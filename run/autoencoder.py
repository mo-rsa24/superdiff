# run/autoencoder.py
import argparse, os, json, math
from datetime import datetime
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes

# --- Local datasets & helpers (same style as cxr.py) ---
from datasets.ChestXRay import ChestXrayDataset
from models.ae_kl import AutoencoderKL
from losses.lpips_gan import LPIPSWithDiscriminatorJAX, LPIPSGANConfig, PerceptualHook

# --- Optional W&B ---
try:
    import wandb
    _WANDB = True
except Exception:
    wandb = None
    _WANDB = False

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def make_grid_torch(imgs_tensor, nrow=None):
    from torchvision.utils import make_grid
    N = imgs_tensor.shape[0]
    if nrow is None:
        nrow = int(math.sqrt(max(1, N)))
    return make_grid(imgs_tensor, nrow=nrow)

def parse_args():
    p = argparse.ArgumentParser("JAX AutoencoderKL (CXR) trainer")

    # Data
    p.add_argument("--data_root", default="../datasets/cleaned")
    p.add_argument("--task", choices=["TB","PNEUMONIA"], default="TB")
    p.add_argument("--split", choices=["train","val","test"], default="train")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--class_filter", type=int, default=1)
    p.add_argument("--overfit_one", action="store_true",
                                       help = "Repeat a single sample to overfit the AE.")
    p.add_argument("--overfit_k", type=int, default=0,
                                       help = "If >0, train on a fixed tiny subset of size K.")
    p.add_argument("--repeat_len", type=int, default=16384,
                                       help = "Virtual length for the repeated one-sample dataset.")

    # Model arch (no YAML)
    p.add_argument("--ch_mults", type=str, default="128,256,512")
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--z_channels", type=int, default=3)

    # Loss settings (LDM-like)
    p.add_argument("--kl_weight", type=float, default=1.0e-6)
    p.add_argument("--pixel_weight", type=float, default=1.0)
    p.add_argument("--disc_start", type=int, default=50001)
    p.add_argument("--disc_factor", type=float, default=1.0)
    p.add_argument("--disc_weight", type=float, default=0.5)
    p.add_argument("--disc_layers", type=int, default=3)
    p.add_argument("--disc_loss", choices=["hinge","vanilla"], default="hinge")
    p.add_argument("--perceptual_weight", type=float, default=0.0)

    # Optimizer
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # Logging & ckpts
    p.add_argument("--output_root", default="runs")
    p.add_argument("--exp_name", default="cxr_ae")
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume_dir", default=None)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=100)

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="cxr-ae")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_tags", default="")
    p.add_argument("--wandb_id", default=None)
    return p.parse_args()

def n_local_devices():
    return jax.local_device_count()

def main():
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    # ----- run dirs -----
    H = W = int(args.img_size)
    per_dev = max(1, args.batch_per_device)
    ndev = n_local_devices()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    ch_mults = tuple(int(c.strip()) for c in args.ch_mults.split(",") if c.strip())
    mode = "full"
    if args.overfit_one:
        mode = "of1"
    elif args.overfit_k > 0:
        mode = f"tiny{args.overfit_k}"
    exp_slug = (
       f"{args.exp_name}"
       f"-{args.task.lower()}-{args.split}"
       f"-cxr{H}-{mode}"
       f"-ch{'x'.join(map(str, ch_mults))}"
       f"-z{args.z_channels}"
       f"-lr{args.lr:g}-b{per_dev}x{ndev}")
    run_dir = args.resume_dir if args.resume_dir else os.path.join(args.output_root, args.run_name or exp_slug, ts)
    ckpt_dir = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(os.path.join(run_dir, "samples"))
    ckpt_latest = os.path.join(ckpt_dir, "last.flax")
    meta_path = os.path.join(run_dir, "run_meta.json")

    with open(meta_path, "w") as f:
        json.dump({**vars(args), "run_dir": run_dir, "ckpt_latest": ckpt_latest}, f, indent=2)

    # ----- dataset -----
    base_ds = ChestXrayDataset(
        root_dir=args.data_root, task=args.task, split=args.split,
        img_size=args.img_size, class_filter=args.class_filter
    )
    label_counts = Counter(base_ds.labels)
    class RepeatOne(Dataset):
        def __init__(self, item, length: int):
            self.x, self.y = item
            self.length = int(length)
        def __len__(self): return self.length
        def __getitem__(self, idx): return self.x, self.y

    if args.overfit_one:
        batch_size = per_dev * ndev
        indices = list(range(batch_size))
        ds = Subset(base_ds, indices)
        ds = torch.utils.data.ConcatDataset([ds] * (args.repeat_len // batch_size))
        shuffle = True
        drop_last = True
    elif args.overfit_k > 0:
        idxs = list(range(min(args.overfit_k, len(base_ds))))
        ds = Subset(base_ds, idxs)
        shuffle = True
        drop_last = True
    else:
        ds = base_ds
        shuffle = True
        drop_last = True
    batch_size = per_dev * ndev
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                         num_workers = 8, drop_last = drop_last, pin_memory = True)

    # ----- model -----
    enc_cfg = dict(ch_mults = ch_mults,
        in_ch = 1, num_res_blocks = args.num_res_blocks, dropout = args.dropout, double_z = True, )
    dec_cfg = dict( ch_mults = ch_mults, out_ch = 1, num_res_blocks = args.num_res_blocks, dropout = args.dropout,)
    ae = AutoencoderKL(enc_cfg=enc_cfg, dec_cfg=dec_cfg, embed_dim=args.z_channels)

    # ----- loss -----
    loss_cfg = LPIPSGANConfig(
        disc_start=args.disc_start, kl_weight=args.kl_weight, pixel_weight=args.pixel_weight,
        disc_num_layers=args.disc_layers, disc_in_channels=1, disc_factor=args.disc_factor,
        disc_weight=args.disc_weight, disc_loss=args.disc_loss, perceptual_weight=args.perceptual_weight
    )
    loss_mod = LPIPSWithDiscriminatorJAX(loss_cfg, perc=PerceptualHook(fn=None, weight=args.perceptual_weight))

    # ----- init -----
    fake = jnp.ones((batch_size, H, W, 1), dtype=jnp.float32)
    variables = ae.init({'params': rng, 'dropout': rng}, fake, rng=rng, sample_posterior=True, train=True)
    params = variables['params']

    # loss_vars = loss_mod.init({'params': rng}, x_in=fake, x_rec=fake, posterior=None, step=jnp.array(0), train=True)
    fake_loss = jnp.ones((1, 32, 32, 1), dtype=jnp.float32)  # tiny tensor for safe init
    loss_vars = loss_mod.init({'params': rng}, x_in = fake_loss, x_rec = fake_loss, posterior = None, step = jnp.array(0), train = True)
    loss_params = loss_vars['params']

    # two optimizers (generator vs discriminator) like LDM
    def tx(lr):
        return optax.chain(
            optax.clip_by_global_norm(args.grad_clip) if args.grad_clip>0 else optax.identity(),
            optax.adamw(lr, weight_decay=args.weight_decay),
        )
    # group params
    def split_gen_disc(ae_params, loss_params):
        # everything in AE is "gen"; discriminator is in loss_params
        disc = {'loss': loss_params}
        gen = {'ae': ae_params}
        return gen, disc

    gen_params, disc_params = split_gen_disc(params, loss_params)
    gen_state  = TrainState.create(apply_fn=None, params=gen_params,  tx=tx(args.lr))
    disc_state = TrainState.create(apply_fn=None, params=disc_params, tx=tx(args.lr))

    # ----- resume -----
    if args.resume_dir and tf.io.gfile.exists(ckpt_latest):
        print(f"[info] resume from {ckpt_latest}")
        with tf.io.gfile.GFile(ckpt_latest, "rb") as f: blob = f.read()
        gen_state, disc_state = from_bytes((gen_state, disc_state), blob)

    # ----- wandb -----
    use_wandb = bool(args.wandb and _WANDB)
    wb = None
    if use_wandb:
        wb = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.run_name or exp_slug, id=args.wandb_id, resume="allow" if args.wandb_id else None,
            dir=run_dir, config={**vars(args), "n_devices": ndev, "label_counts": dict(label_counts)}
        )
        wandb.define_metric("train/step")
        wandb.define_metric("epoch/*", step_metric="epoch/idx")

    # ----- step fns -----
    def model_apply(ae_params, x, *, rng, train):
        return ae.apply({'params': ae_params}, x, rng=rng, sample_posterior=True, train=train)

    @jax.jit
    def gen_step(gen_state, disc_state, x, step):
        def loss_fn(params):
            rng1, rng2 = jax.random.split(jax.random.PRNGKey(step))
            xrec, posterior = model_apply(params['ae'], x, rng=rng1, train=True)
            g_loss, logs_g, d_loss, logs_d = loss_mod.apply(
                {'params': disc_state.params['loss']},
                x_in=x, x_rec=xrec, posterior=posterior, step=jnp.array(step), train=True, mutable=False
            )
            # Only return generator portion (nll+kl+g); discriminator updated separately
            return g_loss, (logs_g, xrec, posterior)
        (g_loss, (logs_g, xrec, posterior)), grads = jax.value_and_grad(loss_fn, has_aux=True)(gen_state.params)
        gen_state = gen_state.apply_gradients(grads=grads)
        return gen_state, logs_g, xrec, posterior

    @jax.jit
    def disc_step(gen_params, disc_state, x, step):
        rng1 = jax.random.PRNGKey(step)
        xrec, posterior = model_apply(gen_params['ae'], x, rng=rng1, train=True)
        def loss_fn(dparams):
            g_loss, logs_g, d_loss, logs_d = loss_mod.apply(
                {'params': dparams['loss']},
                x_in=x, x_rec=xrec, posterior=posterior, step=jnp.array(step), train=True, mutable=False
            )
            return d_loss, logs_d
        (d_loss, logs_d), grads = jax.value_and_grad(loss_fn, has_aux=True)(disc_state.params)
        disc_state = disc_state.apply_gradients(grads=grads)
        return disc_state, logs_d

    # ----- training loop -----
    global_step = 0
    for ep in tqdm.trange(args.epochs, desc="epochs"):
        inner = tqdm.tqdm(loader, desc=f"epoch {ep+1}/{args.epochs}", leave=False)
        for step_i, (batch, _) in enumerate(inner):
            # Convert to JAX/NumPy array first
            x = jnp.asarray(batch.numpy())
            # Permute and normalize in JAX
            x = jnp.transpose(x, (0, 2, 3, 1))  # N, C, H, W -> N, H, W, C
            x = (x + 1.0) * 0.5  # [-1, 1] -> [0,

            gen_state, logs_g, xrec, posterior = gen_step(gen_state, disc_state, x, global_step)
            disc_state, logs_d = disc_step(gen_state.params, disc_state, x, global_step)

            global_step += 1
            if use_wandb and (global_step % max(1, args.log_every) == 0):
                payload = {"train/step": global_step}
                payload.update({k: float(v) for k,v in logs_g.items()})
                payload.update({k: float(v) for k,v in logs_d.items()})
                wandb.log(payload)

        # save ckpt each epoch
        payload = to_bytes((gen_state, disc_state))
        ep_path = os.path.join(ckpt_dir, f"ep{ep+1:04d}.flax")
        with tf.io.gfile.GFile(ep_path, "wb") as f: f.write(payload)
        with tf.io.gfile.GFile(ckpt_latest, "wb") as f: f.write(payload)

        if use_wandb:
            wandb.log({"epoch/idx": ep+1, "ckpt/last_path": ckpt_latest})

        # sampling grid
        if ((ep+1) % max(1, args.sample_every)) == 0:
            with torch.no_grad():
                xnp = np.asarray(xrec)  # last batch reconstructions (0..1)
                xnp = np.transpose(xnp, (0,3,1,2))  # N,1,H,W
                imgs = torch.tensor(xnp).clamp(0,1)
                grid = make_grid_torch(imgs, nrow=8)
                grid_np = grid.permute(1,2,0).numpy()
                out_path = os.path.join(samples_dir, f"recon_ep{ep+1:03d}.png")
                from PIL import Image
                Image.fromarray((grid_np*255).astype(np.uint8)).save(out_path)
                if use_wandb:
                    wandb.log({"samples/recon_grid": wandb.Image(out_path)})

    print(f"[done] run dir: {run_dir}")
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
