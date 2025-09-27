import argparse
import os
import math
import json
import functools
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
import optax

import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

from models.cxr_unet import ScoreNet


def _l2_norm_sq(x, eps=1e-8):
    """Calculates the squared L2 norm for a batch of tensors."""
    return (x ** 2).sum(axis=(1, 2, 3)) + eps


def sampler_superdiff_and_sde(rng, model_a, pa, model_b, pb,
                              marginal_prob_std_fn, diffusion_coeff_fn,
                              shape, num_steps=500, eps=1e-3,
                              log_dir=None, do_diagnostics=False):
    """
    Faithfully implements the SUPERDIFF logical AND using a reverse-SDE sampler.

    This method calculates a dynamic, time-dependent weight `κ` at each step
    to solve for the optimal combination of scores `sA` and `sB`, as described
    in the SUPERDIFF paper and its reference implementation.

    NOTE: This implementation makes two necessary assumptions to adapt the logic
    from the paper's Stable Diffusion (PyTorch/DDPM) notebook to this script's
    unconditional (JAX/VE SDE) framework:
      1. Classifier-Free Guidance is not used, so the `guidance_scale` is set to 1.0.
      2. The change in noise level `dsigma` is approximated from the VE SDE's
         continuous-time `marginal_prob_std_fn`.
    """
    B = shape[0]
    rng, sub = jax.random.split(rng)
    # Initialize noise from the prior distribution at t=1
    x = jax.random.normal(sub, shape) * marginal_prob_std_fn(1.0)

    # Create the reverse-time integration grid
    t_grid = make_time_grid(eps, num_steps, x.dtype, reverse=True)
    dt = t_grid[0] - t_grid[1]
    sqrt_dt = jnp.sqrt(dt)

    # Diagnostics containers
    kappa_list = []

    for i, t in enumerate(t_grid):
        t_batch = jnp.full((B,), t, dtype=x.dtype)

        # 1. Calculate the scores from both of your unconditional models
        # Model A can be considered the 'object' and Model B the 'background'
        sA = score_fn(model_a, pa, x, t_batch)  # s_obj
        sB = score_fn(model_b, pb, x, t_batch)  # s_bg

        # --- Dynamic Kappa Calculation (The Core of SUPERDIFF AND) ---

        # 2. Get the SDE coefficients for the current timestep `t`
        g = diffusion_coeff_fn(t_batch)
        g2 = broadcast_g2(g, x)

        # 3. Approximate `dsigma`, the change in noise std deviation over the step `dt`.
        # This is required by the SUPERDIFF formula.
        sigma_curr = marginal_prob_std_fn(t_batch)
        sigma_next = marginal_prob_std_fn(t_batch - dt)
        dsigma = sigma_curr - sigma_next  # Should be positive as we go from t=1 to t=eps

        # 4. Calculate the hypothetical update `dx_ind` using only the 'background' model (sB)
        # This is a necessary component for the kappa formula.
        rng, sub = jax.random.split(rng)
        noise = jax.random.normal(sub, x.shape)
        drift_B = -0.5 * g2 * sB
        dx_ind = drift_B * dt + g.reshape((-1,) + (1,) * (x.ndim - 1)) * noise * sqrt_dt

        # 5. Solve for kappa `κ` using the closed-form solution from the paper's logic
        s_diff = sA - sB
        s_sum = sA + sB

        # Numerator of the kappa equation
        # Note: In the reference code, (vel_bg-vel_obj) is used, so we use (sB-sA)
        # The equation is: abs(dsigma)*(sB-sA)*(sB+sA) - (dx_ind*(sA-sB))
        num_term1 = (jnp.abs(dsigma.reshape(-1, 1, 1, 1)) * (sB - sA) * s_sum).sum(axis=(1, 2, 3))
        num_term2 = (dx_ind * s_diff).sum(axis=(1, 2, 3))
        numerator = num_term1 - num_term2

        # Denominator of the kappa equation
        guidance_scale = 1.0  # Assuming no CFG, so scale is 1
        denominator = 2 * dsigma.reshape(-1, 1, 1, 1) * guidance_scale * _l2_norm_sq(s_diff)

        kappa = numerator / denominator

        if do_diagnostics:
            kappa_list.append(np.asarray(kappa.mean()))

        # --- End of Kappa Calculation ---

        # 6. Form the composed score `s_combined` using the calculated kappa
        # The composition is: s_bg + kappa * (s_obj - s_bg)
        s_combined = sB + kappa[:, None, None, None] * s_diff

        # 7. Perform the reverse-SDE update step using the dynamically composed score
        drift_combined = -0.5 * g2 * s_combined
        # Use the *same noise* that was generated for the kappa calculation for a consistent update
        x = x + drift_combined * dt + g.reshape((-1,) + (1,) * (x.ndim - 1)) * noise * sqrt_dt

    # Save diagnostics plot for kappa if enabled
    if do_diagnostics and log_dir is not None:
        save_plot([kappa_list], ["kappa"], "Dynamic Kappa (κ) over steps",
                  os.path.join(log_dir, "kappa_dynamic_sde.png"),
                  ylabel="kappa value")

    return x

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _resolve_run_dir(output_root: str, run_name: str):
    base = os.path.join(output_root, run_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base run directory not found: {base}")
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subs:
        return base
    return os.path.join(base, sorted(subs)[-1])

def _latest_ckpt(ckpt_dir: str):
    last = os.path.join(ckpt_dir, "last.flax")
    if tf.io.gfile.exists(last):
        return last
    eps = sorted([p for p in tf.io.gfile.listdir(ckpt_dir)
                  if p.startswith("ep") and p.endswith(".flax")])
    if not eps:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return os.path.join(ckpt_dir, eps[-1])

def _load_config(run_dir: str):
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not tf.io.gfile.exists(meta_path):
        raise FileNotFoundError(f"Missing run_meta.json in {run_dir}")
    with tf.io.gfile.GFile(meta_path, "r") as f:
        return json.load(f)

def _build_sde_from_cfg(cfg):
    sde_type = cfg.get("sde", "VE").upper()
    if sde_type == "VE":
        from diffusion.equations import marginal_prob_std, diffusion_coeff
        sigma_max = float(cfg.get("sigma_max", 12.0))
        mstd_fn = functools.partial(marginal_prob_std, sigma=sigma_max)
        dcoeff_fn = functools.partial(diffusion_coeff, sigma=sigma_max)
        return "VE", mstd_fn, dcoeff_fn
    else:
        # Extend as needed for VP/VESDE etc.
        raise ValueError(f"SDE type '{sde_type}' not supported in this compose.py")

def _load_model_from_run(output_root, run_name):
    print(f"[*] Loading model from run: {run_name}")
    run_dir = _resolve_run_dir(output_root, run_name)
    cfg = _load_config(run_dir)
    sde_name, mstd_fn, dcoeff_fn = _build_sde_from_cfg(cfg)

    channels = tuple(int(c.strip()) for c in cfg.get("channels", "64,128,256,512").split(","))
    embed_dim = int(cfg.get("embed_dim", 256))
    model = ScoreNet(marginal_prob_std=mstd_fn, channels=channels, embed_dim=embed_dim)

    img_size = int(cfg.get("img_size", 256))
    dummy_x = jnp.ones((1, img_size, img_size, 1), dtype=jnp.float32)
    dummy_t = jnp.ones((1,), dtype=jnp.float32)
    params_template = model.init({'params': jax.random.PRNGKey(0)}, dummy_x, dummy_t)

    state_template = TrainState.create(apply_fn=model.apply, params=params_template, tx=optax.adam(1e-4))
    ckpt_path = _latest_ckpt(os.path.join(run_dir, "ckpts"))
    print(f"    - Loading checkpoint: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    try:
        target_template = (None, params_template, None)  # (state, ema_params, ema_decay)
        restored_tuple = from_bytes(target_template, blob)
        restored_ema_params = restored_tuple[1]
        print("    - Successfully loaded EMA parameters.")
        return model, restored_ema_params, cfg, (mstd_fn, dcoeff_fn)
    except Exception as e:
        print("    - FAILED to load EMA params. The model structure in your script might not match the checkpoint.")
        print(f"      Original error: {type(e).__name__}: {e}")
        raise

def _assert_compat(cfg_a, cfg_b):
    keys = ["img_size", "channels", "embed_dim", "sde", "sigma_max"]
    for k in keys:
        if str(cfg_a.get(k)) != str(cfg_b.get(k)):
            raise ValueError(
                f"Incompatible runs: mismatch on '{k}'. "
                f"A:{cfg_a.get(k)} vs B:{cfg_b.get(k)}"
            )
    print("[+] Models appear compatible for composition.")

# -------------------------
# Core math helpers
# -------------------------

def score_fn(model, params, x, t_batch):
    # model.apply expects (x, t)
    return model.apply(params, x, t_batch)

def vel_from_score_VE(diffusion_coeff_fn, s):
    # VE drift/velocity for the probability-flow ODE: v = -(1/2) g(t)^2 * s
    # Here, s shape = (B, H, W, C), g2 broadcast needed outside
    return s  # placeholder; caller multiplies by broadcast g2

def make_time_grid(eps, num_steps, dtype, reverse=True):
    # t: 1 -> eps (descending) for reverse-time integration
    tg = jnp.linspace(1.0, eps, num_steps, dtype=dtype)
    return tg if reverse else tg[::-1]

def broadcast_g2(g, x):
    g2 = (g ** 2).reshape((-1,) + (1,) * (x.ndim - 1))
    return g2

def to_torch_grid_and_save(x_np, out_path_png, nrow=None, cmap='gray'):
    # x_np expected in [B,H,W,1], in [0,1]
    x_np = np.clip(x_np, 0.0, 1.0)
    x_chw = np.transpose(x_np, (0, 3, 1, 2))
    xt = torch.tensor(x_chw)
    if nrow is None:
        nrow = int(math.sqrt(len(xt))) if len(xt) > 0 else 1
    grid_t = make_grid(xt, nrow=nrow)
    save_image(grid_t, out_path_png)
    # mpl version
    grid_np = grid_t.permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_np, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path_png.replace(".png", "_mpl.png"),
                bbox_inches="tight", pad_inches=0)
    plt.close()

# -------------------------
# Diagnostics helpers
# -------------------------

def cosine_similarity(a, b, eps=1e-8):
    # a,b shape: (B,H,W,C)
    num = (a * b).sum(axis=(1, 2, 3))
    den = jnp.sqrt((a**2).sum(axis=(1, 2, 3)) + eps) * jnp.sqrt((b**2).sum(axis=(1, 2, 3)) + eps)
    return num / den

def l2_norm(x):
    return jnp.sqrt((x**2).sum(axis=(1, 2, 3)))

def save_plot(y_list, labels, title, out_png, xlabel="step", ylabel="value"):
    plt.figure()
    for y, lab in zip(y_list, labels):
        plt.plot(y, label=lab)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_histogram(vals, title, out_png, bins=50, xlab=None):
    plt.figure()
    plt.hist(vals, bins=bins)
    plt.title(title)
    if xlab:
        plt.xlabel(xlab)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# -------------------------
# Samplers
# -------------------------

def sampler_poe_ode(rng, model_a, pa, model_b, pb,
                    marginal_prob_std_fn, diffusion_coeff_fn,
                    shape, num_steps=500, eps=1e-3,
                    log_dir=None, do_diagnostics=False):
    """
    VE probability-flow ODE sampler with PoE (sum of scores).
    dx/dt = v = -(1/2) g(t)^2 [s_A + s_B]
    """
    B = shape[0]
    rng, sub = jax.random.split(rng)
    x = jax.random.normal(sub, shape) * marginal_prob_std_fn(1.0)

    t_grid = make_time_grid(eps, num_steps, x.dtype, reverse=True)
    dt = t_grid[0] - t_grid[1]

    # Diagnostics containers
    cos_list, ratio_list, vnorm_list, energy_list = [], [], [], []
    hist_cos_final = None

    for i, t in enumerate(t_grid):
        t_batch = jnp.full((B,), t, dtype=x.dtype)

        sA = score_fn(model_a, pa, x, t_batch)
        sB = score_fn(model_b, pb, x, t_batch)

        g = diffusion_coeff_fn(t_batch)
        g2 = broadcast_g2(g, x)
        v = -0.5 * g2 * (sA + sB)

        if do_diagnostics:
            cos = cosine_similarity(sA, sB)
            cos_list.append(np.asarray(cos.mean()))
            ratio = (l2_norm(sA) / (l2_norm(sB) + 1e-8)).mean()
            ratio_list.append(np.asarray(ratio))
            vnorm_list.append(np.asarray(l2_norm(v).mean()))
            energy_list.append(np.asarray(l2_norm(x).mean()))
            if i == len(t_grid) - 1:
                hist_cos_final = np.asarray(cos)

        x = x + v * dt

    if do_diagnostics and log_dir is not None:
        save_plot([cos_list], ["cos(sA,sB)"], "Score cosine similarity", os.path.join(log_dir, "cosine.png"))
        save_plot([ratio_list], ["||sA||/||sB||"], "Norm ratio over steps", os.path.join(log_dir, "norm_ratio.png"))
        save_plot([vnorm_list], ["||v||"], "Drift norm over steps", os.path.join(log_dir, "drift_norm.png"))
        save_plot([energy_list], ["||x||"], "Sample energy over steps", os.path.join(log_dir, "energy.png"))
        if hist_cos_final is not None:
            save_histogram(hist_cos_final, "Cosine similarity at final step", os.path.join(log_dir, "cosine_hist_final.png"),
                           bins=50, xlab="cos(sA,sB)")

    return x

def sampler_poe_sde(rng, model_a, pa, model_b, pb,
                    marginal_prob_std_fn, diffusion_coeff_fn,
                    shape, num_steps=500, eps=1e-3,
                    log_dir=None, do_diagnostics=False):
    """
    VE reverse-SDE sampler with PoE drift:
      dx = [ -(1/2) g^2 (sA + sB) ] dt + g dW
    Euler–Maruyama with dt > 0 over reversed t-grid (1->eps).
    """
    B = shape[0]
    rng, sub = jax.random.split(rng)
    x = jax.random.normal(sub, shape) * marginal_prob_std_fn(1.0)

    t_grid = make_time_grid(eps, num_steps, x.dtype, reverse=True)
    dt = t_grid[0] - t_grid[1]
    sqrt_dt = jnp.sqrt(dt)

    # Diagnostics containers
    cos_list, ratio_list, vnorm_list, energy_list = [], [], [], []
    hist_cos_final = None

    for i, t in enumerate(t_grid):
        t_batch = jnp.full((B,), t, dtype=x.dtype)

        sA = score_fn(model_a, pa, x, t_batch)
        sB = score_fn(model_b, pb, x, t_batch)

        g = diffusion_coeff_fn(t_batch)
        g2 = broadcast_g2(g, x)
        drift = -0.5 * g2 * (sA + sB)

        # Noise term
        rng, sub = jax.random.split(rng)
        noise = jax.random.normal(sub, x.shape)
        x = x + drift * dt + broadcast_g2(g**0.0, x) * (g.reshape((-1,)+(1,)*(x.ndim-1)) * noise * sqrt_dt)

        if do_diagnostics:
            cos = cosine_similarity(sA, sB)
            cos_list.append(np.asarray(cos.mean()))
            ratio = (l2_norm(sA) / (l2_norm(sB) + 1e-8)).mean()
            ratio_list.append(np.asarray(ratio))
            vnorm_list.append(np.asarray(l2_norm(drift).mean()))
            energy_list.append(np.asarray(l2_norm(x).mean()))
            if i == len(t_grid) - 1:
                hist_cos_final = np.asarray(cos)

    if do_diagnostics and log_dir is not None:
        save_plot([cos_list], ["cos(sA,sB)"], "Score cosine similarity (SDE)", os.path.join(log_dir, "cosine_sde.png"))
        save_plot([ratio_list], ["||sA||/||sB||"], "Norm ratio over steps (SDE)", os.path.join(log_dir, "norm_ratio_sde.png"))
        save_plot([vnorm_list], ["||drift||"], "Drift norm over steps (SDE)", os.path.join(log_dir, "drift_norm_sde.png"))
        save_plot([energy_list], ["||x||"], "Sample energy over steps (SDE)", os.path.join(log_dir, "energy_sde.png"))
        if hist_cos_final is not None:
            save_histogram(hist_cos_final, "Cosine similarity at final step (SDE)",
                           os.path.join(log_dir, "cosine_hist_final_sde.png"),
                           bins=50, xlab="cos(sA,sB)")

    return x

def sampler_temp_grid_ode(rng, model_a, pa, model_b, pb,
                          marginal_prob_std_fn, diffusion_coeff_fn,
                          shape, lambdas, num_steps=500, eps=1e-3):
    """
    Temperature grid over (lambda_A, lambda_B) for PoE ODE:
      v = -(1/2) g^2 (λA sA + λB sB)
    Returns concatenated batch of all grid samples in row-major order.
    """
    B = shape[0]
    img_shape = shape[1:]
    all_samples = []
    labels = []

    # We reuse the same initial noise for fairness across λ pairs
    rng, init_key = jax.random.split(rng)
    x0 = jax.random.normal(init_key, shape) * marginal_prob_std_fn(1.0)

    for lamA, lamB in lambdas:
        x = x0  # start from same init
        t_grid = make_time_grid(eps, num_steps, x.dtype, reverse=True)
        dt = t_grid[0] - t_grid[1]

        for t in t_grid:
            t_batch = jnp.full((B,), t, dtype=x.dtype)
            sA = score_fn(model_a, pa, x, t_batch)
            sB = score_fn(model_b, pb, x, t_batch)
            g = diffusion_coeff_fn(t_batch)
            g2 = broadcast_g2(g, x)
            v = -0.5 * g2 * (lamA * sA + lamB * sB)
            x = x + v * dt

        all_samples.append(x)
        labels.append((lamA, lamB))

    samples = jnp.concatenate(all_samples, axis=0)  # [(k*B), H, W, C]
    return samples, labels

# -------------------------
# Main
# -------------------------

def main(args):
    # Load runs
    model_a, params_a, cfg_a, (mstd_a, dcoeff_a) = _load_model_from_run(args.runs_root, args.run_a)
    model_b, params_b, cfg_b, (mstd_b, dcoeff_b) = _load_model_from_run(args.runs_root, args.run_b)
    _assert_compat(cfg_a, cfg_b)

    # Output dirs
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ensure_dir(os.path.join(args.output_dir, f"compose_{args.run_a}_vs_{args.run_b}_{ts}"))
    diag_dir = ensure_dir(os.path.join(out_dir, "diagnostics"))
    temp_dir = ensure_dir(os.path.join(out_dir, "temp_grid"))

    print(f"[*] Saving samples to: {out_dir}")

    rng = jax.random.PRNGKey(args.seed)
    img_size = int(cfg_a.get("img_size", 256))
    sample_shape = (args.batch_size, img_size, img_size, 1)

    # Use VE functions from run A (compat asserted)
    marginal_prob_std_fn = mstd_a
    diffusion_coeff_fn = dcoeff_a

    # # -------------------------
    # # (1) PoE ODE
    # # -------------------------
    # print("[*] Running PoE ODE sampler ...")
    # rng, sub = jax.random.split(rng)
    # samples_ode = sampler_poe_ode(
    #     sub, model_a, params_a, model_b, params_b,
    #     marginal_prob_std_fn, diffusion_coeff_fn,
    #     sample_shape, num_steps=args.num_steps, eps=args.eps,
    #     log_dir=diag_dir, do_diagnostics=True
    # )
    # to_torch_grid_and_save(
    #     np.asarray(samples_ode),
    #     os.path.join(out_dir, "poe_ode_grid.png"),
    #     nrow=int(math.sqrt(args.batch_size)),
    #     cmap='gray'
    # )
    #
    # # -------------------------
    # # (2) Temperature grid (PoE ODE with λA, λB)
    # # -------------------------
    # print("[*] Running temperature grid (PoE ODE) ...")
    # # Define a reasonable sweep; feel free to change
    # lambdas = []
    # for lamA in args.lambda_list:
    #     for lamB in args.lambda_list:
    #         lambdas.append((lamA, lamB))
    #
    # rng, sub = jax.random.split(rng)
    # samples_temp, labels = sampler_temp_grid_ode(
    #     sub, model_a, params_a, model_b, params_b,
    #     marginal_prob_std_fn, diffusion_coeff_fn,
    #     sample_shape, lambdas, num_steps=args.num_steps, eps=args.eps
    # )
    #
    # # Save one big grid
    # to_torch_grid_and_save(
    #     np.asarray(samples_temp),
    #     os.path.join(temp_dir, "poe_temp_grid_all.png"),
    #     nrow=len(args.lambda_list)*int(math.sqrt(args.batch_size)),
    #     cmap='gray'
    # )
    # # Also save per-(λA,λB)
    # bs = args.batch_size
    # for i, (lamA, lamB) in enumerate(labels):
    #     block = np.asarray(samples_temp[i*bs:(i+1)*bs])
    #     outp = os.path.join(temp_dir, f"lamA_{lamA:.2f}_lamB_{lamB:.2f}.png")
    #     to_torch_grid_and_save(block, outp, nrow=int(math.sqrt(bs)), cmap='gray')

    # -------------------------
    # (3) PoE reverse-SDE
    # -------------------------
    print("[*] Running PoE reverse-SDE sampler ...")
    rng, sub = jax.random.split(rng)
    samples_sde = sampler_superdiff_and_sde(
        sub, model_a, params_a, model_b, params_b,
        marginal_prob_std_fn, diffusion_coeff_fn,
        sample_shape, num_steps=args.num_steps, eps=args.eps,
        log_dir=diag_dir, do_diagnostics=True
    )
    to_torch_grid_and_save(
        np.asarray(samples_sde),
        os.path.join(out_dir, "poe_sde_grid.png"),
        nrow=int(math.sqrt(args.batch_size)),
        cmap='gray'
    )

    print(f"[+] Done. See: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Superposition (PoE) Inference & Diagnostics for Chest X-Ray Models")
    parser.add_argument("--runs_root", default="runs", help="Root directory where training runs are stored.")
    parser.add_argument("--run_a", required=True, help="Name of the first model's run directory (e.g., TB).")
    parser.add_argument("--run_b", required=True, help="Name of the second model's run directory (e.g., Normal).")
    parser.add_argument("--output_dir", default="composition_results", help="Directory to save generated images.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (per sampler run).")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of integration steps.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Final time for the sampler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--lambda_list",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Values for λ sweep in temperature grid for both λA and λB."
    )
    args = parser.parse_args()
    main(args)
