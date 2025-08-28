import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets.Shapes import GrayscaleShapesDataset
from models.VAE import BetaVAE


def loss_function(recon_x, x, mu, logvar, beta=4.0):
    """
    Reconstruction + KL divergence losses summed over all elements and batch.
    """
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD


def train_beta_vae(dataset: GrayscaleShapesDataset, epochs=25, batch_size=128, latent_dim=2, beta=4.0,
                   checkpoint_dir="./checkpoints"):
    """
    Trains the Beta-VAE model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BetaVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting VAE training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'====> Epoch: {epoch + 1} Average loss: {train_loss / len(dataset):.4f}')

    model_path = os.path.join(checkpoint_dir, f"beta_vae_ld{latent_dim}_b{beta}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Beta-VAE model saved to {model_path}")
    return model