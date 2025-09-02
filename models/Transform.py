from datasets.Shapes import GrayscaleShapesDataset
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from torch.utils.data import DataLoader
import joblib
import torch

from models.VAE import BetaVAE
from train.train_vae import train_beta_vae


def GrayscalePCA(dataset: GrayscaleShapesDataset, n_components: int = 2, filename: str = "pca_grayscale.joblib", checkpoint_dir: str = "./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    pca_model_path = os.path.join(checkpoint_dir, filename)
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    images, labels = next(iter(dataloader))
    images_flat = images.view(images.size(0), -1).numpy()
    pca = PCA(n_components=n_components)
    pca.fit(images_flat)
    joblib.dump(pca, pca_model_path)
    print(f"Grayscale PCA model saved to {pca_model_path}")
    return pca


def GrayscaleUMAP(dataset: GrayscaleShapesDataset, n_components: int = 2, filename: str = "umap_grayscale.joblib",
                  checkpoint_dir: str = "./checkpoints"):
    """
    Applies UMAP to reduce the dimensionality of the GrayscaleShapesDataset.

    Args:
        dataset (GrayscaleShapesDataset): The input dataset.
        n_components (int, optional): The number of dimensions to reduce to. Defaults to 2.
        filename (str, optional): The filename to save the UMAP model. Defaults to "umap_grayscale.joblib".
        checkpoint_dir (str, optional): The directory to save the model. Defaults to "./checkpoints".

    Returns:
        umap.UMAP: The trained UMAP reducer model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    umap_model_path = os.path.join(checkpoint_dir, filename)

    # Load the entire dataset into a single batch
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    images, _ = next(iter(dataloader))

    # Flatten the images
    images_flat = images.view(images.size(0), -1).numpy()

    # Initialize and fit UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reducer.fit(images_flat)

    # Save the UMAP model
    joblib.dump(reducer, umap_model_path)
    print(f"Grayscale UMAP model saved to {umap_model_path}")

    return reducer



def GrayscaleTSNE(dataset: GrayscaleShapesDataset, n_components: int = 2, filename: str = "tsne_grayscale.joblib", checkpoint_dir: str = "./checkpoints"):
    """Applies t-SNE and saves the model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    tsne_model_path = os.path.join(checkpoint_dir, filename)
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    images, _ = next(iter(dataloader))
    images_flat = images.view(images.size(0), -1).numpy()
    # Corrected: Changed 'n_iter' to 'max_iter' for compatibility with newer scikit-learn versions.
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, max_iter=1000)
    tsne.fit_transform(images_flat)
    joblib.dump(tsne, tsne_model_path)
    print(f"Grayscale t-SNE model saved to {tsne_model_path}")
    return tsne

def get_vae_embedding(model, dataset):
    """
    Gets the 2D embedding from the trained VAE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    images, labels = next(iter(dataloader))
    images = images.to(device)
    with torch.no_grad():
        mu, _ = model.encode(images)
    return mu.cpu().numpy(), labels.numpy()


def get_latent_codes(method: str, dataset: GrayscaleShapesDataset, n_components: int = 2, vae_params: dict = None,
                     checkpoint_dir: str = "./checkpoints"):
    """
    Produces latent codes for the dataset using the specified dimensionality reduction method.

    Args:
        method (str): The method to use ('pca', 'umap', 'tsne', 'vae').
        dataset (GrayscaleShapesDataset): The input dataset.
        n_components (int): The number of latent dimensions.
        vae_params (dict, optional): Parameters for VAE training if method is 'vae'.
        checkpoint_dir (str, optional): Directory to save/load models.

    Returns:
        (np.ndarray, np.ndarray): A tuple of (latent_codes, labels).
    """
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(dataloader))
    images_flat = images.view(images.size(0), -1).numpy()
    labels_numpy = labels.numpy()

    if method == 'pca':
        print("Generating latent codes using PCA...")
        pca_model = GrayscalePCA(dataset, n_components=n_components, checkpoint_dir=checkpoint_dir)
        latent_codes = pca_model.transform(images_flat)
        return latent_codes, labels_numpy

    elif method == 'umap':
        print("Generating latent codes using UMAP...")
        filename = f"umap_grayscale.joblib"
        model_path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(model_path):
            print(f"Loading pre-trained UMAP model from {model_path}")
            umap_model = joblib.load(model_path)
        else:
            print("No pre-trained UMAP model found, training a new one...")
            umap_model = GrayscaleUMAP(dataset, n_components=n_components, filename=filename,
                                       checkpoint_dir=checkpoint_dir)
        latent_codes = umap_model.transform(images_flat)
        return latent_codes, labels_numpy

    elif method == 'tsne':
        print("Generating latent codes using t-SNE...")
        tsne_model = GrayscaleTSNE(dataset, n_components=n_components, checkpoint_dir=checkpoint_dir)
        latent_codes = tsne_model.embedding_
        return latent_codes, labels_numpy

    elif method == 'vae':
        print("Generating latent codes using Beta-VAE...")
        if vae_params is None:
            vae_params = {'epochs': 10, 'beta': 4.0}

        model_path = os.path.join(checkpoint_dir, f"beta_vae_ld{n_components}_b{vae_params.get('beta', 4.0)}.pth")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae_model = BetaVAE(latent_dim=n_components).to(device)

        if os.path.exists(model_path):
            print(f"Loading pre-trained VAE model from {model_path}")
            vae_model.load_state_dict(torch.load(model_path))
        else:
            print("No pre-trained VAE model found, training a new one...")
            vae_model = train_beta_vae(
                dataset,
                latent_dim=n_components,
                epochs=vae_params.get('epochs', 10),
                beta=vae_params.get('beta', 4.0),
                checkpoint_dir=checkpoint_dir
            )

        latent_codes, labels_from_func = get_vae_embedding(vae_model, dataset)
        return latent_codes, labels_from_func

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'umap', 'tsne', 'vae'.")