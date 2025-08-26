from datasets.Shapes import GrayscaleShapesDataset
import os
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import joblib

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