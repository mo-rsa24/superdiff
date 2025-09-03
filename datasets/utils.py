from src.dataset.ChestXRay import ChestXrayDataset
from src.dataset.transforms import build_preprocessing, safe_augmentation


def load_dataset(root_dir, task: str, normalization: str='minmax', resize_strategy: str='center_crop', hist_eq: bool=False, aug_risk: str='low'):
    transform = build_preprocessing(normalization, resize_strategy, hist_eq) if aug_risk == "none" else None
    aug = safe_augmentation(aug_risk, normalization=normalization)
    return ChestXrayDataset(root_dir, task, transform=transform, aug=aug)
