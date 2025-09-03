import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image

def normalize_image(method="minmax"):
    if method == "zscore":
        return T.Normalize(mean=[0.5], std=[0.25])  # Use only if justified
    elif method == "minmax":
        return T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
    elif method == "tanh":
        return T.Lambda(lambda x: x * 2 - 1)
    else:
        return T.Lambda(lambda x: x)

def get_albu_normalize(normalization):
    if normalization == "zscore":
        return A.Normalize(mean=[0.5], std=[0.25])  # You can calibrate this to dataset stats
    elif normalization == "minmax":
        return A.Normalize(mean=[0.0], std=[1.0])  # Identity scaling — needed for some models
    elif normalization == "tanh":
        return A.Normalize(mean=[0.5], std=[0.5])  # Scale from [0, 1] → [-1, 1]
    else:
        return A.NoOp()


def resize_with_strategy(size=256, strategy="center_crop"):
    if strategy == "pad":
        return T.Compose([
            T.Resize(size),
            T.CenterCrop(size)
        ])
    elif strategy == "center_crop":
        return T.CenterCrop(size)
    else:
        return T.Resize(size)

def histogram_equalization(enabled=False):
    def apply_clahe(image):
        if not enabled:
            return image
        npimg = np.array(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_img = clahe.apply(npimg)
        return Image.fromarray(eq_img)
    return T.Lambda(apply_clahe)

def build_preprocessing(normalization="minmax", resize_strategy="center_crop", hist_eq=False):
    return T.Compose([
        T.Grayscale(),
        resize_with_strategy(256, strategy=resize_strategy),
        histogram_equalization(hist_eq),
        T.ToTensor(),
        normalize_image(normalization),
    ])

def safe_augmentation(risk="low", normalization="minmax"):
    def get_albu_normalize(norm):
        if norm == "zscore":
            return A.Normalize(mean=[0.5], std=[0.25])
        elif norm == "minmax":
            return A.Normalize(mean=[0.0], std=[1.0])
        elif norm == "tanh":
            return A.Normalize(mean=[0.5], std=[0.5])
        else:
            return A.NoOp()

    normalize = get_albu_normalize(normalization)

    if risk == "none":
        return A.Compose([
            A.Resize(256, 256),
            normalize,
            ToTensorV2()
        ])
    elif risk == "low":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(std_range=(0.0088, 0.0124), p=0.2),
            A.Resize(256, 256),
            normalize,
            ToTensorV2()
        ])
    elif risk == "medium":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=1.0),
            A.RandomBrightnessContrast(p=0.4),
            A.Resize(256, 256),
            normalize,
            ToTensorV2()
        ])
    else:
        raise ValueError("Avoid high-risk medical augmentations.")

