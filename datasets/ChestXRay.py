# ChestXRay.py  (JAX-friendly dataset module)
# Minimal, dependency-light dataset returning torch tensors compatible with PyTorch DataLoader.
# Images are loaded as grayscale, resized to img_size, normalized to [-1, 1], shape (1, H, W).
# This mirrors the interface your JAX training expects and keeps loader code simple.

import os
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

def _list_images(dirpath: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg")
    return sorted([p for p in dirpath.rglob("*") if p.suffix.lower() in exts])

class ChestXrayDataset(Dataset):
    """
    ChestXrayDataset(root_dir, task='TB', split='train', img_size=256, class_names=None)

    Directory layout:
        root_dir/
          TB/         # or PNEUMONIA
            train/
              <classA>/*.jpg|png
              <classB>/*.jpg|png
            val/
            test/

    We only need *images* for unconditional SDE (no labels). We keep labels in case you extend later.
    """
    def __init__(
        self,
        root_dir: str = "../datasets/cleaned",
        task: str = "TB",
        split: str = "train",
        img_size: int = 256,
        class_filter: Optional[int] = None,
    ):
        root = Path(root_dir) / task.upper() / split.lower()
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")
        # Optional subfolders for classes; if none, just use all images in split folder
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        img_paths: List[Path] = []
        labels: List[int] = []

        if subdirs:
            # Two-level layout (e.g., NORMAL/TB subfolders)
            for label, d in enumerate(sorted(subdirs)):
                if class_filter is not None and label != class_filter:
                    continue
                for p in _list_images(d):
                    img_paths.append(p)
                    labels.append(label)
        else:
            # Flat layout
            for p in _list_images(root):
                img_paths.append(p)
                labels.append(0)

        if not img_paths:
            raise RuntimeError(f"No images found under {root}")

        self.paths = img_paths
        self.labels = labels
        self.img_size = int(img_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("L").resize((self.img_size, self.img_size), Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0   # [0,1]
        x = x * 2.0 - 1.0                               # [-1,1]
        x = x[None, ...]                                # (1, H, W)
        return torch.from_numpy(x), int(self.labels[idx])
