import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

def _list_images(dirpath: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg")
    return sorted([p for p in dirpath.rglob("*") if p.suffix.lower() in exts])

class ChestXrayDataset(Dataset):
    """
    ChestXrayDataset for multi-class conditional training.

    Directory layout expected:
        root_dir/
            train/
              NORMAL/*.jpg
              PNEUMONIA/*.jpg
              TB/*.jpg
            val/
            test/
    """
    def __init__(
        self,
        root_dir: str = "../datasets/cleaned",
        split: str = "train",
        img_size: int = 256,
    ):
        root = Path(root_dir) / split.lower()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset path not found or is not a directory: {root}")

        # Find class subdirectories (e.g., NORMAL, PNEUMONIA, TB)
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not class_dirs:
            raise RuntimeError(f"No class subdirectories found in {root}")

        self.img_paths: List[Path] = []
        self.labels: List[int] = []
        self.class_map: Dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}

        print(f"[Dataset] Found classes: {list(self.class_map.keys())}")

        for class_name, label in self.class_map.items():
            class_dir = root / class_name
            for p in _list_images(class_dir):
                self.img_paths.append(p)
                self.labels.append(label)

        if not self.img_paths:
            raise RuntimeError(f"No images found under {root}")

        self.img_size = int(img_size)
        print(f"[Dataset] Loaded {len(self.img_paths)} images from {split} split.")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        p = self.img_paths[idx]
        img = Image.open(p).convert("L").resize((self.img_size, self.img_size), Image.BICUBIC)
        x = np.asarray(img, dtype=np.float32) / 255.0   # [0,1]
        x = x * 2.0 - 1.0                               # [-1,1]
        x = x[None, ...]                                # (1, H, W)
        return torch.from_numpy(x), int(self.labels[idx])