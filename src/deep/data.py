"""Image dataset: load -> DIP preprocessing (P2) -> augmentation -> tensor, plus the standardised DIP features."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.config import PROJECT_ROOT
from src.dip.features import FEATURE_NAMES
from src.dip.preprocess import load_rgb, preprocess

CLASSES = ["fire", "smoke", "normal"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def transforms(train: bool, size: int):
    aug = [v2.RandomResizedCrop(size, scale=(0.8, 1.0), antialias=True), v2.RandomHorizontalFlip(),
           v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] if train else []
    return v2.Compose([*aug, v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


class FireSmokeDataset(Dataset):
    def __init__(self, df, pre_cfg: dict, train: bool, feat_mean: np.ndarray, feat_std: np.ndarray):
        self.paths = df["path"].tolist()
        self.y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        f = (df[FEATURE_NAMES].to_numpy(np.float32) - feat_mean) / feat_std
        self.f = np.clip(f, -5, 5).astype(np.float32)
        self.pre_cfg = pre_cfg
        self.tf = transforms(train, pre_cfg["size"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p2 = preprocess(load_rgb(PROJECT_ROOT / self.paths[i]), self.pre_cfg)
        x = self.tf(torch.from_numpy(np.ascontiguousarray(p2)).permute(2, 0, 1))
        return x, torch.from_numpy(self.f[i]), int(self.y[i]), i


def worker_init(_):
    cv2.setNumThreads(0)
