"""Image dataset: load -> DIP preprocessing (P2) -> augmentation -> tensor, plus the standardised DIP features."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.config import PROJECT_ROOT
from src.data.boxes import box_mask, label_path, read_boxes
from src.dip.features import FEATURE_NAMES, extract
from src.dip.noise import add_gaussian_noise
from src.dip.preprocess import load_rgb, preprocess
from src.dip.segmentation import segment

CLASSES = ["fire", "smoke", "normal"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def geometric(size: int):
    return v2.Compose([v2.RandomResizedCrop(size, scale=(0.8, 1.0), antialias=True), v2.RandomHorizontalFlip()])


def photometric(train: bool):
    aug = [v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)] if train else []
    return v2.Compose([*aug, v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


class FireSmokeDataset(Dataset):
    """With live=(seg_cfg, feat_cfg) the DIP features are recomputed on the cropped/flipped training view."""

    def __init__(self, df, pre_cfg: dict, train: bool, feat_mean: np.ndarray, feat_std: np.ndarray, live=None,
                 maps=None, noise_aug=None):
        self.paths = df["path"].tolist()
        self.y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        self.mean, self.std = feat_mean.astype(np.float32), feat_std.astype(np.float32)
        self.f = self._norm(df[FEATURE_NAMES].to_numpy(np.float32))
        self.pre_cfg = pre_cfg
        self.geo = geometric(pre_cfg["size"]) if train else None
        self.photo = photometric(train)
        self.live = live if train else None
        self.maps = maps
        self.noise_aug = noise_aug if train else None
        self.sources = df["source"].tolist()

    def _norm(self, f: np.ndarray) -> np.ndarray:
        return np.clip((f - self.mean) / self.std, -5, 5).astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def _box_masks(self, i, size: int):
        ids = self.maps["box_classes"].get(self.sources[i])
        if not ids:
            return np.zeros((2, size, size), np.uint8), False
        boxes = read_boxes(PROJECT_ROOT / label_path(self.paths[i]))
        m = np.stack([box_mask(boxes, ids["fire"], size), box_mask(boxes, ids["smoke"], size)]) * 255
        return m.astype(np.uint8), bool(self.y[i] == CLASSES.index("normal") or m.any())

    def __getitem__(self, i):
        raw = load_rgb(PROJECT_ROOT / self.paths[i])
        if self.noise_aug is not None and np.random.random() < self.noise_aug["p"]:
            raw = add_gaussian_noise(raw, np.random.uniform(*self.noise_aug["sigma"]), np.random.default_rng())
        p2 = preprocess(raw, self.pre_cfg)
        x = torch.from_numpy(np.ascontiguousarray(p2)).permute(2, 0, 1)
        f = self.f[i]
        target, valid = torch.zeros(2, 1, 1), False
        dip_maps = self.maps is not None and self.maps.get("source", "boxes") == "dip"
        if self.maps is not None and not dip_maps:
            m, valid = self._box_masks(i, p2.shape[0])
            x = torch.cat([x, torch.from_numpy(m)], 0)
        if self.geo is not None:
            x = self.geo(x)
        if self.maps is not None and not dip_maps:
            target = torch.nn.functional.adaptive_avg_pool2d(x[3:].float() / 255.0, self.maps["grid"])
            x = x[:3]
        seg = None
        if dip_maps or (self.geo is not None and self.live is not None):
            view = np.ascontiguousarray(x.permute(1, 2, 0).numpy())
            seg = segment(view, self.maps["seg"] if dip_maps else self.live[0])
        if dip_maps:
            m = torch.from_numpy(np.stack([seg["fire"], seg["smoke"]]).astype(np.float32))
            target, valid = torch.nn.functional.adaptive_avg_pool2d(m, self.maps["grid"]), True
        if self.geo is not None and self.live is not None:
            d = extract(view, seg, self.live[1])
            f = self._norm(np.array([d[k] for k in FEATURE_NAMES], np.float32))
        if self.maps is not None:
            return self.photo(x), torch.from_numpy(f), int(self.y[i]), i, target, valid
        return self.photo(x), torch.from_numpy(f), int(self.y[i]), i


def worker_init(_):
    cv2.setNumThreads(0)
