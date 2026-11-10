"""YOLO label files: read boxes and rasterise them as masks."""
from pathlib import Path

import cv2
import numpy as np


def label_path(img_rel: str) -> Path:
    p = Path(img_rel)
    return p.parent.parent / "labels" / (p.stem + ".txt") if p.parent.name.lower() == "images" else p.with_suffix(".txt")


def read_boxes(lbl: Path) -> np.ndarray:
    """Rows of (class, x_center, y_center, width, height), normalised 0-1. Empty array if no file / no boxes."""
    if not lbl.exists():
        return np.zeros((0, 5), np.float32)
    t = lbl.read_text(encoding="utf-8").split()
    return np.array(t, dtype=np.float32).reshape(-1, 5) if t else np.zeros((0, 5), np.float32)


def box_mask(boxes: np.ndarray, cls: int, size: int) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    b = boxes[boxes[:, 0] == cls]
    xy = np.stack([(b[:, 1] - b[:, 3] / 2), (b[:, 2] - b[:, 4] / 2), (b[:, 1] + b[:, 3] / 2), (b[:, 2] + b[:, 4] / 2)],
                  axis=1) if len(b) else np.zeros((0, 4))
    for x1, y1, x2, y2 in (np.clip(xy, 0, 1) * (size - 1)).round().astype(int):
        cv2.rectangle(m, (x1, y1), (x2, y2), 1, thickness=-1)
    return m
