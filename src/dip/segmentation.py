"""Fire (YCbCr rule) and smoke (HSV + dark channel + Otsu) segmentation with morphological cleaning."""
import cv2
import numpy as np


def fire_rule(rgb: np.ndarray, tau: float) -> np.ndarray:
    """Fire-pixel rule in YCbCr (Celik & Demirel, 2009)."""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((y > cb) & (cr > cb) & (y > y.mean()) & (cb < cb.mean()) & (cr > cr.mean())
         & (np.abs(cb - cr) >= tau))
    return m.astype(np.uint8)


def dark_channel(rgb: np.ndarray, patch: int) -> np.ndarray:
    """He et al. dark channel: minimum over the colour channels, then minimum over a patch (= erosion)."""
    return cv2.erode(rgb.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))


def smoke_rule(rgb: np.ndarray, sat_max: int, val_min: int, dark_patch: int):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    dc = dark_channel(rgb, dark_patch)
    thr, hazy = cv2.threshold(dc, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (hsv[..., 1] < sat_max) & (hsv[..., 2] > val_min) & (hazy > 0)
    return m.astype(np.uint8), dc, float(thr)


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected regions smaller than min_area pixels (lookup table on the label image, no loop)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)


def clean(mask: np.ndarray, kernel: int, min_area: int) -> dict:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)
    return {"open": opened, "close": closed, "final": remove_small(closed, min_area)}


def segment(p2: np.ndarray, scfg: dict) -> dict:
    h, w = p2.shape[:2]
    min_area = max(1, int(scfg["morph"]["min_area_frac"] * h * w))
    fire_raw = fire_rule(p2, scfg["fire"]["tau_cb_cr"])
    f = clean(fire_raw, scfg["morph"]["kernel"], min_area)
    smoke_raw, dc, dc_thr = smoke_rule(p2, **scfg["smoke"])
    smoke_raw = smoke_raw & (1 - f["final"])
    s = clean(smoke_raw, scfg["morph"]["kernel"], min_area)
    return {"fire_raw": fire_raw, "fire_open": f["open"], "fire_close": f["close"], "fire": f["final"],
            "smoke_raw": smoke_raw, "smoke_open": s["open"], "smoke_close": s["close"], "smoke": s["final"],
            "dark": dc, "dark_thr": dc_thr}


def mask_stats(mask: np.ndarray) -> dict:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return {"frac": float(mask.mean()), "regions": int(n - 1),
            "largest_frac": float(areas.max() / mask.size) if len(areas) else 0.0}


def overlay(rgb: np.ndarray, fire: np.ndarray, smoke: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Fire = red, smoke = blue, blended on the image (for figures and predict.py)."""
    out = rgb.astype(np.float32)
    for m, col in ((fire, (255, 40, 0)), (smoke, (0, 120, 255))):
        sel = m.astype(bool)
        out[sel] = (1 - alpha) * out[sel] + alpha * np.array(col, np.float32)
    return out.clip(0, 255).astype(np.uint8)
