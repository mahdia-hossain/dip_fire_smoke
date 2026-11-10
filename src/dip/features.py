"""16 hand-crafted DIP features in five groups: colour, brightness, texture, haze, edges."""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

FEATURE_GROUPS = {
    "colour": ["fire_frac", "smoke_frac", "redness", "saturation"],
    "brightness": ["brightness", "rms_contrast", "entropy"],
    "texture": ["lbp_entropy", "lbp_flat_frac", "glcm_contrast", "glcm_homogeneity"],
    "haze": ["dark_mean", "dark_high_frac"],
    "edges": ["sobel_mean", "edge_frac", "fire_edge"],
}
FEATURE_NAMES = [f for g in FEATURE_GROUPS.values() for f in g]


def _entropy(hist: np.ndarray) -> float:
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def extract(p2: np.ndarray, seg: dict, fcfg: dict) -> dict:
    gray = cv2.cvtColor(p2, cv2.COLOR_RGB2GRAY)
    ycc = cv2.cvtColor(p2, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(p2, cv2.COLOR_RGB2HSV)
    f = {}
    f["fire_frac"] = float(seg["fire"].mean())
    f["smoke_frac"] = float(seg["smoke"].mean())
    f["redness"] = float((ycc[..., 1] - ycc[..., 2]).mean() / 255.0)
    f["saturation"] = float(hsv[..., 1].mean() / 255.0)
    f["brightness"] = float(gray.mean() / 255.0)
    f["rms_contrast"] = float(gray.std() / 255.0)
    f["entropy"] = _entropy(np.bincount(gray.ravel(), minlength=256))
    P, R = fcfg["lbp"]["P"], fcfg["lbp"]["R"]
    lbp = local_binary_pattern(gray, P, R, method="uniform").astype(np.int64)
    h = np.bincount(lbp.ravel(), minlength=P + 2)
    f["lbp_entropy"] = _entropy(h)
    f["lbp_flat_frac"] = float((h[0] + h[P]) / h.sum())
    lv = fcfg["glcm"]["levels"]
    q = (gray.astype(np.int32) * lv // 256).astype(np.uint8)
    glcm = graycomatrix(q, [fcfg["glcm"]["distance"]], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=lv,
                        symmetric=True, normed=True)
    f["glcm_contrast"] = float(graycoprops(glcm, "contrast").mean() / (lv - 1) ** 2)
    f["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
    f["dark_mean"] = float(seg["dark"].mean() / 255.0)
    f["dark_high_frac"] = float((seg["dark"] > seg["dark_thr"]).mean())
    g = gray.astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    f["sobel_mean"] = float(mag.mean() / 255.0)
    f["edge_frac"] = float((mag > fcfg["edge_thr"]).mean())
    fire = seg["fire"].astype(bool)
    f["fire_edge"] = float(mag[fire].mean() / 255.0) if fire.any() else 0.0
    return f
