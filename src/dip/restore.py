"""Lab-manual DIP operations used in the E12 restoration study (all loop-free OpenCV/NumPy)."""
import cv2
import numpy as np

K_WEIGHTED = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0
K_LAPLACE = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
K_POINT = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)


def salt_pepper(img: np.ndarray, amount: float, rng) -> np.ndarray:
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out


def gaussian_noise(img: np.ndarray, sigma: float, rng) -> np.ndarray:
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def min_filter(img, k):
    return cv2.erode(img, np.ones((k, k), np.uint8))


def max_filter(img, k):
    return cv2.dilate(img, np.ones((k, k), np.uint8))


def midpoint_filter(img, k):
    return ((min_filter(img, k).astype(np.uint16) + max_filter(img, k)) // 2).astype(np.uint8)


DENOISERS = {
    "none": lambda x: x,
    "average_3": lambda x: cv2.blur(x, (3, 3)),
    "average_5": lambda x: cv2.blur(x, (5, 5)),
    "average_7": lambda x: cv2.blur(x, (7, 7)),
    "weighted_avg_3": lambda x: cv2.filter2D(x, -1, K_WEIGHTED),
    "gaussian_3": lambda x: cv2.GaussianBlur(x, (3, 3), 0),
    "gaussian_5": lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    "gaussian_7": lambda x: cv2.GaussianBlur(x, (7, 7), 0),
    "median_3": lambda x: cv2.medianBlur(x, 3),
    "median_5": lambda x: cv2.medianBlur(x, 5),
    "median_7": lambda x: cv2.medianBlur(x, 7),
    "min_3": lambda x: min_filter(x, 3),
    "max_3": lambda x: max_filter(x, 3),
    "midpoint_3": lambda x: midpoint_filter(x, 3),
    "bilateral": lambda x: cv2.bilateralFilter(x, 5, 50, 50),
    "nlmeans": lambda x: cv2.fastNlMeansDenoisingColored(x, None, 10, 10, 7, 21),
}


def _lut(f):
    return np.clip(f(np.arange(256, dtype=np.float32)), 0, 255).astype(np.uint8)


LUT_LOG = _lut(lambda r: 255.0 / np.log(256.0) * np.log1p(r))
LUT_GAMMA_04 = _lut(lambda r: 255.0 * (r / 255.0) ** 0.4)
LUT_GAMMA_06 = _lut(lambda r: 255.0 * (r / 255.0) ** 0.6)


def on_l(img: np.ndarray, fn) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = fn(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def contrast_stretch(ch: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    a, b = np.percentile(ch, [lo, hi])
    return np.clip((ch.astype(np.float32) - a) * 255.0 / max(b - a, 1.0), 0, 255).astype(np.uint8)


def match_lut(src: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(np.bincount(src.ravel(), minlength=256)).astype(np.float64)
    cdf /= cdf[-1]
    return np.searchsorted(ref_cdf, cdf).clip(0, 255).astype(np.uint8)[src]


def enhancers(ref_cdf: np.ndarray) -> dict:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return {
        "none": lambda x: x,
        "log": lambda x: on_l(x, lambda c: LUT_LOG[c]),
        "gamma_0.4": lambda x: on_l(x, lambda c: LUT_GAMMA_04[c]),
        "gamma_0.6": lambda x: on_l(x, lambda c: LUT_GAMMA_06[c]),
        "contrast_stretch": lambda x: on_l(x, contrast_stretch),
        "hist_equalization": lambda x: on_l(x, cv2.equalizeHist),
        "hist_matching": lambda x: on_l(x, lambda c: match_lut(c, ref_cdf)),
        "clahe": lambda x: on_l(x, clahe.apply),
    }


def laplace_sharpen(img):
    return np.clip(img.astype(np.float32) + cv2.filter2D(img.astype(np.float32), -1, K_LAPLACE), 0, 255).astype(np.uint8)


def high_boost(img, k: float = 1.0):
    blur = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
    return np.clip(img + k * (img.astype(np.float32) - blur), 0, 255).astype(np.uint8)


SHARPENERS = {
    "none": lambda x: x,
    "laplace_sharpen": laplace_sharpen,
    "unsharp_mask": lambda x: high_boost(x, 1.0),
    "high_boost_2": lambda x: high_boost(x, 2.0),
}


def isolated_point_share(rgb: np.ndarray, thr: float = 600.0) -> float:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float((np.abs(cv2.filter2D(g, -1, K_POINT, borderType=cv2.BORDER_REFLECT)) > thr).mean())


def edge_f1(a: np.ndarray, b: np.ndarray) -> float:
    """Canny edges of a restored image vs the clean image, with 1-pixel tolerance."""
    ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), 100, 200) > 0
    k = np.ones((3, 3), np.uint8)
    da, db = cv2.dilate(ea.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    p = (ea & db).sum() / max(ea.sum(), 1)
    r = (eb & da).sum() / max(eb.sum(), 1)
    return float(2 * p * r / max(p + r, 1e-9))
