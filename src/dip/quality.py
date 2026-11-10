"""Image quality measures (PSNR, SSIM, entropy, contrast, edges) and synthetic noise."""
import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PSNR_CAP = 100.0


def psnr(ref: np.ndarray, out: np.ndarray) -> float:
    if np.array_equal(ref, out):
        return PSNR_CAP
    return float(min(peak_signal_noise_ratio(ref, out, data_range=255), PSNR_CAP))


def ssim(ref: np.ndarray, out: np.ndarray) -> float:
    return float(structural_similarity(ref, out, channel_axis=2, data_range=255))


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def entropy(img: np.ndarray) -> float:
    """Shannon entropy of the grey histogram (bits): how much detail / how spread the grey levels are."""
    return float(shannon_entropy(gray(img)))


def rms_contrast(img: np.ndarray) -> float:
    """Standard deviation of grey levels (0-255): a simple global contrast measure."""
    return float(gray(img).astype(np.float32).std())


def mean_brightness(img: np.ndarray) -> float:
    return float(gray(img).mean())


def edge_strength(img: np.ndarray) -> float:
    g = gray(img).astype(np.float32)
    return float(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())


def edge_preservation(ref: np.ndarray, out: np.ndarray) -> float:
    """Mean Sobel edge strength after / before. 1.0 = edges kept; lower = edges blurred."""
    e = edge_strength(ref)
    return float(edge_strength(out) / e) if e > 0 else 1.0


def colour_shift(ref: np.ndarray, out: np.ndarray) -> float:
    """Mean |Δa| + |Δb| in LAB: how much the COLOUR (not brightness) changed. 0 = colours unchanged."""
    a = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(np.abs(a[..., 1:] - b[..., 1:]).sum(axis=2).mean())


def add_salt_pepper(img: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    mask = rng.random(img.shape[:2])
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def add_gaussian(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
