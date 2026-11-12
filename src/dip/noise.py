"""Noise level estimation (Immerkaer, 1996): Laplacian-difference mask, mean absolute response -> Gaussian sigma."""
import cv2
import numpy as np

_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def estimate_sigma(rgb: np.ndarray) -> float:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = np.abs(cv2.filter2D(g, -1, _MASK, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * r.mean() / 6.0)


def add_gaussian_noise(rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) + rng.normal(0, sigma, rgb.shape), 0, 255).astype(np.uint8)
