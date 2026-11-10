"""Preprocessing: resize, denoise, contrast enhancement."""
import cv2
import numpy as np


def load_rgb(path) -> np.ndarray:
    """Read an image as uint8 RGB. np.fromfile + imdecode also works for Windows paths with non-ASCII letters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """INTER_AREA is the recommended OpenCV method for shrinking (averages pixels, avoids aliasing)."""
    interp = cv2.INTER_AREA if min(img.shape[:2]) >= size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def denoise(img: np.ndarray, method: str = "median", ksize: int = 3, **kw) -> np.ndarray:
    if method == "none":
        return img.copy()
    if method == "median":
        return cv2.medianBlur(img, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(img, kw.get("d", 5), kw.get("sigma_color", 50), kw.get("sigma_space", 50))
    raise ValueError(f"unknown denoise method: {method}")


def enhance(img: np.ndarray, method: str = "clahe_l", clip_limit: float = 2.0, tile: int = 8, **kw) -> np.ndarray:
    if method == "none":
        return img.copy()
    if method == "he_rgb":
        return np.dstack([cv2.equalizeHist(img[..., c]) for c in range(3)])
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if method == "he_l":
        lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    elif method == "clahe_l":
        lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(lab[..., 0])
    else:
        raise ValueError(f"unknown contrast method: {method}")
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_steps(img: np.ndarray, pcfg: dict) -> dict:
    p0 = resize(img, pcfg["size"])
    p1 = denoise(p0, **pcfg["denoise"])
    p2 = enhance(p1, **pcfg["contrast"])
    return {"resized": p0, "denoised": p1, "enhanced": p2}


def preprocess(img: np.ndarray, pcfg: dict) -> np.ndarray:
    return preprocess_steps(img, pcfg)["enhanced"]
