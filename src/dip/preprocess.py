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
    """
    Adaptive mode, pcfg["adaptive"] = {"sigma_thr": t, "dark_thr": d (optional)}:
    noisy (estimated sigma > t) -> denoise + contrast; else dark (mean grey < d) -> contrast only; else unchanged.
    Noise-type mode (adds "points_thr", "heavy_thr", chosen in E12b): clean -> unchanged; salt-and-pepper
    (isolated-point share > points_thr) -> median 3x3; heavy Gaussian (sigma > heavy_thr) -> Gaussian 7x7;
    other Gaussian -> bilateral; then CLAHE on L.
    """
    p0 = resize(img, pcfg["size"])
    ad = pcfg.get("adaptive")
    if ad and "points_thr" in ad:
        from src.dip.noise import estimate_sigma
        from src.dip.restore import isolated_point_share
        sigma = estimate_sigma(p0)
        if sigma <= ad["sigma_thr"]:
            return {"resized": p0, "denoised": p0, "enhanced": p0}
        from src.dip.restore import DENOISERS
        if isolated_point_share(p0) > ad["points_thr"]:
            p1 = DENOISERS[ad.get("sp_filter", "median_3")](p0)
        elif sigma > ad["heavy_thr"]:
            p1 = DENOISERS[ad.get("heavy_filter", "gaussian_7")](p0)
        else:
            p1 = DENOISERS[ad.get("light_filter", "bilateral")](p0)
        return {"resized": p0, "denoised": p1, "enhanced": enhance(p1, **pcfg["contrast"])}
    if ad:
        from src.dip.noise import estimate_sigma
        if estimate_sigma(p0) <= ad["sigma_thr"]:
            if "dark_thr" in ad and float(cv2.cvtColor(p0, cv2.COLOR_RGB2GRAY).mean()) < ad["dark_thr"]:
                return {"resized": p0, "denoised": p0, "enhanced": enhance(p0, **pcfg["contrast"])}
            return {"resized": p0, "denoised": p0, "enhanced": p0}
    p1 = denoise(p0, **pcfg["denoise"])
    p2 = enhance(p1, **pcfg["contrast"])
    return {"resized": p0, "denoised": p1, "enhanced": p2}


def preprocess(img: np.ndarray, pcfg: dict) -> np.ndarray:
    return preprocess_steps(img, pcfg)["enhanced"]
