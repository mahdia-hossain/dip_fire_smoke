"""DIP steps used at inference: resize, noise check, adaptive filtering (median + CLAHE), colour masks, input vs output change."""
import cv2
import numpy as np

NOISE_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)

def load_rgb(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize(img: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if min(img.shape[:2]) >= size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)

def noise_sigma(rgb: np.ndarray) -> float:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = np.abs(cv2.filter2D(g, -1, NOISE_MASK, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * r.mean() / 6.0)

def clahe_l(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8, **_) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def adaptive(p0: np.ndarray, ad: dict, contrast: dict):
    sigma = noise_sigma(p0)
    if sigma <= ad["sigma_thr"]:
        return sigma, "clean", "none", p0, p0
    p1 = cv2.medianBlur(p0, 3)
    return sigma, "noisy", "median_3", p1, clahe_l(p1, **contrast)

def _clean_mask(mask: np.ndarray, kernel: int, min_area: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    m = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8)

def colour_masks(p2: np.ndarray, seg: dict):
    h, w = p2.shape[:2]
    min_area = max(1, int(seg["morph"]["min_area_frac"] * h * w))
    ycc = cv2.cvtColor(p2, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    fire = ((y > cb) & (cr > cb) & (y > y.mean()) & (cb < cb.mean()) & (cr > cr.mean())
            & (np.abs(cb - cr) >= seg["fire"]["tau_cb_cr"])).astype(np.uint8)
    fire = _clean_mask(fire, seg["morph"]["kernel"], min_area)
    sm = seg["smoke"]
    hsv = cv2.cvtColor(p2, cv2.COLOR_RGB2HSV)
    dc = cv2.erode(p2.min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (sm["dark_patch"],) * 2))
    thr, _ = cv2.threshold(dc, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smoke = ((hsv[..., 1] < sm["sat_max"]) & (hsv[..., 2] > sm["val_min"]) & (dc > thr)).astype(np.uint8) & (1 - fire)
    return fire, _clean_mask(smoke, seg["morph"]["kernel"], min_area)

def overlay(rgb: np.ndarray, fire: np.ndarray, smoke: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32)
    for m, col in ((fire, (255, 40, 0)), (smoke, (0, 120, 255))):
        sel = m.astype(bool)
        out[sel] = (1 - alpha) * out[sel] + alpha * np.array(col, np.float32)
    return out.clip(0, 255).astype(np.uint8)

PSNR_CAP = 100.0

def psnr(ref: np.ndarray, out: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - out.astype(np.float64)) ** 2)
    return PSNR_CAP if mse == 0 else float(min(10 * np.log10(255.0 ** 2 / mse), PSNR_CAP))

def ssim(ref: np.ndarray, out: np.ndarray) -> float:
    a, b = ref.astype(np.float64), out.astype(np.float64)
    f = lambda x: cv2.blur(x, (7, 7), borderType=cv2.BORDER_REFLECT)
    n, c1, c2 = 49 / 48, (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = f(a), f(b)
    va, vb, cab = n * (f(a * a) - ma * ma), n * (f(b * b) - mb * mb), n * (f(a * b) - ma * mb)
    s = ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s[3:-3, 3:-3].mean())

def change(ref: np.ndarray, out: np.ndarray, level: int = 10) -> dict:
    d = cv2.absdiff(ref, out).max(axis=2)
    return {"psnr_db": round(psnr(ref, out), 2), "ssim": round(ssim(ref, out), 4),
            "mean_abs_diff": round(float(d.mean()), 2), "changed_pct": round(float((d > level).mean() * 100), 2)}

def diff_map(ref: np.ndarray, out: np.ndarray, gain: float = 4.0) -> np.ndarray:
    d = np.clip(cv2.absdiff(ref, out).max(axis=2).astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(d, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
