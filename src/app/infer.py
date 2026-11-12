"""Inference core shared by predict.py and the demo app: final model + every intermediate DIP step."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.common.cfg import load_config
from src.config import CKPT_ROOT, CONFIG_DIR, RESULTS_ROOT
from src.deep.data import CLASSES, photometric
from src.deep.models import build_model
from src.dip.noise import estimate_sigma
from src.dip.preprocess import enhance, preprocess, resize
from src.dip.restore import DENOISERS, K_POINT, isolated_point_share
from src.dip.segmentation import overlay, segment

CANDIDATES = {"CA3": "e07h_mobilevit_noisetype_adaptive", "CA": "e07d_mobilevit_adaptive_noiseaug"}


def find_final(root: Path = RESULTS_ROOT) -> Path:
    """The finished CA / CA3 run with the highest validation macro-F1."""
    best = None
    for stage in CANDIDATES.values():
        for run in (root / stage).glob("run_*"):
            sj = run / "summary.json"
            if (run / "results.json").exists() and sj.exists():
                s = json.loads(sj.read_text())
                if best is None or s["best_val_macro_f1"] > best[0]:
                    best = (s["best_val_macro_f1"], run)
    if best is None:
        raise SystemExit(f"No finished CA or CA3 run under {root}. Train one first.")
    return best[1]


def checkpoint_of(run: Path) -> Path:
    """best.pt of a run, looked up inside this project's checkpoints folder (works after moving the project)."""
    p = CKPT_ROOT / run.parent.name / run.name / "best.pt"
    if not p.exists():
        raise SystemExit(f"Model file not found: {p}")
    return p


class Pipeline:
    def __init__(self, run: Path):
        s = json.loads((run / "summary.json").read_text())
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck_path = CKPT_ROOT / run.parent.name / run.name / "best.pt"
        if not ck_path.exists():
            ck_path = Path(s["checkpoints"]) / "best.pt"
        if not ck_path.exists():
            raise SystemExit(f"Model file not found: {ck_path}. Copy the checkpoints folder too.")
        ck = torch.load(ck_path, map_location=self.dev, weights_only=False)
        self.cfg = ck["cfg"]
        self.name, self.run, self.val_f1 = s["model"], run, s["best_val_macro_f1"]
        self.model = build_model(self.cfg["model"]["kind"], self.cfg["model"]["arch"], pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.dev).eval()
        self.photo = photometric(False)
        base = load_config(CONFIG_DIR / "base.yaml")
        self.dip_pre, self.seg_cfg = base["preprocess"], load_config(CONFIG_DIR / "deep_train.yaml")["segmentation"]

    def steps(self, rgb: np.ndarray) -> dict:
        pc = self.cfg["preprocess"]
        ad = pc.get("adaptive", {})
        p0 = resize(rgb, pc["size"])
        gray = cv2.cvtColor(p0, cv2.COLOR_RGB2GRAY)
        resp = np.abs(cv2.filter2D(gray.astype(np.float32), -1, K_POINT, borderType=cv2.BORDER_REFLECT))
        sigma, points = estimate_sigma(p0), isolated_point_share(p0)
        if sigma <= ad.get("sigma_thr", -1):
            decision, filt, p1 = "clean", "none", p0
        elif "points_thr" in ad and points > ad["points_thr"]:
            filt = ad.get("sp_filter", "median_3")
            decision, p1 = "salt-and-pepper noise", DENOISERS[filt](p0)
        elif "heavy_thr" in ad and sigma > ad["heavy_thr"]:
            filt = ad.get("heavy_filter", "gaussian_7")
            decision, p1 = "heavy grain", DENOISERS[filt](p0)
        elif "points_thr" in ad:
            filt = ad.get("light_filter", "bilateral")
            decision, p1 = "light grain", DENOISERS[filt](p0)
        else:
            decision, filt, p1 = "noisy", "median 3x3", cv2.medianBlur(p0, 3)
        p2 = p1 if decision == "clean" else enhance(p1, **pc["contrast"])
        model_in = preprocess(rgb, pc)
        assert np.array_equal(model_in, p2), "displayed steps differ from the model's preprocessing"
        with torch.no_grad():
            x = self.photo(torch.from_numpy(np.ascontiguousarray(model_in)).permute(2, 0, 1))[None].to(self.dev)
            prob = torch.softmax(self.model(x).float(), 1)[0].cpu().numpy()
        label = CLASSES[int(prob.argmax())]
        seg = segment(preprocess(rgb, self.dip_pre), self.seg_cfg)
        fire = seg["fire"] if label == "fire" else np.zeros_like(seg["fire"])
        smoke = seg["smoke"] if label == "smoke" else np.zeros_like(seg["smoke"])
        hl = overlay(model_in, fire, smoke)
        return {
            "label": label, "confidence": float(prob.max()), "probs": dict(zip(CLASSES, map(float, prob))),
            "sigma": float(sigma), "points_share": float(points), "decision": decision, "filter": filt,
            "thresholds": ad, "model": self.name,
            "images": [
                ("1. Resized 256x256", p0),
                (f"2. Noise check: sigma = {sigma:.2f}", gray),
                (f"3. Isolated points: {points * 100:.3f}% of pixels", np.clip(resp / 4, 0, 255).astype(np.uint8)),
                (f"4. Decision: {decision} -> {filt}", p1),
                ("5. After CLAHE on L" if decision != "clean" else "5. No change (clean photo)", p2),
                (f"6. Answer: {label.upper()} {prob.max() * 100:.0f}%", hl),
            ],
        }


def explain(r: dict) -> str:
    t = r["thresholds"]
    lines = [f"**Answer: {r['label'].upper()} ({r['confidence'] * 100:.1f}% confident)** — model `{r['model']}`", "",
             f"1. The photo is resized to 256 x 256 pixels.",
             f"2. Noise check (Immerkaer 3x3 mask): sigma = **{r['sigma']:.2f}** "
             f"(threshold {t.get('sigma_thr', '-')}). " + ("Below the threshold, so the photo is **clean**."
                                                            if r["decision"] == "clean" else "Above the threshold, so the photo is **noisy**.")]
    if "points_thr" in t:
        lines.append(f"3. Isolated point check: {r['points_share'] * 100:.3f}% of pixels are lonely dots "
                     f"(threshold {t['points_thr'] * 100:.3f}%).")
    else:
        lines.append("3. Isolated point check: shown for information (this model only decides clean vs noisy).")
    lines += [f"4. Decision: **{r['decision']}**, filter used: **{r['filter']}**.",
              "5. " + ("Clean photos are passed on unchanged, so no detail is lost." if r["decision"] == "clean"
                       else "CLAHE on the brightness channel restores local contrast without changing colours."),
              "6. MobileViT-XS scores: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in r["probs"].items()) + ".",
              "", "_The red/blue highlight comes from the DIP colour masks and is approximate._"]
    return "\n".join(lines)
