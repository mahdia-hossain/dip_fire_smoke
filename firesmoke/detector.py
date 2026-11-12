"""FireSmokeDetector: load one exported model file and predict with every DIP step kept."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

from . import dip


class _Net(torch.nn.Module):
    def __init__(self, arch: str, n_classes: int):
        super().__init__()
        self.net = timm.create_model(arch, pretrained=False, num_classes=n_classes)

    def forward(self, x):
        return self.net(x)


@dataclass
class Result:
    image: str
    label: str
    confidence: float
    probs: dict
    sigma: float
    points_share: float
    decision: str
    filter: str
    thresholds: dict
    model: str
    steps: list = field(repr=False, default_factory=list)

    def explain(self, show: bool = True) -> str:
        t = self.thresholds
        noisy = self.decision != "clean"
        lines = [f"Answer: {self.label.upper()} ({self.confidence * 100:.1f}% confident), model {self.model}",
                 "1. The photo is resized to 256 x 256 pixels.",
                 f"2. Noise check: sigma = {self.sigma:.2f} (threshold {t['sigma_thr']}), so the photo is "
                 + ("noisy." if noisy else "clean."),
                 f"3. Isolated points: {self.points_share * 100:.3f}% of pixels are lonely dots"
                 + (f" (threshold {t['points_thr'] * 100:.3f}%)." if "points_thr" in t else "."),
                 f"4. Decision: {self.decision} -> filter: {self.filter}.",
                 "5. " + ("CLAHE on the brightness channel restores contrast without changing colours." if noisy
                          else "Clean photos are passed on unchanged, so no detail is lost."),
                 "6. MobileViT-XS scores: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in self.probs.items()) + ".",
                 "(The red/blue highlight comes from DIP colour masks and is approximate.)"]
        text = "\n".join(lines)
        if show:
            print(text)
        return text

    def figure(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(self.steps), figsize=(2.2 * len(self.steps), 2.8))
        for ax, (title, im) in zip(axes, self.steps):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
        fig.tight_layout()
        return fig

    def show_steps(self):
        import matplotlib.pyplot as plt
        self.figure()
        plt.show()

    def save(self, out_dir="out") -> Path:
        import matplotlib.pyplot as plt
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(self.image).stem
        fig = self.figure()
        fig.savefig(out / f"{stem}_steps.png", dpi=150)
        plt.close(fig)
        res = cv2.cvtColor(self.steps[-1][1], cv2.COLOR_RGB2BGR)
        cv2.putText(res, f"{self.label.upper()} {self.confidence * 100:.0f}%", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}_result.png"), res)
        (out / f"{stem}_explanation.md").write_text(self.explain(show=False), encoding="utf-8")
        info = {k: v for k, v in self.__dict__.items() if k != "steps"}
        (out / f"{stem}_result.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return out


class FireSmokeDetector:
    def __init__(self, weights, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        b = torch.load(weights, map_location=self.device, weights_only=False)
        self.meta = {k: v for k, v in b.items() if k != "state_dict"}
        self.classes, self.pre, self.seg = b["classes"], b["preprocess"], b["segmentation"]
        self.net = _Net(b["arch"], len(self.classes))
        self.net.load_state_dict(b["state_dict"])
        self.net.to(self.device).eval()
        self.mean = torch.tensor(b["mean"]).view(3, 1, 1)
        self.std = torch.tensor(b["std"]).view(3, 1, 1)

    @torch.no_grad()
    def predict(self, source) -> Result:
        """source: a file path or an RGB uint8 array."""
        rgb = dip.load_rgb(source) if not isinstance(source, np.ndarray) else source
        p0 = dip.resize(rgb, self.pre["size"])
        sigma, points, decision, filt, p1, p2 = dip.adaptive(p0, self.pre["adaptive"], self.pre["contrast"])
        x = (torch.from_numpy(np.ascontiguousarray(p2)).permute(2, 0, 1).float() / 255.0 - self.mean) / self.std
        prob = torch.softmax(self.net(x[None].to(self.device)).float(), 1)[0].cpu().numpy()
        label = self.classes[int(prob.argmax())]
        fire, smoke = dip.colour_masks(clahe_ready(p0, self.pre), self.seg)
        hl = dip.overlay(p2, fire if label == "fire" else 0 * fire, smoke if label == "smoke" else 0 * smoke)
        resp = dip.point_response(p0)
        steps = [("1. Resized 256x256", p0),
                 (f"2. Noise sigma = {sigma:.2f}", cv2.cvtColor(p0, cv2.COLOR_RGB2GRAY)),
                 (f"3. Isolated points {points * 100:.3f}%", np.clip(resp / 4, 0, 255).astype(np.uint8)),
                 (f"4. {decision}: {filt}", p1),
                 ("5. After CLAHE" if decision != "clean" else "5. No change (clean)", p2),
                 (f"6. {label.upper()} {prob.max() * 100:.0f}%", hl)]
        return Result(str(source) if not isinstance(source, np.ndarray) else "array", label, float(prob.max()),
                      dict(zip(self.classes, map(float, prob))), sigma, points, decision, filt,
                      self.pre["adaptive"], self.meta["model_name"], steps)


def clahe_ready(p0: np.ndarray, pre: dict) -> np.ndarray:
    """The colour masks were tuned on median 3x3 + CLAHE images, so the highlight uses that view."""
    return dip.clahe_l(cv2.medianBlur(p0, 3), **pre["contrast"])
