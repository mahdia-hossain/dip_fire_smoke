"""
New image in -> result out, using the final model (the finished CA / CA3 run with the best validation score).

    python predict.py --image photo.jpg
    python predict.py --folder E:\\my_photos
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from src.app.infer import Pipeline, explain, find_final
from src.common.runs import append_ledger, new_run_dir
from src.config import RESULTS_ROOT
from src.dip.preprocess import load_rgb

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_steps(r: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.8))
    for ax, (t, im) in zip(axes, r["images"]):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255), ax.axis("off"), ax.set_title(t, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image")
    g.add_argument("--folder")
    ap.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    root = RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT
    pipe = Pipeline(find_final(root))
    paths = [Path(a.image)] if a.image else sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in EXT)
    out = new_run_dir("e11_predict", root=root)
    print(f"Model {pipe.name} (val macro-F1 {pipe.val_f1}), results -> {out}")
    rows = []
    for p in paths:
        r = pipe.steps(load_rgb(p))
        hl = cv2.cvtColor(r["images"][-1][1], cv2.COLOR_RGB2BGR)
        cv2.putText(hl, f"{r['label'].upper()} {r['confidence'] * 100:.0f}%", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{p.stem}_result.png"), hl)
        save_steps(r, out / f"{p.stem}_steps.png")
        info = {k: v for k, v in r.items() if k != "images"}
        (out / f"{p.stem}_result.json").write_text(json.dumps({"image": str(p), **info}, indent=2))
        (out / f"{p.stem}_explanation.md").write_text(explain(r), encoding="utf-8")
        rows.append({"image": p.name, **{k: info[k] for k in ("label", "confidence", "sigma", "decision", "filter")}})
        print(f"  {p.name}: {r['label']} {r['confidence'] * 100:.1f}%  (sigma {r['sigma']:.2f}, {r['decision']}, {r['filter']})")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    if not a.smoke:
        append_ledger("e11_predict", out, f"predict.py on {len(rows)} images with {pipe.name}")
    print("Done.")


if __name__ == "__main__":
    main()
