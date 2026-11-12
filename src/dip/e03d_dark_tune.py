"""
E03d: choose the brightness threshold for adaptive CLAHE on validation images only.

    python run.py dark-tune [--smoke]

Each image is measured as it is and darkened (x0.5, x0.35). The mean-grey threshold that best separates the
original images from the darkened ones (balanced accuracy) is written to configs/dark_tuned.yaml.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd
import yaml

from src.common.cfg import load_config
from src.common.parallel import run_records
from src.common.runs import append_ledger, atomic_write_json, atomic_write_text, new_or_resume_run, setup_logging, write_meta
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.dip.preprocess import load_rgb, resize

STAGE = "e03d_dark_tune"
FACTORS = [1.0, 0.5, 0.35]


def one(item: dict, cfg: dict) -> dict:
    rec = {"id": item["id"], "path": item["path"], "label": item["label"]}
    try:
        g = cv2.cvtColor(resize(load_rgb(PROJECT_ROOT / item["path"]), cfg["preprocess"]["size"]),
                         cv2.COLOR_RGB2GRAY).astype(np.float32)
        for f in FACTORS:
            rec[f"mean_{f}"] = float((g * f).mean())
        rec["error"] = ""
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    base = load_config(CONFIG_DIR / "base.yaml")
    cfg = {"seed": base["seed"], "preprocess": base["preprocess"], "per_class": 300}
    run_dir, _ = new_or_resume_run(STAGE, cfg, root=RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT)
    log = setup_logging(run_dir, STAGE)
    write_meta(run_dir, cfg, data_files=[MANIFEST_DIR / "manifest.csv"])
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    val = man[man["split"] == "val"]
    n = 5 if a.smoke else cfg["per_class"]
    s = pd.concat([val[val["label"] == c].sample(frac=1, random_state=cfg["seed"]).head(n)
                   for c in ["fire", "smoke", "normal"]], ignore_index=True)[["path", "label"]]
    s.insert(0, "id", np.arange(len(s)))
    rec = run_records(run_dir, s.to_dict("records"), one, cfg, int(os.environ.get("DIP_WORKERS", "2")), log, every=200)
    rec = rec[rec["error"] == ""]
    rec.to_csv(run_dir / "records.csv", index=False)
    orig = rec["mean_1.0"].to_numpy()
    dark = np.concatenate([rec[f"mean_{f}"].to_numpy() for f in FACTORS[1:]])
    grid = np.unique(np.round(np.concatenate([orig, dark]), 1))
    thr, best = max(((t, ((orig >= t).mean() + (dark < t).mean()) / 2) for t in grid), key=lambda x: x[1])
    tab = pd.DataFrame({"factor": FACTORS, "mean_grey": [rec[f"mean_{f}"].mean() for f in FACTORS],
                        "share_below_thr": [(rec[f"mean_{f}"] < thr).mean() for f in FACTORS]}).round(4)
    (run_dir / "tables").mkdir(exist_ok=True), (run_dir / "figures").mkdir(exist_ok=True)
    tab.to_csv(run_dir / "tables" / "dark_levels.csv", index=False)
    log(f"Chosen dark threshold {thr:.1f} (balanced accuracy {best:.4f})\n" + tab.to_string(index=False))
    import matplotlib.pyplot as plt
    from src.viz import style
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for i, f in enumerate(FACTORS):
        ax.hist(rec[f"mean_{f}"], bins=40, alpha=0.55, color=style.OKABE_ITO[i], label=f"brightness x{f}")
    ax.axvline(thr, color="black", ls="--", lw=0.8)
    ax.set_xlabel("mean grey level"), ax.set_ylabel("images"), ax.legend(fontsize=6, frameon=False)
    style.save(fig, run_dir / "figures" / "fig_dark_estimate", tab)
    plt.close(fig)
    text = (f"# Chosen by E03d on validation images: outputs/{STAGE}/{run_dir.name}\n"
            + yaml.safe_dump({"preprocess": {"adaptive": {"dark_thr": round(float(thr), 1)}}}, sort_keys=False))
    atomic_write_text(run_dir / "dark_tuned.yaml", text)
    if not a.smoke:
        atomic_write_text(CONFIG_DIR / "dark_tuned.yaml", text)
    summary = {"dark_thr": float(thr), "balanced_accuracy": float(best), "levels": tab.to_dict("records")}
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "results.json", {"metrics": summary, "records_file": "records.csv"})
    (run_dir / "records.partial.jsonl").unlink(missing_ok=True)
    if not a.smoke:
        append_ledger(STAGE, run_dir, f"E03d dark threshold {thr:.1f} (balanced acc {best:.4f})")
    log("Done.")


if __name__ == "__main__":
    main()
