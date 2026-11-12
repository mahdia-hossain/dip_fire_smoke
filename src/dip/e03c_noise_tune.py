"""
E03c: choose the noise threshold for adaptive DIP preprocessing on validation images only.

    python run.py noise-tune [--smoke]

Each image is measured clean and with Gaussian noise (sigma 5, 10, 15, 25) added before the resize. The threshold
that best separates clean images from images with sigma >= 10 (balanced accuracy) is written to
configs/noise_tuned.yaml.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from src.common.cfg import load_config
from src.common.parallel import run_records
from src.common.runs import append_ledger, atomic_write_json, atomic_write_text, new_or_resume_run, setup_logging, write_meta
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.dip.noise import add_gaussian_noise, estimate_sigma
from src.dip.preprocess import load_rgb, resize

STAGE = "e03c_noise_tune"
LEVELS = [0, 5, 10, 15, 25]


def one(item: dict, cfg: dict) -> dict:
    rec = {"id": item["id"], "path": item["path"], "label": item["label"]}
    try:
        raw = load_rgb(PROJECT_ROOT / item["path"])
        rng = np.random.default_rng(item["id"])
        for s in LEVELS:
            img = raw if s == 0 else add_gaussian_noise(raw, s, rng)
            rec[f"sigma_est_{s}"] = estimate_sigma(resize(img, cfg["preprocess"]["size"]))
        rec["error"] = ""
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(CONFIG_DIR / "base.yaml")
    cfg = {"seed": cfg["seed"], "preprocess": cfg["preprocess"], "per_class": 300, "noisy_from": 10}
    run_dir, _ = new_or_resume_run(STAGE, cfg, root=RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT)
    log = setup_logging(run_dir, STAGE)
    write_meta(run_dir, cfg, data_files=[MANIFEST_DIR / "manifest.csv"])
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    val = man[man["split"] == "val"]
    n = 5 if a.smoke else cfg["per_class"]
    s = pd.concat([val[val["label"] == c].sample(frac=1, random_state=cfg["seed"]).head(n)
                   for c in ["fire", "smoke", "normal"]], ignore_index=True)[["path", "label"]]
    s.insert(0, "id", np.arange(len(s)))
    rec = run_records(run_dir, s.to_dict("records"), one, cfg, int(__import__("os").environ.get("DIP_WORKERS", "2")),
                      log, every=200)
    rec = rec[rec["error"] == ""]
    rec.to_csv(run_dir / "records.csv", index=False)
    clean = rec["sigma_est_0"].to_numpy()
    noisy = np.concatenate([rec[f"sigma_est_{l}"].to_numpy() for l in LEVELS if l >= cfg["noisy_from"]])
    grid = np.unique(np.round(np.concatenate([clean, noisy]), 2))
    bal = [(t, ((clean <= t).mean() + (noisy > t).mean()) / 2) for t in grid]
    thr, best = max(bal, key=lambda x: x[1])
    tab = pd.DataFrame({"level": LEVELS, "mean_sigma_est": [rec[f"sigma_est_{l}"].mean() for l in LEVELS],
                        "share_above_thr": [(rec[f"sigma_est_{l}"] > thr).mean() for l in LEVELS]}).round(4)
    (run_dir / "tables").mkdir(exist_ok=True), (run_dir / "figures").mkdir(exist_ok=True)
    tab.to_csv(run_dir / "tables" / "noise_levels.csv", index=False)
    (run_dir / "tables" / "noise_levels.md").write_text(tab.to_markdown(index=False), encoding="utf-8")
    log(f"Chosen sigma threshold {thr:.2f} (balanced accuracy {best:.4f})\n" + tab.to_string(index=False))
    import matplotlib.pyplot as plt
    from src.viz import style
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for i, l in enumerate(LEVELS):
        ax.hist(rec[f"sigma_est_{l}"], bins=40, alpha=0.55, color=style.OKABE_ITO[i], label=f"added σ = {l}")
    ax.axvline(thr, color="black", ls="--", lw=0.8)
    ax.set_xlabel("estimated noise σ (after resize)"), ax.set_ylabel("images"), ax.legend(fontsize=6, frameon=False)
    style.save(fig, run_dir / "figures" / "fig_noise_estimate", tab)
    plt.close(fig)
    text = (f"# Chosen by E03c on validation images: outputs/{STAGE}/{run_dir.name}\n"
            + yaml.safe_dump({"preprocess": {"adaptive": {"sigma_thr": round(float(thr), 2)}}}, sort_keys=False))
    atomic_write_text(run_dir / "noise_tuned.yaml", text)
    if not a.smoke:
        atomic_write_text(CONFIG_DIR / "noise_tuned.yaml", text)
    summary = {"sigma_thr": float(thr), "balanced_accuracy": float(best), "levels": tab.to_dict("records")}
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "results.json", {"metrics": summary, "records_file": "records.csv"})
    (run_dir / "records.partial.jsonl").unlink(missing_ok=True)
    if not a.smoke:
        append_ledger(STAGE, run_dir, f"E03c noise threshold sigma {thr:.2f} (balanced acc {best:.4f})")
    log("Done.")


if __name__ == "__main__":
    main()
