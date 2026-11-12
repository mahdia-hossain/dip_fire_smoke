"""
E12b: thresholds for the noise-type adaptive pipeline (CA3), from the E12 validation records (no new images).

    python run.py noisetype-tune

points_thr: isolated-point share that separates salt-and-pepper photos from clean and Gaussian-noise photos.
heavy_thr:  noise sigma that separates Gaussian sigma 25 from sigma 10 photos.
Filters:    the filter with the best mean SSIM for each noise type (dots, light grain, heavy grain).
Both are chosen by balanced accuracy and written to configs/noisetype_tuned.yaml together with the E03c sigma_thr.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from src.common.runs import append_ledger, atomic_write_json, atomic_write_text, new_or_resume_run, setup_logging, write_meta
from src.config import CONFIG_DIR, RESULTS_ROOT

STAGE = "e12b_noisetype_tune"


def best_split(neg: np.ndarray, pos: np.ndarray):
    grid = np.unique(np.concatenate([neg, pos]))
    return max(((float(t), ((neg <= t).mean() + (pos > t).mean()) / 2) for t in grid), key=lambda x: x[1])


def main() -> None:
    runs = [p for p in sorted((RESULTS_ROOT / "e12_restoration").glob("run_*/records.csv"))
            if (p.parent / "results.json").exists()]
    nt = CONFIG_DIR / "noise_tuned.yaml"
    if not runs or not nt.exists():
        sys.exit("Needs a finished E12 run and configs/noise_tuned.yaml. Run:  python run.py restore-study / noise-tune")
    rec = pd.read_csv(runs[-1])
    sigma_thr = yaml.safe_load(nt.read_text())["preprocess"]["adaptive"]["sigma_thr"]
    cfg = {"e12_records": str(runs[-1]), "sigma_thr": sigma_thr}
    run_dir, _ = new_or_resume_run(STAGE, cfg)
    log = setup_logging(run_dir, STAGE)
    write_meta(run_dir, cfg, data_files=[runs[-1]])
    sp = np.r_[rec["det|saltpepper_2pct|points"], rec["det|saltpepper_5pct|points"]]
    other = np.r_[rec["det|clean|points"], rec["det|gauss_10|points"], rec["det|gauss_25|points"]]
    points_thr, acc_p = best_split(other, sp)
    heavy_thr, acc_h = best_split(rec["det|gauss_10|sigma"].to_numpy(), rec["det|gauss_25|sigma"].to_numpy())

    def best_filter(kinds):
        cols = {c.split("|")[2]: [f"noise|{k}|{c.split('|')[2]}|ssim" for k in kinds]
                for c in rec.columns if c.startswith(f"noise|{kinds[0]}|") and c.endswith("|ssim")}
        scores = {m: float(rec[cs].mean().mean()) for m, cs in cols.items() if m != "none"}
        m = max(scores, key=scores.get)
        return m, round(scores[m], 4)

    sp_f, sp_s = best_filter(["saltpepper_2pct", "saltpepper_5pct"])
    light_f, light_s = best_filter(["gauss_10"])
    heavy_f, heavy_s = best_filter(["gauss_25"])
    log(f"Best filters (mean SSIM vs clean): salt-and-pepper {sp_f} ({sp_s}), light grain {light_f} ({light_s}), "
        f"heavy grain {heavy_f} ({heavy_s})")
    tab = pd.DataFrame([{"threshold": "points_thr", "value": points_thr, "balanced_accuracy": acc_p,
                         "separates": "salt-and-pepper vs clean/Gaussian"},
                        {"threshold": "heavy_thr", "value": heavy_thr, "balanced_accuracy": acc_h,
                         "separates": "Gaussian sigma 25 vs sigma 10"}]).round(6)
    (run_dir / "tables").mkdir(exist_ok=True)
    tab.to_csv(run_dir / "tables" / "thresholds.csv", index=False)
    log("Chosen thresholds:\n" + tab.to_string(index=False))
    text = (f"# Chosen by E12b on validation records: outputs/{STAGE}/{run_dir.name}\n" + yaml.safe_dump(
        {"preprocess": {"adaptive": {"sigma_thr": sigma_thr, "points_thr": round(points_thr, 6),
                                     "heavy_thr": round(heavy_thr, 4), "sp_filter": sp_f,
                                     "light_filter": light_f, "heavy_filter": heavy_f}}}, sort_keys=False))
    atomic_write_text(run_dir / "noisetype_tuned.yaml", text)
    atomic_write_text(CONFIG_DIR / "noisetype_tuned.yaml", text)
    atomic_write_json(run_dir / "results.json", {"metrics": tab.to_dict("records")})
    append_ledger(STAGE, run_dir, f"E12b points_thr {points_thr:.5f} (bal acc {acc_p:.3f}), heavy_thr {heavy_thr:.3f} "
                  f"(bal acc {acc_h:.3f}); filters: dots {sp_f}, light {light_f}, heavy {heavy_f}")
    log("Done.")


if __name__ == "__main__":
    main()
