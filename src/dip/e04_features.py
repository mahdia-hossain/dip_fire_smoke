"""
E04: 16 DIP features for every image (features.csv, ANOVA ranking, distributions).

    python run.py e04 [--smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.cfg import load_config, validate
from src.common.parallel import run_records
from src.common.runs import append_ledger, atomic_write_json, new_or_resume_run, setup_logging, write_meta
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.dip.features import FEATURE_GROUPS, FEATURE_NAMES, extract
from src.dip.preprocess import load_rgb, preprocess
from src.dip.segmentation import segment

STAGE = "e04_features"
CLASSES = ["fire", "smoke", "normal"]


def one(item: dict, cfg: dict) -> dict:
    t0 = time.perf_counter()
    rec = {k: item[k] for k in ("id", "path", "source", "split", "label", "lookalike_type")}
    try:
        p2 = preprocess(load_rgb(PROJECT_ROOT / item["path"]), cfg["preprocess"])
        rec.update(extract(p2, segment(p2, cfg["segmentation"]), cfg["features"]))
        rec["error"] = ""
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
    rec["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return rec


def build_items() -> pd.DataFrame:
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    look = pd.read_csv(MANIFEST_DIR / "lookalike_types.csv")[["path", "lookalike_type"]]
    df = man.merge(look, on="path", how="left")
    df["lookalike_type"] = df["lookalike_type"].fillna("")
    df = df[(df["split"] != "test_lookalike") | (df["lookalike_type"] != "")]
    df = df[["path", "source", "split", "label", "lookalike_type"]].reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    return df


def report(feat: pd.DataFrame, cfg: dict, run_dir: Path, log) -> dict:
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import f_classif
    from src.viz import style
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    tr = feat[feat["split"] == "train"]
    means = tr.groupby("label")[FEATURE_NAMES].agg(["mean", "std"]).T.unstack()
    means.to_csv(run_dir / "tables" / "feature_means_by_class.csv")
    with np.errstate(divide="ignore", invalid="ignore"):
        F, p = f_classif(tr[FEATURE_NAMES].to_numpy(), tr["label"].to_numpy())
    F = np.where(np.isfinite(F), F, np.nan)
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    rank = pd.DataFrame({"feature": FEATURE_NAMES, "group": [group_of[f] for f in FEATURE_NAMES], "anova_F": F,
                         "p_value": p}).sort_values("anova_F", ascending=False).reset_index(drop=True)
    rank.to_csv(run_dir / "tables" / "feature_ranking.csv", index=False)
    (run_dir / "tables" / "feature_ranking.md").write_text(rank.round(4).to_markdown(index=False), encoding="utf-8")
    log("Feature ranking (ANOVA F on train, higher = separates the classes better):\n"
        + rank.round(3).to_string(index=False))
    sub = tr.sample(min(len(tr), 6000), random_state=cfg["seed"])
    fig, axes = plt.subplots(4, 4, figsize=(7.0, 6.2))
    for ax, f in zip(axes.ravel(), FEATURE_NAMES):
        data = [sub.loc[sub["label"] == c, f].to_numpy() for c in CLASSES]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                        medianprops={"color": "black", "lw": 1})
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(style.CLASS_COLOR[c])
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2, 3], CLASSES, fontsize=6), ax.set_title(f"{f} ({group_of[f]})", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    style.save(fig, run_dir / "figures" / "fig_feature_distributions",
               sub.groupby("label")[FEATURE_NAMES].describe().T.reset_index())
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    colors = [style.OKABE_ITO[list(FEATURE_GROUPS).index(g)] for g in rank["group"]]
    ax.barh(rank["feature"][::-1], rank["anova_F"][::-1].fillna(0), color=colors[::-1])
    ax.set_xlabel("ANOVA F-score (train)"), ax.tick_params(labelsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in zip(FEATURE_GROUPS, style.OKABE_ITO)],
              fontsize=6, frameon=False, loc="lower right")
    style.save(fig, run_dir / "figures" / "fig_feature_ranking", rank)
    plt.close(fig)
    return {"top5_features": rank["feature"].head(5).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "e04_features.yaml"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    validate(cfg, ["seed", "preprocess", "segmentation", "features.lbp", "features.glcm", "features.edge_thr",
                   "e04.workers"], a.config)
    run_dir, resumed = new_or_resume_run(STAGE, cfg, root=RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT)
    log = setup_logging(run_dir, STAGE)
    log(f"{'RESUMING' if resumed else 'New'} run: {run_dir}")
    if not (run_dir / "meta.json").exists():
        write_meta(run_dir, cfg, data_files=[MANIFEST_DIR / "manifest.csv", MANIFEST_DIR / "lookalike_types.csv"],
                   extra={"smoke": a.smoke, "feature_names": FEATURE_NAMES})
    ip = run_dir / "items.csv"
    if ip.exists():
        items = pd.read_csv(ip, keep_default_na=False)
    else:
        items = build_items()
        if a.smoke:
            items = (items.sample(frac=1, random_state=cfg["seed"]).groupby(["split", "label"]).head(3)
                     .reset_index(drop=True))
        items.to_csv(ip, index=False)
    log("Images per split/label:\n" + items.groupby(["split", "label"]).size().to_string())
    feat = run_records(run_dir, items.to_dict("records"), one, cfg, cfg["e04"]["workers"], log)
    feat["lookalike_type"] = feat["lookalike_type"].fillna("")
    failed = feat[feat["error"] != ""]
    if len(failed):
        log(f"!! {len(failed)} images failed (recorded, excluded from training): {failed.iloc[0]['error']}")
    feat.to_csv(run_dir / "features.csv", index=False)
    ok = feat[feat["error"] == ""]
    extra = report(ok, cfg, run_dir, log) if (ok["split"] == "train").any() else {}
    summary = {"images": int(len(feat)), "failed": int(len(failed)), "features": FEATURE_NAMES,
               "mean_ms_per_image": round(float(ok["time_ms"].mean()), 1), **extra}
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "results.json", {"metrics": summary, "records_file": "features.csv",
                                                 "meta": json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))})
    (run_dir / "records.partial.jsonl").unlink(missing_ok=True)
    if not a.smoke:
        append_ledger(STAGE, run_dir, f"E04 16 DIP features for {len(ok)} images")
    log(f"Done. {json.dumps(summary)}")


if __name__ == "__main__":
    main()
