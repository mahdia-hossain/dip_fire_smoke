"""
E01: freeze the labelled look-alike set.

    python run.py lookalike-freeze [--force]
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.cfg import load_config
from src.common.runs import append_ledger, atomic_write_json, new_run_dir, setup_logging, write_meta
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT
from src.data.lookalike_app import LABELS_CSV, OTHER, TARGET_PER_TYPE, TYPES, load_items
from src.viz.grids import group_grid

TYPE_CODES = [v[0] for v in TYPES.values()]
LEFT_OUT = [v[0] for v in OTHER.values()]
FROZEN_FLAG = MANIFEST_DIR / "lookalike_types.FROZEN"


def freeze(force: bool) -> None:
    if FROZEN_FLAG.exists() and not force:
        raise SystemExit(f"Look-alike set is already frozen ({FROZEN_FLAG.name}). Use --force to re-freeze "
                         f"(only before any model has been evaluated on it).")
    if not LABELS_CSV.exists():
        raise SystemExit("No answers yet. Run `python run.py lookalike-label` first.")
    cfg = load_config(CONFIG_DIR / "base.yaml")
    run_dir = new_run_dir("e01_lookalike")
    log = setup_logging(run_dir, "lookalike")
    write_meta(run_dir, cfg, data_files=[LABELS_CSV, MANIFEST_DIR / "manifest.csv"])
    items = load_items()
    ans = pd.read_csv(LABELS_CSV).drop_duplicates("path", keep="last")
    ans = ans[ans["path"].isin(items["path"])]
    shutil.copy2(LABELS_CSV, run_dir / "labels_raw.csv")
    counts = ans["label"].value_counts().reindex(TYPE_CODES + LEFT_OUT, fill_value=0)
    counts["unlabelled"] = len(items) - len(ans)
    counts.rename("n_images").to_csv(run_dir / "counts.csv")
    log("Images per type:\n" + counts.to_string())
    typed = ans[ans["label"].isin(TYPE_CODES)].rename(columns={"label": "lookalike_type"})
    typed = typed.sort_values(["lookalike_type", "path"])[["path", "lookalike_type"]]
    typed.to_csv(run_dir / "lookalike_types.csv", index=False)
    ans[ans["label"].isin(LEFT_OUT)].to_csv(run_dir / "left_out.csv", index=False)
    low = counts[TYPE_CODES][counts[TYPE_CODES] < TARGET_PER_TYPE]
    if len(low):
        log(f"!! Types below the target of {TARGET_PER_TYPE} (report them with this caveat): {low.to_dict()}")
    group_grid({t: typed.loc[typed["lookalike_type"] == t, "path"].tolist() for t in TYPE_CODES},
               run_dir / "lookalike_types_grid.png", PROJECT_ROOT,
               title="Look-alike test set by type (seeded random samples)", seed=cfg["seed"])
    summary = {"typed_images": int(len(typed)), "per_type": {k: int(v) for k, v in counts[TYPE_CODES].items()},
               "rejected_real_fire_smoke": int(counts["reject_has_fire_smoke"]),
               "skipped_unsure": int(counts["skip_unsure"]), "unlabelled": int(counts["unlabelled"])}
    atomic_write_json(run_dir / "summary.json", summary)
    shutil.copy2(run_dir / "lookalike_types.csv", MANIFEST_DIR / "lookalike_types.csv")
    FROZEN_FLAG.write_text(f"frozen from {run_dir.name}\n", encoding="utf-8")
    append_ledger("e01_lookalike", run_dir, f"E01 look-alike set frozen: {len(typed)} typed images, "
                  f"{summary['rejected_real_fire_smoke']} rejected (real fire/smoke visible)")
    log(f"Frozen: {len(typed)} images -> {MANIFEST_DIR / 'lookalike_types.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["freeze"])
    ap.add_argument("--force", action="store_true")
    freeze(ap.parse_args().force)
