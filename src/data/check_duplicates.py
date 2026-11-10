"""
E00: image grid of removed near-duplicate pairs.

    python run.py dupcheck
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.common.cfg import load_config
from src.common.runs import append_ledger, new_run_dir, write_meta
from src.config import CONFIG_DIR, PROJECT_ROOT, RESULTS_ROOT

ROOT = PROJECT_ROOT
CFG = load_config(CONFIG_DIR / "base.yaml")
OUT = RESULTS_ROOT
N_PAIRS = 24

runs = sorted((OUT / "s02_manifest").glob("run_*"))
if not runs:
    sys.exit("No Step 02 run found. Run src\\s02_build_manifest.py first.")
src_run = runs[-1]
run_dir = new_run_dir("s02b_dup_check")
write_meta(run_dir, CFG, extra={"source_run": src_run.name})

dups = pd.read_csv(src_run / "duplicates.csv")
full = pd.read_csv(src_run / "manifest_full.csv", usecols=["path", "phash"], dtype=str).set_index("path")["phash"]
near = dups[dups["exclude_reason"] == "near_duplicate"].copy()
if near.empty:
    sys.exit("No near duplicates in the newest run.")

near["kind"] = near["source"] + ": " + near["split"] + " -> " + near["kept_split"]
per_kind = max(1, N_PAIRS // near["kind"].nunique())
shown = (near.sample(frac=1, random_state=CFG["seed"]).groupby("kind").head(per_kind)
             .sort_values("kind").head(N_PAIRS).reset_index(drop=True))
h1 = shown["path"].map(full).map(lambda x: int(x, 16))
h2 = shown["duplicate_of"].map(full).map(lambda x: int(x, 16))
shown["phash_distance"] = np.bitwise_count(np.array(h1, dtype=np.uint64) ^ np.array(h2, dtype=np.uint64))
shown.to_csv(run_dir / "dup_pairs_shown.csv", index=False)

cols = 4
rows = int(np.ceil(len(shown) / cols))
fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 2.4, rows * 2.7), dpi=120)
axes = np.atleast_2d(axes)
for ax in axes.ravel():
    ax.axis("off")
for i, r in shown.iterrows():
    a, b = axes[i // cols, (i % cols) * 2], axes[i // cols, (i % cols) * 2 + 1]
    a.imshow(Image.open(ROOT / r["path"]).convert("RGB").resize((200, 200)))
    b.imshow(Image.open(ROOT / r["duplicate_of"]).convert("RGB").resize((200, 200)))
    a.set_title(f"REMOVED ({r['split']})\n{r['source']} · d={r['phash_distance']}", fontsize=7, color="#b91c1c")
    b.set_title(f"KEPT ({r['kept_split']})\n{r['label']}→{r['kept_label']}", fontsize=7, color="#15803d")
fig.suptitle(f"Near-duplicate check: removed image (left) vs kept copy (right) · pHash threshold "
             f"{CFG['manifest']['phash_threshold']} · source run {src_run.name}", fontsize=11)
fig.tight_layout()
fig.savefig(run_dir / "dup_pairs_grid.png", facecolor="white")
print(f"Saved {run_dir / 'dup_pairs_grid.png'}  ({len(shown)} pairs, {near['kind'].nunique()} kinds)")
print(f"Label conflicts among all removed near-duplicates: {int(near['label_conflict'].sum())} of {len(near)}")
append_ledger("s02b_dup_check", run_dir, f"duplicate grid from {src_run.name}: {len(shown)} pairs")
