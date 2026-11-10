"""
Single entry point for every stage.

    python run.py help      list stages
    python run.py <stage>   run a stage (extra arguments are passed through)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = {
    "check": ("src.common.check", [], "paths, imports, GPU, disk, pip freeze"),
    "manifest": ("src.data.build_manifest", [], "E00 dataset manifest (labels, duplicates, splits)"),
    "dupcheck": ("src.data.check_duplicates", [], "E00 grid of removed near-duplicate pairs"),
    "lookalike-label": ("src.data.lookalike_app", [], "E01 browser app: label look-alike types (saved per click)"),
    "lookalike-freeze": ("src.data.lookalike", ["freeze"], "E01 save labelled types + freeze the look-alike set"),
    "e02": ("src.dip.e02_preprocess", [], "E02 preprocessing + PSNR/SSIM (add --smoke for a 20-image test)"),
    "e03": ("src.dip.e03_segmentation", [], "E03 segmentation masks + box overlap (--smoke = 20 images)"),
    "e04": ("src.dip.e04_features", [], "E04 16 DIP features for all images (--smoke = 40 images)"),
    "e05": ("src.classic.e05_classic", [], "E05 model A: SVM + Random Forest (needs a finished E04 run)"),
}


def show_help() -> None:
    print(__doc__)
    for name, (_, _, desc) in STAGES.items():
        print(f"  {name:20s} {desc}")
    print(f"  {'status':20s} progress of every stage")


def status() -> None:
    sys.path.insert(0, str(ROOT))
    from src.config import RESULTS_ROOT, free_gb
    print(f"Results root: {RESULTS_ROOT}   (free disk {free_gb():.1f} GB)")
    for stage in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()) if RESULTS_ROOT.exists() else []:
        runs = sorted(stage.glob("run_*"))
        if not runs:
            continue
        last = runs[-1]
        done = any((last / f).exists() for f in ("summary.json", "dup_pairs_grid.png", "results.json"))
        print(f"  {stage.name:22s} runs={len(runs):3d}  newest={last.name}  "
              f"{'finished' if done else 'NOT finished / partial'}  "
              f"({time.strftime('%d %b %H:%M', time.localtime(last.stat().st_mtime))})")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
        return show_help()
    name, extra = sys.argv[1], sys.argv[2:]
    if name == "status":
        return status()
    if name not in STAGES:
        print(f"Unknown stage '{name}'.")
        return show_help()
    module, fixed, _ = STAGES[name]
    cmd = [sys.executable, "-u", "-m", module, *fixed, *extra]
    print(f"> {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
