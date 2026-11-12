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
    "seg-tune": ("src.dip.e03b_seg_tune", [], "E03b choose segmentation thresholds on val (writes configs/segmentation_tuned.yaml)"),
    "noise-tune": ("src.dip.e03c_noise_tune", [], "E03c choose the noise threshold for adaptive preprocessing on val"),
    "dark-tune": ("src.dip.e03d_dark_tune", [], "E03d choose the brightness threshold for adaptive CLAHE on val"),
    "restore-study": ("src.dip.e12_restoration", [], "E12 denoiser + noise-detector study (CPU)"),
    "noisetype-tune": ("src.dip.e12b_noisetype_tune", [], "E12b thresholds for noise-type adaptive preprocessing (CA3)"),
    "e03": ("src.dip.e03_segmentation", [], "E03 segmentation masks + box overlap (--smoke = 20 images)"),
    "e04": ("src.dip.e04_features", [], "E04 16 DIP features for all images (--smoke = 40 images)"),
    "train": ("src.deep.train", [], "E06-E09 train a deep model: --model B|C|D|E|EM|CM|CR|EMD|CA|CRN|CN|CA2|CA3 [--seed N] (add --smoke for a quick test)"),
    "probe": ("src.deep.probe", [], "E10a/b probes on trained models: --mode robust | --mode ablate"),
    "export": ("src.app.export", [], "export the final model as one file for the firesmoke package (release/)"),
    "demo": ("src.app.demo_app", [], "local demo website: upload a photo, see each DIP step and the answer"),
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
