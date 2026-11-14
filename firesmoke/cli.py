"""Command line: firesmoke predict PATH [--weights FILE] [--out DIR] [--steps]"""
import argparse
import glob
import os
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main() -> None:
    ap = argparse.ArgumentParser(prog="firesmoke")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="classify a photo or every photo in a folder")
    p.add_argument("source")
    p.add_argument("--weights", default=os.environ.get("FIRESMOKE_WEIGHTS"))
    p.add_argument("--out", default="firesmoke_out")
    p.add_argument("--steps", action="store_true", help="print the step-by-step explanation")
    a = ap.parse_args()
    weights = a.weights or next(iter(sorted(glob.glob("firesmoke_*.pt"))), None)
    if not weights:
        raise SystemExit("No model file. Pass --weights firesmoke_CA.pt (download it from the GitHub release).")
    from .detector import FireSmokeDetector
    det = FireSmokeDetector(weights)
    src = Path(a.source)
    paths = sorted(q for q in src.iterdir() if q.suffix.lower() in EXT) if src.is_dir() else [src]
    for q in paths:
        r = det.predict(q)
        r.save(a.out)
        c = r.change
        print(f"{q.name}: {r.label} {r.confidence * 100:.1f}%  ({r.decision}, {r.filter})  "
              f"PSNR {c['psnr_db']:.2f} dB  SSIM {c['ssim']:.3f}  changed {c['changed_pct']:.2f}%")
        if a.steps:
            r.explain()
            print()
    print(f"Results saved to {Path(a.out).resolve()}")

if __name__ == "__main__":
    main()
