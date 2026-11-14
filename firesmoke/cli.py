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
        raise SystemExit("No model file. Pass --weights release\\firesmoke_CA.pt")
    if not Path(weights).is_file():
        raise SystemExit(f"Model file not found: {Path(weights).resolve()}")

    src = Path(a.source)
    if not src.exists():
        raise SystemExit(f"Not found: {src}\nCheck the folder or photo name, for example with:  dir {src.parent}")
    if src.is_dir():
        paths = sorted(q for q in src.iterdir() if q.is_file() and q.suffix.lower() in EXT)
        if not paths:
            subs = [q.name for q in src.iterdir() if q.is_dir()]
            hint = f"\nSubfolders here: {', '.join(subs)}  (run the command on one of them)" if subs else ""
            raise SystemExit(f"No photos (.jpg .jpeg .png .bmp .webp) found in {src}{hint}")
    elif src.suffix.lower() in EXT:
        paths = [src]
    else:
        raise SystemExit(f"Not a photo: {src.name} (use .jpg .jpeg .png .bmp or .webp)")

    from .detector import FireSmokeDetector
    det = FireSmokeDetector(weights)
    print(f"{len(paths)} photo(s) found in {src}")

    done, skipped = 0, []
    for q in paths:
        try:
            r = det.predict(q)
        except Exception as e:
            skipped.append(q.name)
            print(f"{q.name}: SKIPPED ({e})")
            continue
        r.save(a.out)
        done += 1
        c = r.change
        print(f"{q.name}: {r.label} {r.confidence * 100:.1f}%  ({r.decision}, {r.filter})  "
              f"PSNR {c['psnr_db']:.2f} dB  SSIM {c['ssim']:.3f}  changed {c['changed_pct']:.2f}%")
        if a.steps:
            r.explain()
            print()

    print(f"\n{done} photo(s) done, {len(skipped)} skipped. Results saved to {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
