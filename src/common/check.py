"""Environment check: paths, imports, GPU, disk, pinned packages."""
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DOTENV_LOADED, PATHS, RESULTS_ROOT, free_gb

MODULES = ["numpy", "pandas", "cv2", "skimage", "sklearn", "pywt", "imagehash", "PIL", "matplotlib", "yaml",
           "tqdm", "dotenv", "torch", "torchvision", "timm", "tensorboard"]


def main() -> None:
    print(f".env loaded: {DOTENV_LOADED}")
    for k, v in PATHS.items():
        print(f"  {k:13s} {v}  {'ok' if v.exists() else 'MISSING'}")
    print(f"  Free disk: {free_gb():.1f} GB")
    bad = []
    for m in MODULES:
        try:
            mod = import_module(m)
            print(f"  import {m:12s} ok  {getattr(mod, '__version__', '')}")
        except Exception as e:
            bad.append(m)
            print(f"  import {m:12s} FAILED ({e.__class__.__name__}: {e})")
    if "torch" not in bad:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  GPU {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | VRAM free {free / 1e9:.2f} / "
                  f"{total / 1e9:.2f} GB | bf16 {torch.cuda.is_bf16_supported()}")
        else:
            print("  !! CUDA NOT available — training would be very slow. Check the torch install (cu130 build).")
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        print("  Processes on the GPU:\n    " + apps.replace("\n", "\n    "))
    except Exception as e:
        print(f"  nvidia-smi not available ({e})")
    env_dir = RESULTS_ROOT / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    out = env_dir / f"pip_freeze_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    out.write_text(frozen, encoding="utf-8")
    print(f"  Pinned package list saved: {out}")
    if bad:
        print(f"!! Missing modules: {bad}. Install with: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
