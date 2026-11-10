"""Project paths (environment variable > .env > defaults)."""
import os
import shutil
from pathlib import Path

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT_DEFAULT / ".env", override=False)
    DOTENV_LOADED = (PROJECT_ROOT_DEFAULT / ".env").exists()
except ImportError:
    DOTENV_LOADED = False


def _p(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).resolve()


PROJECT_ROOT = _p("PROJECT_ROOT", PROJECT_ROOT_DEFAULT)
DATA_ROOT = _p("DATA_ROOT", PROJECT_ROOT / "data")
RESULTS_ROOT = _p("RESULTS_ROOT", PROJECT_ROOT / "outputs")
CKPT_ROOT = _p("CKPT_ROOT", PROJECT_ROOT / "checkpoints")
TB_ROOT = _p("TB_ROOT", PROJECT_ROOT / "runs")
CONFIG_DIR = PROJECT_ROOT / "configs"
MANIFEST_DIR = DATA_ROOT / "manifest"
LEDGER = RESULTS_ROOT / "LEDGER.md"
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "5"))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PATHS = {"PROJECT_ROOT": PROJECT_ROOT, "DATA_ROOT": DATA_ROOT, "RESULTS_ROOT": RESULTS_ROOT,
         "CKPT_ROOT": CKPT_ROOT, "TB_ROOT": TB_ROOT,
         "CONFIG_DIR": CONFIG_DIR, "MANIFEST_DIR": MANIFEST_DIR}


def free_gb(path: Path = PROJECT_ROOT) -> float:
    return shutil.disk_usage(path.anchor or path).free / 1e9


def check_free_space(path: Path = PROJECT_ROOT, warn=print) -> float:
    gb = free_gb(path)
    if gb < MIN_FREE_GB:
        warn(f"!! LOW DISK SPACE: {gb:.1f} GB free on {path.anchor} (warning below {MIN_FREE_GB} GB)")
    return gb


if __name__ == "__main__":
    print(f".env loaded: {DOTENV_LOADED}")
    for k, v in PATHS.items():
        print(f"{k:13s} {v}   {'ok' if v.exists() else 'MISSING'}")
    print(f"HF_HOME       {os.environ['HF_HOME']}")
    print(f"TORCH_HOME    {os.environ['TORCH_HOME']}")
    print(f"Free space    {free_gb():.1f} GB")
