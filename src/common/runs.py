"""Run folders, logging, metadata, atomic writes, seeding, ledger, resume."""
import hashlib
import json
import logging
import os
import platform
import random
import socket
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from src.config import LEDGER, PROJECT_ROOT, RESULTS_ROOT, check_free_space

LIBS = ["numpy", "pandas", "opencv-python", "scikit-image", "scikit-learn", "PyWavelets", "imagehash",
        "pillow", "matplotlib", "pyyaml", "torch", "torchvision", "timm", "tensorboard", "python-dotenv"]


def new_run_dir(stage: str, root: Path = RESULTS_ROOT) -> Path:
    base = root / stage
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = next(base / f"{stamp}{'' if k == 0 else f'_{k}'}" for k in range(1000)
                   if not (base / f"{stamp}{'' if k == 0 else f'_{k}'}").exists())
    run_dir.mkdir(parents=True)
    return run_dir


def setup_logging(run_dir: Path, name: str = "run"):
    logger = logging.getLogger(f"{name}:{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    for h in (logging.FileHandler(run_dir / "log.txt", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    check_free_space(warn=logger.warning)
    return logger.info


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def lib_versions() -> dict:
    out = {}
    for lib in LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def gpu_info() -> dict:
    """Only reads torch if it is ALREADY imported by the stage (keeps CPU-only stages light)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {"cuda": False if torch is not None else "torch not loaded"}
    free, total = torch.cuda.mem_get_info()
    return {"cuda": True, "name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "vram_total_gb": round(total / 1e9, 2), "vram_free_gb_at_start": round(free / 1e9, 2),
            "bf16": bool(torch.cuda.is_bf16_supported())}


def write_meta(run_dir: Path, cfg: dict, data_files: list = (), extra: dict = None) -> dict:
    """config.resolved.yaml + meta.json. Code is not in git, so every .py file in src/ is hashed instead."""
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                                  encoding="utf-8")
    code = {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)[:16]
            for p in sorted((PROJECT_ROOT / "src").rglob("*.py")) + [PROJECT_ROOT / "run.py"] if p.exists()}
    meta = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "libraries": lib_versions(),
        "gpu": gpu_info(),
        "code_sha256_16": code,
        "data_files": {str(Path(f).as_posix()): {"bytes": Path(f).stat().st_size, "sha256": sha256_file(Path(f))}
                       for f in data_files if Path(f).exists()},
        **(extra or {}),
    }
    atomic_write_json(run_dir / "meta.json", meta)
    return meta


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def append_ledger(stage: str, run_dir: Path, note: str) -> None:
    """One line per result folder: what produced it (guide §7, §14). Report numbers are traced through this."""
    new = not LEDGER.exists()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        if new:
            f.write("# Results ledger\n\n| Date | Stage | Run folder | Command | Note |\n| --- | --- | --- | --- | --- |\n")
        rel = run_dir.relative_to(RESULTS_ROOT).as_posix() if run_dir.is_relative_to(RESULTS_ROOT) else str(run_dir)
        f.write(f"| {time.strftime('%Y-%m-%d %H:%M')} | {stage} | `{rel}` | `{' '.join(sys.argv)}` | {note} |\n")


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()[:12]


def new_or_resume_run(stage: str, cfg: dict, final_name: str = "results.json", root: Path = RESULTS_ROOT):
    """Resume the newest UNFINISHED run of this stage that used the SAME config (guide §3);"""
    h = config_hash(cfg)
    for run in sorted((root / stage).glob("run_*"), reverse=True) if (root / stage).exists() else []:
        if (run / final_name).exists():
            break
        if (run / "config_hash.txt").exists() and (run / "config_hash.txt").read_text().strip() == h:
            return run, True
    run_dir = new_run_dir(stage, root)
    (run_dir / "config_hash.txt").write_text(h, encoding="utf-8")
    return run_dir, False


def read_partial(path: Path) -> list:
    """Records already finished in a .partial.jsonl; a truncated last line (killed mid-write) is ignored."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
