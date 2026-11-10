"""YAML configs with `_base_` inheritance and key validation."""
import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = cfg.pop("_base_", None) or []
    merged = {}
    for b in [bases] if isinstance(bases, str) else bases:
        bp = path.parent / b
        if not bp.exists() and b == "segmentation_tuned.yaml":
            raise SystemExit("configs/segmentation_tuned.yaml not found. Run:  python run.py seg-tune")
        merged = deep_merge(merged, load_config(bp))
    return deep_merge(merged, cfg)


def get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def validate(cfg: dict, required: list, where: str = "config") -> None:
    missing = []
    for key in required:
        try:
            get(cfg, key)
        except KeyError:
            missing.append(key)
    if missing:
        raise SystemExit(f"{where}: missing required keys: {', '.join(missing)}")
