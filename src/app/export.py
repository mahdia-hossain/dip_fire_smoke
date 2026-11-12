"""
Export the final model (finished CA / CA3 run with the best validation score) as one self-contained file
for the firesmoke package.

    python run.py export          -> release/firesmoke_<ID>.pt + release/MODEL_CARD.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.app.infer import checkpoint_of, find_final
from src.common.cfg import load_config
from src.config import CONFIG_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.deep.data import CLASSES, MEAN, STD


def main() -> None:
    smoke = "--smoke" in sys.argv
    run = find_final(RESULTS_ROOT / "_smoke" if smoke else RESULTS_ROOT)
    s = json.loads((run / "summary.json").read_text())
    ck = torch.load(checkpoint_of(run), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    pre = cfg["preprocess"]
    bundle = {
        "format": "firesmoke-v1", "model_name": s["model"], "arch": cfg["model"]["arch"], "classes": CLASSES,
        "state_dict": ck["model"], "mean": MEAN, "std": STD,
        "preprocess": {"size": pre["size"], "contrast": pre["contrast"], "adaptive": pre["adaptive"]},
        "segmentation": load_config(CONFIG_DIR / "deep_train.yaml")["segmentation"],
        "val_macro_f1": s["best_val_macro_f1"], "source_run": f"{run.parent.name}/{run.name}",
        "test_metrics": {r["split"]: {k: r[k] for k in ("macro_f1", "false_alarm_rate") if k in r} for r in s["table"]},
    }
    out = PROJECT_ROOT / ("release_smoke" if smoke else "release")
    out.mkdir(exist_ok=True)
    short = s["model"].split("_")[0]
    f = out / f"firesmoke_{short}.pt"
    torch.save(bundle, f)
    card = [f"# Model card: firesmoke_{short}.pt", "",
            f"- Model: `{s['model']}` ({cfg['model']['arch']}), source run `{bundle['source_run']}`",
            f"- Validation macro-F1: {s['best_val_macro_f1']}",
            f"- Adaptive DIP thresholds: {json.dumps(pre['adaptive'])}",
            "- Metrics from the source run:", "",
            "| split | macro-F1 | false-alarm rate |", "|---|---|---|"]
    card += [f"| {k} | {v.get('macro_f1')} | {v.get('false_alarm_rate')} |" for k, v in bundle["test_metrics"].items()]
    (out / "MODEL_CARD.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    print(f"Exported {f} ({f.stat().st_size / 1e6:.1f} MB) from {bundle['source_run']}")


if __name__ == "__main__":
    main()
