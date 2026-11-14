"""
Re-evaluate a trained model from its saved checkpoint, when its result folder was lost.

    python src/deep/reeval.py --model CA2 --run run_20261111_213222

It loads checkpoints/<stage>/<run>/best.pt, evaluates it on val, test_in, test_ext and test_lookalike,
and rebuilds outputs/<stage>/<run>/ (marked restored_from_checkpoint). Nothing is trained.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.common.runs import append_ledger, atomic_write_json, setup_logging, write_meta
from src.config import CKPT_ROOT, RESULTS_ROOT
from src.deep.data import CLASSES, FireSmokeDataset, worker_init
from src.deep.models import build_model, uses_features
from src.deep.probe import STAGES
from src.deep.train import EVAL_SPLITS, predict, records
from src.dip.features import FEATURE_NAMES
from src.eval.metrics import classification_metrics, summary_row
from src.eval.plots import confusion_figure, lookalike_figure


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(STAGES))
    ap.add_argument("--run", required=True)
    a = ap.parse_args()

    stage = STAGES[a.model]
    ck_path = CKPT_ROOT / stage / a.run / "best.pt"
    if not ck_path.exists():
        sys.exit(f"Model file not found: {ck_path}")
    run_dir = RESULTS_ROOT / stage / a.run
    if (run_dir / "results.json").exists():
        sys.exit(f"{run_dir} already has results.json. Nothing to restore.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    log = setup_logging(run_dir, stage + "_reeval")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    cfg, norm = ck["cfg"], ck["feat_norm"]
    name, kind, arch = cfg["model"]["name"], cfg["model"]["kind"], cfg["model"]["arch"]
    model = build_model(kind, arch, pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()
    use_f = uses_features(kind)
    log(f"Re-evaluating {name} from {ck_path} on {dev} (best epoch {ck['best']['epoch']}, "
        f"val macro-F1 at training {ck['best']['val_macro_f1']})")

    feat = pd.read_csv(cfg["features_file"], keep_default_na=False)
    feat = feat[feat["error"] == ""].reset_index(drop=True)
    feat[FEATURE_NAMES] = feat[FEATURE_NAMES].astype(np.float32)
    workers = int(os.environ.get("DIP_WORKERS", "2"))
    bs = cfg["train"]["batch_size"]

    rec, metrics, rows, look_rows, speed = [], {}, [], [], {}
    for s in EVAL_SPLITS:
        frame = feat[feat["split"] == s].reset_index(drop=True)
        ds = FireSmokeDataset(frame, cfg["preprocess"], False, np.array(norm["mean"]), np.array(norm["std"]))
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, worker_init_fn=worker_init,
                        pin_memory=torch.cuda.is_available())
        with torch.no_grad():
            idx, prob, ms = predict(model, dl, dev, use_f, cfg["train"]["amp"])
        r = records(frame, idx, prob, name)
        rec.append(r)
        m = classification_metrics(r, n_boot=cfg["deep"]["n_boot"], seed=cfg["seed"])
        metrics[s] = m
        rows.append(summary_row(name, s, m))
        speed[s] = {"ms_per_image": round(ms, 3), "fps": round(1000 / max(ms, 1e-6), 1), "batch": bs}
        if "confusion_matrix" in m:
            confusion_figure(m["confusion_matrix"]["rows_true_cols_pred"], CLASSES, f"{name} - {s}",
                             run_dir / "figures" / f"cm_{name}_{s}")
        look_rows += [{"model": name, "lookalike_type": k, **v} for k, v in m.get("false_alarm_by_type", {}).items()]
        log(f"{s}: {len(frame)} images done")

    rec = pd.concat(rec, ignore_index=True)
    rec.to_csv(run_dir / "records.csv", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(run_dir / "tables" / "summary.csv", index=False)
    (run_dir / "tables" / "summary.md").write_text(summ.to_markdown(index=False), encoding="utf-8")
    log("Summary:\n" + summ.to_string(index=False))
    if look_rows:
        lk = pd.DataFrame(look_rows)
        lk.to_csv(run_dir / "tables" / "lookalike_false_alarms.csv", index=False)
        lookalike_figure(lk, run_dir / "figures" / "fig_lookalike_false_alarms")
        log("Look-alike false alarms by type:\n" + lk.to_string(index=False))

    params = sum(p.numel() for p in model.parameters()) / 1e6
    summary = {"model": name, "seed": cfg["seed"], "arch": arch, "kind": kind, "params_m": round(params, 3),
               "best_epoch": ck["best"]["epoch"], "best_val_macro_f1": ck["best"]["val_macro_f1"],
               "speed": speed, "table": summ.to_dict("records"), "checkpoints": str(ck_path.parent),
               "restored_from_checkpoint": True}
    atomic_write_json(run_dir / "summary.json", summary)
    write_meta(run_dir, cfg, extra={"restored_from_checkpoint": True, "checkpoint": str(ck_path), "feature_norm": norm})
    atomic_write_json(run_dir / "results.json", {"metrics": metrics, "summary": summary, "records_file": "records.csv",
                                                 "meta": json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))})
    t = summ.set_index("split")["macro_f1"]
    note = (f"{name} seed {cfg['seed']} RESTORED from checkpoint (no training): val {t.get('val')}, "
            f"test_in {t.get('test_in')}, test_ext {t.get('test_ext')}")
    append_ledger(stage, run_dir, note)
    log("Done.")


if __name__ == "__main__":
    main()
