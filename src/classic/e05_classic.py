"""
E05 (model A): SVM and Random Forest on the 16 DIP features.

    python run.py e05 [--smoke] [--features PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.cfg import load_config, validate
from src.common.runs import append_ledger, atomic_write_json, new_or_resume_run, setup_logging, write_meta
from src.config import CKPT_ROOT, CONFIG_DIR, RESULTS_ROOT
from src.dip.features import FEATURE_NAMES
from src.eval.metrics import CLASSES, classification_metrics, summary_row

STAGE = "e05_classic"
EVAL_SPLITS = ["val", "test_in", "test_ext", "test_lookalike"]


def newest_features(root: Path) -> Path:
    runs = sorted((root / "e04_features").glob("run_*/features.csv"))
    runs = [p for p in runs if (p.parent / "results.json").exists()]
    if not runs:
        sys.exit(f"No finished E04 run under {root / 'e04_features'}. Run:  python run.py e04")
    return runs[-1]


def build_models(cfg: dict):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    c = cfg["e05"]
    svm = lambda C: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma="scale",
                                                         class_weight="balanced", random_state=cfg["seed"]))
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=c["rf_trees"], min_samples_leaf=c["rf_min_leaf"], class_weight="balanced",
        n_jobs=c["workers"], random_state=cfg["seed"]))
    return svm, rf


def records_for(model, df: pd.DataFrame, name: str) -> pd.DataFrame:
    X = df[FEATURE_NAMES].to_numpy()
    t0 = time.perf_counter()
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
    else:
        d = model.decision_function(X)
        if d.ndim == 1:
            d = np.stack([-d, d], axis=1)
        prob = np.exp(d - d.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
    ms = (time.perf_counter() - t0) * 1000 / max(len(df), 1)
    cls = list(model.classes_)
    rec = df[["path", "source", "split", "label", "lookalike_type"]].copy()
    rec["model"] = name
    for c in CLASSES:
        rec[f"p_{c}"] = prob[:, cls.index(c)] if c in cls else 0.0
    rec["pred"] = pred
    rec["classify_ms"] = round(ms, 4)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "e05_classic.yaml"))
    ap.add_argument("--features", default=None, help="path to a features.csv (default: newest finished E04 run)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    validate(cfg, ["seed", "e05.svm_C_grid", "e05.rf_trees", "e05.rf_min_leaf", "e05.workers", "e05.n_boot"], a.config)
    root = RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT
    fpath = Path(a.features) if a.features else newest_features(root)
    cfg = {**cfg, "features_file": str(fpath)}
    run_dir, resumed = new_or_resume_run(STAGE, cfg, root=root)
    log = setup_logging(run_dir, STAGE)
    log(f"{'RESUMING' if resumed else 'New'} run: {run_dir}\nFeatures: {fpath}")
    write_meta(run_dir, cfg, data_files=[fpath], extra={"smoke": a.smoke})
    import joblib
    from sklearn.metrics import f1_score
    from src.eval.plots import confusion_figure, lookalike_figure
    from src.viz import style
    import matplotlib.pyplot as plt

    feat = pd.read_csv(fpath, keep_default_na=False)
    feat = feat[feat["error"] == ""]
    feat[FEATURE_NAMES] = feat[FEATURE_NAMES].astype(float)
    tr, va = feat[feat["split"] == "train"], feat[feat["split"] == "val"]
    log(f"train {len(tr)}  val {len(va)}  (classes in train: {tr['label'].value_counts().to_dict()})")
    Xtr, ytr = tr[FEATURE_NAMES].to_numpy(), tr["label"].to_numpy()
    svm_fn, rf = build_models(cfg)
    n_boot = 100 if a.smoke else cfg["e05"]["n_boot"]

    grid = []
    for C in cfg["e05"]["svm_C_grid"]:
        m = svm_fn(C).fit(Xtr, ytr)
        f1 = f1_score(va["label"], m.predict(va[FEATURE_NAMES].to_numpy()), labels=CLASSES, average="macro",
                      zero_division=0)
        grid.append({"C": C, "val_macro_f1": round(float(f1), 4)})
        log(f"SVM C={C}: val macro-F1 {f1:.4f}")
    pd.DataFrame(grid).to_csv(run_dir / "svm_C_search.csv", index=False)
    best_C = max(grid, key=lambda g: g["val_macro_f1"])["C"]
    t0 = time.time()
    svm = svm_fn(best_C).fit(Xtr, ytr)
    fit_time = {"A_svm": round(time.time() - t0, 2)}
    t0 = time.time()
    rf.fit(Xtr, ytr)
    fit_time["A_rf"] = round(time.time() - t0, 2)
    log(f"Chosen C={best_C}. Fit time (s): {fit_time}")
    ck = CKPT_ROOT / STAGE / run_dir.name
    ck.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": svm, "features": FEATURE_NAMES, "classes": CLASSES}, ck / "A_svm.joblib")
    joblib.dump({"model": rf, "features": FEATURE_NAMES, "classes": CLASSES}, ck / "A_rf.joblib")

    rec = pd.concat([records_for(m, feat[feat["split"].isin(EVAL_SPLITS)], n) for n, m in
                     (("A_svm", svm), ("A_rf", rf))], ignore_index=True)
    rec.to_csv(run_dir / "records.csv", index=False)
    metrics, rows, look_rows = {}, [], []
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    for (name, split), g in rec.groupby(["model", "split"], sort=False):
        m = classification_metrics(g, n_boot=n_boot, seed=cfg["seed"])
        metrics.setdefault(name, {})[split] = m
        rows.append(summary_row(name, split, m))
        if "confusion_matrix" in m:
            confusion_figure(m["confusion_matrix"]["rows_true_cols_pred"], CLASSES, f"{name} — {split}",
                             run_dir / "figures" / f"cm_{name}_{split}")
        for t, v in m.get("false_alarm_by_type", {}).items():
            look_rows.append({"model": name, "lookalike_type": t, **v})
    summ = pd.DataFrame(rows)
    summ.to_csv(run_dir / "tables" / "summary.csv", index=False)
    (run_dir / "tables" / "summary.md").write_text(summ.to_markdown(index=False), encoding="utf-8")
    log("Summary:\n" + summ.to_string(index=False))
    if look_rows:
        lk = pd.DataFrame(look_rows)
        lk.to_csv(run_dir / "tables" / "lookalike_false_alarms.csv", index=False)
        lookalike_figure(lk, run_dir / "figures" / "fig_lookalike_false_alarms")
        log("Look-alike false alarms by type:\n" + lk.to_string(index=False))

    imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance": rf[-1].feature_importances_}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(run_dir / "tables" / "rf_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color=style.OKABE_ITO[0])
    ax.set_xlabel("Random Forest importance (mean impurity decrease)"), ax.tick_params(labelsize=7)
    style.save(fig, run_dir / "figures" / "fig_rf_importance", imp)
    plt.close(fig)

    feat_ms = float(feat["time_ms"].mean())
    speed = {n: {"feature_ms_per_image": round(feat_ms, 1),
                 "classify_ms_per_image": float(rec.loc[rec["model"] == n, "classify_ms"].iloc[0]),
                 "fps_cpu": round(1000 / (feat_ms + float(rec.loc[rec["model"] == n, "classify_ms"].iloc[0])), 1)}
             for n in ("A_svm", "A_rf")}
    log(f"Speed (CPU, one process): {speed}")
    summary = {"best_svm_C": best_C, "fit_time_s": fit_time, "speed": speed,
               "table": summ.to_dict("records"), "checkpoints": str(ck)}
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "results.json", {"metrics": metrics, "summary": summary, "records_file": "records.csv",
                                                 "meta": json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))})
    if not a.smoke:
        best = summ[summ["split"] == "test_in"].sort_values("macro_f1", ascending=False).head(1)
        note = f"E05 model A (SVM C={best_C}, RF)"
        if len(best):
            note += f"; best test_in macro-F1 {best['macro_f1'].iloc[0]} ({best['model'].iloc[0]})"
        append_ledger(STAGE, run_dir, note)
    log("Done.")


if __name__ == "__main__":
    main()
