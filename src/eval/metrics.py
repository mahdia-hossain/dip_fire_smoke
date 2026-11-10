"""Classification metrics computed from per-image records."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

CLASSES = ["fire", "smoke", "normal"]


def bootstrap_ci(y_true, y_pred, fn, n_boot: int = 1000, seed: int = 42) -> list:
    """95% bootstrap interval of a metric (resample images with replacement)."""
    if n_boot <= 0:
        return None
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    vals = np.array([fn(y_true[i], y_pred[i]) for i in idx])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def macro_f1(t, p, labels=CLASSES) -> float:
    """Macro-F1 over `labels` (the classes present in the evaluated set; an absent class would add a fake 0)."""
    return f1_score(t, p, labels=labels, average="macro", zero_division=0)


def classification_metrics(rec: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    t, p = rec["label"].to_numpy(), rec["pred"].to_numpy()
    present = [c for c in CLASSES if (t == c).any()]
    out = {"n": int(len(rec)), "class_counts": {c: int((t == c).sum()) for c in CLASSES}}
    normal = t == "normal"
    out["false_alarm_rate"] = round(float((p[normal] != "normal").mean()), 4) if normal.any() else None
    fire = t == "fire"
    out["fire_missed_as_normal"] = round(float((p[fire] == "normal").mean()), 4) if fire.any() else None
    if len(present) == 1 and present[0] == "normal":
        out["predicted_as"] = {c: int((p == c).sum()) for c in CLASSES}
        if "lookalike_type" in rec:
            out["false_alarm_by_type"] = {
                k: {"n": int(len(g)), "false_alarm_rate": round(float((g["pred"] != "normal").mean()), 4),
                    "called_fire": int((g["pred"] == "fire").sum()), "called_smoke": int((g["pred"] == "smoke").sum())}
                for k, g in rec.groupby("lookalike_type")}
        out["false_alarm_ci95"] = bootstrap_ci(np.zeros(len(p)), (p != "normal").astype(float),
                                               lambda a, b: b.mean(), n_boot, seed)
        return out
    out["accuracy"] = round(float(accuracy_score(t, p)), 4)
    out["accuracy_ci95"] = bootstrap_ci(t, p, accuracy_score, n_boot, seed)
    mf1 = lambda a, b: macro_f1(a, b, [c for c in present if (a == c).any()])
    out["macro_f1"] = round(float(mf1(t, p)), 4)
    out["macro_f1_ci95"] = bootstrap_ci(t, p, mf1, n_boot, seed)
    out["macro_f1_classes"] = present
    pr, rc, f1, sup = precision_recall_fscore_support(t, p, labels=CLASSES, zero_division=0)
    out["per_class"] = {c: {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4),
                            "f1": round(float(f1[i]), 4), "support": int(sup[i])} for i, c in enumerate(CLASSES)}
    out["fire_recall"] = out["per_class"]["fire"]["recall"]
    out["confusion_matrix"] = {"labels": CLASSES, "rows_true_cols_pred": confusion_matrix(t, p, labels=CLASSES).tolist()}
    return out


def summary_row(model: str, split: str, m: dict) -> dict:
    r = {"model": model, "split": split, "n": m["n"]}
    if "macro_f1" in m:
        r.update({"accuracy": m["accuracy"], "accuracy_ci95": m["accuracy_ci95"], "macro_f1": m["macro_f1"],
                  "macro_f1_ci95": m["macro_f1_ci95"], **{f"{c}_recall": m["per_class"][c]["recall"] if m["per_class"][c]["support"] else None
                     for c in CLASSES}})
    r["false_alarm_rate"] = m["false_alarm_rate"]
    return r
