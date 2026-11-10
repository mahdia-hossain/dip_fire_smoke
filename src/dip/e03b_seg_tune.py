"""
E03b: choose the segmentation thresholds on validation images only.

    python run.py seg-tune [--smoke]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from src.common.cfg import load_config, validate
from src.common.parallel import run_records
from src.common.runs import append_ledger, atomic_write_json, atomic_write_text, new_or_resume_run, setup_logging, write_meta
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.data.boxes import box_mask, label_path, read_boxes
from src.dip.preprocess import load_rgb, preprocess
from src.dip.segmentation import clean, fire_rule, overlay, segment, smoke_rule

STAGE = "e03b_seg_tune"
CLASSES = ["fire", "smoke", "normal"]


def grids(cfg: dict):
    g = cfg["seg_tune"]
    smoke = list(itertools.product(g["sat_max"], g["val_min"], g["dark_thr"]))
    return g["tau_cb_cr"], smoke


def skey(s) -> str:
    return f"s{s[0]}_v{s[1]}_d{s[2]}"


def one(item: dict, cfg: dict) -> dict:
    t0 = time.perf_counter()
    rec = {k: item[k] for k in ("id", "path", "source", "label")}
    try:
        p2 = preprocess(load_rgb(PROJECT_ROOT / item["path"]), cfg["preprocess"])
        sc = cfg["segmentation"]
        size = p2.shape[0]
        min_area = max(1, int(sc["morph"]["min_area_frac"] * size * size))
        k = sc["morph"]["kernel"]
        ids = sc["box_classes"].get(item["source"])
        boxes = read_boxes(PROJECT_ROOT / label_path(item["path"])) if ids else np.zeros((0, 5))
        fb = box_mask(boxes, ids["fire"], size).astype(bool) if ids else np.zeros((size, size), bool)
        sb = box_mask(boxes, ids["smoke"], size).astype(bool) if ids else np.zeros((size, size), bool)
        rec["has_fire_box"], rec["has_smoke_box"] = bool(fb.any()), bool(sb.any())
        taus, smokes = grids(cfg)
        raw_smoke = {s: smoke_rule(p2, s[0], s[1], sc["smoke"]["dark_patch"], s[2])[0] for s in smokes}
        for t in taus:
            f = clean(fire_rule(p2, t), k, min_area)["final"].astype(bool)
            rec[f"fire_t{t}_area"] = float(f.mean())
            rec[f"fire_t{t}_cov"] = float((f & fb).sum() / fb.sum()) if fb.any() else None
            for s in smokes:
                m = clean(raw_smoke[s] & (~f).astype(np.uint8), k, min_area)["final"].astype(bool)
                rec[f"smoke_t{t}_{skey(s)}_area"] = float(m.mean())
                rec[f"smoke_t{t}_{skey(s)}_cov"] = float((m & sb).sum() / sb.sum()) if sb.any() else None
        rec["error"] = ""
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
    rec["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return rec


def build_sample(cfg: dict) -> pd.DataFrame:
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    val = man[(man["split"] == "val") & man["source"].isin(list(cfg["segmentation"]["box_classes"]))]
    n = cfg["seg_tune"]["per_class"]
    s = pd.concat([val[val["label"] == c].sample(frac=1, random_state=cfg["seed"]).head(n) for c in CLASSES],
                  ignore_index=True)[["path", "source", "label"]]
    s.insert(0, "id", np.arange(len(s)))
    return s


def score(rec: pd.DataFrame, cfg: dict):
    taus, smokes = grids(cfg)
    normal = rec["label"] == "normal"
    fire_rows = []
    for t in taus:
        cov = rec.loc[rec["has_fire_box"], f"fire_t{t}_cov"].astype(float).mean()
        area = rec.loc[normal, f"fire_t{t}_area"].mean()
        fire_rows.append({"tau_cb_cr": t, "fire_box_covered": cov, "normal_fire_area": area, "J": cov - area})
    fire = pd.DataFrame(fire_rows)
    best_t = int(fire.sort_values(["J", "tau_cb_cr"], ascending=[False, False]).iloc[0]["tau_cb_cr"])
    smoke_rows = []
    for s in smokes:
        c = f"smoke_t{best_t}_{skey(s)}"
        cov = rec.loc[rec["has_smoke_box"], c + "_cov"].astype(float).mean()
        area = rec.loc[normal, c + "_area"].mean()
        smoke_rows.append({"sat_max": s[0], "val_min": s[1], "dark_thr": s[2], "smoke_box_covered": cov,
                           "normal_smoke_area": area, "J": cov - area})
    smoke = (pd.DataFrame(smoke_rows).sort_values(["J", "normal_smoke_area"], ascending=[False, True])
             .reset_index(drop=True))
    return fire.round(4), smoke.round(4), best_t, smoke.iloc[0]


def figures(rec: pd.DataFrame, cfg: dict, tuned: dict, fire: pd.DataFrame, smoke: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    from src.viz import style
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(fire["tau_cb_cr"], fire["fire_box_covered"], "o-", color=style.CLASS_COLOR["fire"], label="fire box covered")
    ax.plot(fire["tau_cb_cr"], fire["normal_fire_area"], "s--", color=style.CLASS_COLOR["normal"], label="area on normal")
    ax.plot(fire["tau_cb_cr"], fire["J"], "^-", color="black", label="J = covered - normal area")
    ax.set_xlabel("τ (|Cb - Cr|)"), ax.set_ylabel("fraction")
    ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style.save(fig, out / "fig_fire_tau", fire)
    plt.close(fig)
    top = smoke.head(12).copy()
    top["setting"] = top.apply(lambda r: f"S<{r.sat_max} V>{r.val_min} D>{r.dark_thr}", axis=1)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    y = np.arange(len(top))[::-1]
    ax.barh(y + 0.2, top["smoke_box_covered"], 0.4, color=style.CLASS_COLOR["smoke"], label="smoke box covered")
    ax.barh(y - 0.2, top["normal_smoke_area"], 0.4, color=style.CLASS_COLOR["normal"], label="area on normal")
    ax.set_yticks(y, top["setting"], fontsize=6), ax.set_xlabel("fraction")
    ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style.save(fig, out / "fig_smoke_grid_top12", top)
    plt.close(fig)
    old = load_config(CONFIG_DIR / "base.yaml")["segmentation"]
    new = {**old, "fire": tuned["fire"], "smoke": tuned["smoke"]}
    ex = rec[rec["error"] == ""].sample(frac=1, random_state=cfg["seed"]).groupby("label").head(3)
    fig, axes = plt.subplots(2, len(ex), figsize=(1.25 * len(ex), 2.9), squeeze=False)
    for j, (_, e) in enumerate(ex.iterrows()):
        p2 = preprocess(load_rgb(PROJECT_ROOT / e["path"]), cfg["preprocess"])
        for i, (name, sc) in enumerate((("before", old), ("after", new))):
            sg = segment(p2, sc)
            axes[i, j].imshow(overlay(p2, sg["fire"], sg["smoke"]))
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
            axes[i, j].set_title(f"{e['label']} · {name}", fontsize=6)
    style.save(fig, out / "fig_before_after", ex[["path", "label"]])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "e03b_seg_tune.yaml"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    validate(cfg, ["seed", "preprocess", "segmentation", "seg_tune.per_class", "seg_tune.tau_cb_cr",
                   "seg_tune.sat_max", "seg_tune.val_min", "seg_tune.dark_thr", "seg_tune.workers"], a.config)
    run_dir, resumed = new_or_resume_run(STAGE, cfg, root=RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT)
    log = setup_logging(run_dir, STAGE)
    log(f"{'RESUMING' if resumed else 'New'} run: {run_dir}")
    if not (run_dir / "meta.json").exists():
        write_meta(run_dir, cfg, data_files=[MANIFEST_DIR / "manifest.csv"], extra={"smoke": a.smoke})
    sp = run_dir / "sample.csv"
    if sp.exists():
        sample = pd.read_csv(sp)
    else:
        sample = build_sample(cfg)
        if a.smoke:
            sample = sample.groupby("label").head(5).reset_index(drop=True)
        sample.to_csv(sp, index=False)
    log("Validation images per class: " + json.dumps(sample["label"].value_counts().to_dict()))
    rec = run_records(run_dir, sample.to_dict("records"), one, cfg, cfg["seg_tune"]["workers"], log, every=100)
    rec.to_csv(run_dir / "records.csv", index=False)
    ok = rec[rec["error"] == ""]
    fire, smoke, best_t, best_s = score(ok, cfg)
    (run_dir / "tables").mkdir(exist_ok=True)
    for nm, t in (("fire_grid", fire), ("smoke_grid", smoke)):
        t.to_csv(run_dir / "tables" / f"{nm}.csv", index=False)
        (run_dir / "tables" / f"{nm}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
    log("Fire threshold (J = fire box covered - fire area on normal images):\n" + fire.to_string(index=False))
    log("Smoke settings, best first:\n" + smoke.head(10).to_string(index=False))
    dark = best_s["dark_thr"]
    tuned = {"fire": {"tau_cb_cr": best_t},
             "smoke": {"sat_max": int(best_s["sat_max"]), "val_min": int(best_s["val_min"]),
                       "dark_patch": cfg["segmentation"]["smoke"]["dark_patch"],
                       "dark_thr": dark if dark == "otsu" else int(dark)}}
    text = (f"# Chosen by E03b on validation images: outputs/{STAGE}/{run_dir.name}\n"
            + yaml.safe_dump({"segmentation": tuned}, sort_keys=False))
    atomic_write_text(run_dir / "segmentation_tuned.yaml", text)
    if not a.smoke:
        atomic_write_text(CONFIG_DIR / "segmentation_tuned.yaml", text)
    log(f"Chosen: {json.dumps(tuned)}")
    figures(ok, cfg, tuned, fire, smoke, run_dir / "figures")
    summary = {"images": int(len(rec)), "failed": int(len(rec) - len(ok)), "chosen": tuned,
               "fire_best": fire[fire["tau_cb_cr"] == best_t].iloc[0].to_dict(), "smoke_best": best_s.to_dict(),
               "default_fire": fire[fire["tau_cb_cr"] == cfg["segmentation"]["fire"]["tau_cb_cr"]].to_dict("records")}
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "results.json", {"metrics": summary, "records_file": "records.csv",
                                                 "meta": json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))})
    (run_dir / "records.partial.jsonl").unlink(missing_ok=True)
    if not a.smoke:
        append_ledger(STAGE, run_dir, f"E03b thresholds chosen on val: {json.dumps(tuned)}")
    log("Done.")


if __name__ == "__main__":
    main()
