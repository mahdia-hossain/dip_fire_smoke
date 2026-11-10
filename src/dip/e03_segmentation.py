"""
E03: fire and smoke segmentation on a seeded sample (mask statistics, box overlap, figures).

    python run.py e03 [--smoke]
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
from src.common.parallel import run_records
from src.common.runs import append_ledger, atomic_write_json, new_or_resume_run, setup_logging, write_meta
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.data.boxes import box_mask, label_path, read_boxes
from src.dip.preprocess import load_rgb, preprocess
from src.dip.segmentation import mask_stats, overlay, segment

STAGE = "e03_segmentation"
CLASSES = ["fire", "smoke", "normal"]


def one(item: dict, cfg: dict) -> dict:
    t0 = time.perf_counter()
    rec = {k: item[k] for k in ("id", "path", "source", "split", "label", "lookalike_type")}
    try:
        p2 = preprocess(load_rgb(PROJECT_ROOT / item["path"]), cfg["preprocess"])
        seg = segment(p2, cfg["segmentation"])
        for name in ("fire", "smoke"):
            st = mask_stats(seg[name])
            rec.update({f"{name}_frac": st["frac"], f"{name}_regions": st["regions"],
                        f"{name}_largest_frac": st["largest_frac"], f"{name}_raw_frac": float(seg[f"{name}_raw"].mean())})
        rec["dark_thr"] = seg["dark_thr"]
        ids = cfg["segmentation"]["box_classes"].get(item["source"])
        if ids:
            boxes = read_boxes(PROJECT_ROOT / label_path(item["path"]))
            size = cfg["preprocess"]["size"]
            for name in ("fire", "smoke"):
                bm = box_mask(boxes, ids[name], size).astype(bool)
                m = seg[name].astype(bool)
                rec[f"{name}_box_frac"] = float(bm.mean())
                rec[f"{name}_box_covered"] = float((m & bm).sum() / bm.sum()) if bm.any() else None
                rec[f"{name}_mask_in_box"] = float((m & bm).sum() / m.sum()) if (m.any() and bm.any()) else None
        rec["error"] = ""
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
    rec["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return rec


def build_sample(cfg: dict) -> pd.DataFrame:
    n = cfg["e03"]["sample_per_class"]
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    test = man[man["split"] == "test_in"].assign(lookalike_type="")
    look = pd.read_csv(MANIFEST_DIR / "lookalike_types.csv").merge(man[["path", "source", "split", "label"]], on="path")
    parts = [test[test["label"] == c].sample(frac=1, random_state=cfg["seed"]).head(n) for c in CLASSES]
    parts.append(look.sample(frac=1, random_state=cfg["seed"]).head(cfg["e03"]["lookalike_sample"]))
    s = pd.concat(parts, ignore_index=True)[["path", "source", "split", "label", "lookalike_type"]]
    s.insert(0, "id", np.arange(len(s)))
    return s


def tables(rec: pd.DataFrame, out: Path) -> dict:
    out.mkdir(exist_ok=True)
    rec = rec.assign(group=np.where(rec["lookalike_type"] != "", "look-alike: " + rec["lookalike_type"], rec["label"]))
    area = rec.groupby("group").agg(n=("id", "size"), fire_mask_area=("fire_frac", "mean"),
                                    smoke_mask_area=("smoke_frac", "mean"), fire_regions=("fire_regions", "mean"),
                                    smoke_regions=("smoke_regions", "mean"),
                                    images_with_fire_mask=("fire_frac", lambda s: (s > 0).mean()),
                                    images_with_smoke_mask=("smoke_frac", lambda s: (s > 0).mean())).round(4).reset_index()
    rows = []
    for name in ("fire", "smoke"):
        col = f"{name}_box_covered"
        if col not in rec:
            continue
        sub = rec[rec[col].notna()]
        rows.append({"mask": name, "images_with_this_box": int(len(sub)),
                     "mean_box_area_covered": round(float(sub[col].mean()), 4) if len(sub) else None,
                     "median_box_area_covered": round(float(sub[col].median()), 4) if len(sub) else None,
                     "images_box_covered_>10%": round(float((sub[col] > 0.10).mean()), 4) if len(sub) else None,
                     "mean_mask_inside_box": round(float(sub[f"{name}_mask_in_box"].mean()), 4)
                     if sub[f"{name}_mask_in_box"].notna().any() else None})
    box = pd.DataFrame(rows)
    for nm, t in (("table_mask_area", area), ("table_box_overlap", box)):
        t.to_csv(out / f"{nm}.csv", index=False)
        (out / f"{nm}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
    return {"table_mask_area": area, "table_box_overlap": box}


def figures(rec: pd.DataFrame, cfg: dict, out: Path, log) -> None:
    import cv2
    import matplotlib.pyplot as plt
    from src.viz import style
    out.mkdir(exist_ok=True)
    ok = rec[rec["error"] == ""]
    shuf = lambda d: d.sample(frac=1, random_state=cfg["seed"])
    picks = [(shuf(ok[(ok["label"] == "fire") & (ok["fire_frac"] > 0)]).head(1), "fire"),
             (shuf(ok[(ok["label"] == "smoke") & (ok["smoke_frac"] > 0)]).head(1), "smoke")]
    picks = [(d.iloc[0], k) for d, k in picks if len(d)]
    if picks:
        fig, axes = plt.subplots(len(picks), 5, figsize=(7.0, 1.95 * len(picks)), squeeze=False)
        for r, (e, kind) in enumerate(picks):
            p2 = preprocess(load_rgb(PROJECT_ROOT / e["path"]), cfg["preprocess"])
            seg = segment(p2, cfg["segmentation"])
            if kind == "fire":
                mid = [("YCbCr rule\n(raw mask)", seg["fire_raw"]), ("after\nopening", seg["fire_open"])]
            else:
                mid = [("dark\nchannel", seg["dark"]), ("smoke rule\n(raw mask)", seg["smoke_raw"])]
            panels = [("P2 input\n(preprocessed)", p2), *mid, ("final mask\n(closed, cleaned)", seg[kind]),
                      ("overlay\nred fire / blue smoke", overlay(p2, seg["fire"], seg["smoke"]))]
            for c, (t, im) in enumerate(panels):
                axes[r, c].imshow(im, cmap=None if im.ndim == 3 else "gray")
                axes[r, c].set_xticks([]), axes[r, c].set_yticks([]), axes[r, c].set_title(t, fontsize=7)
            axes[r, 0].set_ylabel(kind, fontsize=8)
        style.save(fig, out / "fig_segmentation_steps", pd.DataFrame([{"path": e["path"], "row": k} for e, k in picks]))
        plt.close(fig)
    groups = [(c, shuf(ok[(ok["label"] == c) & (ok["lookalike_type"] == "")]).head(4)) for c in CLASSES]
    groups += [(f"look-alike\n{t}", shuf(g).head(1)) for t, g in ok[ok["lookalike_type"] != ""].groupby("lookalike_type")]
    tiles = [(name, row) for name, d in groups for _, row in d.iterrows()]
    if tiles:
        cols = 6
        rows_n = int(np.ceil(len(tiles) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(7.0, 1.35 * rows_n), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, (name, e) in enumerate(tiles):
            p2 = preprocess(load_rgb(PROJECT_ROOT / e["path"]), cfg["preprocess"])
            seg = segment(p2, cfg["segmentation"])
            axes[i // cols, i % cols].imshow(overlay(p2, seg["fire"], seg["smoke"]))
            axes[i // cols, i % cols].set_title(f"{name}\nfire {seg['fire'].mean():.0%} · smoke {seg['smoke'].mean():.0%}",
                                                fontsize=6)
        style.save(fig, out / "fig_mask_gallery", pd.DataFrame([{"group": n, "path": e["path"]} for n, e in tiles]))
        plt.close(fig)
    if picks and picks[-1][1] == "smoke":
        e = picks[-1][0]
        p2 = preprocess(load_rgb(PROJECT_ROOT / e["path"]), cfg["preprocess"])
        seg = segment(p2, cfg["segmentation"])
        h = cv2.calcHist([seg["dark"]], [0], None, [256], [0, 256]).ravel()
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        ax.plot(h / h.sum(), color=style.CLASS_COLOR["smoke"])
        ax.axvline(seg["dark_thr"], color="black", ls="--", lw=0.8)
        ax.text(seg["dark_thr"] + 3, ax.get_ylim()[1] * 0.85, f"Otsu threshold = {seg['dark_thr']:.0f}", fontsize=7)
        ax.set_xlabel("dark-channel value"), ax.set_ylabel("fraction of pixels")
        style.save(fig, out / "fig_dark_channel_otsu", pd.DataFrame({"value": range(256), "fraction": h / h.sum()}))
        plt.close(fig)
    log(f"Figures saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "e03_segmentation.yaml"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    validate(cfg, ["seed", "preprocess", "segmentation.fire", "segmentation.smoke", "segmentation.morph",
                   "segmentation.box_classes", "e03.sample_per_class", "e03.lookalike_sample", "e03.workers"], a.config)
    run_dir, resumed = new_or_resume_run(STAGE, cfg, root=RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT)
    log = setup_logging(run_dir, STAGE)
    log(f"{'RESUMING' if resumed else 'New'} run: {run_dir}")
    if not (run_dir / "meta.json").exists():
        write_meta(run_dir, cfg, data_files=[MANIFEST_DIR / "manifest.csv", MANIFEST_DIR / "lookalike_types.csv"],
                   extra={"smoke": a.smoke})
    sp = run_dir / "sample.csv"
    if sp.exists():
        sample = pd.read_csv(sp, keep_default_na=False)
    else:
        sample = build_sample(cfg)
        if a.smoke:
            sample = sample.sample(20, random_state=cfg["seed"]).reset_index(drop=True)
        sample.to_csv(sp, index=False)
    rec = run_records(run_dir, sample.fillna("").to_dict("records"), one, cfg, cfg["e03"]["workers"], log, every=100)
    rec["lookalike_type"] = rec["lookalike_type"].fillna("")
    rec.to_csv(run_dir / "records.csv", index=False)
    ok = rec[rec["error"] == ""]
    if len(ok) < len(rec):
        log(f"!! {len(rec) - len(ok)} images failed (recorded): {rec[rec['error'] != ''].iloc[0]['error']}")
    t = tables(ok, run_dir / "tables")
    for k, v in t.items():
        log(f"{k}:\n{v.to_string(index=False)}")
    figures(rec, cfg, run_dir / "figures", log)
    summary = {"images": int(len(rec)), "failed": int(len(rec) - len(ok)),
               "mask_area_by_group": t["table_mask_area"].set_index("group")[["fire_mask_area", "smoke_mask_area"]]
               .to_dict(orient="index"),
               "box_overlap": t["table_box_overlap"].to_dict(orient="records")}
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "results.json", {"metrics": summary, "meta": json.loads(
        (run_dir / "meta.json").read_text(encoding="utf-8")), "records": rec.to_dict("records")})
    (run_dir / "records.partial.jsonl").unlink(missing_ok=True)
    if not a.smoke:
        append_ledger(STAGE, run_dir, f"E03 segmentation on {len(ok)} images (masks + box overlap)")
    log("Done.")


if __name__ == "__main__":
    main()
