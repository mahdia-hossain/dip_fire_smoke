"""
E12: restoration study with the lab-manual DIP operations (validation images, CPU only, no training).

    python run.py restore-study [--smoke]

Noise is added to the full-size photo before resizing, as in real use (a camera's noise is there before the
pipeline shrinks the photo). --after-resize repeats the first E12 version (noise added after resizing).

1. Denoiser choice: clean validation photos get Gaussian or salt-and-pepper noise, each filter restores them, and
   the result is compared with the clean photo (PSNR, SSIM, Canny edge F1).
2. Noise detectors: Immerkaer sigma vs isolated point detection, for Gaussian and salt-and-pepper noise (ROC AUC).
3. Pipeline gallery: the final adaptive pipeline step by step on a few test photos.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd
import yaml

from src.common.parallel import run_records
from src.common.runs import append_ledger, atomic_write_json, new_or_resume_run, setup_logging, write_meta
from src.common.cfg import load_config
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.dip.noise import estimate_sigma
from src.dip.preprocess import load_rgb, preprocess, resize
from src.dip.quality import psnr, ssim
from src.dip.restore import DENOISERS, edge_f1, gaussian_noise, isolated_point_share, salt_pepper

STAGE = "e12_restoration"
NOISES = {"gauss_10": ("g", 10), "gauss_25": ("g", 25), "saltpepper_2pct": ("s", 0.02), "saltpepper_5pct": ("s", 0.05)}


def degrade(clean, kind, rng):
    t, v = NOISES[kind]
    return gaussian_noise(clean, v, rng) if t == "g" else salt_pepper(clean, v, rng)


def one(item: dict, cfg: dict) -> dict:
    rec = {"id": item["id"], "path": item["path"], "label": item["label"]}
    try:
        raw = load_rgb(PROJECT_ROOT / item["path"])
        clean = resize(raw, cfg["size"])
        rng = np.random.default_rng(item["id"])
        rec["det|clean|sigma"] = estimate_sigma(clean)
        rec["det|clean|points"] = isolated_point_share(clean)
        for kind in NOISES:
            noisy = resize(degrade(raw, kind, rng), cfg["size"]) if cfg["noise_before_resize"] else degrade(clean, kind, rng)
            rec[f"det|{kind}|sigma"] = estimate_sigma(noisy)
            rec[f"det|{kind}|points"] = isolated_point_share(noisy)
            for m, fn in DENOISERS.items():
                t0 = time.perf_counter()
                out = fn(noisy)
                ms = (time.perf_counter() - t0) * 1000
                rec.update({f"noise|{kind}|{m}|psnr": psnr(clean, out), f"noise|{kind}|{m}|ssim": ssim(clean, out),
                            f"noise|{kind}|{m}|edge_f1": edge_f1(out, clean), f"noise|{kind}|{m}|ms": ms})
        rec["error"] = ""
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
    return rec


def long_table(rec: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in rec.columns if c.split("|")[0] == "noise"]
    m = rec[cols].mean().reset_index()
    m.columns = ["key", "value"]
    p = m["key"].str.split("|", expand=True)
    m["task"], m["degradation"], m["method"], m["metric"] = p[0], p[1], p[2], p[3]
    return m.pivot_table(index=["task", "degradation", "method"], columns="metric", values="value").reset_index()


def ref_cdf_from_train(cfg_size: int, n: int, seed: int) -> list:
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    tr = man[man["split"] == "train"].sample(n, random_state=seed)
    hist = np.zeros(256)
    for p in tr["path"]:
        L = cv2.cvtColor(resize(load_rgb(PROJECT_ROOT / p), cfg_size), cv2.COLOR_RGB2LAB)[..., 0]
        hist += np.bincount(L.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    return (cdf / cdf[-1]).round(6).tolist()


def figures(rec, tab, det, cfg, run_dir, log):
    import matplotlib.pyplot as plt
    from src.viz import style
    from sklearn.metrics import roc_curve
    fd = run_dir / "figures"
    fd.mkdir(exist_ok=True)
    nz = tab[tab["task"] == "noise"]
    for metric in ("ssim", "psnr"):
        piv = nz.pivot(index="method", columns="degradation", values=metric).reindex(list(DENOISERS))[list(NOISES)]
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(piv.columns)), piv.columns, fontsize=7, rotation=20)
        ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
        for (i, j), v in np.ndenumerate(piv.to_numpy()):
            ax.text(j, i, f"{v:.3f}" if metric == "ssim" else f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if v < np.nanmean(piv.to_numpy()) else "black")
        fig.colorbar(im, ax=ax, shrink=0.7, label=metric.upper() + " vs clean photo")
        style.save(fig, fd / f"fig_denoise_{metric}", piv.reset_index())
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
    for ax, kind in zip(axes, ["gauss_25", "saltpepper_5pct"]):
        for i, fam in enumerate(["average", "gaussian", "median"]):
            ys = [nz[(nz["degradation"] == kind) & (nz["method"] == f"{fam}_{k}")]["ssim"].iloc[0] for k in (3, 5, 7)]
            ax.plot([3, 5, 7], ys, "o-", color=style.OKABE_ITO[i], label=fam)
        ax.set_title(kind, fontsize=8), ax.set_xlabel("kernel size"), ax.set_xticks([3, 5, 7])
    axes[0].set_ylabel("SSIM"), axes[0].legend(fontsize=7, frameon=False)
    style.save(fig, fd / "fig_kernel_size", nz[nz["method"].str.contains("_[357]$")])
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, (group, kinds) in zip(axes, (("Gaussian noise", ["gauss_10", "gauss_25"]),
                                         ("salt-and-pepper", ["saltpepper_2pct", "saltpepper_5pct"]))):
        for i, d in enumerate(("sigma", "points")):
            y = np.r_[np.zeros(len(rec)), np.ones(len(rec) * len(kinds))]
            s = np.r_[rec[f"det|clean|{d}"], *[rec[f"det|{k}|{d}"] for k in kinds]]
            fpr, tpr, _ = roc_curve(y, s)
            auc = det[(det["noise_type"] == group) & (det["detector"] == d)]["auc"].iloc[0]
            ax.plot(fpr, tpr, color=style.OKABE_ITO[i],
                    label=f"{'Immerkaer sigma' if d == 'sigma' else 'isolated points'} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.6), ax.set_title(group, fontsize=8)
        ax.set_xlabel("false positive rate"), ax.legend(fontsize=6, frameon=False, loc="lower right")
    axes[0].set_ylabel("true positive rate")
    style.save(fig, fd / "fig_noise_detectors", det)
    plt.close(fig)
    illustrations(rec, cfg, fd, style, plt)
    log(f"Figures saved to {fd}")


def illustrations(rec, cfg, fd, style, plt):
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    fire = man[(man["split"] == "val") & (man["label"] == "fire")].sample(1, random_state=cfg["seed"])["path"].iloc[0]
    img = resize(load_rgb(PROJECT_ROOT / fire), cfg["size"])
    rng = np.random.default_rng(0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def grid(panels, name, cols):
        rows = int(np.ceil(len(panels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 1.6 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (t, im) in zip(axes.ravel(), panels):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_title(t, fontsize=6)
        style.save(fig, fd / name, pd.DataFrame({"panel": [p[0] for p in panels], "image": fire}))
        plt.close(fig)

    noisy_sp, noisy_g = salt_pepper(img, 0.05, rng), np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
    sel = ["average_3", "weighted_avg_3", "gaussian_3", "median_3", "min_3", "max_3", "midpoint_3", "bilateral", "nlmeans"]
    grid([("salt & pepper 5%", noisy_sp), *[(m, DENOISERS[m](noisy_sp)) for m in sel],
          ("Gaussian noise σ25", noisy_g), *[(m, DENOISERS[m](noisy_g)) for m in sel]], "fig_filter_examples", 5)
    sob = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    pts = (np.abs(cv2.filter2D(cv2.cvtColor(noisy_sp, cv2.COLOR_RGB2GRAY).astype(np.float32), -1,
                               np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32))) > 600).astype(np.uint8) * 255
    grid([("input", img), ("Sobel", np.clip(sob / sob.max() * 255, 0, 255).astype(np.uint8)),
          ("Laplacian", np.clip(lap / lap.max() * 255 * 3, 0, 255).astype(np.uint8)),
          ("Canny", cv2.Canny(gray, 100, 200)), ("isolated points (on S&P 5%)", pts)],
         "fig_edges_isolated_points", 5)
    nt = CONFIG_DIR / "noise_tuned.yaml"
    if nt.exists():
        thr = yaml.safe_load(nt.read_text())["preprocess"]["adaptive"]["sigma_thr"]
        base = load_config(CONFIG_DIR / "base.yaml")["preprocess"]
        pc = {**base, "adaptive": {"sigma_thr": thr}}
        test = man[man["split"] == "test_in"].sample(frac=1, random_state=cfg["seed"]).groupby("label").head(1)
        fig, axes = plt.subplots(len(test) * 2, 3, figsize=(4.8, 3.2 * len(test)), squeeze=False)
        for i, p in enumerate(test["path"]):
            raw = load_rgb(PROJECT_ROOT / p)
            for k, (tag, x) in enumerate((("original", raw), ("+ noise σ20", np.clip(
                    raw + np.random.default_rng(i).normal(0, 20, raw.shape), 0, 255).astype(np.uint8)))):
                p0 = resize(x, cfg["size"])
                s = estimate_sigma(p0)
                out = preprocess(x, pc)
                row = 2 * i + k
                axes[row, 0].imshow(p0), axes[row, 0].set_title(f"{tag}", fontsize=6)
                axes[row, 1].text(0.5, 0.5, f"noise σ = {s:.2f}\nthreshold {thr}\n" +
                                  ("noisy: median + CLAHE" if s > thr else "clean: no change"),
                                  ha="center", va="center", fontsize=7)
                axes[row, 2].imshow(out)
                axes[row, 2].set_title(f"output  PSNR {psnr(p0, out):.1f}  SSIM {ssim(p0, out):.3f}", fontsize=6)
                for ax in axes[row]:
                    ax.axis("off")
        style.save(fig, fd / "fig_pipeline_gallery", test[["path", "label"]])
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--after-resize", action="store_true")
    a = ap.parse_args()
    base = load_config(CONFIG_DIR / "base.yaml")
    cfg = {"seed": base["seed"], "size": base["preprocess"]["size"], "per_class": 50,
           "noise_before_resize": "--after-resize" not in sys.argv}
    root = RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT
    run_dir, resumed = new_or_resume_run(STAGE, cfg, root=root)
    log = setup_logging(run_dir, STAGE)
    log(f"{'RESUMING' if resumed else 'New'} run: {run_dir}")
    work = cfg
    write_meta(run_dir, cfg, data_files=[MANIFEST_DIR / "manifest.csv"], extra={"smoke": a.smoke})
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    val = man[man["split"] == "val"]
    n = 3 if a.smoke else cfg["per_class"]
    s = pd.concat([val[val["label"] == c].sample(frac=1, random_state=cfg["seed"]).head(n)
                   for c in ["fire", "smoke", "normal"]], ignore_index=True)[["path", "label"]]
    s.insert(0, "id", np.arange(len(s)))
    rec = run_records(run_dir, s.to_dict("records"), one, work, int(os.environ.get("DIP_WORKERS", "2")), log, every=30)
    rec = rec[rec["error"] == ""].reset_index(drop=True)
    rec.to_csv(run_dir / "records.csv", index=False)
    tab = long_table(rec)
    from sklearn.metrics import roc_auc_score
    det = []
    for group, kinds in (("Gaussian noise", ["gauss_10", "gauss_25"]), ("salt-and-pepper", ["saltpepper_2pct", "saltpepper_5pct"])):
        for d in ("sigma", "points"):
            y = np.r_[np.zeros(len(rec)), np.ones(len(rec) * len(kinds))]
            sc = np.r_[rec[f"det|clean|{d}"], *[rec[f"det|{k}|{d}"] for k in kinds]]
            det.append({"noise_type": group, "detector": d, "auc": round(float(roc_auc_score(y, sc)), 4)})
    det = pd.DataFrame(det)
    td = run_dir / "tables"
    td.mkdir(exist_ok=True)
    tab.round(4).to_csv(td / "restoration_all.csv", index=False)
    best = (tab.sort_values("ssim", ascending=False).groupby(["task", "degradation"]).head(1)
            [["task", "degradation", "method", "psnr", "ssim", "edge_f1"]].round(4))
    best.to_csv(td / "best_per_degradation.csv", index=False)
    (td / "best_per_degradation.md").write_text(best.to_markdown(index=False), encoding="utf-8")
    det.to_csv(td / "noise_detectors.csv", index=False)
    (td / "restoration_all.md").write_text(tab.round(4).to_markdown(index=False), encoding="utf-8")
    log("Best method per degradation (by SSIM vs the clean photo):\n" + best.to_string(index=False))
    log("Noise detectors (AUC, 1.0 = perfect):\n" + det.to_string(index=False))
    figures(rec, tab, det, work, run_dir, log)
    summary = {"images": int(len(rec)), "best": best.to_dict("records"), "detectors": det.to_dict("records")}
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "results.json", {"metrics": summary, "records_file": "records.csv"})
    (run_dir / "records.partial.jsonl").unlink(missing_ok=True)
    if not a.smoke:
        append_ledger(STAGE, run_dir, "E12 restoration study: " + "; ".join(
            f"{r.degradation} -> {r.method}" for r in best.itertuples()))
    log("Done.")


if __name__ == "__main__":
    main()
