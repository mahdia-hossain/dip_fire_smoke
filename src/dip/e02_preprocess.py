"""
E02: preprocessing analysis (PSNR/SSIM per step, denoising and contrast comparisons).

    python run.py e02 [--smoke]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.cfg import load_config, validate
from src.common.runs import (append_ledger, atomic_write_json, new_or_resume_run, read_partial, setup_logging,
                             write_meta)
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.dip import quality as Q
from src.dip.preprocess import denoise, enhance, load_rgb, preprocess_steps

STAGE = "e02_preprocess"
CLASSES = ["fire", "smoke", "normal"]


def measure_one(item: dict, cfg: dict) -> dict:
    t0 = time.perf_counter()
    rec = {k: item[k] for k in ("id", "path", "source", "split", "label", "group", "lookalike_type")}
    try:
        img = load_rgb(PROJECT_ROOT / item["path"])
        rec["orig_h"], rec["orig_w"] = img.shape[:2]
        s = preprocess_steps(img, cfg["preprocess"])
        p0, p1, p2 = s["resized"], s["denoised"], s["enhanced"]
        for name, a, b in (("median", p0, p1), ("clahe", p1, p2), ("full", p0, p2)):
            rec[f"psnr_{name}"], rec[f"ssim_{name}"] = Q.psnr(a, b), Q.ssim(a, b)
        for tag, im in (("p0", p0), ("p1", p1), ("p2", p2)):
            rec[f"entropy_{tag}"], rec[f"rms_{tag}"] = Q.entropy(im), Q.rms_contrast(im)
            rec[f"bright_{tag}"] = Q.mean_brightness(im)
        e = cfg["e02"]
        rng = np.random.default_rng(cfg["seed"] + int(item["id"]))
        sp = Q.add_salt_pepper(p0, e["noise"]["salt_pepper_amount"], rng)
        gn = Q.add_gaussian(p0, e["noise"]["gaussian_sigma"], rng)
        rec["sp_psnr_noisy"], rec["sp_ssim_noisy"] = Q.psnr(p0, sp), Q.ssim(p0, sp)
        rec["gn_psnr_noisy"], rec["gn_ssim_noisy"] = Q.psnr(p0, gn), Q.ssim(p0, gn)
        for m, mcfg in e["denoise_methods"].items():
            out = denoise(p0, **mcfg)
            rec[f"dn_{m}_chg_psnr"], rec[f"dn_{m}_chg_ssim"] = Q.psnr(p0, out), Q.ssim(p0, out)
            rec[f"dn_{m}_edge"] = Q.edge_preservation(p0, out)
            r_sp, r_gn = denoise(sp, **mcfg), denoise(gn, **mcfg)
            rec[f"dn_{m}_sp_psnr"], rec[f"dn_{m}_sp_ssim"] = Q.psnr(p0, r_sp), Q.ssim(p0, r_sp)
            rec[f"dn_{m}_gn_psnr"], rec[f"dn_{m}_gn_ssim"] = Q.psnr(p0, r_gn), Q.ssim(p0, r_gn)
        for c, ccfg in e["contrast_methods"].items():
            out = enhance(p1, **ccfg)
            rec[f"ct_{c}_chg_psnr"], rec[f"ct_{c}_chg_ssim"] = Q.psnr(p1, out), Q.ssim(p1, out)
            rec[f"ct_{c}_entropy_gain"] = Q.entropy(out) - rec["entropy_p1"]
            rec[f"ct_{c}_rms_gain"] = Q.rms_contrast(out) - rec["rms_p1"]
            rec[f"ct_{c}_colour_shift"] = Q.colour_shift(p1, out)
        rec["error"] = ""
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
    rec["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return rec


def build_sample(cfg: dict) -> pd.DataFrame:
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    look = pd.read_csv(MANIFEST_DIR / "lookalike_types.csv")
    man = man[man["split"] != "test_lookalike"].assign(lookalike_type="")
    look = look.merge(pd.read_csv(MANIFEST_DIR / "manifest.csv")[["path", "source", "label", "split"]], on="path")
    df = pd.concat([man[["path", "source", "split", "label", "lookalike_type"]],
                    look[["path", "source", "split", "label", "lookalike_type"]]], ignore_index=True)
    df["group"] = df["split"] + "/" + df["label"]
    n = cfg["e02"]["sample_per_group"]
    sample = (df.sample(frac=1, random_state=cfg["seed"]).groupby("group").head(n)
                .sort_values(["group", "path"]).reset_index(drop=True))
    sample.insert(0, "id", np.arange(len(sample)))
    return sample


def mean_std(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std():.3f}"


def make_tables(rec: pd.DataFrame, cfg: dict, out: Path) -> dict:
    out.mkdir(exist_ok=True)
    tables = {}
    rows = []
    for name, sub in [("all", rec)] + [(c, rec[rec["label"] == c]) for c in CLASSES]:
        rows.append({"class": name, "n": len(sub),
                     **{f"PSNR {k} (dB)": mean_std(sub[f"psnr_{k}"]) for k in ("median", "clahe", "full")},
                     **{f"SSIM {k}": mean_std(sub[f"ssim_{k}"]) for k in ("median", "clahe", "full")},
                     "entropy P0 → P2 (bits)": f"{sub['entropy_p0'].mean():.2f} → {sub['entropy_p2'].mean():.2f}",
                     "RMS contrast P0 → P2": f"{sub['rms_p0'].mean():.1f} → {sub['rms_p2'].mean():.1f}"})
    tables["table_step_change"] = pd.DataFrame(rows)
    rows = []
    for m in cfg["e02"]["denoise_methods"]:
        rows.append({"method": m, "change PSNR (dB)": mean_std(rec[f"dn_{m}_chg_psnr"]),
                     "change SSIM": mean_std(rec[f"dn_{m}_chg_ssim"]),
                     "edge preservation": mean_std(rec[f"dn_{m}_edge"]),
                     "salt-pepper → restored PSNR": f"{rec['sp_psnr_noisy'].mean():.2f} → {rec[f'dn_{m}_sp_psnr'].mean():.2f}",
                     "salt-pepper restored SSIM": mean_std(rec[f"dn_{m}_sp_ssim"]),
                     "Gaussian → restored PSNR": f"{rec['gn_psnr_noisy'].mean():.2f} → {rec[f'dn_{m}_gn_psnr'].mean():.2f}",
                     "Gaussian restored SSIM": mean_std(rec[f"dn_{m}_gn_ssim"])})
    tables["table_denoise"] = pd.DataFrame(rows)
    rows = []
    for c in cfg["e02"]["contrast_methods"]:
        rows.append({"method": c, "change PSNR (dB)": mean_std(rec[f"ct_{c}_chg_psnr"]),
                     "change SSIM": mean_std(rec[f"ct_{c}_chg_ssim"]),
                     "entropy gain (bits)": mean_std(rec[f"ct_{c}_entropy_gain"]),
                     "RMS contrast gain": mean_std(rec[f"ct_{c}_rms_gain"]),
                     "colour shift |Δa|+|Δb|": mean_std(rec[f"ct_{c}_colour_shift"])})
    tables["table_contrast"] = pd.DataFrame(rows)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.md").write_text(t.to_markdown(index=False) if hasattr(t, "to_markdown") else t.to_string(),
                                        encoding="utf-8")
    return tables


def make_figures(rec: pd.DataFrame, sample: pd.DataFrame, cfg: dict, out: Path, log) -> None:
    import matplotlib.pyplot as plt
    from src.viz import style
    out.mkdir(exist_ok=True)
    pcfg = cfg["preprocess"]

    k = cfg["e02"]["figure_examples_per_class"]
    okr = rec[rec["error"] == ""]
    pool = okr[okr["split"] == "test_in"]
    if pool.empty:
        pool = okr
    shuf = lambda d: d.sample(frac=1, random_state=cfg["seed"])
    ex = pd.concat([shuf(pool[pool["label"] == c]).head(k) for c in CLASSES]
                   + [shuf(okr[okr["lookalike_type"].fillna("") != ""]).head(1)])
    if ex.empty:
        log("!! No images for figures")
        return
    ex["lookalike_type"] = ex["lookalike_type"].fillna("")
    fig, axes = plt.subplots(len(ex), 4, figsize=(6.6, 2.15 * len(ex)), squeeze=False)
    rows = []
    for r, (_, e) in enumerate(ex.iterrows()):
        img = load_rgb(PROJECT_ROOT / e["path"])
        s = preprocess_steps(img, pcfg)
        panels = [(img, f"Original\n{img.shape[1]}×{img.shape[0]}\n"), (s["resized"], "P0 resized\n256×256\n"),
                  (s["denoised"], f"P1 median 3×3\nPSNR {e['psnr_median']:.1f} dB\nSSIM {e['ssim_median']:.3f}"),
                  (s["enhanced"], f"P2 CLAHE (L)\nPSNR {e['psnr_clahe']:.1f} dB\nSSIM {e['ssim_clahe']:.3f}")]
        for c, (im, t) in enumerate(panels):
            axes[r, c].imshow(im)
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            axes[r, c].set_title(t, fontsize=7)
        lab = e["label"] + (f" ({e['lookalike_type']})" if e["lookalike_type"] else "")
        axes[r, 0].set_ylabel(lab, fontsize=8)
        rows.append({"path": e["path"], "label": lab, "psnr_median": e["psnr_median"], "ssim_median": e["ssim_median"],
                     "psnr_clahe": e["psnr_clahe"], "ssim_clahe": e["ssim_clahe"]})
    style.save(fig, out / "fig_steps_strip", pd.DataFrame(rows))
    plt.close(fig)

    import cv2
    e = ex[ex["label"] == "fire"].iloc[0] if (ex["label"] == "fire").any() else ex.iloc[0]
    p1 = preprocess_steps(load_rgb(PROJECT_ROOT / e["path"]), pcfg)["denoised"]
    variants = {"P1 (no contrast step)": p1, "HE on L": enhance(p1, method="he_l"),
                "CLAHE on L (ours)": enhance(p1, **pcfg["contrast"])}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    hist_rows = []
    for i, (name, im) in enumerate(variants.items()):
        axes[i].imshow(im), axes[i].set_xticks([]), axes[i].set_yticks([]), axes[i].set_title(name, fontsize=7)
        h = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_RGB2LAB)], [0], None, [256], [0, 256]).ravel()
        axes[3].plot(h / h.sum(), label=name, lw=1)
        hist_rows += [{"variant": name, "L_level": j, "fraction": float(v)} for j, v in enumerate(h / h.sum())]
    axes[3].set_xlabel("L (brightness) level"), axes[3].set_ylabel("fraction of pixels"), axes[3].legend(fontsize=6)
    style.save(fig, out / "fig_histogram_clahe", pd.DataFrame(hist_rows))
    plt.close(fig)

    ok = rec[rec["error"] == ""]
    agg = []
    for c in CLASSES:
        sub = ok[ok["label"] == c]
        for step, key in (("median (P0→P1)", "median"), ("CLAHE (P1→P2)", "clahe"), ("full (P0→P2)", "full")):
            agg.append({"class": c, "step": step, "psnr_mean": sub[f"psnr_{key}"].mean(),
                        "psnr_std": sub[f"psnr_{key}"].std(), "ssim_mean": sub[f"ssim_{key}"].mean(),
                        "ssim_std": sub[f"ssim_{key}"].std(), "n": len(sub)})
    agg = pd.DataFrame(agg)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    w = 0.26
    for j, step in enumerate(style.STEP_COLOR):
        d = agg[agg["step"] == step]
        x = np.arange(len(CLASSES)) + (j - 1) * w
        axes[0].bar(x, d["psnr_mean"], w, yerr=d["psnr_std"], color=style.STEP_COLOR[step], label=step, capsize=2)
        axes[1].bar(x, d["ssim_mean"], w, yerr=d["ssim_std"], color=style.STEP_COLOR[step], label=step, capsize=2)
    for a, yl in zip(axes, ("PSNR input vs output (dB)", "SSIM input vs output")):
        a.set_xticks(np.arange(len(CLASSES)), CLASSES), a.set_ylabel(yl)
    axes[1].set_ylim(0, 1.05)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.13), frameon=False)
    style.save(fig, out / "fig_step_change_bars", agg)
    plt.close(fig)

    e = ex.iloc[0]
    p0 = preprocess_steps(load_rgb(PROJECT_ROOT / e["path"]), pcfg)["resized"]
    rng = np.random.default_rng(cfg["seed"])
    sp = Q.add_salt_pepper(p0, cfg["e02"]["noise"]["salt_pepper_amount"], rng)
    panels = [("clean P0", p0), (f"2% salt-pepper\nPSNR {Q.psnr(p0, sp):.1f} dB", sp)]
    panels += [(f"{m}\nPSNR {Q.psnr(p0, denoise(sp, **mc)):.1f} dB", denoise(sp, **mc))
               for m, mc in cfg["e02"]["denoise_methods"].items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.7))
    for a, (t, im) in zip(axes, panels):
        a.imshow(im[64:192, 64:192]), a.set_xticks([]), a.set_yticks([]), a.set_title(t, fontsize=7)
    style.save(fig, out / "fig_denoise_compare",
               pd.DataFrame([{"panel": t.replace("\n", " ")} for t, _ in panels]))
    plt.close(fig)

    e = ex[ex["label"] == "fire"].iloc[0] if (ex["label"] == "fire").any() else ex.iloc[0]
    p1 = preprocess_steps(load_rgb(PROJECT_ROOT / e["path"]), pcfg)["denoised"]
    panels = [("P1", p1)] + [(f"{c}\ncolour shift {Q.colour_shift(p1, enhance(p1, **cc)):.1f}", enhance(p1, **cc))
                             for c, cc in cfg["e02"]["contrast_methods"].items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.9))
    for a, (t, im) in zip(axes, panels):
        a.imshow(im), a.set_xticks([]), a.set_yticks([]), a.set_title(t, fontsize=7)
    style.save(fig, out / "fig_contrast_compare",
               pd.DataFrame([{"panel": t.replace("\n", " ")} for t, _ in panels]))
    plt.close(fig)
    log(f"Figures saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "e02_preprocess.yaml"))
    ap.add_argument("--smoke", action="store_true", help="20 seeded images into outputs/_smoke/")
    a = ap.parse_args()
    cfg = load_config(a.config)
    validate(cfg, ["seed", "preprocess.size", "preprocess.denoise", "preprocess.contrast", "e02.sample_per_group",
                   "e02.denoise_methods", "e02.contrast_methods", "e02.noise"], a.config)
    if not (MANIFEST_DIR / "lookalike_types.FROZEN").exists():
        raise SystemExit("The look-alike set is not frozen yet. Run `python run.py lookalike-freeze` first.")
    root = RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT
    run_dir, resumed = new_or_resume_run(STAGE, cfg, root=root)
    log = setup_logging(run_dir, STAGE)
    log(f"{'RESUMING' if resumed else 'New'} run: {run_dir}")
    if not (run_dir / "meta.json").exists():
        write_meta(run_dir, cfg, data_files=[MANIFEST_DIR / "manifest.csv", MANIFEST_DIR / "lookalike_types.csv"],
                   extra={"smoke": a.smoke})

    sample_path = run_dir / "sample.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path, keep_default_na=False)
    else:
        sample = build_sample(cfg)
        if a.smoke:
            sample = sample.sample(20, random_state=cfg["seed"]).reset_index(drop=True)
        sample.to_csv(sample_path, index=False)
    log("Sample per group:\n" + sample["group"].value_counts().sort_index().to_string())

    partial = run_dir / "records.partial.jsonl"
    done = {r["id"] for r in read_partial(partial)}
    todo = sample[~sample["id"].isin(done)].fillna("").to_dict("records")
    log(f"{len(sample)} images, {len(done)} already done, {len(todo)} to do")
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=cfg["e02"]["workers"]) as ex, open(partial, "a", encoding="utf-8") as f:
        futures = [ex.submit(measure_one, it, cfg) for it in todo]
        for fut in as_completed(futures):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                log(f"...{n}/{len(todo)}  {rate:.2f} s/image  eta {rate * (len(todo) - n) / 60:.1f} min")

    rec = pd.DataFrame(read_partial(partial)).drop_duplicates("id", keep="last").sort_values("id")
    rec.to_csv(run_dir / "records.csv", index=False)
    failed = rec[rec["error"] != ""]
    if len(failed):
        log(f"!! {len(failed)} images failed (recorded, skipped): e.g. {failed.iloc[0]['path']} -> {failed.iloc[0]['error']}")
    ok = rec[rec["error"] == ""]
    tables = make_tables(ok, cfg, run_dir / "tables")
    for name, t in tables.items():
        log(f"{name}:\n{t.to_string(index=False)}")
    make_figures(rec, sample, cfg, run_dir / "figures", log)
    summary = {
        "images": int(len(rec)), "failed": int(len(failed)),
        "mean_psnr_db": {k: round(float(ok[f"psnr_{k}"].mean()), 3) for k in ("median", "clahe", "full")},
        "mean_ssim": {k: round(float(ok[f"ssim_{k}"].mean()), 4) for k in ("median", "clahe", "full")},
        "denoise_salt_pepper_restored_psnr": {m: round(float(ok[f"dn_{m}_sp_psnr"].mean()), 3)
                                              for m in cfg["e02"]["denoise_methods"]},
        "denoise_edge_preservation": {m: round(float(ok[f"dn_{m}_edge"].mean()), 4)
                                      for m in cfg["e02"]["denoise_methods"]},
        "contrast_colour_shift": {c: round(float(ok[f"ct_{c}_colour_shift"].mean()), 3)
                                  for c in cfg["e02"]["contrast_methods"]},
        "contrast_entropy_gain": {c: round(float(ok[f"ct_{c}_entropy_gain"].mean()), 4)
                                  for c in cfg["e02"]["contrast_methods"]},
    }
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "results.json", {"metrics": summary, "meta": json.loads(
        (run_dir / "meta.json").read_text(encoding="utf-8")), "records": rec.to_dict("records")})
    partial.unlink()
    if not a.smoke:
        append_ledger(STAGE, run_dir, f"E02 preprocessing + PSNR/SSIM on {len(ok)} images")
    log(f"Done. Summary: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
