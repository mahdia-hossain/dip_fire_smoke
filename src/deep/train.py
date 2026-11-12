"""
E06-E09: train one deep model (B, C, D or E).

    python run.py train --model B [--smoke]
    python run.py train --model E
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.cfg import load_config, validate
from src.common.runs import (append_ledger, atomic_write_json, new_or_resume_run, seed_everything, setup_logging,
                             write_meta)
from src.config import CKPT_ROOT, CONFIG_DIR, RESULTS_ROOT
from src.deep.data import CLASSES, MEAN, STD, FireSmokeDataset, worker_init
from src.deep.models import build_model, has_maps, uses_features
from src.dip.features import FEATURE_NAMES
from src.eval.metrics import classification_metrics, summary_row

EVAL_SPLITS = ["val", "test_in", "test_ext", "test_lookalike"]


def newest_features(root: Path) -> Path:
    runs = sorted((root / "e04_features").glob("run_*/features.csv"))
    runs = [p for p in runs if (p.parent / "results.json").exists()]
    if not runs:
        sys.exit(f"No finished E04 run under {root / 'e04_features'}. Run:  python run.py e04")
    return runs[-1]


def loaders(feat: pd.DataFrame, cfg: dict, smoke: bool, use_f: bool, maps: bool):
    tr = feat[feat["split"] == "train"]
    if smoke:
        tr = tr.groupby("label").head(16)
    mean = tr[FEATURE_NAMES].to_numpy(np.float32).mean(0)
    std = tr[FEATURE_NAMES].to_numpy(np.float32).std(0) + 1e-6
    t = cfg["train"]
    live = (cfg["segmentation"], cfg["features"]) if use_f and cfg["deep"]["live_features"] else None
    mcfg = {"box_classes": cfg["segmentation"]["box_classes"], "grid": cfg["deep"]["map_grid"],
            "source": cfg["model"]["map_source"], "seg": cfg["segmentation"]} if maps else None
    ds = {"train": FireSmokeDataset(tr, cfg["preprocess"], True, mean, std, live, mcfg, cfg["model"]["noise_aug"])}
    for s in EVAL_SPLITS:
        d = feat[feat["split"] == s]
        if smoke:
            d = d.groupby("label").head(8)
        ds[s] = FireSmokeDataset(d, cfg["preprocess"], False, mean, std)
    common = dict(num_workers=t["num_workers"], pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init)
    keep = t["num_workers"] > 0
    dl = {"train": DataLoader(ds["train"], batch_size=t["batch_size"], shuffle=True, drop_last=True,
                              persistent_workers=keep, **common),
          "val": DataLoader(ds["val"], batch_size=t["batch_size"], shuffle=False, persistent_workers=False, **common)}
    for s in EVAL_SPLITS[1:]:
        dl[s] = DataLoader(ds[s], batch_size=t["batch_size"], shuffle=False, persistent_workers=False, **common)
    frames = {"train": tr, **{s: (feat[feat["split"] == s].groupby("label").head(8) if smoke
                                  else feat[feat["split"] == s]) for s in EVAL_SPLITS}}
    return dl, frames, {"mean": mean.tolist(), "std": std.tolist()}


@torch.no_grad()
def predict(model, loader, dev, use_f: bool, amp: bool):
    model.eval()
    out, t0, n = [], time.perf_counter(), 0
    for x, f, y, idx, *_ in loader:
        x, f = x.to(dev, non_blocking=True), f.to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and dev.type == "cuda"):
            logits = model(x, f if use_f else None)
        out.append((idx.numpy(), torch.softmax(logits.float(), 1).cpu().numpy()))
        n += len(idx)
    ms = (time.perf_counter() - t0) * 1000 / max(n, 1)
    idx = np.concatenate([o[0] for o in out])
    prob = np.concatenate([o[1] for o in out])
    return idx, prob, ms


@torch.no_grad()
def map_gallery(model, frame: pd.DataFrame, cfg: dict, norm: dict, dev, out_base, per_class: int = 3) -> None:
    import cv2
    import matplotlib.pyplot as plt
    from src.viz import style
    model.eval()
    ex = frame.sample(frac=1, random_state=cfg["seed"]).groupby("label").head(per_class)
    ds = FireSmokeDataset(ex, cfg["preprocess"], False, np.array(norm["mean"]), np.array(norm["std"]))
    fig, axes = plt.subplots(3, len(ex), figsize=(1.2 * len(ex), 4.0), squeeze=False)
    for j in range(len(ex)):
        x, f, y, _ = ds[j]
        logits, maps = model.forward_with_maps(x[None].to(dev), f[None].to(dev) if uses_features(cfg["model"]["kind"]) else None)
        prob = torch.sigmoid(maps.float())[0].cpu().numpy()
        img = (x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)
        size = img.shape[0]
        pred = CLASSES[int(logits.argmax(1))]
        panels = [(img, f"true {CLASSES[y]}\npred {pred}", None),
                  (cv2.resize(prob[0], (size, size)), "fire map", "Reds"),
                  (cv2.resize(prob[1], (size, size)), "smoke map", "Blues")]
        for i, (im, title, cmap) in enumerate(panels):
            axes[i, j].imshow(im, cmap=cmap, vmin=0, vmax=1)
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([]), axes[i, j].set_title(title, fontsize=6)
    style.save(fig, out_base, ex[["path", "label"]])
    plt.close(fig)


def records(frame: pd.DataFrame, idx, prob, model_name: str) -> pd.DataFrame:
    rec = frame.iloc[idx][["path", "source", "split", "label", "lookalike_type"]].copy()
    rec["model"] = model_name
    for i, c in enumerate(CLASSES):
        rec[f"p_{c}"] = prob[:, i]
    rec["pred"] = np.array(CLASSES)[prob.argmax(1)]
    return rec.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["B", "C", "D", "E", "EM", "CM", "CR", "EMD", "CA", "CRN", "CN", "CA2", "CA3"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    spec = {"B": ("e06_mobilenetv3", "cnn", "image"), "C": ("e07_mobilevit", "vit", "image"),
            "D": ("e08_mobilevit_concat", "vit", "concat"), "E": ("e09_mobilevit_diptokens", "vit", "tokens"),
            "EM": ("e09b_mobilevit_diptokens_maps", "vit", "tokens_maps"),
            "CM": ("e09c_mobilevit_maps", "vit", "image_maps"),
            "CR": ("e07c_mobilevit_rawinput", "vit", "image"),
            "EMD": ("e09d_mobilevit_diptokens_dipmaps", "vit", "tokens_maps"),
            "CA": ("e07d_mobilevit_adaptive_noiseaug", "vit", "image"),
            "CRN": ("e07e_mobilevit_raw_noiseaug", "vit", "image"),
            "CN": ("e07f_mobilevit_dip_noiseaug", "vit", "image"),
            "CA2": ("e07g_mobilevit_fulladaptive_noiseaug", "vit", "image"),
            "CA3": ("e07h_mobilevit_noisetype_adaptive", "vit", "image")}
    stage, arch_key, kind = spec[a.model]
    model_name = {"B": "B_mobilenetv3", "C": "C_mobilevit", "D": "D_mobilevit_concat",
                  "E": "E_mobilevit_diptokens", "EM": "EM_mobilevit_diptokens_maps", "CM": "CM_mobilevit_maps",
                  "CR": "CR_mobilevit_rawinput", "EMD": "EMD_mobilevit_diptokens_dipmaps",
                  "CA": "CA_mobilevit_adaptive_noiseaug", "CRN": "CRN_mobilevit_raw_noiseaug",
                  "CN": "CN_mobilevit_dip_noiseaug",
                  "CA2": "CA2_mobilevit_fulladaptive_noiseaug",
                  "CA3": "CA3_mobilevit_noisetype_adaptive"}[a.model]
    cfg = load_config(a.config or CONFIG_DIR / "deep_train.yaml")
    validate(cfg, ["seed", "preprocess", "train.img_size", "train.batch_size", "train.epochs", "train.lr",
                   "train.weight_decay", "train.warmup_epochs", "train.label_smoothing", "train.amp",
                   "train.models", "deep.n_boot", "deep.live_features", "deep.map_loss_weight", "deep.map_grid"], "deep_train.yaml")
    if a.seed is not None:
        cfg = {**cfg, "seed": a.seed}
    if a.seed is not None:
        cfg["seed"] = a.seed
    root = RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT
    fpath = Path(a.features) if a.features else newest_features(root)
    arch = cfg["train"]["models"][arch_key]
    cfg = {**cfg, "model": {"id": a.model, "name": model_name, "arch": arch, "kind": kind},
           "features_file": str(fpath)}
    if a.model in ("CR", "CRN"):
        cfg["preprocess"] = {**cfg["preprocess"], "denoise": {"method": "none"}, "contrast": {"method": "none"}}
    if a.model in ("CA", "CA2", "CA3"):
        tuned = ([("noisetype_tuned.yaml", "noisetype-tune")] if a.model == "CA3" else [("noise_tuned.yaml", "noise-tune")])
        tuned += [("dark_tuned.yaml", "dark-tune")] if a.model == "CA2" else []
        adaptive = {}
        for fname, stage_cmd in tuned:
            if not (CONFIG_DIR / fname).exists():
                sys.exit(f"configs/{fname} not found. Run:  python run.py {stage_cmd}")
            adaptive.update(load_config(CONFIG_DIR / fname)["preprocess"]["adaptive"])
        cfg["preprocess"] = {**cfg["preprocess"], "adaptive": adaptive}
    cfg["model"]["noise_aug"] = cfg["deep"]["noise_aug"] if a.model in ("CA", "CRN", "CN", "CA2", "CA3") else None
    cfg["model"]["map_source"] = "dip" if a.model == "EMD" else "boxes"
    if a.smoke:
        cfg["train"] = {**cfg["train"], "epochs": 2, "batch_size": 8, "num_workers": 0}
    run_dir, resumed = new_or_resume_run(stage, cfg, root=root)
    log = setup_logging(run_dir, stage)
    if os.environ.get("DIP_WORKERS"):
        cfg["train"] = {**cfg["train"], "num_workers": int(os.environ["DIP_WORKERS"])}
        log(f"DIP_WORKERS override: {cfg['train']['num_workers']} data workers (does not change the run's config hash)")
    seed_everything(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{'RESUMING' if resumed else 'New'} run: {run_dir}\nModel {model_name} ({arch}, {kind}) on {dev}")
    log(f"Features: {fpath}")

    feat = pd.read_csv(fpath, keep_default_na=False)
    feat = feat[feat["error"] == ""].reset_index(drop=True)
    feat[FEATURE_NAMES] = feat[FEATURE_NAMES].astype(np.float32)
    dl, frames, norm = loaders(feat, cfg, a.smoke, uses_features(kind), has_maps(kind))
    log("Images: " + json.dumps({k: len(v.dataset) for k, v in dl.items()}))

    model = build_model(kind, arch, pretrained=not a.smoke).to(dev)
    use_f, use_maps = uses_features(kind), has_maps(kind)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    t = cfg["train"]
    counts = frames["train"]["label"].value_counts().reindex(CLASSES).fillna(0).to_numpy()
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w if t["class_weighted_loss"] else None,
                               label_smoothing=t["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps = max(len(dl["train"]), 1)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, max(t["warmup_epochs"] * steps, 1))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(t["epochs"] * steps - t["warmup_epochs"] * steps, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [max(t["warmup_epochs"] * steps, 1)])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and dev.type == "cuda")
    log(f"{params:.2f} M parameters, class weights {w.tolist()}, {steps} steps per epoch")

    ck_dir = CKPT_ROOT / stage / run_dir.name
    ck_dir.mkdir(parents=True, exist_ok=True)
    last_ck, best_ck = ck_dir / "last.pt", ck_dir / "best.pt"
    hist, start_ep, best = [], 0, {"val_macro_f1": -1.0, "epoch": -1}
    if last_ck.exists():
        s = torch.load(last_ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]), opt.load_state_dict(s["opt"]), sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        hist, start_ep, best = s["history"], s["epoch"] + 1, s["best"]
        log(f"Resumed: continuing at epoch {start_ep + 1} (best val macro-F1 {best['val_macro_f1']:.4f})"
            if start_ep < t["epochs"] else "Training already finished; running the evaluation only")

    for ep in range(start_ep, t["epochs"]):
        model.train()
        t0, tot, seen, correct, map_tot = time.time(), 0.0, 0, 0, 0.0
        for i, (x, f, y, _, *extra) in enumerate(dl["train"]):
            x, f, y = x.to(dev, non_blocking=True), f.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                if use_maps:
                    logits, maps = model.forward_with_maps(x, f if use_f else None)
                    target, valid = extra[0].to(dev), extra[1].to(dev)
                    map_loss = (bce(maps.float(), target).mean((1, 2, 3)) * valid).sum() / valid.sum().clamp(min=1)
                    loss = crit(logits, y) + cfg["deep"]["map_loss_weight"] * map_loss
                    map_tot += map_loss.item() * len(y)
                else:
                    logits = model(x, f if use_f else None)
                    loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt), scaler.update(), sched.step()
            tot += loss.item() * len(y)
            seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            if i % 50 == 0:
                log(f"  epoch {ep + 1}/{t['epochs']}  step {i}/{steps}  loss {tot / seen:.4f}  "
                    f"acc {correct / seen:.4f}  lr {sched.get_last_lr()[0]:.2e}")
        idx, prob, _ = predict(model, dl["val"], dev, use_f, t["amp"])
        vm = classification_metrics(records(frames["val"], idx, prob, model_name), n_boot=0, seed=cfg["seed"])
        row = {"epoch": ep + 1, "train_loss": round(tot / max(seen, 1), 4),
               **({"train_map_loss": round(map_tot / max(seen, 1), 4)} if use_maps else {}), "train_acc": round(correct / max(seen, 1), 4),
               "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"], "val_fire_recall": vm["fire_recall"],
               "val_false_alarm": vm["false_alarm_rate"], "minutes": round((time.time() - t0) / 60, 2),
               "lr": sched.get_last_lr()[0]}
        hist.append(row)
        log(f"epoch {ep + 1}: " + json.dumps(row))
        pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
        if row["val_macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": row["val_macro_f1"], "epoch": ep + 1}
            torch.save({"model": model.state_dict(), "cfg": cfg, "feat_norm": norm, "best": best}, best_ck)
            log(f"  new best (val macro-F1 {best['val_macro_f1']:.4f}) -> {best_ck}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "history": hist, "best": best}, last_ck)

    del dl["train"]
    gc.collect()
    if best["epoch"] > 0:
        model.load_state_dict(torch.load(best_ck, map_location=dev, weights_only=False)["model"])
    log(f"Evaluating the best epoch ({best['epoch']}) on {', '.join(EVAL_SPLITS)}")
    rec, metrics, rows, look_rows, speed = [], {}, [], [], {}
    (run_dir / "figures").mkdir(exist_ok=True), (run_dir / "tables").mkdir(exist_ok=True)
    from src.eval.plots import confusion_figure, lookalike_figure
    for s in EVAL_SPLITS:
        idx, prob, ms = predict(model, dl[s], dev, use_f, t["amp"])
        if s == "val" and "val" in dl:
            del dl["val"]
            gc.collect()
        r = records(frames[s], idx, prob, model_name)
        rec.append(r)
        m = classification_metrics(r, n_boot=0 if a.smoke else cfg["deep"]["n_boot"], seed=cfg["seed"])
        metrics[s] = m
        rows.append(summary_row(model_name, s, m))
        speed[s] = {"ms_per_image": round(ms, 3), "fps": round(1000 / max(ms, 1e-6), 1), "batch": t["batch_size"]}
        if "confusion_matrix" in m:
            confusion_figure(m["confusion_matrix"]["rows_true_cols_pred"], CLASSES, f"{model_name} - {s}",
                             run_dir / "figures" / f"cm_{model_name}_{s}")
        for k, v in m.get("false_alarm_by_type", {}).items():
            look_rows.append({"model": model_name, "lookalike_type": k, **v})
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
    if use_maps:
        map_gallery(model, frames["test_in"], cfg, norm, dev, run_dir / "figures" / "fig_learned_maps")
    h = pd.DataFrame(hist)
    if len(h) > 1:
        import matplotlib.pyplot as plt
        from src.viz import style
        fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
        ax[0].plot(h["epoch"], h["train_loss"], "o-", color=style.OKABE_ITO[0])
        ax[0].set_xlabel("epoch"), ax[0].set_ylabel("train loss")
        ax[1].plot(h["epoch"], h["val_macro_f1"], "o-", color=style.OKABE_ITO[1], label="val macro-F1")
        ax[1].plot(h["epoch"], h["val_fire_recall"], "s--", color=style.OKABE_ITO[2], label="val fire recall")
        ax[1].set_xlabel("epoch"), ax[1].legend(fontsize=7, frameon=False)
        style.save(fig, run_dir / "figures" / "fig_training_curve", h)
        plt.close(fig)
    summary = {"model": model_name, "seed": cfg["seed"], "arch": arch, "kind": kind, "params_m": round(params, 3),
               "best_epoch": best["epoch"], "best_val_macro_f1": best["val_macro_f1"],
               "epochs_run": len(hist), "total_minutes": round(float(h["minutes"].sum()), 1) if len(h) else 0.0,
               "speed": speed, "table": summ.to_dict("records"), "checkpoints": str(ck_dir)}
    atomic_write_json(run_dir / "summary.json", summary)
    write_meta(run_dir, cfg, data_files=[fpath], extra={"smoke": a.smoke, "feature_norm": norm})
    atomic_write_json(run_dir / "results.json", {"metrics": metrics, "summary": summary, "history": hist,
                                                 "records_file": "records.csv",
                                                 "meta": json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))})
    if not a.smoke:
        best_row = summ[summ["split"] == "test_in"]
        note = f"{model_name} seed {cfg['seed']}: best epoch {best['epoch']}, val macro-F1 {best['val_macro_f1']}"
        if len(best_row):
            note += f", test_in macro-F1 {best_row['macro_f1'].iloc[0]}"
        append_ledger(stage, run_dir, note)
    log("Done.")


if __name__ == "__main__":
    main()
