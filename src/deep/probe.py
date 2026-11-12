"""
E10a / E10b: probes on trained deep models (prediction only, no training).

    python run.py probe --mode robust  [--models C CR D E EM CM EMD] [--seed 42] [--smoke]
    python run.py probe --mode ablate  [--models D E EM EMD]          [--seed 42] [--smoke]

robust: test_ext images are darkened, blurred, noised or JPEG-compressed BEFORE the DIP preprocessing; DIP features
        are recomputed on the damaged image. Shows how much each model (and the DIP preprocessing) helps under damage.
ablate: one DIP hint group (colour, brightness, texture, haze, edges) is switched off (set to its training mean)
        at test time. Shows which DIP features the model relies on.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.runs import append_ledger, atomic_write_json, new_or_resume_run, setup_logging, write_meta
from src.config import CONFIG_DIR, PROJECT_ROOT, RESULTS_ROOT
from src.common.cfg import load_config
from src.deep.data import CLASSES, FireSmokeDataset, photometric, worker_init
from src.deep.models import build_model, uses_features
from src.dip.features import FEATURE_GROUPS, FEATURE_NAMES, extract
from src.dip.preprocess import load_rgb, preprocess
from src.dip.segmentation import segment
from src.eval.metrics import classification_metrics

STAGES = {"B": "e06_mobilenetv3", "C": "e07_mobilevit", "D": "e08_mobilevit_concat", "E": "e09_mobilevit_diptokens",
          "EM": "e09b_mobilevit_diptokens_maps", "CM": "e09c_mobilevit_maps", "CR": "e07c_mobilevit_rawinput",
          "EMD": "e09d_mobilevit_diptokens_dipmaps", "CA": "e07d_mobilevit_adaptive_noiseaug",
          "CRN": "e07e_mobilevit_raw_noiseaug", "CN": "e07f_mobilevit_dip_noiseaug",
          "CA2": "e07g_mobilevit_fulladaptive_noiseaug", "CA3": "e07h_mobilevit_noisetype_adaptive"}
CONDITIONS = ["clean", "dark", "blur", "jpeg", "noise5", "noise10", "noise15", "noise25", "saltpepper2", "saltpepper5"]


def corrupt(rgb: np.ndarray, kind: str, seed: int) -> np.ndarray:
    if kind == "dark":
        return (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(rgb, (7, 7), 0)
    if kind.startswith("saltpepper"):
        from src.dip.restore import salt_pepper
        return salt_pepper(rgb, float(kind[10:]) / 100, np.random.default_rng(seed))
    if kind.startswith("noise"):
        n = np.random.default_rng(seed).normal(0, float(kind[5:] or 15), rgb.shape)
        return np.clip(rgb.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 20])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return rgb


class DamagedDataset(Dataset):
    def __init__(self, df, cfg: dict, kind: str, mean, std):
        self.paths, self.kind, self.cfg = df["path"].tolist(), kind, cfg
        self.y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.photo = photometric(False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        raw = corrupt(load_rgb(PROJECT_ROOT / self.paths[i]), self.kind, i)
        p2 = preprocess(raw, self.cfg["preprocess"])
        d = extract(p2, segment(p2, self.cfg["segmentation"]), self.cfg["features"])
        f = np.clip((np.array([d[k] for k in FEATURE_NAMES], np.float32) - self.mean) / self.std, -5, 5)
        x = torch.from_numpy(np.ascontiguousarray(p2)).permute(2, 0, 1)
        return self.photo(x), torch.from_numpy(f.astype(np.float32)), int(self.y[i]), i


def find_run(model_id: str, seed: int, root: Path = RESULTS_ROOT) -> Path:
    for run in sorted((root / STAGES[model_id]).glob("run_*"), reverse=True):
        sj = run / "summary.json"
        if (run / "results.json").exists() and sj.exists() and json.loads(sj.read_text()).get("seed") == seed:
            return run
    for run in sorted((root / STAGES[model_id]).glob("run_*"), reverse=True):
        if (run / "results.json").exists() and seed == 42 and "seed" not in json.loads((run / "summary.json").read_text()):
            return run
    raise SystemExit(f"No finished run of model {model_id} with seed {seed} under {root / STAGES[model_id]}")


def load(run: Path, dev):
    s = json.loads((run / "summary.json").read_text())
    from src.app.infer import checkpoint_of
    ck = torch.load(checkpoint_of(run), map_location=dev, weights_only=False)
    c = ck["cfg"]
    m = build_model(c["model"]["kind"], c["model"]["arch"], pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), c, ck["feat_norm"], s["model"]


@torch.no_grad()
def run_loader(model, ds, dev, use_f: bool, workers: int):
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=workers, worker_init_fn=worker_init)
    idx, prob = [], []
    for x, f, y, i in dl:
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            out = model(x.to(dev), f.to(dev) if use_f else None)
        idx.append(i.numpy()), prob.append(torch.softmax(out.float(), 1).cpu().numpy())
    return np.concatenate(idx), np.concatenate(prob)


def to_records(frame, idx, prob, name):
    r = frame.iloc[idx][["path", "source", "split", "label", "lookalike_type"]].copy().reset_index(drop=True)
    r["model"], r["pred"] = name, np.array(CLASSES)[prob.argmax(1)]
    for k, c in enumerate(CLASSES):
        r[f"p_{c}"] = prob[:, k]
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["robust", "ablate"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    models = a.models or (["CR", "CA", "CRN", "CN", "CA3"] if a.mode == "robust" else ["D", "E", "EM", "EMD"])
    stage = "e10a_robustness" if a.mode == "robust" else "e10b_hint_ablation"
    cfg = {"mode": a.mode, "models": models, "seed": a.seed, "smoke": a.smoke}
    root = RESULTS_ROOT / "_smoke" if a.smoke else RESULTS_ROOT
    run_dir, _ = new_or_resume_run(stage, cfg, root=root)
    log = setup_logging(run_dir, stage)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = int(os.environ.get("DIP_WORKERS", "2"))
    write_meta(run_dir, cfg, extra={"workers": workers})
    rows, recs = [], []
    for mid in models:
        try:
            run = find_run(mid, a.seed, root)
        except SystemExit as ex:
            log(f"skip {mid}: {ex}")
            continue
        model, mcfg, norm, name = load(run, dev)
        use_f = uses_features(mcfg["model"]["kind"])
        feat = pd.read_csv(mcfg["features_file"], keep_default_na=False)
        feat = feat[feat["error"] == ""].reset_index(drop=True)
        feat[FEATURE_NAMES] = feat[FEATURE_NAMES].astype(np.float32)
        log(f"{name}: run {run.name}")
        if a.mode == "robust":
            frame = feat[feat["split"] == "test_ext"]
            if a.smoke:
                frame = frame.groupby("label").head(8)
            frame = frame.reset_index(drop=True)
            for kind in CONDITIONS:
                idx, prob = run_loader(model, DamagedDataset(frame, mcfg, kind, norm["mean"], norm["std"]),
                                       dev, use_f, workers)
                r = to_records(frame, idx, prob, name).assign(condition=kind)
                m = classification_metrics(r, n_boot=0, seed=a.seed)
                rows.append({"model": name, "condition": kind, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                             "fire_recall": m["fire_recall"], "smoke_recall": m["per_class"]["smoke"]["recall"],
                             "false_alarm_rate": m["false_alarm_rate"]})
                recs.append(r)
                log(f"  {kind:6s} macro-F1 {m['macro_f1']:.4f}  fire recall {m['fire_recall']:.4f}")
        else:
            if not use_f:
                log(f"skip {name}: it does not use DIP hints")
                continue
            groups = ["none", *FEATURE_GROUPS]
            for split in ["val", "test_in", "test_ext"]:
                frame = feat[feat["split"] == split]
                if a.smoke:
                    frame = frame.groupby("label").head(8)
                frame = frame.reset_index(drop=True)
                for g in groups:
                    ds = FireSmokeDataset(frame, mcfg["preprocess"], False, np.array(norm["mean"]), np.array(norm["std"]))
                    if g != "none":
                        cols = [FEATURE_NAMES.index(k) for k in FEATURE_GROUPS[g]]
                        ds.f[:, cols] = 0.0
                    idx, prob = run_loader(model, ds, dev, True, workers)
                    r = to_records(frame, idx, prob, name).assign(hint_group_off=g)
                    m = classification_metrics(r, n_boot=0, seed=a.seed)
                    rows.append({"model": name, "split": split, "hint_group_off": g, "macro_f1": m["macro_f1"],
                                 "fire_recall": m["fire_recall"], "smoke_recall": m["per_class"]["smoke"]["recall"],
                                 "false_alarm_rate": m["false_alarm_rate"]})
                    recs.append(r)
                    log(f"  {split:8s} off={g:10s} macro-F1 {m['macro_f1']:.4f}  smoke recall "
                        f"{m['per_class']['smoke']['recall']:.4f}")
        del model
        torch.cuda.empty_cache()
    if not rows:
        raise SystemExit("No model could be probed.")
    tab = pd.DataFrame(rows)
    (run_dir / "tables").mkdir(exist_ok=True), (run_dir / "figures").mkdir(exist_ok=True)
    pd.concat(recs, ignore_index=True).to_csv(run_dir / "records.csv", index=False)
    import matplotlib.pyplot as plt
    from src.viz import style
    if a.mode == "robust":
        piv = tab.pivot(index="model", columns="condition", values="macro_f1")[CONDITIONS]
        drop = piv.sub(piv["clean"], axis=0).drop(columns="clean").round(4)
        tab.to_csv(run_dir / "tables" / "robustness.csv", index=False)
        (run_dir / "tables" / "robustness.md").write_text(piv.round(4).to_markdown(), encoding="utf-8")
        drop.to_csv(run_dir / "tables" / "robustness_drop.csv")
        log("Macro-F1 on DeepQuestAI by condition:\n" + piv.round(4).to_string())
        fig, ax = plt.subplots(figsize=(5.0, 2.6))
        for m in piv.index:
            st = style.MODEL_STYLE.get(m, {})
            ax.plot(CONDITIONS, piv.loc[m], marker=st.get("marker", "o"), color=st.get("color"), label=m)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.set_ylabel("macro-F1 (DeepQuestAI)"), ax.legend(fontsize=6, frameon=False, ncol=2)
        style.save(fig, run_dir / "figures" / "fig_robustness", piv.reset_index())
    else:
        base = tab[tab["hint_group_off"] == "none"].set_index(["model", "split"])["macro_f1"]
        tab["macro_f1_drop"] = (tab["macro_f1"] - tab.set_index(["model", "split"]).index.map(base)).round(4)
        tab.to_csv(run_dir / "tables" / "hint_ablation.csv", index=False)
        piv = tab[tab["hint_group_off"] != "none"].pivot_table(index=["model", "split"], columns="hint_group_off",
                                                               values="macro_f1_drop")
        (run_dir / "tables" / "hint_ablation.md").write_text(piv.round(4).to_markdown(), encoding="utf-8")
        log("Macro-F1 change when one hint group is switched off:\n" + piv.round(4).to_string())
        sub = tab[(tab["split"] == "test_in") & (tab["hint_group_off"] != "none")]
        fig, ax = plt.subplots(figsize=(5.0, 2.6))
        ms = list(dict.fromkeys(sub["model"]))
        w = 0.8 / max(len(ms), 1)
        for j, m in enumerate(ms):
            d = sub[sub["model"] == m].set_index("hint_group_off").reindex(list(FEATURE_GROUPS))
            ax.bar(np.arange(len(FEATURE_GROUPS)) + (j - (len(ms) - 1) / 2) * w, d["macro_f1_drop"], w, label=m,
                   color=style.MODEL_STYLE.get(m, {}).get("color"))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(FEATURE_GROUPS)), list(FEATURE_GROUPS)), ax.set_ylabel("macro-F1 change (test_in)")
        ax.legend(fontsize=6, frameon=False)
        style.save(fig, run_dir / "figures" / "fig_hint_ablation", sub)
    plt.close("all")
    atomic_write_json(run_dir / "results.json", {"table": tab.to_dict("records"), "records_file": "records.csv"})
    if not a.smoke:
        append_ledger(stage, run_dir, f"{a.mode} probe on {', '.join(models)} (seed {a.seed})")
    log("Done.")


if __name__ == "__main__":
    main()
