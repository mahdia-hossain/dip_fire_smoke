"""
E00: build the dataset manifest (labels, duplicate removal, splits).

    python run.py manifest
"""
import hashlib
import io
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imagehash
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.common.cfg import load_config, validate
from src.common.runs import append_ledger, new_run_dir, setup_logging, write_meta
from src.config import CONFIG_DIR, MANIFEST_DIR, PROJECT_ROOT, RESULTS_ROOT

ROOT = PROJECT_ROOT
CFG = load_config(CONFIG_DIR / "base.yaml")
validate(CFG, ["seed", "paths.raw_dfire", "paths.raw_fireandsmoke", "paths.raw_deepquest", "manifest.phash_threshold",
               "manifest.val_frac", "manifest.keywords", "manifest.dfire_known_box_totals"], "configs/base.yaml")
MCFG = CFG["manifest"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRIORITY = {"train": 0, "val": 1, "test_in": 2, "test_lookalike": 3, "test_ext": 4}
LABELS = ["fire", "smoke", "normal"]

OUT_BASE = RESULTS_ROOT / "s02_manifest"
CACHE = OUT_BASE / "hash_cache.csv"
RUN_DIR = new_run_dir("s02_manifest")
log = setup_logging(RUN_DIR, "manifest")


def rel(p: Path) -> str:
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


def list_images(folder: Path) -> list:
    if not folder.exists():
        log(f"!! Missing folder: {folder}")
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def split_from_parts(parts) -> str:
    """Split from FOLDER names only (file names are ignored, e.g. 'test_123.jpg' inside train/)."""
    low = [x.lower() for x in parts if Path(x).suffix.lower() not in IMG_EXT]
    if "test" in low or any(x.startswith("test") for x in low):
        return "test_in"
    if any(x in ("val", "valid", "validation") for x in low):
        return "val"
    return "train"


def read_yolo_ids(lp: Path):
    """Return the class ids of a YOLO label file, or None if the file is missing."""
    if not lp.exists():
        return None
    tokens = lp.read_text(encoding="utf-8").split()
    return np.array(tokens, dtype=float).reshape(-1, 5)[:, 0].astype(int) if tokens else np.array([], dtype=int)


def load_dfire_split_lists(root: Path) -> dict:
    """Official D-Fire split files (dfire_train.txt, dfire_test.txt) in data/raw/dfire/splits/."""
    split_dir = root / "splits"
    names = {"train": "train", "test": "test_in", "val": "val", "valid": "val"}
    files = [f for f in (sorted(split_dir.glob("*.txt")) if split_dir.exists() else [])
             if "video" not in f.stem.lower() and any(k in f.stem.lower() for k in names)]
    keyed = [(f, next(v for k, v in names.items() if k in f.stem.lower())) for f in files]
    keyed.sort(key=lambda fk: PRIORITY[fk[1]])
    out = {}
    for f, key in keyed:
        stems = [Path(ln.strip().replace("\\", "/")).stem
                 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        overlap = len(set(stems) & set(out))
        out.update(dict.fromkeys(stems, key))
        log(f"D-Fire split file {f.name}: {len(stems)} names -> {key}"
            + (f" ({overlap} also in an earlier list; {key} wins)" if overlap else ""))
    return out


def scan_dfire(root: Path) -> tuple:
    imgs = list_images(root)
    if not imgs:
        return pd.DataFrame(), {}
    split_map = load_dfire_split_lists(root)
    lbl_paths = [p.parent.parent / "labels" / (p.stem + ".txt") if p.parent.name.lower() == "images"
                 else p.with_suffix(".txt") for p in imgs]
    ids = [read_yolo_ids(lp) for lp in tqdm(lbl_paths, desc="D-Fire labels")]
    present = [i for i in ids if i is not None]
    counts = np.bincount(np.concatenate(present), minlength=2) if present else np.zeros(2, int)

    known = MCFG["dfire_known_box_totals"]
    cost_a = abs(counts[0] - known["smoke"]) + abs(counts[1] - known["fire"])
    cost_b = abs(counts[0] - known["fire"]) + abs(counts[1] - known["smoke"])
    id_fire, id_smoke = (1, 0) if cost_a <= cost_b else (0, 1)
    rel_err = {
        "fire": float(abs(counts[id_fire] - known["fire"]) / known["fire"]),
        "smoke": float(abs(counts[id_smoke] - known["smoke"]) / known["smoke"]),
    }
    if max(rel_err.values()) > 0.10:
        log(f"!! WARNING: D-Fire box counts {counts.tolist()} differ >10% from README totals {known}. "
            f"Check the download before trusting labels.")
    mapping = {"id_fire": int(id_fire), "id_smoke": int(id_smoke),
               "box_counts_by_id": counts.tolist(), "relative_error_vs_readme": rel_err}

    has_fire = np.array([(i is not None) and bool((i == id_fire).any()) for i in ids])
    has_smoke = np.array([(i is not None) and bool((i == id_smoke).any()) for i in ids])
    missing = np.array([i is None for i in ids])
    label = np.where(has_fire, "fire", np.where(has_smoke, "smoke", "normal"))
    label = np.where(missing, "unknown", label)
    orig = np.where(has_fire & has_smoke, "fire+smoke",
                    np.where(has_fire, "fire", np.where(has_smoke, "smoke", "none")))
    folder_split = [split_from_parts(p.relative_to(root).parts) for p in imgs]
    if split_map:
        split = [split_map.get(p.stem, "not_in_split_files") for p in imgs]
        n_unlisted = sum(s == "not_in_split_files" for s in split)
        n_disagree = sum(a != b for a, b in zip(split, folder_split) if a != "not_in_split_files")
        log(f"D-Fire split from official lists: {len(imgs) - n_unlisted} matched, {n_unlisted} not listed "
            f"(excluded), {n_disagree} differ from folder names (lists win)")
        mapping["split_source"] = "official split files"
        mapping["images_not_in_split_files"] = int(n_unlisted)
    else:
        split = folder_split
        mapping["split_source"] = "folder names (no split files found)"
    unlisted = np.array([s == "not_in_split_files" for s in split])
    df = pd.DataFrame({
        "path": [rel(p) for p in imgs],
        "source": "dfire",
        "orig_label": np.where(missing, "missing_label", orig),
        "label": label,
        "split": split,
        "exclude_reason": np.where(missing, "missing_label", np.where(unlisted, "not_in_split_files", "")),
        "folder": [f"{root.name}/" + p.parent.relative_to(root).as_posix() for p in imgs],
    })
    return df, mapping


def scan_fireandsmoke(root: Path) -> tuple:
    """Each part folder (part1, part2) has data.yaml with names, e.g. ['fire', 'other', 'smoke'],"""
    imgs = list_images(root)
    if not imgs:
        return pd.DataFrame(), {}
    parts = sorted({p.relative_to(root).parts[0] for p in imgs})
    names = {pt: yaml.safe_load((root / pt / "data.yaml").read_text(encoding="utf-8"))["names"] for pt in parts}
    ids_of = {pt: {n.lower(): i for i, n in enumerate(names[pt])} for pt in parts}
    for pt in parts:
        if not {"fire", "smoke", "other"} <= set(ids_of[pt]):
            log(f"!! {pt}/data.yaml names {names[pt]} do not contain fire/smoke/other - check the download")
    part = [p.relative_to(root).parts[0] for p in imgs]
    lbl_paths = [p.parent.parent / "labels" / (p.stem + ".txt") for p in imgs]
    ids = [read_yolo_ids(lp) for lp in tqdm(lbl_paths, desc="FireAndSmoke labels")]
    missing = np.array([i is None for i in ids])

    def has(cls):
        return np.array([(i is not None) and bool(np.isin(i, ids_of[pt].get(cls, -99)).any())
                         for i, pt in zip(ids, part)])
    has_fire, has_smoke, has_other = has("fire"), has("smoke"), has("other")
    label = np.where(has_fire, "fire", np.where(has_smoke, "smoke", "normal"))
    label = np.where(missing, "unknown", label)
    lookalike = has_other & ~has_fire & ~has_smoke & ~missing
    orig = np.where(has_fire & has_smoke, "fire+smoke", np.where(has_fire, "fire",
                    np.where(has_smoke, "smoke", np.where(has_other, "other_only", "none"))))
    split = np.array([split_from_parts(p.relative_to(root).parts) for p in imgs])
    to_lookalike = (split == "test_in") | ((split == "val") & bool(MCFG.get("lookalike_test_from_val", False)))
    split = np.where(to_lookalike & lookalike, "test_lookalike", split)
    df = pd.DataFrame({
        "path": [rel(p) for p in imgs],
        "source": "fireandsmoke",
        "orig_label": np.where(missing, "missing_label", orig),
        "label": label,
        "split": split,
        "exclude_reason": np.where(missing, "missing_label", ""),
        "folder": [f"{root.name}/" + p.parent.relative_to(root).as_posix() for p in imgs],
        "lookalike": lookalike,
    })
    info = {"class_names_per_part": names,
            "box_class_counts": {pt: np.bincount(np.concatenate([i for i, q in zip(ids, part) if q == pt and i is not None]
                                                                 or [np.array([], int)]), minlength=len(names[pt])).tolist()
                                 for pt in parts},
            "lookalike_images_total": int(lookalike.sum())}
    return df, info


def label_from_folder(rel_parts, kw) -> str:
    """Label from the parent folder name, then the grandparent. Normal is checked first (non_fire etc.)."""
    folders = list(rel_parts[:-1])
    candidates = folders[::-1][:2]
    if len(folders) >= 2:
        candidates = [c for i, c in zip([len(folders) - 1, len(folders) - 2], candidates) if i != 0]
    for part in candidates:
        low = part.lower()
        if any(k in low for k in kw["normal"]):
            return "normal"
        if any(k in low for k in kw["fire"]):
            return "fire"
        if any(k in low for k in kw["smoke"]):
            return "smoke"
    return "unknown"


def scan_folder_dataset(root: Path, source: str, force_split=None) -> pd.DataFrame:
    imgs = list_images(root)
    if not imgs:
        return pd.DataFrame()
    kw = MCFG["keywords"]
    overrides = MCFG.get("folder_label_overrides") or {}
    rel_parts = [p.relative_to(root).parts for p in imgs]
    folder_key = [f"{root.name}/" + "/".join(rp[:-1]) for rp in rel_parts]
    labels = [overrides.get(fk, label_from_folder(rp, kw)) for fk, rp in zip(folder_key, rel_parts)]
    labels = np.array(labels)
    reason = np.where(labels == "skip", "manual_skip", np.where(labels == "unknown", "unknown_label", ""))
    return pd.DataFrame({
        "path": [rel(p) for p in imgs],
        "source": source,
        "orig_label": [rp[-2] if len(rp) > 1 else "" for rp in rel_parts],
        "label": labels,
        "split": force_split or [split_from_parts(rp) for rp in rel_parts],
        "exclude_reason": reason,
        "folder": folder_key,
    })


def hash_one(rel_path: str):
    try:
        data = (ROOT / rel_path).read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            w, h = im.size
            ph = str(imagehash.phash(im.convert("RGB")))
        return rel_path, md5, ph, w, h, ""
    except Exception as e:
        return rel_path, "", "", 0, 0, repr(e)[:200]


def hash_all(paths: list) -> pd.DataFrame:
    cols = ["path", "md5", "phash", "width", "height", "error"]
    cache = pd.read_csv(CACHE, dtype=str, keep_default_na=False) if CACHE.exists() else pd.DataFrame(columns=cols)
    todo = sorted(set(paths) - set(cache["path"]))
    log(f"Hashing: {len(paths)} images, {len(paths) - len(todo)} cached, {len(todo)} to do")
    chunk = 2000
    with ThreadPoolExecutor(MCFG["hash_workers"]) as ex:
        for s in tqdm(range(0, len(todo), chunk), desc="hash chunks"):
            rows = list(ex.map(hash_one, todo[s:s + chunk]))
            pd.DataFrame(rows, columns=cols).to_csv(CACHE, mode="a", header=not CACHE.exists(), index=False)
    cache = pd.read_csv(CACHE, dtype=str, keep_default_na=False).drop_duplicates("path", keep="last")
    cache[["width", "height"]] = cache[["width", "height"]].astype(int)
    return cache


def near_duplicates(phash_hex: pd.Series, priority: np.ndarray, thr: int):
    """For each image, find a near-identical image in a MORE IMPORTANT split (test_ext > test_lookalike >"""
    h = np.array([int(x, 16) for x in phash_hex], dtype=np.uint64)
    n = len(h)
    drop = np.zeros(n, bool)
    dup_of = np.full(n, -1)
    step = 256
    for s in tqdm(range(0, n, step), desc="near-dup search"):
        e = min(s + step, n)
        dist = np.bitwise_count(h[s:e, None] ^ h[None, :])
        m = (dist <= thr) & (priority[None, :] > priority[s:e, None])
        drop[s:e] = m.any(axis=1)
        dup_of[s:e] = np.where(drop[s:e], m.argmax(axis=1), -1)
    return drop, dup_of


def main():
    t0 = time.time()
    P = {k: ROOT / v for k, v in CFG["paths"].items()}
    log(f"Run folder: {RUN_DIR}")
    write_meta(RUN_DIR, CFG, data_files=sorted((P["raw_dfire"] / "splits").glob("*.txt"))
               + sorted(P["raw_fireandsmoke"].glob("*/data.yaml")))

    dfire, mapping = scan_dfire(P["raw_dfire"])
    fas, fas_info = scan_fireandsmoke(P["raw_fireandsmoke"])
    deepq = scan_folder_dataset(P["raw_deepquest"], "deepquest", force_split="test_ext")
    df = pd.concat([dfire, fas, deepq], ignore_index=True)
    df["lookalike"] = df["lookalike"].fillna(False).astype(bool) if "lookalike" in df else False
    if df.empty:
        log("!! No images found. Check the folders in configs/config.yaml -> paths.")
        sys.exit(1)
    df["folder"] = df["folder"].fillna(df["path"].str.rsplit("/", n=1).str[0])
    log(f"Scanned: {df['source'].value_counts().to_dict()}")

    fs = df.groupby(["source", "folder", "label", "split"]).size().rename("n_images").reset_index()
    fs.to_csv(RUN_DIR / "folder_summary.csv", index=False)
    log("Folder -> label check (look at this carefully):\n" + fs.to_string(index=False))

    hashes = hash_all(df["path"].tolist())
    df = df.merge(hashes, on="path", how="left")
    df.loc[(df["error"] != "") & (df["exclude_reason"] == ""), "exclude_reason"] = "corrupt"

    kept = df["exclude_reason"] == ""
    for src in df.loc[kept, "source"].unique():
        has_val = ((df["source"] == src) & kept & (df["split"] == "val")).any()
        pool = df.index[(df["source"] == src) & kept & (df["split"] == "train")]
        if not has_val and len(pool) > 20:
            _, val_idx = train_test_split(pool, test_size=MCFG["val_frac"], random_state=CFG["seed"],
                                          stratify=df.loc[pool, "label"])
            df.loc[val_idx, "split"] = "val"
            log(f"Carved {len(val_idx)} val images from {src} train (stratified, seed {CFG['seed']})")

    ok = df["exclude_reason"] == ""
    use = df[ok].copy()
    use["priority"] = use["split"].map(PRIORITY).to_numpy()
    use["order"] = np.arange(len(use))
    use = use.sort_values(["priority", "order"], ascending=[False, True])
    exact = use.duplicated("md5", keep="first")
    first_of_md5 = use.drop_duplicates("md5").set_index("md5")["path"]
    df.loc[use.index[exact], "exclude_reason"] = "exact_duplicate"
    df.loc[use.index[exact], "duplicate_of"] = use.loc[exact, "md5"].map(first_of_md5).to_numpy()

    rest = use[~exact].sort_values("order")
    drop, dup_of = near_duplicates(rest["phash"], rest["priority"].to_numpy(), MCFG["phash_threshold"])
    near_idx = rest.index[drop]
    df.loc[near_idx, "exclude_reason"] = "near_duplicate"
    df.loc[near_idx, "duplicate_of"] = rest["path"].to_numpy()[dup_of[drop]]

    dups = df[df["exclude_reason"].isin(["exact_duplicate", "near_duplicate"])].copy()
    lab = df.set_index("path")["label"]
    split = df.set_index("path")["split"]
    dups["kept_label"] = dups["duplicate_of"].map(lab)
    dups["kept_split"] = dups["duplicate_of"].map(split)
    dups["label_conflict"] = dups["label"] != dups["kept_label"]
    dups[["path", "source", "split", "label", "exclude_reason", "duplicate_of", "kept_split", "kept_label",
          "label_conflict"]].to_csv(RUN_DIR / "duplicates.csv", index=False)
    dup_table = pd.crosstab([dups["exclude_reason"], dups["source"], dups["split"]], dups["kept_split"]) \
        if len(dups) else pd.DataFrame()
    dup_table.to_csv(RUN_DIR / "duplicates_by_split.csv")
    log("Removed duplicates (rows = removed image, columns = split of the copy that was kept):\n"
        + (dup_table.to_string() if len(dups) else "none"))

    kept = df["exclude_reason"] == ""

    df["use_for_training"] = kept & df["split"].isin(["train", "val"])

    df.to_csv(RUN_DIR / "manifest_full.csv", index=False)
    final = df[df["exclude_reason"] == ""][
        ["path", "source", "orig_label", "label", "split", "lookalike", "width", "height", "md5", "phash",
         "use_for_training"]]
    final.to_csv(RUN_DIR / "manifest.csv", index=False)
    counts = final.pivot_table(index=["source", "split"], columns="label", values="path", aggfunc="count",
                               fill_value=0).reindex(columns=LABELS, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(RUN_DIR / "counts.csv")
    by_split = final.pivot_table(index="split", columns="label", values="path", aggfunc="count",
                                 fill_value=0).reindex(columns=LABELS, fill_value=0)
    by_split.to_csv(RUN_DIR / "counts_by_split.csv")

    summary = {
        "run_dir": rel(RUN_DIR),
        "seconds": round(time.time() - t0, 1),
        "dfire_class_mapping": mapping,
        "fireandsmoke_info": fas_info,
        "lookalike_in_train_val": int((final["lookalike"] & final["split"].isin(["train", "val"])).sum()),
        "lookalike_test_images": int((final["split"] == "test_lookalike").sum()),
        "scanned_per_source": df["source"].value_counts().to_dict(),
        "excluded_by_reason": df.loc[df["exclude_reason"] != "", "exclude_reason"].value_counts().to_dict(),
        "duplicate_label_conflicts": int(dups["label_conflict"].sum()),
        "duplicates_removed_from_train_val_because_of_test": int(
            (dups["split"].isin(["train", "val"]) & dups["kept_split"].isin(["test_in", "test_ext"])).sum()),
        "final_images": int(len(final)),
        "final_counts_by_split": by_split.to_dict(orient="index"),
        "settings": MCFG,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    mdir = MANIFEST_DIR
    mdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN_DIR / "manifest.csv", mdir / "manifest.csv")
    shutil.copy2(RUN_DIR / "summary.json", mdir / "manifest_summary.json")

    log("Counts per source/split:\n" + counts.to_string())
    log("Counts per split:\n" + by_split.to_string())
    log(f"Excluded: {summary['excluded_by_reason']}")
    log(f"D-Fire mapping: {mapping}")
    log(f"FireAndSmoke: {fas_info}")
    log(f"Look-alike images: {summary['lookalike_in_train_val']} in train/val (labelled normal), "
        f"{summary['lookalike_test_images']} in the look-alike test set")
    log(f"Done in {summary['seconds']} s. Official manifest -> {rel(mdir / 'manifest.csv')}")
    append_ledger("s02_manifest", RUN_DIR, f"E00 manifest: {summary['final_images']} images, "
                  f"{sum(summary['excluded_by_reason'].values())} excluded")


if __name__ == "__main__":
    main()
