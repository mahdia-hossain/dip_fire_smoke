"""Resumable per-image processing in worker processes."""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import pandas as pd

from src.common.runs import read_partial


def run_records(run_dir: Path, items: list, fn, cfg: dict, workers: int, log, every: int = 500,
                chunk: int = 2000) -> pd.DataFrame:
    partial_path = run_dir / "records.partial.jsonl"
    done = {r["id"] for r in read_partial(partial_path)}
    todo = [it for it in items if it["id"] not in done]
    log(f"{len(items)} images, {len(done)} already done, {len(todo)} to do ({workers} worker processes)")
    t0, n = time.time(), 0
    work = partial(fn, cfg=cfg)
    with ProcessPoolExecutor(max_workers=workers) as ex, open(partial_path, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), chunk):
            for fut in as_completed([ex.submit(work, it) for it in todo[s:s + chunk]]):
                f.write(json.dumps(fut.result()) + "\n")
                n += 1
                if n % every == 0 or n == len(todo):
                    f.flush()
                    rate = (time.time() - t0) / n
                    log(f"...{n}/{len(todo)}  {rate * 1000:.0f} ms/image  eta {rate * (len(todo) - n) / 60:.1f} min")
    rec = pd.DataFrame(read_partial(partial_path))
    return rec.drop_duplicates("id", keep="last").sort_values("id").reset_index(drop=True)
