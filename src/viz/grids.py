"""Labelled image grids."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def group_grid(groups: dict, out_png: Path, root: Path, per_row: int = 8, thumb: int = 160,
               label_w: int = 230, title: str = "", seed: int = 42) -> Path:
    """One row per group (e.g. class or look-alike type): label + count on the left, `per_row` seeded samples."""
    rng = np.random.default_rng(seed)
    pad, title_h = 4, (40 if title else 0)
    names = list(groups)
    W = label_w + per_row * (thumb + pad) + pad
    H = title_h + len(names) * (thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    f_title, f_lab, f_cnt = _font(20), _font(17), _font(14)
    if title:
        draw.text((pad + 6, 10), title, fill="#111827", font=f_title)
    shown = {}
    for r, name in enumerate(names):
        items = list(groups[name])
        pick = [items[i] for i in rng.choice(len(items), size=min(per_row, len(items)), replace=False)] if items else []
        shown[name] = pick
        y = title_h + pad + r * (thumb + pad)
        draw.text((pad + 6, y + thumb // 2 - 18), name, fill="#111827", font=f_lab)
        draw.text((pad + 6, y + thumb // 2 + 4), f"n = {len(items)}", fill="#4b5563", font=f_cnt)
        for c, rel in enumerate(pick):
            with Image.open(root / rel) as im:
                tile = ImageOps.fit(im.convert("RGB"), (thumb, thumb))
            canvas.paste(tile, (label_w + pad + c * (thumb + pad), y))
    out_png = Path(out_png)
    canvas.save(out_png)
    out_png.with_suffix(".json").write_text(json.dumps({"seed": seed, "shown": shown}, indent=2), encoding="utf-8")
    return out_png
