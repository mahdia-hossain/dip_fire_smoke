"""
E01: local browser app for labelling look-alike types.

    python run.py lookalike-label
"""
import csv
import json
import mimetypes
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.config import DATA_ROOT, MANIFEST_DIR, PROJECT_ROOT

PORT = 8770
TARGET_PER_TYPE = 50
TYPES = {"1": ("sunset_sunrise", "Sunset / sunrise"), "2": ("lamp_streetlight", "Lamp / street light"),
         "3": ("vehicle_light", "Vehicle / emergency light"), "4": ("reflection_glow", "Reflection / glow / screen"),
         "5": ("cloud_fog_haze", "Cloud / fog / haze")}
OTHER = {"r": ("reject_has_fire_smoke", "Real fire/smoke (reject)"), "s": ("skip_unsure", "Skip / unsure")}
LABEL_DIR = DATA_ROOT / "lookalike_labels"
LABELS_CSV = LABEL_DIR / "labels.csv"

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Look-alike labelling</title>
<style>
 body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#111827;color:#f9fafb}
 header{display:flex;gap:18px;align-items:center;padding:10px 18px;background:#1f2937;flex-wrap:wrap}
 #prog{font-weight:600} .cnt{padding:3px 9px;border-radius:12px;background:#374151;font-size:13px}
 .ok{background:#065f46} main{display:flex;height:calc(100vh - 56px)}
 #imgbox{flex:1;display:flex;align-items:center;justify-content:center;padding:12px}
 #img{max-width:100%;max-height:100%;object-fit:contain;border:2px solid #374151}
 aside{width:330px;padding:14px;background:#1f2937;overflow:auto}
 button{display:block;width:100%;margin:6px 0;padding:11px;font-size:15px;text-align:left;border:0;
   border-radius:8px;background:#374151;color:#f9fafb;cursor:pointer} button:hover{background:#4b5563}
 .k{display:inline-block;width:26px;font-weight:700;color:#fbbf24} .rej{background:#7f1d1d} .skip{background:#3f3f46}
 #info{font-size:12px;color:#9ca3af;margin-top:12px;word-break:break-all} #msg{color:#34d399;margin-top:8px}
 .help{font-size:13px;color:#d1d5db;line-height:1.5;margin-top:14px}
</style></head><body>
<header><span id="prog">loading…</span><span id="counts"></span></header>
<main><div id="imgbox"><img id="img" alt=""></div>
<aside>
 <div id="btns"></div>
 <button class="skip" onclick="undo()"><span class="k">Z</span>Undo last</button>
 <div id="msg"></div><div id="info"></div>
 <div class="help"><b>What counts as what</b><br>
 1: sun, orange sky at sunrise/sunset<br>2: street lamps, indoor lamps, bulbs, spotlights<br>
 3: headlights, taillights, fire-truck / police lights<br>4: reflections (water, glass, wet road), glow, neon, screens, signs<br>
 5: clouds, fog, haze, steam<br><b>R</b>: you can clearly see REAL fire or smoke<br><b>S</b>: not sure</div>
</aside></main>
<script>
const KEYS = __KEYS__; let cur = null;
const btns = document.getElementById('btns');
for (const [k,[code,name]] of Object.entries(KEYS)) {
  const b = document.createElement('button'); b.innerHTML = `<span class="k">${k.toUpperCase()}</span>${name}`;
  if (k==='r') b.className='rej'; if (k==='s') b.className='skip'; b.onclick=()=>answer(k); btns.appendChild(b);
}
function show(d){
  document.getElementById('prog').textContent = `${d.done} labelled of ${d.total}  ·  ${d.total-d.done} left`;
  document.getElementById('counts').innerHTML = Object.entries(d.counts).map(([n,c]) =>
     `<span class="cnt ${d.types.includes(n)&&c>=d.target?'ok':''}">${n}: ${c}${d.types.includes(n)?' / '+d.target:''}</span>`).join(' ');
  if (!d.item){ document.getElementById('img').src=''; document.getElementById('msg').textContent='All images labelled. Run: python run.py lookalike-freeze'; cur=null; return; }
  cur = d.item; document.getElementById('img').src = '/img?i=' + d.item.i;
  document.getElementById('info').textContent = `#${d.item.i} · ${d.item.source} · ${d.item.path}`;
  if (d.all_done) document.getElementById('msg').textContent = 'Every type has reached the target. You can stop and run lookalike-freeze (or keep going).';
}
async function load(){ show(await (await fetch('/api/next')).json()); }
async function answer(k){ if(!cur) return; const r = await fetch('/api/label',{method:'POST',body:JSON.stringify({i:cur.i,key:k})}); show(await r.json()); }
async function undo(){ show(await (await fetch('/api/undo',{method:'POST'})).json()); }
document.addEventListener('keydown', e => { const k=e.key.toLowerCase();
  if (KEYS[k]) answer(k); else if (k==='z'||k==='backspace'){ e.preventDefault(); undo(); } });
load();
</script></body></html>"""


def load_items() -> pd.DataFrame:
    man = pd.read_csv(MANIFEST_DIR / "manifest.csv")
    look = man[man["split"] == "test_lookalike"][["path", "source"]].reset_index(drop=True)
    if look.empty:
        raise SystemExit("No test_lookalike images in data/manifest/manifest.csv. Run `python run.py manifest` first.")
    order = np.random.default_rng(42).permutation(len(look))
    return look.iloc[order].reset_index(drop=True)


def read_labels() -> dict:
    if not LABELS_CSV.exists():
        return {}
    rows = list(csv.DictReader(open(LABELS_CSV, encoding="utf-8")))
    return {r["path"]: r["label"] for r in rows}


class State:
    def __init__(self):
        self.items = load_items()
        LABEL_DIR.mkdir(parents=True, exist_ok=True)
        if not LABELS_CSV.exists():
            LABELS_CSV.write_text("path,label,time\n", encoding="utf-8")

    def payload(self) -> dict:
        labels = read_labels()
        names = [v[0] for v in TYPES.values()] + [v[0] for v in OTHER.values()]
        counts = {n: sum(1 for x in labels.values() if x == n) for n in names}
        types = [v[0] for v in TYPES.values()]
        todo = self.items[~self.items["path"].isin(labels)]
        item = None if todo.empty else {"i": int(todo.index[0]), **todo.iloc[0].to_dict()}
        return {"done": len(labels), "total": len(self.items), "counts": counts, "types": types,
                "target": TARGET_PER_TYPE, "all_done": all(counts[t] >= TARGET_PER_TYPE for t in types), "item": item}

    def label(self, i: int, key: str) -> None:
        code = {**TYPES, **OTHER}[key][0]
        with open(LABELS_CSV, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([self.items.at[i, "path"], code, time.strftime("%Y-%m-%d %H:%M:%S")])

    def undo(self) -> None:
        lines = LABELS_CSV.read_text(encoding="utf-8").splitlines()
        if len(lines) > 1:
            tmp = LABELS_CSV.with_suffix(".tmp")
            tmp.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            tmp.replace(LABELS_CSV)


STATE = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path == "/":
            keys = {k: list(v) for k, v in {**TYPES, **OTHER}.items()}
            return self._send(PAGE.replace("__KEYS__", json.dumps(keys)).encode(), "text/html; charset=utf-8")
        if self.path == "/api/next":
            return self._json(STATE.payload())
        if self.path.startswith("/img?i="):
            try:
                i = int(self.path.split("=", 1)[1])
                f = PROJECT_ROOT / STATE.items.at[i, "path"]
                return self._send(f.read_bytes(), mimetypes.guess_type(f.name)[0] or "image/jpeg")
            except (ValueError, KeyError, OSError):
                return self._send(b"not found", "text/plain", 404)
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        if self.path == "/api/label":
            d = json.loads(body or b"{}")
            if d.get("key") in {**TYPES, **OTHER} and int(d.get("i", -1)) in STATE.items.index:
                STATE.label(int(d["i"]), d["key"])
            return self._json(STATE.payload())
        if self.path == "/api/undo":
            STATE.undo()
            return self._json(STATE.payload())
        self._send(b"not found", "text/plain", 404)

    def log_message(self, *args):
        pass


def main(open_browser: bool = True) -> None:
    global STATE
    STATE = State()
    p = STATE.payload()
    print(f"{p['total']} look-alike images, {p['done']} already labelled. Answers saved to {LABELS_CSV}")
    print(f"Open http://127.0.0.1:{PORT}  (stop with Ctrl + C)")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Your answers are saved. Run again any time to continue.")


if __name__ == "__main__":
    main(open_browser="--no-browser" not in sys.argv)
