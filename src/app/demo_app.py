"""
Local demo website: upload a photo, see every DIP step and the answer.

    python run.py demo          then open http://127.0.0.1:7860
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import gradio as gr

from src.app.infer import Pipeline, explain, find_final
from src.config import RESULTS_ROOT

SMOKE = "--smoke" in sys.argv
ROOT = RESULTS_ROOT / "_smoke" if SMOKE else RESULTS_ROOT
PIPE = Pipeline(find_final(ROOT))
SAVE = ROOT / "e11_demo" / time.strftime("session_%Y%m%d_%H%M%S")


def run(img):
    if img is None:
        return None, "Upload a photo first.", []
    r = PIPE.steps(img)
    SAVE.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%H%M%S")
    cv2.imwrite(str(SAVE / f"{stem}_input.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(SAVE / f"{stem}_result.png"), cv2.cvtColor(r["images"][-1][1], cv2.COLOR_RGB2BGR))
    (SAVE / f"{stem}_result.json").write_text(json.dumps({k: v for k, v in r.items() if k != "images"}, indent=2))
    return r["probs"], explain(r), [(im, t) for t, im in r["images"]]


with gr.Blocks(title="Fire / smoke detection: double-adaptive DIP + MobileViT-XS") as demo:
    gr.Markdown(f"## Fire and smoke detection\nModel: `{PIPE.name}` (validation macro-F1 {PIPE.val_f1}). "
                "Upload a photo to see each DIP step and the answer.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Input photo")
            btn = gr.Button("Analyse", variant="primary")
            probs = gr.Label(num_top_classes=3, label="Answer")
        with gr.Column(scale=2):
            text = gr.Markdown()
            gallery = gr.Gallery(label="Step by step", columns=3, height="auto")
    btn.click(run, inputs=inp, outputs=[probs, text, gallery])

if __name__ == "__main__":
    print(f"Saving every analysed photo to {SAVE}")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=not SMOKE, prevent_thread_lock=SMOKE)
    if SMOKE:
        import numpy as np
        p, t, g = run((np.random.default_rng(0).random((300, 400, 3)) * 255).astype("uint8"))
        print("smoke ok:", p, len(g))
        demo.close()
