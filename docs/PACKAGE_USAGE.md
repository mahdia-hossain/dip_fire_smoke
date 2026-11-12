# Using the firesmoke package

`firesmoke` runs the final pipeline on any photo: resize → noise check → (if noisy) the right filter + CLAHE →
MobileViT-XS → fire / smoke / normal, and keeps every step.

## Install

```bash
pip install git+https://github.com/mahdia-hossain/dip_fire_smoke
```

Download the model file `firesmoke_CA.pt` from the repository's **Releases** page.
Install the CUDA build of PyTorch from pytorch.org first if you want to use an NVIDIA GPU.

## Python

```python
from firesmoke import FireSmokeDetector

model = FireSmokeDetector("firesmoke_CA.pt")
result = model.predict("kitchen.jpg")

print(result.label, result.confidence)   # e.g. fire 0.96
print(result.decision, result.filter)    # e.g. clean none
result.explain()                         # every step in sentences
result.show_steps()                      # one figure with the 6 steps
result.save("out/")                      # *_steps.png, *_result.png, *_explanation.md, *_result.json
```

## Command line

```bash
firesmoke predict kitchen.jpg --weights firesmoke_CA.pt --steps
firesmoke predict photos_folder --weights firesmoke_CA.pt --out results
```

## Google Colab

Open `demo.ipynb` in Colab (File → Open notebook → GitHub → paste the repository link) and run the cells.

## The 6 steps

1. Resize to 256 × 256 (area interpolation).
2. Noise check: noise sigma from a 3 × 3 mask (Immerkær, 1996), compared with a threshold chosen on validation.
3. Isolated point detection: share of lonely bright/dark pixels (used by the double-adaptive model).
4. Decision and filter: clean → none; noisy → median 3 × 3 (single adaptive), or by noise type: dots → median,
   light grain → bilateral, heavy grain → Gaussian 7 × 7 (double adaptive).
5. CLAHE on the L channel of LAB for noisy photos; clean photos are unchanged.
6. MobileViT-XS gives the class and confidence; fire (red) or smoke (blue) areas are highlighted with the DIP
   colour masks (approximate).
