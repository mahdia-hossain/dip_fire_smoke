# All runs and results (as of 12 Nov 2026)

**Final model: CA (adaptive DIP + noise training)**, chosen because it has the highest validation score of all models.

Every number below is copied from the saved result files in `outputs/` (seed 42, one run per model).
"TODO" means the run has not been reported yet.

Metric reminder:
- **macro-F1**: the average F1 score of the three classes (fire, smoke, normal). 1.0 = perfect.
- **val** is used to choose; **test_in**, **DeepQuestAI** (new source) and **look-alikes** are only for reporting.
- **False alarms**: share of normal photos called fire or smoke (lower is better).

---

## Part 1: Data and DIP steps

### E00: dataset manifest (s02_manifest)
- Train 24,047, val 7,325, test_in 6,810, DeepQuestAI 2,931, look-alikes 451 images. Copies removed between splits.
- **Why:** D-Fire and FireAndSmoke contain many near-identical video frames, so copies had to be removed between train and test.
- Without this, test scores would look better than they really are.
- DeepQuestAI was kept completely separate as a "new source" exam.

### E01: look-alike set (e01_lookalike)
- 451 normal photos hand-labelled into 5 types (fog 75, lamps 52, glow 51, sunsets 88, car lights 185); 19 rejected because they showed real fire or smoke.
- **Why:** these are the photos most likely to fool a fire detector.
- Sorting them into types lets us say which kind fools which model.
- The 19 rejected photos showed label mistakes in the original data.

### E02: preprocessing analysis (e02_preprocess)
- PSNR and SSIM between each step's input and output on 1,300 images; median best against salt-and-pepper dots, CLAHE on L keeps colours.
- **Why:** each step changes the photo a little, and PSNR/SSIM measure how much.
- The median filter removes dot noise without blurring as much as other filters on that noise type.
- CLAHE on the brightness channel boosts contrast without shifting the colours of the flames.

### E03 / E03b: segmentation and threshold check (e03_segmentation, e03b_seg_tune)
- Fire mask covers 33% of fire-box area with 4% area on normal photos (score 0.29); smoke mask covers 34% with 19% on normal photos (score 0.15). The default thresholds were already the best.
- **Why:** fire has a distinct colour, so the YCbCr rule works reasonably well.
- Smoke is grey, like sky, roads and walls, so a colour rule cannot separate them.
- No threshold setting fixed this, which shows the limit of rule-based segmentation.

### E04: 16 DIP features (e04_features)
- All 41,564 images processed, 0 failures, about 22 ms per image. Best features: fire edges, redness, fire area, haze.
- **Why:** real flames have sharp red-orange edges, and smoke makes images hazy.
- These features match what a person would look for.
- Texture and contrast numbers separated the classes much less.

### E03c / E03d: noise and brightness thresholds (e03c_noise_tune, e03d_dark_tune)
- Noise threshold σ 2.56 (balanced accuracy 0.78): detects 97% of photos with added noise σ 15, but also flags 36% of the original photos as noisy. Brightness threshold: mean grey 71.4 (balanced accuracy 0.87); it also flags 22% of the original photos as dark.
- **Why:** heavy noise is easy to detect, light noise is hard to tell apart from real image texture.
- Many CCTV photos in the datasets already contain some noise, so flagging them is partly correct.
- The thresholds were chosen on val only.

---

## Part 2: Models (val | test_in | DeepQuestAI macro-F1, DeepQuestAI false alarms, look-alike false alarms)

### A: SVM / Random Forest on the 16 features (e05_classic)
- Random Forest: test_in 0.711, DeepQuestAI 0.508, DeepQuestAI false alarms 81.6%, look-alike false alarms 40.8%.
- **Why:** 16 numbers describe the photo only roughly, so many cases look alike.
- On a new camera source, the numbers shift and the model calls most normal photos fire or smoke.
- This shows the DIP features carry information but are not enough alone.

### B: MobileNetV3-Small, CNN (e06_mobilenetv3)
- 0.886 | 0.878 | 0.856, false alarms 22.2%, look-alikes 3.6%.
- **Why:** a CNN learns its own features from the pixels, which is far richer than 16 numbers.
- It already ignores most look-alikes.
- It struggles more with photos from a new source.

### C: MobileViT-XS, transformer with DIP preprocessing (e07_mobilevit)
- 0.908 | 0.890 | 0.883, false alarms 15.5%, look-alikes 3.3%.
- **Why:** the transformer looks at the whole photo at once, so it uses context (sky, building, smoke shape).
- That helps most on the new source.
- Smoke recall is still the weak spot (0.78 on test_in).

### CR: MobileViT-XS on raw photos, no median/CLAHE (e07c_mobilevit_rawinput)
- 0.916 | 0.902 | 0.900, false alarms 12.0%, look-alikes 3.6%.
- **Why:** the photos in these datasets are mostly clean, so cleaning them adds nothing.
- The median filter slightly softens fine detail that the transformer can use.
- But see the robustness results: on noisy photos this model collapses.

### D: MobileViT-XS + 16 features joined at the end (e08_mobilevit_concat)
- 0.913 | 0.891 | 0.898, false alarms 10.7%, look-alikes 2.7%.
- **Why:** the extra numbers give a small push on val and on the new source.
- The hint test showed that switching off any feature group changes the score by less than 0.5 points.
- So most of the gain comes from the transformer itself, not the numbers.

### E: MobileViT-XS + 16 features as hint tokens inside (e09_mobilevit_diptokens)
- 0.913 | 0.895 | 0.878, false alarms 18.7%, look-alikes 3.6%.
- **Why:** best smoke recall on test_in (0.83), but more false alarms on the new source.
- The transformer already sees colour and haze in the pixels, so the hints add little.
- Putting the numbers inside did not beat putting them at the end (D).

### EM: E + learned fire and smoke maps from the boxes (e09b_mobilevit_diptokens_maps)
- 0.916 | 0.889 | 0.888, false alarms 17.4%, look-alikes 3.3%.
- **Why:** drawing the maps forces the model to find the actual flames and smoke.
- That raised val smoke recall to 0.907, the best of the first models.
- The learned maps look sensible (fire on flames, empty on normal photos), but test scores did not improve.

### CM: C + learned maps, no hints (e09c_mobilevit_maps)
- 0.913 | 0.894 | 0.875, false alarms 16.5%, look-alikes 4.2%.
- **Why:** this checks whether EM's gain comes from the maps or the hints.
- The maps alone help a little on val; the hints add a little more.
- On the tests, the differences stay within about 1 point.

### EMD: E + maps taught by our DIP masks instead of boxes (e09d_mobilevit_diptokens_dipmaps)
- 0.910 | 0.887 | 0.883, false alarms 13.1%, look-alikes 2.7%.
- **Why:** the DIP masks mark sky and grey walls as smoke, and the model copied those mistakes.
- Its learned smoke maps light up the sky, unlike EM's.
- So rule-based DIP masks cannot replace human-drawn boxes as a teacher.

### CRN: raw photos + training with added noise (e07e_mobilevit_raw_noiseaug)
- 0.909 | 0.891 | 0.897, false alarms 13.2%, look-alikes 3.6%.
- **Why:** seeing noisy photos during training teaches the model to cope with noise.
- This is the fair "why not just train with noise?" comparison.
- It gives most of the noise protection without any DIP step.

### CN: always median + CLAHE + training with noise (e07f_mobilevit_dip_noiseaug)
- 0.899 | 0.887 | 0.887, false alarms 16.8%, look-alikes 2.9%.
- **Why:** cleaning every photo, even clean ones, loses some useful detail.
- It is robust to noise, but slightly worse on clean photos.
- This shows "always clean" is not the best choice.

### CA: adaptive DIP (clean only noisy photos) + training with noise (e07d_mobilevit_adaptive_noiseaug)
- **0.917** | 0.895 | **0.900**, false alarms 12.4%, look-alikes **2.2%**. Best val score and fewest look-alike false alarms of all models.
- **Why:** clean photos keep their full detail, and noisy photos get cleaned first.
- Training with noise teaches the model to handle both kinds.
- It combines the strengths of CR (clean photos) and CN (noisy photos).

### CA2: fully adaptive DIP (noisy → median + CLAHE, dark → CLAHE, else raw) + training with noise (e07g_mobilevit_fulladaptive_noiseaug)
- val 0.908, test_in 0.890 (from LEDGER.md). Lower than CA on val (0.917), so CA stays the final model. Its robustness numbers are missing (see note).
- **Why:** the brightness rule flagged 22% of ordinary photos as "dark", so CLAHE was applied to many photos that did not need it.
- Earlier results already showed that cleaning photos that don't need it costs a little accuracy (C vs CR).
- So adding the brightness rule did not help; the simpler noise-only rule (CA) works better.

---

## Part 3: Probes (no training)

### Robustness (e10a_robustness): DeepQuestAI macro-F1 under damage

| Model | clean | dark | blur | JPEG | noise σ5 | σ10 | σ15 | σ25 |
|---|---|---|---|---|---|---|---|---|
| C (always DIP) | 0.883 | **0.895** | 0.878 | 0.828 | 0.877 | 0.820 | 0.727 | 0.546 |
| CR (raw) | 0.900 | 0.890 | **0.906** | 0.873 | 0.879 | 0.797 | 0.661 | 0.396 |
| CRN (raw + noise training) | 0.897 | 0.878 | 0.886 | **0.886** | **0.898** | **0.888** | **0.870** | 0.806 |
| CN (always DIP + noise training) | 0.887 | 0.884 | 0.868 | 0.861 | 0.882 | 0.870 | 0.857 | 0.798 |
| **CA (adaptive DIP + noise training)** | **0.900** | 0.868 | 0.889 | 0.877 | 0.894 | 0.876 | 0.860 | **0.815** |

Fire recall at heavy noise (σ 25): C 0.79, CR **0.38**, CRN 0.89, CN 0.87, **CA 0.91**.

- **Why:** a model trained only on clean photos has never seen heavy noise, so it misses most fires (CR catches only 38%).
- The median filter (C) or noise training (CRN, CA) protects against this.
- CA keeps clean-photo accuracy and stays strong under heavy noise; its weak spot is dark photos (CA2 tried to fix this but scored lower on val).

Note: CA2 was not included in the final robustness run (its run folder was not found when the probe ran).

### Hint ablation (e10b_hint_ablation)
- Switching off one DIP hint group changed macro-F1 by less than 0.5 points for D and EM; E dropped up to 1.3 points on DeepQuestAI (texture, edges).
- **Why:** the transformer already learns colour, haze and edge information from the pixels.
- The 16 numbers therefore repeat what it knows.
- This explains why D and E did not clearly beat C.

---

## Part 4: Honest summary

1. Deep models beat DIP features alone by a wide margin (0.71 → 0.89–0.90 on test_in).
2. The transformer beats the CNN, most clearly on the new source (0.856 → 0.883–0.900).
3. DIP features as hints help only slightly; the transformer barely uses them.
4. DIP preprocessing is not needed on clean photos, but it protects against noise; doing it **adaptively** (only when needed) gives the best val score (CA, 0.917) and the fewest look-alike false alarms (2.2%). Adding a brightness rule (CA2) did not help.
5. All results are from one training seed. Differences below about 1 point are not proven.
