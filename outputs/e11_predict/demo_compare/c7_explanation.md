Answer: NORMAL (96.5% confident), model CA_mobilevit_adaptive_noiseaug
1. The photo is resized to 256 x 256 pixels.
2. Noise check: sigma = 8.42 (threshold 2.56), so the photo is noisy.
3. Isolated points: 0.014% of pixels are lonely dots.
4. Decision: noisy -> filter: median_3.
5. CLAHE on the brightness channel restores contrast without changing colours.
6. Input vs output: PSNR 19.41 dB, SSIM 0.748, mean change 20.86 grey levels, 64.46% of pixels changed.
7. MobileViT-XS scores: fire 1.5%, smoke 2.0%, normal 96.5%.
(The red/blue highlight comes from DIP colour masks and is approximate.)