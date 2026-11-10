# Results ledger

| Date | Stage | Run folder | Command | Note |
| --- | --- | --- | --- | --- |
| 2026-11-09 14:50 | e01_lookalike | `e01_lookalike/run_20261109_145044` | `E:\dip_fire_smoke\src\data\lookalike.py freeze` | E01 look-alike set frozen: 451 typed images, 19 rejected (real fire/smoke visible) |
| 2026-11-09 15:04 | e02_preprocess | `e02_preprocess/run_20261109_150207` | `E:\dip_fire_smoke\src\dip\e02_preprocess.py` | E02 preprocessing + PSNR/SSIM on 1300 images |
| 2026-11-09 15:08 | e02_preprocess | `e02_preprocess/run_20261109_150612` | `E:\dip_fire_smoke\src\dip\e02_preprocess.py` | E02 preprocessing + PSNR/SSIM on 1300 images |
| 2026-11-10 12:10 | e03_segmentation | `e03_segmentation/run_20261110_120959` | `E:\dip_fire_smoke\src\dip\e03_segmentation.py` | E03 segmentation on 600 images (masks + box overlap) |
| 2026-11-10 12:13 | e04_features | `e04_features/run_20261110_121007` | `E:\dip_fire_smoke\src\dip\e04_features.py` | E04 16 DIP features for 41564 images |
| 2026-11-10 12:16 | e05_classic | `e05_classic/run_20261110_121346` | `E:\dip_fire_smoke\src\classic\e05_classic.py` | E05 model A (SVM C=10, RF); best test_in macro-F1 0.7105 (A_rf) |
| 2026-11-10 12:42 | e03b_seg_tune | `e03b_seg_tune/run_20261110_124124` | `E:\dip_fire_smoke\src\dip\e03b_seg_tune.py` | E03b thresholds chosen on val: {"fire": {"tau_cb_cr": 40}, "smoke": {"sat_max": 70, "val_min": 80, "dark_patch": 15, "dark_thr": "otsu"}} |
| 2026-11-10 12:42 | e03_segmentation | `e03_segmentation/run_20261110_124216` | `E:\dip_fire_smoke\src\dip\e03_segmentation.py` | E03 segmentation on 600 images (masks + box overlap) |
| 2026-11-10 12:45 | e04_features | `e04_features/run_20261110_124222` | `E:\dip_fire_smoke\src\dip\e04_features.py` | E04 16 DIP features for 41564 images |
| 2026-11-10 12:48 | e05_classic | `e05_classic/run_20261110_124508` | `E:\dip_fire_smoke\src\classic\e05_classic.py` | E05 model A (SVM C=10, RF); best test_in macro-F1 0.7105 (A_rf) |
