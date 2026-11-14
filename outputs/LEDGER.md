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
| 2026-11-10 14:13 | e06_mobilenetv3 | `e06_mobilenetv3/run_20261110_132052` | `E:\dip_fire_smoke\src\deep\train.py --model B` | B_mobilenetv3: best epoch 13, val macro-F1 0.8859, test_in macro-F1 0.8776 |
| 2026-11-10 14:55 | e07_mobilevit | `e07_mobilevit/run_20261110_141639` | `E:\dip_fire_smoke\src\deep\train.py --model C` | C_mobilevit: best epoch 8, val macro-F1 0.9077, test_in macro-F1 0.8897 |
| 2026-11-10 15:45 | e08_mobilevit_concat | `e08_mobilevit_concat/run_20261110_150015` | `E:\dip_fire_smoke\src\deep\train.py --model D` | D_mobilevit_concat: best epoch 14, val macro-F1 0.9132, test_in macro-F1 0.8912 |
| 2026-11-10 16:29 | e09_mobilevit_diptokens | `e09_mobilevit_diptokens/run_20261110_154514` | `E:\dip_fire_smoke\src\deep\train.py --model E` | E_mobilevit_diptokens: best epoch 12, val macro-F1 0.913, test_in macro-F1 0.8945 |
| 2026-11-11 10:20 | e09b_mobilevit_diptokens_maps | `e09b_mobilevit_diptokens_maps/run_20261110_172623` | `E:\dip_fire_smoke\src\deep\train.py --model EM` | EM_mobilevit_diptokens_maps seed 42: best epoch 11, val macro-F1 0.9158, test_in macro-F1 0.8888 |
| 2026-11-11 11:20 | e09c_mobilevit_maps | `e09c_mobilevit_maps/run_20261111_102345` | `E:\dip_fire_smoke\src\deep\train.py --model CM` | CM_mobilevit_maps seed 42: best epoch 13, val macro-F1 0.9133, test_in macro-F1 0.8944 |
| 2026-11-11 12:24 | e07c_mobilevit_rawinput | `e07c_mobilevit_rawinput/run_20261111_112855` | `E:\dip_fire_smoke\src\deep\train.py --model CR` | CR_mobilevit_rawinput seed 42: best epoch 15, val macro-F1 0.9164, test_in macro-F1 0.9017 |
| 2026-11-11 14:01 | e09d_mobilevit_diptokens_dipmaps | `e09d_mobilevit_diptokens_dipmaps/run_20261111_122411` | `E:\dip_fire_smoke\src\deep\train.py --model EMD` | EMD_mobilevit_diptokens_dipmaps seed 42: best epoch 10, val macro-F1 0.9102, test_in macro-F1 0.8871 |
| 2026-11-11 14:26 | e10a_robustness | `e10a_robustness/run_20261111_140206` | `E:\dip_fire_smoke\src\deep\probe.py --mode robust` | robust probe on C, CR, D, E, EM, CM, EMD (seed 42) |
| 2026-11-11 15:02 | e10b_hint_ablation | `e10b_hint_ablation/run_20261111_142606` | `E:\dip_fire_smoke\src\deep\probe.py --mode ablate` | ablate probe on D, E, EM, EMD (seed 42) |
| 2026-11-11 15:28 | e03c_noise_tune | `e03c_noise_tune/run_20261111_152748` | `E:\dip_fire_smoke\src\dip\e03c_noise_tune.py` | E03c noise threshold sigma 2.56 (balanced acc 0.7841) |
| 2026-11-11 16:42 | e07d_mobilevit_adaptive_noiseaug | `e07d_mobilevit_adaptive_noiseaug/run_20261111_153248` | `E:\dip_fire_smoke\src\deep\train.py --model CA` | CA_mobilevit_adaptive_noiseaug seed 42: best epoch 15, val macro-F1 0.9174, test_in macro-F1 0.895 |
| 2026-11-11 17:47 | e07e_mobilevit_raw_noiseaug | `e07e_mobilevit_raw_noiseaug/run_20261111_164258` | `E:\dip_fire_smoke\src\deep\train.py --model CRN` | CRN_mobilevit_raw_noiseaug seed 42: best epoch 11, val macro-F1 0.9092, test_in macro-F1 0.8905 |
| 2026-11-11 18:56 | e07f_mobilevit_dip_noiseaug | `e07f_mobilevit_dip_noiseaug/run_20261111_174750` | `E:\dip_fire_smoke\src\deep\train.py --model CN` | CN_mobilevit_dip_noiseaug seed 42: best epoch 11, val macro-F1 0.8991, test_in macro-F1 0.8868 |
| 2026-11-11 19:24 | e10a_robustness | `e10a_robustness/run_20261111_185704` | `E:\dip_fire_smoke\src\deep\probe.py --mode robust` | robust probe on C, CR, CA, CRN, CN (seed 42) |
| 2026-11-11 21:14 | e03d_dark_tune | `e03d_dark_tune/run_20261111_211449` | `E:\dip_fire_smoke\src\dip\e03d_dark_tune.py` | E03d dark threshold 71.4 (balanced acc 0.8697) |
| 2026-11-11 22:45 | e07g_mobilevit_fulladaptive_noiseaug | `e07g_mobilevit_fulladaptive_noiseaug/run_20261111_213222` | `E:\dip_fire_smoke\src\deep\train.py --model CA2` | CA2_mobilevit_fulladaptive_noiseaug seed 42: best epoch 15, val macro-F1 0.9078, test_in macro-F1 0.8901 |
| 2026-11-12 10:48 | e10a_robustness | `e10a_robustness/run_20261111_224521` | `E:\dip_fire_smoke\src\deep\probe.py --mode robust` | robust probe on C, CR, CA, CRN, CN, CA2 (seed 42) |
| 2026-11-12 11:21 | e12_restoration | `e12_restoration/run_20261112_111932` | `E:\dip_fire_smoke\src\dip\e12_restoration.py` | E12 restoration study: saltpepper_2pct -> median_3; saltpepper_5pct -> median_3; gauss_10 -> bilateral; gauss_25 -> gaussian_7 |
| 2026-11-12 11:28 | e12b_noisetype_tune | `e12b_noisetype_tune/run_20261112_112818` | `E:\dip_fire_smoke\src\dip\e12b_noisetype_tune.py` | E12b points_thr 0.00989 (bal acc 0.994), heavy_thr 10.250 (bal acc 0.977) |
| 2026-11-12 12:38 | e07h_mobilevit_noisetype_adaptive | `e07h_mobilevit_noisetype_adaptive/run_20261112_112929` | `E:\dip_fire_smoke\src\deep\train.py --model CA3` | CA3_mobilevit_noisetype_adaptive seed 42: best epoch 15, val macro-F1 0.9089, test_in macro-F1 0.8883 |
| 2026-11-12 13:11 | e10a_robustness | `e10a_robustness/run_20261112_123843` | `E:\dip_fire_smoke\src\deep\probe.py --mode robust` | robust probe on CR, CA, CRN, CN, CA3 (seed 42) |
| 2026-11-12 14:07 | e12_restoration | `e12_restoration/run_20261112_140532` | `E:\dip_fire_smoke\src\dip\e12_restoration.py` | E12 restoration study: gauss_10 -> bilateral; saltpepper_2pct -> median_3; gauss_25 -> bilateral; saltpepper_5pct -> median_3 |
| 2026-11-12 14:07 | e12b_noisetype_tune | `e12b_noisetype_tune/run_20261112_140758` | `E:\dip_fire_smoke\src\dip\e12b_noisetype_tune.py` | E12b points_thr 0.00046 (bal acc 0.606), heavy_thr 4.551 (bal acc 0.853); filters: dots median_3, light bilateral, heavy bilateral |
| 2026-11-12 15:16 | e07h_mobilevit_noisetype_adaptive | `e07h_mobilevit_noisetype_adaptive/run_20261112_140818` | `E:\dip_fire_smoke\src\deep\train.py --model CA3` | CA3_mobilevit_noisetype_adaptive seed 42: best epoch 15, val macro-F1 0.9142, test_in macro-F1 0.891 |
| 2026-11-12 15:48 | e10a_robustness | `e10a_robustness/run_20261112_151618` | `E:\dip_fire_smoke\src\deep\probe.py --mode robust` | robust probe on CR, CA, CRN, CN, CA3 (seed 42) |
| 2026-11-14 17:41 | e07g_mobilevit_fulladaptive_noiseaug | `e07g_mobilevit_fulladaptive_noiseaug/run_20261111_213222` | `src\deep\reeval.py --model CA2 --run run_20261111_213222` | CA2_mobilevit_fulladaptive_noiseaug seed 42 RESTORED from checkpoint (no training): val 0.9078, test_in 0.8901, test_ext 0.901 |
| 2026-11-14 17:50 | e10a_robustness | `e10a_robustness/run_20261114_174400` | `E:\dip_fire_smoke\src\deep\probe.py --mode robust --models CA2` | robust probe on CA2 (seed 42) |
