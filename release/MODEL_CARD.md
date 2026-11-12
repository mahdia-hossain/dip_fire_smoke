# Model card: firesmoke_CA.pt

- Model: `CA_mobilevit_adaptive_noiseaug` (mobilevit_xs), source run `e07d_mobilevit_adaptive_noiseaug/run_20261111_153248`
- Validation macro-F1: 0.9174
- Adaptive DIP thresholds: {"sigma_thr": 2.56}
- Metrics from the source run:

| split | macro-F1 | false-alarm rate |
|---|---|---|
| val | 0.9174 | 0.0552 |
| test_in | 0.895 | 0.0826 |
| test_ext | 0.9002 | 0.1242 |
| test_lookalike | nan | 0.0222 |
