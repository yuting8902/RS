# Validation status — v1

## Tests performed during packaging (CPU-only)

- `python -m compileall -q .` — passed.
- `python -m pytest -q tests` — passed: **20 tests**.
- `python scripts/smoke_test.py` — passed: same CPU test suite.
- `python scripts/test_pca.py` — passed: independent PCA diagnostic using small synthetic features.

Test coverage includes:

1. Stage 1 full Patch mean Coverage Gain matches explicit brute-force Cosine calculation.
2. Greedy acceptance updates Coverage; rejection does not commit Coverage.
3. Exact Stream and Disk-precompute Coverage produce the same scores on a small example.
4. Per-image PCA sufficient statistics reproduce recomputation on concatenated features.
5. PCA Reconstruction Residual matches direct projected reconstruction.
6. PCA fixed dimension errors on insufficient effective rank.
7. Residual mean, P90 and high-error ratio are computed correctly.
8. MVTec-like class discovery, candidate-source label handling, cross-split leakage detection.
9. DoRA wrapper injection does not wrap `patch_embed.proj` and retains correct QKV/attention projection keys.
10. User-provided offline DINO wrapper performs strictly matching fake checkpoint loading on a synthetic minimal model, and complete two-stage accept/reject/Resume Controller integration tests pass with synthetic features.
11. Missing verified-good Bootstrap candidates fail **before** expensive feature extraction; a completed Resume skips all feature processing.
12. `--no-resume` protects existing jobs from accidental overwrite; Bootstrap may be disabled when an initial PCA training set exists.

Additional dedicated `tests/test_pca.py` covers NumPy covariance/eigen decomposition agreement, minimum dimension from EV, accumulated statistics equivalence, residual monotonicity, image-wise reference aggregation, and effective-rank rejection.

## NOT yet tested

- Actual DINOv2-L/14-Reg + AnomalyVFM adapted checkpoint on the company H100.
- Exact feature numerical agreement with the current deployed three-stage model.
- Exact PCA numerical agreement with the current deployed three-stage model for real data.
- Full 5,000-image production run, GPU benchmark, memory pressure, and runtime.
- Final anomaly detection FPR, Recall, AUROC (these are deliberately outside this independent module).
- Rotate augmentation × 30 (not implemented in this v1).

Do not treat the CPU checks as deployment validation. Follow README steps on the company server before running full-scale selection.
