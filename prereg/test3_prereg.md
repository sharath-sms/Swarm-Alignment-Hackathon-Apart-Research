# Preregistration — Test 3 (fresh targets X/Y, DeepSeek-v4-flash) — frozen and published before any live run

**Published:** this file, the frozen predictions and the analysis script are committed and pushed to GitHub before the
first Test 3 live population is run; the public commit timestamp is the external record. All analysis is fixed here.

**Why this test.** Independent review asked for a fresh, externally timestamped forecast on additional targets with
prespecified numerical criteria. A second live model was planned but is infeasible before the deadline (account rate
limit ~50 requests/min); this is stated in the report.

**Disclosure.** Test 1 (post hoc) contained 3 populations pushing X (all failed) and 3 pushing Y (2 flipped) at c = 2–4;
that observation informed interest in this pair. Test 3 uses new populations and frozen numerical predictions.

## Design
- Model `deepseek-v4-flash` (Adaption API), environment `swarm_ews/envs/naming_game.py` (unchanged), N = 24, memory 5,
  T = 0.7, reasoning off, fixed words. Directions: **XY** = established X, committed agents push Y; **YX** = established
  Y, push X. Committed counts c in {3, 4} (chosen as the levels with largest predicted difference between directions).
- 4 populations per (direction, c) → 16 populations; new seeds 200–203; experiment tag `live_test3`.
- Horizon **100 rounds**; flip = 2-round mean of f_t ≥ 0.5 within 100 rounds (same T50 definition as Tests 1–2).
- Policies: target-conditional, unsmoothed MLE from cached + top-up samples (`results/policies/deepseek-v4-flash__XY.json`,
  `__YX.json`; ≥ 16 samples on states covering 90% of emulated near-threshold visits).
- Predictions: `results/predictions/test3_deepseek-v4-flash.json` (point flip probability from 1,000 emulator runs; 95%
  predictive intervals for the number of flips from a parametric bootstrap of the MLE policy, 40 draws × 200 runs).

## Frozen predictions
| direction | c | predicted flip prob. | 95% PI for flips out of 4 |
|---|---|---|---|
| XY (push Y) | 3 | 0.877 | [1, 4] |
| XY (push Y) | 4 | 1.000 | [4, 4] |
| YX (push X) | 3 | 0.000 | [0, 0] |
| YX (push X) | 4 | 0.068 | [0, 2] |
Emulated critical mass: XY 0.106, YX 0.191.

## Hypotheses (analysis: `scripts/analyze_test3.py`)
- **H6 (primary, directional):** pooled over c in {3, 4}, XY (predicted easier) flips more often than YX; one-sided
  Fisher exact test, α = 0.05.
- **H7 (numerical accuracy):** Brier score of the frozen point predictions is lower than that of a leave-one-out
  constant reference; cell-stratified paired bootstrap 95% CI of (Brier_ref − Brier_pred) excludes 0.
- **H8 (interval):** all four cells' flip counts inside their 95% PIs (reported jointly, no correction).
- Exclusions: none planned; runs aborted by errors are re-run with the same seed (cached calls identical).
- Minimal detectable effect: with 8 vs 8 populations H6 detects only large differences (≈ 0.85 vs 0.15 at 80% power).
