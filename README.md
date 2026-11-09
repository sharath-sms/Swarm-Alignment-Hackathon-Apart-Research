# swarm-ews — forecasting takeover in LLM agent populations

Apart Research **Swarm Alignment Hackathon 2026** (Track 2, bridging Track 4). Submission:
`report/Apart Research hackathon submission template.docx` (source: the `.md` next to it; build with
`python report/build_docx.py`); figures in `report/figures/`.

**Findings in one line each.** Three preregistered live forecasts (DeepSeek-v4-flash): an emulator built from a
policy pooled across word pairs failed and showed no improvement over a constant reference; a target-conditional
emulator correctly forecast a large directional asymmetry; a third, externally timestamped forecast (pushed before its
runs, commit 64f280d) confirmed a directional prediction on new populations of a previously explored pair, although a
direction-only rule scores almost as well. Live decisions track the extracted policy closely on average (estimated 95%
upper bound on RMS difference 0.024), but small deviations can move single-level forecasts near the threshold (up to
0.31). Exact log-probability policies (Qwen3-8B, Gemma-3-12B, Mistral-7B) trace forecast errors to aggregation across
contexts and to biased small-sample smoothed estimators. In emulation, calibrated early-warning indicators showed no
consistent incremental discrimination beyond population level and trend.

## Layout
| path | what |
|---|---|
| `swarm_ews/envs/naming_game.py` | live LLM naming game (committed minority, memory poisoning, pulses) |
| `swarm_ews/emulator.py` | Markov emulator with identical protocol; best-response reference agent |
| `swarm_ews/llm.py` | async Adaption client: SQLite cache, spend ledger + cap, 95 req/min limiter, reasoning off |
| `swarm_ews/ews.py`, `ews_calibrated.py`, `survival.py` | EWS indicators, surrogate-calibrated scores, KM/RMST |
| `scripts/extract_policy*.py`, `refine_*.py` | sampled (API) and exact-logprob (MLX) policy extraction |
| `scripts/run_naming.py` | resumable live experiment runner |
| `scripts/make_predictions.py`, `analyze_live*.py` | preregistered predictions and analyses |
| `scripts/m4_v4.py`, `export_decisions.py` | live-vs-extraction analysis (population jackknife, simulated coverage); decision export |
| `scripts/r5_v2_scores.py`, `analyze_test3.py`, `test3_h7_sensitivity.py` | frozen-prediction scores; Test 3 analysis; post-hoc H7 diagnostics |
| `scripts/r5_r6_scores.py` | event-definition/horizon sensitivity, rate bounds (scores superseded by r5_v2) |
| `scripts/policy_sensitivity_v2.py`, `policy_sensitivity_v3.py`, `pc_mc_uncertainty.py` | estimator bias (balanced design); interval coverage; Monte Carlo error of p_c |
| `scripts/monitor_v3.py` | level-matched monitoring with cluster + refit bootstrap (v2 kept for the record) |
| `scripts/study1_*.py`, `reanalysis_ashery.py` | exploratory reanalysis of Ashery et al. public data |
| `tests/` | state encoding, protocol correspondence, event definition, dataset consistency |
| `results/decisions/deepseek_decisions.csv.gz` | sanitized decision-level data (pair, option order, state, choice) |
| `results/` | all other outputs (runs, policies, predictions, analyses) |

## Reproduce (authoritative analyses)
```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt      # pinned; or requirements.txt
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/m4_v4.py                       # §4.1 live vs extraction (reads results/decisions CSV)
.venv/bin/python scripts/r5_v2_scores.py                # §4.2/§4.4 proper scores with frozen predictions
.venv/bin/python scripts/analyze_test3.py               # §4.4b Test 3 (preregistered analysis)
.venv/bin/python scripts/test3_h7_sensitivity.py        # §4.4b post-hoc H7 diagnostics
.venv/bin/python scripts/policy_sensitivity_v2.py qwen3-8b   # §4.3 estimator bias (needs exact policy, below)
.venv/bin/python scripts/policy_sensitivity_v3.py qwen3-8b   # §4.3 interval coverage
.venv/bin/python scripts/monitor_v3.py --best-response  # §4.5 monitoring (emulation)
.venv/bin/python scripts/extract_policy_local.py --model mlx-community/Qwen3-8B-4bit --name qwen3-8b   # exact policy (MLX)
```
**Historical scripts** kept for the record (superseded): `m4_live_vs_extraction*.py`, `m4_v3_equivalence.py`,
`r5_r6_scores.py`, `policy_sensitivity.py`, `monitor_v2.py`. Live/API steps need `cp .env.example .env` and an
`ADAPTION_API_KEY`; the raw API cache is not released, but `results/decisions/deepseek_decisions.csv.gz` (Tests 1–2)
contains what the live-vs-extraction analysis uses. Test 3 preregistration: `prereg/test3_prereg.md`.

## Pre-sprint vs sprint
All code, experiments and analysis were produced during the sprint (Nov 6–8, 2026). Pre-existing: Ashery et al.'s
public dataset and code (MIT), whose prompt we adapted. Preregistration documents and freeze hashes are available from
the author on request (hashes establish content identity, not independent chronology).
AI assistance (Claude) was used for coding, analysis and drafting.
