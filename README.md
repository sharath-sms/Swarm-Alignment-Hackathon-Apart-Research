# swarm-ews — forecasting takeover in LLM agent populations

Apart Research **Swarm Alignment Hackathon 2026** (Track 2, bridging Track 4). Submission:
`report/Apart Research hackathon submission template.docx` (source: the `.md` next to it; build with
`python report/build_docx.py`); figures in `report/figures/`.

**Findings in one line each.** Two preregistered live forecasts (DeepSeek-v4-flash): an emulator built from a policy
pooled across word pairs failed and had no skill over the base rate; a target-conditional emulator correctly forecast a
large directional asymmetry. Live decisions deviate detectably but slightly from the extracted policy, with negligible
effect on the critical-mass forecast. Exact log-probability policies (Qwen3-8B, Gemma-3-12B, Mistral-7B) trace the
failure to aggregation across contexts and to biased small-sample smoothed estimators. In emulation, calibrated
early-warning indicators showed no consistent incremental discrimination beyond population level and trend.

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
| `scripts/m4_v3_equivalence.py`, `export_decisions.py` | calibrated live-vs-extraction analysis; sanitized decision export |
| `scripts/r5_r6_scores.py` | proper scores with CIs, event-definition/horizon sensitivity, rate bounds |
| `scripts/policy_sensitivity_v2.py`, `pc_mc_uncertainty.py` | balanced-design estimator study; Monte Carlo error of p_c |
| `scripts/monitor_v3.py` | level-matched monitoring with cluster + refit bootstrap (v2 kept for the record) |
| `scripts/study1_*.py`, `reanalysis_ashery.py` | exploratory reanalysis of Ashery et al. public data |
| `tests/` | state encoding, protocol correspondence, event definition, dataset consistency |
| `results/decisions/deepseek_decisions.csv.gz` | sanitized decision-level data (pair, option order, state, choice) |
| `results/` | all other outputs (runs, policies, predictions, analyses) |

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt mlx-lm statsmodels pytest
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/m4_v3_equivalence.py --csv          # §4.1 from the released decision data (no API cache needed)
.venv/bin/python scripts/r5_r6_scores.py                      # §4.2/§4.4 scores and sensitivity
.venv/bin/python scripts/extract_policy_local.py --model mlx-community/Qwen3-8B-4bit --name qwen3-8b
.venv/bin/python scripts/policy_sensitivity_v2.py qwen3-8b    # §4.3 estimator study
.venv/bin/python scripts/monitor_v3.py --best-response        # §4.5 (emulation)
.venv/bin/python scripts/make_figures.py
```
Live/API steps need `cp .env.example .env` and an `ADAPTION_API_KEY`. The raw API cache (prompts and responses) is not
released; the decision-level CSV contains everything the live-vs-extraction and scoring analyses use. The provider
exposes no seeds. Third-party data (`data/raw/ashery`, MIT) is downloaded from github.com/Ariel-Flint-Ashery/AI-norms.

## Pre-sprint vs sprint
All code, experiments and analysis were produced during the sprint (Nov 6–8, 2026). Pre-existing: Ashery et al.'s
public dataset and code (MIT), whose prompt we adapted. Preregistration documents and freeze hashes are available from
the author on request (hashes establish content identity, not independent chronology).
AI assistance (Claude) was used for coding, analysis and drafting.
