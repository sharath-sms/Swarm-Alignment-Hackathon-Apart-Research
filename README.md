# swarm-ews — takeover vulnerability in LLM agent populations

Apart Research **Swarm Alignment Hackathon 2026** (Track 2, bridging Track 4). Report: `report/report.pdf`
(source `report/report.md`, figures in `report/figures/`).

**Findings in one line each.** Live LLM agents follow their extracted per-context policy (121k decisions); emulators
built by pooling across contexts or smoothing sampled policies misstate critical mass near the threshold; the
adversary's target convention changes the critical mass up to ~6× (same hardest target in 4 models); a
token-conditional emulator predicted live behaviour in a preregistered test; generic early-warning signals add
essentially nothing to a monitor once level and trend are matched.

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
| `scripts/policy_sensitivity.py`, `local_policy_studies.py` | pooling / smoothing / scaling studies on exact policies |
| `scripts/monitor_v2.py`, `validate_pipeline_v2.py` | level-matched monitoring; synthetic validation |
| `scripts/study1_*.py`, `reanalysis_ashery.py` | exploratory reanalysis of Ashery et al. public data |
| `results/` | all outputs (runs, policies, predictions, analyses) |

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt mlx-lm statsmodels
cp .env.example .env            # add ADAPTION_API_KEY for live/API steps (not needed for emulator/local steps)
.venv/bin/python scripts/validate_pipeline_v2.py
.venv/bin/python scripts/extract_policy_local.py --model mlx-community/Qwen3-8B-4bit --name qwen3-8b
.venv/bin/python scripts/policy_sensitivity.py results/policies_local/qwen3-8b.json
.venv/bin/python scripts/analyze_live_v2.py      # uses results/runs/live_token
.venv/bin/python scripts/make_figures.py
```
Live API calls are cached in `data/cache/` (not committed). The provider exposes no seeds; reruns reproduce via cache.
Third-party data (`data/raw/ashery`, MIT) is downloaded from github.com/Ariel-Flint-Ashery/AI-norms and not committed.

## Pre-sprint vs sprint
All code, experiments and analysis were produced during the sprint (Nov 6–8, 2026). Pre-existing: Ashery et al.'s
public dataset and code (MIT), whose prompt we adapted. The preregistration documents, freeze hashes and experiment log are available from the author on request.
AI assistance (Claude) was used for coding, analysis and drafting.
