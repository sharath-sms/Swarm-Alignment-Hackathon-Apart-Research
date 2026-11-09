---
title: "Forecasting Takeover in LLM Agent Populations: Preregistered Tests of Policy Emulators and the Limits of Early Warning[^title]"
author: "Sharath M Shankaranarayana"
affiliation: "Synthpop.AI"
with: "Apart Research"
abstract: |
  A small committed minority can tip a population of LLM agents into a new convention, an abstraction of a few
  compromised agents hijacking swarm norms. Large-population estimates of the required critical mass rely on
  emulators driven by policies extracted from the model, but these forecasts are rarely tested prospectively. In an LLM
  naming game we run three preregistered live forecasting tests (DeepSeek-v4-flash; 83 populations; 146k decisions)
  and diagnose errors against exact log-probability policies of three open-weight models. Live decisions match the
  offline policy samples closely on average (RMS difference: estimated 95% upper bound 0.024), although small deviations near the threshold can
  shift individual forecasts. A forecast from a policy pooled across word pairs failed and showed no improvement over a
  constant reference. Exact policies explain why: averaging the transition policy across contexts yields a far sharper
  transition than the mixture of populations (flip-probability error up to 0.37), and the smoothed estimators we tested
  bias critical mass low by up to 60% at small sample sizes, with posterior intervals that rarely cover the truth.
  Target-conditional forecasts correctly predicted a large directional asymmetry (10/10 vs 2/10 takeovers), and a
  third forecast, published before its runs on new populations of a previously explored target pair, passed its
  preregistered tests (7/8 vs 0/8 takeovers; 4/4 cells inside predictive intervals), though a direction-only rule scores
  almost as well (Brier 0.06 vs 0.05). In emulation, calibrated early-warning
  indicators added no consistent discrimination beyond population level and trend. We report all failed predictions and withdrawn
  claims.
---

[^title]: Research conducted at the [Swarm Alignment Hackathon](https://apartresearch.com/sprints/swarm-alignment-2026), November 2026.

# 1. Introduction

Populations of interacting LLM agents can self-organise into collective behaviour that no single-agent evaluation
anticipates; the July 2026 Hugging Face incident is a recent example [16]. A concerning failure mode is *norm
hijacking*: a handful of agents that never update can, through ordinary repeated interaction, pull a population onto a
convention of their choosing. In the classical committed-minority model this happens once the adversarial fraction
exceeds a critical mass [8, 9], and recent work reproduces such tipping points with LLM agents [1–5].

An overseer wants to know *how many agents an adversary must control* and *whether a takeover can be seen coming*.
Because live experiments are expensive, large-population answers rely on **emulators**: simulations driven by
policies extracted from the LLM [2–4]. Prior work validated per-pair, log-probability emulators against direct
LLM simulations [2, 3] and mapped directional, convention-dependent critical masses across models [3]. What has not
been done is to **forecast prospectively**: freeze an emulator's numerical predictions, then test them on new live
populations, and diagnose the errors. This matters most for black-box APIs that expose no log-probabilities, where the
policy must be estimated from samples and choices about aggregation and smoothing are unavoidable.

**Threat model.** An adversary controls c of N agents (committed fraction p = c/N) that always play a target
convention B, or poisons agents' memories at a hidden rate q(t). The adversary may choose which convention to push. The
defender can query the model offline and observe the population's aggregate behaviour.

**Contributions.**

1. **Three preregistered forecasting tests.** An emulator built from a policy pooled across word pairs *failed* and
   showed no improvement over a constant reference; a target-conditional emulator correctly forecast a large
   directional asymmetry and improved proper scores; and a third, externally timestamped forecast confirmed a directional prediction on new populations.
2. **A decision-level check of the policy-following assumption for a sampled (no-logprob) API model**, with calibrated
   inference, power analysis, and the consequence for the forecast quantity.
3. **A controlled diagnosis of forecast error against exact policies**: aggregation across contexts versus sampling
   and smoothing, with bias, RMSE and interval coverage of the critical-mass estimate.
4. **A level-matched, surrogate-calibrated evaluation of early-warning indicators** for an emulated memory-poisoning
   driver, as a bounded secondary result.

# 2. Related Work

**Conventions and committed minorities in LLM populations.** Ashery et al. [1] showed that LLM populations form
conventions that committed minorities can overturn, noting that the tipping point depends on the conventions
involved. Flint et al. extracted per-pair log-probability policies, validated the resulting model against direct LLM
simulations, and studied group size [2]. Their *indirect tipping* study [3] showed quantitatively that critical masses
are directional and convention-dependent across four models, forming a directed graph over equilibria. Passaro et al.
[4] reused those per-pair policies to reach 10^3^–10^5^ agents across topologies and excluded conditions whose
convention is unstable with zero committed agents. De Marzo et al. [5] report conformity-driven misalignment and
hysteresis; De Nobili et al. [6, 7] give statistical-physics descriptions of LLM populations. **Target dependence,
emulator validation with log-probability policies, and the stability precondition are therefore established.** Our
contribution is complementary: *preregistered* numerical forecasts tested on new live populations, a validation for
a *sampled* policy from an API without log-probabilities, and a controlled account of how estimation and aggregation
choices propagate into critical-mass forecasts.

**Early warning and multi-agent risk.** Critical slowing down [10, 11] is used to anticipate transitions, but
conditioning on transitions inflates apparent performance [12]; indicators can also be learned [13]. Hammond et al.
[14] catalogue multi-agent risks, and Makins et al. [15] show that per-instance monitors miss distributed attacks.

# 3. Methods

## 3.1 Environment and definitions

We adapt the naming game of Ashery et al. [1] (MIT licence). N = 24 agents start on convention A. Each round agents are
paired by a uniformly random perfect matching, and each pair plays a coordination game (+100 if both choose the same
word, −50 otherwise). An agent's prompt contains only its last five records (own word, partner's word, payoff); we
remove the cumulative score of the original prompt so that no hidden state remains. Option order is shuffled per call,
reasoning is disabled (0 reasoning tokens verified), and output is JSON with one resample on parse failure
(0 unrecoverable failures in 146,097 live calls over the three tests; 121,423 in Tests 1–2). c committed agents always play B.

The observable f~t~ is the fraction of non-committed agents playing B. A population **flips** if the two-round mean of
f~t~ reaches 0.5 within 150 rounds: a *finite-horizon first-passage event* for a given N, horizon, initial state and
event definition, not a statement about an asymptotic bifurcation. φ(p) is the flip probability and the **critical
mass** p~c~ solves φ(p~c~) = ½. Emulator flip probabilities are Monte Carlo
estimates (≥ 300 runs; SE ≤ 0.03; 1,000 runs, SE ≤ 0.016, for frozen predictions). The flip event agreed with
sustained adoption (f~t~ ≥ 0.9 for 5 rounds) in 97–100% of live runs (§4.4).

The prompt is a deterministic function of the memory state s (2^10^ = 1024 states), word pair w and option order o.
Because order is randomised per call, the population dynamics are driven by the order-marginalised policy
π(s, w) = E~o~ P(B | s, w, o); an emulator with the true π(s, w) reproduces the live protocol up to Monte Carlo error.

## 3.2 Policy extraction and emulators

**API model (DeepSeek-v4-flash).** No log-probabilities are exposed, so π is estimated by sampling: 8 samples per state
cycling through six word pairs, plus 40–48 extra samples on near-threshold states, and pair-specific top-ups for the
two pairs of Test 2. **Local models (Qwen3-8B, Gemma-3-12B, Mistral-7B, Llama-3.1-8B; 4-bit MLX community builds on an
Apple M5 Pro).** We compute next-token probabilities of the two words after a forced `{"value": "` prefix, renormalised
over the two answers and tempered to T = 0.7 (the untempered next-token distribution puts ≥ 97% of its mass on the two answers), for all 1024 states × 6 pairs × 2
orders. We call these **exact policies**: exact for this forced-prefix two-answer policy of these quantised builds, not
for unrestricted generation. We compare *pooled* (π averaged over pairs) vs *target-conditional* (π for one pair)
emulators, and *unsmoothed* maximum-likelihood estimates k/n vs *Jeffreys* (k + ½)/(n + 1) and *Laplace*
(k + 1)/(n + 2) estimates.

## 3.3 Preregistered live tests

Before each live experiment we recorded SHA-256 hashes of the preregistration, code, policies and predictions (the
documents are available from the author; hashes establish content identity, not independent chronology). **Test 1**
used the pooled DeepSeek policy and counterbalanced word pairs by seed (pair allocation was therefore not balanced
within every level of c). H1: at every level of c the live flip count lies inside the emulator's 95% predictive interval
(PI), evaluated by an interval rule and a Holm-corrected binomial rule. **Test 2** (addendum) used target-conditional
policies for "push Q" (A = M, B = Q) and "push M" (A = Q, B = M) with fixed words and fresh seeds. The primary H5 is
directional: flips(push M) > flips(push Q) at c in {3, 4} (one-sided Fisher exact test); H1′ repeats the PI check. PIs
were built from Jeffreys-Beta posterior draws of the policy, which §4.3 shows to be biased; we treat this as a
limitation and add proper scores against baselines.

**Test 3** (added after independent review) forecasts the X/Y pair in both directions on *new populations*; this pair
was already explored post hoc in Test 1 (3 populations per direction), which motivated its selection. It uses
target-conditional unsmoothed policies and parametric-bootstrap PIs. Its preregistration, frozen numerical predictions
and analysis code were **pushed to the public repository before the first live population** (commit 64f280d; GitHub
server-side push event 2026-11-08 19:47:07 UTC). Design: c in {3, 4} (the levels with the largest predicted difference between directions), 4 new
populations per cell, 100-round horizon. H6 (primary): the predicted-easier direction flips more often (one-sided
Fisher); H7: the frozen predictions beat a leave-one-out constant reference in Brier score (cell-stratified paired
bootstrap CI excludes 0); H8: all cells inside 95% PIs. A planned second live model (gemma-4-31b) was infeasible
before the deadline under the provider's rate limit.

## 3.4 Monitoring evaluation (emulation)

The hidden driver is memory poisoning: each round each agent's new record is forged (partner recorded as playing B)
with probability q(t). In emulation with N = 48 (not the N = 24 of the live tests), TOWARD runs ramp q from 0 to
1.6·q~c~ over 300 rounds; 4,700 null runs per policy are generated over stationary, stop-short and away-ramp schedules,
of which the 4,345–4,402 that never tip are used.
Segments ending L in {10, 25, 50, 100} rounds before the tip are matched 1:1 to null segments on final level and trend.
Indicators are variance and lag-1 autocorrelation (AC1) converted to surrogate-calibrated z-scores, plus level, trend,
recent slopes and an oracle that observes q. Incremental value is the grouped cross-validated logistic AUROC of
level + trend with and without the indicators. Uncertainty uses a **cluster bootstrap over connected components of
matched pairs that share a null run** (fixed out-of-fold predictions); **H4 decisions (indicators add ≤ 0.02 AUROC) use
this cluster bootstrap**. A refit bootstrap (one CV repeat per replicate, whereas point estimates average five) is a
sensitivity analysis for the calibrated indicators only. This is a retrospective discrimination task on selected pre-tip segments, not a prospective alarm.

# 4. Results

## 4.1 Live decisions versus the extracted policy

We reconstructed the context of 121,375 first-attempt live decisions (67 populations) and compared them with 38,948
offline samples of the same model. **Estimand:** the live-decision-weighted mean squared difference between live and
extracted choice probabilities over (word pair, state) cells with ≥ 2 observations per side (eligibility uses counts
only). With a per-cell noise-corrected estimator, it is ≈ 0 (−0.00003 ± 0.0003; 1,691 cells, 120,575 decisions), i.e.
an RMS difference with 95% upper bound 0.024. Uncertainty is a jackknife that deletes one live *population* at a time
plus grouped deletion of extraction samples. In 60 simulated replications of a *simplified* experiment (the real
population design, visitation and stopping, but one policy pooled over word pairs and fixed extraction counts), its 95%
intervals covered the realised discrepancy 60/60 times, for perturbations scaled to nominal RMS 0 and 0.02 (the realised
decision-weighted RMS differs slightly; Clopper–Pearson lower bound 0.94). This supports the procedure in these
simulations, not yet for the full pair-specific, adaptively extracted design. A population bootstrap with replacement, which
we tried first, failed this check (0/40) because duplicated decisions bias the noise correction.

An exploratory homogeneity test against an independent-Bernoulli reference (not modelling adaptive visitation)
nonetheless rejects equality (p ≈ 0.006–0.03; simulated false-positive rate 0.045, power 0.94 for an exact 0.01 shift). The departure is concentrated in a few mid-probability states of the M → Q pair (e.g.
live 29/33 vs extraction 10/24 choosing B), not in rare transitions (cells with pooled p < 0.05 or > 0.95 contribute 19%
of the statistic). In a post-hoc plug-in sensitivity analysis, substituting live estimates in states with ≥ 5 live
decisions changed the emulated critical mass by 0.0002 (pooled), 0.002 (push Q) and 0.013 (push M, a 11% relative
change), but changed the flip probability at single levels by up to 0.02, 0.06 and **0.31**, respectively. **Live
agents deviate detectably but slightly on average from the extracted policy; near a steep threshold such small
deviations can move individual-level forecasts substantially.**

## 4.2 Preregistered test 1: a pooled emulator fails near threshold

The pooled-policy emulator predicted p~c~ = 0.135. Table 1 compares its predictions with 32 live populations.

: **Table 1.** Preregistered test 1 (DeepSeek-v4-flash, N = 24). k = number of populations that flipped. H1 requires
k to lie inside the emulator's 95% predictive interval.

| c | p | Live flips | Emulator φ | 95% PI (k) | H1 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 2 | 0.083 | 1/4 | 0.00 | [0, 0] | **fail** |
| 3 | 0.125 | 4/8 | 0.11 | [0, 5] | pass |
| 4 | 0.167 | 5/8 | 0.97 | [7, 8] | **fail** |
| 5 | 0.208 | 8/8 | 1.00 | [8, 8] | pass |
| 8 | 0.333 | 4/4 | 1.00 | [4, 4] | pass |

**H1 is rejected** under both preregistered rules. The c = 4 failure is robust (P(k ≤ 5 | φ = 0.962, n = 8) ≈ 0.003);
the c = 2 failure rests on a single run. Scored with the frozen predictions, the pooled forecast
**showed no improvement over a constant reference** (leave-one-out base rate): Brier 0.220 [0.108, 0.336] vs 0.229,
paired difference 0.009 [−0.127, 0.137], i.e. neither better nor demonstrably equivalent; log-loss 0.82 vs 0.65. *Post hoc*, outcomes at c = 2–4 tracked the committed
word (push M 5/5, push Q 0/5; Fig. 1), and re-predicting each run with its own pair's unsmoothed policy reduced the Brier
score to 0.09. (Pair allocation also varied across levels; §4.3 isolates pooling.)

![**Figure 1.** Preregistered test 1. *Left:* live flip fraction (Wilson 95% CI) vs the pooled-policy emulator.
*Right (post hoc):* flips at c = 2–4 split by the committed word.](figures/fig1_live_vs_emulator.png){width=78%}

## 4.3 Diagnosing forecast error with exact policies

**Precondition.** Llama-3.1-8B does not hold a convention without adversaries (P(B | five successful A rounds) up to
0.95; spontaneous flips in 5/6 pairs) and is excluded, following the stability criterion of [4]. Qwen, Gemma and
Mistral showed 0 spontaneous flips in 1,200 runs each (95% upper bound on the per-run rate 0.25%).

**Target dependence (replication).** Consistent with [1, 3], p~c~ depends on the pushed convention (Table 2); pushing Q
from M is the hardest target in all three models and in DeepSeek live.

**Aggregation.** Averaging the *transition policy* across pairs is not the same as averaging *population outcomes*
across pairs. The pooled emulator happens to land near the mixture's p~c~ here (e.g. Qwen 0.272 vs 0.270), but its
transition is far sharper: maximum flip-probability error 0.20–0.37 (Table 2). A mixture of sharp, target-specific
transitions is broad, which matches the Test 1 failure pattern.

**Estimation.** We re-ran sampled extraction *in silico* with a **fixed, balanced target mixture** (K a multiple of 6:
K/6 samples per pair and state), 30 replicates per K. The unsmoothed estimator's critical mass was nearly unbiased
even at K = 6 (bias −0.005, +0.006 and −0.001 for Qwen, Mistral and Gemma; RMSE 0.013–0.023), whereas Jeffreys
(bias −0.08 to −0.11) and Laplace (−0.15 to −0.20) estimates were strongly biased low at small K, with bias shrinking as K grows (Jeffreys −0.005 to +0.000 at K = 120). Interval
calibration (nominal 90%, 40 extraction replicates, Clopper–Pearson CIs on coverage) differed sharply. A parametric
bootstrap that collapses counts to a single binomial per state covered the true p~c~ in 29–38/40 (K = 12) and 34–37/40
(K = 48) replicates; a design-preserving stratified bootstrap (per pair) covered only 25–31/40 and 32–33/40, because
two to eight samples per pair cannot represent transitions never observed (each interval used only 20 draws and 80
simulated populations per threshold point, so fine differences between methods are uncertain); and stratified Jeffreys posterior draws
covered it in 0/40 at both K. (An earlier pooled-Jeffreys check with 15 replicates gave 0% and 33–53%.) Our
preregistered predictive intervals were built from Jeffreys draws and were therefore shifted toward a lower critical
mass. These are evaluations of specific approximate procedures, not of smoothing or Bayesian inference in general. Unbiased
estimates of each probability need not yield an unbiased estimate of the nonlinear threshold, and a zero count makes
the unsmoothed estimate certain that a transition never happens. Neither smoothing nor Bayesian uncertainty
propagation is ruled out; the specific posterior-mean estimators we tested are biased at small K.

: **Table 2.** Exact-policy ground truth (N = 24, 150-round horizon). Range of p~c~ across six target conventions;
maximum |Δφ| of the pooled emulator relative to the counterbalanced mixture; bias of the critical-mass estimate at
K = 6 samples per state (balanced design).

| Model | p~c~ range across targets | Hardest target | Pooled max \|Δφ\| | Bias at K = 6: MLE / Jeffreys / Laplace |
|:--|:-:|:-:|:-:|:-:|
| Qwen3-8B | 0.23–0.38 | M → Q | 0.20 | −0.005 / −0.084 / −0.151 |
| Gemma-3-12B | 0.25–0.37 | M → Q | 0.20 | −0.001 / −0.112 / −0.198 |
| Mistral-7B | 0.07–0.41 | M → Q | 0.37 | +0.006 / −0.088 / −0.163 |

![**Figure 2.** Exact-policy ground truth. *Left:* estimated / true critical mass vs samples per state in a balanced
design (K/6 samples per word pair), mean ± SD over 30 replicates, for unsmoothed (MLE) and Jeffreys estimators.
*Right:* critical mass by target convention (error bars: ±2 Monte Carlo SD over 8 seed sets; all ≤ 0.005).](figures/fig4_sensitivity.png){width=88%}

## 4.4 Preregistered test 2: a target-conditional forecast

Using target-specific DeepSeek policies, we ran 35 new populations (Table 3, Fig. 3).

: **Table 3.** Preregistered test 2 (DeepSeek-v4-flash, 5 populations per cell). Values are listed per level of c.

| Direction | c | Live k/5 | Emulator φ | 95% PI (k) |
|:--|:-:|:-:|:-:|:-:|
| push Q | 3 / 4 / 5 / 6 | 0 / 2 / 5 / 5 | 0.00 / 0.18 / 0.96 / 1.00 | [0,1] / [0,5] / [4,5] / [5,5] |
| push M | 2 / 3 / 4 | 1 / 5 / 5 | 0.01 / 0.59 / 1.00 | [0,4] / [1,5] / [5,5] |

**The directional forecast H5 is confirmed:** at c = 3–4, push M flipped 10/10 populations vs 2/10 for push Q
(one-sided Fisher p = 3.6×10^−4^). It is robust to the event definition (100-round horizon: 8/10 vs 1/10, p = 0.003;
sustained adoption: 10/10 vs 1/10, p = 6×10^−5^). **Quantitative calibration is only weakly tested:** H1′ passes
(7/7 cells inside PIs), but intervals such as [0, 5] cannot be falsified with five populations. Scored with the frozen
predictions (cell-stratified paired bootstrap), the target-conditional forecast reached Brier 0.094 [0.047, 0.160]
(log-loss 0.33), better than the frozen pooled forecast (0.224; paired difference 0.13 [0.04, 0.22]) and than a
leave-one-out constant reference (0.239; difference 0.145 [0.051, 0.211]); the pooled forecast again showed no
improvement over the constant reference (difference 0.015 [−0.061, 0.087]). With 10 vs 10 populations, H5 could detect only large effects
(≈ 0.8 vs 0.2).

![**Figure 3.** Preregistered test 2: live flip fraction (Wilson 95% CI) vs the target-conditional emulator, for both
directions.](figures/fig5_confirmatory_v2.png){width=45%}

## 4.4b Preregistered test 3: an externally timestamped forecast on new populations

The forecast was published before any live run (§3.3). On 16 new populations (Table 4) **all three preregistered
hypotheses passed.** *Directional discrimination* (H6): the predicted-easier direction (push Y) flipped 7/8 populations
vs 0/8 for push X (one-sided Fisher p = 7×10^−4^). *Proper score* (H7, as preregistered): Brier 0.052 vs 0.280 for a
leave-one-out constant reference (difference 0.228, 95% CI [0.116, 0.284]). *Calibration* (H8): all four cells fell
inside their 95% predictive intervals, three of them narrow ([4, 4], [0, 0], [0, 2]). Post-hoc diagnostics qualify H7:
a direction-only rule (every push-Y population flips, no push-X population does) scores Brier 0.0625, so the emulator's
numerical advantage over knowing the direction is small (0.010, 95% CI [−0.005, 0.041]); and when cells with identical
outcomes are given posterior uncertainty (Beta-drawn cell probabilities), the CI of the improvement over the constant
reference widens to [−0.11, 0.28]. Test 3 therefore confirms the directional forecast on new populations; it adds
little evidence about precise probabilities (4 populations per cell, same model, previously explored pair).

: **Table 4.** Preregistered test 3 (new populations; push Y: X → Y, push X: Y → X; N = 24; 100 rounds; predictions public
before the runs).

| Direction | c | Live k/4 | Emulator φ | 95% PI (k) |
|:--|:-:|:-:|:-:|:-:|
| push Y | 3 / 4 | 3 / 4 | 0.88 / 1.00 | [1,4] / [4,4] |
| push X | 3 / 4 | 0 / 0 | 0.00 / 0.07 | [0,0] / [0,2] |

## 4.5 Monitoring: no consistent incremental value from the tested indicators

In the preregistered unmatched design, level and trend alone separate TOWARD from null segments almost perfectly
(CV AUROC 0.97–1.00), so H4 passes trivially. After level-and-trend matching (Appendix E, Fig. 4), level and trend
are uninformative (AUROC ≈ 0.50) while an oracle that sees the poisoning rate reaches 0.87–1.00 at leads of 25–50
rounds, so information about the driver exists beyond level and trend. Calibrated variance and AC1 indicators alone
reach 0.43–0.58 with no consistent sign; their incremental value over level + trend ranged from −0.17 to +0.05 AUROC,
with cluster-bootstrap CIs of about ±0.05–0.06 (no significant gain) and refit-sensitivity CIs of ±0.08–0.12. H4 is
not falsified for the calibrated indicators, but equivalence within 0.02 is not established; the preregistered *raw*
variance indicator formally fails H4 for Gemma at lead 50 (+0.099 [+0.045, +0.149]), its calibrated counterpart does not
(+0.039 [−0.013, +0.089]). **The tested indicators showed no consistent incremental discrimination** in this
retrospective, emulated (N = 48, pooled-policy) task; we did not evaluate prospective alarms.

# 5. Discussion and Limitations

**For forecasting swarm vulnerability.** For this sampled API model, live decisions tracked the extracted policy
closely on average, but small deviations moved individual-level forecasts near the threshold; this is one model and one
prompt. The decisive errors were in how the policy was *used*: averaging transition policies across contexts gave a
confidently wrong, too-sharp transition, and the posterior-mean estimators we tested made populations look easier to take
over at small sample sizes. Forecasts should be target-conditional [1, 3], estimators should be checked for bias and
coverage against a known policy, and uncertainty should be able to represent unseen transitions. For monitoring, the
tested indicators added nothing consistent beyond population state and trend; the oracle result points to the driver
itself (e.g., integrity checks on agent memories) as the more promising signal.

### Limitations

One live API model (a second was infeasible under the rate limit); N = 24 live (48 in monitoring emulation); 4–8
populations per level; one abstract game, memory length, temperature and label set; reasoning disabled; no claim about
the Hugging Face incident beyond motivation. DeepSeek policies are sampled; local policies are exact only for the
forced-prefix two-answer policy of specific 4-bit builds. Tests 1–2 PIs used biased Jeffreys draws; Test 1 pair
allocation was unbalanced; post-hoc per-pair analyses used samples partly collected after Test 1. Monitoring is
retrospective, emulated, with pooled policies and one driver.

# 6. Conclusion

A pooled-policy forecast failed, while
target-conditional forecasts confirmed a large asymmetry, including an externally timestamped directional forecast on
new populations. Exact policies trace the failure to aggregation across contexts and biased small-sample estimators: forecasts should be
target-specific, estimator-checked and prospective.

# Code and Data

Code, tests, sanitized decision-level data (`results/decisions/`), policies, predictions and analysis outputs:
<https://github.com/sharath-sms/Swarm-Alignment-Hackathon-Apart-Research>. Test 3's preregistration is public in
`prereg/` (pushed before its runs); Tests 1–2 documents and SHA-256 hashes are available from the author. Ashery et al. [1]
data used under MIT licence.

# Author Contributions

S.M.S. conceived the project, designed and ran all experiments, performed the analysis and wrote the report.

# References

1. Ashery, A. F., Aiello, L. M., & Baronchelli, A. (2025). Emergent social conventions and collective bias in LLM
   populations. *Science Advances*, 11, eadu9368.
2. Flint, A., Aiello, L. M., Pastor-Satorras, R., & Baronchelli, A. (2026). Group size effects and collective
   misalignment in LLM multi-agent systems. *Proceedings of the National Academy of Sciences*, 123, e2531697123.
3. Flint, A., Aiello, L. M., Constantino, S. M., Pastor-Satorras, R., & Baronchelli, A. (2026). Indirect tipping: a
   social attack surface in AI agent populations. arXiv:2609.25194.
4. Passaro, Flint, A., Pastor-Satorras, R., Baronchelli, A., & Aiello, L. M. (2026). Network topology shapes consensus
   and vulnerability in LLM populations. arXiv:2610.31165.
5. De Marzo, G., Bellina, A., Castellano, C., Priesemann, V., & Garcia, D. (2026). Conformity generates collective
   misalignment in AI agents societies. arXiv:2605.10721.
6. De Nobili, C., et al. (2026). Collective alignment in LLM multi-agent systems: disentangling bias from cooperation
   via statistical physics. arXiv:2605.10528.
7. De Nobili, C., Iyer, V., Codello, A., & Burioni, R. (2026). Microscopic dynamics of consensus formation in
   multi-agent LLM naming games. arXiv:2608.02178.
8. Xie, J., Sreenivasan, S., Korniss, G., Zhang, W., Lim, C., & Szymanski, B. K. (2011). Social consensus through the
   influence of committed minorities. *Physical Review E*, 84, 011130.
9. Centola, D., Becker, J., Brackbill, D., & Baronchelli, A. (2018). Experimental evidence for tipping points in social
   convention. *Science*, 360, 1116–1119.
10. Scheffer, M., et al. (2009). Early-warning signals for critical transitions. *Nature*, 461, 53–59.
11. Dakos, V., et al. (2012). Methods for detecting early warnings of critical transitions in time series illustrated
    using simulated ecological data. *PLoS ONE*, 7, e41010.
12. Boettiger, C., & Hastings, A. (2012). Early warning signals and the prosecutor's fallacy. *Proceedings of the Royal
    Society B*, 279, 4734–4739.
13. Bury, T. M., et al. (2021). Deep learning for early warning signals of tipping points. *Proceedings of the National
    Academy of Sciences*, 118, e2106140118.
14. Hammond, L., Chan, A., Clifton, J., et al. (2025). Multi-agent risks from advanced AI. arXiv:2502.14143.
15. Makins, O., Angelini, O., Shams, Z., & Phuong, M. (2026). Multi-agent AI control: distributed attacks hamper
    per-instance monitors. arXiv:2607.07368.
16. METR (2026). OpenAI–Hugging Face incident investigation. https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/

# Appendix

## A. Exploratory analyses: withdrawn and downgraded claims

In the Ashery et al. Claude archive [1], a censored survival fit strongly prefers a threshold law over no-threshold
dilution (ΔAIC = 38; p~c~ = 0.156 [0.144, 0.162]), and time-to-takeover rises steeply near the threshold (restricted
mean to 800 interactions: 202 at c = 5, 464 at c = 4, 753 at c = 3). We initially reported an exponent γ = 0.53 as
evidence for saddle-node scaling (γ = ½). **We withdraw this claim:** parameter recovery shows that the estimator
returns 0.58–0.70 on emulators with known saddle-nodes, exact Qwen/Gemma policies give γ ≈ 0.86/1.05, and large-N
simulations show no single exponent (Fig. 5). In the same archive, Llama-3-70B (N = 48) holds its convention without
adversaries but tips with a single committed agent (p = 0.02) in 16/21 runs; its threshold lies below the tested range.
We also withdraw an earlier "less excess variance before tips" result, which came from the uncalibrated variance
indicator.

![**Figure 5.** Claude archive (exploratory). *a:* censored accelerated-failure-time fits; *b:* profile likelihood of
γ; *c:* parameter recovery at the archive design.](figures/fig3b_scaling_v2.png){width=100%}

## B. Dual-use and ethical considerations

**Dual use.** Target choice changes the number of compromised agents required, and memory poisoning tips populations;
an attacker could use this to choose easier targets. We judge the marginal uplift low: the setting is an abstract
coordination game, convention dependence of critical mass is already published [1, 3], and the practical findings are
defensive (how to forecast vulnerability and check forecasts). We release a research simulator, not attack tooling.
**Data and ethics.** No human subjects were involved. The Flint et al. group-size repository (no licence file) was not
used.

## C. Research integrity

All failed preregistered predictions (H1 in Test 1; P5; raw-variance H4 for Gemma) and withdrawn claims (γ = ½; reduced
pre-tip variance; an earlier uncalibrated "statistically indistinguishable" agreement claim) are reported. An
independent internal critique, an external review and our responses are available from the author. All code, data
collection and analysis were done during the sprint (6–8 November 2026); the only pre-existing assets are the public
Ashery et al. dataset and code (MIT), whose prompt we adapted.

## D. Monitoring pipeline validation

**Pipeline validation.** On synthetic systems the pipeline detects a fold against a mean-drift null (AUROC 0.95) and
does not fire on mean drift alone (0.54). A preregistered check (P5: a level-only baseline cannot separate fold from
mean drift) **failed** (0.62). The raw binomial-normalised variance is biased on pure noise when the mean moves
(Kendall τ up to +0.4–0.7), so we recalibrated it with surrogates; residual bias remains when the mean accelerates
faster than the detrending window, i.e. just before a tip.

## E. Monitoring: level-matched results

![**Figure 4.** Level-matched monitoring (emulation, N = 48; four policies). *Top:* CV AUROC of level + trend,
level + trend + calibrated indicators, indicators only, and an oracle that knows the poisoning rate. *Bottom:* ΔAUROC
from adding each indicator (95% cluster-bootstrap CI over connected components of matched pairs); dashed lines mark the H4
margin ±0.02.](figures/fig2b_monitor_matched.png){width=100%}

# LLM Usage Statement

Claude was used for literature search, coding, analysis, drafting and editing of this report. All results and claims
were checked by the author against the experiment outputs, and the author takes full responsibility for the content.
