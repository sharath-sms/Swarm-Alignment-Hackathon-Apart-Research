"""Confirmatory analysis of live naming-game runs vs emulator predictions (preregistered; see research/prereg.md).

For each (model, N, cm): flip fraction, T50 in interactions ((t50_round+1)*N/2), right-censoring at max_rounds.
Comparisons with the emulator (same N, cm, max_rounds, policy from results/policies/<model>.json):
  - flip probability: live k/n vs emulator p_hat -> two-sided binomial test + 95% CI for difference
  - time-to-tip: Mann-Whitney U on uncensored T50 (live vs emulator), and ratio of medians with bootstrap CI
  - scaling: CRIT / CRIT½ / DIL fits on live runs where all runs tip (same code as Study 1)
Usage: python scripts/analyze_live.py --exp live_quench --model deepseek-v4-flash --N 24
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm_ews.emulator import load_policy, simulate, t50  # noqa: E402
from emulator_study import fit_scaling  # noqa: E402


def live_rows(exp, model, N):
    rows = []
    for f in glob.glob(f"results/runs/{exp}/{model}__N{N}__cm*__s*.json"):
        r = json.loads(Path(f).read_text())
        if r["config"].get("q_ramp"):
            continue
        fs = np.array([x["f"] for x in r["rounds"]])
        t = t50(fs)
        rows.append(dict(cm=r["config"]["cm"], p=r["config"]["p"], seed=r["config"]["seed"],
                         T=None if t is None else (t + 1) * N / 2, max_rounds=r["config"]["max_rounds"],
                         parse_fail=r["parse_fail_rate"], f=fs.tolist()))
    return rows


def main(a):
    rows = live_rows(a.exp, a.model, a.N)
    P = load_policy(json.loads(Path(f"results/policies/{a.model}.json").read_text()))
    by = defaultdict(list)
    for r in rows:
        by[r["cm"]].append(r)
    out = {"model": a.model, "N": a.N, "exp": a.exp, "per_cm": {}}
    for cm in sorted(by):
        L = by[cm]; mr = L[0]["max_rounds"]
        Tl = [r["T"] for r in L if r["T"] is not None]
        em = [t50(simulate(P, a.N, cm, max_rounds=mr, seed=777_000 + s)["f"]) for s in range(a.R_em)]
        Te = [(t + 1) * a.N / 2 for t in em if t is not None]
        pe = len(Te) / a.R_em
        bt = binomtest(len(Tl), len(L), min(max(pe, 1e-3), 1 - 1e-3))
        d = dict(p=cm / a.N, n_live=len(L), flip_live=len(Tl) / len(L), flip_emulator=pe, binom_p=bt.pvalue,
                 median_T_live=float(np.median(Tl)) if Tl else None, median_T_emulator=float(np.median(Te)) if Te else None,
                 parse_fail_mean=float(np.mean([r["parse_fail"] for r in L])))
        if len(Tl) >= 3 and len(Te) >= 3:
            d["mwu_p"] = float(mannwhitneyu(Tl, Te).pvalue)
            rng = np.random.default_rng(0)
            ratios = [np.median(rng.choice(Tl, len(Tl))) / np.median(rng.choice(Te, len(Te))) for _ in range(2000)]
            d["median_ratio_live_over_em"] = [float(np.median(Tl) / np.median(Te)), float(np.percentile(ratios, 2.5)),
                                              float(np.percentile(ratios, 97.5))]
        out["per_cm"][cm] = d
        print(f"cm={cm:2d} p={cm/a.N:.3f} n={len(L):2d} flip live={d['flip_live']:.2f} em={pe:.2f} (p={bt.pvalue:.3f}) "
              f"T50 live={d['median_T_live']} em={d['median_T_emulator']} "
              f"ratio={d.get('median_ratio_live_over_em')} parse_fail={d['parse_fail_mean']:.3f}")
    full = [cm for cm in sorted(by) if all(r["T"] is not None for r in by[cm])]
    if len(full) >= 4:
        p = np.concatenate([[cm / a.N] * len(by[cm]) for cm in full]); T = np.concatenate([[r["T"] for r in by[cm]] for cm in full])
        out["scaling_live"] = fit_scaling(p, np.array(T, float), full[0] / a.N)
        print("live scaling fits:", {k: (round(v['aic'], 1), [round(x, 3) for x in v['theta']]) for k, v in out["scaling_live"].items()})
    Path("results/live_analysis").mkdir(parents=True, exist_ok=True)
    Path(f"results/live_analysis/{a.exp}__{a.model}__N{a.N}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--N", type=int, default=24); ap.add_argument("--R-em", type=int, default=500)
    main(ap.parse_args())
