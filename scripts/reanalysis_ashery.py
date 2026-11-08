"""Step 2 (exploratory): EWS before committed-minority tipping in Ashery et al. (Sci. Adv. 2025) public LLM data.

Data: github.com/Ariel-Flint-Ashery/AI-norms (MIT). Files *_swap_0_{cm}cmtd_*.pkl: after a convention is
established, `cm` agents are swapped to committed agents (fixed forcing, NOT a slow ramp), then the population
plays the naming game. Pickles contain only builtin types; loaded with a class-blocking unpickler.

Observable: fraction of NON-committed agents' answers equal to the committed word, in blocks of B answers.
Tip: first block where the 3-block rolling mean exceeds 0.5. Tipping runs: pre-tip segment (minus buffer).
Null runs: runs that never exceed 0.5; length-matched segments.
Indicators: variance, AC1, and binomial-normalised variance (variance / [p(1-p)/B]) to control for the
mechanical rise of binomial variance as p moves toward 0.5.

Caveat: forcing is a step, not a ramp, so this tests "do runs that will tip look different beforehand"
rather than classical slow-forcing CSD.
"""
from __future__ import annotations

import glob
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews import ews  # noqa: E402

RAW = Path("data/raw/ashery")
OUT = Path("results/ashery_reanalysis")
OUT.mkdir(parents=True, exist_ok=True)
B, BUFFER, MIN_BLOCKS = 12, 2, 20
WF, BW = 0.5, 0.2


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_runs():
    runs = []
    for f in sorted(glob.glob(str(RAW / "*_data/*swap*.pkl"))):
        m = re.search(r"(\w+?)_swap_0_(\d+)cmtd_(\w)_(\w)_.*_(\d+)ps", Path(f).name)
        model, cm, _, _, N = m.groups()
        with open(f, "rb") as fh:
            d = SafeUnpickler(fh).load()
        for run_id, r in d.items():
            sim, word = r["simulation"], r["convergence"]["committed_to"]
            committed = {k for k, v in sim.items() if v["committed_tag"]}
            seq = [a == word for (p1, p2), (a1, a2) in zip(r["tracker"]["players"], r["tracker"]["answers"])
                   for p, a in ((p1, a1), (p2, a2)) if p not in committed]
            seq = np.array(seq, float)
            nb = len(seq) // B
            if nb < 3:
                continue
            x = seq[: nb * B].reshape(nb, B).mean(1)
            runs.append(dict(model=model, N=int(N), cm=int(cm), run=run_id, x=x, file=Path(f).name))
    return runs


def tip_index(x):
    rm = np.convolve(x, np.ones(3) / 3, mode="valid")
    idx = np.where(rm > 0.5)[0]
    return int(idx[0]) if len(idx) else None


def indicators(seg):
    taus = ews.ews_taus(seg, WF, BW)
    ind = ews.rolling_indicators(seg, WF, BW)
    _, trend = ews.detrend(seg, BW)
    p = np.clip(trend, 1e-3, 1 - 1e-3)
    norm = ind["variance"] / (p * (1 - p) / B)
    taus["var_binom_norm"] = ews.trend_tau(norm)
    return {k: taus[k] for k in ("variance", "ac1", "var_binom_norm")}


def main():
    runs = load_runs()
    rng = np.random.default_rng(0)
    results = {}
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["N"])].append(r)
    for (model, N), rs in groups.items():
        tips, nulls = [], []
        for r in rs:
            t = tip_index(r["x"])
            if t is not None:
                seg = r["x"][: t - BUFFER] if t - BUFFER > 0 else r["x"][:0]
                if len(seg) >= MIN_BLOCKS:
                    tips.append((r, seg))
            elif len(r["x"]) >= MIN_BLOCKS:
                nulls.append((r, r["x"]))
        key = f"{model}_N{N}"
        info = dict(n_runs=len(rs), n_tip_usable=len(tips), n_null=len(nulls),
                    tip_cms=sorted({r["cm"] for r, _ in tips}), null_cms=sorted({r["cm"] for r, _ in nulls}))
        if len(tips) >= 4 and len(nulls) >= 4:
            lengths = [len(s) for _, s in tips]
            null_segs = []
            for r, x in nulls:
                L = min(int(rng.choice(lengths)), len(x))
                null_segs.append(x[:L])
            pos = [indicators(s) for _, s in tips]
            neg = [indicators(s) for s in null_segs]
            for k in pos[0]:
                auc, lo, hi = ews.auroc_ci([p[k] for p in pos], [q[k] for q in neg])
                info[k] = dict(auroc=round(auc, 3), ci=[round(lo, 3), round(hi, 3)])
            info["median_tip_len_blocks"] = float(np.median(lengths))
            # figure
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
            for r, s in tips[:8]:
                ax[0].plot(s, color="C3", alpha=0.6)
            for s in null_segs[:8]:
                ax[0].plot(s, color="C0", alpha=0.6)
            ax[0].set_title(f"{key}: committed-word fraction (red=tips, blue=null)")
            ax[0].set_xlabel(f"block ({B} answers)")
            for j, k in enumerate(("ac1", "var_binom_norm")):
                ax[j + 1].hist([p[k] for p in pos], alpha=0.6, color="C3", label="tip", bins=10)
                ax[j + 1].hist([q[k] for q in neg], alpha=0.6, color="C0", label="null", bins=10)
                ax[j + 1].set_title(f"Kendall tau of {k}: AUROC {info[k]['auroc']}")
                ax[j + 1].legend()
            fig.tight_layout()
            fig.savefig(OUT / f"{key}.png", dpi=140)
            plt.close(fig)
        results[key] = info
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
