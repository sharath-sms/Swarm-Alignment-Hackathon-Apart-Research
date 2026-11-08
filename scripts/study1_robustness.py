"""Study 1 robustness (exploratory): tip threshold, window, leave-one-cm-out, gamma=1 alternative."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, "scripts")
import study1_time_to_tip as s1
from collections import defaultdict

def load_thr(thr, Wmult):
    import glob, re, pickle
    rows = []
    for f in sorted(glob.glob(str(s1.RAW / "claude_data/*swap*24ps*.pkl"))):
        cm = int(re.search(r"swap_0_(\d+)cmtd", f)[1]); N = 24
        d = s1.Safe(open(f, "rb")).load()
        for k, r in d.items():
            tr, word = r["tracker"], r["convergence"]["committed_to"]
            com = {p for p, v in r["simulation"].items() if v["committed_tag"]}
            ans, idx = [], []
            for t, ((p1, p2), (a1, a2)) in enumerate(zip(tr["players"], tr["answers"])):
                for p, a in ((p1, a1), (p2, a2)):
                    if p not in com: ans.append(a == word); idx.append(t)
            W = int(Wmult * N); roll = np.convolve(np.array(ans, float), np.ones(W) / W, mode="valid")
            hit = np.where(roll >= thr)[0]
            T, e = (idx[hit[0] + W - 1] + 1, 1) if len(hit) else (len(tr["outcome"]), 0)
            rows.append(dict(cm=cm, N=N, p=cm / N, T=T, event=e))
    return rows

def fitall(rows, cm_min=4, drop=None):
    fr = [r for r in rows if r["cm"] >= cm_min and r["event"] and r["cm"] != drop]
    p = np.array([r["p"] for r in fr]); T = np.array([r["T"] for r in fr], float); pcm = cm_min / 24
    c, h, dl = (s1.fit(p, T, k, pcm) for k in ("CRIT", "CRIT_half", "DIL"))
    # gamma = 1 alternative (transcritical-like)
    from scipy.optimize import least_squares
    y = np.log(T); f1 = lambda th: th[0] - np.log(np.clip(p - th[1], 1e-9, None))
    r1 = least_squares(lambda th: f1(th) - y, [np.log(np.median(T)), pcm * .7], bounds=([-np.inf, 0], [np.inf, pcm - 1e-6]))
    aic1 = len(y) * np.log(np.sum(r1.fun ** 2) / len(y)) + 6
    return dict(pc=round(c["theta"][1], 3), gamma=round(c["theta"][2], 3), aic_crit_half=round(h["aic"], 1),
                aic_gamma1=round(float(aic1), 1), aic_dil=round(dl["aic"], 1), n=len(fr))

out = {}
for thr in (0.5, 0.7, 0.9):
    for Wm in (1, 2, 3):
        out[f"thr={thr},W={Wm}N"] = fitall(load_thr(thr, Wm))
base = load_thr(0.5, 2)
for cm in range(4, 13):
    out[f"LOO drop cm={cm}"] = fitall(base, drop=cm)
for k, v in out.items(): print(f"{k:22s} {v}")
Path("results/study1_time_to_tip/robustness.json").write_text(json.dumps(out, indent=2))
