"""Classical (best-response) reference: critical mass and time-to-tip scaling in the same protocol. Zero cost."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import best_response_policy, simulate, t50
N, R = 48, 200
out = {}
for eps in (0.02, 0.1):
    P = best_response_policy(5, eps)
    rows = []
    for cm in range(1, 25):
        Ts = []
        for s in range(R):
            sim = simulate(P, N, cm, max_rounds=400, seed=s)
            Ts.append(t50(sim["f"]))
        tipped = [t for t in Ts if t is not None]
        rows.append(dict(cm=cm, p=cm/N, frac_tip=len(tipped)/R, median_T=float(np.median(tipped)) if tipped else None))
    out[f"eps={eps}"] = rows
    print(f"eps={eps}:", " ".join(f"cm{r['cm']}:{r['frac_tip']:.2f}/{r['median_T']}" for r in rows if r['cm'] <= 16))
Path("results/emulator").mkdir(parents=True, exist_ok=True)
Path("results/emulator/best_response_reference.json").write_text(json.dumps(out, indent=1))
