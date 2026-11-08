"""Scaling-law (prospective test of Study 1) and monitoring replication on EXACT local policies.
For each model passing the cm=0 precondition and each word pair: per-pair effective policy P_j = mean over option
order; emulator quench sweep at N in {24, 48, 96} with fine cm grid; fit CRIT / CRIT½ / DIL on cm levels where all
runs tip; then q(t)-ramp monitoring (incremental CV-AUROC) for the pooled policy. Writes results/emulator/local_<model>.json
"""
import json, sys, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from emulator_study import quench, scaling_from_rows, pc_from_flip

for name in sys.argv[1:]:
    js = json.loads(Path(f"results/policies_local/{name}.json").read_text()); Pj = np.array(js["P"]).mean(axis=1)
    out = {"model": name, "per_pair": {}}
    for j, pr in enumerate(js["pairs"]):
        key = "".join(pr); out["per_pair"][key] = {}
        for N in (24, 48, 96):
            rows = quench(Pj[j], N, R=120, max_rounds=1200, cms=list(range(1, N // 2 + 1)))
            sc = scaling_from_rows(rows, N); pc = pc_from_flip(rows)
            d = dict(pc_flip=pc)
            if sc:
                f = sc["fits"]; d.update(gamma=f["CRIT"]["theta"][2], pc_fit=f["CRIT"]["theta"][1],
                                         dAIC_DIL_minus_CRIThalf=f["DIL"]["aic"] - f["CRIT_half"]["aic"],
                                         dAIC_gamma_free_vs_half=f["CRIT_half"]["aic"] - f["CRIT"]["aic"])
            out["per_pair"][key][f"N{N}"] = d
            print(f"{name} {key} N={N}: " + ", ".join(f"{k}={v:.3f}" for k, v in d.items() if v is not None), flush=True)
    g = [v[f"N{N}"]["gamma"] for v in out["per_pair"].values() for N in (24, 48, 96) if "gamma" in v[f"N{N}"]]
    out["gamma_summary"] = dict(median=float(np.median(g)), q10=float(np.percentile(g, 10)), q90=float(np.percentile(g, 90)), n=len(g))
    print(f"{name} gamma across pairs×N: median {np.median(g):.3f} [10–90%: {np.percentile(g,10):.3f}, {np.percentile(g,90):.3f}]", flush=True)
    Path(f"results/emulator/local_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    # monitoring replication on pooled exact policy (emulator_study expects extract_policy-style JSON)
    pol = {"model": f"local_{name}_pooled", "memory": 5, "policy": {
        "".join(str((i // 4 ** (4 - d)) % 4) for d in range(5)): {"P_B": float(Pj.mean(0)[i]), "n": 10**6, "fail": 0} for i in range(1024)}}
    Path(f"results/policies/local_{name}_pooled.json").write_text(json.dumps(pol))
    subprocess.run([sys.executable, "scripts/emulator_study.py", "--policy", f"results/policies/local_{name}_pooled.json",
                    "--parts", "qcrit,monitor", "--R-monitor", "150"], check=False)
