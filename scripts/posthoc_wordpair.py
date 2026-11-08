"""POST-HOC / EXPLORATORY (not preregistered): does token identity explain the live-vs-emulator H1 failure?
1) Live outcomes by committed word. 2) Rebuild word-pair-conditional policies from cached policy samples (no new
calls): each cached request's prompt contains the words; tag gives state. 3) Emulate per word pair and compare."""
import json, re, sqlite3, sys, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import simulate, t50
from swarm_ews.envs.naming_game import parse_value
MODEL, N, MR = "deepseek-v4-flash", 24, 150
# 1) live outcomes by committed word
live = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"results/runs/live_quench/{MODEL}__N{N}__cm*__s*.json"):
    r = json.load(open(f)); A, B = r["config"]["words"]
    tip = t50(np.array([x["f"] for x in r["rounds"]])) is not None
    live[(A, B)][r["config"]["cm"]].append(int(tip))
print("LIVE flips by (A,B) and cm:")
for pair in sorted(live): print(" ", pair, {cm: f"{sum(v)}/{len(v)}" for cm, v in sorted(live[pair].items())})
# 2) word-pair-conditional policies from cache
con = sqlite3.connect("data/cache/llm_cache.sqlite")
cnt = defaultdict(lambda: np.zeros((1024, 2)))
for req, text in con.execute("select request, text from cache where request like ?", (f'%"tag": "policy/{MODEL}/%',)):
    q = json.loads(req); m = re.match(rf"policy/{re.escape(MODEL)}/(\d{{5}})/k(\d+)", q["tag"])
    if not m: continue
    state, k = m.group(1), int(m.group(2))
    pairs = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]
    A, B = pairs[k % 6]; v = parse_value(text, (A, B))
    if v is None: continue
    cnt[(A, B)][int(state, 4)] += [v == B, 1]
res = {}
for pair, c in sorted(cnt.items()):
    P = np.where(c[:, 1] > 0, (c[:, 0] + 0.5) / (c[:, 1] + 1), np.nan)
    # fall back to pooled estimate where this pair has no samples
    pooled = sum(cnt.values()); Pp = (pooled[:, 0] + 0.5) / (pooled[:, 1] + 1)
    P = np.where(np.isnan(P), Pp, P)
    key = {s: round(float(P[int(s, 4)]), 3) for s in ("00001", "00011", "00111")}
    flips = {}
    for cm in (2, 3, 4, 5):
        fl = np.mean([t50(simulate(P, N, cm, max_rounds=MR, seed=5000 + s)["f"]) is not None for s in range(400)])
        flips[cm] = round(float(fl), 2)
    res[str(pair)] = dict(n_samples=int(c[:, 1].sum()), key_states=key, emulated_flip=flips)
    print(f"EMULATOR pair A={pair[0]} B={pair[1]} n={int(c[:,1].sum())} P_B(last1/2/3 B)={list(key.values())} flips={flips}")
Path("results/live_analysis/posthoc_wordpair_deepseek.json").write_text(json.dumps(dict(live={str(k): dict(v) for k, v in live.items()}, emulator=res), indent=1))
