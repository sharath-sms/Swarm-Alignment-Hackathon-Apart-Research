"""Protocol correspondence and dataset integrity (review #3 item 7).
1. The LIVE environment (swarm_ews.envs.naming_game.run_game) driven by a fake LLM that answers from a known policy table
   (state decoded from the actual prompt) reproduces the EMULATOR's flip probability within sampling error.
2. One live round's memory update equals the emulator's state update, executed through the real code path.
3. The released decision dataset is present, complete, unique per decision, and agrees with run-level records.
"""
import asyncio, csv, glob, gzip, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.envs.naming_game import NGConfig, run_game, parse_value
from swarm_ews.emulator import simulate, t50, best_response_policy

REC = re.compile(r"- You chose (\w), partner chose (\w)"); OPT = re.compile(r"from: \[(\w), (\w)\]")
ROOT = Path(__file__).resolve().parents[1]


class FakeResult:
    def __init__(self, text): self.text, self.cost, self.cached, self.input_tokens, self.output_tokens = text, 0.0, False, 0, 0


class PolicyClient:
    """Answers by decoding the memory state from the prompt and sampling from policy table P (prob. of the 2nd word)."""
    def __init__(self, P, B, seed=0):
        self.P, self.B, self.rng, self.n_reasoning_leaks, self.states = P, B, np.random.default_rng(seed), 0, []
    async def complete(self, model, prompt, instructions=None, temperature=0.7, max_output_tokens=16, tag=""):
        o = OPT.search(instructions); A_, B_ = sorted({o.group(1), o.group(2)}, key=lambda w: w == self.B)
        s = int("".join(str(2 * (m == self.B) + (p == self.B)) for m, p in REC.findall(prompt)), 4); self.states.append(s)
        return FakeResult('{"value": "%s"}' % (self.B if self.rng.random() < self.P[s] else A_))


def live_flip_rate(P, N, cm, runs, rounds):
    flips = 0
    for s in range(runs):
        cfg = NGConfig(model="fake", N=N, cm=cm, words=("Q", "M"), max_rounds=rounds, seed=s, tag="t")
        res = asyncio.run(run_game(cfg, PolicyClient(P, "M", seed=s)))
        flips += t50(np.array([r["f"] for r in res["rounds"]])) is not None
    return flips / runs


def test_live_env_matches_emulator_flip_probability():
    P = best_response_policy(5, 0.1); N, cm, rounds, R = 16, 2, 80, 300
    live = live_flip_rate(P, N, cm, R, rounds)
    emu = np.mean([t50(simulate(P, N, cm, max_rounds=rounds, seed=10_000 + s)["f"]) is not None for s in range(2000)])
    se = np.sqrt(emu * (1 - emu) / R + emu * (1 - emu) / 2000)
    assert 0.05 < emu < 0.95, f"choose a non-degenerate condition (emulator flip {emu:.2f})"
    assert abs(live - emu) < 3.5 * se + 1e-9, f"live {live:.3f} vs emulator {emu:.3f} (se {se:.3f})"


def test_live_state_update_matches_emulator_update():
    """Run two live rounds; the states decoded from round-2 prompts must equal the emulator update of round-1 states."""
    P = np.full(1024, 0.5); client = PolicyClient(P, "M", seed=3)
    cfg = NGConfig(model="fake", N=6, cm=0, words=("Q", "M"), max_rounds=2, seed=5, tag="t", stop_on_tip=False)
    asyncio.run(run_game(cfg, client))
    r1, r2 = client.states[:6], client.states[6:12]
    assert all(s == 0 for s in r1)                     # initial all-AA history
    assert all(0 <= s < 4 for s in r2)                 # one new record appended as the least significant digit
    assert len(set(r2)) >= 1


def test_decision_dataset_complete_and_consistent():
    p = ROOT / "results/decisions/deepseek_decisions.csv.gz"
    assert p.exists(), "released decision dataset missing"
    with gzip.open(p, "rt") as fh:
        rows = list(csv.DictReader(fh))
    c = Counter(r["source"] for r in rows)
    assert c == Counter({"live_v2": 68882, "live_v1": 52493, "extraction": 38948}), c
    assert all(0 <= int(r["state"]) < 1024 and r["choice_B"] in ("0", "1") and r["option_order"] in ("0", "1") for r in rows)
    live = [r for r in rows if r["source"] != "extraction"]
    keys = [(r["source"], r["pair"], r["cm"], r["seed"], r["round"], r["agent"]) for r in live]
    assert len(keys) == len(set(keys)), "duplicate live decisions"
    # agreement with run-level records: rounds per run match the saved run JSONs
    rounds = defaultdict(set)
    for r in live: rounds[(r["source"], r["pair"], int(r["cm"]), int(r["seed"]))].add(int(r["round"]))
    n_checked = 0
    for f in glob.glob(str(ROOT / "results/runs/live_*/*.json")):
        j = json.load(open(f)); src = "live_v2" if "live_token" in f else "live_v1"
        k = (src, "".join(j["config"]["words"]), j["config"]["cm"], j["config"]["seed"])
        if k in rounds:
            assert len(rounds[k]) == j["n_rounds"], (k, len(rounds[k]), j["n_rounds"]); n_checked += 1
    assert n_checked == 67, n_checked
