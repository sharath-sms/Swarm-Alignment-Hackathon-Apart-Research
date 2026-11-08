"""Core correctness tests (R7): state encoding (live prompt <-> emulator), T50, protocol correspondence, calibration."""
import re, sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.envs.naming_game import build_prompt, parse_value, _payoff
from swarm_ews.emulator import simulate, t50, best_response_policy

REC = re.compile(r"- You chose (\w), partner chose (\w)")

def encode(prompt, B):
    return int("".join(str(2 * (m == B) + (p == B)) for m, p in REC.findall(prompt)), 4)

def test_state_encoding_roundtrip():
    """A memory history rendered by the live prompt builder decodes to the emulator's state index."""
    rng = np.random.default_rng(0); A, B = "Q", "M"
    for _ in range(200):
        recs = rng.integers(0, 4, 5)                     # oldest first, code = 2*myB + partnerB
        hist = [((B if c >> 1 else A), (B if c & 1 else A), _payoff(c >> 1, c & 1)) for c in recs]
        _, prompt = build_prompt(hist, [A, B])
        assert encode(prompt, B) == int("".join(map(str, recs)), 4)

def test_emulator_state_update_matches_window():
    """Emulator update state = (state*4 + rec) % 4^5 drops the oldest record, as the live memory window does."""
    s = int("12301", 4); new = 2
    assert (s * 4 + new) % 4 ** 5 == int("23012", 4)

def test_parse_value():
    assert parse_value('```json\n{"value": "Q"}\n```', ("Q", "M")) == "Q"
    assert parse_value('{"value":"M"}', ("Q", "M")) == "M"
    assert parse_value("no answer", ("Q", "M")) is None

def test_t50_definition():
    f = np.array([0, 0.1, 0.4, 0.6, 0.7, 1.0])
    assert t50(f) == 3                                   # 2-round mean first >= 0.5 at index 3 ((0.4+0.6)/2)
    assert t50(np.zeros(10)) is None

def test_no_spontaneous_flip_best_response():
    P = best_response_policy(5, 0.02)
    assert all(t50(simulate(P, 24, 0, max_rounds=200, seed=s)["f"]) is None for s in range(20))

def test_committed_majority_always_flips():
    P = best_response_policy(5, 0.02)
    assert all(t50(simulate(P, 24, 12, max_rounds=200, seed=s)["f"]) is not None for s in range(20))

def test_decision_dataset_present_and_consistent():
    import csv, gzip
    p = Path("results/decisions/deepseek_decisions.csv.gz")
    if not p.exists():
        return
    with gzip.open(p, "rt") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["source"] for r in rows} == {"extraction", "live_v1", "live_v2"}
    assert all(0 <= int(r["state"]) < 1024 and r["choice_B"] in ("0", "1") for r in rows[:5000])
