"""Exact (logprob) policy extraction with a local MLX model — no sampling noise, no rate limit.

For every memory state (1024), word pair (6) and option order (2) we build the *identical* prompt used by the live
environment (swarm_ews.envs.naming_game.build_prompt), apply the model's chat template (thinking disabled where
supported), append the forced assistant prefix '{"value": "', and read the next-token logits of the two words.
P(B) = softmax([l_A, l_B] / T)[1] with T = 0.7 (the live sampling temperature), i.e. renormalised over the two
valid answers. Because the live prompt is a deterministic function of (state, pair, order), these probabilities
define the live system exactly (up to the forced-prefix answer format).
Output: results/policies_local/<name>.json with P[pair][order][state] and the pooled / per-pair averages.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.envs.naming_game import build_prompt, _payoff  # noqa: E402

WORD_PAIRS = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]
RECS = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
PREFIX = '{"value": "'


def chat_ids(tok, instr, prompt):
    msgs = [{"role": "system", "content": instr}, {"role": "user", "content": prompt}]
    kw = dict(add_generation_prompt=True, tokenize=False)
    try:
        text = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        text = tok.apply_chat_template(msgs, **kw)
    except Exception:  # templates without a system role (e.g. some Gemma/Mistral) -> merge into user turn
        text = tok.apply_chat_template([{"role": "user", "content": instr + "\n\n" + prompt}], **kw)
    return tok.encode(text + PREFIX, add_special_tokens=False)


def word_token(tok, base_ids, w):
    ids = tok.encode(tok.decode(base_ids) + w, add_special_tokens=False)
    extra = ids[len(base_ids):] if ids[:len(base_ids)] == base_ids else None
    if not extra or len(extra) != 1:
        # fall back: tokenise the word alone
        alone = tok.encode(w, add_special_tokens=False)
        assert len(alone) == 1, f"word {w!r} is not a single token"
        return alone[0]
    return extra[0]


def main(a):
    model, tok = load(a.model)
    name = a.name or a.model.split("/")[-1]
    T = a.temperature
    P = np.full((6, 2, 1024), np.nan)
    mass = np.full((6, 2, 1024), np.nan)
    states = list(itertools.product(range(4), repeat=5))
    t0 = time.time()
    tok_cache = {}
    for j, (A, B) in enumerate(WORD_PAIRS):
        for o in range(2):
            opts = [A, B] if o == 0 else [B, A]
            seqs = []
            for si, s in enumerate(states):
                hist = [({"A": A, "B": B}[RECS[i][0]], {"A": A, "B": B}[RECS[i][1]], _payoff(RECS[i][0], RECS[i][1])) for i in s]
                seqs.append(chat_ids(tok, *build_prompt(hist, opts)))
            if (A, B) not in tok_cache:
                tok_cache[(A, B)] = (word_token(tok, seqs[0], A), word_token(tok, seqs[0], B))
            ia, ib = tok_cache[(A, B)]
            by_len = {}
            for si, ids in enumerate(seqs):
                by_len.setdefault(len(ids), []).append(si)
            for L, idx in by_len.items():           # equal-length batches -> no padding needed
                for b0 in range(0, len(idx), a.batch):
                    bi = idx[b0:b0 + a.batch]
                    logits = model(mx.array([seqs[i] for i in bi]))[:, -1].astype(mx.float32)
                    lp = np.array((logits - mx.logsumexp(logits, axis=-1, keepdims=True))[:, mx.array([ia, ib])])
                    P[j, o, bi] = 1.0 / (1.0 + np.exp((lp[:, 0] - lp[:, 1]) / T))
                    mass[j, o, bi] = np.exp(lp).sum(1)
            print(f"{name} pair={A}{B} order={o} done ({time.time()-t0:.0f}s) min mass on 2 answers={np.nanmin(mass[j, o]):.3f}", flush=True)
    out = dict(model=a.model, name=name, temperature=T, memory=5, pairs=WORD_PAIRS, P=P.tolist(), answer_mass_min=float(np.nanmin(mass)), answer_mass_mean=float(np.nanmean(mass)),
               note="P[pair][order][state]; state index = base-4 digits (oldest first), record code 2*myB+partnerB")
    Path("results/policies_local").mkdir(parents=True, exist_ok=True)
    Path(f"results/policies_local/{name}.json").write_text(json.dumps(out))
    print("saved", name, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", default=None)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--batch", type=int, default=32)
    main(ap.parse_args())
