"""Paper figures. Palette: reference categorical slots 1-3 (validated all-pairs, light mode) + neutral gray.
Fig1 live vs emulator flip curve (DeepSeek) + flips by committed token. Fig2 monitor AUROC vs lead time.
Fig3 time-to-tip scaling (Ashery Claude archive)."""
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import load_policy, simulate, t50

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "axes.grid": True, "grid.color": "#ecebe8", "grid.linewidth": 0.8, "legend.frameon": False})
OUT = Path("report/figures"); OUT.mkdir(parents=True, exist_ok=True)


def fig1():
    M, N, MR = "deepseek-v4-flash", 24, 150
    P = load_policy(json.loads(Path(f"results/policies/{M}.json").read_text()))
    cms = np.arange(1, 10)
    em = [np.mean([t50(simulate(P, N, c, max_rounds=MR, seed=9000 + s)["f"]) is not None for s in range(600)]) for c in cms]
    live = defaultdict(list); tok = defaultdict(list)
    for f in glob.glob(f"results/runs/live_quench/{M}__N{N}__cm*__s*.json"):
        r = json.load(open(f)); tip = int(t50(np.array([x["f"] for x in r["rounds"]])) is not None)
        live[r["config"]["cm"]].append(tip)
        if r["config"]["cm"] in (2, 3, 4): tok[r["config"]["words"][1]].append(tip)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    a.plot(cms / N, em, color=BLUE, label="Emulator (pooled extracted policy)")
    ks = sorted(live); p = np.array([np.mean(live[k]) for k in ks]); n = np.array([len(live[k]) for k in ks])
    lo, hi = proportion_confint(p * n, n, method="wilson")
    a.errorbar(np.array(ks) / N, p, yerr=[p - lo, hi - p], fmt="o", ms=6, color=ORANGE, ecolor=ORANGE, elinewidth=1.2,
               capsize=0, mec="white", mew=1.5, label="Live LLM populations (Wilson 95% CI)")
    for k, pp, nn in zip(ks, p, n):
        a.annotate(f"{int(round(pp*nn))}/{nn}", (k / N, pp), xytext=(6, -10), textcoords="offset points", fontsize=7, color=INK2)
    a.set_xlabel("committed fraction p"); a.set_ylabel("P(population flips within 150 rounds)")
    a.set_title("Preregistered test: live vs emulator (DeepSeek-v4-flash, N=24)", fontsize=9, loc="left", color=INK)
    a.legend(fontsize=7, loc="lower right"); a.set_ylim(-0.05, 1.08)
    b = ax[1]
    order = ["M", "Y", "F", "J", "X", "Q"]
    vals = [np.mean(tok[t]) for t in order]; nn = [len(tok[t]) for t in order]
    b.bar(range(len(order)), vals, color=GRAY, width=0.6)
    for i, (v, m) in enumerate(zip(vals, nn)):
        b.text(i, v + 0.03, f"{int(round(v*m))}/{m}", ha="center", fontsize=7, color=INK2)
    b.set_xticks(range(len(order)), order); b.set_ylim(0, 1.15)
    b.set_xlabel("committed (adversarial) word"); b.set_ylabel("flip fraction, cm = 2–4")
    b.set_title("Post-hoc: token identity", fontsize=9, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(OUT / "fig1_live_vs_emulator.pdf"); fig.savefig(OUT / "fig1_live_vs_emulator.png", dpi=200)


def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for a, name, title in ((ax[0], "best_response", "Classical best-response agents"),
                           (ax[1], "deepseek-v4-flash", "DeepSeek extracted policy")):
        r = json.loads(Path(f"results/emulator/{name}.json").read_text())["monitor_offline"]
        leads = sorted(int(k[4:]) for k in r)
        get = lambda key: [r[f"lead{L}"]["incremental"]["POOLED"][key] for L in leads]
        a.plot(leads, get("level_trend"), color=BLUE, marker="o", ms=5, mec="white", label="level + trend")
        a.plot(leads, get("plus_ews"), color=ORANGE, marker="o", ms=5, mec="white", ls="--", label="level + trend + EWS")
        a.plot(leads, get("ews_only"), color=AQUA, marker="o", ms=5, mec="white", label="EWS only (variance, AC1)")
        a.plot(leads, [r[f"lead{L}"]["auroc"]["POOLED"]["oracle"] for L in leads], color=GRAY, ls=":", label="oracle (knows q)")
        a.axhline(0.5, color="#c9c8c3", lw=1); a.set_xlabel("lead time before tip (rounds)")
        a.set_title(title, fontsize=9, loc="left", color=INK); a.set_xticks(leads)
    ax[0].set_ylabel("5-fold CV AUROC (vs pooled nulls)"); ax[1].legend(fontsize=7, loc="lower left"); ax[0].set_ylim(0.45, 1.02)
    fig.tight_layout(); fig.savefig(OUT / "fig2_monitor_leadtime.pdf"); fig.savefig(OUT / "fig2_monitor_leadtime.png", dpi=200)


def fig3():
    sys.path.insert(0, "scripts")
    import study1_time_to_tip as s1
    rows = s1.load("claude_data/*swap*24ps*.pkl")
    js = json.loads(Path("results/study1_time_to_tip/summary.json").read_text())["claude_N24"]
    fig, a = plt.subplots(figsize=(3.6, 2.8))
    tip = [r for r in rows if r["event"]]; cen = [r for r in rows if not r["event"]]
    a.scatter([r["p"] for r in tip], [r["T"] for r in tip], s=10, color=BLUE, alpha=0.5, lw=0, label="tipped run")
    a.scatter([r["p"] for r in cen], [r["T"] for r in cen], s=16, marker="^", facecolor="white", edgecolor=GRAY, label="censored (no tip)")
    pc = js["fits"]["CRIT_half"]["theta"][1]; A = js["fits"]["CRIT_half"]["theta"][0]
    pg = np.linspace(pc + 0.004, 0.5, 200); a.plot(pg, np.exp(A - 0.5 * np.log(pg - pc)), color=ORANGE, label=f"saddle-node, γ=½, p_c={pc:.3f}")
    Ad, gd = js["fits"]["DIL"]["theta"]; pg2 = np.linspace(0.08, 0.5, 200)
    a.plot(pg2, np.exp(Ad - gd * np.log(pg2)), color=GRAY, ls="--", lw=1.5, label="no threshold (T ∝ p^−γ)")
    a.set_yscale("log"); a.set_xlabel("committed fraction p"); a.set_ylabel("time to tip T50 (interactions)")
    a.set_title("Ashery et al. Claude archive (exploratory)", fontsize=9, loc="left", color=INK); a.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(OUT / "fig3_scaling_claude.pdf"); fig.savefig(OUT / "fig3_scaling_claude.png", dpi=200)




def fig4():
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    a = ax[0]
    for name, c, lab in (("qwen3-8b", BLUE, "Qwen3-8B"), ("gemma3-12b", ORANGE, "Gemma-3-12B"), ("mistral-7b", AQUA, "Mistral-7B")):
        r = json.loads(Path(f"results/sensitivity/{name}.json").read_text()); Ks = sorted(int(k) for k in r["sampling"])
        truth = r["pooled"]["p_c"]
        for est, ls, mk in (("mle", "-", "o"), ("jeffreys", "--", "s")):
            v = np.array([r["sampling"][str(K)][f"pooled_pc_{est}"] for K in Ks]) / truth
            a.plot(Ks, v[:, 1], color=c, ls=ls, marker=mk, ms=4, mec="white", label=f"{lab}, {'MLE' if est=='mle' else 'Jeffreys-smoothed'}")
            a.fill_between(Ks, v[:, 0], v[:, 2], color=c, alpha=0.12, lw=0)
    a.axhline(1, color=INK2, lw=1); a.set_xscale("log"); a.set_xticks([4, 8, 16, 48, 128], ["4", "8", "16", "48", "128"])
    a.set_xlabel("samples per memory state (K)"); a.set_ylabel("estimated p_c / true p_c")
    a.set_title("Policy-estimation bias in emulator critical mass", fontsize=9, loc="left", color=INK); a.legend(fontsize=6.5)
    b = ax[1]; pairs = ["MQ", "YX", "FJ", "JF", "XY", "QM"]; x = np.arange(len(pairs)); w = 0.27
    for k, (name, c, lab) in enumerate((("qwen3-8b", BLUE, "Qwen3-8B"), ("gemma3-12b", ORANGE, "Gemma-3-12B"), ("mistral-7b", AQUA, "Mistral-7B"))):
        r = json.loads(Path(f"results/sensitivity/{name}.json").read_text())
        b.bar(x + (k - 1) * w, [r["per_pair"][p]["p_c"] for p in pairs], width=w - 0.03, color=c, label=lab)
    b.set_xticks(x, [f"{p[0]}→{p[1]}" for p in pairs]); b.set_ylabel("critical mass p_c (exact policy)")
    b.set_xlabel("established → pushed word"); b.legend(fontsize=7)
    b.set_title("Critical mass depends on the target", fontsize=9, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(OUT / "fig4_sensitivity.pdf"); fig.savefig(OUT / "fig4_sensitivity.png", dpi=200)




def fig5():
    M, N = "deepseek-v4-flash", 24
    fig, a = plt.subplots(figsize=(3.8, 2.8))
    for d, c, lab in (("QM", BLUE, "push M (Q→M)"), ("MQ", ORANGE, "push Q (M→Q)")):
        P = load_policy(json.loads(Path(f"results/policies/{M}__{d}.json").read_text()))
        cms = np.arange(1, 9)
        em = [np.mean([t50(simulate(P, N, c_, max_rounds=150, seed=4000 + s)["f"]) is not None for s in range(400)]) for c_ in cms]
        a.plot(cms / N, em, color=c, label=f"emulator, {lab}")
        live = defaultdict(list)
        for f in glob.glob(f"results/runs/live_token/{M}__N{N}__cm*__s*__w{d}.json"):
            r = json.load(open(f)); live[r["config"]["cm"]].append(int(t50(np.array([x["f"] for x in r["rounds"]])) is not None))
        ks = sorted(live); p = np.array([np.mean(live[k]) for k in ks]); n = np.array([len(live[k]) for k in ks])
        lo, hi = proportion_confint(p * n, n, method="wilson")
        a.errorbar(np.array(ks) / N, p, yerr=[p - lo, hi - p], fmt="o", ms=6, color=c, elinewidth=1.2, mec="white", mew=1.5,
                   label=f"live, {lab}")
    a.set_xlabel("committed fraction p"); a.set_ylabel("P(flip within 150 rounds)"); a.set_ylim(-0.05, 1.08)
    a.set_title("Confirmatory v2: token-conditional emulator", fontsize=9, loc="left", color=INK); a.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig5_confirmatory_v2.pdf"); fig.savefig(OUT / "fig5_confirmatory_v2.png", dpi=200)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5"]:
        globals()[f](); print("made", f)
