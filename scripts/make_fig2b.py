"""Fig 2b: level-matched monitoring (scripts/monitor_v3.py outputs; cluster-bootstrap CIs). Top row: CV AUROC vs lead on the level-matched
sample. Bottom row: Delta CV-AUROC from adding one calibrated EWS indicator to level+trend, matched vs original
(unmatched) design, with paired-bootstrap 95% CIs; dashed lines = preregistered H4 margin +/-0.02."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
                     "axes.grid": True, "grid.color": "#ecebe8", "grid.linewidth": 0.6, "legend.frameon": False})
NAMES = {"best_response": "Best-response reference", "deepseek-v4-flash": "DeepSeek (extracted)",
         "local_qwen3-8b_pooled": "Qwen3-8B (exact)", "local_gemma3-12b_pooled": "Gemma-3-12B (exact)"}


def main(names):
    R = [(n, json.loads(Path(f"results/emulator/monitor_v3_{n}.json").read_text())) for n in names
         if Path(f"results/emulator/monitor_v3_{n}.json").exists()]
    fig, ax = plt.subplots(2, len(R), figsize=(2.1 * len(R) + 2.2, 4.8), sharey="row", squeeze=False)
    for j, (n, r) in enumerate(R):
        a = ax[0, j]; M = r["matched"]
        L = sorted(int(k[4:]) for k in M)
        g = lambda key: [M[f"lead{l}"]["auroc"][key]["est"] for l in L]
        a.plot(L, g("cv:LT"), color=BLUE, marker="o", ms=4, mec="white", label="level + trend")
        a.plot(L, g("cv:LT+ews_cal"), color=ORANGE, marker="o", ms=4, mec="white", ls="--", label="level + trend + calibrated EWS")
        a.plot(L, g("cv:ews_only_cal"), color=AQUA, marker="o", ms=4, mec="white", label="calibrated EWS only")
        a.plot(L, g("feat:oracle"), color=GRAY, ls=":", marker="o", ms=3, label="oracle (knows q)")
        a.axhline(0.5, color="#c9c8c3", lw=0.8); a.set_xticks(L); a.set_ylim(0.3, 1.02)
        a.set_title(NAMES.get(n, n), fontsize=8, loc="left", color=INK)
        b = ax[1, j]
        for k, (ind, col) in enumerate((("var_cal", ORANGE), ("ac1_cal", AQUA))):
            for design, off, mfc in (("matched", -2.0, col), ("unmatched_orig", 2.0, "white")):
                D = r[design]; LL = sorted(int(x[4:]) for x in D)
                xs = [l + off + (k - 0.5) * 3 for l in LL]
                est = [D[f"lead{l}"]["delta"][ind]["est"] for l in LL]
                lo = [e - D[f"lead{l}"]["delta"][ind]["ci"][0] for e, l in zip(est, LL)]
                hi = [D[f"lead{l}"]["delta"][ind]["ci"][1] - e for e, l in zip(est, LL)]
                b.errorbar(xs, est, yerr=[lo, hi], fmt="o", ms=3.5, color=col, mfc=mfc, mec=col, elinewidth=0.9, capsize=0)
        for y in (0.02, -0.02):
            b.axhline(y, color=GRAY, lw=0.8, ls="--")
        b.axhline(0, color="#c9c8c3", lw=0.8); b.set_xticks([10, 25, 50, 100]); b.set_xlabel("lead time before tip (rounds)")
    ax[0, 0].set_ylabel("5-fold CV AUROC\n(level-matched nulls)")
    ax[1, 0].set_ylabel("Δ CV-AUROC from adding EWS\n(95% bootstrap CI)")
    ax[0, -1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.03, 0.5))
    h = [Line2D([], [], color=ORANGE, marker="o", ls="", ms=4, label="+ variance z"),
         Line2D([], [], color=AQUA, marker="o", ls="", ms=4, label="+ AC1 z"),
         Line2D([], [], color=INK2, marker="o", ls="", ms=4, label="level-matched"),
         Line2D([], [], color=INK2, marker="o", mfc="white", ls="", ms=4, label="original (unmatched)"),
         Line2D([], [], color=GRAY, ls="--", lw=0.8, label="H4 margin ±0.02")]
    ax[1, -1].legend(handles=h, fontsize=6.5, loc="center left", bbox_to_anchor=(1.03, 0.5))
    fig.suptitle("Monitoring with level-matched nulls: do calibrated EWS add to level + trend?", x=0.01, ha="left",
                 fontsize=9, color=INK)
    fig.tight_layout()
    out = Path("report/figures"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig2b_monitor_matched.png", dpi=200); fig.savefig(out / "fig2b_monitor_matched.pdf")


if __name__ == "__main__":
    main(sys.argv[1:] or list(NAMES))
