"""Print markdown tables from results/emulator/monitor_v2_*.json (for the experiment log)."""
import json
from pathlib import Path

NAMES = ["best_response", "deepseek-v4-flash", "local_qwen3-8b_pooled", "local_gemma3-12b_pooled"]
f3 = lambda d: f"{d['est']:.3f}"
fd = lambda d: f"{d['est']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"

for n in NAMES:
    p = Path(f"results/emulator/monitor_v2_{n}.json")
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    print(f"\n**{n}** (q_c = {r['qcrit']['q_c']:.3f}; TOWARD tipped {r['toward']['n_tipped']}/{r['toward']['n_runs']}, "
          f"tip round p10/50/90 = {r['toward']['tip_round_pct10_50_90']}; runtime {r['runtime_s']:.0f} s)\n")
    print("| lead | matched/TOWARD | median dist | SMD level/trend unmatched → matched | null mix (STAT/STOPSHORT/AWAY) |")
    print("|---|---|---|---|---|")
    for L, m in r["match_quality"].items():
        sm = m["smd_matched"] or {}
        print(f"| {L[4:]} | {m['n_matched']}/{m['n_toward']} | {m['median_dist']} | "
              f"{m['smd_unmatched_orig']['level']}/{m['smd_unmatched_orig']['trend']} → {sm.get('level')}/{sm.get('trend')} | "
              f"{m['null_class_mix']['STAT']}/{m['null_class_mix']['STOPSHORT']}/{m['null_class_mix']['AWAY']} |")
    print("\n| lead | design | AUROC level | trend | var z | AC1 z | oracle q | CV L+T | CV L+T+EWS(z) | CV EWS(z) only | ΔAUROC +var z | ΔAUROC +AC1 z | CV L+T+slopes | Δ +EWS(z) over L+T+slopes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for design in ("unmatched_orig", "matched"):
        for L, d in r[design].items():
            A, D = d["auroc"], d["delta"]
            print(f"| {L[4:]} | {'orig' if design == 'unmatched_orig' else 'matched'} | {f3(A['feat:level'])} | "
                  f"{f3(A['feat:trend'])} | {f3(A['feat:var_z'])} | {f3(A['feat:ac1_z'])} | {f3(A['feat:oracle'])} | "
                  f"{f3(A['cv:LT'])} | {f3(A['cv:LT+ews_cal'])} | {f3(A['cv:ews_only_cal'])} | {fd(D['var_cal'])} | "
                  f"{fd(D['ac1_cal'])} | {f3(A['cv:LTP'])} | {fd(D['P_ews_cal'])} |")
    print("\nH4 decisions:")
    for k, v in r["H4"].items():
        tests = "; ".join(f"L{t['lead']} {t['indicator']} {t['delta']:+.3f} [{t['ci'][0]:+.3f},{t['ci'][1]:+.3f}] "
                          f"p_sup={t['p_sup_holm']:.3f} p_eq={t['p_eq_holm']:.3f}" for t in (v["tests"] if isinstance(v.get("tests"), list) else []))
        print(f"- {k}: **{v['decision']}** {('— ' + tests) if tests else ''}")
