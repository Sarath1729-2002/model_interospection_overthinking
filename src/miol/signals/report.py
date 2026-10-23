"""Plain-text gate report: does the forced-answer signal exist and move, before any probe is trained?"""
import numpy as np
import pandas as pd


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def gate_report(df: pd.DataFrame, tr: pd.DataFrame, split: str) -> str:
    L = []
    n_tr, n_ck = len(tr), len(df)
    cor = tr[tr["correct"] == 1]
    wrong = tr[tr["correct"] == 0]
    L.append(f"=== feasibility gate: split={split} ===")
    L.append(f"traces {n_tr}  checkpoints {n_ck} ({n_ck / max(n_tr, 1):.1f}/trace)  trace accuracy {_pct(tr['correct'].mean())}"
             f"  (wrong {len(wrong)})")
    # G1: the cue works
    last = df.sort_values(["sample", "k"]).groupby("sample").tail(1)
    first = df.sort_values(["sample", "k"]).groupby("sample").head(1)
    last_c = last[last["correct"] == 1]
    p_last = np.exp(np.clip(last_c["a"].astype(float), -50, 0))
    med_p = float(np.nanmedian(p_last)) if len(p_last) else float("nan")
    parse = float(np.isfinite(df["a"].astype(float)).mean())
    g1 = med_p >= 0.9 and parse >= 0.95
    L.append(f"G1 cue works          median P(gold) at the final checkpoint of correct traces = {med_p:.3f} (need >= 0.90); "
             f"gold answer scorable on {_pct(parse)} of checkpoints (need >= 95%)   -> {'PASS' if g1 else 'FAIL'}")
    # G2: a(k) moves
    m = first.set_index("sample")["a"].astype(float)
    rise = (last.set_index("sample")["a"].astype(float) - m).dropna()
    frac_rise = float((rise >= 3.0).mean()) if len(rise) else float("nan")
    g2 = frac_rise >= 0.5
    L.append(f"G2 a_k varies         traces whose a rises >= 3 nats from the prompt end to the last checkpoint = {_pct(frac_rise)} "
             f"(need >= 50%); median rise {np.nanmedian(rise):.2f} nats   -> {'PASS' if g2 else 'FAIL'}")
    a0 = np.exp(np.clip(first["a"].astype(float), -50, 0))
    L.append(f"                      P(gold) with nothing written yet: median {np.nanmedian(a0):.3f}, "
             f">0.5 on {_pct(float((a0 > 0.5).mean()))} of traces (answered from the question alone)")
    # G3: convergence before the end
    early = float(((cor["t_star"] >= 0) & (cor["t_star"] < cor["n_ckpt"] - 1)).mean()) if len(cor) else float("nan")
    over = float(cor["cond_overthink"].mean()) if len(cor) else float("nan")
    g3 = early >= 0.20
    L.append(f"G3 converges early    correct traces whose forced answer is gold-and-stays before the final checkpoint = {_pct(early)} "
             f"(need >= 20%); with >= tau tokens still to write = {_pct(over)}   -> {'PASS' if g3 else 'FAIL'}")
    sg = cor.loc[cor["slack_gold"] >= 0, "slack_gold"]
    if len(sg):
        waste = (cor.loc[cor["slack_gold"] >= 0, "slack_gold"] / cor.loc[cor["slack_gold"] >= 0, "n_tok"]).mean()
        L.append(f"                      slack after convergence: median {sg.median():.0f} tokens, mean {sg.mean():.0f}; "
                 f"{_pct(float(waste))} of a correct trace's tokens are written after the answer has stabilised")
    # wrong traces
    if len(wrong):
        cm = float((wrong["c_star"] >= 0).mean())
        rat = float(wrong["cond_rationalize"].mean())
        lost = float(wrong["cond_lost"].mean())
        L.append(f"wrong traces ({len(wrong)})   committed to their own wrong final answer before the end = {_pct(cm)}; "
                 f"with >= tau tokens still to write (rationalising) = {_pct(rat)}; "
                 f"had the gold answer at some checkpoint k>=1 and lost it = {_pct(lost)}")
    L.append(f"flips                 forced answer changes along the trace: mean {tr['flips'].mean():.2f}, "
             f">=2 flips on {_pct(float(tr['cond_dither'].mean()))} of traces")
    L.append("condition prevalence  " + "  ".join(
        f"{c[5:]}={_pct(float(tr[c].mean()))}" for c in ("cond_overthink", "cond_rationalize", "cond_lost", "cond_dither")
        if c in tr.columns) + "   (over ALL traces)")
    L.append(f"GATE: {'GO - signals look usable' if (g1 and g2 and g3) else 'CHECK - at least one gate failed; read the lines above before spending probe time'}")
    return "\n".join(L)
