"""Side-by-side comparison of two finished MATH runs (e.g. Qwen2.5-7B-Instruct vs DeepSeek-R1-Distill-Qwen-7B) on the SAME problems.

    python cluster/msig_compare.py --a $ROOT/outputs/qwen2.5_7b-math/signals --b $ROOT/outputs/r1_distill_7b-math/signals \
        --name-a qwen --name-b r1 [--out compare.md]
Needs for each run: traces_labeled_eval.csv, probes/scores_test.csv; optional: analysis/{prevalence,controls,monitors,binary_events}.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import analysis as an  # noqa: E402

TARGETS = ["rem", "d_gold", "d_commit", "a", "f", "fg", "ff", "a_next", "loss_next", "gain_next", "y_end", "ent_next", "hedge_next"]


def _read(p):
    try:
        d = pd.read_csv(p) if Path(p).exists() else None
    except pd.errors.EmptyDataError:
        return None
    return d if d is not None and len(d) else None


def behaviour(tr: pd.DataFrame, name: str) -> dict:
    cor = tr[(tr["correct"] == 1) & (tr["slack_gold"] >= 0)]
    r = {"run": name, "problems": len(tr), "terminated %": 100 * tr["has_eos"].mean(), "accuracy %": 100 * tr["correct"].mean(),
         "tokens median": tr["n_tok"].median(), "tokens mean": tr["n_tok"].mean(), "tokens p90": tr["n_tok"].quantile(0.9),
         "checkpoints/trace": tr["n_ckpt"].mean(), "median slack after answer stabilised (tok)": cor["slack_gold"].median(),
         "share of correct-trace tokens after it %": 100 * (cor["slack_gold"] / cor["n_tok"]).mean()}
    for c in ("cond_overthink_core", "cond_rationalize_core", "cond_wobble", "cond_lost", "cond_dither"):
        if c in tr:
            r[c[5:] + " %"] = 100 * tr[c].mean()
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default=None)
    x = ap.parse_args(argv)
    A, B = Path(x.a), Path(x.b)
    na, nb = x.name_a, x.name_b
    ta, tb = pd.read_csv(A / "traces_labeled_eval.csv"), pd.read_csv(B / "traces_labeled_eval.csv")
    L = [f"# {na} vs {nb} on MATH (held-out problems)\n"]

    L.append("## Behaviour (all held-out problems each run solved)\n" + an.md(pd.DataFrame([behaviour(ta, na), behaviour(tb, nb)])))
    if "level" in ta and "level" in tb:
        lv = pd.concat([ta.groupby("level").agg(acc_a=("correct", "mean"), tok_a=("n_tok", "median")),
                        tb.groupby("level").agg(acc_b=("correct", "mean"), tok_b=("n_tok", "median"))], axis=1).reset_index()
        lv.columns = ["level", f"accuracy {na}", f"median tokens {na}", f"accuracy {nb}", f"median tokens {nb}"]
        L.append("## By level\n" + an.md(lv))

    m = ta[["sample", "correct", "n_tok", "has_eos"]].merge(tb[["sample", "correct", "n_tok", "has_eos"]], on="sample", suffixes=("_a", "_b"))
    both = m[(m.has_eos_a == 1) & (m.has_eos_b == 1)]
    if len(both):
        tab = pd.crosstab(both["correct_a"].map({1: f"{na} right", 0: f"{na} wrong"}), both["correct_b"].map({1: f"{nb} right", 0: f"{nb} wrong"}))
        L.append(f"## Same problems, both finished ({len(both)} of {len(m)} shared)\n```\n{tab.to_string()}\n```\n"
                 f"median length ratio {nb}/{na}: {float((both['n_tok_b'] / both['n_tok_a']).median()):.2f}\n")

    sa, sb = _read(A / "probes" / "scores_test.csv"), _read(B / "probes" / "scores_test.csv")
    if sa is not None and sb is not None:
        rows = []
        for t in TARGETS:
            r = {"target": t}
            for nm, s in ((na, sa), (nb, sb)):
                d = s[s.target == t].set_index("model")["r2"]
                if "ALL" not in d.index:
                    continue
                txt = [k for k in ("B5", "B4", "Bq") if k in d.index]
                rival = max(txt, key=lambda k: d[k]) if txt else None
                r[f"{nm} probe R2"] = d["ALL"]
                r[f"{nm} best text R2"] = d[rival] if rival else np.nan
                r[f"{nm} gap"] = d["ALL"] - d[rival] if rival else np.nan
            rows.append(r)
        L.append("## Probe R2 on the test split vs the strongest linear text baseline (ALL = all-layer probe)\n" + an.md(pd.DataFrame(rows)))

    ca, cb = _read(A / "analysis" / "controls.csv"), _read(B / "analysis" / "controls.csv")
    if ca is not None and cb is not None:
        rows = []
        for t in TARGETS:
            r = {"target": t}
            for nm, c in ((na, ca), (nb, cb)):
                d = c[(c.target == t) & c.model.str.startswith("ADDED VALUE") & (c.model.str.contains(r"B5\]") | c.model.str.contains(r"Bq\]"))]
                if len(d):
                    r[f"{nm} dR2 over Q0+B1+text"] = float(d["R2"].iloc[0])
                    r[f"{nm} 95% CI"] = d["dR2 95% CI"].iloc[0]
            if len(r) > 1:
                rows.append(r)
        L.append("## Added value of the probe over [question-only probe + position + text] (R2 points)\n" + an.md(pd.DataFrame(rows)))

    for fname, title, keep in (("binary_events.csv", "Events (AUROC, probe vs B5)", "event"), ("monitors.csv", "Condition monitors (AUROC, probe vs B5)", "monitor")):
        da, db = _read(A / "analysis" / fname), _read(B / "analysis" / fname)
        if da is None or db is None:
            continue
        bcol = "baseline" if "baseline" in da else "base"
        pa, pb = ("AUROC probe", "AUROC baseline") if "AUROC baseline" in da else ("AUROC probe", "AUROC base")
        ka = [keep, "score"] if "score" in da else [keep]
        fa, fb = da[da[bcol] == "B5"], db[db[bcol] == "B5"]
        if len(fa) and len(fb):
            mg = fa[ka + [pa, pb]].merge(fb[ka + [pa, pb]], on=ka, suffixes=(f" {na}", f" {nb}"))
            L.append(f"## {title}\n" + an.md(mg))
    txt = "\n".join(L)
    out = Path(x.out) if x.out else B / "analysis" / f"compare_{na}_vs_{nb}.md"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(txt)
    print(txt)
    return txt


if __name__ == "__main__":
    main()
