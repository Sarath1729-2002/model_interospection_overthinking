"""Stage 5 (CPU, minutes): evidence tables + plots + report.md from the probe predictions.

    python cluster/sig_analyze.py --run-dir $RUN_DIR [--out ...] [--boot 400]
Writes <out>/analysis/{report.md, *.csv, layer_curves.png, trajectories.png}
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import analysis as an  # noqa: E402
from miol.signals import io as sio  # noqa: E402


def headline(D, cmp_df):
    L = []
    for t in ("a", "f", "d_gold", "d_commit", "rem"):
        sub = cmp_df[(cmp_df.target == t) & (cmp_df.vs == "B4")]
        if len(sub) == 0:
            continue
        allr = sub[sub.probe.str.startswith("ALL")]
        r = (allr if len(allr) else sub).iloc[0]
        L.append(f"- **{t}**: probe {r['probe']}  R2 {r['probe_R2']};  vs text baseline B4: dMAE {r['dMAE (probe-base)']}  -> {r['verdict']}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    D = an.Data(out)
    B = a.boot
    lc = an.layer_curves(D)
    lc.to_csv(adir / "layer_curves.csv", index=False)
    cmp_df = an.compare_table(D, B)
    cmp_df.to_csv(adir / "probe_vs_baselines.csv", index=False)
    wt = an.within_trace(D, B)
    wt.to_csv(adir / "within_trace.csv", index=False)
    bt = an.before_text(D, max(B // 2, 100))
    bt.to_csv(adir / "before_text.csv", index=False)
    mon = an.monitors(D, max(B // 2, 100))
    mon.to_csv(adir / "monitors.csv", index=False)
    ee = an.early_exit(D)
    ee.to_csv(adir / "early_exit.csv", index=False)
    rv = an.rollout_validation(D)
    rv.to_csv(adir / "rollout_validation.csv", index=False)
    pv = an.prevalence(D)
    pv.to_csv(adir / "prevalence.csv", index=False)
    plots = an.make_plots(D, adir)

    # best-layer summary
    best = []
    for t in D.names:
        if not D.has(t) or not D.layers:
            continue
        m = an.best_layer(D, t)
        row = lc[(lc.target == t) & (lc.model == m)]
        allr = lc[(lc.target == t) & (lc.model == "ALL")]
        b4 = lc[(lc.target == t) & (lc.model == "B4")]
        b1 = lc[(lc.target == t) & (lc.model == "B1")]
        best.append({"target": t, "best layer (by val)": m, "R2 best layer": float(row["r2"].iloc[0]) if len(row) else np.nan,
                     "R2 all layers": float(allr["r2"].iloc[0]) if len(allr) else np.nan,
                     "R2 B4 (text)": float(b4["r2"].iloc[0]) if len(b4) else np.nan,
                     "R2 B1 (position)": float(b1["r2"].iloc[0]) if len(b1) else np.nan,
                     "MAE all layers": float(allr["mae"].iloc[0]) if len(allr) else np.nan,
                     "MAE B4": float(b4["mae"].iloc[0]) if len(b4) else np.nan})
    import pandas as pd
    best = pd.DataFrame(best)
    best.to_csv(adir / "best_layers.csv", index=False)

    n = D.meta["n"]
    R = []
    R.append("# Signals report\n")
    R.append(f"Rows fit/val/test = {n['fit']}/{n['val']}/{n['test']} checkpoints. Everything below is on the TEST split (eval problems, "
             f"never used for a choice); uncertainty is a bootstrap over problems. Probes are ridge regressions (closed form, lambda "
             f"chosen on validation) on frozen hidden states.\n")
    R.append("## How to read this\n"
             "- Every target is an exact function of the model's own answer distribution at a line boundary (no judge, no sampling noise), "
             "except `r_*` which come from resampled continuations.\n"
             "- The comparison that matters is **probe vs B4** (position + prompt length + 'is the answer already written' flags + recent wording). "
             "A probe that only matches B1/B3 is reading position or text.\n"
             "- 'verdict' compares MAE with a 95% interval over problems; 'no clear difference' means the data cannot separate them.\n")
    R.append("## Headline (all-layer probe vs text baseline B4)\n" + (headline(D, cmp_df) or "_(not available)_") + "\n")
    R.append("## Condition prevalence\n" + an.md(pv))
    R.append("## Best layer per target and the baselines (test R2 / MAE)\n" + an.md(best))
    R.append("## Probe vs baselines (paired, cluster bootstrap)\n" + an.md(cmp_df))
    R.append("## Within-trace ordering (does the probe follow the target INSIDE a trace?)\n" + an.md(wt))
    R.append("## Before the text shows it\n" + an.md(bt))
    R.append("## Condition monitors\n" + an.md(mon))
    R.append("## Early exit on predicted P(gold) (eval problems)\n"
             "Exit at the first line boundary (k>=1, not the last) whose P(gold) estimate reaches theta; accuracy = the forced answer there. "
             "The ORACLE and TRUE rows need the gold answer / a model call per step and are upper bounds, not deployable.\n" + an.md(ee))
    if len(rv):
        R.append("## Rollout validation of the forced-answer signal\n" + an.md(rv))
    if plots:
        R.append("## Plots\n" + "\n".join(f"- {p}" for p in plots) + "\n")
    (adir / "report.md").write_text("\n".join(R))
    print("\n".join(R[:8]))
    print(f"\nfull report: {adir / 'report.md'}")


if __name__ == "__main__":
    main()
