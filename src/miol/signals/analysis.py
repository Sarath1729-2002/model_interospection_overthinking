"""Turn probes + labels into the evidence tables.  Everything is computed on the TEST split (the eval problems); the validation
split is used only to pick 'the best layer' per target.  Uncertainty = bootstrap over problems (clusters), not over checkpoints.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from miol.signals import io as sio
from miol.signals.metrics import (auroc, boot_auroc, boot_paired, boot_reg, boot_within, mae, r2, spearman,
                                  within_group_spearman)
from miol.signals.targets import build_targets

KEY = ["a", "f", "g", "rem", "d_gold", "d_commit", "fg", "ff", "r_pgold", "r_pfinal", "r_H"]
LOG_HALF = float(np.log(0.5))


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                         else ("" if isinstance(v, float) and not np.isfinite(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ci(d, nd=3):
    f = f"{{:.{nd}f}}"
    return f"{f.format(d['est'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


class Data:
    """Loaded once: predictions, labels and targets for val and test, aligned."""

    def __init__(self, out: Path):
        self.out = Path(out)
        pdir = self.out / "probes"
        self.meta = json.load(open(pdir / "meta.json"))
        z = np.load(pdir / "preds.npz")
        self.names: List[str] = self.meta["targets"]
        self.models: List[str] = self.meta["models"]
        self.P = {s: {m: z[f"{s}/{m}"] for m in self.models} for s in ("val", "test")}
        tr = sio.read_table(self.out / "labels_train.csv")
        ev = sio.read_table(self.out / "labels_eval.csv")
        self.df = {"test": ev.set_index("row").loc[z["row_test"]].reset_index(),
                   "val": tr.set_index("row").loc[z["row_val"]].reset_index()}
        self.Y = {s: build_targets(self.df[s], names=self.names)[0] for s in ("val", "test")}
        self.traces = {"eval": sio.read_table(self.out / "traces_labeled_eval.csv"),
                       "train": sio.read_table(self.out / "traces_labeled_train.csv")}
        self.layers = sorted(int(m[1:]) for m in self.models if m[0] == "L" and m[1:].isdigit())

    def j(self, t):
        return self.names.index(t)

    def y(self, t, s="test"):
        return self.Y[s][:, self.j(t)]

    def p(self, m, t, s="test"):
        return self.P[s][m][:, self.j(t)]

    def ok(self, t, s="test"):
        return np.isfinite(self.y(t, s))

    def has(self, t):
        return t in self.names and self.ok(t).sum() > 50


def r2_val(D: Data, m: str, t: str) -> float:
    ok = D.ok(t, "val") & np.isfinite(D.p(m, t, "val"))
    return r2(D.y(t, "val")[ok], D.p(m, t, "val")[ok]) if ok.sum() > 10 else float("nan")


def best_layer(D: Data, t: str) -> str:
    cand = [f"L{l}" for l in D.layers]
    sc = [r2_val(D, m, t) for m in cand]
    return cand[int(np.nanargmax(sc))] if cand and np.isfinite(sc).any() else (cand[0] if cand else "ALL")


def layer_curves(D: Data) -> pd.DataFrame:
    rec = []
    for t in D.names:
        ok = D.ok(t)
        for m in D.models:
            p = D.p(m, t)
            k = ok & np.isfinite(p)
            if k.sum() > 3:
                rec.append({"target": t, "model": m, "mae": mae(D.y(t)[k], p[k]), "r2": r2(D.y(t)[k], p[k]),
                            "r2_val": r2_val(D, m, t), "n": int(k.sum())})
    return pd.DataFrame(rec)


def compare_table(D: Data, B: int = 400) -> pd.DataFrame:
    """probe (ALL and best layer) vs each baseline: paired delta MAE / R2 with CI over problems."""
    rows = []
    g = D.df["test"]["sample"].values
    for t in KEY:
        if not D.has(t):
            continue
        ok = D.ok(t)
        y = D.y(t)[ok]
        grp = g[ok]
        probes = [("ALL", "ALL")] if "ALL" in D.models else []
        probes.append((best_layer(D, t), "best layer"))
        for pm, tag in probes:
            pp = D.p(pm, t)[ok]
            est = boot_reg(y, pp, grp, B)
            for bm in ("B0", "B1", "B3_mlp", "B4"):
                if bm not in D.models:
                    continue
                d = boot_paired(y, pp, D.p(bm, t)[ok], grp, B)
                v = ("probe better" if d["delta_mae"]["hi"] < 0 else "probe WORSE" if d["delta_mae"]["lo"] > 0 else "no clear difference")
                rows.append({"target": t, "probe": f"{pm} ({tag})", "vs": bm, "probe_MAE": ci(est["mae"]), "probe_R2": ci(est["r2"]),
                             "dMAE (probe-base)": ci(d["delta_mae"]), "dR2": ci(d["delta_r2"]), "verdict": v})
    return pd.DataFrame(rows)


def within_trace(D: Data, B: int = 300) -> pd.DataFrame:
    rows = []
    g = D.df["test"]["sample"].values
    for t in ("a", "f", "d_gold", "r_pgold"):
        if not D.has(t):
            continue
        ok = D.ok(t)
        pm = "ALL" if "ALL" in D.models else best_layer(D, t)
        va, ca = within_group_spearman(D.y(t)[ok], D.p(pm, t)[ok], g[ok])
        for bm in ("B1", "B4"):
            vb, cb = within_group_spearman(D.y(t)[ok], D.p(bm, t)[ok], g[ok])
            d = boot_within(ca, cb, B)
            rows.append({"target": t, "probe": pm, "within-trace Spearman probe": va, "baseline": bm,
                         "within-trace Spearman base": vb, "diff [95% CI]": ci(d), "traces": d["n_groups"]})
    return pd.DataFrame(rows)


def before_text(D: Data, B: int = 200) -> pd.DataFrame:
    """Rows where the written prefix does not yet contain the answer.  Can the probe tell what the model will say anyway?"""
    df = D.df["test"]
    rows = []
    pm = "ALL" if "ALL" in D.models else best_layer(D, "a")
    g = df["sample"].values
    sub = {
        "gold not yet written (k>=1)": (df["gold_in"].values == 0) & (df["k"].values >= 1),
        "gold not yet written, CORRECT traces": (df["gold_in"].values == 0) & (df["k"].values >= 1) & (df["correct"].values == 1),
    }
    ya = D.y("a")
    for name, m in sub.items():
        m = m & np.isfinite(ya)
        if m.sum() < 50:
            continue
        lab = ya[m] > LOG_HALF
        if lab.sum() < 5 or (~lab).sum() < 5:
            continue
        for bm in ("B1", "B4"):
            d = boot_auroc(lab, D.p(pm, "a")[m], g[m], B, s2=D.p(bm, "a")[m])
            rows.append({"subset": name, "n rows": int(m.sum()), "label": "model already puts >50% on gold (a > log .5)",
                         "positive rate": float(lab.mean()), "AUROC probe": auroc(lab, D.p(pm, "a")[m]),
                         "AUROC baseline": auroc(lab, D.p(bm, "a")[m]), "baseline": bm, "AUROC diff [95% CI]": ci(d)})
    if D.has("ff"):
        yf = D.y("ff")
        m = (df["final_in"].values == 0) & (df["k"].values >= 1) & np.isfinite(yf)
        if m.sum() >= 50 and 5 <= yf[m].sum() <= m.sum() - 5:
            for bm in ("B1", "B4"):
                d = boot_auroc(yf[m] > 0.5, D.p(pm, "f")[m], g[m], B, s2=D.p(bm, "f")[m])
                rows.append({"subset": "own final answer not yet written (k>=1)", "n rows": int(m.sum()),
                             "label": "forced answer already equals own final answer", "positive rate": float((yf[m] > 0.5).mean()),
                             "AUROC probe": auroc(yf[m] > 0.5, D.p(pm, "f")[m]),
                             "AUROC baseline": auroc(yf[m] > 0.5, D.p(bm, "f")[m]), "baseline": bm, "AUROC diff [95% CI]": ci(d)})
    return pd.DataFrame(rows)


def monitors(D: Data, B: int = 200) -> pd.DataFrame:
    """Condition monitors: can a probe flag the condition at a checkpoint, before the text shows it?"""
    df = D.df["test"]
    g = df["sample"].values
    pm = "ALL" if "ALL" in D.models else best_layer(D, "a")
    rows = []

    def add(name, mask, lab, score_name, t, higher_is_positive=True):
        m = mask & np.isfinite(lab.astype(float))
        if m.sum() < 50 or lab[m].sum() < 5 or (~lab[m].astype(bool)).sum() < 5:
            return
        sgn = 1.0 if higher_is_positive else -1.0
        for bm in ("B1", "B4"):
            d = boot_auroc(lab[m].astype(bool), sgn * D.p(pm, t)[m], g[m], B, s2=sgn * D.p(bm, t)[m])
            rows.append({"monitor": name, "score": score_name, "n": int(m.sum()), "positive rate": float(lab[m].mean()),
                         "AUROC probe": auroc(lab[m].astype(bool), sgn * D.p(pm, t)[m]),
                         "AUROC base": auroc(lab[m].astype(bool), sgn * D.p(bm, t)[m]), "base": bm, "diff [95% CI]": ci(d)})

    cor = (df["correct"].values == 1) & (df["k"].values >= 1)
    wrong = (df["correct"].values == 0) & (df["k"].values >= 1) & (df["final"].fillna("").astype(str).values != "")
    if D.has("d_gold"):
        add("overthinking: has the answer already converged (correct traces)", cor, df["conv_gold"].values, "pred d_gold", "d_gold")
    add("overthinking: has the answer already converged (correct traces)", cor, df["conv_gold"].values, "pred a", "a")
    if D.has("d_commit"):
        add("rationalisation: committed to own wrong answer (wrong traces)", wrong, df["commit"].values, "pred d_commit", "d_commit")
    add("rationalisation: committed to own wrong answer (wrong traces)", wrong, df["commit"].values, "pred f", "f")
    # trace-level: will this trace end wrong?  score = -pred a at the k-th checkpoint
    for k in (1, 2, 3):
        m = df["k"].values == k
        lab = (df["correct"].values == 0)
        add(f"will the trace end WRONG (at checkpoint k={k})", m, lab, "-pred a", "a", higher_is_positive=False)
    kmax = df.groupby("sample")["k"].transform("max").values
    last_nf = (df["k"].values == kmax - 1) & (kmax >= 2)
    add("will the trace end WRONG (last checkpoint before the end)", last_nf, (df["correct"].values == 0), "-pred a", "a", False)
    return pd.DataFrame(rows)


def early_exit(D: Data, thetas=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99)) -> pd.DataFrame:
    """Stop at the first checkpoint whose (predicted) P(gold) >= theta.  Tokens saved vs accuracy of the forced answer there."""
    df = D.df["test"].copy()
    df["_i"] = np.arange(len(df))
    tr = D.traces["eval"]
    tr = tr[tr["sample"].isin(set(df["sample"]))]
    full_acc = float(tr["correct"].mean())
    n_tr = len(tr)
    tot_tok = float(tr["n_tok"].sum())
    cor = tr[(tr["correct"] == 1) & (tr["slack_gold"] >= 0)]
    orc = float(cor["slack_gold"].sum()) / tot_tok
    rows = [{"policy": "never exit (full trace)", "theta": np.nan, "tokens saved": 0.0, "accuracy": full_acc, "d accuracy": 0.0, "exits": 0.0},
            {"policy": "ORACLE: exit at t* (needs the gold answer)", "theta": np.nan, "tokens saved": orc, "accuracy": full_acc,
             "d accuracy": 0.0, "exits": len(cor) / n_tr}]
    srcs = [("ALL" if "ALL" in D.models else best_layer(D, "a"), "probe"), ("B4", "text baseline"), ("B1", "position baseline"),
            ("TRUE", "true forced-answer score")]
    srcs = [(m, n) for m, n in srcs if m in D.models or m == "TRUE"]
    for th in thetas:
        for m, name in srcs:
            pa = np.exp(df["a"].astype(float).values) if m == "TRUE" else np.exp(D.p(m, "a"))
            pa = np.nan_to_num(pa, nan=0.0)
            df["_p"] = pa
            sel = df[(df["kind"] != "final") & (df["k"] >= 1) & (df["_p"] >= th)]
            first = sel.sort_values(["sample", "k"]).groupby("sample").head(1)
            saved = float(first["rem"].sum()) / tot_tok
            exit_ok = first["eq_gold"].values.astype(float)
            fin = tr.set_index("sample")["correct"]
            acc_ex = fin.copy().astype(float)
            acc_ex.loc[first["sample"].values] = exit_ok
            rows.append({"policy": f"exit when pred P(gold) >= theta [{name}: {m}]", "theta": th, "tokens saved": saved,
                         "accuracy": float(acc_ex.mean()), "d accuracy": float(acc_ex.mean()) - full_acc,
                         "exits": len(first) / n_tr})
    return pd.DataFrame(rows)


def rollout_validation(D: Data) -> pd.DataFrame:
    if not D.has("r_pgold"):
        return pd.DataFrame()
    y = D.y("r_pgold")
    a_true = D.y("a")
    ok = np.isfinite(y) & np.isfinite(a_true)
    rows = [{"check": "Spearman(forced-answer a, rollout P(gold))", "value": spearman(a_true[ok], y[ok]), "n": int(ok.sum())}]
    m = ok & (y > 0.99) | ok & (y < 0.01)
    if m.sum() > 20:
        rows.append({"check": "AUROC of a for 'all rollouts end on gold' vs 'none do'", "value": auroc(y[m] > 0.5, a_true[m]), "n": int(m.sum())})
    pm = "ALL" if "ALL" in D.models else best_layer(D, "a")
    ok2 = np.isfinite(y) & np.isfinite(D.p(pm, "a"))
    rows.append({"check": f"Spearman(probe-predicted a [{pm}], rollout P(gold))", "value": spearman(D.p(pm, "a")[ok2], y[ok2]), "n": int(ok2.sum())})
    return pd.DataFrame(rows)


def prevalence(D: Data) -> pd.DataFrame:
    rows = []
    for sp, tr in D.traces.items():
        r = {"split": sp, "traces": len(tr), "accuracy": tr["correct"].mean()}
        for c in ("cond_overthink", "cond_rationalize", "cond_lost", "cond_dither"):
            if c in tr:
                r[c[5:]] = tr[c].mean()
        cor = tr[(tr["correct"] == 1) & (tr["slack_gold"] >= 0)]
        r["median slack (correct, tokens)"] = float(cor["slack_gold"].median()) if len(cor) else np.nan
        r["share of tokens after convergence"] = float((cor["slack_gold"] / cor["n_tok"]).mean()) if len(cor) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def make_plots(D: Data, outdir: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    made = []
    lc = layer_curves(D)
    ts = [t for t in ("a", "f", "rem", "d_gold", "d_commit", "fg") if D.has(t)]
    if D.layers and ts:
        fig, axes = plt.subplots(1, len(ts), figsize=(3.4 * len(ts), 3.2), squeeze=False)
        for ax, t in zip(axes[0], ts):
            s = lc[lc.target == t].set_index("model")["r2"]
            ax.plot(D.layers, [s.get(f"L{l}", np.nan) for l in D.layers], "o-", ms=3, label="layer probe")
            for m, ls in (("ALL", "-"), ("B4", "--"), ("B1", ":")):
                if m in s.index:
                    ax.axhline(s[m], ls=ls, color={"ALL": "C3", "B4": "C2", "B1": "C7"}[m], label=m)
            ax.set_title(f"{t}"), ax.set_xlabel("layer"), ax.set_ylabel("test R2")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        p = outdir / "layer_curves.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p.name)
    df = D.df["test"]
    pm = "ALL" if "ALL" in D.models else (best_layer(D, "a") if D.layers else None)
    if pm:
        tr = D.traces["eval"]
        picks = []
        for c in ("cond_overthink", "cond_rationalize", "cond_lost", "cond_dither"):
            if c in tr:
                picks += tr.loc[tr[c] == 1, "sample"].head(2).tolist()
        picks = list(dict.fromkeys(picks))[:8]
        if picks:
            fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), squeeze=False)
            for ax, sid in zip(axes.ravel(), picks):
                m = (df["sample"].values == sid)
                sub = df[m]
                ax.plot(sub["pos"] + 1, np.exp(sub["a"].astype(float)), "k-o", ms=3, label="true P(gold)")
                ax.plot(sub["pos"] + 1, np.exp(D.p(pm, "a")[m]), "C3--o", ms=3, label=f"probe {pm}")
                if (sub["f"].astype(float) > -50).any():
                    ax.plot(sub["pos"] + 1, np.exp(sub["f"].astype(float)), "C0:", label="true P(own final)")
                ax.set_title(f"sample {sid} ({'ok' if sub['correct'].iloc[0] else 'wrong'})", fontsize=8)
                ax.set_ylim(-0.03, 1.03)
            axes[0][0].legend(fontsize=6)
            fig.tight_layout()
            p = outdir / "trajectories.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            made.append(p.name)
    return made
