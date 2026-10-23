"""Stage 6 (GPU, minutes): does the probe read the REASONING STATE, or just 'how hard / solvable is this question'?

Two controls the main report cannot give:
  Q0   a linear probe whose input is the hidden state at the PROMPT END of the same trace (k=0: sees only the question), used to
       predict every later checkpoint's targets.  If Q0 already explains most of the R2, the signal is problem difficulty.
  Bq   B4 + a hashed bag-of-words of the QUESTION text (the text rival B4 never sees the question).
Then stacking on the validation split (3-5 coefficients): do the all-layer probe's predictions add anything on top of
[Q0 + position], [+ B4], [+ Bq / B5]?  Reported as test R2 / MAE with a paired bootstrap over problems.

    python cluster/sig_controls.py --run-dir $RUN_DIR [--layers 16,20,22] [--device cuda]
Needs the outputs of sig_probes.py.  Writes <out>/analysis/controls.md and controls.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import analysis as an  # noqa: E402
from miol.signals import io as sio  # noqa: E402
from miol.signals.baselines import feature_matrix  # noqa: E402
from miol.signals.metrics import boot_paired, mae, r2  # noqa: E402
from miol.signals.ridge import fit_layer, pick_ratio  # noqa: E402
from miol.signals.targets import build_targets  # noqa: E402
from miol.signals.text_utils import hash_bow, question_from_prompt  # noqa: E402

TARGETS = ["a", "f", "fg", "ff", "rem", "d_gold", "d_commit", "a_next", "da", "ent_next", "hedge_next", "r_pgold"]


def question_bow(run_dir, split, samples, dim=2048):
    out = {}
    for s in sorted(set(samples)):
        meta = json.load(open(sio.hs_dir(run_dir, split) / f"sample_{int(s):06d}.json"))
        q = question_from_prompt(meta["prompt"]) or ""
        v = np.zeros(dim, np.float32)
        v[hash_bow(q, dim)] = 1.0
        out[s] = v
    return np.stack([out[s] for s in samples])


def ols_stack(P_val, y_val, P_test):
    """least squares with intercept on validation rows; returns test predictions."""
    m = np.isfinite(y_val) & np.isfinite(P_val).all(1)
    A = np.column_stack([np.ones(m.sum()), P_val[m]])
    w, *_ = np.linalg.lstsq(A, y_val[m], rcond=None)
    ok = np.isfinite(P_test).all(1)
    out = np.full(len(P_test), np.nan)
    out[ok] = np.column_stack([np.ones(ok.sum()), P_test[ok]]) @ w
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", default="16,20,22")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--split-source", default="files", choices=["files", "labels"])
    ap.add_argument("--no-bq", action="store_true", help="skip the internal B4+question baseline (MATH run: B4 already holds the question)")
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    log = sio.log_to("[controls]")
    out = sio.out_dir(a.run_dir, a.out)
    adir = out / "analysis"
    adir.mkdir(exist_ok=True)
    D = an.Data(out)
    names = D.names
    T = len(names)
    dev = torch.device(a.device)

    tr, ev = sio.read_table(out / "labels_train.csv"), sio.read_table(out / "labels_eval.csv")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sig_probes as sp
    val_ids = sp.split_samples(a.run_dir, a.val_fraction, a.split_seed,
                               ids=sorted(int(x) for x in tr["sample"].unique()) if a.split_source == "labels" else None)
    is_val = tr["sample"].isin(val_ids).values
    full = pd.concat([tr, ev], ignore_index=True)
    Yall, _ = build_targets(full, names=names)
    n_tr = len(tr)
    parts = {"fit": np.arange(n_tr)[~is_val], "val": np.arange(n_tr)[is_val], "test": np.arange(n_tr, len(full))}
    dfs = {k: full.iloc[v].reset_index(drop=True) for k, v in parts.items()}
    Y = {k: Yall[v] for k, v in parts.items()}
    split_of = {"fit": "train", "val": "train", "test": "eval"}
    assert np.array_equal(dfs["test"]["row"].values, D.df["test"]["row"].values), "row order differs from the probes stage"

    # row index of each trace's prompt-end checkpoint (k=0), in its own split's states array
    k0 = {s: d[d.k == 0].set_index("sample")["row"] for s, d in (("train", tr), ("eval", ev))}
    S = {"train": np.load(out / "states_train.npy", mmap_mode="r"), "eval": np.load(out / "states_eval.npy", mmap_mode="r")}
    layers = [int(x) for x in a.layers.split(",")]
    P = {"val": {}, "test": {}}

    # ---- Q0: prompt-end state of the same trace ---------------------------------------------------------------
    def load0(k, l):
        rows = k0[split_of[k]].loc[dfs[k]["sample"].values].values
        order = np.argsort(rows, kind="stable")
        X = np.empty((len(rows), S[split_of[k]].shape[2]), np.float32)
        X[order] = np.asarray(S[split_of[k]][l][rows[order]], dtype=np.float32)
        return torch.from_numpy(X).to(dev)

    for l in layers:
        lf = fit_layer(load0("fit", l), Y["fit"])
        Xv, Xt = load0("val", l), load0("test", l)
        pv, pt = np.zeros((len(Xv), T), np.float32), np.zeros((len(Xt), T), np.float32)
        for j in range(T):
            cv = lf.predict(Xv, j)
            r = pick_ratio(cv, Y["val"][:, j])
            pv[:, j], pt[:, j] = cv[:, r].cpu().numpy(), lf.predict(Xt, j)[:, r].cpu().numpy()
        P["val"][f"Q0_L{l}"], P["test"][f"Q0_L{l}"] = sp.postprocess(pv, names), sp.postprocess(pt, names)
        log(f"Q0 layer {l} done")

    # ---- Bq: B4 + question bag-of-words (only where B4 does not already hold the question) ------------------------
    use_bq = not a.no_bq and "qbow" not in tr.columns
    X5 = {}
    if use_bq:
        for k in ("fit", "val", "test"):
            qb = question_bow(a.run_dir, split_of[k], dfs[k]["sample"].values)
            X5[k] = torch.tensor(np.column_stack([feature_matrix(dfs[k], "B4"), qb]), device=dev)
        lf = fit_layer(X5["fit"], Y["fit"])
        pv, pt = np.zeros((len(dfs["val"]), T), np.float32), np.zeros((len(dfs["test"]), T), np.float32)
        for j in range(T):
            cv = lf.predict(X5["val"], j)
            r = pick_ratio(cv, Y["val"][:, j])
            pv[:, j], pt[:, j] = cv[:, r].cpu().numpy(), lf.predict(X5["test"], j)[:, r].cpu().numpy()
        P["val"]["Bq"], P["test"]["Bq"] = sp.postprocess(pv, names), sp.postprocess(pt, names)
        log("Bq done")
    txtq = "Bq" if use_bq else ("B5" if "B5" in D.models else "B4")      # the strongest text(+question) rival available

    # ---- table ---------------------------------------------------------------------------------------------------
    g = dfs["test"]["sample"].values
    rows = []
    for t in TARGETS:
        if t not in names:
            continue
        j = names.index(t)
        yt, yv = Y["test"][:, j], Y["val"][:, j]
        ok = np.isfinite(yt)
        if ok.sum() < a.min_rows:
            continue

        def col(model, s):
            src = P[s] if model.startswith(("Q0", "Bq")) else D.P[s]
            return src[model][:, j]
        qbest = max((f"Q0_L{l}" for l in layers), key=lambda m: r2(yv[np.isfinite(yv)], col(m, "val")[np.isfinite(yv)]))
        pos = "B1"
        probe = "ALL" if "ALL" in D.models else an.best_layer(D, t)

        def stack(ms):
            pv_ = np.column_stack([col(m, "val") for m in ms])
            pt_ = np.column_stack([col(m, "test") for m in ms])
            return ols_stack(pv_, yv, pt_)

        preds = {f"probe {probe}": col(probe, "test"), f"Q0 only ({qbest})": col(qbest, "test"), "B1 position": col("B1", "test"),
                 "B4 text": col("B4", "test"), f"{txtq} text+question": col(txtq, "test"),
                 "stack Q0+B1": stack([qbest, pos]), f"stack Q0+B1+{txtq}": stack([qbest, pos, txtq]),
                 f"stack Q0+B1+{txtq}+probe": stack([qbest, pos, txtq, probe]), "stack Q0+B1+probe": stack([qbest, pos, probe])}
        for name, p in preds.items():
            k = ok & np.isfinite(p)
            rows.append({"target": t, "model": name, "R2": r2(yt[k], p[k]), "MAE": mae(yt[k], p[k]), "n": int(k.sum())})
        for base, withp in (("stack Q0+B1", "stack Q0+B1+probe"), (f"stack Q0+B1+{txtq}", f"stack Q0+B1+{txtq}+probe")):
            k = ok & np.isfinite(preds[base]) & np.isfinite(preds[withp])
            d = boot_paired(yt[k], preds[withp][k], preds[base][k], g[k], a.boot)
            rows.append({"target": t, "model": f"ADDED VALUE of probe over [{base}]", "R2": d["delta_r2"]["est"], "MAE": d["delta_mae"]["est"],
                         "n": int(k.sum()), "dR2 95% CI": f"[{d['delta_r2']['lo']:.3f}, {d['delta_r2']['hi']:.3f}]",
                         "dMAE 95% CI": f"[{d['delta_mae']['lo']:.3f}, {d['delta_mae']['hi']:.3f}]"})
    df = pd.DataFrame(rows)
    df.to_csv(adir / "controls.csv", index=False)
    txt = ["# Controls: question-only probe (Q0) and text+question baseline\n",
           "Q0 = ridge on the PROMPT-END hidden state of the same trace (it has seen only the question). "
           "Bq/B5 = text baseline that also sees the question. 'stack' = least squares on the validation split.\n",
           "Read the ADDED VALUE rows: dR2 > 0 with a CI above 0 means the all-layer probe carries information about the reasoning state that "
           "neither the question nor the written text provides. If it is ~0, the earlier gain over B4 was problem difficulty.\n",
           an.md(df.fillna(""))]
    (adir / "controls.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
