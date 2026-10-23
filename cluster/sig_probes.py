"""Stage 4 (GPU, one process): fit every linear probe + baseline for every target, closed form (ridge), validation-tuned.

    python cluster/sig_probes.py --run-dir $RUN_DIR [--layers all] [--kernel-max-fit 60000]

Inputs  (<out> = <run>/signals):  labels_train.csv labels_eval.csv states_train.npy states_eval.npy
Split   fit/val = the SAME fixed 95/5 split of the train problems as the paper's probes (split_indices(n_files, 0.05, 0));
        test = the eval split (never used for any choice).
Models  B0 B1 B2 B3 B3_mlp B4  (boring explanations)   L0..L28  (one linear probe per layer)   ALL (all layers concatenated, exact
        dual ridge on the 29x3584 features)
Output  <out>/probes/preds.npz   keys  val|test/<model>  [n,T]   row_val row_test   (+ meta.json, scores_test.csv)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.data.hidden_states_dataset import split_indices  # noqa: E402
from miol.signals import io as sio  # noqa: E402
from miol.signals.baselines import default_levels, run_baselines  # noqa: E402
from miol.signals.metrics import mae, r2  # noqa: E402
from miol.signals.ridge import RATIOS, fit_layer, kernel_allcat, pick_ratio  # noqa: E402
from miol.signals.targets import CLIP, build_targets  # noqa: E402


UNIT = {"fg", "ff", "r_pgold", "r_pfinal", "pct", "loss_next", "gain_next", "loss_end", "y_end", "p1max"}
NONNEG = {"rem", "r_H", "ent1", "ent_next", "surp_next", "ent_rem", "hedge_next", "hedge_rem", "flips_rem", "steps_rem", "len_next"}


def postprocess(P: np.ndarray, names):
    """Clip predictions to each target's feasible range (applied identically to every model)."""
    P = P.copy()
    for j, n in enumerate(names):
        if n in ("a", "f", "a_next"):
            P[:, j] = np.clip(P[:, j], CLIP, 0.0)
        elif n in UNIT:
            P[:, j] = np.clip(P[:, j], 0.0, 1.0)
        elif n in NONNEG:
            P[:, j] = np.maximum(P[:, j], 0.0)
        elif n == "da":
            P[:, j] = np.clip(P[:, j], -10.0, 10.0)
    return P


def split_samples(run_dir, val_fraction, split_seed, ids=None):
    """Validation problems = the fixed split_indices() split of the train problems (same rule as the paper's probes).
    ids=None: the cached safetensors files of the train split (GSM8K run); else an explicit sorted list of sample ids (MATH run)."""
    if ids is None:
        files = sorted(sio.hs_dir(run_dir, "train").glob("sample_*.safetensors"))
        ids = [int(f.stem.split("_")[1]) for f in files]
    _, va = split_indices(len(ids), val_fraction, split_seed)
    return {ids[i] for i in va}


def cap_rows(samples: np.ndarray, cap: int, seed: int):
    """indices into the fit rows, whole traces at random, until `cap` rows (cap<=0 or fewer rows -> all)."""
    n = len(samples)
    if cap <= 0 or n <= cap:
        return np.arange(n)
    u = np.unique(samples)
    rng = np.random.RandomState(seed)
    keep, tot = [], 0
    for s in u[rng.permutation(len(u))]:
        idx = np.where(samples == s)[0]
        if tot + len(idx) > cap:
            continue
        keep.append(idx)
        tot += len(idx)
    return np.sort(np.concatenate(keep))


def pick_and_predict(cands, yv):
    """cands: list of {'ratio','val','test'} -> (val_pred, test_pred, ratio) at the lowest validation MSE."""
    m = ~np.isnan(yv)
    best, bi = float("inf"), 0
    for i, c in enumerate(cands):
        e = float(np.mean((c["val"][m] - yv[m]) ** 2)) if m.any() else 0.0
        if e < best:
            best, bi = e, i
    return cands[bi]["val"], cands[bi]["test"], cands[bi]["ratio"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--kernel-max-fit", type=int, default=60000)
    ap.add_argument("--no-allcat", action="store_true")
    ap.add_argument("--no-rollouts", action="store_true")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--split-source", default="files", choices=["files", "labels"], help="labels: validation split drawn from the label table (MATH run)")
    ap.add_argument("--save-weights", action="store_true", help="save the per-layer probe weights (for the transfer test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    log = sio.log_to("[probes]")
    out = sio.out_dir(a.run_dir, a.out)
    pdir = out / "probes"
    pdir.mkdir(exist_ok=True)
    dev = torch.device(a.device)
    t0 = time.time()

    tr, ev = sio.read_table(out / "labels_train.csv"), sio.read_table(out / "labels_eval.csv")
    val_ids = split_samples(a.run_dir, a.val_fraction, a.split_seed,
                            ids=sorted(int(x) for x in tr["sample"].unique()) if a.split_source == "labels" else None)
    is_val = tr["sample"].isin(val_ids).values
    full = pd.concat([tr, ev], ignore_index=True)
    Yall, names = build_targets(full, include_rollouts=not a.no_rollouts)
    T = len(names)
    n_tr = len(tr)
    parts = {"fit": np.arange(n_tr)[~is_val], "val": np.arange(n_tr)[is_val], "test": np.arange(n_tr, len(full))}
    dfs = {k: full.iloc[v].reset_index(drop=True) for k, v in parts.items()}
    Y = {k: Yall[v] for k, v in parts.items()}
    log(f"rows fit/val/test = {len(parts['fit'])}/{len(parts['val'])}/{len(parts['test'])}   targets {names}")
    log("finite targets (fit): " + " ".join(f"{n}={int(np.isfinite(Y['fit'][:, j]).sum())}" for j, n in enumerate(names)))

    preds = {"val": {}, "test": {}}
    chosen = {}
    weights = {}

    # ---- baselines ----------------------------------------------------------------------------------------
    levels = [l for l in default_levels(dfs["fit"]) if not (a.no_mlp and l.endswith("_mlp"))]
    bl = run_baselines(dfs, Y, a.device, levels, log)
    for lv in levels:
        for s in ("val", "test"):
            preds[s][lv] = postprocess(bl[lv][s], names)
    log(f"baselines done ({(time.time() - t0) / 60:.1f} min)")

    # ---- states -------------------------------------------------------------------------------------------
    S = {"train": np.load(out / "states_train.npy", mmap_mode="r"), "eval": np.load(out / "states_eval.npy", mmap_mode="r")}
    n_layers = S["train"].shape[0]
    layers = list(range(n_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    rows = {"fit": tr["row"].values[parts["fit"]], "val": tr["row"].values[parts["val"] ],
            "test": ev["row"].values[parts["test"] - n_tr]}
    src = {"fit": "train", "val": "train", "test": "eval"}

    def load(split_key, l, sel=None):
        r = rows[split_key] if sel is None else rows[split_key][sel]
        return torch.from_numpy(np.ascontiguousarray(S[src[split_key]][l][r])).to(dev).float()

    # ---- per-layer ridge ----------------------------------------------------------------------------------
    for l in layers:
        Xf = load("fit", l)
        lf = fit_layer(Xf, Y["fit"])
        del Xf
        Xv, Xt = load("val", l), load("test", l)
        pv = np.zeros((len(Xv), T), np.float32)
        pt = np.zeros((len(Xt), T), np.float32)
        rr = []
        for j in range(T):
            cand_v = lf.predict(Xv, j)
            r = pick_ratio(cand_v, Y["val"][:, j])
            pv[:, j] = cand_v[:, r].cpu().numpy()
            pt[:, j] = lf.predict(Xt, j)[:, r].cpu().numpy()
            rr.append(RATIOS[r])
        preds["val"][f"L{l}"], preds["test"][f"L{l}"] = postprocess(pv, names), postprocess(pt, names)
        chosen[f"L{l}"] = dict(zip(names, rr))
        if a.save_weights:
            ridx = [RATIOS.index(r_) for r_ in rr]
            weights[l] = {"mu": lf.mu.cpu(), "sd": lf.sd.cpu(),
                          "W": torch.stack([lf.W[j][ridx[j]] for j in range(T)]).cpu(), "b": torch.tensor([float(lf.b[j][ridx[j]]) for j in range(T)])}
        del Xv, Xt, lf
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        ja = names.index("a")
        log(f"layer {l:2d}  val R2(a)={r2(Y['val'][:, ja], preds['val'][f'L{l}'][:, ja]):.3f}  "
            f"test R2(a)={r2(Y['test'][:, ja], preds['test'][f'L{l}'][:, ja]):.3f}   ({(time.time() - t0) / 60:.1f} min)")

    # ---- all layers concatenated (dual ridge) --------------------------------------------------------------
    if not a.no_allcat and len(layers) == n_layers:
        sub = cap_rows(tr["sample"].values[parts["fit"]], a.kernel_max_fit, a.split_seed)
        log(f"all-layer kernel ridge on {len(sub)} of {len(parts['fit'])} fit rows")

        def get_layer(l):
            return {"fit": load("fit", l, sub), "val": load("val", l), "test": load("test", l)}

        cand = kernel_allcat(get_layer, n_layers, Y["fit"][sub], RATIOS, log)
        pv = np.zeros((len(parts["val"]), T), np.float32)
        pt = np.zeros((len(parts["test"]), T), np.float32)
        rr = []
        for j in range(T):
            if not cand[j]:
                pv[:, j] = pt[:, j] = np.nan
                rr.append(float("nan"))
                continue
            v, t_, r = pick_and_predict(cand[j], Y["val"][:, j])
            pv[:, j], pt[:, j] = v, t_
            rr.append(r)
        preds["val"]["ALL"], preds["test"]["ALL"] = postprocess(pv, names), postprocess(pt, names)
        chosen["ALL"] = dict(zip(names, rr))
        log(f"all-layer done ({(time.time() - t0) / 60:.1f} min)")

    # ---- save ---------------------------------------------------------------------------------------------
    arrays = {"row_val": rows["val"], "row_test": rows["test"], "sample_val": dfs["val"]["sample"].values,
              "sample_test": dfs["test"]["sample"].values}
    for s in ("val", "test"):
        for m, P in preds[s].items():
            arrays[f"{s}/{m}"] = P.astype(np.float32)
    np.savez_compressed(pdir / "preds.npz", **arrays)
    if weights:
        torch.save({"targets": names, "layers": weights}, pdir / "weights_layers.pt")
    meta = {"targets": names, "models": list(preds["test"].keys()), "chosen_ratio": chosen,
            "n": {k: int(len(v)) for k, v in parts.items()}, "val_fraction": a.val_fraction, "split_seed": a.split_seed,
            "kernel_max_fit": a.kernel_max_fit, "layers": layers, "minutes": (time.time() - t0) / 60}
    (pdir / "meta.json").write_text(json.dumps(meta, indent=1))
    recs = []
    for m, P in preds["test"].items():
        for j, n in enumerate(names):
            ok = np.isfinite(Y["test"][:, j]) & np.isfinite(P[:, j])
            if ok.sum() > 3:
                recs.append({"model": m, "target": n, "n": int(ok.sum()), "mae": mae(Y["test"][ok, j], P[ok, j]),
                             "r2": r2(Y["test"][ok, j], P[ok, j])})
    pd.DataFrame(recs).to_csv(pdir / "scores_test.csv", index=False)
    log(f"saved {pdir}  total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
