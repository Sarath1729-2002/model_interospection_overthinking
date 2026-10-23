"""The baseline ladder, each rung blocking one 'boring' explanation of a probe's score.

  B0  constant                        the train mean
  B1  position                        how far into the trace we are (tokens, checkpoint index)
  B2  + prompt length                 harder / longer problems
  B3  + is it already written down?   gold / own answer appears among the numbers so far, the last number equals it,
                                      counts of numbers, '=', lines and hedge words           (ridge, and a small MLP)
  B4  + recent wording                hashed bag-of-words of the last ~200 characters         (the strongest text rival)

Every rung uses ONLY the written prefix (plus the labels the target itself is defined with: for the 'is it already
written down' flags that is an oracle-style upper bound on what the text can tell).
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from miol.signals.ridge import RATIOS, fit_layer, pick_ratio
from miol.signals.text_utils import hash_bow

LEVELS = ["B0", "B1", "B2", "B3", "B3_mlp", "B4"]
LEVELS_MATH = LEVELS + ["B5", "B5_mlp"]
ENT_COLS = ["ent_here", "ent_past", "ent_last", "surp_last", "surp_past"]


def default_levels(df: pd.DataFrame) -> List[str]:
    return LEVELS_MATH if "pbow" in df.columns else LEVELS


def _bow_dense(series: pd.Series, dim: int) -> np.ndarray:
    """'12 55 901' strings of hashed-word indices -> dense 0/1 matrix."""
    out = np.zeros((len(series), dim), dtype=np.float32)
    for i, t in enumerate(series.fillna("").astype(str).values):
        if t:
            out[i, np.array(t.split(), dtype=np.int64)] = 1.0
    return out


def feature_matrix(df: pd.DataFrame, level: str, bow_dim: int = 2048) -> np.ndarray:
    pos1 = df["pos"].values.astype(float) + 1.0
    k = df["k"].values.astype(float)
    kcap = np.minimum(k, 12).astype(int)
    onehot = np.eye(13)[kcap]
    cols = [pos1 / 100.0, np.sqrt(pos1) / 10.0, np.log1p(pos1), k, (k == 0).astype(float)]
    X = np.column_stack(cols + [onehot])
    if level in ("B2", "B3", "B3_mlp", "B4", "B5", "B5_mlp"):
        X = np.column_stack([X, df["prompt_len"].values.astype(float) / 100.0])
    if level in ("B3", "B3_mlp", "B4", "B5", "B5_mlp"):
        txt = np.column_stack([df["gold_in"], df["final_in"], df["last_eq_gold"], df["last_eq_final"],
                               df["n_numbers"] / 10.0, df["n_eq"] / 10.0, df["n_lines"] / 10.0, df["n_hedge"],
                               df["has_boxed"]]).astype(float)
        X = np.column_stack([X, txt])
    if level in ("B4", "B5", "B5_mlp"):
        bow = np.zeros((len(df), bow_dim), dtype=np.float32)
        for i, t in enumerate(df["tail"].fillna("").astype(str).values):
            for j in hash_bow(t, bow_dim):
                bow[i, j] = 1.0
        parts = [X, bow]
        if "qbow" in df.columns:                                       # the QUESTION text (B4 of the first GSM8K report lacked it)
            parts.append(_bow_dense(df["qbow"], bow_dim))
        if level in ("B5", "B5_mlp") and "pbow" in df.columns:
            parts.append(_bow_dense(df["pbow"], bow_dim))              # the WHOLE written prefix, as a bag of words
            parts.append(np.column_stack([df[c].astype(float).fillna(0.0).values for c in ENT_COLS if c in df.columns]))   # logit statistics
        X = np.column_stack(parts)
    return X.astype(np.float32)


def fit_mlp_multi(Xf, Yf, Xv, Yv, Xt, hidden=(128, 128), epochs=60, lr=2e-3, wd=1e-4, batch=512, patience=8,
                  device="cpu", seed=0, log=None):
    """Multi-target MLP with a NaN-masked MSE on z-scored targets; early stopping on validation. -> (pred_val, pred_test)."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    mu, sd = np.nanmean(Yf, 0), np.nanstd(Yf, 0) + 1e-8
    zf, zv = (Yf - mu) / sd, (Yv - mu) / sd
    xm, xs = Xf.mean(0), Xf.std(0)
    xs[xs < 1e-5] = 1.0
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    Xf_, Xv_, Xt_ = T((Xf - xm) / xs), T((Xv - xm) / xs), T((Xt - xm) / xs)
    Yf_, Yv_ = T(np.nan_to_num(zf)), T(np.nan_to_num(zv))
    Mf_, Mv_ = T(~np.isnan(zf)), T(~np.isnan(zv))
    layers, d = [], Xf.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.GELU()]
        d = h
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, Yf.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = float("inf"), None, 0
    n = Xf_.shape[0]
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            loss = (((net(Xf_[b]) - Yf_[b]) ** 2) * Mf_[b]).sum() / Mf_[b].sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float((((net(Xv_) - Yv_) ** 2) * Mv_).sum() / Mv_.sum().clamp_min(1))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pv, pt = net(Xv_).cpu().numpy() * sd + mu, net(Xt_).cpu().numpy() * sd + mu
    if log:
        log(f"  mlp: best val z-MSE {best:.4f} after {ep + 1} epochs")
    return pv, pt


def run_baselines(dfs: Dict[str, pd.DataFrame], Y: Dict[str, np.ndarray], device="cpu", levels: Optional[List[str]] = None,
                  log=print) -> Dict[str, Dict[str, np.ndarray]]:
    """dfs/Y keyed 'fit','val','test'.  -> {level: {'val': [nv,T], 'test': [nt,T]}}"""
    out = {}
    T = Y["fit"].shape[1]
    levels = levels if levels is not None else default_levels(dfs["fit"])
    cache = {}
    for lv in levels:
        if lv == "B0":
            m = np.nanmean(Y["fit"], 0)
            out[lv] = {s: np.tile(m, (len(dfs[s]), 1)) for s in ("val", "test")}
            continue
        base = {"B3_mlp": "B3", "B5_mlp": "B5"}.get(lv, lv)
        if base not in cache:
            cache.clear()
            cache[base] = {s: feature_matrix(dfs[s], base) for s in ("fit", "val", "test")}
        X = cache[base]
        if lv.endswith("_mlp"):
            pv, pt = fit_mlp_multi(X["fit"], Y["fit"], X["val"], Y["val"], X["test"], device=device, log=log)
            out[lv] = {"val": pv, "test": pt}
            continue
        dev = torch.device(device)
        t = {s: torch.tensor(X[s], device=dev) for s in X}
        lf = fit_layer(t["fit"], Y["fit"])
        pv = np.zeros((len(X["val"]), T), np.float32)
        pt = np.zeros((len(X["test"]), T), np.float32)
        for j in range(T):
            rv = lf.predict(t["val"], j)
            r = pick_ratio(rv, Y["val"][:, j])
            pv[:, j] = rv[:, r].cpu().numpy()
            pt[:, j] = lf.predict(t["test"], j)[:, r].cpu().numpy()
        out[lv] = {"val": pv, "test": pt}
        log(f"  baseline {lv}: features {X['fit'].shape[1]}")
    return out
