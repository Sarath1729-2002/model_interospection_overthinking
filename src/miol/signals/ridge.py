"""Closed-form ridge probes (torch, GPU if available).

fit_layer     : one hidden layer -> a linear readout per target, for a grid of ridge strengths.  Features are
                standardised per dimension with FIT-row statistics; targets may contain NaN (those rows are ignored
                for that target only).  Solved by eigendecomposition of the D x D covariance, so the whole
                lambda path costs one eigh per (layer, set-of-valid-rows).
kernel_allcat : all layers concatenated (what the paper's headline probe sees) via the linear kernel
                K = sum_l Z_l Z_l^T / (L*D) and a Cholesky solve - exact ridge on 29*3584 dims without forming them.
"""
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

RATIOS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]      # lambda = ratio * n_rows  (standardised features: diag(Z'Z) ~ n)


def standardise_stats(Z: torch.Tensor):
    mu = Z.mean(0)
    sd = Z.std(0)
    sd = torch.where(sd < 1e-5, torch.ones_like(sd), sd)
    return mu, sd


def _eigh(G: torch.Tensor):
    """eigh that survives rank-deficient Gram matrices (duplicate / constant columns): retry with a tiny diagonal jitter
    (1e-10 of the mean diagonal - far below the smallest ridge strength), then on CPU."""
    try:
        return torch.linalg.eigh(G)
    except torch._C._LinAlgError:
        pass
    jit = 1e-10 * float(G.diagonal().mean().clamp_min(1e-12))
    Gj = G + jit * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    try:
        return torch.linalg.eigh(Gj)
    except torch._C._LinAlgError:
        e, Q = torch.linalg.eigh(Gj.cpu())
        return e.to(G.device), Q.to(G.device)


class LayerFit:
    def __init__(self, mu, sd, W, b, ratios):
        self.mu, self.sd, self.W, self.b, self.ratios = mu, sd, W, b, ratios     # W[t]: [R, D], b[t]: [R]

    @torch.no_grad()
    def predict(self, X: torch.Tensor, t: int) -> torch.Tensor:
        """-> [n, R] predictions of target t for every lambda in the grid."""
        Zx = (X - self.mu) / self.sd
        return Zx @ self.W[t].T + self.b[t]


@torch.no_grad()
def fit_layer(Xfit: torch.Tensor, Y: np.ndarray, ratios: Sequence[float] = RATIOS) -> LayerFit:
    """Xfit [n, D] float32 (device); Y [n, T] float with NaN allowed."""
    dev = Xfit.device
    n, D = Xfit.shape
    T = Y.shape[1]
    mu, sd = standardise_stats(Xfit)
    Z = (Xfit - mu) / sd
    G_all = (Z.T @ Z).double()
    S_all = Z.sum(0).double()
    W = {t: torch.zeros(len(ratios), D, device=dev) for t in range(T)}
    b = {t: torch.zeros(len(ratios), device=dev) for t in range(T)}
    sig = {}
    for t in range(T):
        sig.setdefault(np.isnan(Y[:, t]).tobytes(), []).append(t)
    for key, ts in sig.items():
        m = ~np.isnan(Y[:, ts[0]])
        ng = int(m.sum())
        if ng < 10:
            continue
        mt = torch.tensor(m, device=dev)
        if m.all():
            G, S = G_all, S_all
        else:
            Ze = Z[~mt]
            G, S = G_all - (Ze.T @ Ze).double(), S_all - Ze.sum(0).double()
        mu_g = S / ng
        Gc = G - ng * torch.outer(mu_g, mu_g)
        Yg = torch.tensor(Y[m][:, ts], dtype=torch.float64, device=dev)
        ybar = Yg.mean(0)
        B = Z[mt].double().T @ (Yg - ybar)                                    # [D, Tg]
        evals, Q = _eigh(Gc)
        evals = evals.clamp_min(0)
        QB = Q.T @ B
        for r_i, r in enumerate(ratios):
            Wr = Q @ (QB / (evals + r * ng)[:, None])                         # [D, Tg]
            for j, t in enumerate(ts):
                W[t][r_i] = Wr[:, j].float()
                b[t][r_i] = float(ybar[j] - (mu_g * Wr[:, j]).sum())
    return LayerFit(mu, sd, W, b, list(ratios))


def pick_ratio(pred_val: torch.Tensor, y_val: np.ndarray) -> int:
    """index of the lambda with the lowest validation MSE (rows with NaN target ignored)."""
    m = ~np.isnan(y_val)
    if m.sum() == 0:
        return 0
    yv = torch.tensor(y_val[m], dtype=torch.float32, device=pred_val.device)
    mse = ((pred_val[torch.tensor(m, device=pred_val.device)] - yv[:, None]) ** 2).mean(0)
    return int(torch.argmin(mse))


@torch.no_grad()
def kernel_allcat(get_layer: Callable[[int], Dict[str, torch.Tensor]], n_layers: int, Y: np.ndarray,
                  ratios: Sequence[float] = RATIOS, log=print):
    """All-layer ridge in dual form.

    get_layer(l) -> {"fit": [n,D], "val": [nv,D], "test": [nt,D]} float32 tensors on the device (raw, unstandardised).
    Returns {t: [ {"ratio": r, "val": [nv], "test": [nt]}, ... ]} - one candidate per lambda; the caller picks the
    lambda by validation MSE.  Primal lambda = ratio * n_rows, which in kernel units (K has a unit diagonal) is
    ratio * n_rows / (n_layers * D).
    Approximation: a target with NaN rows uses the globally-centred kernel restricted to its valid rows and its own
    mean (the intercept is not re-fit per row subset); with ~1% missing rows this is immaterial.
    """
    Kff = Kvf = Ktf = None
    dev, D = None, None
    for l in range(n_layers):
        d = get_layer(l)
        mu, sd = standardise_stats(d["fit"])
        Zf, Zv, Zt = [(d[k] - mu) / sd for k in ("fit", "val", "test")]
        dev, D = Zf.device, Zf.shape[1]
        scale = 1.0 / (n_layers * D)
        if Kff is None:
            Kff = torch.zeros(Zf.shape[0], Zf.shape[0], device=dev)
            Kvf = torch.zeros(Zv.shape[0], Zf.shape[0], device=dev)
            Ktf = torch.zeros(Zt.shape[0], Zf.shape[0], device=dev)
        Kff += (Zf @ Zf.T) * scale
        Kvf += (Zv @ Zf.T) * scale
        Ktf += (Zt @ Zf.T) * scale
        if l % 5 == 0:
            log(f"  allcat kernel: layer {l + 1}/{n_layers}")
    T = Y.shape[1]
    out = {t: [] for t in range(T)}
    groups = {}
    for t in range(T):
        groups.setdefault(np.isnan(Y[:, t]).tobytes(), []).append(t)
    for ts in groups.values():
        m = ~np.isnan(Y[:, ts[0]])
        ng = int(m.sum())
        if ng < 10:
            continue
        mi = torch.tensor(np.where(m)[0], device=dev)
        full = bool(m.all())
        Kg = Kff if full else Kff.index_select(0, mi).index_select(1, mi)
        Kv = Kvf if full else Kvf.index_select(1, mi)
        Kt = Ktf if full else Ktf.index_select(1, mi)
        Yg = torch.tensor(Y[m][:, ts], dtype=torch.float32, device=dev)
        ybar = Yg.mean(0)
        for r in ratios:
            lam = r * ng / (n_layers * D)
            A = Kg.clone()
            A.diagonal().add_(lam)
            L, info = torch.linalg.cholesky_ex(A)
            if int(info) != 0:      # float32 cannot factor this lambda: skip it (no jitter - that would silently change lambda)
                log(f"  allcat: cholesky failed at ratio {r}; candidate skipped")
                del A, L
                continue
            alpha = torch.cholesky_solve(Yg - ybar, L)
            del A, L
            pv, pt = Kv @ alpha + ybar, Kt @ alpha + ybar
            for j, t in enumerate(ts):
                out[t].append({"ratio": r, "val": pv[:, j].cpu().numpy(), "test": pt[:, j].cpu().numpy()})
        if not full:
            del Kg
    return out
