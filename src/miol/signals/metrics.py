"""Metrics with cluster (per-problem) bootstrap.  numpy only (no scipy / sklearn on the cluster venv)."""
from typing import Dict, Optional

import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auroc(y, s) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - p) ** 2).sum() / den) if den > 0 else float("nan")


def _cluster_sums(y, p, groups):
    u, inv = np.unique(groups, return_inverse=True)
    e = np.asarray(p, float) - np.asarray(y, float)
    y = np.asarray(y, float)
    return dict(n=np.bincount(inv).astype(float), sabs=np.bincount(inv, np.abs(e)), sse=np.bincount(inv, e * e),
                sy=np.bincount(inv, y), sy2=np.bincount(inv, y * y)), len(u)


def _mae_r2(cs, counts):
    n = counts @ cs["n"]
    mae_ = (counts @ cs["sabs"]) / n
    sy, sy2 = counts @ cs["sy"], counts @ cs["sy2"]
    den = sy2 - sy ** 2 / n
    r2_ = 1.0 - (counts @ cs["sse"]) / np.where(den > 0, den, np.nan)
    return mae_, r2_


def _ci(x, est):
    return {"est": float(est), "lo": float(np.nanpercentile(x, 2.5)), "hi": float(np.nanpercentile(x, 97.5))}


def boot_reg(y, p, groups, B: int = 500, seed: int = 0) -> Dict[str, Dict[str, float]]:
    cs, C = _cluster_sums(y, p, groups)
    counts = np.random.RandomState(seed).multinomial(C, np.ones(C) / C, size=B).astype(float)
    mb, rb = _mae_r2(cs, counts)
    m0, r0 = _mae_r2(cs, np.ones((1, C)))
    return {"mae": _ci(mb, m0[0]), "r2": _ci(rb, r0[0])}


def boot_paired(y, pa, pb, groups, B: int = 500, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """a minus b: delta_mae (negative = a better), delta_r2 (positive = a better)."""
    ca, C = _cluster_sums(y, pa, groups)
    cb, _ = _cluster_sums(y, pb, groups)
    counts = np.random.RandomState(seed).multinomial(C, np.ones(C) / C, size=B).astype(float)
    ma, ra = _mae_r2(ca, counts)
    mb, rb = _mae_r2(cb, counts)
    one = np.ones((1, C))
    ma0, ra0 = _mae_r2(ca, one)
    mb0, rb0 = _mae_r2(cb, one)
    return {"delta_mae": _ci(ma - mb, ma0[0] - mb0[0]), "delta_r2": _ci(ra - rb, ra0[0] - rb0[0])}


def boot_auroc(y, s, groups, B: int = 200, seed: int = 0, s2: Optional[np.ndarray] = None):
    """AUROC of s (and, if s2 is given, of s minus s2's AUROC) with a cluster bootstrap."""
    y, s = np.asarray(y).astype(bool), np.asarray(s, float)
    u, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == c)[0] for c in range(len(u))]
    rng = np.random.RandomState(seed)
    est = auroc(y, s) - (auroc(y, s2) if s2 is not None else 0.0)
    vals = []
    for _ in range(B):
        pick = rng.randint(0, len(u), len(u))
        idx = np.concatenate([members[c] for c in pick])
        a = auroc(y[idx], s[idx])
        if s2 is not None:
            a -= auroc(y[idx], np.asarray(s2, float)[idx])
        vals.append(a)
    return _ci(np.array(vals), est)


def within_group_spearman(y, p, groups, min_n: int = 4):
    """Mean over groups of Spearman(y, p) inside each group (groups with < min_n rows or constant y are skipped)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    u, inv = np.unique(groups, return_inverse=True)
    vals = {}
    for c in range(len(u)):
        idx = np.where(inv == c)[0]
        if len(idx) >= min_n and np.ptp(y[idx]) > 0:
            v = spearman(y[idx], p[idx])
            if not np.isnan(v):
                vals[u[c]] = v
    return (float(np.mean(list(vals.values()))) if vals else float("nan")), vals


def boot_within(vals_a: dict, vals_b: Optional[dict] = None, B: int = 500, seed: int = 0):
    """CI of the mean within-group Spearman (and of a-b on the common groups)."""
    keys = sorted(vals_a if vals_b is None else set(vals_a) & set(vals_b))
    if not keys:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_groups": 0}
    d = np.array([vals_a[k] - (vals_b[k] if vals_b is not None else 0.0) for k in keys])
    rng = np.random.RandomState(seed)
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)]
    out = _ci(np.array(bs), d.mean())
    out["n_groups"] = len(keys)
    return out
