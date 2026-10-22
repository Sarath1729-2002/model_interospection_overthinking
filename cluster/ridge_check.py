"""Closed-form ridge cross-check for the per-layer linear probes.

Why: the trained probes are raw nn.Linear heads fit with Adam on unnormalised residuals, so a
poor MAE can mean "hard to optimise" rather than "not linearly decodable".  Ridge regression has
no learning rate / schedule / step budget: if it beats the trained probe clearly, the trained
probe is optimisation-limited.

For each requested layer it fits  remaining_tokens ~ w . h + b  on the TRAIN part of the train
split (same fixed split as training), picks the ridge strength on the held-out VALIDATION part, and
reports the TEST MAE (mean over completions of per-completion token-level MAE - the same number
`evaluate_probes.py` / `summarize_eval.py` print) with a bootstrap 95% CI, next to simple baselines.

    python cluster/ridge_check.py --run-dir <run_dir> [--layers 8,16,24,27,28 | all]

Output: <run_dir>/ridge_check/ridge.json and summary.txt.  Features are standardised per
dimension; sufficient statistics (X'X, X'y) are streamed in float64 so memory is O(D^2) per layer,
not O(tokens).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.data.hidden_states_dataset import split_indices  # noqa: E402

LAMBDA_RATIOS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]   # ridge strength / n_tokens (Z'Z has diag ~ n)


class CompletionTokens(Dataset):
    """One item per sample: hidden states of the chosen layers at completion positions + targets."""

    def __init__(self, files, layers):
        self.files, self.layers = files, layers

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        meta = json.load(open(path.with_suffix(".json")))
        pl, cl = int(meta["prompt_length"]), int(meta["completion_length"])
        with safe_open(str(path), "pt") as f:
            layers = self.layers if self.layers is not None else list(range(int(meta["num_hidden_layers"])))
            X = torch.stack([f.get_tensor(f"hidden_states_{l}")[pl:] for l in layers], dim=1)   # [T, nL, D]
        T = X.shape[0]
        y = (cl - torch.arange(T) - 1).float()                                                     # remaining tokens
        return X, y


def _loader(files, layers, workers):
    return DataLoader(CompletionTokens(files, layers), batch_size=None, shuffle=False,
                      num_workers=workers, prefetch_factor=2 if workers else None)


def _sample_files(run_dir, split):
    d = Path(run_dir) / "hidden_states" / split
    files = sorted(d.glob("sample_*.safetensors"))
    if not files:
        raise SystemExit(f"no samples in {d}")
    return files


def _first_meta(f):
    return json.load(open(f.with_suffix(".json")))


def run(run_dir, layers=None, val_fraction=0.05, split_seed=0, workers=8, device=None, ratios=LAMBDA_RATIOS,
        n_boot=1000, log=print):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_all, test_files = _sample_files(run_dir, "train"), _sample_files(run_dir, "eval")
    n_layers = int(_first_meta(train_all[0])["num_hidden_layers"])
    layer_ids = list(range(n_layers)) if layers in (None, "all") else [int(x) for x in layers]
    tr_idx, va_idx = split_indices(len(train_all), val_fraction, split_seed)
    train_files = [train_all[i] for i in tr_idx]
    val_files = [train_all[i] for i in va_idx]
    log(f"train {len(train_files)} | val {len(val_files)} | test {len(test_files)} | layers {layer_ids} | device {device}")

    # ---- pass A: sufficient statistics on the train part ----
    G = S = B = None
    n = 0
    sy = 0.0
    train_y = []
    for k, (X, y) in enumerate(_loader(train_files, layer_ids, workers)):
        X = X.to(device, torch.float64)
        yd = y.to(device, torch.float64)
        if G is None:
            nL, D = X.shape[1], X.shape[2]
            G = torch.zeros(nL, D, D, dtype=torch.float64, device=device)
            S = torch.zeros(nL, D, dtype=torch.float64, device=device)
            B = torch.zeros(nL, D, dtype=torch.float64, device=device)
        for j in range(nL):
            Xj = X[:, j]
            G[j] += Xj.T @ Xj
            S[j] += Xj.sum(0)
            B[j] += Xj.T @ yd
        n += X.shape[0]
        sy += float(yd.sum())
        train_y.append(y.numpy().astype(np.int32))
        if (k + 1) % 500 == 0:
            log(f"  pass A: {k + 1}/{len(train_files)} samples")
    ybar = sy / n
    log(f"pass A done: {n} train tokens")

    # ---- solve for every (layer, lambda) ----
    lambdas = [r * n for r in ratios]
    Wx = torch.zeros(nL, D, len(lambdas), dtype=torch.float64, device=device)
    b0 = torch.zeros(nL, len(lambdas), dtype=torch.float64, device=device)
    for j in range(nL):
        mu = S[j] / n
        var = torch.diagonal(G[j]) / n - mu ** 2
        keep = var > 1e-10
        sigma = torch.where(keep, var.clamp_min(1e-10).sqrt(), torch.ones_like(var))
        Gz = (G[j] - n * torch.outer(mu, mu)) / torch.outer(sigma, sigma)
        Gz[~keep, :] = 0
        Gz[:, ~keep] = 0
        bz = (B[j] - mu * sy) / sigma
        bz[~keep] = 0
        evals, Q = torch.linalg.eigh(Gz)
        evals = evals.clamp_min(0)
        Qb = Q.T @ bz
        for m, lam in enumerate(lambdas):
            wz = Q @ (Qb / (evals + lam))
            wx = wz / sigma
            Wx[j, :, m] = wx
            b0[j, m] = ybar - (mu * wx).sum()
    del G

    # ---- pass B: predictions on val and test ----
    def evaluate(files):
        pooled_abs = torch.zeros(nL, len(lambdas), dtype=torch.float64, device=device)
        tokens = 0
        per_comp = []
        for X, y in _loader(files, layer_ids, workers):
            X = X.to(device, torch.float64)
            yd = y.to(device, torch.float64)
            errs = torch.stack([(X[:, j] @ Wx[j] + b0[j] - yd[:, None]).abs() for j in range(nL)], dim=0)  # [nL,T,nλ]
            pooled_abs += errs.sum(1)
            per_comp.append(errs.mean(1).cpu().numpy())                                                     # [nL,nλ]
            tokens += X.shape[0]
        return pooled_abs.cpu().numpy() / tokens, np.stack(per_comp, axis=0)                               # [nL,nλ], [N,nL,nλ]

    val_pooled, _ = evaluate(val_files)
    test_pooled, test_comp = evaluate(test_files)

    # ---- baselines on test (per-completion MAE, same convention) ----
    all_y = np.concatenate(train_y)
    med = float(np.median(all_y))
    maxlen = max(len(a) for a in train_y)
    pos_med = np.array([np.median(np.array([a[t] for a in train_y if len(a) > t])) for t in range(maxlen)])
    base_const, base_pos = [], []
    for _, y in _loader(test_files, layer_ids[:1], workers):
        y = y.numpy()
        base_const.append(np.abs(y - med).mean())
        base_pos.append(np.abs(y - pos_med[np.minimum(np.arange(len(y)), maxlen - 1)]).mean())

    rng = np.random.RandomState(0)

    def boot(v):
        v = np.asarray(v)
        idx = rng.randint(0, len(v), size=(n_boot, len(v)))
        m = v[idx].mean(1)
        return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    result = {"layers": {}, "n_train_tokens": n, "n_val": len(val_files), "n_test": len(test_files),
              "ratios": list(ratios),
              "baseline_constant_median": float(np.mean(base_const)), "baseline_position_median": float(np.mean(base_pos))}
    for j, L in enumerate(layer_ids):
        m = int(np.argmin(val_pooled[j]))
        comp = test_comp[:, j, m]
        lo, hi = boot(comp)
        result["layers"][str(L)] = {"best_lambda_ratio": ratios[m], "val_mae_pooled": float(val_pooled[j, m]),
                                    "test_mae": float(comp.mean()), "test_ci95": [lo, hi],
                                    "test_mae_pooled": float(test_pooled[j, m])}
    return result


def format_summary(res):
    lines = [f"ridge cross-check: {res['n_train_tokens']} train tokens, {res['n_val']} val / {res['n_test']} test completions",
             f"constant-median baseline MAE {res['baseline_constant_median']:.1f} | position-median baseline MAE {res['baseline_position_median']:.1f}",
             "layer   test_MAE  [95% CI]            val_MAE(pooled)  lambda/n"]
    for L, r in res["layers"].items():
        lines.append(f"{int(L):5d}  {r['test_mae']:8.2f}  [{r['test_ci95'][0]:6.2f}, {r['test_ci95'][1]:6.2f}]   "
                     f"{r['val_mae_pooled']:12.2f}   {r['best_lambda_ratio']:g}")
    best = min(res["layers"].items(), key=lambda kv: kv[1]["test_mae"])
    lines.append(f"best layer: {best[0]} (test MAE {best[1]['test_mae']:.2f})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--layers", default="8,12,16,20,24,26,27,28", help="comma list or 'all'")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    layers = "all" if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    res = run(a.run_dir, layers, a.val_fraction, a.split_seed, a.workers, a.device)
    out = Path(a.run_dir) / "ridge_check"
    out.mkdir(exist_ok=True)
    (out / "ridge.json").write_text(json.dumps(res, indent=1))
    text = format_summary(res)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
