"""Per-layer evaluation of a layer-sweep run (paper appendix A.2, Figure 4).

`train_offline.py data.layers_to_train=all` trains one Remaining-Count head per layer
(`count_L0_head.pt` ... `count_L28_head.pt`) and one prompt-end head per layer
(`prompt_only_count_L*_head.pt`). Their eval metrics otherwise only live in (offline)
W&B; this script re-evaluates every saved head on the eval split, from disk, in ONE pass:

    python -m miol.analysis.layer_sweep_eval \
        --probe-dir outputs/run/run-0/seed_0/trained_probe  [--probe-dir ...seed_1 ...] \
        --hidden-states-dir outputs/run/hidden_states --output-dir outputs/run/layer_sweep

Outputs (in --output-dir): layer_sweep.json, layer_sweep.csv, layer_sweep.png.

Metrics (completions that ended in EOS only):
  count  : per-token MAE of remaining-token prediction, pooled over all eval tokens
           (`mae_pooled`) and mean of per-completion MAEs (`mae_per_completion`, the repo's
           convention). 95% CI = bootstrap over eval completions of the seed-averaged MAE.
  prompt : |T_hat - T| at the last prompt token (prompt-end probe), mean over completions.
  constant-median reference is computed from the eval targets, as in the paper (Sec 3.4).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from miol.data.hidden_states_dataset import HiddenStatesDataset


def discover_layers(probe_dir: Path, probe: str) -> list[int]:
    pat = re.compile(rf"^{re.escape(probe)}_L(\d+)_head\.pt$")
    return sorted(int(m.group(1)) for p in probe_dir.iterdir() if (m := pat.match(p.name)))


def load_stacked_heads(probe_dir: Path, probe: str, layers: list[int]):
    """-> W [n_layers, H], b [n_layers] for a regression head per layer."""
    ws, bs = [], []
    for L in layers:
        sd = torch.load(probe_dir / f"{probe}_L{L}_head.pt", map_location="cpu")["head"]
        ws.append(sd["weight"].reshape(-1).float())
        bs.append(sd["bias"].reshape(-1).float())
    return torch.stack(ws), torch.cat(bs)


def evaluate(probe_dirs: list[Path], hidden_states_dir: Path, probes=("count", "prompt_only_count"),
             device="cpu", num_workers=0, max_samples=None):
    layers = discover_layers(probe_dirs[0], probes[0])
    if not layers:
        raise SystemExit(f"no {probes[0]}_L*_head.pt files in {probe_dirs[0]} - was the run trained with data.layers_to_train=all?")
    for pd in probe_dirs:
        for pr in probes:
            if discover_layers(pd, pr) != layers:
                raise SystemExit(f"{pd}: layers of '{pr}' differ from {probes[0]} in {probe_dirs[0]}")
    ds = HiddenStatesDataset(str(hidden_states_dir), split="eval", layers_to_train=layers)
    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    # heads[(seed_idx, probe)] = (W, b) on device
    heads = {(s, pr): tuple(t.to(device) for t in load_stacked_heads(pd, pr, layers))
             for s, pd in enumerate(probe_dirs) for pr in probes}
    S, nL = len(probe_dirs), len(layers)

    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=None,
                                         shuffle=False, num_workers=num_workers)
    tok_abs = {pr: np.zeros((S, nL)) for pr in probes}        # pooled abs-error sums (count) / unused
    n_tok = 0
    ex_mae = []        # per completion: [S, nL] mean abs err over its tokens (count probe)
    ex_prompt_ae = []  # per completion: [S, nL] |T_hat - T|
    targets_all, T_all = [], []
    for sample in loader:
        if not sample["has_eos"]:
            continue
        P, C = int(sample["prompt_length"]), int(sample["completion_length"])
        hs = sample["hidden_states"].to(device).float()           # [seq, nL, H]
        tgt = torch.arange(C - 1, -1, -1, device=device, dtype=torch.float32)
        targets_all.append(tgt.cpu().numpy()); T_all.append(C)
        if "count" in probes:
            h = hs[P:P + C]                                        # completion positions
            row = np.zeros((S, nL))
            for s in range(S):
                W, b = heads[(s, "count")]
                pred = torch.einsum("tld,ld->tl", h, W) + b        # [C, nL]
                err = (pred - tgt[:, None]).abs()
                row[s] = err.mean(0).cpu().numpy()
                tok_abs["count"][s] += err.sum(0).cpu().numpy()
            ex_mae.append(row)
        if "prompt_only_count" in probes:
            h0 = hs[P - 1]                                         # prompt-end position
            row = np.zeros((S, nL))
            for s in range(S):
                W, b = heads[(s, "prompt_only_count")]
                row[s] = ((h0 * W).sum(-1) + b - C).abs().cpu().numpy()
            ex_prompt_ae.append(row)
        n_tok += C
    if not T_all:
        raise SystemExit("no eval completions with EOS were found")

    pooled_t = np.concatenate(targets_all)
    out = {"layers": layers, "n_completions": len(T_all), "n_tokens": int(n_tok),
           "n_seeds": S, "probe_dirs": [str(p) for p in probe_dirs],
           "reference": {"constant_median_mae": float(np.abs(pooled_t - np.median(pooled_t)).mean()),
                         "constant_median_prompt_ae": float(np.abs(np.array(T_all) - np.median(T_all)).mean())}}
    rng = np.random.default_rng(0)

    def boot(per_ex):                        # per_ex [n_ex, nL] (seed-averaged) -> 95% CI over completions
        idx = rng.integers(0, len(per_ex), size=(1000, len(per_ex)))
        means = per_ex[idx].mean(1)          # [1000, nL]
        return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)

    if "count" in probes:
        em = np.stack(ex_mae)                # [n_ex, S, nL]
        pooled_seed = tok_abs["count"] / n_tok            # [S, nL]
        lo, hi = boot(em.mean(1))
        out["count"] = {
            "mae_pooled": pooled_seed.mean(0).tolist(),
            "mae_pooled_seed_std": pooled_seed.std(0).tolist(),
            "mae_per_completion": em.mean((0, 1)).tolist(),
            "mae_per_completion_ci95": [lo.tolist(), hi.tolist()],
            "mae_pooled_per_seed": pooled_seed.tolist(),
            "best_layer": int(layers[int(np.argmin(pooled_seed.mean(0)))]),
        }
    if "prompt_only_count" in probes:
        ea = np.stack(ex_prompt_ae)
        lo, hi = boot(ea.mean(1))
        out["prompt_only_count"] = {
            "prompt_ae": ea.mean((0, 1)).tolist(),
            "prompt_ae_seed_std": ea.mean(0).std(0).tolist(),
            "prompt_ae_ci95": [lo.tolist(), hi.tolist()],
            "best_layer": int(layers[int(np.argmin(ea.mean((0, 1))))]),
        }
    return out


def write_outputs(res: dict, out_dir: Path, all_layers_mae: float | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if all_layers_mae is not None:
        res["reference"]["all_layers_concat_mae"] = all_layers_mae
    (out_dir / "layer_sweep.json").write_text(json.dumps(res, indent=1))
    layers = res["layers"]
    with open(out_dir / "layer_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "count_mae_pooled", "count_mae_seed_std", "count_mae_per_completion",
                    "count_ci95_lo", "count_ci95_hi", "prompt_ae", "prompt_ae_ci95_lo", "prompt_ae_ci95_hi"])
        c, p = res.get("count"), res.get("prompt_only_count")
        for i, L in enumerate(layers):
            w.writerow([L,
                        *(( c["mae_pooled"][i], c["mae_pooled_seed_std"][i], c["mae_per_completion"][i],
                            c["mae_per_completion_ci95"][0][i], c["mae_per_completion_ci95"][1][i]) if c else [""] * 5),
                        *(( p["prompt_ae"][i], p["prompt_ae_ci95"][0][i], p["prompt_ae_ci95"][1][i]) if p else [""] * 3)])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    panels = [k for k in ("count", "prompt_only_count") if k in res]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4), squeeze=False)
    for ax, k in zip(axes[0], panels):
        if k == "count":
            y, (lo, hi), ref, lab = res[k]["mae_per_completion"], res[k]["mae_per_completion_ci95"], res["reference"]["constant_median_mae"], "per-token MAE (remaining tokens)"
        else:
            y, (lo, hi), ref, lab = res[k]["prompt_ae"], res[k]["prompt_ae_ci95"], res["reference"]["constant_median_prompt_ae"], "prompt-end |T_hat - T|"
        ax.plot(layers, y, marker="o", ms=3, label="single-layer probe")
        ax.fill_between(layers, lo, hi, alpha=0.2)
        ax.axhline(ref, ls="--", c="gray", label="constant median")
        if k == "count" and res["reference"].get("all_layers_concat_mae") is not None:
            ax.axhline(res["reference"]["all_layers_concat_mae"], ls=":", c="C3", label="all-layers concat")
        ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel(lab); ax.set_title(k); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "layer_sweep.png", dpi=150); plt.close(fig)


def format_table(res: dict) -> str:
    c, p, ref = res.get("count"), res.get("prompt_only_count"), res["reference"]
    lines = [f"{res['n_completions']} eval completions, {res['n_tokens']} tokens, {res['n_seeds']} seed(s)",
             f"constant-median MAE {ref['constant_median_mae']:.1f} | prompt-end AE {ref['constant_median_prompt_ae']:.1f}"
             + (f" | all-layers-concat MAE {ref['all_layers_concat_mae']:.1f}" if "all_layers_concat_mae" in ref else ""),
             "layer  count_MAE(+-seed std)   prompt_AE"]
    for i, L in enumerate(res["layers"]):
        cm = f"{c['mae_pooled'][i]:8.1f} +-{c['mae_pooled_seed_std'][i]:5.1f}" if c else " " * 16
        pa = f"{p['prompt_ae'][i]:8.1f}" if p else ""
        lines.append(f"{L:>5}  {cm}   {pa}")
    if c: lines.append(f"best count layer: {c['best_layer']}")
    if p: lines.append(f"best prompt-end layer: {p['best_layer']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", type=Path, action="append", required=True, help="repeat once per seed")
    ap.add_argument("--hidden-states-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--probes", nargs="+", default=["count", "prompt_only_count"])
    ap.add_argument("--all-layers-mae", type=float, default=None, help="draw the all-layers-concat line (from eval_results.json)")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = evaluate(a.probe_dir, a.hidden_states_dir, tuple(a.probes), dev, a.num_workers, a.max_samples)
    write_outputs(res, a.output_dir, a.all_layers_mae)
    print(format_table(res))
    print(f"wrote {a.output_dir}")


if __name__ == "__main__":
    main()
