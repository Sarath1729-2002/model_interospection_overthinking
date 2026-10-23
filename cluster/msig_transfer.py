"""Stage (GPU/CPU, minutes): does a probe trained on MATH read the same thing on GSM8K?  (zero-shot transfer, no refitting)

Applies the per-layer MATH-trained ridge weights (probes/weights_layers.pt, written by sig_probes.py --save-weights) to the
cached GSM8K checkpoint states and compares with the GSM8K labels.  Reports R2 and Spearman per target and layer, next to the
GSM8K-trained probe at the same layer when its scores exist.

    python cluster/msig_transfer.py --math-out $RUN_DIR_MATH/signals --gsm-out $RUN_DIR_GSM/signals [--layers 16,20,22,24]
Writes <math-out>/analysis/transfer.csv and transfer.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import analysis as an  # noqa: E402
from miol.signals import io as sio  # noqa: E402
from miol.signals.metrics import r2, spearman  # noqa: E402
from miol.signals.targets import build_targets  # noqa: E402

TARGETS = ["a", "f", "rem", "d_gold", "d_commit", "fg", "ff"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-out", required=True)
    ap.add_argument("--gsm-out", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args(argv)
    mo, go = Path(a.math_out), Path(a.gsm_out)
    W = torch.load(mo / "probes" / "weights_layers.pt", map_location="cpu")
    wn, wl = W["targets"], W["layers"]
    lab = sio.read_table(go / "labels_eval.csv")
    S = np.load(go / "states_eval.npy", mmap_mode="r")
    Y, names = build_targets(lab, include_rollouts=False)
    rows = lab["row"].values
    if a.max_rows and len(rows) > a.max_rows:
        sel = np.sort(np.random.RandomState(0).permutation(len(rows))[:a.max_rows])
        rows, Y = rows[sel], Y[sel]
    layers = sorted(wl) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    own = None
    f = go / "probes" / "scores_test.csv"
    if f.exists():
        own = pd.read_csv(f)
    out = []
    for l in layers:
        if l not in wl:
            continue
        X = torch.from_numpy(np.ascontiguousarray(S[l][rows])).to(a.device).float()
        w = wl[l]
        P = (((X - w["mu"].to(a.device)) / w["sd"].to(a.device)) @ w["W"].to(a.device).T + w["b"].to(a.device)).cpu().numpy()
        for t in TARGETS:
            if t not in wn or t not in names:
                continue
            y, p = Y[:, names.index(t)], P[:, wn.index(t)]
            m = np.isfinite(y)
            if m.sum() < 50:
                continue
            r = {"layer": l, "target": t, "n": int(m.sum()), "R2 (MATH probe on GSM8K)": r2(y[m], p[m]), "Spearman": spearman(y[m], p[m])}
            if own is not None:
                o = own[(own.model == f"L{l}") & (own.target == t)]
                if len(o):
                    r["R2 (GSM8K-trained)"] = float(o["r2"].iloc[0])
            out.append(r)
        print(f"[transfer] layer {l} done", flush=True)
    df = pd.DataFrame(out)
    (mo / "analysis").mkdir(exist_ok=True)
    df.to_csv(mo / "analysis" / "transfer.csv", index=False)
    best = df.sort_values("Spearman", ascending=False).groupby("target").head(1)
    txt = ["# Transfer: MATH-trained probes applied to GSM8K states (no refit)\n",
           "Low R2 with a high Spearman means the direction transfers but the scale/offset does not (the distributions differ). "
           "R2 near 0 and Spearman near 0 means the probe reads something dataset-specific.\n",
           "Best layer per target (by Spearman):\n", an.md(best)]
    (mo / "analysis" / "transfer.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
