"""Stage (GPU, ~10 min, optional): is the probe direction CAUSAL?  Add alpha * (unit probe direction) to the residual stream of
layer L while the forced-answer cue is scored, and see whether log P(gold) moves; a random direction of the same norm is the control.

    python cluster/msig_steer.py --run-dir $RUN_DIR [--layer 20] [--n-traces 150] [--alphas=-0.3,-0.1,0.1,0.3]
alpha is a fraction of the mean norm of the layer's checkpoint states.  Needs probes/weights_layers.pt (sig_probes.py --save-weights).
Writes <out>/analysis/steer.csv and steer.md.  A causal effect = the probe-direction rows move `a` monotonically in alpha and the
random rows do not.  (Additive steering at all positions is a blunt instrument; a null here does not rule out a causal role.)
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
from miol.signals.mathdata import SYSTEM_PROMPT_MATH  # noqa: E402
from miol.signals.scoring import DEFAULT_CUE  # noqa: E402
from miol.signals.worker import Worker  # noqa: E402


def layer_module(model, l):
    """hidden_states[l] is the output of decoder layer l-1 (hidden_states[0] = embeddings)."""
    base = model.model if hasattr(model, "model") else model
    return base.layers[l - 1]


def main(argv=None, tok=None, model=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--target", default="a")
    ap.add_argument("--n-traces", type=int, default=150)
    ap.add_argument("--per-trace", type=int, default=4, help="checkpoints per trace")
    ap.add_argument("--alphas", default="-0.3,-0.1,0.1,0.3")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--system", default=SYSTEM_PROMPT_MATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    W = torch.load(out / "probes" / "weights_layers.pt", map_location="cpu")
    w = W["layers"][a.layer]
    j = W["targets"].index(a.target)
    v = (w["W"][j] / w["sd"]).float()
    v = v / v.norm()
    ck = sio.read_table(out / "ckpt_eval.csv")
    ck = ck[ck["kind"] != "final"]
    rng = np.random.RandomState(a.seed)
    samples = sorted(ck["sample"].unique())
    samples = sorted(rng.permutation(samples)[:a.n_traces].tolist())
    parts = []
    for s in samples:
        g = ck[ck["sample"] == s]
        parts.append(g.iloc[np.sort(rng.permutation(len(g))[:a.per_trace])])
    sub = pd.concat(parts, ignore_index=True)
    ids = {}
    want = set(samples)
    for line in open(out / "ids_eval.jsonl"):
        r = json.loads(line)
        if r["sample"] in want:
            ids[r["sample"]] = (r["prompt_ids"], r["completion_ids"])
    S = np.load(out / "states_eval.npy", mmap_mode="r")
    scale = float(np.linalg.norm(np.asarray(S[a.layer][np.sort(sub["row"].values)], dtype=np.float32), axis=1).mean())
    tok = tok or sio.load_tokenizer(a.model_id)
    model = model or sio.load_model(a.model_id, a.device)
    wk = Worker(model, tok, a.device, a.system, DEFAULT_CUE, mode="lines", stride=8, seed=a.seed, log=sio.log_to("[steer]"))
    mod = layer_module(model, a.layer)
    state = {"vec": None}

    def hook(_m, _i, o):
        if state["vec"] is None:
            return o
        if isinstance(o, tuple):
            return (o[0] + state["vec"].to(o[0].dtype).to(o[0].device),) + tuple(o[1:])
        return o + state["vec"].to(o.dtype).to(o.device)
    h = mod.register_forward_hook(hook)

    def run(vec):
        state["vec"] = vec
        sc = wk.score(sub[["sample", "k", "pos", "gold", "final"]].copy(), ids)
        return sc["a"].values.astype(float), (sc["forced"].astype(str).values == sub["gold"].astype(str).values).astype(float)
    rows = []
    try:
        a0, g0 = run(None)
        rows.append({"direction": "none", "alpha": 0.0, "mean_a": a0.mean(), "d_a": 0.0, "d_a_se": 0.0, "forced_gold": g0.mean()})
        for al in [float(x) for x in a.alphas.split(",")]:
            for name, vecs in (("probe", [v]), ("random", [torch.nn.functional.normalize(torch.randn_like(v), dim=0) for _ in range(a.n_random)])):
                for vi, vec in enumerate(vecs):
                    ai, gi = run(vec * al * scale)
                    d = ai - a0
                    rows.append({"direction": name if name == "probe" else f"random{vi}", "alpha": al, "mean_a": ai.mean(), "d_a": d.mean(),
                                 "d_a_se": d.std(ddof=1) / np.sqrt(len(d)), "forced_gold": gi.mean()})
    finally:
        h.remove()
    df = pd.DataFrame(rows)
    (out / "analysis").mkdir(exist_ok=True)
    df.to_csv(out / "analysis" / "steer.csv", index=False)
    txt = [f"# Steering along the layer-{a.layer} probe direction for `{a.target}` ({len(sub)} checkpoints, {len(samples)} traces)\n",
           f"alpha x mean state norm ({scale:.1f}) is added at every position of layer {a.layer} while the forced answer is scored. "
           "d_a = change in log P(gold) vs unsteered. The probe rows should move monotonically with alpha and the random rows should not.\n",
           an.md(df)]
    (out / "analysis" / "steer.md").write_text("\n".join(txt))
    print("\n".join(txt))
    return df


if __name__ == "__main__":
    main()
