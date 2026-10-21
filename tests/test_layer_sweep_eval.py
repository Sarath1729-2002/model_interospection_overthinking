"""layer_sweep_eval agrees with a naive per-layer reference computed via the single-layer dataset path."""
import json

import numpy as np
import torch
from safetensors.torch import save_file

from miol.analysis.layer_sweep_eval import discover_layers, evaluate, write_outputs
from miol.data.hidden_states_dataset import HiddenStatesDataset

NL, H = 4, 5


def _make_eval(tmp_path, specs):
    d = tmp_path / "hs" / "eval"
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    for i, (P, C, eos) in enumerate(specs):
        t = {"prompt_ids": torch.arange(P), "completion_ids": torch.arange(C),
             "full_sequence_ids": torch.arange(P + C)}
        for k in range(NL):
            t[f"hidden_states_{k}"] = torch.randn(P + C, H, generator=g)
        save_file(t, str(d / f"sample_{i:06d}.safetensors"))
        (d / f"sample_{i:06d}.json").write_text(json.dumps(
            {"prompt": "p", "completion": "c", "has_eos": eos, "prompt_length": P,
             "completion_length": C, "num_hidden_layers": NL}))
    return tmp_path / "hs"


def _make_heads(tmp_path, name, seed):
    d = tmp_path / name
    d.mkdir()
    g = torch.Generator().manual_seed(seed)
    for pr in ("count", "prompt_only_count"):
        for L in range(NL):
            torch.save({"head": {"weight": torch.randn(1, H, generator=g), "bias": torch.randn(1, generator=g)}},
                       d / f"{pr}_L{L}_head.pt")
    return d


def test_matches_naive_reference(tmp_path):
    specs = [(3, 6, True), (4, 9, True), (2, 5, False), (5, 4, True)]
    hs = _make_eval(tmp_path, specs)
    pds = [_make_heads(tmp_path, "s0", 1), _make_heads(tmp_path, "s1", 2)]
    assert discover_layers(pds[0], "count") == list(range(NL))
    res = evaluate(pds, hs, num_workers=0)
    assert res["n_completions"] == 3 and res["n_tokens"] == 6 + 9 + 4

    pooled = np.zeros((2, NL)); prompt = np.zeros((2, NL, 3)); percomp = np.zeros((2, NL, 3))
    for L in range(NL):
        ds = HiddenStatesDataset(str(hs), split="eval", hidden_layer=str(L))
        j = 0
        for i, (P, C, eos) in enumerate(specs):
            if not eos:
                continue
            s = ds[i]
            tgt = s["remaining_token_counts"][P:].float()
            for si, pd in enumerate(pds):
                for pr in ("count", "prompt_only_count"):
                    sd = torch.load(pd / f"{pr}_L{L}_head.pt")["head"]
                    pred = s["hidden_states"] @ sd["weight"].reshape(-1) + sd["bias"]
                    if pr == "count":
                        err = (pred[P:] - tgt).abs()
                        pooled[si, L] += err.sum().item() / res["n_tokens"]
                        percomp[si, L, j] = err.mean().item()
                    else:
                        prompt[si, L, j] = abs(pred[P - 1].item() - C)
            j += 1
    np.testing.assert_allclose(res["count"]["mae_pooled"], pooled.mean(0), rtol=1e-4)
    np.testing.assert_allclose(res["count"]["mae_per_completion"], percomp.mean((0, 2)), rtol=1e-4)
    np.testing.assert_allclose(res["prompt_only_count"]["prompt_ae"], prompt.mean((0, 2)), rtol=1e-4)
    assert res["count"]["best_layer"] == int(np.argmin(pooled.mean(0)))
    lo, hi = res["count"]["mae_per_completion_ci95"]
    assert all(l <= m <= h for l, m, h in zip(lo, res["count"]["mae_per_completion"], hi))
    # constant-median reference
    allt = np.concatenate([np.arange(C - 1, -1, -1) for P, C, e in specs if e])
    assert abs(res["reference"]["constant_median_mae"] - np.abs(allt - np.median(allt)).mean()) < 1e-6

    write_outputs(res, tmp_path / "out", all_layers_mae=3.0)
    assert (tmp_path / "out" / "layer_sweep.json").exists()
    rows = (tmp_path / "out" / "layer_sweep.csv").read_text().strip().splitlines()
    assert len(rows) == NL + 1
