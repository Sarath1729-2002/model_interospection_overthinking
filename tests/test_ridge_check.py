"""ridge_check recovers a planted linear signal in one layer and not in a noise layer."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

SPEC = importlib.util.spec_from_file_location("ridge_check", Path(__file__).resolve().parent.parent / "cluster" / "ridge_check.py")
rc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc)


def _write(split_dir, n, rng, direction, D=16, layers=3, prompt=5):
    split_dir.mkdir(parents=True)
    for i in range(n):
        cl = int(rng.randint(30, 80))
        T = prompt + cl
        rem = np.concatenate([np.zeros(prompt), cl - np.arange(cl) - 1])
        tensors = {f"hidden_states_{l}": torch.tensor(rng.randn(T, D) * 3.0, dtype=torch.float32) for l in range(layers)}
        tensors["hidden_states_1"] = torch.tensor(rng.randn(T, D) * 0.05 + np.outer(rem, direction), dtype=torch.float32)  # signal layer
        tensors["prompt_ids"] = torch.zeros(prompt, dtype=torch.long)
        tensors["completion_ids"] = torch.zeros(cl, dtype=torch.long)
        tensors["full_sequence_ids"] = torch.zeros(T, dtype=torch.long)
        save_file(tensors, str(split_dir / f"sample_{i:06d}.safetensors"))
        json.dump({"prompt_length": prompt, "completion_length": cl, "has_eos": True, "num_hidden_layers": layers,
                   "prompt": "", "completion": ""}, open(split_dir / f"sample_{i:06d}.json", "w"))


def test_ridge_finds_signal_layer(tmp_path):
    rng = np.random.RandomState(0)
    direction = rng.randn(16)                              # same planted direction in train and eval
    _write(tmp_path / "hidden_states" / "train", 60, rng, direction)
    _write(tmp_path / "hidden_states" / "eval", 20, rng, direction)
    res = rc.run(tmp_path, layers=[0, 1, 2], val_fraction=0.2, workers=0, device="cpu", n_boot=50, log=lambda *_: None)
    r = res["layers"]
    assert r["1"]["test_mae"] < 5.0                      # planted layer is decodable
    assert r["0"]["test_mae"] > 0.7 * res["baseline_constant_median"]   # noise layers are not
    assert r["1"]["test_mae"] < 0.2 * r["0"]["test_mae"]
    assert "best layer: 1" in rc.format_summary(res)
