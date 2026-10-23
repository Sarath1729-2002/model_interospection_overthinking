"""Paths and loaders shared by the cluster/sig_*.py scripts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def out_dir(run_dir, out=None) -> Path:
    d = Path(out) if out else Path(run_dir) / "signals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hs_dir(run_dir, split: str) -> Path:
    return Path(run_dir) / "hidden_states" / split


def write_table(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=["", "nan", "NaN"], dtype={"gold": str, "final": str, "forced": str},
                       low_memory=False)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    return model


def log_to(prefix: str):
    def _log(*a):
        print(prefix, *a, flush=True)
    return _log
