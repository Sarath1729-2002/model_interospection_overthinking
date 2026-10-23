"""Regression targets, all exact functions of the scored checkpoints (NaN = undefined for that row)."""
from typing import List, Tuple

import numpy as np
import pandas as pd

CLIP = -20.0      # log-probabilities below e^-20 are floored: a probability that small is 'not at all', not an outlier to fit

DESCRIPTIONS = {
    "a": "log P(gold answer | prefix + cue)           what the model KNOWS at this point",
    "f": "log P(own final answer | prefix + cue)      what the trace is COMMITTED to (own answer parsed from the completion)",
    "g": "f - a                                       committed against the truth (large when it will end on a wrong answer)",
    "rem": "completion tokens still to be written     the length probe (the paper's target) on the same checkpoints",
    "pct": "rem / total completion length",
    "d_gold": "tokens from this checkpoint to the stable-convergence point (negative = not there yet)",
    "d_commit": "tokens from this checkpoint to the commitment point (negative = not there yet)",
    "fg": "forced greedy answer == gold (0/1)",
    "ff": "forced greedy answer == own final answer (0/1)",
    "r_H": "entropy of the final answer over sampled continuations (rollouts)",
    "r_pgold": "fraction of sampled continuations ending on gold (rollouts)",
    "r_pfinal": "fraction of sampled continuations ending on the trace's own final answer (rollouts)",
}
CORE = ["a", "f", "g", "rem", "pct", "d_gold", "d_commit", "fg", "ff"]
ROLL = {"r_H": "roll_H", "r_pgold": "roll_pgold", "r_pfinal": "roll_pfinal"}


def build_targets(df: pd.DataFrame, include_rollouts: bool = True, names=None) -> Tuple[np.ndarray, List[str]]:
    """names: force exactly this target list (missing rollout columns -> all-NaN), so splits built separately line up."""
    cols = {}
    a = np.clip(df["a"].values.astype(float), CLIP, 0.0)
    f = np.clip(df["f"].values.astype(float), CLIP, 0.0)        # NaN stays NaN
    cols["a"] = a
    cols["f"] = f
    cols["g"] = f - a
    cols["rem"] = df["rem"].values.astype(float)
    cols["pct"] = df["rem"].values / df["n_tok"].values
    cols["d_gold"] = df["d_gold"].values.astype(float)
    cols["d_commit"] = df["d_commit"].values.astype(float)
    cols["fg"] = df["eq_gold"].values.astype(float)
    ff = df["eq_final"].values.astype(float)
    ff[(df["final"].fillna("").astype(str) == "").values] = np.nan
    cols["ff"] = ff
    if names is not None:
        for name, col in ROLL.items():
            cols[name] = df[col].values.astype(float) if col in df.columns else np.full(len(df), np.nan)
        return np.stack([cols[n] for n in names], axis=1), list(names)
    names = list(CORE)
    if include_rollouts:
        for name, col in ROLL.items():
            if col in df.columns and np.isfinite(df[col].values.astype(float)).sum() > 100:
                cols[name] = df[col].values.astype(float)
                names.append(name)
    Y = np.stack([cols[n] for n in names], axis=1)
    return Y, names
