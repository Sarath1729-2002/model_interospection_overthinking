"""
Run all trained probes on the evaluation set and produce a single JSON file
with per-token predictions, per-token metrics, and aggregate metrics (full
sequence + last 10 tokens) for every probe and every statistical baseline.

The output JSON is meant to drive a static HTML dashboard (see dashboard/).

Usage:
    python src/miol/evaluate_probes.py \
        --probe-dir outputs/.../trained_probe \
        --hidden-states-dir outputs/.../hidden_states \
        --output outputs/.../eval_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from miol.data.hidden_states_dataset import HiddenStatesDataset
from miol.model.offline_trainer import MultiProbeOfflineTrainer
from miol.model.probe_specs import PROBE_SPECS
from miol.statistical_baseline import (
    _safe_mape,
    _safe_smape,
    compute_train_mean_completion_length,
    compute_train_mean_token_target,
    compute_train_median_completion_length,
    compute_train_median_token_target,
)


# -------------------------
# Probe inference helpers
# -------------------------


def build_probe_input(spec: dict, hidden_states: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Build the input tensor for a probe head given its spec.

    Mirrors MultiProbeOfflineTrainer._build_probe_input but operates on a single
    (unbatched) sequence of shape [seq_len, hidden_size].
    Returns [seq_len, input_dim] (input_dim is hidden_size / 1 / 1 depending on type).
    """
    input_type = spec.get("input_type", "hidden_states")
    seq_len = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype

    if input_type == "hidden_states":
        return hidden_states
    if input_type == "constant":
        return torch.ones(seq_len, 1, device=device, dtype=dtype)
    if input_type == "position":
        seq_idx = torch.arange(seq_len, device=device)
        completion_position = (seq_idx - prompt_length).clamp(min=0)
        return completion_position.unsqueeze(-1).to(dtype)
    raise ValueError(f"Unknown input_type: {input_type}")


def load_probe_heads(probe_dir: Path, active_probes: list[str], hidden_size: int) -> dict[str, torch.nn.Linear]:
    """Load each saved probe head into a fresh Linear layer."""
    heads: dict[str, torch.nn.Linear] = {}
    for name in active_probes:
        spec = PROBE_SPECS[name]
        input_type = spec.get("input_type", "hidden_states")
        if input_type == "hidden_states":
            in_dim = hidden_size
        elif input_type in ("constant", "position"):
            in_dim = 1
        else:
            raise ValueError(f"Unknown input_type for probe {name!r}: {input_type}")

        head = torch.nn.Linear(in_dim, spec["output_dim"])
        ckpt_path = probe_dir / f"{name}_head.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing probe checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        head.load_state_dict(state["head"])
        head.eval()
        heads[name] = head
    return heads


# -------------------------
# Per-token metric helpers
# -------------------------


def compute_non_monotonicity(series: list[float], expected_direction: str = "decreasing") -> dict:
    """Per-token non-monotonicity statistics.

    Quantifies how often (and how strongly) a per-token prediction series moves
    *against* its expected direction. For the count / percentage probes the
    ground truth target is monotonically decreasing in $t$, so any upward jump
    in the prediction is information that no position-only predictor can
    reproduce — a position baseline always satisfies `nm_score = 0` by
    construction. For classification probes' derived completion-progress, the
    expected direction is increasing.

    Args:
        series: list of per-token prediction values.
        expected_direction: "decreasing" (count, percentage, implied remaining,
            static decay) or "increasing" (cls derived progress).

    Returns:
        dict with:
            nm_score:               sum of jumps against expected direction (>= 0).
            nm_max:                 largest single such jump.
            nm_count:               number of positions with a violating jump.
            nm_score_per_token:     nm_score / max(len(series) - 1, 1).
            nm_max_relative:        nm_max / max(|series[0] - series[-1]|, eps),
                                    a scale-free measure of "biggest local
                                    reversal vs. total range traversed."
    """
    if not series or len(series) < 2:
        return {
            "nm_score": 0.0,
            "nm_max": 0.0,
            "nm_count": 0,
            "nm_score_per_token": 0.0,
            "nm_max_relative": 0.0,
        }
    diffs = [series[i] - series[i - 1] for i in range(1, len(series))]
    if expected_direction == "decreasing":
        violations = [max(d, 0.0) for d in diffs]
    elif expected_direction == "increasing":
        violations = [max(-d, 0.0) for d in diffs]
    else:
        raise ValueError(f"Unknown expected_direction: {expected_direction!r}")

    score = sum(violations)
    nm_max = max(violations) if violations else 0.0
    range_ = abs(series[0] - series[-1])
    return {
        "nm_score": float(score),
        "nm_max": float(nm_max),
        "nm_count": int(sum(1 for v in violations if v > 0)),
        "nm_score_per_token": float(score / max(len(diffs), 1)),
        "nm_max_relative": float(nm_max / range_) if range_ > 0 else 0.0,
    }


def compute_regression_metrics(predictions: list[float], targets: list[float], epsilon: float = 1e-8) -> dict:
    """Compute per-token MAE/MAPE/SMAPE and aggregate scalars (full + last-10).

    Aggregate MAE/MSE: per-token mean. MAPE/SMAPE: per-sequence mean-of-means
    (we're working on a single sequence here, so it's just the mean of per-token
    values — matches the trainer / statistical baseline convention).
    """
    assert len(predictions) == len(targets), f"{len(predictions)} vs {len(targets)}"
    mae_per_token = [abs(p - t) for p, t in zip(predictions, targets)]
    mse_per_token = [(p - t) ** 2 for p, t in zip(predictions, targets)]
    mape_per_token = [_safe_mape(t, p, epsilon) for p, t in zip(predictions, targets)]
    smape_per_token = [_safe_smape(t, p, epsilon) for p, t in zip(predictions, targets)]

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    def _mean_last(xs: list[float], k: int = 10) -> float | None:
        return _mean(xs[-k:])

    mae_scalar = _mean(mae_per_token)
    mse_scalar = _mean(mse_per_token)

    # Non-monotonicity: regression probes target a monotonically decreasing
    # series, so any upward motion in the prediction is dynamic re-estimation —
    # a signal no position-only predictor can produce.
    nm = compute_non_monotonicity(predictions, expected_direction="decreasing")

    return {
        "mae_per_token": mae_per_token,
        "mse_per_token": mse_per_token,
        "mape_per_token": mape_per_token,
        "smape_per_token": smape_per_token,
        "aggregate": {
            "mae": mae_scalar,
            "mse": mse_scalar,
            "mape": _mean(mape_per_token),
            "smape": _mean(smape_per_token),
            "mae_last_10": _mean_last(mae_per_token),
            "mse_last_10": _mean_last(mse_per_token),
            "mape_last_10": _mean_last(mape_per_token),
            "smape_last_10": _mean_last(smape_per_token),
            "nm_score": nm["nm_score"],
            "nm_max": nm["nm_max"],
            "nm_count": nm["nm_count"],
            "nm_score_per_token": nm["nm_score_per_token"],
            "nm_max_relative": nm["nm_max_relative"],
        },
    }


def compute_prompt_only_probe_metrics(
    preds_all: list[float],
    prompt_length: int,
    completion_length: int,
    count_targets: list[float] | None,
    T: int | None,
    has_eos: bool,
    epsilon: float = 1e-8,
) -> dict:
    """Build the dashboard entry for a prompt-end-trained regression probe.

    The probe predicts total completion length T. We expose three groups of
    numbers:

    - prompt_*: error at the single prompt-end position (the headline number;
      "how well does the model predict T before generating?").
    - completion_*_T: error at every completion position with target = T (the
      same constant). Tests whether the prompt-trained direction is preserved
      mid-completion.
    - completion_*_remaining: error treating the per-position prediction as
      \\hat{T}_t and converting to implied remaining (\\hat{T}_t - t - 1) before
      comparing to r_t. MAE is mathematically identical to completion_mae_T,
      but MAPE/SMAPE differ because the denominator changes.
    """
    # Predictions exposed for the dashboard, sliced to completion tokens.
    preds_completion = preds_all[prompt_length:prompt_length + completion_length]

    # Headline: prediction at the prompt's last hidden state.
    prompt_end_pred = preds_all[prompt_length - 1] if prompt_length >= 1 else None

    out: dict[str, Any] = {
        "predictions": _clean_list(preds_completion),
        "prompt_end_prediction": _clean_float(prompt_end_pred),
    }

    if not has_eos or T is None or count_targets is None or prompt_end_pred is None:
        out["aggregate"] = None
        return out

    # Per-completion-position errors. Implied-remaining at completion index k:
    #   max(\hat{T}_k - k - 1, 0)
    # which compared to count_targets[k] = T - k - 1 yields the same absolute
    # error as |\hat{T}_k - T|; we expose both for clarity.
    mae_per_token_T = [abs(p - T) for p in preds_completion]
    mse_per_token_T = [(p - T) ** 2 for p in preds_completion]
    implied_remaining = [max(p - k - 1, 0.0) for k, p in enumerate(preds_completion)]
    mae_per_token_remaining = [abs(ir - tgt) for ir, tgt in zip(implied_remaining, count_targets)]
    mse_per_token_remaining = [(ir - tgt) ** 2 for ir, tgt in zip(implied_remaining, count_targets)]

    mape_per_token_T = [_safe_mape(T, p, epsilon) for p in preds_completion]
    smape_per_token_T = [_safe_smape(T, p, epsilon) for p in preds_completion]
    mape_per_token_remaining = [
        _safe_mape(tgt, ir, epsilon) for ir, tgt in zip(implied_remaining, count_targets)
    ]
    smape_per_token_remaining = [
        _safe_smape(tgt, ir, epsilon) for ir, tgt in zip(implied_remaining, count_targets)
    ]

    # Static decay: prompt-end \hat{T}_0 broadcast to every completion position
    # with deterministic decrement max(\hat{T}_0 - k - 1, 0). The "no signal
    # from the residual stream during generation" floor — distinct from the
    # implied-remaining series, which uses the per-position prediction \hat{T}_k.
    static_decay = [max(prompt_end_pred - k - 1, 0.0) for k in range(len(count_targets))]
    mae_per_token_static = [abs(s - tgt) for s, tgt in zip(static_decay, count_targets)]
    mse_per_token_static = [(s - tgt) ** 2 for s, tgt in zip(static_decay, count_targets)]
    mape_per_token_static = [_safe_mape(tgt, s, epsilon) for s, tgt in zip(static_decay, count_targets)]
    smape_per_token_static = [_safe_smape(tgt, s, epsilon) for s, tgt in zip(static_decay, count_targets)]

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    def _mean_last(xs: list[float], k: int = 10) -> float | None:
        return _mean(xs[-k:])

    prompt_err = abs(prompt_end_pred - T)
    prompt_sq_err = (prompt_end_pred - T) ** 2
    out["mae_per_token"] = _clean_list(mae_per_token_remaining)
    out["mse_per_token"] = _clean_list(mse_per_token_remaining)
    out["mape_per_token"] = _clean_list(mape_per_token_remaining)
    out["smape_per_token"] = _clean_list(smape_per_token_remaining)
    out["mae_per_token_T"] = _clean_list(mae_per_token_T)
    out["mse_per_token_T"] = _clean_list(mse_per_token_T)
    out["static_decay_predictions"] = _clean_list(static_decay)
    out["static_decay_mae_per_token"] = _clean_list(mae_per_token_static)
    out["static_decay_mse_per_token"] = _clean_list(mse_per_token_static)
    out["static_decay_mape_per_token"] = _clean_list(mape_per_token_static)
    out["static_decay_smape_per_token"] = _clean_list(smape_per_token_static)

    # Non-monotonicity on the per-position implied-remaining series (this is
    # the prompt-only probe's "comparable to the count probe" view). The
    # static-decay series is monotonically decreasing by construction, so its
    # nm_score is exactly 0 — included as a sanity-check baseline.
    nm_implied = compute_non_monotonicity(implied_remaining, expected_direction="decreasing")
    nm_static = compute_non_monotonicity(static_decay, expected_direction="decreasing")

    out["aggregate"] = {
        "prompt_mae": _clean_float(prompt_err),
        "prompt_mse": _clean_float(prompt_sq_err),
        "prompt_mape": _clean_float(_safe_mape(T, prompt_end_pred, epsilon)),
        "prompt_smape": _clean_float(_safe_smape(T, prompt_end_pred, epsilon)),
        "completion_mae_T": _clean_float(_mean(mae_per_token_T)),
        "completion_mse_T": _clean_float(_mean(mse_per_token_T)),
        "completion_mape_T": _clean_float(_mean(mape_per_token_T)),
        "completion_smape_T": _clean_float(_mean(smape_per_token_T)),
        "completion_mae_remaining": _clean_float(_mean(mae_per_token_remaining)),
        "completion_mse_remaining": _clean_float(_mean(mse_per_token_remaining)),
        "completion_mape_remaining": _clean_float(_mean(mape_per_token_remaining)),
        "completion_smape_remaining": _clean_float(_mean(smape_per_token_remaining)),
        "static_decay_mae": _clean_float(_mean(mae_per_token_static)),
        "static_decay_mse": _clean_float(_mean(mse_per_token_static)),
        "static_decay_mape": _clean_float(_mean(mape_per_token_static)),
        "static_decay_smape": _clean_float(_mean(smape_per_token_static)),
        "mae_last_10": _clean_float(_mean_last(mae_per_token_remaining)),
        "mse_last_10": _clean_float(_mean_last(mse_per_token_remaining)),
        "mape_last_10": _clean_float(_mean_last(mape_per_token_remaining)),
        "smape_last_10": _clean_float(_mean_last(smape_per_token_remaining)),
        "nm_score": _clean_float(nm_implied["nm_score"]),
        "nm_max": _clean_float(nm_implied["nm_max"]),
        "nm_count": nm_implied["nm_count"],
        "nm_score_per_token": _clean_float(nm_implied["nm_score_per_token"]),
        "nm_max_relative": _clean_float(nm_implied["nm_max_relative"]),
        "nm_score_static_decay": _clean_float(nm_static["nm_score"]),
        # Aliases so dashboard sort menus work without special-casing.
        "mae": _clean_float(_mean(mae_per_token_remaining)),
        "mse": _clean_float(_mean(mse_per_token_remaining)),
        "mape": _clean_float(_mean(mape_per_token_remaining)),
        "smape": _clean_float(_mean(smape_per_token_remaining)),
    }
    return out


def _build_confusion(pred_classes: list[int], true_classes: list[int], num_bins: int) -> np.ndarray:
    """[num_bins, num_bins] confusion matrix, indexed [true, pred]."""
    cm = np.zeros((num_bins, num_bins), dtype=np.int64)
    if pred_classes:
        np.add.at(cm, (np.asarray(true_classes, dtype=np.int64), np.asarray(pred_classes, dtype=np.int64)), 1)
    return cm


def compute_classification_metrics(
    pred_classes: list[int],
    true_classes: list[int],
    derived_pct: list[float],
    pct_targets: list[float],
    num_bins: int,
    epsilon: float = 1e-8,
) -> dict:
    """Compute per-token correctness, derived-pct MAE, and aggregates including
    Cohen's kappa (chance-corrected accuracy). Aggregates are reported full and
    over the last 10 tokens.
    """
    correct_per_token = [int(p == t) for p, t in zip(pred_classes, true_classes)]
    derived_pct_mae_per_token = [abs(d - g) for d, g in zip(derived_pct, pct_targets)]
    derived_pct_mse_per_token = [(d - g) ** 2 for d, g in zip(derived_pct, pct_targets)]

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    def _mean_last(xs: list[float], k: int = 10) -> float | None:
        return _mean(xs[-k:])

    # Per-example Cohen's kappa from the full sequence and from the last 10 tokens.
    # Note: kappa over very few tokens (last-10) is noisy; treat as diagnostic.
    cm_full = _build_confusion(pred_classes, true_classes, num_bins)
    kappa_full = MultiProbeOfflineTrainer._compute_cohen_kappa(cm_full)
    last10_pred = pred_classes[-10:]
    last10_true = true_classes[-10:]
    cm_last10 = _build_confusion(last10_pred, last10_true, num_bins)
    kappa_last10 = (
        MultiProbeOfflineTrainer._compute_cohen_kappa(cm_last10) if last10_pred else None
    )

    # Non-monotonicity on the derived completion-progress series. cls probes
    # target a monotonically *increasing* progress signal, so violations are
    # downward jumps in derived_pct.
    nm = compute_non_monotonicity(derived_pct, expected_direction="increasing")

    return {
        "correct_per_token": correct_per_token,
        "derived_pct_mae_per_token": derived_pct_mae_per_token,
        "derived_pct_mse_per_token": derived_pct_mse_per_token,
        "confusion_matrix": cm_full.tolist(),
        "aggregate": {
            "accuracy": _mean(correct_per_token),
            "accuracy_last_10": _mean_last(correct_per_token),
            "cohen_kappa": kappa_full,
            "cohen_kappa_last_10": kappa_last10,
            "derived_pct_mae": _mean(derived_pct_mae_per_token),
            "derived_pct_mae_last_10": _mean_last(derived_pct_mae_per_token),
            "derived_pct_mse": _mean(derived_pct_mse_per_token),
            "derived_pct_mse_last_10": _mean_last(derived_pct_mse_per_token),
            "nm_score": nm["nm_score"],
            "nm_max": nm["nm_max"],
            "nm_count": nm["nm_count"],
            "nm_score_per_token": nm["nm_score_per_token"],
            "nm_max_relative": nm["nm_max_relative"],
        },
    }


# -------------------------
# Statistical baselines
# -------------------------


def build_statistical_baselines(train_dir: Path) -> dict[str, dict]:
    """Fit the 8 statistical baselines on the training split and return a dict
    of (name -> {predictor_type, constant_value | base_length}).
    """
    baseline_mean, _stats = compute_train_mean_token_target(train_dir)
    baseline_median = compute_train_median_token_target(train_dir)
    mean_completion_length = compute_train_mean_completion_length(train_dir)
    median_completion_length = compute_train_median_completion_length(train_dir)

    # 4 "token-level" baselines: natural predictor types from the training script
    # 4 "seqlen" baselines: each treated as length_minus_position with the seqlen
    # constant as base_length (as in the latest statistical_baseline.py).
    return {
        "baseline_constant_mean":        {"predictor_type": "constant",              "constant_value": baseline_mean},
        "baseline_constant_median":      {"predictor_type": "constant",              "constant_value": baseline_median},
        "baseline_mean_plus_position":   {"predictor_type": "length_minus_position", "base_length":    mean_completion_length},
        "baseline_median_plus_position": {"predictor_type": "length_minus_position", "base_length":    median_completion_length},
        "seqlen_constant_mean":          {"predictor_type": "length_minus_position", "base_length":    baseline_mean},
        "seqlen_constant_median":        {"predictor_type": "length_minus_position", "base_length":    baseline_median},
        "seqlen_mean_completion":        {"predictor_type": "length_minus_position", "base_length":    mean_completion_length},
        "seqlen_median_completion":      {"predictor_type": "length_minus_position", "base_length":    median_completion_length},
    }


def predict_statistical(spec: dict, completion_length: int) -> list[float]:
    """Compute the per-token predictions for a statistical baseline on one sequence."""
    if spec["predictor_type"] == "constant":
        c = spec["constant_value"]
        return [float(c)] * completion_length
    if spec["predictor_type"] == "length_minus_position":
        b = spec["base_length"]
        return [max(b - k - 1, 0.0) for k in range(completion_length)]
    raise ValueError(f"Unknown predictor_type: {spec['predictor_type']}")


# -------------------------
# Main evaluation loop
# -------------------------


def _clean_float(x: float | int | None) -> float | int | None:
    """Ensure JSON-safe floats (no NaN / inf)."""
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if math.isnan(x) or math.isinf(x):
        return None
    return float(x)


def _fmt_duration(seconds: float) -> str:
    """Compact `HhMMmSSs` / `MmSSs` / `Ss` format for progress output."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _clean_list(xs: list[float]) -> list[float | None]:
    return [_clean_float(x) for x in xs]


def _merge_scalar(values: list[Any]) -> dict[str, float] | None:
    """Aggregate a list of per-seed scalar values into a `{mean, std}` dict.

    Drops `None` / NaN / Inf entries. Returns `None` when nothing valid remains.
    For a single valid value, std is 0 (population std semantics).
    """
    cleaned: list[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            cleaned.append(float(v))
            continue
        if isinstance(v, (int, float)) and not math.isnan(float(v)) and not math.isinf(float(v)):
            cleaned.append(float(v))
    if not cleaned:
        return None
    return {
        "mean": statistics.fmean(cleaned),
        "std": statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0,
    }


def _merge_aggregate_dict(per_seed: list[dict | None]) -> dict | None:
    """Merge per-seed aggregate dicts into a single dict whose values are
    `{mean, std}` per scalar key. Falls back to None when every seed reported
    `None` for the example (e.g. has_eos=False).
    """
    if all(d is None for d in per_seed):
        return None
    keys: set[str] = set()
    for d in per_seed:
        if isinstance(d, dict):
            keys.update(d.keys())
    out: dict[str, Any] = {}
    for k in keys:
        vals = [(d.get(k) if isinstance(d, dict) else None) for d in per_seed]
        merged = _merge_scalar(vals)
        if merged is not None:
            out[k] = merged
    return out


def _merge_examples(per_seed_examples: list[list[dict]]) -> list[dict]:
    """Walk per-seed example lists and merge into a single list with
    `{mean, std}` aggregate scalars. Per-token arrays and per-example
    metadata come from seed 0.
    """
    if not per_seed_examples:
        return []
    n_examples = len(per_seed_examples[0])
    if any(len(s) != n_examples for s in per_seed_examples):
        raise ValueError(
            "Per-seed example lists must have the same length; got "
            f"{[len(s) for s in per_seed_examples]}"
        )

    merged: list[dict] = []
    for i in range(n_examples):
        base = per_seed_examples[0][i]
        new_ex: dict[str, Any] = dict(base)
        # Merge `probes` aggregate scalars across seeds; per-token arrays come
        # from seed 0 (shared hidden states cache → seed-invariant inputs).
        merged_probes: dict[str, Any] = {}
        for probe_name, base_entry in (base.get("probes") or {}).items():
            new_entry = dict(base_entry) if isinstance(base_entry, dict) else base_entry
            if isinstance(base_entry, dict):
                # Aggregate dict: scalar → {mean, std}.
                if "aggregate" in base_entry:
                    per_seed_agg = [
                        ((s[i].get("probes") or {}).get(probe_name) or {}).get("aggregate")
                        for s in per_seed_examples
                    ]
                    new_entry["aggregate"] = _merge_aggregate_dict(per_seed_agg)
                # `prompt_end_prediction` is a top-level scalar on prompt-only
                # probes (not nested under aggregate). Merge it separately.
                if "prompt_end_prediction" in base_entry:
                    per_seed_pep = [
                        ((s[i].get("probes") or {}).get(probe_name) or {}).get("prompt_end_prediction")
                        for s in per_seed_examples
                    ]
                    merged_pep = _merge_scalar(per_seed_pep)
                    if merged_pep is not None:
                        new_entry["prompt_end_prediction"] = merged_pep
            merged_probes[probe_name] = new_entry
        new_ex["probes"] = merged_probes
        # `statistical_baselines` are seed-invariant (fit on the same train
        # split metadata regardless of probe seed) — pass through from seed 0.
        merged.append(new_ex)
    return merged


def _apply_heads_to_sample(
    sample: dict,
    heads: dict[str, torch.nn.Linear],
    cls_anchors: dict[str, torch.Tensor],
    *,
    has_eos: bool,
    prompt_length: int,
    completion_length: int,
    count_targets: list[float] | None,
    pct_targets: list[float] | None,
) -> dict[str, dict[str, Any]]:
    """Apply every probe head in `heads` to one already-loaded sample and
    return the per-probe sub-dict that goes into `example["probes"]`.

    Pulled out of `evaluate_one_seed` so the same per-example work can be
    shared across many probe-dirs in `evaluate_many_probe_dirs` (the eval
    dataset's `__getitem__` is the dominant cost — once we have a sample
    in hand, applying N heads is O(N) cheap matmuls).
    """
    hidden_states = sample["hidden_states"]  # [seq_len, hidden_size]
    probes_out: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        for name, head in heads.items():
            spec = PROBE_SPECS[name]
            probe_in = build_probe_input(spec, hidden_states, prompt_length)
            logits = head(probe_in.float())  # [seq_len, output_dim]

            if spec["type"] == "regression":
                preds_all = logits.squeeze(-1).cpu().tolist()

                # Prompt-end-trained probe: predicts T from the last prompt
                # token; we additionally evaluate it at every completion
                # position to test whether the prompt-trained direction
                # transfers to mid-completion residuals.
                if spec.get("train_position") == "prompt_end":
                    probes_out[name] = compute_prompt_only_probe_metrics(
                        preds_all, prompt_length, completion_length,
                        count_targets if has_eos else None,
                        T=completion_length if has_eos else None,
                        has_eos=has_eos,
                    )
                    continue

                # Slice to completion tokens only
                preds_completion = preds_all[prompt_length:prompt_length + completion_length]
                # log_count probes were trained on log(1+count); map predictions
                # back to count space so the stored `predictions` and all
                # metrics are directly comparable to the `count` probe.
                if spec["target_key"] == "remaining_log_count":
                    preds_completion = [max(math.expm1(p), 0.0) for p in preds_completion]
                probe_entry: dict[str, Any] = {"predictions": _clean_list(preds_completion)}
                if has_eos:
                    if spec["target_key"] in ("remaining_token_counts", "remaining_log_count"):
                        metrics = compute_regression_metrics(preds_completion, count_targets)
                    else:  # remaining_percentage
                        metrics = compute_regression_metrics(preds_completion, pct_targets)
                    probe_entry["mae_per_token"] = _clean_list(metrics["mae_per_token"])
                    probe_entry["mse_per_token"] = _clean_list(metrics["mse_per_token"])
                    probe_entry["mape_per_token"] = _clean_list(metrics["mape_per_token"])
                    probe_entry["smape_per_token"] = _clean_list(metrics["smape_per_token"])
                    probe_entry["aggregate"] = {k: _clean_float(v) for k, v in metrics["aggregate"].items()}
                else:
                    probe_entry["aggregate"] = None
                probes_out[name] = probe_entry

            else:  # classification
                n_bins = spec["output_dim"]
                anchors = cls_anchors[name]
                probs_all = torch.softmax(logits, dim=-1)          # [seq_len, n_bins]
                derived_all = (probs_all * anchors).sum(dim=-1)     # [seq_len]
                argmax_all = probs_all.argmax(dim=-1)               # [seq_len]

                probs_completion = probs_all[prompt_length:prompt_length + completion_length].cpu().tolist()
                derived_completion = derived_all[prompt_length:prompt_length + completion_length].cpu().tolist()
                argmax_completion = argmax_all[prompt_length:prompt_length + completion_length].cpu().tolist()

                probe_entry = {
                    "predictions_probs": [[_clean_float(p) for p in row] for row in probs_completion],
                    "predictions_argmax": argmax_completion,
                    "derived_percentage": _clean_list(derived_completion),
                }
                if has_eos:
                    # Classification probes are trained on *completion progress*
                    # (anchor=0 at the start of the completion, anchor=1 at the end),
                    # not on the regression percentage probe's *remaining percentage*
                    # convention. See compute_soft_class_targets in generation_utils.
                    if completion_length > 1:
                        cls_progress_targets = [k / (completion_length - 1) for k in range(completion_length)]
                    else:
                        cls_progress_targets = [0.0]

                    anchors_list = anchors.tolist()
                    true_classes = []
                    for pct in cls_progress_targets:
                        best = 0
                        best_d = abs(anchors_list[0] - pct)
                        for i in range(1, n_bins):
                            d = abs(anchors_list[i] - pct)
                            if d < best_d:
                                best_d = d
                                best = i
                        true_classes.append(best)

                    cls_metrics = compute_classification_metrics(
                        argmax_completion, true_classes, derived_completion, cls_progress_targets,
                        num_bins=n_bins,
                    )
                    probe_entry["ground_truth_argmax"] = true_classes
                    probe_entry["correct_per_token"] = cls_metrics["correct_per_token"]
                    probe_entry["derived_pct_mae_per_token"] = _clean_list(cls_metrics["derived_pct_mae_per_token"])
                    probe_entry["derived_pct_mse_per_token"] = _clean_list(cls_metrics["derived_pct_mse_per_token"])
                    probe_entry["confusion_matrix"] = cls_metrics["confusion_matrix"]
                    probe_entry["aggregate"] = {k: _clean_float(v) for k, v in cls_metrics["aggregate"].items()}
                else:
                    probe_entry["aggregate"] = None
                probes_out[name] = probe_entry

    return probes_out


def evaluate_many_probe_dirs(
    probe_dirs: list[Path],
    eval_dataset: HiddenStatesDataset,
    hidden_size: int,
    active_probes: list[str],
    classification_bins: dict[str, int],
    cls_anchors: dict[str, torch.Tensor],
    stat_baselines: dict,
    tokenizer,
) -> dict[Path, tuple[list[dict], dict[str, float]]]:
    """Evaluate many probe-dirs in a single pass over the eval dataset.

    `eval_dataset[idx]` (hidden-states load + per-layer concat for
    `hidden_layer="all"`) is the dominant per-example cost. By looping over
    examples *once* and applying every probe-dir's heads to each loaded
    sample, the work scales as O(N_examples) instead of
    O(N_examples × N_probe_dirs). Per-dir output shape matches what
    `evaluate_one_seed` returns today, so callers (the seed-merge layer in
    `evaluate()` and the cross-dataset eval loop) can consume each entry
    as a drop-in seed.

    A probe-dir whose `load_probe_heads` call raises (typically a missing
    head checkpoint) is skipped with a warning and absent from the result;
    callers should treat its absence as "this seed didn't run."

    Seed-invariant per-example fields (`tokens_decoded`, `ground_truth`,
    `statistical_baselines`) are computed once and shared (by reference)
    across the per-dir example dicts. The downstream seed-merge layer
    treats these as immutable, so reference-sharing is safe.
    """
    if not probe_dirs:
        return {}

    # Pre-load every probe-dir's heads. Skip dirs whose head files are missing
    # (the cross-dataset eval caller already filters by the plotted probe's
    # head; this catches missing other-probe heads gracefully too).
    heads_per_dir: dict[Path, dict[str, torch.nn.Linear]] = {}
    for pd in probe_dirs:
        try:
            heads_per_dir[pd] = load_probe_heads(pd, active_probes, hidden_size)
        except FileNotFoundError as e:
            print(f"  [skip] {pd}: {e}")

    if not heads_per_dir:
        return {}

    examples_per_dir: dict[Path, list[dict[str, Any]]] = {pd: [] for pd in heads_per_dir}

    # Progress: emit ~50 lines over the run (every K examples), and at least
    # every 30 s so a slow per-example loop still gives feedback. Time-based
    # ETA / elapsed lets the operator spot stalls and budget cluster time.
    n_total = len(eval_dataset)
    n_dirs = len(heads_per_dir)
    print(f"  evaluating {n_total} examples × {n_dirs} probe-dirs...")
    start_time = time.monotonic()
    last_print_time = start_time
    print_every_n = max(1, n_total // 50)
    print_every_secs = 30.0

    for idx in range(n_total):
        sample = eval_dataset[idx]  # ONE expensive load shared across every probe-dir
        full_ids = sample["input_ids"].tolist()
        prompt_length = int(sample["prompt_length"])
        completion_length = int(sample["completion_length"])
        has_eos = bool(sample["has_eos"])

        tokens_decoded = [tokenizer.decode([tid]) for tid in full_ids]

        # Ground truth for completion tokens — seed-invariant.
        if has_eos and completion_length > 0:
            count_targets: list[float] | None = [
                float(completion_length - k - 1) for k in range(completion_length)
            ]
            if completion_length > 1:
                pct_targets: list[float] | None = [
                    (completion_length - k - 1) / (completion_length - 1)
                    for k in range(completion_length)
                ]
            else:
                pct_targets = [0.0]
            ground_truth: dict[str, Any] = {
                "remaining_count": count_targets,
                "percentage": pct_targets,
            }
        else:
            count_targets = None
            pct_targets = None
            ground_truth = {"remaining_count": None, "percentage": None}

        # Statistical baselines — seed-invariant; computed once per example.
        baselines_out: dict[str, dict[str, Any]] = {}
        for bname, bspec in stat_baselines.items():
            preds = predict_statistical(bspec, completion_length)
            entry: dict[str, Any] = {"predictions": _clean_list(preds)}
            if has_eos:
                metrics = compute_regression_metrics(preds, count_targets)
                entry["mae_per_token"] = _clean_list(metrics["mae_per_token"])
                entry["mse_per_token"] = _clean_list(metrics["mse_per_token"])
                entry["mape_per_token"] = _clean_list(metrics["mape_per_token"])
                entry["smape_per_token"] = _clean_list(metrics["smape_per_token"])
                entry["aggregate"] = {k: _clean_float(v) for k, v in metrics["aggregate"].items()}
            else:
                entry["aggregate"] = None
            baselines_out[bname] = entry

        # Apply each probe-dir's heads to the same loaded sample.
        for pd, heads in heads_per_dir.items():
            probes_out = _apply_heads_to_sample(
                sample=sample,
                heads=heads,
                cls_anchors=cls_anchors,
                has_eos=has_eos,
                prompt_length=prompt_length,
                completion_length=completion_length,
                count_targets=count_targets,
                pct_targets=pct_targets,
            )
            examples_per_dir[pd].append({
                "id": idx,
                "prompt": sample["prompt"],
                "completion": sample["completion"],
                "tokens_decoded": tokens_decoded,
                "prompt_length": prompt_length,
                "completion_length": completion_length,
                "has_eos": has_eos,
                "ground_truth": ground_truth,
                "probes": probes_out,
                "statistical_baselines": baselines_out,
            })

        n_done = idx + 1
        now = time.monotonic()
        if (
            n_done == n_total
            or n_done % print_every_n == 0
            or (now - last_print_time) >= print_every_secs
        ):
            elapsed = now - start_time
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (n_total - n_done) / rate if rate > 0 else 0.0
            pct = 100.0 * n_done / n_total
            print(
                f"  [{n_done:>5d}/{n_total:<5d} {pct:5.1f}%]  "
                f"elapsed {_fmt_duration(elapsed):>7s}  "
                f"ETA {_fmt_duration(eta):>7s}  "
                f"({rate:5.2f} ex/s × {n_dirs} probe-dirs)",
                flush=True,
            )
            last_print_time = now

    # Per-dir dataset-wide Cohen's kappa — accumulate confusion matrices across
    # the dir's examples (each ex's `probes[name]["confusion_matrix"]` is dir-
    # specific, so the totals naturally separate by dir).
    result: dict[Path, tuple[list[dict], dict[str, float]]] = {}
    for pd, examples_pd in examples_per_dir.items():
        kappa: dict[str, float] = {}
        for name, n_bins in classification_bins.items():
            cm_total = np.zeros((n_bins, n_bins), dtype=np.int64)
            for ex in examples_pd:
                entry = ex.get("probes", {}).get(name)
                if entry and entry.get("confusion_matrix"):
                    cm_total += np.asarray(entry["confusion_matrix"], dtype=np.int64)
            if cm_total.sum() > 0:
                kappa[name] = float(MultiProbeOfflineTrainer._compute_cohen_kappa(cm_total))
        result[pd] = (examples_pd, kappa)

    return result


def evaluate_one_seed(
    probe_dir: Path,
    eval_dataset: HiddenStatesDataset,
    hidden_size: int,
    active_probes: list[str],
    classification_bins: dict[str, int],
    cls_anchors: dict[str, torch.Tensor],
    stat_baselines: dict,
    tokenizer,
) -> tuple[list[dict], dict[str, float]]:
    """Run the per-example evaluation for one trained-probe checkpoint.

    Backward-compatible thin wrapper around `evaluate_many_probe_dirs` for
    callers that handle one probe-dir at a time. New code should prefer
    `evaluate_many_probe_dirs` to avoid re-loading the eval dataset for
    each seed / probe-dir.
    """
    results = evaluate_many_probe_dirs(
        probe_dirs=[probe_dir],
        eval_dataset=eval_dataset,
        hidden_size=hidden_size,
        active_probes=active_probes,
        classification_bins=classification_bins,
        cls_anchors=cls_anchors,
        stat_baselines=stat_baselines,
        tokenizer=tokenizer,
    )
    if probe_dir not in results:
        # `evaluate_many_probe_dirs` skipped the dir (missing heads). Surface
        # as the underlying error rather than returning empty results, so the
        # caller's error-handling path is the same as it was pre-refactor.
        load_probe_heads(probe_dir, active_probes, hidden_size)  # re-raise
        raise FileNotFoundError(f"Probe dir not evaluated: {probe_dir}")
    return results[probe_dir]


def evaluate(args: argparse.Namespace) -> None:
    # `--probe-dir` is `nargs='+'`, so always a list. One element = single-seed
    # mode (std=0 in every aggregate); >1 elements = multi-seed aggregation.
    probe_dirs = [Path(p) for p in args.probe_dir]
    hidden_states_dir = Path(args.hidden_states_dir)
    output_path = Path(args.output)

    # Load the training config from the FIRST probe dir; warn if subsequent
    # dirs have a different model / dataset / probe-list (still proceed —
    # users may legitimately compare seeds with slightly different settings).
    first_cfg_path = probe_dirs[0] / "config.yaml"
    if not first_cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {first_cfg_path}")
    cfg = OmegaConf.load(first_cfg_path)

    hidden_layer = str(cfg.data.get("hidden_layer", "-1"))
    model_name = cfg.model.model_name
    active_probes = list(cfg.optimization.get("probes", list(PROBE_SPECS.keys())))
    classification_sigma = float(cfg.optimization.get("classification_sigma", 0.15))

    print(f"Probes:     {active_probes}")
    print(f"Model:      {model_name}")
    print(f"Hidden layer: {hidden_layer}")
    print(f"Probe dirs: {len(probe_dirs)} ({'multi-seed aggregation' if len(probe_dirs) > 1 else 'single seed'})")

    # Verify other probe-dirs match the first; warn-and-continue if not.
    for pd in probe_dirs[1:]:
        cp = pd / "config.yaml"
        if not cp.exists():
            print(f"[warning] {cp} missing — proceeding with seed-0 config.")
            continue
        other = OmegaConf.load(cp)
        if other.model.model_name != model_name:
            print(f"[warning] {pd}: model differs ({other.model.model_name} vs {model_name})")
        if str(other.data.get("hidden_layer", "-1")) != hidden_layer:
            print(f"[warning] {pd}: hidden_layer differs")
        if list(other.optimization.get("probes", [])) != active_probes:
            print(f"[warning] {pd}: probes list differs")

    # Load eval dataset once — shared across seeds.
    eval_dataset = HiddenStatesDataset(
        hidden_states_dir=str(hidden_states_dir),
        split="eval",
        hidden_layer=hidden_layer,
    )
    if len(eval_dataset) == 0:
        raise ValueError(f"No eval samples found in {hidden_states_dir}")
    hidden_size = eval_dataset[0]["hidden_states"].shape[-1]
    print(f"Eval samples: {len(eval_dataset)}, hidden size: {hidden_size}")

    # Precompute classification anchors once.
    cls_anchors: dict[str, torch.Tensor] = {}
    classification_bins: dict[str, int] = {}
    for name in active_probes:
        if PROBE_SPECS[name]["type"] == "classification":
            n_bins = PROBE_SPECS[name]["output_dim"]
            cls_anchors[name] = torch.linspace(0, 1, n_bins)
            classification_bins[name] = n_bins

    # Statistical baselines fit on train split — seed-invariant.
    train_dir = hidden_states_dir / "train"
    print("Fitting statistical baselines on training split...")
    stat_baselines = build_statistical_baselines(train_dir)
    print(f"  {len(stat_baselines)} baselines")

    # Tokenizer: prefer the first probe-dir (so any saved tokenizer files take
    # precedence), fall back to the HF model name.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(probe_dirs[0]))
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -------- Run all seeds in one pass over the eval dataset --------
    print(f"\nEvaluating {len(probe_dirs)} probe-dir(s) in one pass...")
    results = evaluate_many_probe_dirs(
        probe_dirs=probe_dirs,
        eval_dataset=eval_dataset,
        hidden_size=hidden_size,
        active_probes=active_probes,
        classification_bins=classification_bins,
        cls_anchors=cls_anchors,
        stat_baselines=stat_baselines,
        tokenizer=tokenizer,
    )
    # Preserve probe-dir input order; drop dirs whose heads failed to load.
    per_seed_examples: list[list[dict]] = []
    per_seed_kappa: list[dict[str, float]] = []
    for probe_dir in probe_dirs:
        if probe_dir not in results:
            print(f"[warning] {probe_dir}: skipped (missing head checkpoint).")
            continue
        ex_s, kappa_s = results[probe_dir]
        per_seed_examples.append(ex_s)
        per_seed_kappa.append(kappa_s)
    if not per_seed_examples:
        raise RuntimeError("No probe-dirs produced results — cannot write output JSON.")

    # -------- Merge across seeds --------
    if len(probe_dirs) == 1:
        # Single-seed shortcut: still wrap aggregate scalars in {mean, std=0}
        # so the output JSON shape is consistent regardless of seed count.
        examples = _merge_examples(per_seed_examples)
    else:
        examples = _merge_examples(per_seed_examples)

    # Dataset kappa: merge per-name across seeds.
    dataset_cohen_kappa: dict[str, dict[str, float] | None] = {}
    all_kappa_names = set()
    for d in per_seed_kappa:
        all_kappa_names.update(d.keys())
    for name in sorted(all_kappa_names):
        merged = _merge_scalar([d.get(name) for d in per_seed_kappa])
        if merged is not None:
            dataset_cohen_kappa[name] = merged

    # -------- Write JSON --------
    output_data = {
        "metadata": {
            "model_name": model_name,
            "probe_dir": str(probe_dirs[0]) if len(probe_dirs) == 1 else None,
            "probe_dirs": [str(p) for p in probe_dirs],
            "n_seeds": len(probe_dirs),
            "seed_aggregated": len(probe_dirs) > 1,
            "hidden_states_dir": str(hidden_states_dir),
            "hidden_layer": hidden_layer,
            "probes": active_probes,
            "baselines": list(stat_baselines.keys()),
            "classification_bins": classification_bins,
            "classification_sigma": classification_sigma,
            "num_examples": len(examples),
            "dataset_cohen_kappa": dataset_cohen_kappa,
        },
        "examples": examples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output_data, f)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-dir",
        required=True,
        type=str,
        nargs="+",
        help=(
            "Path(s) to one or more trained_probe/ directories (each containing "
            "*_head.pt and config.yaml). When multiple paths are given, the "
            "evaluation is run once per probe-dir (treated as one seed each) and "
            "every aggregate scalar in the output JSON is replaced with "
            "{mean, std} computed across seeds. Per-token arrays and per-example "
            "metadata are taken from the first probe-dir."
        ),
    )
    parser.add_argument("--hidden-states-dir", required=True, type=str, help="Path to hidden_states directory (containing train/ and eval/).")
    parser.add_argument("--output", required=True, type=str, help="Output JSON file path.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
