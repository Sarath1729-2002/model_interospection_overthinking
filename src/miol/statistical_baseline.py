"""
Compute constant baselines from hidden-state metadata.

Probe-aligned baseline (default):
- Fit on training split using only samples with has_eos == True.
- Use all subsequence targets (remaining-token counts for each completion token).
- Predict one constant value for every token target.

Evaluation:
- Compute token-level MAPE on eval split (default: only samples with has_eos == True).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# All scalars and figures logged to wandb live under this top-level namespace,
# so a single wandb run can host both the trained-probe metrics and the
# baselines side-by-side without naming collisions.
WANDB_PREFIX = "statistical_baseline"


def is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in [-1, 0]


@dataclass
class SplitStats:
    # total_files: JSON files discovered in split directory
    # eos_files: files where has_eos=True
    # used_files: files included in metric computation after filtering
    # used_targets: token-level targets used (sum of completion lengths)
    total_files: int
    eos_files: int
    used_files: int
    used_targets: int


def _update_bucket(buckets: dict[int, tuple[float, int]], bucket_start: int, value: float) -> None:
    if bucket_start in buckets:
        prev_sum, prev_count = buckets[bucket_start]
        buckets[bucket_start] = (prev_sum + value, prev_count + 1)
    else:
        buckets[bucket_start] = (value, 1)


def _update_bucket_with_count(
    buckets: dict[int, tuple[float, int]],
    bucket_start: int,
    value_sum: float,
    value_count: int,
) -> None:
    if bucket_start in buckets:
        prev_sum, prev_count = buckets[bucket_start]
        buckets[bucket_start] = (prev_sum + value_sum, prev_count + value_count)
    else:
        buckets[bucket_start] = (value_sum, value_count)


def _bucket_means_and_counts(
    buckets: dict[int, tuple[float, int]],
) -> tuple[dict[int, float], dict[int, int]]:
    metric_per_bin: dict[int, float] = {}
    counts_per_bin: dict[int, int] = {}
    for bucket_start, (sum_values, count) in sorted(buckets.items()):
        metric_per_bin[bucket_start] = (sum_values / count) if count > 0 else 0.0
        counts_per_bin[bucket_start] = count
    return metric_per_bin, counts_per_bin


def _save_metric_chart(
    metric_per_bin: dict[int, float],
    counts_per_bin: dict[int, int],
    split_name: str,
    metric: str,
    output_path: Path,
) -> None:
    if not metric_per_bin:
        return

    bins = sorted(metric_per_bin.keys())
    metric_values = [metric_per_bin[b] for b in bins]
    count_values = [counts_per_bin[b] for b in bins]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    color = "steelblue" if split_name == "train" else "coral"
    ax1.plot(
        bins,
        metric_values,
        marker="o",
        linewidth=2,
        markersize=6,
        color=color,
        label=f"{split_name.capitalize()} {metric.upper()}",
    )
    ax1.set_xlabel("Token Length Bin Start", fontsize=12)
    ax1.set_ylabel("Mean Value", fontsize=12, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    bar_color = "lightsteelblue" if split_name == "train" else "lightcoral"
    ax2.bar(bins, count_values, alpha=0.3, color=bar_color, label="Sample Count", width=2)
    ax2.set_ylabel("Number of Samples", fontsize=12, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title(f"{metric.upper()} per Token Length Bin ({split_name.capitalize()}) - Baseline", fontsize=14)
    ax1.set_xticks(bins)
    ax1.set_xticklabels([str(b) for b in bins], rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, linestyle="--")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _log_figure_to_wandb(
    image_path: Path,
    split_name: str,
    metric: str,
    enabled: bool,
    key_prefix: str = "",
) -> None:
    if not enabled or not is_main_process():
        return

    # Lazy import keeps script runnable even when wandb is not installed.
    import wandb

    if wandb.run is None:
        return

    metric_key = f"{metric}_per_bin/{split_name}_chart"
    if key_prefix:
        metric_key = f"{key_prefix}/{metric_key}"
    # Single point of namespacing: every figure logged from this script is
    # routed under WANDB_PREFIX so it doesn't collide with the trained
    # probes' figures (which live under count/, percentage/, etc.).
    metric_key = f"{WANDB_PREFIX}/{metric_key}"

    # Anchor pre-training baselines at epoch 0 so they line up on the left edge
    # of charts when the x-axis is `train/epoch` (see CustomTrainer step-metric).
    wandb.log({metric_key: wandb.Image(str(image_path)), "train/epoch": 0.0})


def _iter_sample_json_files(split_dir: Path) -> Iterable[Path]:
    # Keep deterministic ordering for reproducible stats.
    return sorted(split_dir.glob("sample_*.json"))


def _load_metadata(path: Path) -> dict:
    # Each metadata file contains has_eos and completion_length, which are
    # sufficient for this baseline.
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mape(y_true: float, y_pred: float, epsilon: float) -> float:
    # Match trainer behavior for zero targets: denom 1.0 instead of 0.
    if y_true == 0:
        denom = 1.0
    else:
        denom = max(abs(y_true), epsilon)
    return abs(y_true - y_pred) / denom


def _safe_smape(y_true: float, y_pred: float, epsilon: float) -> float:
    # SMAPE: |error| / ((|pred| + |target|) / 2)
    # Falls back to 0 when both are zero (perfect prediction).
    denom = (abs(y_true) + abs(y_pred)) / 2.0
    if denom == 0:
        return 0.0
    return abs(y_true - y_pred) / max(denom, epsilon)


def _resolve_hidden_states_root(root: Path) -> Path:
    """
    Accept either:
    - a directory that directly contains train/ and eval/
    - a parent directory that contains hidden_states/train and hidden_states/eval
    """
    # Allow both direct split roots and parent directories.
    if (root / "train").is_dir() and (root / "eval").is_dir():
        return root

    nested = root / "hidden_states"
    if (nested / "train").is_dir() and (nested / "eval").is_dir():
        return nested

    raise ValueError(
        "Could not find train/ and eval/ splits. "
        f"Checked: {root} and {nested}"
    )


def compute_train_mean_token_target(train_dir: Path) -> tuple[float, SplitStats]:
    # Fit one constant predictor for all token-level remaining-count targets.
    json_files = _iter_sample_json_files(train_dir)

    total = 0
    eos_count = 0
    used = 0
    used_targets = 0
    target_sum = 0.0

    for file_path in json_files:
        total += 1
        meta = _load_metadata(file_path)

        # Match training: use only sequences where EOS was produced.
        has_eos = bool(meta.get("has_eos", False))
        if not has_eos:
            continue

        eos_count += 1
        completion_length = int(meta["completion_length"])

        # For completion length L, remaining-token targets are:
        # L-1, L-2, ..., 0  (L targets in total)
        # Sum = L * (L - 1) / 2
        target_sum += (completion_length * (completion_length - 1)) / 2.0
        used_targets += completion_length
        used += 1

    if used == 0 or used_targets == 0:
        raise ValueError(
            "No training samples with has_eos=True were found; cannot compute baseline mean."
        )

    return (target_sum / used_targets, SplitStats(total, eos_count, used, used_targets))


def compute_train_mean_completion_length(train_dir: Path) -> float:
    """Mean completion length over training samples with EOS."""
    json_files = _iter_sample_json_files(train_dir)

    total = 0.0
    count = 0
    for file_path in json_files:
        meta = _load_metadata(file_path)
        if not bool(meta.get("has_eos", False)):
            continue
        total += float(meta["completion_length"])
        count += 1

    if count == 0:
        raise ValueError("No training samples with has_eos=True were found; cannot compute mean completion length.")

    return total / count


def compute_train_median_token_target(train_dir: Path) -> float:
    """Median of all token-level remaining-count targets in the training split.

    For each sample with EOS and completion length L, the token-level targets
    are L-1, L-2, ..., 0. We compute the median over the flattened list.
    """
    json_files = _iter_sample_json_files(train_dir)

    targets: list[int] = []
    for file_path in json_files:
        meta = _load_metadata(file_path)
        if not bool(meta.get("has_eos", False)):
            continue
        completion_length = int(meta["completion_length"])
        # Extend with L-1, L-2, ..., 0
        targets.extend(range(completion_length - 1, -1, -1))

    if not targets:
        raise ValueError(
            "No training samples with has_eos=True were found; cannot compute median token target."
        )

    return float(statistics.median(targets))


def compute_train_median_completion_length(train_dir: Path) -> float:
    """Median completion length over training samples with EOS."""
    json_files = _iter_sample_json_files(train_dir)

    lengths: list[int] = []
    for file_path in json_files:
        meta = _load_metadata(file_path)
        if not bool(meta.get("has_eos", False)):
            continue
        lengths.append(int(meta["completion_length"]))

    if not lengths:
        raise ValueError("No training samples with has_eos=True were found; cannot compute median completion length.")

    return float(statistics.median(lengths))


def compute_split_metrics(
    split_dir: Path,
    epsilon: float,
    eos_only: bool,
    bin_size: int,
    predictor_type: str,
    constant_value: float | None = None,
    base_length: float | None = None,
) -> tuple[
    float,  # MAE (average across tokens)
    float,  # MSE (average across tokens)
    float,  # MAPE (average across tokens)
    float,  # SMAPE (average across tokens)
    SplitStats,
    dict[int, tuple[float, int]],  # sequence_mae_buckets
    dict[int, tuple[float, int]],  # token_mae_buckets
    dict[int, tuple[float, int]],  # token_mape_buckets
    dict[int, tuple[float, int]],  # token_smape_buckets
]:
    """Evaluate a constant/position-aware predictor at token level on a split.

    Predictor types:
    - "constant":              y_pred = constant_value for every position
    - "length_minus_position": y_pred = max(base_length - tokens_generated - 1, 0)
    """
    json_files = _iter_sample_json_files(split_dir)

    total = 0
    eos_count = 0
    used = 0
    used_targets = 0
    mae_sum = 0.0
    mse_sum = 0.0
    # MAPE / SMAPE scalars are aggregated as "mean of per-sequence means" to
    # match the trainer (CustomTrainer._update_count_buckets), so the baseline
    # numbers are directly comparable with the trainer's count/*_mape and
    # count/*_smape wandb scalars. MAE / MSE stay as per-token means
    # (the trainer's count/*_mae scalar is also per-token).
    seq_mape_mean_sum = 0.0
    seq_smape_mean_sum = 0.0

    sequence_mae_buckets: dict[int, tuple[float, int]] = {}
    token_mae_buckets: dict[int, tuple[float, int]] = {}
    token_mape_buckets: dict[int, tuple[float, int]] = {}
    token_smape_buckets: dict[int, tuple[float, int]] = {}

    for file_path in json_files:
        total += 1
        meta = _load_metadata(file_path)

        has_eos = bool(meta.get("has_eos", False))
        if has_eos:
            eos_count += 1

        if eos_only and not has_eos:
            continue

        completion_length = int(meta["completion_length"])
        sum_errors = 0.0
        # Per-sequence accumulators for MAPE/SMAPE — averaged at end of sequence.
        seq_mape_sum = 0.0
        seq_smape_sum = 0.0

        for tokens_generated in range(completion_length):
            y_true = float(completion_length - tokens_generated - 1)

            if predictor_type == "constant":
                if constant_value is None:
                    raise ValueError("constant_value is required for predictor_type='constant'.")
                y_pred = constant_value
            elif predictor_type == "length_minus_position":
                if base_length is None:
                    raise ValueError("base_length is required for predictor_type='length_minus_position'.")
                y_pred = max(base_length - tokens_generated - 1, 0.0)
            else:
                raise ValueError(f"Unknown predictor_type: {predictor_type}")

            error = abs(y_pred - y_true)
            sq_error = (y_pred - y_true) ** 2
            mape_val = _safe_mape(y_true, y_pred, epsilon)
            smape_val = _safe_smape(y_true, y_pred, epsilon)

            mae_sum += error
            mse_sum += sq_error
            seq_mape_sum += mape_val
            seq_smape_sum += smape_val
            used_targets += 1
            sum_errors += error

            token_bucket_start = (int(y_true) // bin_size) * bin_size
            _update_bucket(token_mae_buckets, token_bucket_start, error)
            _update_bucket(token_mape_buckets, token_bucket_start, mape_val)
            _update_bucket(token_smape_buckets, token_bucket_start, smape_val)

        seq_bucket_start = (completion_length // bin_size) * bin_size
        _update_bucket_with_count(sequence_mae_buckets, seq_bucket_start, sum_errors, completion_length)

        # Mean-of-means aggregation: contribute this sequence's mean MAPE/SMAPE.
        if completion_length > 0:
            seq_mape_mean_sum += seq_mape_sum / completion_length
            seq_smape_mean_sum += seq_smape_sum / completion_length

        used += 1

    if used == 0 or used_targets == 0:
        raise ValueError("No evaluation samples matched the filter; cannot compute metrics.")

    return (
        mae_sum / used_targets,              # MAE: per-token mean (matches trainer's count/*_mae)
        mse_sum / used_targets,              # MSE: per-token mean
        seq_mape_mean_sum / used,            # MAPE: mean of per-sequence means (matches trainer)
        seq_smape_mean_sum / used,           # SMAPE: mean of per-sequence means (matches trainer)
        SplitStats(total, eos_count, used, used_targets),
        sequence_mae_buckets,
        token_mae_buckets,
        token_mape_buckets,
        token_smape_buckets,
    )


def compute_prompt_end_baseline_metrics(
    split_dir: Path,
    epsilon: float,
    eos_only: bool,
    bin_size: int,
    constant_value: float,
) -> tuple[
    float,                             # MAE  (mean over sequences)
    float,                             # MSE  (mean over sequences)
    float,                             # MAPE (mean over sequences)
    float,                             # SMAPE (mean over sequences)
    SplitStats,
    dict[int, tuple[float, int]],      # seq_mae_buckets   (binned by T)
    dict[int, tuple[float, int]],      # tok_mae_buckets   (always empty here)
    dict[int, tuple[float, int]],      # mape buckets      (binned by T)
    dict[int, tuple[float, int]],      # smape buckets     (binned by T)
]:
    """No-information mirror of the `prompt_only_count` probe.

    The probe predicts total completion length T from the prompt's last hidden
    state. The corresponding "no signal" baseline outputs a fixed scalar (the
    train-split mean or median completion length) regardless of the prompt
    and regardless of position. Since the predictor is constant per sequence,
    we compute one error per sequence (against the true T = completion_length)
    rather than per token — that's what makes this a *prompt-end* baseline as
    opposed to the per-token `length_minus_position` baselines.

    Errors:
      - MAE   = mean over sequences of |constant - T_i|
      - MSE   = mean over sequences of (constant - T_i)^2
      - MAPE  = mean over sequences of |constant - T_i| / max(T_i, eps)
      - SMAPE = mean over sequences of |constant - T_i| / max((|constant|+|T_i|)/2, eps)

    All buckets are binned by T (= completion length). The token-level
    `tok_mae_buckets` slot is left empty so the per-token MAE figure is
    skipped (`_save_metric_chart` returns early on empty dicts).
    """
    json_files = _iter_sample_json_files(split_dir)

    total = 0
    eos_count = 0
    used = 0
    mae_sum = 0.0
    mse_sum = 0.0
    mape_sum = 0.0
    smape_sum = 0.0

    seq_mae_buckets: dict[int, tuple[float, int]] = {}
    mape_buckets: dict[int, tuple[float, int]] = {}
    smape_buckets: dict[int, tuple[float, int]] = {}

    for file_path in json_files:
        total += 1
        meta = _load_metadata(file_path)

        has_eos = bool(meta.get("has_eos", False))
        if has_eos:
            eos_count += 1
        if eos_only and not has_eos:
            continue

        T = float(meta["completion_length"])
        err = abs(constant_value - T)
        sq_err = (constant_value - T) ** 2
        mape_val = _safe_mape(T, constant_value, epsilon)
        smape_val = _safe_smape(T, constant_value, epsilon)

        mae_sum += err
        mse_sum += sq_err
        mape_sum += mape_val
        smape_sum += smape_val
        used += 1

        bucket = (int(T) // bin_size) * bin_size
        _update_bucket(seq_mae_buckets, bucket, err)
        _update_bucket(mape_buckets, bucket, mape_val)
        _update_bucket(smape_buckets, bucket, smape_val)

    if used == 0:
        raise ValueError(
            "No evaluation samples matched the filter; cannot compute prompt-end baseline."
        )

    return (
        mae_sum / used,
        mse_sum / used,
        mape_sum / used,
        smape_sum / used,
        # used_targets == used (one prediction per sequence).
        SplitStats(total, eos_count, used, used),
        seq_mae_buckets,
        {},  # tok_mae_buckets — N/A for prompt-end baseline.
        mape_buckets,
        smape_buckets,
    )


def compute_baselines(
    hidden_states_root: Path,
    epsilon: float = 1e-8,
    bin_size: int = 10,
    include_non_eos_eval: bool = False,
) -> dict[str, float]:
    """Compute every statistical baseline on a hidden-states cache and return
    the prefixed scalars as a flat dict ready for `wandb.log(...)`.

    No figures, no printing, no side effects beyond reading the metadata
    JSONs in `hidden_states_root/{train,eval}/sample_*.json`. The caller is
    responsible for logging — typically `wandb.log(compute_baselines(root))`.

    Returns:
        Dict like `{"statistical_baseline/baseline_constant_mean/eval_mae": 12.34, ...}`,
        already namespaced under WANDB_PREFIX so it can sit beside trained-probe
        metrics in the same wandb run.
    """
    hs_root = _resolve_hidden_states_root(hidden_states_root)
    train_dir = hs_root / "train"
    eval_dir = hs_root / "eval"

    baseline_mean, _train_stats = compute_train_mean_token_target(train_dir)
    mean_completion_length = compute_train_mean_completion_length(train_dir)
    baseline_median = compute_train_median_token_target(train_dir)
    median_completion_length = compute_train_median_completion_length(train_dir)

    # Same baseline list as `main()` uses, minus the figure / print machinery.
    token_baselines = [
        ("baseline_constant_mean",        "constant",              baseline_mean,   None),
        ("baseline_constant_median",      "constant",              baseline_median, None),
        ("baseline_mean_plus_position",   "length_minus_position", None, mean_completion_length),
        ("baseline_median_plus_position", "length_minus_position", None, median_completion_length),
    ]
    seqlen_baselines = [
        ("seqlen_constant_mean",     "length_minus_position", None, baseline_mean),
        ("seqlen_constant_median",   "length_minus_position", None, baseline_median),
        ("seqlen_mean_completion",   "length_minus_position", None, mean_completion_length),
        ("seqlen_median_completion", "length_minus_position", None, median_completion_length),
    ]
    prompt_only_specs = [
        ("prompt_only_mean_completion", mean_completion_length),
    ]

    out: dict[str, float] = {}

    for key_prefix, predictor_type, constant_value, base_length in token_baselines + seqlen_baselines:
        for split_name, split_dir, eos_only in [
            ("train", train_dir, True),
            ("eval", eval_dir, not include_non_eos_eval),
        ]:
            mae, mse, mape, smape, *_ = compute_split_metrics(
                split_dir=split_dir,
                epsilon=epsilon,
                eos_only=eos_only,
                bin_size=bin_size,
                predictor_type=predictor_type,
                constant_value=constant_value,
                base_length=base_length,
            )
            out[f"{key_prefix}/{split_name}_mae"]   = mae
            out[f"{key_prefix}/{split_name}_mse"]   = mse
            out[f"{key_prefix}/{split_name}_mape"]  = mape
            out[f"{key_prefix}/{split_name}_smape"] = smape

    for key_prefix, constant_value in prompt_only_specs:
        for split_name, split_dir, eos_only in [
            ("train", train_dir, True),
            ("eval", eval_dir, not include_non_eos_eval),
        ]:
            mae, mse, mape, smape, *_ = compute_prompt_end_baseline_metrics(
                split_dir=split_dir,
                epsilon=epsilon,
                eos_only=eos_only,
                bin_size=bin_size,
                constant_value=constant_value,
            )
            out[f"{key_prefix}/{split_name}_mae"]   = mae
            out[f"{key_prefix}/{split_name}_mse"]   = mse
            out[f"{key_prefix}/{split_name}_mape"]  = mape
            out[f"{key_prefix}/{split_name}_smape"] = smape

    # Baseline-specific scalars under their own subnamespace.
    out["baseline/mean_remaining_token_count"]   = baseline_mean
    out["baseline/median_remaining_token_count"] = baseline_median
    out["baseline/mean_completion_length"]       = mean_completion_length
    out["baseline/median_completion_length"]     = median_completion_length

    # Single point of namespacing — keep parity with `main()`'s wandb logging.
    return {f"{WANDB_PREFIX}/{k}": v for k, v in out.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute token-level baseline metrics and figures for remaining-token prediction."
    )
    parser.add_argument(
        "hidden_states_dir",
        type=Path,
        help=(
            "Path to hidden states root. Can be either a directory containing "
            "train/ and eval/, or a parent containing hidden_states/train and "
            "hidden_states/eval."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon used in MAPE denominator to avoid division by zero.",
    )
    parser.add_argument(
        "--include-non-eos-eval",
        action="store_true",
        help="If set, include eval samples with has_eos=False when computing MAPE.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=10,
        help="Bin size used for bucketed figures (same semantics as trainer).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory where train/eval baseline figures are saved.",
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log generated figures to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help=(
            "W&B project name. Resolved with this precedence: CLI override > "
            "wandb.project from --config > built-in default 'token-count-prediction'."
        ),
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity/team name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help=(
            "Optional W&B run name. Resolved with this precedence: CLI override "
            "> wandb.run_name from --config > unset (wandb auto-generates one)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to a Hydra config.yaml (typically saved by train_offline.py at "
            "<output_dir>/seed_<N>/trained_probe/config.yaml). When provided, the "
            "file's contents are merged into the wandb run's config so the "
            "baseline run carries the same model / data / seed metadata as the "
            "training run it complements."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load the saved Hydra config (if --config was given) so the wandb run
    # carries the same model / data / seed metadata as the training run this
    # baseline complements.
    saved_config_dict: dict = {}
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"--config path does not exist: {args.config}")
        loaded = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        saved_config_dict = loaded if isinstance(loaded, dict) else {}

    # Pull wandb project / run_name from the saved config if the CLI didn't
    # override them. Lets a baseline run inherit `cfg.wandb.project` and
    # `cfg.wandb.run_name` from the training run it complements without
    # repeating them on the CLI.
    saved_wandb_cfg = saved_config_dict.get("wandb") or {}
    if not isinstance(saved_wandb_cfg, dict):
        saved_wandb_cfg = {}
    wandb_project = (
        args.wandb_project
        or saved_wandb_cfg.get("project")
        or "token-count-prediction"
    )
    wandb_run_name = args.wandb_run_name or saved_wandb_cfg.get("run_name")

    wandb_run = None
    if args.log_wandb and is_main_process():
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config={
                **saved_config_dict,
                "statistical_baseline_args": {
                    "hidden_states_dir": str(args.hidden_states_dir),
                    "epsilon": args.epsilon,
                    "bin_size": args.bin_size,
                    "include_non_eos_eval": args.include_non_eos_eval,
                    "config_path": str(args.config) if args.config else None,
                },
            },
        )
        # Mirror the trainer's step-metric setup so charts in this standalone
        # run are also plottable against `train/epoch`.
        wandb.define_metric("train/epoch")
        wandb.define_metric("*", step_metric="train/epoch")

    # Resolve dataset root and split directories.
    hs_root = _resolve_hidden_states_root(args.hidden_states_dir)
    train_dir = hs_root / "train"
    eval_dir = hs_root / "eval"

    # Fit mean and median baselines on training split.
    baseline_mean, train_stats = compute_train_mean_token_target(train_dir)
    mean_completion_length = compute_train_mean_completion_length(train_dir)
    baseline_median = compute_train_median_token_target(train_dir)
    median_completion_length = compute_train_median_completion_length(train_dir)

    # Four baselines to evaluate on both splits.
    # Each entry: (key_prefix, predictor_type, constant_value, base_length)
    baselines = [
        ("baseline_constant_mean",    "constant",              baseline_mean,   None),
        ("baseline_constant_median",  "constant",              baseline_median, None),
        ("baseline_mean_plus_position",   "length_minus_position", None, mean_completion_length),
        ("baseline_median_plus_position", "length_minus_position", None, median_completion_length),
    ]

    # Evaluate each baseline on train and eval.
    # results[key_prefix][split] = (mae, mse, mape, stats, seq_mae, tok_mae, tok_mape)
    results: dict[str, dict[str, tuple]] = {}
    for key_prefix, predictor_type, constant_value, base_length in baselines:
        results[key_prefix] = {}
        for split_name, split_dir, eos_only in [
            ("train", train_dir, True),
            ("eval", eval_dir, not args.include_non_eos_eval),
        ]:
            results[key_prefix][split_name] = compute_split_metrics(
                split_dir=split_dir,
                epsilon=args.epsilon,
                eos_only=eos_only,
                bin_size=args.bin_size,
                predictor_type=predictor_type,
                constant_value=constant_value,
                base_length=base_length,
            )

    # Use stats from the first baseline (they all process the same files)
    first_key = baselines[0][0]
    eval_stats = results[first_key]["eval"][4]

    # Sequence-level baselines: predictor outputs a single "predicted total completion
    # length" per sequence, but we evaluate at the token level using the same
    # compute_split_metrics function — the predicted total length becomes the
    # base_length of a length_minus_position predictor (y_pred(k) = base - k - 1).
    # Note: seqlen_mean_completion / seqlen_median_completion will be numerically
    # identical to baseline_mean_plus_position / baseline_median_plus_position.
    seqlen_baselines = [
        ("seqlen_constant_mean",     "length_minus_position", None, baseline_mean),
        ("seqlen_constant_median",   "length_minus_position", None, baseline_median),
        ("seqlen_mean_completion",   "length_minus_position", None, mean_completion_length),
        ("seqlen_median_completion", "length_minus_position", None, median_completion_length),
    ]

    seq_results: dict[str, dict[str, tuple]] = {}
    for key_prefix, predictor_type, constant_value, base_length in seqlen_baselines:
        seq_results[key_prefix] = {}
        for split_name, split_dir, eos_only in [
            ("train", train_dir, True),
            ("eval", eval_dir, not args.include_non_eos_eval),
        ]:
            seq_results[key_prefix][split_name] = compute_split_metrics(
                split_dir=split_dir,
                epsilon=args.epsilon,
                eos_only=eos_only,
                bin_size=args.bin_size,
                predictor_type=predictor_type,
                constant_value=constant_value,
                base_length=base_length,
            )

    # No-information mirror of the `prompt_only_count` probe: predict the
    # train-split mean completion length as the total T, no per-token decay.
    # Evaluated as ONE error per sequence (target = T = completion_length).
    prompt_only_baselines = [
        ("prompt_only_mean_completion", mean_completion_length),
    ]

    prompt_only_results: dict[str, dict[str, tuple]] = {}
    for key_prefix, constant_value in prompt_only_baselines:
        prompt_only_results[key_prefix] = {}
        for split_name, split_dir, eos_only in [
            ("train", train_dir, True),
            ("eval", eval_dir, not args.include_non_eos_eval),
        ]:
            prompt_only_results[key_prefix][split_name] = compute_prompt_end_baseline_metrics(
                split_dir=split_dir,
                epsilon=args.epsilon,
                eos_only=eos_only,
                bin_size=args.bin_size,
                constant_value=constant_value,
            )

    figures_dir = args.figures_dir if args.figures_dir is not None else (hs_root / "baseline_figures")

    # Save figures for every (baseline, split) pair.
    for key_prefix, _, _, _ in baselines:
        baseline_fig_dir = figures_dir / key_prefix
        for split_name in ("train", "eval"):
            (_, _, _, _, _,
             seq_mae_buckets, tok_mae_buckets, tok_mape_buckets, tok_smape_buckets,
             ) = results[key_prefix][split_name]

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(seq_mae_buckets)
            mae_path = baseline_fig_dir / f"{split_name}_mae_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "mae", mae_path)
            _log_figure_to_wandb(mae_path, split_name, "mae", args.log_wandb, key_prefix=key_prefix)

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(tok_mae_buckets)
            mae_ind_path = baseline_fig_dir / f"{split_name}_mae_individual_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "mae_individual", mae_ind_path)
            _log_figure_to_wandb(mae_ind_path, split_name, "mae_individual", args.log_wandb, key_prefix=key_prefix)

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(tok_mape_buckets)
            mape_path = baseline_fig_dir / f"{split_name}_mape_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "mape", mape_path)
            _log_figure_to_wandb(mape_path, split_name, "mape", args.log_wandb, key_prefix=key_prefix)

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(tok_smape_buckets)
            smape_path = baseline_fig_dir / f"{split_name}_smape_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "smape", smape_path)
            _log_figure_to_wandb(smape_path, split_name, "smape", args.log_wandb, key_prefix=key_prefix)

    # Save figures for sequence-level baselines (same chart set as token baselines,
    # since they're evaluated through compute_split_metrics the same way).
    for key_prefix, _, _, _ in seqlen_baselines:
        baseline_fig_dir = figures_dir / key_prefix
        for split_name in ("train", "eval"):
            (_, _, _, _, _,
             seq_mae_buckets, tok_mae_buckets, tok_mape_buckets, tok_smape_buckets,
             ) = seq_results[key_prefix][split_name]

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(seq_mae_buckets)
            mae_path = baseline_fig_dir / f"{split_name}_mae_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "mae", mae_path)
            _log_figure_to_wandb(mae_path, split_name, "mae", args.log_wandb, key_prefix=key_prefix)

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(tok_mae_buckets)
            mae_ind_path = baseline_fig_dir / f"{split_name}_mae_individual_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "mae_individual", mae_ind_path)
            _log_figure_to_wandb(mae_ind_path, split_name, "mae_individual", args.log_wandb, key_prefix=key_prefix)

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(tok_mape_buckets)
            mape_path = baseline_fig_dir / f"{split_name}_mape_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "mape", mape_path)
            _log_figure_to_wandb(mape_path, split_name, "mape", args.log_wandb, key_prefix=key_prefix)

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(tok_smape_buckets)
            smape_path = baseline_fig_dir / f"{split_name}_smape_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "smape", smape_path)
            _log_figure_to_wandb(smape_path, split_name, "smape", args.log_wandb, key_prefix=key_prefix)

    # Save figures for prompt-end baselines: only mae/mape/smape per T-bin
    # are meaningful (no per-token-target binning since the prediction is one
    # scalar per sequence). The empty `tok_mae_buckets` skips the
    # mae_individual figure naturally.
    for key_prefix, _ in prompt_only_baselines:
        baseline_fig_dir = figures_dir / key_prefix
        for split_name in ("train", "eval"):
            (_, _, _, _, _,
             seq_mae_buckets, tok_mae_buckets, tok_mape_buckets, tok_smape_buckets,
             ) = prompt_only_results[key_prefix][split_name]

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(seq_mae_buckets)
            mae_path = baseline_fig_dir / f"{split_name}_mae_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "mae", mae_path)
            _log_figure_to_wandb(mae_path, split_name, "mae", args.log_wandb, key_prefix=key_prefix)

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(tok_mape_buckets)
            mape_path = baseline_fig_dir / f"{split_name}_mape_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "mape", mape_path)
            _log_figure_to_wandb(mape_path, split_name, "mape", args.log_wandb, key_prefix=key_prefix)

            metric_per_bin, counts_per_bin = _bucket_means_and_counts(tok_smape_buckets)
            smape_path = baseline_fig_dir / f"{split_name}_smape_per_bin.png"
            _save_metric_chart(metric_per_bin, counts_per_bin, split_name, "smape", smape_path)
            _log_figure_to_wandb(smape_path, split_name, "smape", args.log_wandb, key_prefix=key_prefix)

    print("=== Token-Level Remaining-Count Baselines ===")
    print(f"hidden_states_dir: {hs_root}")
    print("")
    print("Train stats:")
    print(f"  total files: {train_stats.total_files}")
    print(f"  files with EOS: {train_stats.eos_files}")
    print(f"  files used for baseline: {train_stats.used_files}")
    print(f"  token targets used for baseline: {train_stats.used_targets}")
    print(f"  baseline mean remaining_token_count:   {baseline_mean:.6f}")
    print(f"  baseline median remaining_token_count: {baseline_median:.6f}")
    print(f"  mean completion_length:   {mean_completion_length:.6f}")
    print(f"  median completion_length: {median_completion_length:.6f}")
    print("")

    # Per-baseline, per-split MAE / MSE / MAPE / SMAPE table.
    for split_name in ("train", "eval"):
        print(f"{split_name.capitalize()} metrics — TOKEN-LEVEL (remaining count at each position):")
        print(f"  {'Baseline':<40} {'MAE':>10} {'MSE':>12} {'MAPE':>10} {'SMAPE':>10}")
        for key_prefix, _, _, _ in baselines:
            mae, mse, mape, smape, _, _, _, _, _ = results[key_prefix][split_name]
            print(f"  {key_prefix:<40} {mae:>10.4f} {mse:>12.4f} {mape * 100:>9.2f}% {smape * 100:>9.2f}%")
        print("")

        print(f"{split_name.capitalize()} metrics — SEQUENCE-LEVEL predictors (total length, evaluated per-token):")
        print(f"  {'Baseline':<40} {'MAE':>10} {'MSE':>12} {'MAPE':>10} {'SMAPE':>10}")
        for key_prefix, _, _, _ in seqlen_baselines:
            mae, mse, mape, smape, _, _, _, _, _ = seq_results[key_prefix][split_name]
            print(f"  {key_prefix:<40} {mae:>10.4f} {mse:>12.4f} {mape * 100:>9.2f}% {smape * 100:>9.2f}%")
        print("")

        print(f"{split_name.capitalize()} metrics — PROMPT-END predictors (total length, ONE error per sequence):")
        print(f"  {'Baseline':<40} {'MAE':>10} {'MSE':>12} {'MAPE':>10} {'SMAPE':>10}")
        for key_prefix, _ in prompt_only_baselines:
            mae, mse, mape, smape, _, _, _, _, _ = prompt_only_results[key_prefix][split_name]
            print(f"  {key_prefix:<40} {mae:>10.4f} {mse:>12.4f} {mape * 100:>9.2f}% {smape * 100:>9.2f}%")
        print("")

    print(f"Saved figures to: {figures_dir}")

    if wandb_run is not None and is_main_process():
        import wandb

        wandb_log: dict = {}
        for key_prefix, _, _, _ in baselines:
            for split_name in ("train", "eval"):
                mae, mse, mape, smape, _, _, _, _, _ = results[key_prefix][split_name]
                wandb_log[f"{key_prefix}/{split_name}_mae"] = mae
                wandb_log[f"{key_prefix}/{split_name}_mse"] = mse
                wandb_log[f"{key_prefix}/{split_name}_mape"] = mape
                wandb_log[f"{key_prefix}/{split_name}_smape"] = smape

        for key_prefix, _, _, _ in seqlen_baselines:
            for split_name in ("train", "eval"):
                mae, mse, mape, smape, _, _, _, _, _ = seq_results[key_prefix][split_name]
                wandb_log[f"{key_prefix}/{split_name}_mae"] = mae
                wandb_log[f"{key_prefix}/{split_name}_mse"] = mse
                wandb_log[f"{key_prefix}/{split_name}_mape"] = mape
                wandb_log[f"{key_prefix}/{split_name}_smape"] = smape

        for key_prefix, _ in prompt_only_baselines:
            for split_name in ("train", "eval"):
                mae, mse, mape, smape, _, _, _, _, _ = prompt_only_results[key_prefix][split_name]
                wandb_log[f"{key_prefix}/{split_name}_mae"] = mae
                wandb_log[f"{key_prefix}/{split_name}_mse"] = mse
                wandb_log[f"{key_prefix}/{split_name}_mape"] = mape
                wandb_log[f"{key_prefix}/{split_name}_smape"] = smape

        wandb_log.update({
            "train/sequences_total": train_stats.total_files,
            "train/sequences_with_eos": train_stats.eos_files,
            "train/sequences_with_eos_fraction": (train_stats.eos_files / train_stats.total_files) if train_stats.total_files > 0 else 0.0,
            "eval/sequences_total": eval_stats.total_files,
            "eval/sequences_with_eos": eval_stats.eos_files,
            "eval/sequences_truncated": eval_stats.total_files - eval_stats.eos_files,
            # Baseline-specific scalars retained under explicit namespace.
            "baseline/mean_remaining_token_count": baseline_mean,
            "baseline/median_remaining_token_count": baseline_median,
            "baseline/mean_completion_length": mean_completion_length,
            "baseline/median_completion_length": median_completion_length,
        })
        # Single point of namespacing: every scalar logged from this script
        # lives under WANDB_PREFIX so it can sit beside the trained-probe
        # metrics in the same wandb run / project without collisions.
        wandb_log = {f"{WANDB_PREFIX}/{k}": v for k, v in wandb_log.items()}
        wandb_log["train/epoch"] = 0.0
        wandb.log(wandb_log)
        wandb.finish()


if __name__ == "__main__":
    main()
