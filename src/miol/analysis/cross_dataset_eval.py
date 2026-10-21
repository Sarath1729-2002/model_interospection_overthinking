"""Cross-dataset probe evaluation.

For one fixed model and a list of datasets, this script runs every
combination of (train_dataset, eval_dataset, seed) — taking probe heads
trained on `train_dataset` and applying them to `eval_dataset`'s hidden
states. The chosen (probe, metric) is averaged across the eval split's
examples to give one scalar per (train, eval, seed); mean ± std across
the seeds populates each cell of the resulting heatmap.

Same model only — probe weights are tied to a specific hidden_size.

Usage:
    python -m miol.analysis.cross_dataset_eval \\
        --model qwen3_4b \\
        --datasets gsm8k math openthoughts-1k trivia_qa \\
        --seeds 0 1 2 \\
        --probe count \\
        --metric mae \\
        --output figures/cross_dataset/qwen3_4b_count_mae.pdf

For paths that don't match the default layout
`outputs/{model}-{dataset}/seed_{seed}/trained_probe/`, override with
`--probe-dir-pattern` and `--hidden-states-pattern`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from miol.data.hidden_states_dataset import HiddenStatesDataset
from miol.evaluate_probes import (
    _merge_examples,
    _merge_scalar,
    build_statistical_baselines,
    evaluate_many_probe_dirs,
)
from miol.model.probe_specs import PROBE_SPECS


def _build_eval_context(
    model: str,
    eval_dataset_name: str,
    hs_pattern: str,
    active_probes: list[str],
    cfg,
    load_stat_baselines: bool,
) -> dict[str, Any]:
    """Load eval-side state once per eval_dataset (reused across all
    train_dataset × seed combos to avoid re-instantiating the dataset).
    Statistical baselines are fit on the eval_dataset's *train* split — they're
    a property of the eval dataset, not of the trained probe — so they're
    cached here too when `load_stat_baselines` is set.
    """
    hs_dir = Path(hs_pattern.format(model=model, dataset=eval_dataset_name))
    hidden_layer = str(cfg.data.get("hidden_layer", "-1"))
    eval_dataset = HiddenStatesDataset(
        hidden_states_dir=str(hs_dir),
        split="eval",
        hidden_layer=hidden_layer,
    )
    if len(eval_dataset) == 0:
        raise ValueError(f"No eval samples for {eval_dataset_name} at {hs_dir}")
    hidden_size = eval_dataset[0]["hidden_states"].shape[-1]

    cls_anchors: dict[str, torch.Tensor] = {}
    classification_bins: dict[str, int] = {}
    for name in active_probes:
        if PROBE_SPECS[name]["type"] == "classification":
            n_bins = PROBE_SPECS[name]["output_dim"]
            cls_anchors[name] = torch.linspace(0, 1, n_bins)
            classification_bins[name] = n_bins

    stat_baselines: dict[str, dict] = {}
    if load_stat_baselines:
        train_dir = hs_dir / "train"
        try:
            stat_baselines = build_statistical_baselines(train_dir)
        except Exception as exc:
            print(
                f"  [warning] could not fit statistical baselines for "
                f"{eval_dataset_name} at {train_dir}: {type(exc).__name__}: {exc}"
            )
            stat_baselines = {}

    return {
        "eval_dataset": eval_dataset,
        "hidden_size": hidden_size,
        "cls_anchors": cls_anchors,
        "classification_bins": classification_bins,
        "stat_baselines": stat_baselines,
        "hidden_states_dir": str(hs_dir),
    }


_PER_TOKEN_ARRAY_KEYS = {
    # Maps an aggregate metric_key to the per-token array it is the mean of.
    # When a key is in this map, the cell scalar is computed as
    # `sum(per_token_array) / total_tokens` (length-weighted) rather than
    # cross-example mean of per-example means. This matches the trainer's
    # wandb convention (`total_err / total_cnt` from `mae_buckets`), so the
    # diagonal of the cross-dataset matrix lines up with `count/eval_mae`
    # for the same (probe, dataset) on wandb.
    "mae": "mae_per_token",
    "mse": "mse_per_token",
    "mape": "mape_per_token",
    "smape": "smape_per_token",
    "accuracy": "correct_per_token",
    "derived_pct_mae": "derived_pct_mae_per_token",
    "derived_pct_mse": "derived_pct_mse_per_token",
}


def _extract_cell_metric(
    examples: list[dict],
    name: str,
    metric_key: str,
    bucket: str = "probes",
) -> float | None:
    """Length-weighted cross-token mean of `<metric>_per_token` when that
    array exists for this metric, else cross-example mean of
    `aggregate.<metric_key>`. Returns None when nothing valid is found.

    `bucket` selects between `probes` (probe heads, the default) and
    `statistical_baselines` (the dataset-fit constant / mean-plus-position
    predictors). Both share the same per-example entry shape, so the same
    aggregation logic applies.

    The per-token path matches the trainer's wandb convention for the
    headline metrics (mae / mape / smape / accuracy / derived_pct_mae);
    other aggregate scalars (nm_score, prompt_mae, cohen_kappa, ...) don't
    have a per-token array so the equal-weight per-example mean is used.
    """
    per_token_key = _PER_TOKEN_ARRAY_KEYS.get(metric_key)

    if per_token_key is not None:
        total_sum = 0.0
        total_cnt = 0
        for ex in examples:
            entry = (ex.get(bucket) or {}).get(name)
            if not isinstance(entry, dict):
                continue
            arr = entry.get(per_token_key)
            if not isinstance(arr, list):
                continue
            for v in arr:
                if isinstance(v, bool):
                    total_sum += float(v)
                    total_cnt += 1
                    continue
                if isinstance(v, (int, float)) and not (
                    math.isnan(float(v)) or math.isinf(float(v))
                ):
                    total_sum += float(v)
                    total_cnt += 1
        if total_cnt > 0:
            return total_sum / total_cnt
        # Fall through to the aggregate path if no per-token data was found
        # (e.g. has_eos=False everywhere) — keeps the cell populated when
        # the aggregate scalar still has data to offer.

    vals: list[float] = []
    for ex in examples:
        agg = (ex.get(bucket) or {}).get(name, {}).get("aggregate")
        if not isinstance(agg, dict):
            continue
        v = agg.get(metric_key)
        # `evaluate_many_probe_dirs` returns plain scalars per seed; the
        # {mean, std} shape only appears post-merge. Be defensive anyway.
        if isinstance(v, dict):
            v = v.get("mean")
        if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v)):
            vals.append(float(v))
    if not vals:
        return None
    return statistics.fmean(vals)


def _compute_all_cell_metrics(
    per_seed_examples: list[list[dict]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """For every (name, aggregate metric_key) found in the per-seed examples,
    compute the heatmap cell scalar — per-seed cross-example mean → mean ± std
    across seeds — and bundle them all into a `{name: {metric: {...}}}` dict.

    Walks both the `probes` and `statistical_baselines` buckets so the
    statistical-baseline cells (e.g. `baseline_constant_median.mae`) land in
    the JSON alongside probe cells. Probe and baseline names don't overlap,
    so flattening into one dict keeps the schema simple.

    Stashed in the per-pair JSON's `metadata.cell_metrics` so `--from-jsons`
    can render a heatmap (or compute a per-column normalization scalar) for
    any (name, metric) the inference run produced without re-running
    inference. Pulls `aggregate.keys()` from seed-0's examples (names /
    metrics are seed-invariant; only the values differ).
    """
    if not per_seed_examples:
        return {}

    name_metrics: dict[str, set[str]] = {}
    name_buckets: dict[str, str] = {}
    for ex in per_seed_examples[0]:
        for bucket in ("probes", "statistical_baselines"):
            for name, entry in (ex.get(bucket) or {}).items():
                agg = entry.get("aggregate") if isinstance(entry, dict) else None
                if isinstance(agg, dict):
                    name_metrics.setdefault(name, set()).update(agg.keys())
                    name_buckets[name] = bucket

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, metric_keys in name_metrics.items():
        bucket = name_buckets[name]
        per_metric: dict[str, dict[str, Any]] = {}
        for metric_key in sorted(metric_keys):
            seed_values: list[float] = []
            for ex_list in per_seed_examples:
                v = _extract_cell_metric(ex_list, name, metric_key, bucket=bucket)
                if v is not None:
                    seed_values.append(v)
            if not seed_values:
                continue
            per_metric[metric_key] = {
                "per_seed": [float(v) for v in seed_values],
                "mean": float(statistics.fmean(seed_values)),
                "std": (
                    float(statistics.pstdev(seed_values))
                    if len(seed_values) > 1 else 0.0
                ),
            }
        if per_metric:
            out[name] = per_metric
    return out


def _write_matrix_csv(
    datasets: list[str],
    Z: np.ndarray,
    output_path: Path,
    fmt: str = ".6f",
) -> None:
    """Dump a heatmap matrix to CSV with `train_dataset` rows and `eval_dataset`
    columns. NaN cells render as empty strings (loads cleanly back into pandas
    via `pd.read_csv(...).set_index(...)`). Numbers are formatted with `fmt`
    so the file is human-readable; the heatmap PDF uses .3f for display, the
    CSV uses .6f so downstream analysis isn't quantized.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_dataset \\ eval_dataset"] + list(datasets))
        for i, train_ds in enumerate(datasets):
            row: list[str] = [train_ds]
            for j in range(len(datasets)):
                v = Z[i, j]
                row.append("" if np.isnan(v) else format(float(v), fmt))
            writer.writerow(row)


def _write_heatmap_csvs(
    datasets: list[str],
    Z_mean: np.ndarray,
    Z_std: np.ndarray,
    pdf_path: Path,
) -> None:
    """Sibling of `_render_heatmap`: writes <stem>_mean.csv and <stem>_std.csv
    next to the figure (same directory, derived from the PDF path's stem).
    """
    mean_path = pdf_path.with_name(f"{pdf_path.stem}_mean.csv")
    std_path = pdf_path.with_name(f"{pdf_path.stem}_std.csv")
    _write_matrix_csv(datasets, Z_mean, mean_path)
    _write_matrix_csv(datasets, Z_std, std_path)
    print(f"Saved CSVs: {mean_path}  {std_path}")


def _render_heatmap(
    datasets: list[str],
    Z_mean: np.ndarray,
    Z_std: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str = "Blues_r",
) -> None:
    n = len(datasets)
    fig_w = 1.6 + 1.4 * n
    fig_h = 1.4 + 1.4 * n
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(Z_mean, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)

    for i in range(n):
        for j in range(n):
            m = Z_mean[i, j]
            s = Z_std[i, j]
            if np.isnan(m):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#999999")
            else:
                txt = f"{m:.3f}\n±{s:.3f}" if not np.isnan(s) else f"{m:.3f}"
                # Pick text color based on background brightness.
                bg = im.cmap(im.norm(m))
                lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                color = "white" if lum < 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Eval dataset")
    ax.set_ylabel("Train dataset")
    ax.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=str,
                        help="Model slug (used to format probe / hidden-states paths).")
    parser.add_argument("--datasets", required=True, type=str, nargs="+",
                        help="Dataset slugs. Used as both train rows and (default) eval columns.")
    parser.add_argument(
        "--eval-datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Subset of --datasets to use as eval columns this run. Defaults to "
            "the full --datasets list. Use this to fan out one eval-dataset per "
            "process for shell-level parallelism (each process loads exactly one "
            "hidden-states cache)."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--probe", required=True, type=str,
                        help="Probe name (e.g. 'count', 'cls_5', 'prompt_only_count').")
    parser.add_argument("--metric", required=True, type=str,
                        help="Aggregate-key inside probes[probe].aggregate (e.g. 'mae', 'accuracy').")
    parser.add_argument(
        "--relative-to",
        type=str,
        default="baseline_constant_median:mae",
        help=(
            "Render an additional column-normalized heatmap where each cell "
            "is divided by the per-eval_dataset value of `<name>:<metric>` "
            "(taken from `examples[i].statistical_baselines[<name>]`). "
            "Format: `<baseline_name>:<metric_key>`. Set to '' (empty) to "
            "skip the second heatmap. Default normalizes by the eval split's "
            "constant-median MAE — score < 1 means the probe beats that "
            "baseline; cross-dataset comparable since the denominator is "
            "dataset-specific."
        ),
    )
    parser.add_argument(
        "--probe-dir-pattern",
        type=str,
        default="outputs/{model}-{dataset}/{model}-{dataset}-{seed}/seed_{seed}/trained_probe",
        help="Format string with {model}, {dataset}, {seed} placeholders.",
    )
    parser.add_argument(
        "--hidden-states-pattern",
        type=str,
        default="outputs/{model}-{dataset}/hidden_states",
        help="Format string with {model}, {dataset} placeholders.",
    )
    parser.add_argument("--output", required=False, type=str, default=None,
                        help="Output figure path (PDF or PNG). Required unless --skip-heatmap.")
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help=(
            "Skip rendering the heatmap. Use this in parallel worker invocations "
            "that only write per-pair JSONs; stitch the heatmap afterward with "
            "--from-jsons."
        ),
    )
    parser.add_argument(
        "--from-jsons",
        action="store_true",
        help=(
            "Skip evaluation entirely. Read per-pair JSONs from --json-dir "
            "(produced by prior --json-dir invocations) and render a heatmap from "
            "their cached cell metrics. Use after all parallel workers finish."
        ),
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues_r",
        help="matplotlib colormap. Use 'Blues_r' for lower-is-better metrics (MAE), 'Blues' for higher-is-better (accuracy).",
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to write per-(train_dataset, eval_dataset) "
            "eval JSONs in the same shape as `evaluate_probes.py`'s output "
            "(seed-merged: aggregate scalars become {mean, std}). When set, "
            "every active probe from the saved config is included in each "
            "JSON, plus statistical baselines fit on the eval dataset. The "
            "files are named train_<train_ds>__eval_on_<eval_ds>.json and "
            "are loadable by the dashboard."
        ),
    )
    args = parser.parse_args()
    write_json = args.json_dir is not None
    render_heatmap = not args.skip_heatmap

    # Parse --relative-to "<name>:<metric>"; empty string disables.
    relative_to: tuple[str, str] | None = None
    if args.relative_to:
        if ":" not in args.relative_to:
            parser.error("--relative-to must be of the form '<baseline_name>:<metric>'.")
        rel_name, rel_metric = args.relative_to.split(":", 1)
        if not rel_name or not rel_metric:
            parser.error("--relative-to must be of the form '<baseline_name>:<metric>'.")
        relative_to = (rel_name, rel_metric)

    if render_heatmap and args.output is None:
        parser.error("--output is required unless --skip-heatmap is set.")
    if args.from_jsons:
        if args.json_dir is None:
            parser.error("--from-jsons requires --json-dir.")
        if args.output is None:
            parser.error("--from-jsons requires --output.")
        _stitch_from_jsons(args, relative_to=relative_to)
        return

    if args.probe not in PROBE_SPECS:
        raise ValueError(
            f"Unknown probe '{args.probe}'. Known: {sorted(PROBE_SPECS.keys())}"
        )

    eval_datasets: list[str] = args.eval_datasets or list(args.datasets)
    unknown = [d for d in eval_datasets if d not in args.datasets]
    if unknown:
        parser.error(
            f"--eval-datasets entries must appear in --datasets; unknown: {unknown}"
        )

    # Find any probe-dir with config.yaml so we can read the model_name and
    # hidden_layer convention. All probe-dirs under one model should share these.
    first_probe_dir: Path | None = None
    for ds in args.datasets:
        for s in args.seeds:
            p = Path(args.probe_dir_pattern.format(model=args.model, dataset=ds, seed=s))
            if (p / "config.yaml").exists():
                first_probe_dir = p
                break
        if first_probe_dir is not None:
            break
    if first_probe_dir is None:
        raise FileNotFoundError(
            "No probe-dir with config.yaml found across the requested combinations. "
            "Check --probe-dir-pattern / --datasets / --seeds, or pass "
            "--probe-dir-pattern with the right path layout."
        )
    cfg = OmegaConf.load(first_probe_dir / "config.yaml")
    print(f"Using config from {first_probe_dir / 'config.yaml'}")

    # Pull active_probes from the saved config so the JSON output and the
    # cell-metric extraction agree on the probe set. When --json-dir is off,
    # we still load all probes (cheap inference) so the heatmap result is
    # identical regardless of the flag.
    cfg_active_probes = list(
        cfg.optimization.get("probes", list(PROBE_SPECS.keys()))
    )
    if args.probe not in cfg_active_probes:
        # The user asked to plot a probe the saved config doesn't list. Still
        # try — `evaluate_many_probe_dirs` will look for the head file directly.
        cfg_active_probes = [args.probe] + cfg_active_probes
    classification_sigma = float(cfg.optimization.get("classification_sigma", 0.15))

    # Tokenizer: prefer one stashed alongside the probe dir, else HF model name.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(first_probe_dir))
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # Build eval contexts once per eval_dataset (only the ones this process owns).
    eval_contexts: dict[str, dict[str, Any]] = {}
    for eval_ds in eval_datasets:
        print(f"\nLoading eval context for {eval_ds}...")
        try:
            eval_contexts[eval_ds] = _build_eval_context(
                args.model,
                eval_ds,
                args.hidden_states_pattern,
                cfg_active_probes,
                cfg,
                load_stat_baselines=write_json,
            )
        except Exception as exc:
            print(f"  [warning] eval dataset {eval_ds} unavailable: {type(exc).__name__}: {exc}")

    # Compute per-cell, per-seed dataset means.
    n = len(args.datasets)
    Z_mean = np.full((n, n), np.nan)
    Z_std = np.full((n, n), np.nan)
    eval_idx = {ds: j for j, ds in enumerate(args.datasets)}
    # Per-eval_ds normalization scalars for the optional column-relative
    # heatmap. Computed once per eval_ds (the baseline value is dataset-fit
    # and seed-invariant; we still average across seeds for float stability).
    rel_per_col: dict[int, float] = {}
    print()
    # Outer loop on eval_ds: every (train_ds × seed) probe-dir that's evaluated
    # on the same eval_ds shares the same hidden-states cache, so a single pass
    # through the eval dataset (`evaluate_many_probe_dirs`) amortizes the
    # per-`__getitem__` load (the dominant cost — ~188 MB read + concat at
    # `hidden_layer="all"`) across them.
    for eval_ds in eval_datasets:
        j = eval_idx[eval_ds]
        if eval_ds not in eval_contexts:
            continue

        # Gather every (train_ds, seed) probe-dir whose plotted-probe head file
        # exists; only those are worth running. `evaluate_many_probe_dirs` will
        # additionally drop dirs whose other-probe heads are missing.
        triple_meta: list[tuple[str, int, Path]] = []
        for train_ds in args.datasets:
            for s in args.seeds:
                probe_dir = Path(args.probe_dir_pattern.format(
                    model=args.model, dataset=train_ds, seed=s,
                ))
                head_path = probe_dir / f"{args.probe}_head.pt"
                if not head_path.exists():
                    print(f"  [skip] {train_ds}/seed_{s} -> {eval_ds}: missing {head_path.name} in {head_path}")
                    continue
                triple_meta.append((train_ds, s, probe_dir))

        if not triple_meta:
            continue

        # ONE pass over this eval_ds for every (train_ds, seed) probe-dir.
        try:
            results = evaluate_many_probe_dirs(
                probe_dirs=[pd for (_, _, pd) in triple_meta],
                eval_dataset=eval_contexts[eval_ds]["eval_dataset"],
                hidden_size=eval_contexts[eval_ds]["hidden_size"],
                active_probes=cfg_active_probes,
                classification_bins=eval_contexts[eval_ds]["classification_bins"],
                cls_anchors=eval_contexts[eval_ds]["cls_anchors"],
                stat_baselines=eval_contexts[eval_ds]["stat_baselines"],
                tokenizer=tokenizer,
            )
        except Exception as exc:
            print(f"  [error] eval_ds={eval_ds} pass failed: {type(exc).__name__}: {exc}")
            continue

        # Group results by train_ds so the existing per-(train_ds, eval_ds)
        # merge / cell-metric / JSON-write code can run unchanged.
        by_train: dict[str, dict[str, list]] = {}
        for (train_ds, _s, probe_dir) in triple_meta:
            if probe_dir not in results:
                continue
            examples_s, kappa_s = results[probe_dir]
            bucket = by_train.setdefault(
                train_ds,
                {"per_seed_examples": [], "per_seed_kappa": [], "used_probe_dirs": []},
            )
            bucket["per_seed_examples"].append(examples_s)
            bucket["per_seed_kappa"].append(kappa_s)
            bucket["used_probe_dirs"].append(str(probe_dir))

        # Per-eval_ds normalization scalar for the column-relative heatmap.
        # Pulled from any one (train_ds, eval_ds) pair's per-seed examples
        # since `statistical_baselines` is fit on the eval_ds train split,
        # not on probe weights — the value is identical across pairs.
        if relative_to is not None and by_train:
            rel_name, rel_metric = relative_to
            sample_pair = next(iter(by_train.values()))
            rel_seed_values: list[float] = []
            for ex_list in sample_pair["per_seed_examples"]:
                v = _extract_cell_metric(
                    ex_list, rel_name, rel_metric, bucket="statistical_baselines",
                )
                if v is not None:
                    rel_seed_values.append(v)
            if rel_seed_values:
                rel_per_col[j] = float(statistics.fmean(rel_seed_values))

        for i, train_ds in enumerate(args.datasets):
            if train_ds not in by_train:
                continue
            per_seed_examples: list[list[dict]] = by_train[train_ds]["per_seed_examples"]
            per_seed_kappa: list[dict[str, float]] = by_train[train_ds]["per_seed_kappa"]
            used_probe_dirs: list[str] = by_train[train_ds]["used_probe_dirs"]

            # Cell metric: per-seed dataset mean → mean ± std across seeds.
            seed_values: list[float] = []
            for ex_list in per_seed_examples:
                v = _extract_cell_metric(ex_list, args.probe, args.metric)
                if v is not None:
                    seed_values.append(v)
            if seed_values:
                Z_mean[i, j] = statistics.fmean(seed_values)
                Z_std[i, j] = (
                    statistics.pstdev(seed_values) if len(seed_values) > 1 else 0.0
                )
            print(
                f"  ({train_ds:>22s} -> {eval_ds:<22s}) "
                f"seeds={len(seed_values)}/{len(args.seeds)} "
                f"mean={Z_mean[i, j]:.4f} std={Z_std[i, j]:.4f}"
            )

            # Per-pair JSON: merge across seeds and write in the same format
            # as `evaluate_probes.py`. The dashboard can load this directly.
            if write_json and per_seed_examples:
                merged_examples = _merge_examples(per_seed_examples)

                # Merge dataset-wide kappa per classification probe.
                merged_kappa: dict[str, dict[str, float] | None] = {}
                kappa_names: set[str] = set()
                for k in per_seed_kappa:
                    kappa_names.update(k.keys())
                for name in sorted(kappa_names):
                    merged = _merge_scalar([k.get(name) for k in per_seed_kappa])
                    if merged is not None:
                        merged_kappa[name] = merged

                # Per-(probe, metric) heatmap cells for every aggregate found in
                # the per-example dicts. Stashed so `--from-jsons` can re-render
                # any heatmap from this JSON without re-running inference.
                all_cell_metrics = _compute_all_cell_metrics(per_seed_examples)

                metadata = {
                    "model_name": cfg.model.model_name,
                    "probe_dir": used_probe_dirs[0] if len(used_probe_dirs) == 1 else None,
                    "probe_dirs": used_probe_dirs,
                    "n_seeds": len(used_probe_dirs),
                    "seed_aggregated": len(used_probe_dirs) > 1,
                    "hidden_states_dir": eval_contexts[eval_ds]["hidden_states_dir"],
                    "hidden_layer": str(cfg.data.get("hidden_layer", "-1")),
                    "probes": cfg_active_probes,
                    "baselines": list(eval_contexts[eval_ds]["stat_baselines"].keys()),
                    "classification_bins": eval_contexts[eval_ds]["classification_bins"],
                    "classification_sigma": classification_sigma,
                    "num_examples": len(merged_examples),
                    "dataset_cohen_kappa": merged_kappa,
                    # Cross-dataset-specific metadata: makes the JSON
                    # self-describing so the dashboard can show train_ds /
                    # eval_ds without parsing the filename.
                    "train_dataset": train_ds,
                    "eval_dataset": eval_ds,
                    "model": args.model,
                    # Per-(probe, metric) cell metrics for heatmap stitching
                    # from JSONs. `--from-jsons` uses this dict to rebuild the
                    # matrix for any (probe, metric) the inference produced,
                    # without re-running inference.
                    "cell_metrics": all_cell_metrics,
                    # Backward-compat: keep the singular `cell_metric` populated
                    # for the (probe, metric) the run was parameterized with,
                    # so older stitcher / downstream code still finds it.
                    "cell_metric": {
                        "probe": args.probe,
                        "metric_key": args.metric,
                        "per_seed": [float(v) for v in seed_values],
                        "mean": (None if math.isnan(Z_mean[i, j])
                                 else float(Z_mean[i, j])),
                        "std": (None if math.isnan(Z_std[i, j])
                                else float(Z_std[i, j])),
                    },
                }
                output_data = {"metadata": metadata, "examples": merged_examples}

                json_path = Path(args.json_dir) / f"train_{train_ds}__eval_on_{eval_ds}.json"
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with json_path.open("w") as f:
                    json.dump(output_data, f)
                print(f"     [json] {json_path}")

    # Render.
    if not render_heatmap:
        print("\n--skip-heatmap set; not rendering. Stitch later with --from-jsons.")
        return
    title = args.title or (
        f"{args.model}: {args.probe}/{args.metric}\n"
        f"mean ± std across {len(args.seeds)} seeds"
    )
    _render_heatmap(
        datasets=args.datasets,
        Z_mean=Z_mean,
        Z_std=Z_std,
        output_path=Path(args.output),
        title=title,
        cmap=args.cmap,
    )
    print(f"\nSaved heatmap to {args.output}")
    _write_heatmap_csvs(args.datasets, Z_mean, Z_std, Path(args.output))

    # Second heatmap: each column normalized by its eval_dataset's baseline
    # scalar. Score < 1 means the probe beats that baseline; cross-dataset
    # comparable since the denominator is dataset-specific.
    if relative_to is not None:
        _render_relative_heatmap(
            datasets=args.datasets,
            Z_mean=Z_mean,
            Z_std=Z_std,
            rel_per_col=rel_per_col,
            relative_to=relative_to,
            base_output=Path(args.output),
            title_prefix=f"{args.model}: {args.probe}/{args.metric}",
            n_seeds=len(args.seeds),
            cmap=args.cmap,
        )


def _render_relative_heatmap(
    datasets: list[str],
    Z_mean: np.ndarray,
    Z_std: np.ndarray,
    rel_per_col: dict[int, float],
    relative_to: tuple[str, str],
    base_output: Path,
    title_prefix: str,
    n_seeds: int,
    cmap: str,
) -> None:
    """Render the column-normalized heatmap as a sibling of the main figure.

    `Z_mean[i, j] / rel_per_col[j]` and `Z_std[i, j] / rel_per_col[j]` per
    cell. Columns lacking a normalizer (the baseline wasn't in the JSON or
    couldn't be computed) render as NaN.
    """
    rel_name, rel_metric = relative_to
    n = len(datasets)
    Zr_mean = np.full((n, n), np.nan)
    Zr_std = np.full((n, n), np.nan)
    for j in range(n):
        denom = rel_per_col.get(j)
        if denom is None or denom == 0:
            continue
        col_mean = Z_mean[:, j]
        col_std = Z_std[:, j]
        Zr_mean[:, j] = col_mean / denom
        Zr_std[:, j] = col_std / denom

    out_path = base_output.with_name(
        f"{base_output.stem}__rel_{rel_name}_{rel_metric}{base_output.suffix}"
    )
    title = (
        f"{title_prefix} relative to {rel_name}.{rel_metric} (per eval-ds)\n"
        f"mean ± std across {n_seeds} seeds  •  values < 1 beat the baseline"
    )
    _render_heatmap(
        datasets=datasets,
        Z_mean=Zr_mean,
        Z_std=Zr_std,
        output_path=out_path,
        title=title,
        cmap=cmap,
    )
    print(f"Saved relative heatmap to {out_path}")
    _write_heatmap_csvs(datasets, Zr_mean, Zr_std, out_path)


def _stitch_from_jsons(args, relative_to: tuple[str, str] | None = None) -> None:
    """Read per-pair JSONs produced by `--json-dir` runs and render the heatmap.

    First-class source of cell scalars is `metadata.cell_metrics[probe][metric]`,
    which inference now populates for every (probe, metric) the per-example
    aggregates contain — so any (`--probe`, `--metric`) the run produced can be
    rendered without re-running. Falls back to the legacy
    `metadata.cell_metric` (singular) for JSONs written before this change.
    Pairs whose JSON is missing — or whose JSON has neither path populated for
    the requested combo — render as NaN.
    """
    json_dir = Path(args.json_dir)
    n = len(args.datasets)
    Z_mean = np.full((n, n), np.nan)
    Z_std = np.full((n, n), np.nan)
    idx = {ds: k for k, ds in enumerate(args.datasets)}
    rel_per_col: dict[int, float] = {}

    found = 0
    for train_ds in args.datasets:
        for eval_ds in args.datasets:
            path = json_dir / f"train_{train_ds}__eval_on_{eval_ds}.json"
            if not path.exists():
                continue
            try:
                with path.open() as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"  [warn] could not read {path.name}: {type(exc).__name__}: {exc}")
                continue
            md = data.get("metadata") or {}
            cm = md.get("cell_metrics") or {}

            # New path: cell_metrics[probe][metric] indexes any (probe, metric).
            entry = (cm.get(args.probe) or {}).get(args.metric)
            # Legacy path: cell_metric (singular) — only the run's parameterized pair.
            if entry is None:
                legacy = md.get("cell_metric") or {}
                if (
                    legacy.get("probe") == args.probe
                    and legacy.get("metric_key") == args.metric
                ):
                    entry = legacy
            if entry is None:
                continue

            m = entry.get("mean")
            s = entry.get("std")
            j = idx[eval_ds]
            if isinstance(m, (int, float)):
                Z_mean[idx[train_ds], j] = float(m)
            if isinstance(s, (int, float)):
                Z_std[idx[train_ds], j] = float(s)
            found += 1

            # Capture this eval_ds's normalization scalar from cell_metrics.
            # Same value across all (train_ds, eval_ds) pairs for one eval_ds,
            # so first-write-wins is fine. Falls back to recomputing from the
            # JSON's `examples[i].statistical_baselines` when this JSON predates
            # the change that started caching baselines under `cell_metrics`.
            if relative_to is not None and j not in rel_per_col:
                rel_name, rel_metric = relative_to
                rel_entry = (cm.get(rel_name) or {}).get(rel_metric)
                rel_mean = (rel_entry or {}).get("mean") if isinstance(rel_entry, dict) else None
                if not isinstance(rel_mean, (int, float)):
                    # Examples-array fallback — cheap (per-token arrays only),
                    # no re-inference needed.
                    rel_mean = _extract_cell_metric(
                        data.get("examples") or [],
                        rel_name,
                        rel_metric,
                        bucket="statistical_baselines",
                    )
                if isinstance(rel_mean, (int, float)) and rel_mean != 0:
                    rel_per_col[j] = float(rel_mean)

    if found == 0:
        raise FileNotFoundError(
            f"No matching JSONs in {json_dir} for probe={args.probe} metric={args.metric}. "
            f"Make sure the parallel workers ran with --probe/--metric matching."
        )

    title = args.title or (
        f"{args.model}: {args.probe}/{args.metric}\n"
        f"mean ± std across {len(args.seeds)} seeds (stitched from {found} JSONs)"
    )
    _render_heatmap(
        datasets=args.datasets,
        Z_mean=Z_mean,
        Z_std=Z_std,
        output_path=Path(args.output),
        title=title,
        cmap=args.cmap,
    )
    print(f"\nStitched heatmap from {found} JSON(s); saved to {args.output}")
    _write_heatmap_csvs(args.datasets, Z_mean, Z_std, Path(args.output))

    # Second heatmap: column-normalized by per-eval_ds baseline scalar.
    if relative_to is not None:
        _render_relative_heatmap(
            datasets=args.datasets,
            Z_mean=Z_mean,
            Z_std=Z_std,
            rel_per_col=rel_per_col,
            relative_to=relative_to,
            base_output=Path(args.output),
            title_prefix=f"{args.model}: {args.probe}/{args.metric}",
            n_seeds=len(args.seeds),
            cmap=args.cmap,
        )


if __name__ == "__main__":
    main()
