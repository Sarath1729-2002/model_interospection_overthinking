"""
Offline probe training script using pre-extracted hidden states.

This script trains the probe WITHOUT the base model, using pre-saved hidden states.
Much faster than online training since no generation is needed.

Usage:
    # Basic usage (switch data builder to hidden_states and point to saved files)
    uv run python src/miol/train_offline.py data=hidden_states data.hidden_states_dir=./outputs/hidden_states

    # With custom config
    uv run python src/miol/train_offline.py \
        data=hidden_states \
        data.hidden_states_dir=./hidden_states_gsm8k/hidden_states \
        learning_rate=0.001 \
        max_steps=1000

    # Use all layers instead of just last layer
    uv run python src/miol/train_offline.py \
        data=hidden_states \
        data.hidden_states_dir=./outputs/hidden_states \
        data.hidden_layer=all

    # Multi-layer probe sweep: train every hidden-state probe on each listed
    # layer in a single run, sharing one disk pass per batch. Baselines that
    # don't read hidden states (constant / position) are not duplicated.
    uv run python src/miol/train_offline.py \
        data=hidden_states \
        data.hidden_states_dir=./outputs/hidden_states \
        'data.layers_to_train=[1,8,16,24,32,-1]'

    # Same, sweeping every available layer:
    uv run python src/miol/train_offline.py \
        data=hidden_states \
        data.hidden_states_dir=./outputs/hidden_states \
        data.layers_to_train=all
"""

import os
from functools import partial
from pathlib import Path

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from miol.model import MAEPerBinCallback
from miol.model.offline_trainer import MultiProbeOfflineTrainer
from miol.model.callback import ExampleDisplayCallback2
from miol.model.probe_specs import PROBE_SPECS
from miol.model.generation_utils import compute_percentage_targets, compute_soft_class_targets
from miol.data import (
    compute_metrics,
    InfiniteSamplerWithReplacement,
    HiddenStatesDatasetBuilder,
)


def is_main_process():
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in [-1, 0]


def print_main(*args, **kwargs):
    """Print only from the main process."""
    if is_main_process():
        print(*args, **kwargs)


class MultiProbeModel(torch.nn.Module):
    """Wraps multiple probe heads in a ModuleDict for multi-probe training.

    The head's input dimension depends on spec['input_type']:
    - 'hidden_states' (default): hidden_size from the base model
    - 'constant':                1 (probe head is effectively a learnable bias)
    - 'position':                1 (completion position scalar)

    In multi-layer mode the spec dict contains layer-namespaced names like
    `count_L8` (with `base_probe_name="count"` and a `layer_index` slice
    pointer). `token_count_head` is then aliased to whichever count-base head
    has the smallest layer_index — the canonical "count" probe used by the
    display callback and by the trainer's primary-loss bookkeeping.
    """

    def __init__(self, hidden_size, probe_specs):
        super().__init__()
        heads = {}
        for name, spec in probe_specs.items():
            input_type = spec.get("input_type", "hidden_states")
            if input_type == "hidden_states":
                in_dim = hidden_size
            elif input_type in ("constant", "position"):
                in_dim = 1
            else:
                raise ValueError(f"Unknown input_type for probe {name!r}: {input_type}")
            heads[name] = torch.nn.Linear(in_dim, spec["output_dim"])
        self.heads = torch.nn.ModuleDict(heads)
        # Backward compat: ExampleDisplayCallback2 accesses model.token_count_head.
        canonical_count = pick_canonical_probe_name(probe_specs, "count")
        self.token_count_head = self.heads[canonical_count]


def pick_canonical_probe_name(probe_specs, base_name: str):
    """Return the spec key for the 'canonical' instance of a base probe.

    In single-layer mode this is just `base_name` (e.g. "count").
    In multi-layer mode it is the layer-namespaced variant with the smallest
    `layer_index` (e.g. "count_L0" if layers_to_train=[0, 8, 16]).

    Used to drive the trainer's hardcoded count/percentage hooks (HF Trainer
    primary loss, count probe per-bin chart, percentage probe per-bin chart).
    """
    if base_name in probe_specs:
        return base_name
    candidates = [
        (spec.get("layer_index", 0), name)
        for name, spec in probe_specs.items()
        if spec.get("base_probe_name") == base_name
    ]
    if not candidates:
        raise KeyError(
            f"No probe spec entry found for base_probe_name={base_name!r}; "
            f"existing keys: {list(probe_specs.keys())}"
        )
    candidates.sort()
    return candidates[0][1]


def expand_probe_specs_for_layers(active_probe_specs, layers_to_train):
    """Multi-layer expansion: duplicate every hidden-state probe per requested layer.

    Each duplicated spec carries:
      - "layer_index":     the layer index in the data tensor's layer axis
                           (i.e. the slice position, NOT the absolute layer
                           number). The dataset returns layers in the order
                           given by `layers_to_train`, so slice index k
                           corresponds to layers_to_train[k].
      - "layer_label":     the absolute layer number used for the user-facing
                           name and WandB scalar prefix (e.g. `count_L8`).
      - "base_probe_name": the original probe name before namespacing, used
                           for spec-based filtering (e.g. baselines / prompt-only).

    Non-hidden-state probes (input_type in {"constant", "position"}) are not
    duplicated since their output is layer-independent by construction —
    they are passed through once with no layer_index.
    """
    if not layers_to_train:
        return active_probe_specs
    expanded = {}
    for name, spec in active_probe_specs.items():
        if spec.get("input_type", "hidden_states") == "hidden_states":
            for slice_idx, layer_label in enumerate(layers_to_train):
                key = f"{name}_L{layer_label}"
                expanded[key] = {
                    **spec,
                    "layer_index": slice_idx,
                    "layer_label": int(layer_label),
                    "base_probe_name": name,
                }
        else:
            expanded[name] = spec
    return expanded


def hidden_states_collator(batch, probe_specs, classification_sigma=0.15):
    """
    Collate function for batching hidden states with different sequence lengths.
    Pads sequences to the max length in the batch and computes targets for all probes.
    """
    def get_field(item, field):
        if isinstance(item, dict):
            return item[field]
        return getattr(item, field)

    max_len = max(len(get_field(item, "input_ids")) for item in batch)
    batch_size = len(batch)
    sample_hs_shape = get_field(batch[0], "hidden_states").shape
    multi_layer = (sample_hs_shape.__len__() == 3)

    # Core tensors
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    remaining_counts = torch.full((batch_size, max_len), -100, dtype=torch.long)
    if multi_layer:
        # [seq_len, num_layers, hidden_size] from HiddenStatesDataset's multi-layer mode.
        num_layers_in_batch = int(sample_hs_shape[1])
        hidden_size = int(sample_hs_shape[2])
        hidden_states = torch.zeros(
            (batch_size, max_len, num_layers_in_batch, hidden_size), dtype=torch.float32,
        )
    else:
        hidden_size = int(sample_hs_shape[-1])
        hidden_states = torch.zeros((batch_size, max_len, hidden_size), dtype=torch.float32)

    # Per-probe target tensors (only for active probes beyond "count")
    soft_targets = {}

    # Always compute percentage targets — used by the percentage probe AND by
    # the classification probes' derived-percentage metric.
    remaining_percentage = torch.full((batch_size, max_len), -100.0, dtype=torch.float32)

    # Prompt-end-only regression target: total completion length T placed at
    # index (prompt_length - 1) per example, -100 elsewhere. Built only when
    # at least one active probe needs it (train_position == "prompt_end" and
    # target_key is the matching key).
    needs_prompt_end_total_length = any(
        spec.get("train_position") == "prompt_end"
        and spec.get("target_key") == "total_completion_length_at_prompt_end"
        for spec in probe_specs.values()
    )
    total_length_at_prompt_end = (
        torch.full((batch_size, max_len), -100.0, dtype=torch.float32)
        if needs_prompt_end_total_length else None
    )

    for name, spec in probe_specs.items():
        if spec["type"] == "classification":
            num_bins = spec["output_dim"]
            soft_targets[name] = torch.full((batch_size, max_len, num_bins), -100.0, dtype=torch.float32)

    # Metadata
    has_eos_list = []
    prompt_lengths = []
    completion_lengths = []

    # Fill in data and compute targets
    for i, item in enumerate(batch):
        item_input_ids = get_field(item, "input_ids")
        seq_len = len(item_input_ids)
        item_has_eos = get_field(item, "has_eos")
        item_prompt_len = get_field(item, "prompt_length")

        input_ids[i, :seq_len] = item_input_ids
        attention_mask[i, :seq_len] = get_field(item, "attention_mask")
        remaining_counts[i, :seq_len] = get_field(item, "remaining_token_counts")
        hidden_states[i, :seq_len] = get_field(item, "hidden_states")
        has_eos_list.append(item_has_eos)
        prompt_lengths.append(item_prompt_len)
        completion_lengths.append(get_field(item, "completion_length"))

        # Compute percentage targets
        pct = compute_percentage_targets(seq_len, item_prompt_len, item_has_eos, device=torch.device("cpu"))
        remaining_percentage[i, :seq_len] = pct

        # Compute soft classification targets for each classification probe
        for name, spec in probe_specs.items():
            if spec["type"] == "classification":
                num_bins = spec["output_dim"]
                soft = compute_soft_class_targets(
                    seq_len, item_prompt_len, item_has_eos,
                    device=torch.device("cpu"), sigma=classification_sigma, num_bins=num_bins,
                )
                soft_targets[name][i, :seq_len] = soft

        # Place T (= completion_length) at the prompt-end position. Skip
        # truncated sequences (has_eos=False) — their completion_length
        # doesn't reflect the model's intended T.
        if total_length_at_prompt_end is not None and item_has_eos and item_prompt_len >= 1:
            item_completion_len = get_field(item, "completion_length")
            total_length_at_prompt_end[i, item_prompt_len - 1] = float(item_completion_len)

    # log(1 + count) target, with -100 preserved as -100 (so masked positions
    # remain masked under the same convention as remaining_token_counts).
    remaining_log_count = torch.where(
        remaining_counts == -100,
        torch.full_like(remaining_counts, -100, dtype=torch.float32),
        torch.log1p(remaining_counts.to(torch.float32)),
    )

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "remaining_token_counts": remaining_counts,
        "hidden_states": hidden_states,
        "has_eos": torch.tensor(has_eos_list, dtype=torch.bool),
        "prompt_length": torch.tensor(prompt_lengths, dtype=torch.long),
        "completion_length": torch.tensor(completion_lengths, dtype=torch.long),
    }

    result["remaining_percentage"] = remaining_percentage
    result["remaining_log_count"] = remaining_log_count

    if total_length_at_prompt_end is not None:
        result["total_completion_length_at_prompt_end"] = total_length_at_prompt_end

    for name, tensor in soft_targets.items():
        result[probe_specs[name]["target_key"]] = tensor

    return result


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    try:
        return _main_impl(cfg)
    except Exception as e:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        print(f"[Rank {local_rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise


def _resolve_resume(cfg: DictConfig) -> tuple[DictConfig, str | None]:
    """If `cfg.resume_from` is set, replace cfg with the saved snapshot's config
    (preserving a small allowlist of overrides from the current cfg) and return
    a (new_cfg, resume_dir) pair. Otherwise return (cfg, None).
    """
    resume_from = str(cfg.get("resume_from", "") or "").strip()
    if not resume_from:
        return cfg, None

    # Try `{resume_from}/config.yaml` first; fall back to one parent up
    # (covers `resume_from` pointing at a `checkpoint-N/` subdir whose sibling
    # `trained_probe/` holds the saved config).
    candidates = [
        os.path.join(resume_from, "config.yaml"),
        os.path.join(resume_from, "trained_probe", "config.yaml"),
        os.path.join(os.path.dirname(resume_from.rstrip("/")), "trained_probe", "config.yaml"),
        os.path.join(os.path.dirname(resume_from.rstrip("/")), "config.yaml"),
    ]
    config_path = next((p for p in candidates if os.path.isfile(p)), None)
    if config_path is None:
        raise FileNotFoundError(
            f"Could not locate config.yaml near resume_from={resume_from}. Tried: {candidates}"
        )

    saved_cfg = OmegaConf.load(config_path)
    print_main(f"[resume] Loaded saved config from {config_path}")

    # Allowlist: keys we let the current CLI keep overriding even on resume.
    allowlist = ["resume_from", "max_steps", "output_dir"]
    if "wandb" in cfg:
        allowlist.append("wandb.run_name")

    # Start from the saved snapshot, then re-apply the allowed CLI overrides.
    new_cfg = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=True))
    for key in allowlist:
        # OmegaConf.select returns None when missing, so we only re-apply when present.
        cur_val = OmegaConf.select(cfg, key)
        if cur_val is not None and cur_val != "":
            OmegaConf.update(new_cfg, key, cur_val, merge=True)

    return new_cfg, resume_from


def _main_impl(cfg: DictConfig):
    # If resuming, replace cfg with the saved snapshot's config (with a small
    # allowlist of CLI overrides preserved). resume_dir is the path to load
    # probe state from and to pass to trainer.train(resume_from_checkpoint=...).
    cfg, resume_dir = _resolve_resume(cfg)

    # Seed all sources of randomness as early as possible. transformers.set_seed
    # covers Python random, numpy, torch (CPU + CUDA). The HF Trainer's own
    # `seed` arg seeds dataloader shuffling and dropout init separately.
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    # Print config
    if is_main_process():
        print_main("=" * 80)
        print_main("OFFLINE PROBE TRAINING - Using Pre-Extracted Hidden States")
        if resume_dir:
            print_main(f"RESUMING FROM: {resume_dir}")
        print_main(f"SEED: {seed}")
        print_main("=" * 80)
        print_main(OmegaConf.to_yaml(cfg))
        print_main("=" * 80 + "\n")

    # Setup. Append seed to output_dir so multi-seed runs don't collide.
    output_dir = os.path.join(cfg.output_dir, f"{cfg.wandb.run_name}/seed_{seed}")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    os.makedirs(output_dir, exist_ok=True)

    probe_save_dir = f"{output_dir}/trained_probe"

    # Save the resolved hydra config alongside the probe checkpoints
    if is_main_process():
        os.makedirs(probe_save_dir, exist_ok=True)
        with open(f"{probe_save_dir}/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        print_main(f"Config saved to {probe_save_dir}/config.yaml")

    # Initialize wandb
    if cfg.wandb.enabled and is_main_process():
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        # Declare `train/epoch` as the canonical step metric before any logging
        # (including the pre-training baseline scalars below). CustomTrainer
        # repeats this call defensively; `define_metric` is idempotent.
        wandb.define_metric("train/epoch")
        wandb.define_metric("*", step_metric="train/epoch")

    # Load hidden states dataset
    print_main("=" * 80)
    print_main("LOADING PRE-EXTRACTED HIDDEN STATES")
    print_main("=" * 80)

    dataset_builder = instantiate(cfg.data)
    streaming = not isinstance(dataset_builder, HiddenStatesDatasetBuilder)

    # Compute statistical baselines once up front and log them to wandb so the
    # run carries the baseline numbers alongside the trained-probe metrics. The
    # baselines are seed-independent; we re-compute per training run anyway so
    # every wandb run is self-contained (no need to merge with a separate
    # baseline run for the figures table to find the values).
    if not streaming and cfg.wandb.enabled and is_main_process():
        try:
            from miol.statistical_baseline import compute_baselines
            baseline_scalars = compute_baselines(
                hidden_states_root=Path(cfg.data.hidden_states_dir),
                bin_size=int(cfg.optimization.get("bin_size", 10)),
            )
            # Anchor pre-training baselines at epoch 0 so they line up on the
            # left edge of charts when the x-axis is `train/epoch`.
            baseline_scalars["train/epoch"] = 0.0
            wandb.log(baseline_scalars)
            print_main(f"Logged {len(baseline_scalars)} statistical-baseline scalars to wandb.")
        except Exception as exc:
            # A baseline failure should never kill training. Common causes:
            # no train samples with has_eos=True, hidden_states dir layout
            # not matching the expected train/ + eval/ subdirs.
            print_main(f"[warning] failed to compute statistical baselines: {type(exc).__name__}: {exc}")

    # Tokenizer is needed by both paths (streaming uses it during generation,
    # offline uses it just for the per-token display callback).
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Hidden layer config — same key used in offline (cfg.data.hidden_layer);
    # absent for prompt-bearing datasets, in which case it defaults to 'all'.
    hidden_layer_cfg = str(cfg.data.get('hidden_layer', 'all'))

    if streaming:
        from miol.data import tokenize_function
        train_dataset_raw = dataset_builder.get_dataset(split='train', tokenizer=tokenizer)
        eval_dataset_raw = dataset_builder.get_dataset(split='validation', tokenizer=tokenizer)
        train_dataset = train_dataset_raw.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=train_dataset_raw.column_names,
            desc="Tokenizing train",
        )
        eval_dataset = eval_dataset_raw.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=eval_dataset_raw.column_names,
            desc="Tokenizing eval",
        )
    else:
        train_dataset = dataset_builder.get_dataset(split='train')
        eval_dataset = dataset_builder.get_dataset(split='validation')
        # Hold out part of TRAIN as the validation set that drives the LR
        # scheduler, early stopping and best-head selection. The real eval
        # (test) split is then touched only by evaluate_probes.py at the end.
        val_fraction = float(cfg.get("val_fraction", 0.0) or 0.0)
        if val_fraction > 0:
            from miol.data.hidden_states_dataset import split_indices
            tr_idx, va_idx = split_indices(len(train_dataset), val_fraction, int(cfg.get("val_split_seed", 0)))
            print_main(f"Validation split: {len(tr_idx)} train / {len(va_idx)} val "
                       f"(val_fraction={val_fraction}); test split ({len(eval_dataset)}) not used during training")
            eval_dataset = train_dataset.subset(va_idx)
            train_dataset = train_dataset.subset(tr_idx)

    print_main(f"Train dataset size: {len(train_dataset)}")
    print_main(f"Eval dataset size: {len(eval_dataset)}")
    print_main(f"Mode: {'STREAMING (generate + extract on the fly)' if streaming else 'OFFLINE (cached hidden states)'}")
    print_main(f"Using hidden layer: {hidden_layer_cfg}")

    # Multi-layer mode: when set, every hidden-state probe is duplicated per
    # listed layer. The dataset returns 3-D hidden states [seq_len, L, hidden]
    # and we resolve any negative / "all" specifiers up front so the trainer
    # gets a concrete list of *absolute* layer indices for naming.
    layers_to_train_raw = cfg.data.get("layers_to_train", None) if "data" in cfg else None
    layers_to_train = None
    if layers_to_train_raw is not None and layers_to_train_raw != "":
        if streaming:
            raise NotImplementedError(
                "cfg.data.layers_to_train is not supported in streaming mode yet."
            )
        # The builder already resolved "all"; here we only see a list (or None).
        # Pull the resolved list off the eval dataset which the builder built.
        ds_layers = getattr(train_dataset, "layers_to_train", None)
        if ds_layers is None:
            raise RuntimeError(
                "cfg.data.layers_to_train was set, but the dataset returned no resolved layer list. "
                "Check that the active data builder is HiddenStatesDatasetBuilder."
            )
        layers_to_train = list(ds_layers)
        print_main(f"Multi-layer mode: training probes on layers {layers_to_train}")

    # Filter probe specs based on config, then optionally expand across layers.
    active_probe_names = list(cfg.optimization.get("probes", list(PROBE_SPECS.keys())))
    base_active_probe_specs = {k: PROBE_SPECS[k] for k in active_probe_names if k in PROBE_SPECS}
    if "count" not in base_active_probe_specs:
        raise ValueError("The 'count' probe is required (used for primary metrics and callbacks).")
    active_probe_specs = expand_probe_specs_for_layers(base_active_probe_specs, layers_to_train)

    model_name = cfg.model.model_name
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    # display_every_n_steps == 0 disables the per-token display callback. The
    # base model is loaded only if it's needed (always in streaming mode for
    # generation; otherwise only for the display callback).
    display_callback_enabled = int(cfg.display_every_n_steps) > 0
    needs_base_model = streaming or display_callback_enabled
    base_model = None
    if needs_base_model:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if streaming:
            for p in base_model.parameters():
                p.requires_grad = False
            base_model.eval()
    else:
        print_main("display_every_n_steps=0 (offline mode) → skipping base model load and per-token display callback")

    # Hidden size for the probe heads' input dimension. In multi-layer mode
    # the data tensor is 3-D ([seq_len, num_layers, hidden_size]); the probe
    # input remains 2-D after the per-layer slice in `_build_probe_input`, so
    # the head's input dim is still `hidden_size`.
    if streaming:
        # Read from the base-model config (no cached hidden states to inspect).
        base_hidden = int(base_model.config.hidden_size)
        if hidden_layer_cfg == "all":
            hidden_size = base_hidden * (int(base_model.config.num_hidden_layers) + 1)
        else:
            hidden_size = base_hidden
    else:
        # Read from the first cached sample. Last dim is the hidden size in
        # both single-layer (2-D) and multi-layer (3-D) modes.
        sample = train_dataset[0]
        hidden_size = sample["hidden_states"].shape[-1]
    print_main(f"Hidden size: {hidden_size}")

    # Create multi-probe model
    print_main("\n" + "=" * 80)
    print_main(f"CREATING {len(active_probe_specs)} PROBE HEADS")
    print_main("=" * 80)
    for name, spec in active_probe_specs.items():
        layer_tag = f" [layer {spec['layer_label']}]" if "layer_label" in spec else ""
        print_main(f"  {name}: {spec['type']}, output_dim={spec['output_dim']}{layer_tag}")
    model = MultiProbeModel(hidden_size, active_probe_specs)

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print_main(f"Probe parameters: {total_params:,}")
    print_main("Note: Base model not loaded - using pre-extracted hidden states!")

    # Training configuration
    print_main("\n" + "=" * 80)
    print_main("TRAINING CONFIGURATION")
    print_main("=" * 80)
    print_main(f"Training mode: Offline (pre-extracted hidden states)")
    print_main(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print_main(f"Train batch size per device: {cfg.batch_size}")
    print_main(f"Eval batch size per device: {cfg.per_device_eval_batch_size}")
    print_main(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
    print_main(f"Effective batch size: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    print_main(f"Learning rate: {cfg.learning_rate}")
    print_main(f"Max training steps: {cfg.max_steps}")
    print_main(f"Output directory: {output_dir}")
    print_main(f"Wandb logging: {'Enabled' if cfg.wandb.enabled else 'Disabled'}")
    print_main("=" * 80 + "\n")

    # Training arguments
    training_args_dict = {
        "output_dir": output_dir,
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.optimization.weight_decay,
        "logging_steps": cfg.optimization.logging_steps,
        "save_steps": cfg.optimization.save_steps,
        "save_total_limit": cfg.optimization.save_total_limit,
        "lr_scheduler_type": "constant",
        "eval_strategy": cfg.eval_strategy,
        "eval_steps": cfg.eval_steps,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": False,
        "dataloader_pin_memory": cfg.dataloader.pin_memory,
        "dataloader_num_workers": cfg.dataloader.num_workers,
        "max_grad_norm": cfg.optimization.max_grad_norm,
        "report_to": "wandb" if cfg.wandb.enabled else "none",
        "run_name": cfg.wandb.run_name,
        "disable_tqdm": local_rank not in [-1, 0],
        "remove_unused_columns": False,
        "metric_for_best_model": "mae",
        "seed": seed,
        "data_seed": seed,
    }

    training_args = TrainingArguments(**training_args_dict)

    # Create infinite sampler for train dataset (seeded so each run with a
    # different cfg.seed observes a different sequence of training batches).
    train_sampler = InfiniteSamplerWithReplacement(
        data_source=train_dataset,
        seed=seed,
    )

    # Setup scheduler factory if using ReduceLROnPlateau
    scheduler_factory = None
    if cfg.scheduler.get("scheduler_factory") is not None:
        scheduler_factory = instantiate(cfg.scheduler.scheduler_factory)

    # Prepare generation config (used in both modes — for streaming generation
    # and for the display callback's per-token preview).
    generation_config = {
        'max_new_tokens': cfg.generation.max_new_tokens,
        'do_sample': cfg.generation.do_sample,
        'temperature': cfg.generation.temperature,
        'top_p': cfg.generation.top_p,
        'top_k': cfg.generation.top_k,
    }

    # Create per-probe optimizers
    probe_optimizers = {}
    for name in active_probe_specs:
        probe_optimizers[name] = torch.optim.AdamW(
            model.heads[name].parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.optimization.weight_decay,
        )

    classification_sigma = cfg.optimization.get("classification_sigma", 0.15)

    # Collator: streaming uses HF's default (per-item dicts already have
    # {input_ids, attention_mask} as fixed-length lists from tokenize_function);
    # offline uses the rich collator that pads hidden_states + computes targets.
    if streaming:
        from transformers import default_data_collator as _hf_default_collator
        data_collator = _hf_default_collator
    else:
        data_collator = partial(
            hidden_states_collator,
            probe_specs=active_probe_specs,
            classification_sigma=classification_sigma,
        )

    # Create callback to display example predictions (count probe only).
    # Only when explicitly enabled — this is the path that requires the base model.
    callbacks = []
    # MAE/MAPE/SMAPE bucket charts: share display cadence when display is on,
    # otherwise fire on every eval step.
    mae_log_every = int(cfg.display_every_n_steps) if display_callback_enabled else int(cfg.eval_steps)
    mae_callback = MAEPerBinCallback(log_every_n_steps=mae_log_every)
    callbacks.append(mae_callback)

    es_cfg = cfg.get("early_stopping", None)
    plateau_controller = None
    if (not streaming) and es_cfg is not None and bool(es_cfg.get("enabled", False)):
        from miol.model.callback import ValPlateauController
        plateau_controller = ValPlateauController(
            patience=int(es_cfg.get("patience", 5)),
            min_delta=float(es_cfg.get("min_delta", 0.25)),
            stop=True,
        )
        callbacks.append(plateau_controller)

    if display_callback_enabled:
        display_callback = ExampleDisplayCallback2(
            base_model=base_model,
            probe_model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            display_every_n_steps=cfg.display_every_n_steps,
            generation_config=generation_config,
        )
        callbacks.append(display_callback)

    # Resolve canonical probe names. In single-layer mode these are just
    # "count" / "percentage"; in multi-layer mode they become e.g. "count_L0"
    # — the layer-namespaced variant the trainer's hardcoded count/percentage
    # hooks will operate on.
    canonical_count_name = pick_canonical_probe_name(active_probe_specs, "count")
    canonical_percentage_name = (
        pick_canonical_probe_name(active_probe_specs, "percentage")
        if any(spec.get("base_probe_name") == "percentage" or name == "percentage"
               for name, spec in active_probe_specs.items())
        else None
    )

    # Create multi-probe trainer
    trainer = MultiProbeOfflineTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        bin_size=cfg.optimization.bin_size,
        generation_config=generation_config if streaming else {},
        train_sampler=train_sampler,
        scheduler_factory=scheduler_factory,
        loss_fn_name=cfg.optimization.loss_function,
        probe_specs=active_probe_specs,
        probe_optimizers=probe_optimizers,
        streaming=streaming,
        streaming_base_model=base_model if streaming else None,
        streaming_tokenizer=tokenizer if streaming else None,
        streaming_hidden_layer=hidden_layer_cfg,
        canonical_count_name=canonical_count_name,
        canonical_percentage_name=canonical_percentage_name,
    )
    # Pass classification sigma through for streaming target computation.
    trainer._streaming_classification_sigma = classification_sigma

    # Link callback to trainer
    mae_callback.trainer = trainer
    if plateau_controller is not None:
        plateau_controller.trainer = trainer

    # Restore probe heads + per-probe optimizer states if resuming.
    # HF Trainer.train(resume_from_checkpoint=...) handles its own state
    # (global_step, best_metric, RNG, the single self.optimizer).
    if resume_dir:
        trainer.load_probe_state(resume_dir)

    # Train
    print_main("Starting offline multi-probe training...")
    print_main(f"Training {len(active_probe_specs)} probes on {len(train_dataset)} samples, evaluating on {len(eval_dataset)} samples")
    if resume_dir:
        trainer.train(resume_from_checkpoint=resume_dir)
    else:
        trainer.train()

    # Save final probe heads + per-probe optimizer states alongside the
    # already-saved config.yaml. Delegating to trainer._save keeps the layout
    # consistent with the HF periodic checkpoints.
    print_main("\nSaving trained probes...")
    trainer._save(probe_save_dir)
    print_main(f"  probes + optimizers saved to {probe_save_dir}")

    if cfg.wandb.enabled and is_main_process():
        wandb.finish()

    print_main("\nOffline training complete!")


if __name__ == "__main__":
    main()
