"""
Training script for training token count prediction probe on frozen language model.

This script uses Hydra for configuration management. Config file: conf/config.yaml
Note: Only the token count probe is trained - the base model is frozen.

Basic usage:
    uv run python src/miol/train.py

Override config parameters:
    uv run python src/miol/train.py batch_size=8 learning_rate=0.0001

For multi-GPU training, run with torchrun:
    uv run torchrun --nproc_per_node=NUM_GPUS src/miol/train.py

Or with accelerate:
    uv run accelerate launch src/miol/train.py
"""

import logging
import os

import hydra
import numpy as np
import torch
import wandb
from datasets import Dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as transformers_logging

from miol.model import ModelWithTokenCountHead, CustomTrainer, ExampleDisplayCallback, MAEPerBinCallback
from miol.data import tokenize_function, compute_metrics, InfiniteSamplerWithReplacement


def is_main_process():
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in [-1, 0]


def print_main(*args, **kwargs):
    """Print only from the main process."""
    if is_main_process():
        print(*args, **kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    try:
        return _main_impl(cfg)
    except Exception as e:
        # Print errors on all ranks to help debug distributed training issues
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        print(f"[Rank {local_rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Clean up distributed training before exiting
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise


def _main_impl(cfg: DictConfig):
    # Enable better error messages from torch distributed (set to INFO to avoid too much output)
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")

    # Clean up any stale distributed environment variables if not using torchrun
    # This prevents errors when leftover env vars exist from previous runs
    if "LOCAL_RANK" not in os.environ:
        # Not using torchrun, so clean up any distributed env vars
        for key in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
            os.environ.pop(key, None)

    # Print config for debugging (only on main process)
    if is_main_process():
        print_main("=" * 80)
        print_main("HYDRA CONFIGURATION")
        print_main("=" * 80)
        print_main(OmegaConf.to_yaml(cfg))
        print_main("=" * 80 + "\n")

    # Detect distributed training setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Configure logging to only show on main process
    if local_rank not in [-1, 0]:
        # Disable all logging on non-main processes
        transformers_logging.set_verbosity_error()
        logging.getLogger().setLevel(logging.ERROR)
        # Disable datasets logging
        logging.getLogger("datasets").setLevel(logging.ERROR)
        # Disable warnings
        import warnings

        warnings.filterwarnings("ignore")
    else:
        # Main process uses default logging
        transformers_logging.set_verbosity_info()

    if local_rank != -1:
        print_main(f"Running distributed training: Rank {local_rank}/{world_size}")
    elif torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print_main(f"Running on {num_gpus} GPU(s)")
        if num_gpus > 1:
            print_main(
                "Note: For optimal multi-GPU performance, use: torchrun --nproc_per_node=NUM_GPUS"
            )

    # Configuration from Hydra config
    model_name = cfg.model.model_name

    # Seed all sources of randomness as early as possible. Append the seed to
    # output_dir so multi-seed runs don't collide on disk.
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    output_dir = os.path.join(cfg.output_dir, f"{cfg.wandb.run_name}/seed_{seed}")

    # Initialize wandb (only on main process)
    if cfg.wandb.enabled and is_main_process():
        # Convert OmegaConf to dict for wandb
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Add computed/runtime values
        config_dict["effective_batch_size"] = cfg.batch_size * cfg.gradient_accumulation_steps
        config_dict["bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        config_dict["fp16"] = False
        config_dict["training_mode"] = "probe_only"
        config_dict["base_model_frozen"] = True

        # Extract dataset type from _target_ (e.g., "miol.data.SyntheticDatasetBuilder" -> "SyntheticDatasetBuilder")
        dataset_type = cfg.data._target_.split('.')[-1] if '_target_' in cfg.data else 'Unknown'

        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=config_dict,
            tags=[cfg.model.model_name, dataset_type],
        )
        # Declare `train/epoch` as the canonical step metric so all charts in
        # the wandb UI can be re-plotted against epoch. CustomTrainer also
        # calls this on init; `define_metric` is idempotent.
        wandb.define_metric("train/epoch")
        wandb.define_metric("*", step_metric="train/epoch")

    print_main("=" * 80)
    print_main("TRAINING CONFIGURATION")
    print_main("=" * 80)
    print_main(f"Model: {model_name}")
    print_main(f"Training mode: Probe only (base model frozen)")

    # GPU configuration
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if local_rank != -1:
            print_main(
                f"Multi-GPU training: {world_size} GPUs (DDP - Rank {local_rank})"
            )
        else:
            print_main(f"Available GPUs: {num_gpus}")
            if num_gpus > 1:
                print_main(f"⚠️  Multiple GPUs detected but not using DDP!")
                print_main(
                    f"   For faster training, run: torchrun --nproc_per_node={num_gpus} src/miol/train.py"
                )
    else:
        print_main("Device: CPU")

    print_main(f"Training samples: {cfg.data.num_train_samples}")
    print_main(f"Evaluation samples: {cfg.data.num_eval_samples}")
    print_main(f"Train batch size per device: {cfg.batch_size}")
    print_main(f"Eval batch size per device: {cfg.per_device_eval_batch_size}")
    print_main(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
    total_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
    if local_rank != -1:
        total_batch_size *= world_size
    print_main(f"Effective batch size: {total_batch_size}")
    print_main(f"Learning rate: {cfg.learning_rate}")
    print_main(f"Max training steps: {cfg.max_steps}")
    print_main(f"Gradient checkpointing: Disabled (base model frozen)")
    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print_main(f"Mixed precision: {'BF16' if bf16_enabled else 'Disabled (FP32)'}")
    print_main(f"Output directory: {output_dir}")
    print_main(
        f"Wandb logging: {'Disabled' if not cfg.wandb.enabled else f'Enabled (project: {cfg.wandb.project})'}"
    )
    print_main(f"\nGeneration settings:")
    print_main(f"  Max new tokens: {cfg.generation.max_new_tokens}")
    print_main(f"  Sampling: {'Enabled' if cfg.generation.do_sample else 'Disabled (greedy)'}")
    if cfg.generation.do_sample:
        print_main(f"  Temperature: {cfg.generation.temperature}")
        print_main(f"  Top-p: {cfg.generation.top_p}")
        print_main(f"  Top-k: {cfg.generation.top_k}")
    print_main("=" * 80 + "\n")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for decoder-only models (required for batch generation)
    # Left padding ensures all prompts "end" at the same position for correct generation
    tokenizer.padding_side = 'left'

    # For multi-GPU training with Trainer, don't use device_map="auto"
    # Let the Trainer handle device placement
    # Use BF16 if supported, otherwise FP32 (avoid FP16 due to gradient scaler issues)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
    )

    # Freeze all base model parameters - only train the probe
    print_main("Freezing base model parameters...")
    for param in base_model.parameters():
        param.requires_grad = False

    # NOTE: Gradient checkpointing is NOT enabled because the base model is frozen
    # We only train the small probe head, so gradient checkpointing would provide no benefit

    # Wrap with token count prediction head
    print_main("Adding token count prediction head...")
    model = ModelWithTokenCountHead(base_model, hidden_layer=cfg.model.probe_hidden_layer)

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_main(f"Total parameters: {total_params:,}")
    print_main(f"Trainable parameters (token count probe only): {trainable_params:,}")
    print_main(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

    # Instantiate dataset builder from config
    dataset_builder = instantiate(cfg.data)
    dataset_type = cfg.data._target_.split('.')[-1]
    print_main(f"Loading training data (builder: {dataset_type})...")
    train_dataset = dataset_builder.get_dataset(split='train', tokenizer=tokenizer)

    print_main(f"Loading evaluation data (builder: {dataset_type})...")
    eval_dataset = dataset_builder.get_dataset(split='validation', tokenizer=tokenizer)

    print_main(f"Train dataset size: {len(train_dataset)}")
    print_main(f"Eval dataset size: {len(eval_dataset)}")

    # Tokenize datasets
    print_main("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval",
    )

    # Note: Average completion length cannot be computed ahead of time since
    # completions will be generated dynamically during training
    print_main("Completions will be generated dynamically during training")

    # Training arguments - optimized for multi-GPU training
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
        # Learning rate scheduler
        "lr_scheduler_type": "constant",  # No decay, constant learning rate
        # Evaluation configuration
        "eval_strategy": cfg.eval_strategy,
        "eval_steps": cfg.eval_steps,
        # Mixed precision training for faster computation
        # Use BF16 instead of FP16 to avoid gradient scaler issues with custom compute_loss in DDP
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": False,  # Disabled - causes gradient scaler issues with generation in compute_loss
        # Multi-GPU optimization
        "ddp_find_unused_parameters": True,  # Required when base model is frozen (unused parameters in backward pass)
        "ddp_backend": "nccl" if torch.cuda.is_available() else None,  # Fastest backend for NVIDIA GPUs
        "dataloader_pin_memory": cfg.dataloader.pin_memory,
        "dataloader_num_workers": cfg.dataloader.num_workers,
        # Gradient optimization
        "gradient_checkpointing": False,  # Will enable separately on model
        "max_grad_norm": cfg.optimization.max_grad_norm,
        # Logging and checkpointing
        "report_to": "wandb" if cfg.wandb.enabled else "none",
        "run_name": cfg.wandb.run_name,
        "disable_tqdm": local_rank not in [-1, 0],  # Disable progress bars on non-main processes
        # Performance optimization
        "group_by_length": False,  # Can enable for variable length sequences
        # Reproducibility
        "seed": seed,
        "data_seed": seed,
    }

    # Only add prefetch_factor if num_workers > 0
    if cfg.dataloader.num_workers > 0 and cfg.dataloader.prefetch_factor is not None:
        training_args_dict["dataloader_prefetch_factor"] = cfg.dataloader.prefetch_factor

    # Add scheduler-specific TrainingArguments parameters (filter out null values)
    if hasattr(cfg.scheduler, 'metric_for_best_model') and cfg.scheduler.metric_for_best_model is not None:
        training_args_dict["metric_for_best_model"] = cfg.scheduler.metric_for_best_model
    if hasattr(cfg.scheduler, 'greater_is_better') and cfg.scheduler.greater_is_better is not None:
        training_args_dict["greater_is_better"] = cfg.scheduler.greater_is_better
    if hasattr(cfg.scheduler, 'load_best_model_at_end') and cfg.scheduler.load_best_model_at_end is not None:
        training_args_dict["load_best_model_at_end"] = cfg.scheduler.load_best_model_at_end

    training_args = TrainingArguments(**training_args_dict)

    # Prepare generation config
    generation_config = {
        'max_new_tokens': cfg.generation.max_new_tokens,
        'do_sample': cfg.generation.do_sample,
        'temperature': cfg.generation.temperature,
        'top_p': cfg.generation.top_p,
        'top_k': cfg.generation.top_k,
    }

    # Create callback to display example predictions
    display_callback = ExampleDisplayCallback(
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        display_every_n_steps=cfg.display_every_n_steps,
        generation_config=generation_config,
    )

    # Create callback to log MAE per bin charts
    mae_per_bin_callback = MAEPerBinCallback(log_every_n_steps=cfg.eval_steps)

    # Create infinite sampler for training dataset (with replacement)
    train_sampler = InfiniteSamplerWithReplacement(
        data_source=tokenized_train_dataset,
        shuffle=True,
        seed=seed,
    )

    # Instantiate scheduler factory from config
    scheduler_factory = instantiate(cfg.scheduler.scheduler_factory)

    # Create Custom Trainer (handles tied weights properly)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer (new API)
        compute_metrics=compute_metrics,
        callbacks=[display_callback, mae_per_bin_callback],
        bin_size=cfg.optimization.bin_size,
        generation_config=generation_config,
        train_sampler=train_sampler,  # Pass infinite sampler for training
        scheduler_factory=scheduler_factory,  # Pass scheduler factory
        loss_fn_name=cfg.optimization.loss_function,  # Pass configured loss function
    )

    # Set trainer reference in callback so it can trigger trainer methods
    mae_per_bin_callback.trainer = trainer

    # Train
    print_main("Starting training...")
    print_main(
        f"Training on {cfg.data.num_train_samples} samples, evaluating on {cfg.data.num_eval_samples} samples"
    )
    # Print token range only for synthetic data (which has min_tokens and max_tokens)
    if hasattr(cfg.data, 'min_tokens') and hasattr(cfg.data, 'max_tokens'):
        print_main(f"Token range: {cfg.data.min_tokens}-{cfg.data.max_tokens} tokens per completion")
    trainer.train()

    # Save the trained probe (only on main process)
    if is_main_process():
        print_main(f"Saving trained probe to {output_dir}")
        model.save_pretrained(f"{output_dir}/trained_probe")
        tokenizer.save_pretrained(f"{output_dir}/trained_probe")

    # Finish wandb run (only on main process)
    if cfg.wandb.enabled and is_main_process():
        wandb.finish()

    print_main("Training complete!")

    # Clean up distributed training
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
