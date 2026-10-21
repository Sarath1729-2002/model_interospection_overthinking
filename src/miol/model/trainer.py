"""
Custom Trainer for token count prediction probe training.
"""

import io
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb
from transformers import Trainer
from PIL import Image

from miol.model.generation_utils import (
    build_generation_kwargs,
    extract_completion_with_eos,
    compute_remaining_token_targets,
)
from miol.model.loss_functions import create_loss_function


def is_main_process():
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in [-1, 0]


def print_main(*args, **kwargs):
    """Print only from the main process."""
    if is_main_process():
        print(*args, **kwargs)


class CustomTrainer(Trainer):
    """
    Custom trainer that handles saving models with tied weights properly
    and uses token count predictions for evaluation metrics.
    Handles NaN values in loss/metrics (from truncated sequences) by excluding them from averages.
    """

    def __init__(self, *args, bin_size, generation_config, train_sampler=None, scheduler_factory=None, loss_fn_name="mse", **kwargs):
        super().__init__(*args, **kwargs)
        self.bin_size = bin_size
        self.generation_config = generation_config or {}
        self.train_sampler = train_sampler  # Store custom sampler
        self.scheduler_factory = scheduler_factory  # Store scheduler factory
        # Set loss_fn
        self.loss_fn = create_loss_function(loss_fn_name)
        # Dictionaries: {bucket_start: (sum_of_errors, count)}
        self.train_mae_buckets = {}
        self.eval_mae_buckets = {}
        self.train_mae_individual_buckets = {}
        self.eval_mae_individual_buckets = {}
        # MSE buckets mirror the MAE structure: {bucket_start: (sum_of_squared_errors, count)}
        self.train_mse_buckets = {}
        self.eval_mse_buckets = {}
        self.train_mse_individual_buckets = {}
        self.eval_mse_individual_buckets = {}
        self.train_mape_buckets = {}
        self.eval_mape_buckets = {}
        self.train_smape_buckets = {}
        self.eval_smape_buckets = {}
        # Track token counts: (sum_of_token_counts, num_sequences)
        self.train_token_count_sum = 0
        self.train_token_count_num = 0
        self.eval_token_count_sum = 0
        self.eval_token_count_num = 0
        # Track MAPE: (sum_of_mape_values, num_sequences)
        self.train_mape_sum = 0
        self.train_mape_num = 0
        self.eval_mape_sum = 0
        self.eval_mape_num = 0
        # Track SMAPE: (sum_of_smape_values, num_sequences)
        self.train_smape_sum = 0
        self.train_smape_num = 0
        self.eval_smape_sum = 0
        self.eval_smape_num = 0
        # Declare `train/epoch` as the canonical step metric for every other key,
        # so charts can be re-plotted against epoch in the wandb UI.
        self._setup_wandb_step_metrics()

    def _setup_wandb_step_metrics(self):
        """Tell wandb that `train/epoch` is the step metric for all other keys.

        Idempotent: redefining the same metric is a no-op in the wandb SDK.
        """
        if wandb.run is None:
            return
        wandb.define_metric("train/epoch")
        wandb.define_metric("*", step_metric="train/epoch")

    def _log_to_wandb(self, payload, commit=None):
        """Forward to `wandb.log` after injecting `train/epoch` from trainer state.

        Every custom metric logged from the trainers flows through here so that
        the run remains plottable against `train/epoch` on the wandb UI x-axis.
        """
        if wandb.run is None:
            return
        payload = dict(payload)
        epoch = getattr(self.state, "epoch", 0.0)
        payload.setdefault("train/epoch", float(epoch if epoch is not None else 0.0))
        if commit is None:
            wandb.log(payload)
        else:
            wandb.log(payload, commit=commit)

    def _update_buckets(self, predictions_np, targets_np, has_eos_np, prefix):
        """Update MAE/MAPE/SMAPE bucket statistics for the main count probe.

        Thin wrapper around _update_count_buckets that reads/writes the
        self.{prefix}_* attributes used by the main count probe.
        """
        state = {
            "mae_buckets": getattr(self, f"{prefix}_mae_buckets"),
            "mae_individual_buckets": getattr(self, f"{prefix}_mae_individual_buckets"),
            "mse_buckets": getattr(self, f"{prefix}_mse_buckets"),
            "mse_individual_buckets": getattr(self, f"{prefix}_mse_individual_buckets"),
            "mape_buckets": getattr(self, f"{prefix}_mape_buckets"),
            "smape_buckets": getattr(self, f"{prefix}_smape_buckets"),
            "token_count_sum": getattr(self, f"{prefix}_token_count_sum"),
            "token_count_num": getattr(self, f"{prefix}_token_count_num"),
            "mape_sum": getattr(self, f"{prefix}_mape_sum"),
            "mape_num": getattr(self, f"{prefix}_mape_num"),
            "smape_sum": getattr(self, f"{prefix}_smape_sum"),
            "smape_num": getattr(self, f"{prefix}_smape_num"),
        }
        self._update_count_buckets(predictions_np, targets_np, has_eos_np, state)
        # Write back scalar counters (dicts were mutated in place)
        for k in ("token_count_sum", "token_count_num", "mape_sum", "mape_num", "smape_sum", "smape_num"):
            setattr(self, f"{prefix}_{k}", state[k])

    def _update_count_buckets(self, predictions_np, targets_np, has_eos_np, state):
        """Update MAE/MAPE/SMAPE bucket stats for a count-valued regression probe.

        `state` is a dict with the bucket dicts (mutated in place) and scalar
        counters (returned via the dict — caller must read updated values).
        """
        for pred_seq, target_seq, has_eos in zip(predictions_np, targets_np, has_eos_np):
            valid_indices = target_seq != -100
            if not valid_indices.any():
                continue

            valid_preds = pred_seq[valid_indices]
            valid_targets = target_seq[valid_indices]
            completion_length = len(valid_targets)

            errors = abs(valid_preds - valid_targets)
            sq_errors = errors ** 2
            sum_errors = errors.sum()
            sum_sq_errors = sq_errors.sum()
            count = len(errors)

            if has_eos:
                state["token_count_sum"] += completion_length
                state["token_count_num"] += 1
                # MAPE: |error| / |target|
                sequence_mape = errors / np.where(valid_targets == 0, 1, valid_targets)
                state["mape_sum"] += sequence_mape.mean()
                state["mape_num"] += 1
                # SMAPE: |error| / ((|pred| + |target|) / 2)
                denom = (np.abs(valid_preds) + np.abs(valid_targets)) / 2.0
                sequence_smape = errors / np.where(denom == 0, 1, denom)
                state["smape_sum"] += sequence_smape.mean()
                state["smape_num"] += 1

            bucket_start = (completion_length // self.bin_size) * self.bin_size
            if bucket_start in state["mae_buckets"]:
                prev_sum, prev_count = state["mae_buckets"][bucket_start]
                state["mae_buckets"][bucket_start] = (prev_sum + sum_errors, prev_count + count)
                prev_sq_sum, prev_sq_count = state["mse_buckets"][bucket_start]
                state["mse_buckets"][bucket_start] = (prev_sq_sum + sum_sq_errors, prev_sq_count + count)
            else:
                state["mae_buckets"][bucket_start] = (sum_errors, count)
                state["mse_buckets"][bucket_start] = (sum_sq_errors, count)

            for error, sq_error, target, pred in zip(errors, sq_errors, valid_targets, valid_preds):
                bucket_start = (target // self.bin_size) * self.bin_size

                mape_val = (error / target) if target != 0 else error
                denom = (abs(pred) + abs(target)) / 2.0
                smape_val = (error / denom) if denom != 0 else error

                if bucket_start in state["mae_individual_buckets"]:
                    prev_sum, prev_count = state["mae_individual_buckets"][bucket_start]
                    state["mae_individual_buckets"][bucket_start] = (prev_sum + error, prev_count + 1)
                    prev_sq_sum, prev_sq_count = state["mse_individual_buckets"][bucket_start]
                    state["mse_individual_buckets"][bucket_start] = (prev_sq_sum + sq_error, prev_sq_count + 1)
                    prev_sum, prev_count = state["mape_buckets"][bucket_start]
                    state["mape_buckets"][bucket_start] = (prev_sum + mape_val, prev_count + 1)
                    prev_sum, prev_count = state["smape_buckets"][bucket_start]
                    state["smape_buckets"][bucket_start] = (prev_sum + smape_val, prev_count + 1)
                else:
                    state["mae_individual_buckets"][bucket_start] = (error, 1)
                    state["mse_individual_buckets"][bucket_start] = (sq_error, 1)
                    state["mape_buckets"][bucket_start] = (mape_val, 1)
                    state["smape_buckets"][bucket_start] = (smape_val, 1)

    def _log_regression_buckets(self, state, global_step, split, namespace_prefix, include_mape=True, include_smape=True):
        """Log MAE / MAPE / SMAPE per-bin charts from a count-probe state dict.

        Args:
            state: dict with keys 'mae_buckets', 'mae_individual_buckets', 'mape_buckets', 'smape_buckets'
            global_step: current step (used as both step and global_step for chart)
            split: 'train' or 'eval'
            namespace_prefix: wandb key prefix (e.g. 'baseline_count/')
            include_mape: log MAPE chart
            include_smape: log SMAPE chart
        """
        if state["mae_buckets"]:
            per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(state["mae_buckets"].items())}
            counts = {b: c for b, (s, c) in state["mae_buckets"].items()}
            self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mae", prefix=namespace_prefix)

        if state.get("mse_buckets"):
            per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(state["mse_buckets"].items())}
            counts = {b: c for b, (s, c) in state["mse_buckets"].items()}
            self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mse", prefix=namespace_prefix, ylabel="MSE")

        if state["mae_individual_buckets"]:
            per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(state["mae_individual_buckets"].items())}
            counts = {b: c for b, (s, c) in state["mae_individual_buckets"].items()}
            self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mae_individual", prefix=namespace_prefix)

        if state.get("mse_individual_buckets"):
            per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(state["mse_individual_buckets"].items())}
            counts = {b: c for b, (s, c) in state["mse_individual_buckets"].items()}
            self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mse_individual", prefix=namespace_prefix, ylabel="MSE")

        if include_mape and state["mape_buckets"]:
            per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(state["mape_buckets"].items())}
            counts = {b: c for b, (s, c) in state["mape_buckets"].items()}
            self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mape", prefix=namespace_prefix)

        if include_smape and state["smape_buckets"]:
            per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(state["smape_buckets"].items())}
            counts = {b: c for b, (s, c) in state["smape_buckets"].items()}
            self._log_mae_chart(per_bin, counts, global_step, global_step, split, "smape", prefix=namespace_prefix, ylabel="SMAPE")

    def _get_train_sampler(self, train_dataset):
        """
        Override to return custom infinite sampler if provided.
        """
        if self.train_sampler is not None:
            return self.train_sampler
        return super()._get_train_sampler(train_dataset)

    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Override to use custom scheduler factory if provided.
        """
        if self.scheduler_factory is not None:
            self.lr_scheduler = self.scheduler_factory(optimizer if optimizer is not None else self.optimizer)
        else:
            super().create_scheduler(num_training_steps, optimizer)

    def log(self, logs, start_time=None):
        """
        Override log to handle NaN values in metrics (from truncated sequences).
        Uses nanmean to exclude NaN values when computing averages.
        """
        import math

        # Filter out NaN values from logs
        filtered_logs = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if not math.isnan(value):
                    filtered_logs[key] = value
                # else: skip NaN values - they won't be logged
            else:
                filtered_logs[key] = value

        # Mirror HF's `epoch` into `train/epoch` so the wandb WandbCallback ships
        # the step-metric key alongside auto-logged loss/lr/eval_* values.
        if "epoch" in filtered_logs and "train/epoch" not in filtered_logs:
            filtered_logs["train/epoch"] = filtered_logs["epoch"]

        # Call parent with filtered logs
        if start_time is not None:
            super().log(filtered_logs, start_time)
        else:
            super().log(filtered_logs)

    def on_step_end(self, global_step):
        """Compute MAE per bin averages and log to wandb, then reset buckets."""
        if not is_main_process():
            return

        if wandb.run is None:
            return

        # Compute and log train MAE per bin
        if self.train_mae_buckets:
            train_mae_per_bin = {}
            train_counts_per_bin = {}
            for bucket_start, (sum_errors, count) in sorted(self.train_mae_buckets.items()):
                train_mae_per_bin[bucket_start] = sum_errors / count if count > 0 else 0
                train_counts_per_bin[bucket_start] = count

            self._log_mae_chart(train_mae_per_bin, train_counts_per_bin, global_step, global_step, "train", "mae", prefix="count/")

        if self.train_mse_buckets:
            train_mse_per_bin = {}
            train_counts_per_bin = {}
            for bucket_start, (sum_sq, count) in sorted(self.train_mse_buckets.items()):
                train_mse_per_bin[bucket_start] = sum_sq / count if count > 0 else 0
                train_counts_per_bin[bucket_start] = count

            self._log_mae_chart(train_mse_per_bin, train_counts_per_bin, global_step, global_step, "train", "mse", prefix="count/", ylabel="MSE")

        # Compute and log train MAE per bin
        if self.train_mae_individual_buckets:
            train_mae_individual_per_bin = {}
            train_counts_per_bin = {}
            for bucket_start, (sum_errors, count) in sorted(self.train_mae_individual_buckets.items()):
                train_mae_individual_per_bin[bucket_start] = sum_errors / count if count > 0 else 0
                train_counts_per_bin[bucket_start] = count

            self._log_mae_chart(train_mae_individual_per_bin, train_counts_per_bin, global_step, global_step, "train", "mae_individual", prefix="count/")

        if self.train_mse_individual_buckets:
            train_mse_individual_per_bin = {}
            train_counts_per_bin = {}
            for bucket_start, (sum_sq, count) in sorted(self.train_mse_individual_buckets.items()):
                train_mse_individual_per_bin[bucket_start] = sum_sq / count if count > 0 else 0
                train_counts_per_bin[bucket_start] = count

            self._log_mae_chart(train_mse_individual_per_bin, train_counts_per_bin, global_step, global_step, "train", "mse_individual", prefix="count/", ylabel="MSE")

        # Compute and log train MAPE per bin
        if self.train_mape_buckets:
            train_mape_per_bin = {}
            train_counts_per_bin = {}
            for bucket_start, (sum_errors, count) in sorted(self.train_mape_buckets.items()):
                train_mape_per_bin[bucket_start] = sum_errors / count if count > 0 else 0
                train_counts_per_bin[bucket_start] = count

            self._log_mae_chart(train_mape_per_bin, train_counts_per_bin, global_step, global_step, "train", "mape", prefix="count/")

        # Compute and log train SMAPE per bin
        if self.train_smape_buckets:
            train_smape_per_bin = {}
            train_counts_per_bin = {}
            for bucket_start, (sum_errors, count) in sorted(self.train_smape_buckets.items()):
                train_smape_per_bin[bucket_start] = sum_errors / count if count > 0 else 0
                train_counts_per_bin[bucket_start] = count

            self._log_mae_chart(train_smape_per_bin, train_counts_per_bin, global_step, global_step, "train", "smape", prefix="count/", ylabel="SMAPE")

        # Compute and log eval MAE per bin
        if self.eval_mae_buckets:
            eval_mae_per_bin = {}
            eval_counts_per_bin = {}
            for bucket_start, (sum_errors, count) in sorted(self.eval_mae_buckets.items()):
                eval_mae_per_bin[bucket_start] = sum_errors / count if count > 0 else 0
                eval_counts_per_bin[bucket_start] = count

            self._log_mae_chart(eval_mae_per_bin, eval_counts_per_bin, global_step, global_step, "eval", "mae", prefix="count/")

        if self.eval_mse_buckets:
            eval_mse_per_bin = {}
            eval_counts_per_bin = {}
            for bucket_start, (sum_sq, count) in sorted(self.eval_mse_buckets.items()):
                eval_mse_per_bin[bucket_start] = sum_sq / count if count > 0 else 0
                eval_counts_per_bin[bucket_start] = count

            self._log_mae_chart(eval_mse_per_bin, eval_counts_per_bin, global_step, global_step, "eval", "mse", prefix="count/", ylabel="MSE")

        # Compute and log eval MAE per bin
        if self.eval_mae_individual_buckets:
            eval_mae_individual_per_bin = {}
            eval_counts_per_bin = {}
            for bucket_start, (sum_errors, count) in sorted(self.eval_mae_individual_buckets.items()):
                eval_mae_individual_per_bin[bucket_start] = sum_errors / count if count > 0 else 0
                eval_counts_per_bin[bucket_start] = count

            self._log_mae_chart(eval_mae_individual_per_bin, eval_counts_per_bin, global_step, global_step, "eval", "mae_individual", prefix="count/")

        if self.eval_mse_individual_buckets:
            eval_mse_individual_per_bin = {}
            eval_counts_per_bin = {}
            for bucket_start, (sum_sq, count) in sorted(self.eval_mse_individual_buckets.items()):
                eval_mse_individual_per_bin[bucket_start] = sum_sq / count if count > 0 else 0
                eval_counts_per_bin[bucket_start] = count

            self._log_mae_chart(eval_mse_individual_per_bin, eval_counts_per_bin, global_step, global_step, "eval", "mse_individual", prefix="count/", ylabel="MSE")

        # Compute and log eval MAPE per bin
        if self.eval_mape_buckets:
            eval_mape_per_bin = {}
            eval_counts_per_bin = {}
            for bucket_start, (sum_errors, count) in sorted(self.eval_mape_buckets.items()):
                eval_mape_per_bin[bucket_start] = sum_errors / count if count > 0 else 0
                eval_counts_per_bin[bucket_start] = count

            self._log_mae_chart(eval_mape_per_bin, eval_counts_per_bin, global_step, global_step, "eval", "mape", prefix="count/")

        # Compute and log eval SMAPE per bin
        if self.eval_smape_buckets:
            eval_smape_per_bin = {}
            eval_counts_per_bin = {}
            for bucket_start, (sum_errors, count) in sorted(self.eval_smape_buckets.items()):
                eval_smape_per_bin[bucket_start] = sum_errors / count if count > 0 else 0
                eval_counts_per_bin[bucket_start] = count

            self._log_mae_chart(eval_smape_per_bin, eval_counts_per_bin, global_step, global_step, "eval", "smape", prefix="count/", ylabel="SMAPE")

        # Compute and log average token counts (let wandb use automatic step tracking)
        if self.train_token_count_num > 0:
            avg_train_tokens = self.train_token_count_sum / self.train_token_count_num
            self._log_to_wandb({"train/avg_generated_tokens": avg_train_tokens})

        if self.eval_token_count_num > 0:
            avg_eval_tokens = self.eval_token_count_sum / self.eval_token_count_num
            self._log_to_wandb({"eval/avg_generated_tokens": avg_eval_tokens})

        # Compute and log MAE (aggregate across all bins)
        if self.train_mae_buckets:
            total_err = sum(s for s, _ in self.train_mae_buckets.values())
            total_cnt = sum(c for _, c in self.train_mae_buckets.values())
            if total_cnt > 0:
                self._log_to_wandb({"count/train_mae": total_err / total_cnt})

        if self.eval_mae_buckets:
            total_err = sum(s for s, _ in self.eval_mae_buckets.values())
            total_cnt = sum(c for _, c in self.eval_mae_buckets.values())
            if total_cnt > 0:
                self._log_to_wandb({"count/eval_mae": total_err / total_cnt})

        # Compute and log MSE (aggregate across all bins)
        if self.train_mse_buckets:
            total_sq = sum(s for s, _ in self.train_mse_buckets.values())
            total_cnt = sum(c for _, c in self.train_mse_buckets.values())
            if total_cnt > 0:
                self._log_to_wandb({"count/train_mse": total_sq / total_cnt})

        if self.eval_mse_buckets:
            total_sq = sum(s for s, _ in self.eval_mse_buckets.values())
            total_cnt = sum(c for _, c in self.eval_mse_buckets.values())
            if total_cnt > 0:
                self._log_to_wandb({"count/eval_mse": total_sq / total_cnt})

        # Compute and log MAPE
        if self.train_mape_num > 0:
            avg_train_mape = self.train_mape_sum / self.train_mape_num
            self._log_to_wandb({"count/train_mape": avg_train_mape})

        if self.eval_mape_num > 0:
            avg_eval_mape = self.eval_mape_sum / self.eval_mape_num
            self._log_to_wandb({"count/eval_mape": avg_eval_mape})

        # Compute and log SMAPE
        if self.train_smape_num > 0:
            avg_train_smape = self.train_smape_sum / self.train_smape_num
            self._log_to_wandb({"count/train_smape": avg_train_smape})

        if self.eval_smape_num > 0:
            avg_eval_smape = self.eval_smape_sum / self.eval_smape_num
            self._log_to_wandb({"count/eval_smape": avg_eval_smape})

        # Reset buckets and token counts after logging
        self.train_mae_buckets = {}
        self.train_mae_individual_buckets = {}
        self.eval_mae_buckets = {}
        self.eval_mae_individual_buckets = {}
        self.train_mse_buckets = {}
        self.train_mse_individual_buckets = {}
        self.eval_mse_buckets = {}
        self.eval_mse_individual_buckets = {}
        self.train_mape_buckets = {}
        self.eval_mape_buckets = {}
        self.train_smape_buckets = {}
        self.eval_smape_buckets = {}
        self.train_token_count_sum = 0
        self.train_token_count_num = 0
        self.eval_token_count_sum = 0
        self.eval_token_count_num = 0
        self.train_mape_sum = 0
        self.train_mape_num = 0
        self.eval_mape_sum = 0
        self.eval_mape_num = 0
        self.train_smape_sum = 0
        self.train_smape_num = 0
        self.eval_smape_sum = 0
        self.eval_smape_num = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to:
        1. Generate completions from prompts (inputs only contain tokenized prompts)
        2. Extract hidden states from generation and compute probe predictions
        3. Compute loss directly from precomputed predictions
        4. Collect training predictions for bucket statistics
        """
        # Inputs only contain tokenized prompts - always generate completions and compute targets
        assert "remaining_token_counts" not in inputs, (
            f"[compute_loss] Expected inputs to only contain prompts, not completions. "
            f"Input keys: {list(inputs.keys())}"
        )
        inputs = self._generate_and_compute_targets(model, inputs)

        # Log EOS statistics to wandb
        if is_main_process() and wandb.run is not None and self.state.global_step > 0:
            num_with_eos = inputs["_num_sequences_with_eos"]
            num_total = inputs["_num_sequences_total"]

            # Determine if we're in training or evaluation
            is_training = model.training
            prefix = "train" if is_training else "eval"

            self._log_to_wandb({
                f"{prefix}/sequences_with_eos": num_with_eos,
                f"{prefix}/sequences_total": num_total,
                f"{prefix}/sequences_with_eos_fraction": (num_with_eos / num_total),
            })

        # Get outputs to capture predictions (only on main process)
        should_collect = is_main_process() and self.state.global_step > 0

        # If predictions are already precomputed from generation, compute loss directly
        if inputs.get("predictions_precomputed", False):
            # Compute loss from precomputed predictions
            predictions = inputs["token_count_predictions"]
            targets = inputs["remaining_token_counts"]

            # Calculate MSE loss for token count prediction
            remaining_counts_flat = targets.view(-1)
            predictions_flat = predictions.view(-1)

            # Find positions that are not masked (-100)
            active_positions = remaining_counts_flat != -100

            # Filter to only active positions
            active_predictions = predictions_flat[active_positions]
            active_labels = remaining_counts_flat[active_positions].float()

            # Calculate loss using configured loss function
            if len(active_predictions) > 0:
                loss = self.loss_fn(active_predictions, active_labels)
            else:
                # Every sequence in the batch was truncated (no EOS), so skip the update by
                # returning a zero loss that still participates in autograd.
                loss = predictions.sum() * 0.0

            # Create a simple output object for compatibility
            from miol.model.model import DualHeadModelOutput
            outputs = DualHeadModelOutput(
                loss=loss,
                logits=None,
                token_count_predictions=predictions,
            )
        else:
            # Standard path: call model forward
            if should_collect or return_outputs:
                loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
                outputs = None

        # Collect predictions and update buckets (only during training)
        if should_collect and outputs is not None:
            preds_np = outputs.token_count_predictions.detach().cpu().numpy()
            targets_np = inputs["remaining_token_counts"].cpu().numpy()
            has_eos_np = inputs.get("_sequence_has_eos", torch.ones(len(targets_np), dtype=torch.bool)).cpu().numpy()
            self._update_buckets(preds_np, targets_np, has_eos_np, "train")

        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to:
        1. Generate completions and compute targets (inputs only contain tokenized prompts)
        2. Return token_count_predictions for metrics computation
        3. Collect eval predictions for bucket statistics
        """
        inputs = self._prepare_inputs(inputs)

        # Inputs only contain tokenized prompts - always generate completions and compute targets
        assert "remaining_token_counts" not in inputs, (
            f"[prediction_step] Expected inputs to only contain prompts, not completions. "
            f"Input keys: {list(inputs.keys())}"
        )
        inputs = self._generate_and_compute_targets(model, inputs)

        # Log EOS statistics to wandb (for evaluation)
        if is_main_process() and wandb.run is not None:
            num_with_eos = inputs.get("_num_sequences_with_eos", 0)
            num_total = inputs.get("_num_sequences_total", 0)

            if num_total > 0:
                pct_with_eos = (num_with_eos / num_total) * 100

                self._log_to_wandb({
                    "eval/sequences_with_eos": num_with_eos,
                    "eval/sequences_total": num_total,
                    "eval/sequences_truncated": num_total - num_with_eos,
                })

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            # Use token_count_predictions instead of logits
            predictions = outputs.token_count_predictions

        # Collect predictions and update eval buckets (only on main process)
        if is_main_process() and self.state.global_step > 0:
            preds_np = predictions.cpu().numpy()
            targets_np = inputs.get("remaining_token_counts").cpu().numpy()
            has_eos_np = inputs.get("_sequence_has_eos", torch.ones(len(targets_np), dtype=torch.bool)).cpu().numpy()
            self._update_buckets(preds_np, targets_np, has_eos_np, "eval")

        if prediction_loss_only:
            return (loss, None, None)

        # Return predictions and labels for metrics computation
        labels = inputs.get("remaining_token_counts")

        return (loss, predictions, labels)

    def _generate_and_compute_targets(self, model, inputs):
        """
        Generate completions from prompts, extract hidden states, and apply the
        count probe head to obtain per-token predictions.

        Args:
            model: The model to use for generation (may be wrapped in DDP)
            inputs: Dict with 'input_ids' (prompts only) and 'attention_mask'

        Returns:
            Dict matching the offline collator format plus 'token_count_predictions'
            and 'predictions_precomputed=True'.
        """
        # Unwrap model if it's wrapped in DistributedDataParallel
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Generate + extract hidden states (no gradients to base model).
        from miol.model.generation_utils import generate_and_extract_hidden_states
        batch = generate_and_extract_hidden_states(
            base_model=unwrapped_model.base_model,
            hidden_layer=unwrapped_model.hidden_layer,
            prompt_input_ids=inputs["input_ids"],
            prompt_attention_mask=inputs["attention_mask"],
            generation_config=self.generation_config,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            bos_token_id=self.processing_class.bos_token_id,
        )

        # Apply the count probe head to the hidden states (with gradients).
        hidden_states_fp32 = batch["hidden_states"].float()
        token_count_logits = unwrapped_model.token_count_head(hidden_states_fp32)
        token_count_predictions = token_count_logits.squeeze(-1)

        inputs["input_ids"] = batch["input_ids"]
        inputs["attention_mask"] = batch["attention_mask"]
        inputs["remaining_token_counts"] = batch["remaining_token_counts"]
        inputs["token_count_predictions"] = token_count_predictions
        inputs["predictions_precomputed"] = True
        inputs["_num_sequences_with_eos"] = int(batch["has_eos"].sum().item())
        inputs["_num_sequences_total"] = int(batch["has_eos"].shape[0])
        inputs["_sequence_has_eos"] = batch["has_eos"]
        return inputs

    def _save(self, output_dir=None, state_dict=None):
        """Override save to use our custom save method."""
        # Only save on main process in distributed training
        if not is_main_process():
            return

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save using our custom model's save_pretrained method
        # Only save the token count head - base model is frozen
        print_main(f"Saving checkpoint to {output_dir}")

        # Save only the token count head (probe)
        if hasattr(self.model, "token_count_head"):
            token_count_head_path = f"{output_dir}/token_count_head.pt"
            torch.save(
                {
                    "token_count_head": self.model.token_count_head.state_dict(),
                },
                token_count_head_path,
            )
            print_main(f"Token count probe saved to {token_count_head_path}")

    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to use custom save."""
        # Only save on main process in distributed training
        if not is_main_process():
            return

        if output_dir is None:
            output_dir = self.args.output_dir

        self._save(output_dir)

        # Save processing_class (tokenizer)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)


    def _log_mae_chart(self, mae_per_bin, counts_per_bin, step, global_step, split_name, metric="mae", prefix="", ylabel=None):
        """Create and log matplotlib line chart with sample counts to wandb."""
        if not mae_per_bin:
            return

        if ylabel is None:
            ylabel = "Mean Absolute Error"

        # Extract data for plotting
        bins = sorted(mae_per_bin.keys())
        mae_values = [mae_per_bin[b] for b in bins]
        count_values = [counts_per_bin[b] for b in bins]

        # Create matplotlib figure with wider aspect ratio
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Plot line with markers on left y-axis
        color = 'steelblue' if split_name == 'train' else 'coral'
        ax1.plot(bins, mae_values, marker='o', linewidth=2, markersize=6, color=color, label=f'{split_name.capitalize()} {metric.upper()}')
        ax1.set_xlabel('Token Length Bin Start', fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create secondary y-axis for sample counts
        ax2 = ax1.twinx()
        bar_color = 'lightsteelblue' if split_name == 'train' else 'lightcoral'
        ax2.bar(bins, count_values, alpha=0.3, color=bar_color, label='Sample Count', width=2)
        ax2.set_ylabel('Number of Samples', fontsize=12, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Set title
        ax1.set_title(f'{metric.upper()} per Token Length Bin ({split_name.capitalize()}) - Step {step}', fontsize=14)

        # Ensure all bins are shown on x-axis
        ax1.set_xticks(bins)
        ax1.set_xticklabels([str(b) for b in bins], rotation=45, ha='right')

        # Add grid for better readability
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()

        # Convert to image for wandb
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image = Image.open(buf)

        # Log to wandb (let wandb use automatic step tracking)
        self._log_to_wandb({
            f"{prefix}{metric}_per_bin/{split_name}": wandb.Image(image, caption=f"Step {step}"),
        })

        plt.close(fig)
        buf.close()
