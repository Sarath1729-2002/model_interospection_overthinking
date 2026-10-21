"""
Offline Probe Trainers - train probes using pre-extracted hidden states.

These trainers skip generation and load pre-computed hidden states from disk,
making training much faster.

OfflineProbeTrainer: Single regression probe (original).
MultiProbeOfflineTrainer: Multiple probes (regression + classification) with
    separate backward passes per probe head.
"""

import io

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import wandb
from PIL import Image

from miol.model.trainer import (
    CustomTrainer,
    is_main_process,
)


class OfflineProbeTrainer(CustomTrainer):
    """
    Trainer for offline probe training using pre-extracted hidden states.

    Key differences from CustomTrainer:
    - No generation during training (hidden states pre-computed)
    - Hidden states loaded from dataset
    - Much faster training
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to use pre-loaded hidden states instead of generating.

        Inputs already contain:
        - hidden_states: [batch_size, seq_len, hidden_size]
        - remaining_token_counts: [batch_size, seq_len]
        - input_ids: [batch_size, seq_len]
        - attention_mask: [batch_size, seq_len]
        """
        # Extract hidden states and targets from inputs
        hidden_states = inputs["hidden_states"]  # [batch_size, seq_len, hidden_size]
        targets = inputs["remaining_token_counts"]  # [batch_size, seq_len]

        # Unwrap model if wrapped in DDP
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Apply probe head to hidden states
        # hidden_states shape: [batch_size, seq_len, hidden_size]
        hidden_states_fp32 = hidden_states.float()
        token_count_logits = unwrapped_model.token_count_head(hidden_states_fp32)  # [batch_size, seq_len, 1]
        predictions = token_count_logits.squeeze(-1)  # [batch_size, seq_len]

        # Calculate MSE loss for token count prediction
        remaining_counts_flat = targets.view(-1)
        predictions_flat = predictions.view(-1)

        # Find positions that are not masked (-100)
        active_positions = remaining_counts_flat != -100

        # Filter to only active positions
        active_predictions = predictions_flat[active_positions]
        active_labels = remaining_counts_flat[active_positions].float()

        # Calculate loss using configured loss function for token count prediction
        if len(active_predictions) > 0:
            loss = self.loss_fn(active_predictions, active_labels)
        else:
            # Skip optimizer update when no EOS tokens are present by returning
            # a zero-valued loss still connected to the probe head.
            loss = predictions.sum() * 0.0

        # Collect predictions and update buckets (only during training, only on main process)
        if is_main_process() and self.state.global_step > 0:
            has_eos_list = inputs.get("has_eos", torch.ones(len(targets), dtype=torch.bool))
            preds_np = predictions.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()
            has_eos_np = has_eos_list.cpu().numpy() if isinstance(has_eos_list, torch.Tensor) else has_eos_list
            self._update_buckets(preds_np, targets_np, has_eos_np, "train")

        # Create output for compatibility
        from miol.model.model import DualHeadModelOutput
        outputs = DualHeadModelOutput(
            loss=loss,
            logits=None,
            token_count_predictions=predictions,
        )

        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to use pre-loaded hidden states.
        """
        inputs = self._prepare_inputs(inputs)

        # Extract hidden states and targets from inputs
        hidden_states = inputs["hidden_states"]  # [batch_size, seq_len, hidden_size]
        targets = inputs["remaining_token_counts"]  # [batch_size, seq_len]

        # Unwrap model if wrapped in DDP
        unwrapped_model = model.module if hasattr(model, 'module') else model

        with torch.no_grad():
            # Apply probe head to hidden states
            hidden_states_fp32 = hidden_states.float()
            token_count_logits = unwrapped_model.token_count_head(hidden_states_fp32)
            predictions = token_count_logits.squeeze(-1)

            # Calculate loss
            remaining_counts_flat = targets.view(-1)
            predictions_flat = predictions.view(-1)
            active_positions = remaining_counts_flat != -100
            active_predictions = predictions_flat[active_positions]
            active_labels = remaining_counts_flat[active_positions].float()

            if len(active_predictions) > 0:
                loss = self.loss_fn(active_predictions, active_labels)
            else:
                loss = predictions.sum() * 0.0

        # Collect predictions and update eval buckets (only on main process)
        if is_main_process() and self.state.global_step > 0:
            has_eos_list = inputs.get("has_eos", torch.ones(len(targets), dtype=torch.bool))
            preds_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()
            has_eos_np = has_eos_list.cpu().numpy() if isinstance(has_eos_list, torch.Tensor) else has_eos_list
            self._update_buckets(preds_np, targets_np, has_eos_np, "eval")

        if prediction_loss_only:
            return (loss, None, None)

        # Return predictions and labels for metrics computation
        labels = targets

        return (loss, predictions, labels)


class MultiProbeOfflineTrainer(CustomTrainer):
    """
    Trainer for multi-probe offline training using pre-extracted hidden states.

    Trains multiple independent probe heads simultaneously with separate
    backward passes per head. Each probe has its own optimizer and loss function.

    Probe types:
    - regression: MSE loss on scalar targets (count, percentage)
    - classification: KL divergence on soft probability targets (cls_N)
    """

    def __init__(
        self,
        *args,
        probe_specs,
        probe_optimizers,
        streaming: bool = False,
        streaming_base_model=None,
        streaming_tokenizer=None,
        streaming_hidden_layer=-1,
        canonical_count_name: str = "count",
        canonical_percentage_name: str | None = "percentage",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.probe_specs = probe_specs
        self.probe_optimizers = probe_optimizers
        # Canonical probe names used for the trainer's hardcoded primary-loss
        # / per-bin-chart hooks. In single-layer mode these are simply "count"
        # and "percentage"; in multi-layer mode they are the layer-namespaced
        # variants (e.g. "count_L0", "percentage_L0").
        self._canonical_count_name = canonical_count_name
        self._canonical_percentage_name = canonical_percentage_name
        # Streaming mode: generate completions and extract hidden states inside
        # training_step / prediction_step instead of reading them from a
        # HiddenStatesDataset on disk. The base model is required and stays
        # frozen on the GPU.
        self.streaming = streaming
        self._streaming_base_model = streaming_base_model
        self._streaming_tokenizer = streaming_tokenizer
        self._streaming_hidden_layer = streaming_hidden_layer
        if self.streaming:
            if streaming_base_model is None or streaming_tokenizer is None:
                raise ValueError("streaming=True requires streaming_base_model and streaming_tokenizer")
            for p in streaming_base_model.parameters():
                p.requires_grad = False
            streaming_base_model.eval()
        # Per-probe loss accumulators for logging
        self._probe_loss_sums = {name: 0.0 for name in probe_specs}
        self._probe_loss_counts = {name: 0 for name in probe_specs}
        # Per-bin classification metrics: lists of per-batch rates for mean+std
        self._cls_metrics = self._make_empty_cls_metrics()
        # Percentage probe MAE buckets: {bucket_start: (sum_errors, count)}
        # Keyed by completion length and by remaining token count
        self._pct_buckets = self._make_empty_pct_buckets()
        # Classification probe accuracy buckets: {bucket_start: (sum_correct, total)}
        # Keyed by completion length and by remaining token count
        self._cls_acc_buckets = self._make_empty_cls_acc_buckets()
        # Classification probe derived-percentage MAE buckets: {bucket_start: (sum_errors, count)}
        # Derived percentage = sum(softmax(logits) * anchors), compared to true percentage target
        self._cls_derived_pct_buckets = self._make_empty_cls_derived_pct_buckets()
        # Per-probe confusion matrices for Cohen's kappa (chance-corrected accuracy).
        # Shape: {probe_name: {split: ndarray[num_bins, num_bins]}}, indexed [true, pred].
        self._cls_confusion = self._make_empty_cls_confusion()
        # Precompute anchors per classification probe (for E[position] = sum(p_i * a_i))
        self._cls_anchors = {
            name: torch.linspace(0, 1, spec["output_dim"])
            for name, spec in probe_specs.items()
            if spec["type"] == "classification"
        }
        # Baseline regression probes — per-bin bucket tracking for floor comparisons.
        # (The main count/percentage probes already have per-bin tracking via the parent.)
        self._baseline_names = [
            name for name, spec in probe_specs.items()
            if spec["type"] == "regression" and spec.get("input_type", "hidden_states") != "hidden_states"
        ]
        self._count_baseline_names = [
            name for name in self._baseline_names
            if probe_specs[name]["target_key"] == "remaining_token_counts"
        ]
        self._pct_baseline_names = [
            name for name in self._baseline_names
            if probe_specs[name]["target_key"] == "remaining_percentage"
        ]
        # Prompt-only probes: trained at prompt-end position, evaluated everywhere.
        self._prompt_only_probe_names = [
            name for name, spec in probe_specs.items()
            if spec.get("train_position") == "prompt_end"
        ]
        # Per-probe count and percentage bucket tracking. In single-layer mode
        # this list contains just `count` / `percentage`; in multi-layer mode
        # it expands to `count_L0, count_L8, ...` so each layer gets its own
        # MAE/MAPE/SMAPE charts and scalars under its own WandB namespace.
        self._count_probe_names = [
            name for name, spec in probe_specs.items()
            if spec["type"] == "regression"
            and spec.get("input_type", "hidden_states") == "hidden_states"
            and spec.get("target_key") == "remaining_token_counts"
            and spec.get("train_position") != "prompt_end"
        ]
        self._percentage_probe_names = [
            name for name, spec in probe_specs.items()
            if spec["type"] == "regression"
            and spec.get("input_type", "hidden_states") == "hidden_states"
            and spec.get("target_key") == "remaining_percentage"
            and spec.get("train_position") != "prompt_end"
        ]
        # log_count probes: train on log(1+count), but report metrics in count
        # space (expm1 the predictions before bucketing) so they're directly
        # comparable to the `count` probe.
        self._log_count_probe_names = [
            name for name, spec in probe_specs.items()
            if spec["type"] == "regression"
            and spec.get("input_type", "hidden_states") == "hidden_states"
            and spec.get("target_key") == "remaining_log_count"
            and spec.get("train_position") != "prompt_end"
        ]
        self._per_probe_count_buckets = {
            name: {split: self._make_empty_count_bucket_state() for split in ("train", "eval")}
            for name in self._count_probe_names
        }
        # Count-space metric buckets for log_count probes (predictions are
        # expm1'd before going in here).
        self._per_probe_log_count_buckets = {
            name: {split: self._make_empty_count_bucket_state() for split in ("train", "eval")}
            for name in self._log_count_probe_names
        }
        self._per_probe_pct_buckets = {
            name: {
                split: {
                    "by_completion_length": {},
                    "by_remaining_count": {},
                    "by_completion_length_sq": {},
                    "by_remaining_count_sq": {},
                }
                for split in ("train", "eval")
            }
            for name in self._percentage_probe_names
        }
        # Per-prompt-only-probe metric accumulators. "prompt" bucket = prompt-end
        # position only; "completion" bucket = every completion position with
        # implied-remaining derived from the per-position prediction.
        self._prompt_only_metrics = self._make_empty_prompt_only_metrics()
        # Count-based baselines: MAE/MAPE/SMAPE buckets (mirroring the count probe structure)
        self._baseline_count_buckets = self._make_empty_baseline_count_buckets()
        # Percentage-based baselines: MAE-only buckets (percentages near 0 make MAPE meaningless)
        self._baseline_pct_buckets = self._make_empty_baseline_pct_buckets()

    def _make_empty_cls_metrics(self):
        """Create empty classification metrics accumulators."""
        metrics = {}
        for name, spec in self.probe_specs.items():
            if spec["type"] != "classification":
                continue
            num_bins = spec["output_dim"]
            metrics[name] = {
                prefix: {
                    # Per-bin lists of per-batch rates
                    "accuracy": [[] for _ in range(num_bins)],
                    "fpr": [[] for _ in range(num_bins)],  # false positive rate
                    "fnr": [[] for _ in range(num_bins)],  # false negative rate
                }
                for prefix in ("train", "eval")
            }
        return metrics

    def _make_empty_pct_buckets(self):
        """Create empty percentage probe MAE / MSE bucket accumulators."""
        return {
            prefix: {
                "by_completion_length": {},  # {bucket_start: (sum_errors, count)} (MAE)
                "by_remaining_count": {},    # {bucket_start: (sum_errors, count)} (MAE)
                "by_completion_length_sq": {},  # {bucket_start: (sum_sq_errors, count)} (MSE)
                "by_remaining_count_sq": {},    # {bucket_start: (sum_sq_errors, count)} (MSE)
            }
            for prefix in ("train", "eval")
        }

    def _make_empty_prompt_only_metrics(self):
        """Create empty per-prompt-only-probe bucket states.

        Layout: {probe_name: {split: {bucket: count_bucket_state}}}
        - split:  "train" or "eval"
        - bucket: "prompt"     (prompt-end position only — one prediction per
                                example, target = total completion length T)
                  "completion" (every completion position — prediction $\\hat{T}_k$
                                converted to implied remaining vs. count target r_k,
                                so it's directly comparable to the count probe)

        Each leaf is a `_make_empty_count_bucket_state` dict so we can reuse
        `_log_regression_buckets` for chart logging.
        """
        return {
            name: {
                split: {
                    "prompt": self._make_empty_count_bucket_state(),
                    "completion": self._make_empty_count_bucket_state(),
                    # Static decay: prompt-end \hat{T}_0 broadcast across every
                    # completion position with deterministic decrement
                    # max(\hat{T}_0 - k - 1, 0). The "no residual-stream signal
                    # during generation" floor — distinct from the per-position
                    # "completion" bucket which re-evaluates the probe at each h_k.
                    "static_decay": self._make_empty_count_bucket_state(),
                }
                for split in ("train", "eval")
            }
            for name in self._prompt_only_probe_names
        }

    def _update_prompt_only_metrics(
        self,
        preds_np,
        prompt_end_targets_np,
        count_targets_np,
        has_eos_np,
        prompt_lengths_np,
        probe_name,
        split,
        update_completion=True,
        epsilon: float = 1e-8,
    ):
        """Accumulate MAE/MAPE/SMAPE bucket state for a prompt-only probe.

        Args:
            preds_np: [B, seq_len] per-position scalar predictions of T.
            prompt_end_targets_np: [B, seq_len] target T placed at index
                (prompt_length - 1), -100 elsewhere. Drives the "prompt" bucket.
            count_targets_np: [B, seq_len] remaining-count targets r_k (-100 on
                masked positions). Drives the "completion" bucket.
            has_eos_np: [B] booleans (truncated sequences are skipped).
            prompt_lengths_np: [B] prompt lengths.
            probe_name: which prompt-only probe to update.
            split: "train" or "eval".
            update_completion: if False, only the "prompt" bucket is updated
                (used during training, where loss / metrics are prompt-end only).
        """
        eps = float(epsilon)
        state_prompt = self._prompt_only_metrics[probe_name][split]["prompt"]

        # ---- Prompt bucket: one prediction per example, target = T ----
        for row_idx in range(preds_np.shape[0]):
            if not bool(has_eos_np[row_idx]):
                continue
            prompt_end_idx = int(prompt_lengths_np[row_idx]) - 1
            if prompt_end_idx < 0:
                continue
            T_target = prompt_end_targets_np[row_idx, prompt_end_idx]
            if T_target == -100:
                continue

            T = float(T_target)
            pred = float(preds_np[row_idx, prompt_end_idx])
            err = abs(pred - T)
            mape = err / max(abs(T), eps)
            smape = err / max((abs(pred) + abs(T)) / 2.0, eps)

            # Bin by total completion length T (== sequence-level bin on the
            # count-probe convention since this example has T completion tokens).
            bucket = (int(T) // self.bin_size) * self.bin_size

            for key, val in (("mae_buckets", err), ("mape_buckets", mape), ("smape_buckets", smape)):
                if bucket in state_prompt[key]:
                    ps, pc = state_prompt[key][bucket]
                    state_prompt[key][bucket] = (ps + val, pc + 1)
                else:
                    state_prompt[key][bucket] = (val, 1)

            state_prompt["mape_sum"] += mape; state_prompt["mape_num"] += 1
            state_prompt["smape_sum"] += smape; state_prompt["smape_num"] += 1
            state_prompt["token_count_sum"] += 1; state_prompt["token_count_num"] += 1

        # Shared per-position offset: k = position - prompt_length (clamped at 0).
        seq_idx = np.arange(preds_np.shape[1])[None, :]                       # [1, S]
        completion_offset = (seq_idx - prompt_lengths_np[:, None]).clip(min=0)  # [B, S]

        # ---- Static-decay bucket (always updated): \hat{T}_0 broadcast across
        # all completion positions, then deterministically decremented. Tests
        # the "no residual-stream signal during generation" floor.
        prompt_end_preds = np.zeros(preds_np.shape[0], dtype=preds_np.dtype)
        for row_idx in range(preds_np.shape[0]):
            pe_idx = int(prompt_lengths_np[row_idx]) - 1
            if pe_idx >= 0:
                prompt_end_preds[row_idx] = preds_np[row_idx, pe_idx]
        static_decay = np.maximum(prompt_end_preds[:, None] - completion_offset - 1.0, 0.0)
        self._update_count_buckets(
            static_decay, count_targets_np, has_eos_np,
            self._prompt_only_metrics[probe_name][split]["static_decay"],
        )

        # ---- Completion bucket: every completion position with a valid r_k ----
        if not update_completion:
            return

        # Convert per-position T-prediction $\hat{T}_k$ into implied remaining
        # max($\hat{T}_k$ - k - 1, 0). The count target r_k = T - k - 1 is in
        # `count_targets_np`; `_update_count_buckets` filters masked positions.
        implied_remaining = np.maximum(preds_np - completion_offset - 1.0, 0.0)
        self._update_count_buckets(
            implied_remaining, count_targets_np, has_eos_np,
            self._prompt_only_metrics[probe_name][split]["completion"],
        )

    def _update_pct_buckets(self, pct_preds_np, pct_targets_np, count_targets_np, has_eos_np, prefix):
        """Accumulate percentage probe MAE, binned by completion length and remaining token count.

        Args:
            pct_preds_np: [batch, seq_len] percentage predictions
            pct_targets_np: [batch, seq_len] percentage targets (-100 = masked)
            count_targets_np: [batch, seq_len] remaining count targets (for x-axis binning)
            has_eos_np: [batch] booleans
            prefix: 'train' or 'eval'
        """
        by_compl = self._pct_buckets[prefix]["by_completion_length"]
        by_count = self._pct_buckets[prefix]["by_remaining_count"]
        by_compl_sq = self._pct_buckets[prefix]["by_completion_length_sq"]
        by_count_sq = self._pct_buckets[prefix]["by_remaining_count_sq"]

        for pct_pred, pct_tgt, count_tgt, has_eos in zip(
            pct_preds_np, pct_targets_np, count_targets_np, has_eos_np
        ):
            valid = pct_tgt != -100
            if not valid.any():
                continue

            preds = pct_pred[valid]
            targets = pct_tgt[valid]
            counts = count_tgt[valid]
            errors = np.abs(preds - targets)
            sq_errors = errors ** 2

            # By completion length (one entry per sequence)
            completion_length = len(targets)
            bucket = (completion_length // self.bin_size) * self.bin_size
            sum_err = errors.sum()
            sum_sq = sq_errors.sum()
            n = len(errors)
            if bucket in by_compl:
                prev_sum, prev_count = by_compl[bucket]
                by_compl[bucket] = (prev_sum + sum_err, prev_count + n)
                prev_sq, prev_sq_count = by_compl_sq[bucket]
                by_compl_sq[bucket] = (prev_sq + sum_sq, prev_sq_count + n)
            else:
                by_compl[bucket] = (sum_err, n)
                by_compl_sq[bucket] = (sum_sq, n)

            # By remaining token count (one entry per token position)
            for error, sq_error, count_val in zip(errors, sq_errors, counts):
                bucket = (int(count_val) // self.bin_size) * self.bin_size
                if bucket in by_count:
                    prev_sum, prev_count = by_count[bucket]
                    by_count[bucket] = (prev_sum + error, prev_count + 1)
                    prev_sq, prev_sq_count = by_count_sq[bucket]
                    by_count_sq[bucket] = (prev_sq + sq_error, prev_sq_count + 1)
                else:
                    by_count[bucket] = (error, 1)
                    by_count_sq[bucket] = (sq_error, 1)

    def _make_empty_cls_acc_buckets(self):
        """Create empty classification accuracy bucket accumulators."""
        buckets = {}
        for name, spec in self.probe_specs.items():
            if spec["type"] != "classification":
                continue
            buckets[name] = {
                prefix: {
                    "by_completion_length": {},  # {bucket_start: (sum_correct, total)}
                    "by_remaining_count": {},    # {bucket_start: (sum_correct, total)}
                }
                for prefix in ("train", "eval")
            }
        return buckets

    def _update_cls_acc_buckets(self, logits, targets, count_targets_np, has_eos_np, probe_name, prefix):
        """Accumulate classification accuracy, binned by completion length and remaining token count.

        Args:
            logits: [batch, seq_len, num_bins] raw logits (tensor)
            targets: [batch, seq_len, num_bins] soft targets (tensor, -100 for masked)
            count_targets_np: [batch, seq_len] remaining count targets (numpy, for x-axis)
            has_eos_np: [batch] booleans (numpy)
            probe_name: name of the classification probe
            prefix: 'train' or 'eval'
        """
        mask = targets[..., 0] != -100  # [B, seq_len]
        pred_classes = logits.argmax(dim=-1).cpu().numpy()   # [B, seq_len]
        true_classes = targets.argmax(dim=-1).cpu().numpy()  # [B, seq_len]
        mask_np = mask.cpu().numpy()

        by_compl = self._cls_acc_buckets[probe_name][prefix]["by_completion_length"]
        by_count = self._cls_acc_buckets[probe_name][prefix]["by_remaining_count"]

        for pred_seq, true_seq, mask_seq, count_seq, has_eos in zip(
            pred_classes, true_classes, mask_np, count_targets_np, has_eos_np
        ):
            valid = mask_seq
            if not valid.any():
                continue

            preds = pred_seq[valid]
            trues = true_seq[valid]
            counts = count_seq[valid]
            correct = (preds == trues).astype(np.float64)

            # By completion length (one entry per sequence)
            completion_length = len(preds)
            bucket = (completion_length // self.bin_size) * self.bin_size
            n_correct = correct.sum()
            n_total = len(correct)
            if bucket in by_compl:
                prev_correct, prev_total = by_compl[bucket]
                by_compl[bucket] = (prev_correct + n_correct, prev_total + n_total)
            else:
                by_compl[bucket] = (n_correct, n_total)

            # By remaining token count (one entry per token position)
            for is_correct, count_val in zip(correct, counts):
                bucket = (int(count_val) // self.bin_size) * self.bin_size
                if bucket in by_count:
                    prev_correct, prev_total = by_count[bucket]
                    by_count[bucket] = (prev_correct + is_correct, prev_total + 1)
                else:
                    by_count[bucket] = (is_correct, 1)

    def _make_empty_cls_confusion(self):
        """Create empty per-probe / per-split confusion matrices."""
        matrices = {}
        for name, spec in self.probe_specs.items():
            if spec["type"] != "classification":
                continue
            n = int(spec["output_dim"])
            matrices[name] = {prefix: np.zeros((n, n), dtype=np.int64) for prefix in ("train", "eval")}
        return matrices

    def _update_cls_confusion(self, logits, targets, probe_name, prefix):
        """Accumulate confusion-matrix counts for a classification probe.
        Used to compute Cohen's kappa (chance-corrected accuracy) at log time.

        Args:
            logits: [B, seq_len, num_bins] raw logits (tensor)
            targets: [B, seq_len, num_bins] soft targets (tensor, -100 for masked)
            probe_name: name of the classification probe
            prefix: 'train' or 'eval'
        """
        mask = targets[..., 0] != -100
        if not mask.any():
            return
        pred_classes = logits[mask].argmax(dim=-1).cpu().numpy()
        true_classes = targets[mask].argmax(dim=-1).cpu().numpy()
        cm = self._cls_confusion[probe_name][prefix]
        # Vectorized increment: cm[true, pred] += 1
        np.add.at(cm, (true_classes, pred_classes), 1)

    @staticmethod
    def _compute_cohen_kappa(cm: np.ndarray) -> float:
        """Cohen's kappa for an [N, N] confusion matrix indexed [true, pred].

        kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and
        p_e is the agreement expected by chance (from marginals). Returns 0.0
        on an empty matrix; clamps to 1.0 if p_e == 1 (degenerate single-class).
        """
        n = float(cm.sum())
        if n == 0:
            return 0.0
        p_o = float(np.trace(cm)) / n
        row = cm.sum(axis=1) / n
        col = cm.sum(axis=0) / n
        p_e = float((row * col).sum())
        if p_e >= 1.0:
            return 1.0
        return (p_o - p_e) / (1.0 - p_e)

    def _make_empty_cls_derived_pct_buckets(self):
        """Create empty classification derived-percentage MAE / MSE bucket accumulators."""
        buckets = {}
        for name, spec in self.probe_specs.items():
            if spec["type"] != "classification":
                continue
            buckets[name] = {
                prefix: {
                    "by_completion_length": {},  # {bucket_start: (sum_errors, count)} (MAE)
                    "by_remaining_count": {},    # {bucket_start: (sum_errors, count)} (MAE)
                    "by_completion_length_sq": {},  # {bucket_start: (sum_sq_errors, count)} (MSE)
                    "by_remaining_count_sq": {},    # {bucket_start: (sum_sq_errors, count)} (MSE)
                }
                for prefix in ("train", "eval")
            }
        return buckets

    def _update_cls_derived_pct_buckets(self, logits, pct_targets_np, count_targets_np, probe_name, prefix):
        """Accumulate derived-percentage MAE for a classification probe.

        Derived percentage = sum(softmax(logits) * anchors). Classification probes
        are trained on *completion progress* (anchor 0 at completion start,
        anchor 1 at completion end, see compute_soft_class_targets), so we
        compare against (1 - remaining_percentage), not remaining_percentage.

        Args:
            logits: [batch, seq_len, num_bins] raw logits (tensor)
            pct_targets_np: [batch, seq_len] remaining-percentage targets (-100 = masked)
            count_targets_np: [batch, seq_len] remaining count targets (numpy, for x-axis)
            probe_name: name of the classification probe
            prefix: 'train' or 'eval'
        """
        anchors = self._cls_anchors[probe_name].to(logits.device)  # [num_bins]
        probs = F.softmax(logits, dim=-1)  # [B, seq_len, num_bins]
        derived_pct = (probs * anchors).sum(dim=-1).cpu().numpy()  # [B, seq_len]

        by_compl = self._cls_derived_pct_buckets[probe_name][prefix]["by_completion_length"]
        by_count = self._cls_derived_pct_buckets[probe_name][prefix]["by_remaining_count"]
        by_compl_sq = self._cls_derived_pct_buckets[probe_name][prefix]["by_completion_length_sq"]
        by_count_sq = self._cls_derived_pct_buckets[probe_name][prefix]["by_remaining_count_sq"]

        for pred_seq, tgt_seq, count_seq in zip(derived_pct, pct_targets_np, count_targets_np):
            valid = tgt_seq != -100
            if not valid.any():
                continue

            preds = pred_seq[valid]
            # Convert remaining-percentage targets to completion-progress targets
            # so they match the convention the cls probe was trained on.
            targets = 1.0 - tgt_seq[valid]
            counts = count_seq[valid]
            errors = np.abs(preds - targets)
            sq_errors = errors ** 2

            # By completion length (one entry per sequence)
            completion_length = len(targets)
            bucket = (completion_length // self.bin_size) * self.bin_size
            sum_err = errors.sum()
            sum_sq = sq_errors.sum()
            n = len(errors)
            if bucket in by_compl:
                prev_sum, prev_count = by_compl[bucket]
                by_compl[bucket] = (prev_sum + sum_err, prev_count + n)
                prev_sq, prev_sq_count = by_compl_sq[bucket]
                by_compl_sq[bucket] = (prev_sq + sum_sq, prev_sq_count + n)
            else:
                by_compl[bucket] = (sum_err, n)
                by_compl_sq[bucket] = (sum_sq, n)

            # By remaining token count (one entry per token position)
            for error, sq_error, count_val in zip(errors, sq_errors, counts):
                bucket = (int(count_val) // self.bin_size) * self.bin_size
                if bucket in by_count:
                    prev_sum, prev_count = by_count[bucket]
                    by_count[bucket] = (prev_sum + error, prev_count + 1)
                    prev_sq, prev_sq_count = by_count_sq[bucket]
                    by_count_sq[bucket] = (prev_sq + sq_error, prev_sq_count + 1)
                else:
                    by_count[bucket] = (error, 1)
                    by_count_sq[bucket] = (sq_error, 1)

    def create_optimizer(self):
        """Override to prevent Trainer from creating its own optimizer.

        Real optimization is handled per-probe in training_step.
        We return the count probe's optimizer to satisfy Trainer internals.
        """
        self.optimizer = self.probe_optimizers[self._canonical_count_name]
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Override to use custom scheduler factory if provided, or skip."""
        if self.scheduler_factory is not None:
            # Apply scheduler to each probe optimizer
            self.lr_scheduler = self.scheduler_factory(self.optimizer)
        else:
            super().create_scheduler(num_training_steps, optimizer)

    def _make_empty_count_bucket_state(self):
        """Empty state dict compatible with CustomTrainer._update_count_buckets."""
        return {
            "mae_buckets": {},
            "mae_individual_buckets": {},
            "mse_buckets": {},
            "mse_individual_buckets": {},
            "mape_buckets": {},
            "smape_buckets": {},
            "token_count_sum": 0,
            "token_count_num": 0,
            "mape_sum": 0,
            "mape_num": 0,
            "smape_sum": 0,
            "smape_num": 0,
        }

    def _make_empty_baseline_count_buckets(self):
        """Per-probe bucket state for count-predicting regression baselines."""
        return {
            name: {split: self._make_empty_count_bucket_state() for split in ("train", "eval")}
            for name in self._count_baseline_names
        }

    def _make_empty_baseline_pct_buckets(self):
        """Per-probe bucket state for percentage-predicting regression baselines."""
        return {
            name: {
                split: {
                    "by_completion_length": {},
                    "by_remaining_count": {},
                    "by_completion_length_sq": {},
                    "by_remaining_count_sq": {},
                }
                for split in ("train", "eval")
            }
            for name in self._pct_baseline_names
        }

    def _update_pct_probe_buckets(self, pct_preds_np, pct_targets_np, count_targets_np, probe_name, split):
        """Per-probe variant of _update_pct_buckets, used by the per-(probe, layer)
        percentage probes so each layer's MAE/MSE charts and scalars land under
        its own WandB namespace.
        """
        buckets = self._per_probe_pct_buckets[probe_name][split]
        by_compl = buckets["by_completion_length"]
        by_count = buckets["by_remaining_count"]
        by_compl_sq = buckets["by_completion_length_sq"]
        by_count_sq = buckets["by_remaining_count_sq"]

        for pct_pred, pct_tgt, count_tgt in zip(pct_preds_np, pct_targets_np, count_targets_np):
            valid = pct_tgt != -100
            if not valid.any():
                continue
            preds = pct_pred[valid]
            targets = pct_tgt[valid]
            counts = count_tgt[valid]
            errors = np.abs(preds - targets)
            sq_errors = errors ** 2

            completion_length = len(targets)
            bucket = (completion_length // self.bin_size) * self.bin_size
            sum_err = errors.sum()
            sum_sq = sq_errors.sum()
            n = len(errors)
            if bucket in by_compl:
                prev_sum, prev_count = by_compl[bucket]
                by_compl[bucket] = (prev_sum + sum_err, prev_count + n)
                prev_sq, prev_sq_count = by_compl_sq[bucket]
                by_compl_sq[bucket] = (prev_sq + sum_sq, prev_sq_count + n)
            else:
                by_compl[bucket] = (sum_err, n)
                by_compl_sq[bucket] = (sum_sq, n)

            for error, sq_error, count_val in zip(errors, sq_errors, counts):
                bucket = (int(count_val) // self.bin_size) * self.bin_size
                if bucket in by_count:
                    prev_sum, prev_count = by_count[bucket]
                    by_count[bucket] = (prev_sum + error, prev_count + 1)
                    prev_sq, prev_sq_count = by_count_sq[bucket]
                    by_count_sq[bucket] = (prev_sq + sq_error, prev_sq_count + 1)
                else:
                    by_count[bucket] = (error, 1)
                    by_count_sq[bucket] = (sq_error, 1)

    def _update_baseline_pct_buckets(self, pct_preds_np, pct_targets_np, count_targets_np, probe_name, split):
        """Accumulate percentage baseline MAE / MSE (same structure as main percentage probe)."""
        buckets = self._baseline_pct_buckets[probe_name][split]
        by_compl = buckets["by_completion_length"]
        by_count = buckets["by_remaining_count"]
        by_compl_sq = buckets["by_completion_length_sq"]
        by_count_sq = buckets["by_remaining_count_sq"]

        for pct_pred, pct_tgt, count_tgt in zip(pct_preds_np, pct_targets_np, count_targets_np):
            valid = pct_tgt != -100
            if not valid.any():
                continue
            preds = pct_pred[valid]
            targets = pct_tgt[valid]
            counts = count_tgt[valid]
            errors = np.abs(preds - targets)
            sq_errors = errors ** 2

            completion_length = len(targets)
            bucket = (completion_length // self.bin_size) * self.bin_size
            sum_err = errors.sum()
            sum_sq = sq_errors.sum()
            n = len(errors)
            if bucket in by_compl:
                prev_sum, prev_count = by_compl[bucket]
                by_compl[bucket] = (prev_sum + sum_err, prev_count + n)
                prev_sq, prev_sq_count = by_compl_sq[bucket]
                by_compl_sq[bucket] = (prev_sq + sum_sq, prev_sq_count + n)
            else:
                by_compl[bucket] = (sum_err, n)
                by_compl_sq[bucket] = (sum_sq, n)

            for error, sq_error, count_val in zip(errors, sq_errors, counts):
                bucket = (int(count_val) // self.bin_size) * self.bin_size
                if bucket in by_count:
                    prev_sum, prev_count = by_count[bucket]
                    by_count[bucket] = (prev_sum + error, prev_count + 1)
                    prev_sq, prev_sq_count = by_count_sq[bucket]
                    by_count_sq[bucket] = (prev_sq + sq_error, prev_sq_count + 1)
                else:
                    by_count[bucket] = (error, 1)
                    by_count_sq[bucket] = (sq_error, 1)

    def _generate_and_extract_inputs(self, inputs):
        """Streaming-mode entrypoint: generate completions + extract hidden states
        + populate per-probe targets, returning an `inputs` dict in the same
        shape the offline `hidden_states_collator` produces. The rest of the
        per-probe training_step / prediction_step logic is then unchanged.
        """
        from miol.model.generation_utils import (
            compute_percentage_targets,
            compute_soft_class_targets,
            generate_and_extract_hidden_states,
        )
        tok = self._streaming_tokenizer
        batch = generate_and_extract_hidden_states(
            base_model=self._streaming_base_model,
            hidden_layer=self._streaming_hidden_layer,
            prompt_input_ids=inputs["input_ids"],
            prompt_attention_mask=inputs["attention_mask"],
            generation_config=self.generation_config,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            bos_token_id=tok.bos_token_id,
        )

        device = batch["hidden_states"].device
        batch_size, max_seq_len = batch["input_ids"].shape

        # Compute the additional targets the multi-probe loop needs
        # (remaining_percentage and one soft_targets_N per active classification
        # probe). Right-padded to max_seq_len with -100 to match the offline
        # collator layout.
        remaining_percentage = torch.full((batch_size, max_seq_len), -100.0, dtype=torch.float32, device=device)
        soft_targets: dict[str, torch.Tensor] = {}
        cls_specs = [(name, spec) for name, spec in self.probe_specs.items() if spec["type"] == "classification"]
        for name, spec in cls_specs:
            soft_targets[name] = torch.full(
                (batch_size, max_seq_len, spec["output_dim"]),
                -100.0,
                dtype=torch.float32,
                device=device,
            )

        sigma = float(getattr(self, "_streaming_classification_sigma", 0.15))
        prompt_lengths = batch["prompt_length"].tolist()
        has_eos_list = batch["has_eos"].tolist()
        completion_lengths = batch["completion_length"].tolist()

        # Prompt-end-only regression target: T at index (prompt_length - 1),
        # -100 elsewhere. Built only when at least one active probe needs it.
        needs_prompt_end_total_length = any(
            spec.get("train_position") == "prompt_end"
            and spec.get("target_key") == "total_completion_length_at_prompt_end"
            for spec in self.probe_specs.values()
        )
        total_length_at_prompt_end = (
            torch.full((batch_size, max_seq_len), -100.0, dtype=torch.float32, device=device)
            if needs_prompt_end_total_length else None
        )

        for i in range(batch_size):
            seq_len = int(prompt_lengths[i] + completion_lengths[i])
            if seq_len <= 0:
                continue
            pct = compute_percentage_targets(
                sequence_length=seq_len,
                prompt_length=int(prompt_lengths[i]),
                has_eos=bool(has_eos_list[i]),
                device=device,
            )
            remaining_percentage[i, :seq_len] = pct
            for name, spec in cls_specs:
                soft = compute_soft_class_targets(
                    sequence_length=seq_len,
                    prompt_length=int(prompt_lengths[i]),
                    has_eos=bool(has_eos_list[i]),
                    device=device,
                    sigma=sigma,
                    num_bins=spec["output_dim"],
                )
                soft_targets[name][i, :seq_len] = soft

            if total_length_at_prompt_end is not None and bool(has_eos_list[i]) and int(prompt_lengths[i]) >= 1:
                total_length_at_prompt_end[i, int(prompt_lengths[i]) - 1] = float(completion_lengths[i])

        # log(1 + count) target, masked positions preserved as -100.
        rtc = batch["remaining_token_counts"]
        remaining_log_count = torch.where(
            rtc == -100,
            torch.full_like(rtc, -100, dtype=torch.float32),
            torch.log1p(rtc.to(torch.float32)),
        )

        out = dict(inputs)  # preserve any extra keys the trainer expects
        out["input_ids"] = batch["input_ids"]
        out["attention_mask"] = batch["attention_mask"]
        out["hidden_states"] = batch["hidden_states"]
        out["remaining_token_counts"] = batch["remaining_token_counts"]
        out["remaining_percentage"] = remaining_percentage
        out["remaining_log_count"] = remaining_log_count
        if total_length_at_prompt_end is not None:
            out["total_completion_length_at_prompt_end"] = total_length_at_prompt_end
        for name, t in soft_targets.items():
            out[self.probe_specs[name]["target_key"]] = t
        out["has_eos"] = batch["has_eos"]
        out["prompt_length"] = batch["prompt_length"]
        out["completion_length"] = batch["completion_length"]
        return out

    def _build_probe_input(self, spec, hidden_states, prompt_lengths):
        """Build the input tensor for a probe based on its input_type.

        Args:
            spec: probe spec dict. In multi-layer mode the spec also carries
                a `layer_index` field (slice position in the hidden_states
                tensor's layer axis); single-layer specs have no layer_index.
            hidden_states: [B, S, H] in single-layer mode, or
                [B, S, num_selected_layers, H] in multi-layer mode.
            prompt_lengths: [B] (used only for input_type='position')

        Returns:
            Tensor of shape [B, S, input_dim] matching the probe head's expected input.
        """
        input_type = spec.get("input_type", "hidden_states")
        if input_type == "hidden_states":
            if hidden_states.dim() == 4:
                # Multi-layer mode: slice the layer axis; the spec's
                # layer_index is the slice position (NOT the absolute layer
                # number; train_offline.py builds this mapping).
                layer_index = spec.get("layer_index")
                if layer_index is None:
                    raise ValueError(
                        f"Multi-layer hidden_states (shape {tuple(hidden_states.shape)}) "
                        f"requires every hidden-state probe spec to carry a layer_index; "
                        f"got spec={spec}"
                    )
                return hidden_states[:, :, layer_index, :]
            return hidden_states
        B, S = hidden_states.shape[:2]
        device = hidden_states.device
        if input_type == "constant":
            return torch.ones(B, S, 1, device=device, dtype=hidden_states.dtype)
        if input_type == "position":
            seq_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)  # [B, S]
            completion_position = (seq_idx - prompt_lengths.unsqueeze(-1)).clamp(min=0)
            return completion_position.unsqueeze(-1).to(hidden_states.dtype)  # [B, S, 1]
        raise ValueError(f"Unknown input_type: {input_type}")

    def _compute_probe_loss(self, logits, targets, spec):
        """Compute loss for a single probe given its type."""
        if spec["type"] == "regression":
            preds = logits.squeeze(-1)  # [B, seq_len]
            flat_preds = preds.view(-1)
            flat_targets = targets.view(-1)
            active = flat_targets != -100
            if active.any():
                return self.loss_fn(flat_preds[active], flat_targets[active].float())
            return preds.sum() * 0.0
        else:
            # Classification: targets [B, seq_len, num_bins], logits [B, seq_len, num_bins]
            mask = targets[..., 0] != -100  # [B, seq_len]
            if mask.any():
                # F.kl_div expects log-probabilities as input; log_softmax is more
                # numerically stable than log(softmax(x))
                log_probs = F.log_softmax(logits[mask], dim=-1)
                return F.kl_div(log_probs, targets[mask], reduction="batchmean")
            return logits.sum() * 0.0

    def _update_cls_metrics(self, logits, targets, probe_name, prefix):
        """Accumulate per-bin accuracy/FPR/FNR for a classification probe batch.

        Args:
            logits: [B, seq_len, num_bins] raw logits
            targets: [B, seq_len, num_bins] soft targets (-100 for masked)
            probe_name: name of the classification probe
            prefix: 'train' or 'eval'
        """
        mask = targets[..., 0] != -100  # [B, seq_len]
        if not mask.any():
            return

        pred_classes = logits[mask].argmax(dim=-1).cpu().numpy()
        true_classes = targets[mask].argmax(dim=-1).cpu().numpy()
        num_bins = self.probe_specs[probe_name]["output_dim"]
        bins_metrics = self._cls_metrics[probe_name][prefix]

        for b in range(num_bins):
            true_is_b = (true_classes == b)
            pred_is_b = (pred_classes == b)

            # Accuracy for bin b: TP / (TP + FN) = correct / total where true == b
            total_true_b = true_is_b.sum()
            if total_true_b > 0:
                acc = (true_is_b & pred_is_b).sum() / total_true_b
                fnr = 1.0 - acc
                bins_metrics["accuracy"][b].append(float(acc))
                bins_metrics["fnr"][b].append(float(fnr))

            # FPR for bin b: FP / (FP + TN) = wrong pred as b / total where true != b
            total_not_b = (~true_is_b).sum()
            if total_not_b > 0:
                fpr = (~true_is_b & pred_is_b).sum() / total_not_b
                bins_metrics["fpr"][b].append(float(fpr))

    def _log_cls_chart(self, probe_name, prefix, global_step):
        """Create and log a grouped bar chart of accuracy/FPR/FNR per bin."""
        bins_metrics = self._cls_metrics[probe_name][prefix]
        num_bins = self.probe_specs[probe_name]["output_dim"]

        # Compute mean and std per bin
        acc_means, acc_stds = [], []
        fpr_means, fpr_stds = [], []
        fnr_means, fnr_stds = [], []

        for b in range(num_bins):
            acc_vals = bins_metrics["accuracy"][b]
            fpr_vals = bins_metrics["fpr"][b]
            fnr_vals = bins_metrics["fnr"][b]

            acc_means.append(np.mean(acc_vals) if acc_vals else 0.0)
            acc_stds.append(np.std(acc_vals) if acc_vals else 0.0)
            fpr_means.append(np.mean(fpr_vals) if fpr_vals else 0.0)
            fpr_stds.append(np.std(fpr_vals) if fpr_vals else 0.0)
            fnr_means.append(np.mean(fnr_vals) if fnr_vals else 0.0)
            fnr_stds.append(np.std(fnr_vals) if fnr_vals else 0.0)

        # Skip if no data
        if all(m == 0.0 and s == 0.0 for m, s in zip(acc_means, acc_stds)):
            return

        x = np.arange(num_bins)
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, num_bins * 2), 6))

        bars_acc = ax.bar(
            x - bar_width, acc_means, bar_width,
            yerr=acc_stds, capsize=4, label="Accuracy", color="steelblue",
        )
        bars_fpr = ax.bar(
            x, fpr_means, bar_width,
            yerr=fpr_stds, capsize=4, label="False Positive Rate", color="coral",
        )
        bars_fnr = ax.bar(
            x + bar_width, fnr_means, bar_width,
            yerr=fnr_stds, capsize=4, label="False Negative Rate", color="goldenrod",
        )

        ax.set_xlabel("Classification Bin", fontsize=12)
        ax.set_ylabel("Rate", fontsize=12)
        ax.set_title(
            f"{probe_name} — Per-Bin Metrics ({prefix.capitalize()}) — Step {global_step}",
            fontsize=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Bin {b}" for b in range(num_bins)])
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        image = Image.open(buf)

        self._log_to_wandb({
            f"{probe_name}/cls_metrics/{prefix}_chart": wandb.Image(
                image, caption=f"Step {global_step}"
            ),
        })

        plt.close(fig)
        buf.close()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to do separate forward+backward per probe head."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        if self.streaming:
            inputs = self._generate_and_extract_inputs(inputs)
        hidden_states = inputs["hidden_states"].float()

        unwrapped = model.module if hasattr(model, 'module') else model
        should_collect = is_main_process() and self.state.global_step > 0

        # Pre-compute shared numpy arrays for metric collection
        if should_collect:
            has_eos_list = inputs.get("has_eos", torch.ones(hidden_states.shape[0], dtype=torch.bool))
            has_eos_np = has_eos_list.cpu().numpy() if isinstance(has_eos_list, torch.Tensor) else has_eos_list
            count_targets_np = inputs["remaining_token_counts"].cpu().numpy()

        prompt_lengths = inputs["prompt_length"]

        total_loss = 0.0
        for name, spec in self.probe_specs.items():
            optimizer = self.probe_optimizers[name]
            optimizer.zero_grad()

            # Forward through this probe's head with probe-specific input
            probe_input = self._build_probe_input(spec, hidden_states, prompt_lengths)
            logits = unwrapped.heads[name](probe_input)
            targets = inputs[spec["target_key"]]
            loss = self._compute_probe_loss(logits, targets, spec)

            # Backward + step (independent graphs since hidden_states has no grad)
            loss.backward()
            optimizer.step()

            loss_val = loss.detach().item()
            total_loss += loss_val

            # Accumulate per-probe loss for logging
            self._probe_loss_sums[name] += loss_val
            self._probe_loss_counts[name] += 1

            # Accumulate per-bin classification metrics
            if spec["type"] == "classification" and should_collect:
                with torch.no_grad():
                    logits_detached = logits.detach()
                    self._update_cls_metrics(logits_detached, targets, name, "train")
                    self._update_cls_acc_buckets(logits_detached, targets, count_targets_np, has_eos_np, name, "train")
                    self._update_cls_confusion(logits_detached, targets, name, "train")
                    pct_targets_np = inputs["remaining_percentage"].cpu().numpy()
                    self._update_cls_derived_pct_buckets(logits_detached, pct_targets_np, count_targets_np, name, "train")

            # Per-bin bucket tracking for regression baselines
            if should_collect and name in self._count_baseline_names:
                with torch.no_grad():
                    preds_np = logits.detach().squeeze(-1).cpu().numpy()
                self._update_count_buckets(preds_np, count_targets_np, has_eos_np, self._baseline_count_buckets[name]["train"])
            if should_collect and name in self._pct_baseline_names:
                with torch.no_grad():
                    preds_np = logits.detach().squeeze(-1).cpu().numpy()
                pct_targets_np = inputs["remaining_percentage"].cpu().numpy()
                self._update_baseline_pct_buckets(preds_np, pct_targets_np, count_targets_np, name, "train")

            # Prompt-only probe: only the prompt-end position contributes during
            # training (loss is masked elsewhere); completion-position metrics
            # are computed at eval time.
            if should_collect and name in self._prompt_only_probe_names:
                with torch.no_grad():
                    preds_np = logits.detach().squeeze(-1).cpu().numpy()
                pe_targets_np = targets.cpu().numpy()
                prompt_lengths_np = prompt_lengths.cpu().numpy()
                self._update_prompt_only_metrics(
                    preds_np, pe_targets_np, count_targets_np, has_eos_np,
                    prompt_lengths_np,
                    probe_name=name, split="train", update_completion=False,
                )

        # Update per-(probe, split) count and percentage bucket stats. Every
        # hidden-state count probe (count, count_L0, count_L8, ... in
        # multi-layer mode) and every percentage probe gets its own state, so
        # the WandB namespace is the probe's name and per-layer curves don't
        # collapse into a single `count/` series.
        if should_collect:
            pct_targets_np = (
                inputs["remaining_percentage"].cpu().numpy()
                if self._percentage_probe_names else None
            )
            for name in self._count_probe_names:
                spec = self.probe_specs[name]
                probe_input = self._build_probe_input(spec, hidden_states, prompt_lengths)
                with torch.no_grad():
                    preds = unwrapped.heads[name](probe_input).squeeze(-1)
                self._update_count_buckets(
                    preds.cpu().numpy(), count_targets_np, has_eos_np,
                    self._per_probe_count_buckets[name]["train"],
                )
            # log_count probes: expm1 predictions back to count space and
            # bucket against the same count targets as the count probe so
            # log_count/ metrics are directly comparable to count/.
            for name in self._log_count_probe_names:
                spec = self.probe_specs[name]
                probe_input = self._build_probe_input(spec, hidden_states, prompt_lengths)
                with torch.no_grad():
                    log_preds = unwrapped.heads[name](probe_input).squeeze(-1)
                count_preds_np = np.expm1(np.clip(log_preds.cpu().numpy(), 0.0, None))
                self._update_count_buckets(
                    count_preds_np, count_targets_np, has_eos_np,
                    self._per_probe_log_count_buckets[name]["train"],
                )
            for name in self._percentage_probe_names:
                spec = self.probe_specs[name]
                probe_input = self._build_probe_input(spec, hidden_states, prompt_lengths)
                with torch.no_grad():
                    pct_preds = unwrapped.heads[name](probe_input).squeeze(-1)
                self._update_pct_probe_buckets(
                    pct_preds.cpu().numpy(), pct_targets_np, count_targets_np,
                    name, "train",
                )

        return torch.tensor(total_loss / len(self.probe_specs), device=hidden_states.device)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Run all probes for evaluation, return count probe as primary."""
        inputs = self._prepare_inputs(inputs)
        if self.streaming:
            inputs = self._generate_and_extract_inputs(inputs)
        hidden_states = inputs["hidden_states"].float()

        unwrapped = model.module if hasattr(model, 'module') else model

        prompt_lengths = inputs["prompt_length"]

        with torch.no_grad():
            # Evaluate all probes and log per-probe eval losses
            for name, spec in self.probe_specs.items():
                probe_input = self._build_probe_input(spec, hidden_states, prompt_lengths)
                logits = unwrapped.heads[name](probe_input)
                targets = inputs[spec["target_key"]]
                loss = self._compute_probe_loss(logits, targets, spec)

                # Log per-probe eval loss
                self._probe_loss_sums[name] += loss.item()
                self._probe_loss_counts[name] += 1

                # Accumulate per-bin classification metrics for eval
                if spec["type"] == "classification" and is_main_process():
                    self._update_cls_metrics(logits, targets, name, "eval")
                    count_targets_np_eval = inputs["remaining_token_counts"].cpu().numpy()
                    has_eos_eval = inputs.get("has_eos", torch.ones(hidden_states.shape[0], dtype=torch.bool))
                    has_eos_np_eval = has_eos_eval.cpu().numpy() if isinstance(has_eos_eval, torch.Tensor) else has_eos_eval
                    self._update_cls_acc_buckets(logits, targets, count_targets_np_eval, has_eos_np_eval, name, "eval")
                    self._update_cls_confusion(logits, targets, name, "eval")
                    pct_targets_np_eval = inputs["remaining_percentage"].cpu().numpy()
                    self._update_cls_derived_pct_buckets(logits, pct_targets_np_eval, count_targets_np_eval, name, "eval")

                # Per-bin bucket tracking for regression baselines
                if is_main_process() and name in self._count_baseline_names:
                    preds_np = logits.squeeze(-1).cpu().numpy()
                    eval_count_targets_np = inputs["remaining_token_counts"].cpu().numpy()
                    eval_has_eos = inputs.get("has_eos", torch.ones(hidden_states.shape[0], dtype=torch.bool))
                    eval_has_eos_np = eval_has_eos.cpu().numpy() if isinstance(eval_has_eos, torch.Tensor) else eval_has_eos
                    self._update_count_buckets(preds_np, eval_count_targets_np, eval_has_eos_np, self._baseline_count_buckets[name]["eval"])
                if is_main_process() and name in self._pct_baseline_names:
                    preds_np = logits.squeeze(-1).cpu().numpy()
                    eval_pct_targets_np = inputs["remaining_percentage"].cpu().numpy()
                    eval_count_targets_np = inputs["remaining_token_counts"].cpu().numpy()
                    self._update_baseline_pct_buckets(preds_np, eval_pct_targets_np, eval_count_targets_np, name, "eval")

                # Prompt-only probe: at eval time we update both the prompt-end
                # bucket AND the completion-position bucket so we can observe
                # how well the prompt-trained direction transfers downstream.
                if is_main_process() and name in self._prompt_only_probe_names:
                    preds_np = logits.squeeze(-1).cpu().numpy()
                    eval_count_targets_np = inputs["remaining_token_counts"].cpu().numpy()
                    pe_targets_np = targets.cpu().numpy()
                    prompt_lengths_np = prompt_lengths.cpu().numpy()
                    eval_has_eos = inputs.get("has_eos", torch.ones(hidden_states.shape[0], dtype=torch.bool))
                    eval_has_eos_np_local = eval_has_eos.cpu().numpy() if isinstance(eval_has_eos, torch.Tensor) else eval_has_eos
                    self._update_prompt_only_metrics(
                        preds_np, pe_targets_np, eval_count_targets_np, eval_has_eos_np_local,
                        prompt_lengths_np,
                        probe_name=name, split="eval", update_completion=True,
                    )

            # Canonical count probe predictions for primary metrics. In
            # multi-layer mode this is the layer-namespaced canonical (e.g.
            # count_L0); the per-bin chart and HF Trainer's primary loss
            # follow whichever layer is canonical.
            count_spec = self.probe_specs[self._canonical_count_name]
            count_input = self._build_probe_input(count_spec, hidden_states, prompt_lengths)
            count_preds = unwrapped.heads[self._canonical_count_name](count_input).squeeze(-1)
            count_targets = inputs["remaining_token_counts"]

            # Count probe loss
            flat_preds = count_preds.view(-1)
            flat_targets = count_targets.view(-1)
            active = flat_targets != -100
            if active.any():
                count_loss = self.loss_fn(flat_preds[active], flat_targets[active].float())
            else:
                count_loss = count_preds.sum() * 0.0

        # Update per-(probe, split) eval count and percentage bucket stats.
        # See training_step for the rationale; same pattern.
        if is_main_process() and self.state.global_step > 0:
            has_eos_list = inputs.get("has_eos", torch.ones(count_preds.shape[0], dtype=torch.bool))
            has_eos_np = has_eos_list.cpu().numpy() if isinstance(has_eos_list, torch.Tensor) else has_eos_list
            count_targets_np = count_targets.cpu().numpy()
            pct_targets_np = (
                inputs["remaining_percentage"].cpu().numpy()
                if self._percentage_probe_names else None
            )

            for name in self._count_probe_names:
                spec = self.probe_specs[name]
                probe_input = self._build_probe_input(spec, hidden_states, prompt_lengths)
                with torch.no_grad():
                    preds = unwrapped.heads[name](probe_input).squeeze(-1)
                self._update_count_buckets(
                    preds.cpu().numpy(), count_targets_np, has_eos_np,
                    self._per_probe_count_buckets[name]["eval"],
                )
            for name in self._log_count_probe_names:
                spec = self.probe_specs[name]
                probe_input = self._build_probe_input(spec, hidden_states, prompt_lengths)
                with torch.no_grad():
                    log_preds = unwrapped.heads[name](probe_input).squeeze(-1)
                count_preds_np = np.expm1(np.clip(log_preds.cpu().numpy(), 0.0, None))
                self._update_count_buckets(
                    count_preds_np, count_targets_np, has_eos_np,
                    self._per_probe_log_count_buckets[name]["eval"],
                )
            for name in self._percentage_probe_names:
                spec = self.probe_specs[name]
                probe_input = self._build_probe_input(spec, hidden_states, prompt_lengths)
                with torch.no_grad():
                    pct_preds = unwrapped.heads[name](probe_input).squeeze(-1)
                self._update_pct_probe_buckets(
                    pct_preds.cpu().numpy(), pct_targets_np, count_targets_np,
                    name, "eval",
                )

        if prediction_loss_only:
            return (count_loss, None, None)

        return (count_loss, count_preds, count_targets)

    def on_step_end(self, global_step):
        """Log per-probe losses and classification charts, then delegate to parent."""
        if is_main_process() and wandb.run is not None:
            for name, spec in self.probe_specs.items():
                count = self._probe_loss_counts[name]
                if count > 0:
                    avg_loss = self._probe_loss_sums[name] / count
                    self._log_to_wandb({f"{name}/loss": avg_loss}, commit=False)

                # Log classification charts (per-class accuracy/FPR/FNR)
                if spec["type"] == "classification":
                    self._log_cls_chart(name, "train", global_step)
                    self._log_cls_chart(name, "eval", global_step)

                    # Log accuracy-by-position charts (by completion length & remaining count)
                    for split in ("train", "eval"):
                        probe_buckets = self._cls_acc_buckets[name][split]

                        if probe_buckets["by_completion_length"]:
                            acc_per_bin = {}
                            counts_per_bin = {}
                            for b, (s, c) in sorted(probe_buckets["by_completion_length"].items()):
                                acc_per_bin[b] = s / c if c > 0 else 0
                                counts_per_bin[b] = c
                            self._log_mae_chart(
                                acc_per_bin, counts_per_bin, global_step, global_step, split,
                                metric="accuracy", prefix=f"{name}/", ylabel="Accuracy",
                            )

                            # Log aggregate accuracy scalar
                            total_correct = sum(s for s, _ in probe_buckets["by_completion_length"].values())
                            total = sum(c for _, c in probe_buckets["by_completion_length"].values())
                            if total > 0:
                                self._log_to_wandb({f"{name}/{split}_accuracy": total_correct / total}, commit=False)

                        # Cohen's kappa (chance-corrected accuracy) — uses the
                        # full confusion matrix accumulated this window.
                        cm = self._cls_confusion[name][split]
                        if cm.sum() > 0:
                            kappa = self._compute_cohen_kappa(cm)
                            self._log_to_wandb({f"{name}/{split}_cohen_kappa": kappa}, commit=False)

                        if probe_buckets["by_remaining_count"]:
                            acc_per_bin = {}
                            counts_per_bin = {}
                            for b, (s, c) in sorted(probe_buckets["by_remaining_count"].items()):
                                acc_per_bin[b] = s / c if c > 0 else 0
                                counts_per_bin[b] = c
                            self._log_mae_chart(
                                acc_per_bin, counts_per_bin, global_step, global_step, split,
                                metric="accuracy_individual", prefix=f"{name}/", ylabel="Accuracy",
                            )

                        # Derived-percentage MAE (expected position from soft probabilities)
                        dp_buckets = self._cls_derived_pct_buckets[name][split]

                        if dp_buckets["by_completion_length"]:
                            mae_per_bin = {}
                            counts_per_bin = {}
                            for b, (s, c) in sorted(dp_buckets["by_completion_length"].items()):
                                mae_per_bin[b] = s / c if c > 0 else 0
                                counts_per_bin[b] = c
                            self._log_mae_chart(
                                mae_per_bin, counts_per_bin, global_step, global_step, split,
                                metric="derived_pct_mae", prefix=f"{name}/",
                            )

                            # Log aggregate derived-percentage MAE scalar
                            total_err = sum(s for s, _ in dp_buckets["by_completion_length"].values())
                            total_cnt = sum(c for _, c in dp_buckets["by_completion_length"].values())
                            if total_cnt > 0:
                                self._log_to_wandb({f"{name}/{split}_derived_pct_mae": total_err / total_cnt}, commit=False)

                        if dp_buckets["by_completion_length_sq"]:
                            mse_per_bin = {}
                            counts_per_bin = {}
                            for b, (s, c) in sorted(dp_buckets["by_completion_length_sq"].items()):
                                mse_per_bin[b] = s / c if c > 0 else 0
                                counts_per_bin[b] = c
                            self._log_mae_chart(
                                mse_per_bin, counts_per_bin, global_step, global_step, split,
                                metric="derived_pct_mse", prefix=f"{name}/", ylabel="MSE",
                            )

                            total_sq = sum(s for s, _ in dp_buckets["by_completion_length_sq"].values())
                            total_cnt = sum(c for _, c in dp_buckets["by_completion_length_sq"].values())
                            if total_cnt > 0:
                                self._log_to_wandb({f"{name}/{split}_derived_pct_mse": total_sq / total_cnt}, commit=False)

                        if dp_buckets["by_remaining_count"]:
                            mae_per_bin = {}
                            counts_per_bin = {}
                            for b, (s, c) in sorted(dp_buckets["by_remaining_count"].items()):
                                mae_per_bin[b] = s / c if c > 0 else 0
                                counts_per_bin[b] = c
                            self._log_mae_chart(
                                mae_per_bin, counts_per_bin, global_step, global_step, split,
                                metric="derived_pct_mae_individual", prefix=f"{name}/",
                            )

                        if dp_buckets["by_remaining_count_sq"]:
                            mse_per_bin = {}
                            counts_per_bin = {}
                            for b, (s, c) in sorted(dp_buckets["by_remaining_count_sq"].items()):
                                mse_per_bin[b] = s / c if c > 0 else 0
                                counts_per_bin[b] = c
                            self._log_mae_chart(
                                mse_per_bin, counts_per_bin, global_step, global_step, split,
                                metric="derived_pct_mse_individual", prefix=f"{name}/", ylabel="MSE",
                            )

            # Log percentage probe MAE charts
            # Per-probe count probe charts + aggregate scalars (mae/mape/smape
            # by completion length and by individual remaining count). One set
            # of charts per probe, namespaced by probe name so multi-layer
            # variants like `count_L0/`, `count_L8/` each get their own.
            for name in self._count_probe_names:
                for split in ("train", "eval"):
                    state = self._per_probe_count_buckets[name][split]
                    self._log_regression_buckets(state, global_step, split, f"{name}/")
                    if state["mae_buckets"]:
                        total_err = sum(s for s, _ in state["mae_buckets"].values())
                        total_cnt = sum(c for _, c in state["mae_buckets"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mae": total_err / total_cnt}, commit=False)
                    if state["mse_buckets"]:
                        total_sq = sum(s for s, _ in state["mse_buckets"].values())
                        total_cnt = sum(c for _, c in state["mse_buckets"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mse": total_sq / total_cnt}, commit=False)
                    if state["mape_num"] > 0:
                        self._log_to_wandb({f"{name}/{split}_mape": state["mape_sum"] / state["mape_num"]}, commit=False)
                    if state["smape_num"] > 0:
                        self._log_to_wandb({f"{name}/{split}_smape": state["smape_sum"] / state["smape_num"]}, commit=False)

            # log_count probes: same per-bin charts + aggregate scalars as the
            # count probe, but using the expm1-transformed (count-space) buckets
            # so the wandb namespace `log_count/...` is directly comparable to
            # `count/...`.
            for name in self._log_count_probe_names:
                for split in ("train", "eval"):
                    state = self._per_probe_log_count_buckets[name][split]
                    self._log_regression_buckets(state, global_step, split, f"{name}/")
                    if state["mae_buckets"]:
                        total_err = sum(s for s, _ in state["mae_buckets"].values())
                        total_cnt = sum(c for _, c in state["mae_buckets"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mae": total_err / total_cnt}, commit=False)
                    if state["mse_buckets"]:
                        total_sq = sum(s for s, _ in state["mse_buckets"].values())
                        total_cnt = sum(c for _, c in state["mse_buckets"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mse": total_sq / total_cnt}, commit=False)
                    if state["mape_num"] > 0:
                        self._log_to_wandb({f"{name}/{split}_mape": state["mape_sum"] / state["mape_num"]}, commit=False)
                    if state["smape_num"] > 0:
                        self._log_to_wandb({f"{name}/{split}_smape": state["smape_sum"] / state["smape_num"]}, commit=False)

            # Per-probe percentage probe charts + aggregate MAE / MSE scalars.
            for name in self._percentage_probe_names:
                for split in ("train", "eval"):
                    buckets = self._per_probe_pct_buckets[name][split]
                    if buckets["by_completion_length"]:
                        per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(buckets["by_completion_length"].items())}
                        counts = {b: c for b, (s, c) in buckets["by_completion_length"].items()}
                        self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mae", prefix=f"{name}/")
                        total_err = sum(s for s, _ in buckets["by_completion_length"].values())
                        total_cnt = sum(c for _, c in buckets["by_completion_length"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mae": total_err / total_cnt}, commit=False)
                    if buckets["by_completion_length_sq"]:
                        per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(buckets["by_completion_length_sq"].items())}
                        counts = {b: c for b, (s, c) in buckets["by_completion_length_sq"].items()}
                        self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mse", prefix=f"{name}/", ylabel="MSE")
                        total_sq = sum(s for s, _ in buckets["by_completion_length_sq"].values())
                        total_cnt = sum(c for _, c in buckets["by_completion_length_sq"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mse": total_sq / total_cnt}, commit=False)
                    if buckets["by_remaining_count"]:
                        per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(buckets["by_remaining_count"].items())}
                        counts = {b: c for b, (s, c) in buckets["by_remaining_count"].items()}
                        self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mae_individual", prefix=f"{name}/")
                    if buckets["by_remaining_count_sq"]:
                        per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(buckets["by_remaining_count_sq"].items())}
                        counts = {b: c for b, (s, c) in buckets["by_remaining_count_sq"].items()}
                        self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mse_individual", prefix=f"{name}/", ylabel="MSE")

            # Baseline probes: per-bin charts + aggregate MAE / MSE scalars
            for name in self._count_baseline_names:
                for split in ("train", "eval"):
                    state = self._baseline_count_buckets[name][split]
                    self._log_regression_buckets(state, global_step, split, f"{name}/")
                    if state["mae_buckets"]:
                        total_err = sum(s for s, _ in state["mae_buckets"].values())
                        total_cnt = sum(c for _, c in state["mae_buckets"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mae": total_err / total_cnt}, commit=False)
                    if state["mse_buckets"]:
                        total_sq = sum(s for s, _ in state["mse_buckets"].values())
                        total_cnt = sum(c for _, c in state["mse_buckets"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mse": total_sq / total_cnt}, commit=False)
                    if state["mape_num"] > 0:
                        self._log_to_wandb({f"{name}/{split}_mape": state["mape_sum"] / state["mape_num"]}, commit=False)
                    if state["smape_num"] > 0:
                        self._log_to_wandb({f"{name}/{split}_smape": state["smape_sum"] / state["smape_num"]}, commit=False)

            for name in self._pct_baseline_names:
                for split in ("train", "eval"):
                    buckets = self._baseline_pct_buckets[name][split]

                    if buckets["by_completion_length"]:
                        per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(buckets["by_completion_length"].items())}
                        counts = {b: c for b, (s, c) in buckets["by_completion_length"].items()}
                        self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mae", prefix=f"{name}/")
                        total_err = sum(s for s, _ in buckets["by_completion_length"].values())
                        total_cnt = sum(c for _, c in buckets["by_completion_length"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mae": total_err / total_cnt}, commit=False)

                    if buckets["by_completion_length_sq"]:
                        per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(buckets["by_completion_length_sq"].items())}
                        counts = {b: c for b, (s, c) in buckets["by_completion_length_sq"].items()}
                        self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mse", prefix=f"{name}/", ylabel="MSE")
                        total_sq = sum(s for s, _ in buckets["by_completion_length_sq"].values())
                        total_cnt = sum(c for _, c in buckets["by_completion_length_sq"].values())
                        if total_cnt > 0:
                            self._log_to_wandb({f"{name}/{split}_mse": total_sq / total_cnt}, commit=False)

                    if buckets["by_remaining_count"]:
                        per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(buckets["by_remaining_count"].items())}
                        counts = {b: c for b, (s, c) in buckets["by_remaining_count"].items()}
                        self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mae_individual", prefix=f"{name}/")

                    if buckets["by_remaining_count_sq"]:
                        per_bin = {b: s / c if c > 0 else 0 for b, (s, c) in sorted(buckets["by_remaining_count_sq"].items())}
                        counts = {b: c for b, (s, c) in buckets["by_remaining_count_sq"].items()}
                        self._log_mae_chart(per_bin, counts, global_step, global_step, split, "mse_individual", prefix=f"{name}/", ylabel="MSE")

            # Prompt-only probes: scalar MAE/MAPE/SMAPE per split per bucket,
            # plus per-bin bar charts (by completion length, and "individual"
            # by remaining count for the completion bucket only — there is no
            # remaining-count axis for the prompt bucket since it has one
            # position per example).
            for name in self._prompt_only_probe_names:
                for split in ("train", "eval"):
                    state = self._prompt_only_metrics[name][split]

                    # Helper: aggregate scalar from bucket sums.
                    def _bucket_mean(bdict):
                        total_s = sum(s for s, _ in bdict.values())
                        total_c = sum(c for _, c in bdict.values())
                        return (total_s / total_c) if total_c > 0 else None

                    if split == "train":
                        # Train: prompt bucket (loss-driven) + static_decay (a
                        # per-position curve derivable from the prompt-end
                        # prediction; useful for monitoring the MAPE tail).
                        pb = state["prompt"]
                        m = _bucket_mean(pb["mae_buckets"])
                        if m is not None:
                            self._log_to_wandb({f"{name}/train_mae": m}, commit=False)
                        m_sq = _bucket_mean(pb["mse_buckets"])
                        if m_sq is not None:
                            self._log_to_wandb({f"{name}/train_mse": m_sq}, commit=False)
                        if pb["mape_num"] > 0:
                            self._log_to_wandb({f"{name}/train_mape": pb["mape_sum"] / pb["mape_num"]}, commit=False)
                        if pb["smape_num"] > 0:
                            self._log_to_wandb({f"{name}/train_smape": pb["smape_sum"] / pb["smape_num"]}, commit=False)
                        self._log_regression_buckets(
                            pb, global_step, "train_prompt", f"{name}/",
                            include_mape=True, include_smape=True,
                        )

                        sd = state["static_decay"]
                        m = _bucket_mean(sd["mae_buckets"])
                        if m is not None:
                            self._log_to_wandb({f"{name}/train_mae_static_decay": m}, commit=False)
                        m_sq = _bucket_mean(sd["mse_buckets"])
                        if m_sq is not None:
                            self._log_to_wandb({f"{name}/train_mse_static_decay": m_sq}, commit=False)
                        if sd["mape_num"] > 0:
                            self._log_to_wandb({f"{name}/train_mape_static_decay": sd["mape_sum"] / sd["mape_num"]}, commit=False)
                        if sd["smape_num"] > 0:
                            self._log_to_wandb({f"{name}/train_smape_static_decay": sd["smape_sum"] / sd["smape_num"]}, commit=False)
                        self._log_regression_buckets(
                            sd, global_step, "train_static_decay", f"{name}/",
                            include_mape=True, include_smape=True,
                        )
                    else:
                        for bucket in ("prompt", "completion", "static_decay"):
                            b = state[bucket]
                            m = _bucket_mean(b["mae_buckets"])
                            if m is not None:
                                self._log_to_wandb({f"{name}/eval_mae_{bucket}": m}, commit=False)
                            m_sq = _bucket_mean(b["mse_buckets"])
                            if m_sq is not None:
                                self._log_to_wandb({f"{name}/eval_mse_{bucket}": m_sq}, commit=False)
                            if b["mape_num"] > 0:
                                self._log_to_wandb({f"{name}/eval_mape_{bucket}": b["mape_sum"] / b["mape_num"]}, commit=False)
                            if b["smape_num"] > 0:
                                self._log_to_wandb({f"{name}/eval_smape_{bucket}": b["smape_sum"] / b["smape_num"]}, commit=False)
                            self._log_regression_buckets(
                                b, global_step, f"eval_{bucket}", f"{name}/",
                                include_mape=True, include_smape=True,
                            )

        # Reset per-probe accumulators
        self._probe_loss_sums = {name: 0.0 for name in self.probe_specs}
        self._probe_loss_counts = {name: 0 for name in self.probe_specs}
        self._cls_metrics = self._make_empty_cls_metrics()
        self._pct_buckets = self._make_empty_pct_buckets()
        self._cls_acc_buckets = self._make_empty_cls_acc_buckets()
        self._cls_derived_pct_buckets = self._make_empty_cls_derived_pct_buckets()
        self._cls_confusion = self._make_empty_cls_confusion()
        self._baseline_count_buckets = self._make_empty_baseline_count_buckets()
        self._baseline_pct_buckets = self._make_empty_baseline_pct_buckets()
        self._prompt_only_metrics = self._make_empty_prompt_only_metrics()
        self._per_probe_count_buckets = {
            name: {split: self._make_empty_count_bucket_state() for split in ("train", "eval")}
            for name in self._count_probe_names
        }
        self._per_probe_log_count_buckets = {
            name: {split: self._make_empty_count_bucket_state() for split in ("train", "eval")}
            for name in self._log_count_probe_names
        }
        self._per_probe_pct_buckets = {
            name: {
                split: {
                    "by_completion_length": {},
                    "by_remaining_count": {},
                    "by_completion_length_sq": {},
                    "by_remaining_count_sq": {},
                }
                for split in ("train", "eval")
            }
            for name in self._percentage_probe_names
        }

        # We've moved count/percentage tracking to the per-probe path above;
        # the parent's on_step_end no longer has bucket data to log.
        super().on_step_end(global_step)

    def _save(self, output_dir=None, state_dict=None):
        """Save each probe head + per-probe optimizer state, plus a backward-
        compat token_count_head.pt. Called by HuggingFace Trainer for periodic
        checkpoints (`{args.output_dir}/checkpoint-N/`) and explicitly at
        end-of-training from train_offline.py.
        """
        if not is_main_process():
            return

        import os
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        for name in self.probe_specs:
            torch.save(
                {"head": unwrapped.heads[name].state_dict()},
                f"{output_dir}/{name}_head.pt",
            )
            # Per-probe optimizer state — required for proper resume since HF
            # Trainer only restores the single self.optimizer (the count probe).
            opt = self.probe_optimizers.get(name)
            if opt is not None:
                torch.save(
                    {"state_dict": opt.state_dict()},
                    f"{output_dir}/optimizer_{name}.pt",
                )

        # Backward compat: also save as token_count_head.pt — uses the canonical
        # count head (== "count" in single-layer mode, or e.g. "count_L0" in
        # multi-layer mode). Inference scripts that load this file will see
        # only the canonical-layer weights; multi-layer downstream consumers
        # should walk the per-{name}_head.pt files instead.
        torch.save(
            {"token_count_head": unwrapped.heads[self._canonical_count_name].state_dict()},
            f"{output_dir}/token_count_head.pt",
        )

    def load_probe_state(self, checkpoint_dir):
        """Restore probe head weights + per-probe optimizer states from a
        directory previously written by `_save`.

        HF Trainer's `train(resume_from_checkpoint=...)` will separately handle
        global_step / best_metric / RNG / its single tracked optimizer.
        """
        import os
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"resume_from path is not a directory: {checkpoint_dir}")

        unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        loaded_heads, loaded_opts, missing = [], [], []
        for name in self.probe_specs:
            head_path = f"{checkpoint_dir}/{name}_head.pt"
            if os.path.isfile(head_path):
                ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
                unwrapped.heads[name].load_state_dict(ckpt["head"])
                loaded_heads.append(name)
            else:
                missing.append(f"{name}_head.pt")

            opt_path = f"{checkpoint_dir}/optimizer_{name}.pt"
            if os.path.isfile(opt_path) and name in self.probe_optimizers:
                ckpt = torch.load(opt_path, map_location="cpu", weights_only=False)
                self.probe_optimizers[name].load_state_dict(ckpt["state_dict"])
                loaded_opts.append(name)
            else:
                missing.append(f"optimizer_{name}.pt")

        if is_main_process():
            print(f"[resume] Loaded {len(loaded_heads)} probe heads and "
                  f"{len(loaded_opts)} probe optimizers from {checkpoint_dir}")
            if missing:
                print(f"[resume] Missing files (skipped, will keep fresh state): {missing}")

    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to use custom save."""
        if not is_main_process():
            return
        if output_dir is None:
            output_dir = self.args.output_dir
        self._save(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """No-op override: the base model is frozen, so no standard
        `model.safetensors` / `pytorch_model.bin` is ever written by `_save`.
        HF Trainer's default `_load_from_checkpoint` would raise here looking
        for one of those files. Probe head weights are restored separately by
        `load_probe_state()` (called from `train_offline.py` before
        `trainer.train(resume_from_checkpoint=...)`), so there's nothing left
        for HF Trainer to load at the model level.

        HF Trainer's other resume hooks (`_load_rng_state`,
        `_load_optimizer_and_scheduler`, `trainer_state.json`) run separately
        and are unaffected.
        """
        if is_main_process():
            print(
                f"[resume] Skipping HF Trainer model load from {resume_from_checkpoint} "
                f"— frozen base model + probe heads restored via load_probe_state()."
            )
