"""
Callbacks for training token count prediction models.
"""

import os
import random
from miol.model.model import ModelWithTokenCountHead
import torch
import wandb
from transformers import TrainerCallback

from miol.data import tokenize_function
from miol.model.generation_utils import (
    build_generation_kwargs,
    extract_completion_with_eos,
    compute_remaining_token_targets,
)


def is_main_process():
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in [-1, 0]


class ValPlateauController(TrainerCallback):
    """Validation-driven control for offline multi-probe training.

    After every evaluation (on the held-out VALIDATION split, never the test
    split) it
      1. gets each count-style head's pooled validation MAE (all-reduced, so
         every rank sees identical numbers and makes identical decisions),
      2. steps the LR schedulers of the non-canonical count heads on their own MAE,
      3. keeps a CPU snapshot of every count head at ITS best validation MAE
         (other heads follow the monitored best step),
      4. requests an early stop when the monitored MAE (mean over the headline
         `count` heads) has not improved by `min_delta` for `patience` evals,
      5. at the end of training restores the best snapshots into the heads.
    """

    def __init__(self, patience=5, min_delta=0.25, stop=True):
        self.trainer = None  # set after trainer construction
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.stop = bool(stop)
        self.best = float("inf")          # monitored (mean over headline heads)
        self.best_step = None
        self.bad_evals = 0
        self.best_head_mae = {}           # name -> best own MAE
        self.best_head_step = {}          # name -> step of that best
        self._state = {}                  # name -> cpu state_dict at best
        self.history = []                 # [(step, monitored_mae)]

    # -- helpers -----------------------------------------------------------
    def _heads(self):
        m = self.trainer.model
        return (m.module if hasattr(m, "module") else m).heads

    def _snapshot(self, name):
        self._state[name] = {k: v.detach().cpu().clone() for k, v in self._heads()[name].state_dict().items()}

    def _headline(self, per_head):
        specs = self.trainer.probe_specs
        return [n for n in per_head if specs[n].get("base_probe_name", n) == "count"]

    # -- hooks -------------------------------------------------------------
    def on_evaluate(self, args, state, control, **kwargs):
        import math
        tr = self.trainer
        per_head = tr.consume_val_mae()
        if not per_head:
            return control
        step = int(state.global_step)

        for name, sched in tr._extra_schedulers.items():
            v = per_head.get(name)
            if v is not None and math.isfinite(v):
                sched.step(v)

        for name, v in per_head.items():
            if math.isfinite(v) and v < self.best_head_mae.get(name, float("inf")):
                self.best_head_mae[name], self.best_head_step[name] = v, step
                self._snapshot(name)

        headline = [n for n in self._headline(per_head) if math.isfinite(per_head[n])]
        if not headline:
            return control
        mon = sum(per_head[n] for n in headline) / len(headline)
        self.history.append((step, mon))
        if mon < self.best - self.min_delta:
            self.best, self.best_step, self.bad_evals = mon, step, 0
            for name in self._heads().keys():          # heads without a val MAE follow the best step
                if name not in per_head:
                    self._snapshot(name)
        else:
            self.bad_evals += 1
        if is_main_process():
            print(f"[val] step {step}: monitored MAE {mon:.2f} (best {self.best:.2f} @ {self.best_step}); "
                  f"{self.bad_evals}/{self.patience} evals without improvement")
        if self.stop and self.bad_evals >= self.patience:
            if is_main_process():
                print(f"[early-stop] no improvement > {self.min_delta} for {self.patience} evals; stopping at step {step}")
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self._state:
            return control
        heads = self._heads()
        for name, sd in self._state.items():
            heads[name].load_state_dict(sd)
        if is_main_process():
            print(f"[val] restored {len(self._state)} heads to their own best validation step "
                  f"(monitored MAE {self.best:.2f}, first reached at step {self.best_step})")
            import json
            out = {"best_monitored_mae": self.best, "best_step": self.best_step, "stopped_step": int(state.global_step),
                   "patience": self.patience, "min_delta": self.min_delta,
                   "best_head_step": self.best_head_step, "best_head_val_mae": self.best_head_mae,
                   "history": self.history}
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, "early_stopping.json"), "w") as f:
                    json.dump(out, f, indent=1)
            except OSError as exc:
                print(f"[warning] could not write early_stopping.json: {exc}")
        return control


class MAEPerBinCallback(TrainerCallback):
    """
    Callback that triggers MAE per bin computation and logging periodically based on steps.
    """

    def __init__(self, log_every_n_steps=100):
        """
        Args:
            log_every_n_steps: How often to log MAE per bin (in steps)
        """
        self.trainer = None  # Will be set after trainer instantiation
        self.log_every_n_steps = log_every_n_steps

    def on_evaluate(self, args, state, control, **kwargs):
        """Trigger trainer's logging method after evaluation."""
        if not is_main_process():
            return control

        # Only log every N steps
        if state.global_step % self.log_every_n_steps == 0:
            if self.trainer and hasattr(self.trainer, 'on_step_end'):
                self.trainer.on_step_end(state.global_step)
        return control


class ExampleDisplayCallback(TrainerCallback):
    """
    Callback that displays example predictions every N steps.
    Shows token-by-token predictions vs targets for remaining token counts.
    """

    def __init__(self, tokenizer, eval_dataset, display_every_n_steps=100, generation_config=None, num_examples=3):
        """
        Args:
            tokenizer: The tokenizer for decoding tokens
            eval_dataset: Dataset to sample from
            display_every_n_steps: How often to display predictions (in steps)
            generation_config: Dictionary with generation parameters (max_new_tokens, do_sample, etc.)
            num_examples: Number of examples to display each time (default: 3)
        """
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.display_every_n_steps = display_every_n_steps
        self.generation_config = generation_config or {}
        self.num_examples = num_examples

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called after evaluation completes."""
        # Only run on main process to avoid duplicate logging
        if not is_main_process():
            return control

        # Only display every N steps
        if state.global_step % self.display_every_n_steps == 0:
            for i in range(self.num_examples):
                self._display_predictions(model, state.global_step, state.global_step)

        return control

    def _display_predictions(self, model, step, global_step):
        """Display predictions for a random evaluation sample."""
        # Unwrap model if it's wrapped in DistributedDataParallel
        unwrapped_model = model.module if hasattr(model, 'module') else model

        # Get random sample from eval dataset
        idx = random.randint(0, len(self.eval_dataset) - 1)
        eval_sample = self.eval_dataset[idx]

        # Tokenize only the prompt (no completion)
        tokenized = tokenize_function(
            {
                "prompt": [eval_sample["prompt"]],
            },
            self.tokenizer,
        )

        # Convert to tensors and move to model device
        device = next(model.parameters()).device
        prompt_input_ids = torch.tensor([tokenized["input_ids"][0]]).to(device)
        prompt_attention_mask = torch.tensor([tokenized["attention_mask"][0]]).to(device)

        # Trim padding from prompt before generation
        # With left-padding, extract the rightmost tokens (actual prompt)
        prompt_len = prompt_attention_mask.sum().item()
        prompt_input_ids_trimmed = prompt_input_ids[:, -prompt_len:] if prompt_len > 0 else prompt_input_ids[:, :0]
        prompt_attention_mask_trimmed = prompt_attention_mask[:, -prompt_len:] if prompt_len > 0 else prompt_attention_mask[:, :0]

        # Generate completion from the model
        model.eval()
        with torch.no_grad():
            # Build generation kwargs using utility function
            gen_kwargs = build_generation_kwargs(
                input_ids=prompt_input_ids_trimmed,
                attention_mask=prompt_attention_mask_trimmed,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                include_hidden_states=False,
            )

            # Generate using the model (with trimmed prompt)
            generated_ids = unwrapped_model.generate(**gen_kwargs)

            # Extract completion using utility function (no BOS check for callback)
            completion_ids, has_eos, _ = extract_completion_with_eos(
                generated_ids=generated_ids[0],
                prompt_length=prompt_len,
                eos_token_id=self.tokenizer.eos_token_id,
                check_bos=False,  # Simpler for callback
            )

            # Concatenate prompt + completion
            # With left padding, extract the rightmost prompt_len tokens (actual prompt)
            prompt_ids = prompt_input_ids[0, -prompt_len:] if prompt_len > 0 else prompt_input_ids[0, :0]
            full_sequence = torch.cat([prompt_ids, completion_ids])
            full_length = len(full_sequence)

            # Compute remaining_token_counts using utility function
            remaining_counts = compute_remaining_token_targets(
                sequence_length=full_length,
                prompt_length=prompt_len,
                has_eos=has_eos,
                device=device,
            )

            # Create labels (for display compatibility)
            labels = torch.full((full_length,), -100, dtype=torch.long, device=device)
            labels[prompt_len:full_length] = full_sequence[prompt_len:full_length]

            # Pad to match expected size
            max_length = 512
            if full_length < max_length:
                padding_length = max_length - full_length
                padded_input_ids = torch.cat([
                    full_sequence,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
                ])
                padded_attention_mask = torch.cat([
                    torch.ones(full_length, dtype=torch.long, device=device),
                    torch.zeros(padding_length, dtype=torch.long, device=device)
                ])
                padded_remaining_counts = torch.cat([
                    remaining_counts,
                    torch.full((padding_length,), -100, dtype=torch.long, device=device)
                ])
                padded_labels = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=torch.long, device=device)
                ])
            else:
                padded_input_ids = full_sequence[:max_length]
                padded_attention_mask = torch.ones(max_length, dtype=torch.long, device=device)
                padded_remaining_counts = remaining_counts[:max_length]
                padded_labels = labels[:max_length]

            # Add batch dimension
            input_ids = padded_input_ids.unsqueeze(0)
            attention_mask = padded_attention_mask.unsqueeze(0)
            remaining_counts = padded_remaining_counts.unsqueeze(0)
            labels = padded_labels.unsqueeze(0)

            # Run forward pass to get predictions + hidden states (for all probes)
            outputs = unwrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                remaining_token_counts=remaining_counts,
                output_hidden_states=True,
            )

            # Apply every hidden-states-based probe head (only in callback2, which has probe_model)
            probe_outputs = {}
            if hasattr(self, "probe_model") and outputs.hidden_states is not None:
                if unwrapped_model.hidden_layer == "all":
                    hidden_states = torch.cat(outputs.hidden_states, dim=-1)
                else:
                    hidden_states = outputs.hidden_states[unwrapped_model.hidden_layer]
                hidden_states_fp32 = hidden_states.float()
                for name, head in self.probe_model.heads.items():
                    # Skip baselines (constant/position inputs) — only hidden_states-based probes
                    if head.in_features == hidden_states_fp32.shape[-1]:
                        probe_outputs[name] = head(hidden_states_fp32)[0]  # [seq, out_dim]

        model.train()

        # Get predictions
        token_count_predictions = outputs.token_count_predictions[0].cpu().tolist()
        input_ids_list = input_ids[0].cpu().tolist()
        attention_mask_list = attention_mask[0].cpu().tolist()
        labels_list = labels[0].cpu().tolist()
        remaining_counts_list = remaining_counts[0].cpu().tolist()

        # Percentage probe predictions (if present)
        pct_preds_list = None
        if "percentage" in probe_outputs:
            pct_preds_list = probe_outputs["percentage"].squeeze(-1).cpu().tolist()

        # Classification probes: derived percentage = sum(softmax(logits) * anchors)
        cls_derived_preds = {}
        cls_names = sorted([n for n in probe_outputs if n.startswith("cls_")])
        for name in cls_names:
            logits = probe_outputs[name]
            num_bins = logits.shape[-1]
            anchors = torch.linspace(0, 1, num_bins, device=logits.device)
            probs = torch.softmax(logits, dim=-1)
            cls_derived_preds[name] = (probs * anchors).sum(dim=-1).cpu().tolist()

        # Completion length for computing ground-truth percentage per position
        completion_length_total = len(completion_ids) if has_eos else 0

        # Decode the generated completion for display
        generated_completion_text = self.tokenizer.decode(completion_ids.cpu(), skip_special_tokens=True)

        # Decode tokens
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids_list]

        # Print header
        print("\n" + "=" * 100)
        print(f"📊 EXAMPLE PREDICTION - Step {step}")
        print("=" * 100)
        print(f"Prompt: {eval_sample['prompt']}")
        if not has_eos:
            print(f"⚠️  WARNING: Generation was cut off at max_new_tokens (no EOS found)")
        print(f"\nPrompt tokens: {prompt_len}, Completion tokens: {len(completion_ids)}, Total: {full_length}")
        print(f"Loss: {outputs.loss.item():.4f}")
        # Build header dynamically based on available probes
        base_cols = [
            ("Token", 20), ("Type", 7),
            ("cnt_pred", 9), ("cnt_tgt", 8), ("cnt_err", 8), ("cnt_mape", 9), ("cnt_smape", 9),
            ("pct_gt", 8),
        ]
        extra_cols = []
        if pct_preds_list is not None:
            extra_cols += [("pct_pred", 9), ("pct_err", 8)]
        for name in cls_names:
            extra_cols += [(f"{name}_pct", 9), (f"{name}_err", 8)]
        all_cols = base_cols + extra_cols

        header = " ".join(f"{title:<{w}}" for title, w in all_cols)
        width = len(header)
        print("\n" + "-" * width)
        print(header)
        print("-" * width)

        # Prepare data for wandb table
        table_data = []
        table_columns = [c[0] for c in all_cols]

        # Count prompt and completion tokens for display
        prompt_count = 0
        completion_count = 0

        # Display token-by-token
        for i, (token, pred, target, label, att_mask) in enumerate(
            zip(tokens, token_count_predictions, remaining_counts_list, labels_list, attention_mask_list)
        ):
            # Skip padding positions (where attention_mask is 0)
            if att_mask == 0:
                break

            # Token type based on position (not just target == -100)
            is_prompt_token = i < prompt_len
            token_display = repr(token)[:18]

            # Build row with placeholders
            row_display = {c: "-" for c, _ in all_cols}
            row_values = {c: None for c in table_columns}
            row_display["Token"] = token_display

            if is_prompt_token:
                row_display["Type"] = "PROMPT"
                prompt_count += 1
            else:
                row_display["Type"] = "COMPL"
                completion_count += 1
                row_display["cnt_pred"] = f"{pred:.2f}"
                row_values["cnt_pred"] = pred

                # Ground-truth percentage (only when completion length is known, i.e. has_eos)
                pct_gt = None
                if completion_length_total > 1 and target != -100:
                    pct_gt = target / (completion_length_total - 1)
                    row_display["pct_gt"] = f"{pct_gt:.3f}"
                    row_values["pct_gt"] = pct_gt
                elif completion_length_total == 1 and target != -100:
                    pct_gt = 0.0  # single-token completion
                    row_display["pct_gt"] = f"{pct_gt:.3f}"
                    row_values["pct_gt"] = pct_gt

                if target != -100:
                    row_display["cnt_tgt"] = f"{target}"
                    row_values["cnt_tgt"] = target
                    err = abs(pred - target)
                    row_display["cnt_err"] = f"{err:.2f}"
                    row_values["cnt_err"] = err
                    mape = err / target if target != 0 else err
                    row_display["cnt_mape"] = f"{mape:.3f}"
                    row_values["cnt_mape"] = mape
                    denom = (abs(pred) + abs(target)) / 2.0
                    smape = err / denom if denom != 0 else err
                    row_display["cnt_smape"] = f"{smape:.3f}"
                    row_values["cnt_smape"] = smape

                # Percentage probe
                if pct_preds_list is not None:
                    pct_pred_val = pct_preds_list[i]
                    row_display["pct_pred"] = f"{pct_pred_val:.3f}"
                    row_values["pct_pred"] = pct_pred_val
                    if pct_gt is not None:
                        pct_err = abs(pct_pred_val - pct_gt)
                        row_display["pct_err"] = f"{pct_err:.3f}"
                        row_values["pct_err"] = pct_err

                # Classification probes (derived percentage)
                for name in cls_names:
                    derived = cls_derived_preds[name][i]
                    row_display[f"{name}_pct"] = f"{derived:.3f}"
                    row_values[f"{name}_pct"] = derived
                    if pct_gt is not None:
                        err_val = abs(derived - pct_gt)
                        row_display[f"{name}_err"] = f"{err_val:.3f}"
                        row_values[f"{name}_err"] = err_val

            print(" ".join(f"{row_display[c]:<{w}}" for c, w in all_cols))
            table_data.append([row_values[c] if c != "Token" else token_display for c in table_columns])

        # Add summary line
        print("-" * 100)
        print(f"Total tokens displayed: {prompt_count} prompt + {completion_count} completion = {prompt_count + completion_count} total")

        # Calculate metrics for completion tokens only
        completion_predictions = []
        completion_targets = []
        for pred, target in zip(token_count_predictions, remaining_counts_list):
            if target != -100:
                completion_predictions.append(pred)
                completion_targets.append(target)

        if completion_predictions:
            errors = [abs(p - t) for p, t in zip(completion_predictions, completion_targets)]
            mae = sum(errors) / len(errors)
            max_error = max(errors)
            completion_length = len(completion_predictions)
            mape = [e / t if t != 0 else e for e, t in zip(errors, completion_targets)]
            mape = sum(mape) / len(mape)

            print("-" * 100)
            print(f"Completion Metrics:")
            print(f"  Mean Absolute Error (MAE): {mae:.4f}")
            print(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f}")
            print(f"  Max Error: {max_error:.4f}")
            print(f"  Completion Length: {completion_length} tokens")
            print("=" * 100 + "\n")



class ExampleDisplayCallback2(TrainerCallback):
    """
    Callback that displays example predictions every N steps.
    Shows token-by-token predictions vs targets for remaining token counts.
    """

    def __init__(self, base_model, probe_model, tokenizer, eval_dataset, display_every_n_steps=100, generation_config=None, num_examples=3):
        """
        Args:
            tokenizer: The tokenizer for decoding tokens
            eval_dataset: Dataset to sample from
            display_every_n_steps: How often to display predictions (in steps)
            generation_config: Dictionary with generation parameters (max_new_tokens, do_sample, etc.)
            num_examples: Number of examples to display each time (default: 3)
        """
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.display_every_n_steps = display_every_n_steps
        self.generation_config = generation_config or {}
        self.num_examples = num_examples
        self.base_model = base_model
        self.probe_model = probe_model
        self.model = ModelWithTokenCountHead(base_model, hidden_layer="all")
        self.model.token_count_head.load_state_dict(probe_model.token_count_head.state_dict())

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called after evaluation completes."""
        # Only run on main process to avoid duplicate logging
        if not is_main_process():
            return control

        # Only display every N steps
        if state.global_step % self.display_every_n_steps == 0:
            for i in range(self.num_examples):
                self._display_predictions(model, state.global_step, state.global_step)

        return control

    def _display_predictions(self, model, step, global_step):
        """Display predictions for a random evaluation sample."""
        # Unwrap model if it's wrapped in DistributedDataParallel
        unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Get random sample from eval dataset
        idx = random.randint(0, len(self.eval_dataset) - 1)
        eval_sample = self.eval_dataset[idx]

        # Tokenize only the prompt (no completion)
        tokenized = tokenize_function(
            {
                "prompt": [eval_sample["prompt"]],
            },
            self.tokenizer,
        )

        # Convert to tensors and move to model device
        device = next(model.parameters()).device
        prompt_input_ids = torch.tensor([tokenized["input_ids"][0]]).to(device)
        prompt_attention_mask = torch.tensor([tokenized["attention_mask"][0]]).to(device)
        #unwrapped_model.to(device)

        # Trim padding from prompt before generation
        # With left-padding, extract the rightmost tokens (actual prompt)
        prompt_len = prompt_attention_mask.sum().item()
        prompt_input_ids_trimmed = prompt_input_ids[:, -prompt_len:] if prompt_len > 0 else prompt_input_ids[:, :0]
        prompt_attention_mask_trimmed = prompt_attention_mask[:, -prompt_len:] if prompt_len > 0 else prompt_attention_mask[:, :0]

        # Generate completion from the model
        model.eval()
        with torch.no_grad():
            # Build generation kwargs using utility function
            gen_kwargs = build_generation_kwargs(
                input_ids=prompt_input_ids_trimmed,
                attention_mask=prompt_attention_mask_trimmed,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                include_hidden_states=False,
            )

            # Generate using the model (with trimmed prompt)
            generated_ids = unwrapped_model.generate(**gen_kwargs)

            # Extract completion using utility function (no BOS check for callback)
            completion_ids, has_eos, _ = extract_completion_with_eos(
                generated_ids=generated_ids[0],
                prompt_length=prompt_len,
                eos_token_id=self.tokenizer.eos_token_id,
                check_bos=False,  # Simpler for callback
            )

            # Concatenate prompt + completion
            # With left padding, extract the rightmost prompt_len tokens (actual prompt)
            prompt_ids = prompt_input_ids[0, -prompt_len:] if prompt_len > 0 else prompt_input_ids[0, :0]
            full_sequence = torch.cat([prompt_ids, completion_ids])
            full_length = len(full_sequence)

            # Compute remaining_token_counts using utility function
            remaining_counts = compute_remaining_token_targets(
                sequence_length=full_length,
                prompt_length=prompt_len,
                has_eos=has_eos,
                device=device,
            )

            # Create labels (for display compatibility)
            labels = torch.full((full_length,), -100, dtype=torch.long, device=device)
            labels[prompt_len:full_length] = full_sequence[prompt_len:full_length]

            # Pad to match expected size
            max_length = 512
            if full_length < max_length:
                padding_length = max_length - full_length
                padded_input_ids = torch.cat([
                    full_sequence,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
                ])
                padded_attention_mask = torch.cat([
                    torch.ones(full_length, dtype=torch.long, device=device),
                    torch.zeros(padding_length, dtype=torch.long, device=device)
                ])
                padded_remaining_counts = torch.cat([
                    remaining_counts,
                    torch.full((padding_length,), -100, dtype=torch.long, device=device)
                ])
                padded_labels = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=torch.long, device=device)
                ])
            else:
                padded_input_ids = full_sequence[:max_length]
                padded_attention_mask = torch.ones(max_length, dtype=torch.long, device=device)
                padded_remaining_counts = remaining_counts[:max_length]
                padded_labels = labels[:max_length]

            # Add batch dimension
            input_ids = padded_input_ids.unsqueeze(0)
            attention_mask = padded_attention_mask.unsqueeze(0)
            remaining_counts = padded_remaining_counts.unsqueeze(0)
            labels = padded_labels.unsqueeze(0)

            # Run forward pass to get predictions + hidden states (for all probes)
            outputs = unwrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                remaining_token_counts=remaining_counts,
                output_hidden_states=True,
            )

            # Apply every hidden-states-based probe head (only in callback2, which has probe_model)
            probe_outputs = {}
            if hasattr(self, "probe_model") and outputs.hidden_states is not None:
                if unwrapped_model.hidden_layer == "all":
                    hidden_states = torch.cat(outputs.hidden_states, dim=-1)
                else:
                    hidden_states = outputs.hidden_states[unwrapped_model.hidden_layer]
                hidden_states_fp32 = hidden_states.float()
                for name, head in self.probe_model.heads.items():
                    # Skip baselines (constant/position inputs) — only hidden_states-based probes
                    if head.in_features == hidden_states_fp32.shape[-1]:
                        probe_outputs[name] = head(hidden_states_fp32)[0]  # [seq, out_dim]

        model.train()

        # Get predictions
        token_count_predictions = outputs.token_count_predictions[0].cpu().tolist()
        input_ids_list = input_ids[0].cpu().tolist()
        attention_mask_list = attention_mask[0].cpu().tolist()
        labels_list = labels[0].cpu().tolist()
        remaining_counts_list = remaining_counts[0].cpu().tolist()

        # Percentage probe predictions (if present)
        pct_preds_list = None
        if "percentage" in probe_outputs:
            pct_preds_list = probe_outputs["percentage"].squeeze(-1).cpu().tolist()

        # Classification probes: derived percentage = sum(softmax(logits) * anchors)
        cls_derived_preds = {}
        cls_names = sorted([n for n in probe_outputs if n.startswith("cls_")])
        for name in cls_names:
            logits = probe_outputs[name]
            num_bins = logits.shape[-1]
            anchors = torch.linspace(0, 1, num_bins, device=logits.device)
            probs = torch.softmax(logits, dim=-1)
            cls_derived_preds[name] = (probs * anchors).sum(dim=-1).cpu().tolist()

        # Completion length for computing ground-truth percentage per position
        completion_length_total = len(completion_ids) if has_eos else 0

        # Decode the generated completion for display
        generated_completion_text = self.tokenizer.decode(completion_ids.cpu(), skip_special_tokens=True)

        # Decode tokens
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids_list]

        # Print header
        print("\n" + "=" * 100)
        print(f"📊 EXAMPLE PREDICTION - Step {step}")
        print("=" * 100)
        print(f"Prompt: {eval_sample['prompt']}")
        if not has_eos:
            print(f"⚠️  WARNING: Generation was cut off at max_new_tokens (no EOS found)")
        print(f"\nPrompt tokens: {prompt_len}, Completion tokens: {len(completion_ids)}, Total: {full_length}")
        print(f"Loss: {outputs.loss.item():.4f}")
        # Build header dynamically based on available probes
        base_cols = [
            ("Token", 20), ("Type", 7),
            ("cnt_pred", 9), ("cnt_tgt", 8), ("cnt_err", 8), ("cnt_mape", 9), ("cnt_smape", 9),
            ("pct_gt", 8),
        ]
        extra_cols = []
        if pct_preds_list is not None:
            extra_cols += [("pct_pred", 9), ("pct_err", 8)]
        for name in cls_names:
            extra_cols += [(f"{name}_pct", 9), (f"{name}_err", 8)]
        all_cols = base_cols + extra_cols

        header = " ".join(f"{title:<{w}}" for title, w in all_cols)
        width = len(header)
        print("\n" + "-" * width)
        print(header)
        print("-" * width)

        # Prepare data for wandb table
        table_data = []
        table_columns = [c[0] for c in all_cols]

        # Count prompt and completion tokens for display
        prompt_count = 0
        completion_count = 0

        # Display token-by-token
        for i, (token, pred, target, label, att_mask) in enumerate(
            zip(tokens, token_count_predictions, remaining_counts_list, labels_list, attention_mask_list)
        ):
            # Skip padding positions (where attention_mask is 0)
            if att_mask == 0:
                break

            # Token type based on position (not just target == -100)
            is_prompt_token = i < prompt_len
            token_display = repr(token)[:18]

            # Build row with placeholders
            row_display = {c: "-" for c, _ in all_cols}
            row_values = {c: None for c in table_columns}
            row_display["Token"] = token_display

            if is_prompt_token:
                row_display["Type"] = "PROMPT"
                prompt_count += 1
            else:
                row_display["Type"] = "COMPL"
                completion_count += 1
                row_display["cnt_pred"] = f"{pred:.2f}"
                row_values["cnt_pred"] = pred

                # Ground-truth percentage (only when completion length is known, i.e. has_eos)
                pct_gt = None
                if completion_length_total > 1 and target != -100:
                    pct_gt = target / (completion_length_total - 1)
                    row_display["pct_gt"] = f"{pct_gt:.3f}"
                    row_values["pct_gt"] = pct_gt
                elif completion_length_total == 1 and target != -100:
                    pct_gt = 0.0  # single-token completion
                    row_display["pct_gt"] = f"{pct_gt:.3f}"
                    row_values["pct_gt"] = pct_gt

                if target != -100:
                    row_display["cnt_tgt"] = f"{target}"
                    row_values["cnt_tgt"] = target
                    err = abs(pred - target)
                    row_display["cnt_err"] = f"{err:.2f}"
                    row_values["cnt_err"] = err
                    mape = err / target if target != 0 else err
                    row_display["cnt_mape"] = f"{mape:.3f}"
                    row_values["cnt_mape"] = mape
                    denom = (abs(pred) + abs(target)) / 2.0
                    smape = err / denom if denom != 0 else err
                    row_display["cnt_smape"] = f"{smape:.3f}"
                    row_values["cnt_smape"] = smape

                # Percentage probe
                if pct_preds_list is not None:
                    pct_pred_val = pct_preds_list[i]
                    row_display["pct_pred"] = f"{pct_pred_val:.3f}"
                    row_values["pct_pred"] = pct_pred_val
                    if pct_gt is not None:
                        pct_err = abs(pct_pred_val - pct_gt)
                        row_display["pct_err"] = f"{pct_err:.3f}"
                        row_values["pct_err"] = pct_err

                # Classification probes (derived percentage)
                for name in cls_names:
                    derived = cls_derived_preds[name][i]
                    row_display[f"{name}_pct"] = f"{derived:.3f}"
                    row_values[f"{name}_pct"] = derived
                    if pct_gt is not None:
                        err_val = abs(derived - pct_gt)
                        row_display[f"{name}_err"] = f"{err_val:.3f}"
                        row_values[f"{name}_err"] = err_val

            print(" ".join(f"{row_display[c]:<{w}}" for c, w in all_cols))
            table_data.append([row_values[c] if c != "Token" else token_display for c in table_columns])

        # Add summary line
        print("-" * 100)
        print(f"Total tokens displayed: {prompt_count} prompt + {completion_count} completion = {prompt_count + completion_count} total")

        # Calculate metrics for completion tokens only
        completion_predictions = []
        completion_targets = []
        for pred, target in zip(token_count_predictions, remaining_counts_list):
            if target != -100:
                completion_predictions.append(pred)
                completion_targets.append(target)

        if completion_predictions:
            errors = [abs(p - t) for p, t in zip(completion_predictions, completion_targets)]
            mae = sum(errors) / len(errors)
            max_error = max(errors)
            completion_length = len(completion_predictions)
            mape = [e / t if t != 0 else e for e, t in zip(errors, completion_targets)]
            mape = sum(mape) / len(mape)

            print("-" * 100)
            print(f"Completion Metrics:")
            print(f"  Mean Absolute Error (MAE): {mae:.4f}")
            print(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f}")
            print(f"  Max Error: {max_error:.4f}")
            print(f"  Completion Length: {completion_length} tokens")
            print("=" * 100 + "\n")
