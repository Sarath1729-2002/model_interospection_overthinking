"""
Custom model with dual heads: language modeling + remaining token count prediction.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


def is_main_process():
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in [-1, 0]


def print_main(*args, **kwargs):
    """Print only from the main process."""
    if is_main_process():
        print(*args, **kwargs)


@dataclass
class DualHeadModelOutput(ModelOutput):
    """
    Output class for model with token count prediction probe.

    Args:
        loss: Token count prediction loss (MSE)
        logits: Language modeling logits (from frozen base model)
        token_count_predictions: Predicted remaining token counts
        hidden_states: Hidden states from the model
        attentions: Attention weights
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    token_count_predictions: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ModelWithTokenCountHead(nn.Module):
    """
    Wrapper around a frozen causal language model that adds a regression probe
    to predict the remaining number of tokens until end of sequence.

    Note: The base model is frozen and only the probe is trained.
    No language model loss is computed.
    """

    def __init__(self, base_model: AutoModelForCausalLM, hidden_layer: str | int = -1):
        """
        Args:
            base_model: The frozen base causal language model
            hidden_layer: Which hidden layer(s) to use for probe:
                         -1 = last layer, -2 = second-to-last, etc.
                         0 = first layer, 1 = second layer, etc.
                         "all" = concatenate all layers
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_layer = hidden_layer

        # Get hidden size from the model config
        hidden_size = base_model.config.hidden_size

        # Calculate probe input size
        if hidden_layer == "all":
            # num_hidden_layers + 1 for embeddings layer
            num_layers = base_model.config.num_hidden_layers + 1
            probe_input_size = hidden_size * num_layers
        else:
            probe_input_size = hidden_size

        # Simple linear probe: probe_input_size -> 1 (predicting remaining token count)
        # Keep in FP32 for numerical stability, even if base model is FP16
        self.token_count_head = nn.Linear(probe_input_size, 1)

        # Move token_count_head to the same device as base_model
        # This ensures compatibility with multi-GPU training
        if hasattr(base_model, 'device'):
            self.token_count_head = self.token_count_head.to(base_model.device)

        # Loss function for token count prediction (MSE for regression)
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,  # Kept for compatibility but not used
        remaining_token_counts: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> DualHeadModelOutput:
        """
        Forward pass for probe model (base model frozen).

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Kept for compatibility, not used (base model is frozen)
            remaining_token_counts: Remaining token counts for each position [batch_size, seq_len]
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict

        Returns:
            DualHeadModelOutput with token count loss and predictions
        """
        # Get outputs from frozen base model (no labels, no loss computation)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # We need hidden states for token count prediction
            return_dict=True,
            **kwargs
        )

        # Get the hidden states based on configuration
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor is [batch_size, seq_len, hidden_size]
        if self.hidden_layer == "all":
            # Concatenate all hidden states along the hidden dimension
            hidden_states = torch.cat(outputs.hidden_states, dim=-1)  # [batch_size, seq_len, hidden_size * num_layers]
        else:
            # Use specific layer (supports negative indexing)
            hidden_states = outputs.hidden_states[self.hidden_layer]

        # Convert to FP32 for token count head (for numerical stability)
        # The token count head stays in FP32 even during mixed precision training
        hidden_states_fp32 = hidden_states.float()

        # Predict remaining token counts for each position
        # Shape: [batch_size, seq_len, 1]
        token_count_logits = self.token_count_head(hidden_states_fp32)
        # Squeeze to get [batch_size, seq_len]
        token_count_predictions = token_count_logits.squeeze(-1)

        # Calculate loss
        loss = None

        if remaining_token_counts is not None:
            # Only calculate loss for non-masked positions
            # We use -100 to mask positions where we don't want to compute loss
            # (prompt tokens, padding tokens)
            remaining_counts_flat = remaining_token_counts.view(-1)
            predictions_flat = token_count_predictions.view(-1)

            # Find positions that are not masked (-100)
            active_positions = remaining_counts_flat != -100

            # Filter to only active positions
            active_predictions = predictions_flat[active_positions]
            # Convert labels to FP32 (probe works in FP32)
            active_labels = remaining_counts_flat[active_positions].float()

            # Calculate MSE loss for token count prediction only on completion tokens
            if len(active_predictions) > 0:
                loss = self.loss_fn(active_predictions, active_labels)
            else:
                # No active positions (all sequences truncated, no valid targets)
                # Return a zero loss that's connected to model parameters so gradients work
                # Use the token_count_head's weight parameter to ensure requires_grad=True
                loss = (self.token_count_head.weight.sum() * 0.0)

        return DualHeadModelOutput(
            loss=loss,
            logits=outputs.logits,
            token_count_predictions=token_count_predictions,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    def generate(self, *args, **kwargs):
        """Forward generate calls to the base model."""
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model and token count head."""
        # Save base model
        self.base_model.save_pretrained(save_directory, **kwargs)

        # Save token count head separately
        token_count_head_path = f"{save_directory}/token_count_head.pt"
        torch.save({
            'token_count_head': self.token_count_head.state_dict(),
        }, token_count_head_path)
        print_main(f"Token count head saved to {token_count_head_path}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model: AutoModelForCausalLM,
        hidden_layer: str | int = -1,
    ):
        """Load the model with token count head."""
        # Create instance with base model, respecting hidden layer selection
        instance = cls(base_model, hidden_layer=hidden_layer)

        # Load token count head if it exists
        token_count_head_path = f"{model_path}/token_count_head.pt"
        try:
            checkpoint = torch.load(token_count_head_path, map_location='cpu')
            instance.token_count_head.load_state_dict(checkpoint['token_count_head'])
            print_main(f"Token count head loaded from {token_count_head_path}")
        except FileNotFoundError:
            print_main(f"No token count head found at {token_count_head_path}, using randomly initialized head")

        return instance
