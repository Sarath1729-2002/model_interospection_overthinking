"""
Dataset class for loading pre-extracted hidden states for offline probe training.
"""

import copy
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


def split_indices(n: int, val_fraction: float, split_seed: int = 0):
    """Deterministic train/validation split of `n` samples.

    The split depends only on (n, val_fraction, split_seed) - NOT on the
    training seed - so every probe seed and the ridge cross-check see the same
    validation samples. Returns (train_idx, val_idx) as sorted int lists.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    n_val = min(max(1, int(round(n * val_fraction))), n - 1)
    perm = np.random.RandomState(int(split_seed)).permutation(n)
    val = sorted(int(i) for i in perm[:n_val])
    train = sorted(int(i) for i in perm[n_val:])
    return train, val


class HiddenStatesDataset(Dataset):
    """
    Dataset that loads pre-extracted hidden states from disk.

    This allows offline probe training without running the base model.

    Three loading modes are supported:
      * Single-layer (default): the requested layer is returned as a 2-D
        tensor [seq_len, hidden_size]. Selected via `hidden_layer` (string,
        supporting negative indexing or the literal "all" for concatenated).
      * Concatenated all-layer: `hidden_layer="all"` returns a 2-D tensor
        of shape [seq_len, num_layers * hidden_size].
      * Multi-layer (per-layer probe sweep): `layers_to_train=[i, j, ...]`
        returns a 3-D tensor of shape [seq_len, len(layers_to_train), hidden_size]
        with one slice per requested layer, in the user-given order. Used by
        the multi-layer training mode in train_offline.py to amortize disk I/O
        across many per-layer probes.
    """

    def __init__(
        self,
        hidden_states_dir: str,
        split: str = "train",
        hidden_layer: str = "-1",
        layers_to_train: List[int] | None = None,
    ):
        """
        Args:
            hidden_states_dir: Directory containing saved hidden states
            split: 'train' or 'eval'
            hidden_layer: Which layer to use ('-1' for last, 'all' for concatenate all).
                Ignored when `layers_to_train` is set.
            layers_to_train: Optional list of layer indices (negative indices and the
                string "all" are accepted at the builder level; this constructor
                expects a resolved list of non-negative ints, or None for single-layer mode).
        """
        self.hidden_states_dir = Path(hidden_states_dir)
        self.split = split
        self.hidden_layer = hidden_layer
        self.layers_to_train = layers_to_train  # None = single-layer mode

        # Find all sample files
        split_dir = self.hidden_states_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        self.sample_files = sorted(list(split_dir.glob("sample_*.safetensors")))

        if len(self.sample_files) == 0:
            raise ValueError(f"No sample files found in {split_dir}")

        print(f"Loaded {len(self.sample_files)} samples from {split_dir}")

    def __len__(self):
        return len(self.sample_files)

    def subset(self, indices: Sequence[int]) -> "HiddenStatesDataset":
        """Shallow copy restricted to `indices` (same loading mode, no I/O)."""
        new = copy.copy(self)
        new.sample_files = [self.sample_files[int(i)] for i in indices]
        return new

    def __getitem__(self, idx):
        """Load and return a sample with hidden states."""
        sample_file = self.sample_files[idx]
        base_path = sample_file.with_suffix('')  # Remove .safetensors extension

        # Load tensors from safetensors
        tensors = load_file(str(sample_file))

        # Load metadata from JSON
        metadata_file = f"{base_path}.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Extract data
        prompt_ids = tensors["prompt_ids"]
        completion_ids = tensors["completion_ids"]
        full_sequence_ids = tensors["full_sequence_ids"]
        has_eos = metadata["has_eos"]
        prompt_length = metadata["prompt_length"]
        completion_length = metadata["completion_length"]
        num_hidden_layers = metadata["num_hidden_layers"]
        prompt = metadata["prompt"]
        completion = metadata["completion"]

        # Reconstruct hidden states list from individual layer tensors
        hidden_states_all_layers = [
            tensors[f"hidden_states_{i}"] for i in range(num_hidden_layers)
        ]

        # Select hidden states based on the loading mode.
        if self.layers_to_train is not None:
            # Multi-layer mode. Resolve any negative indices against the actual
            # number of layers in this sample, then stack along a new axis.
            resolved = [
                (L + num_hidden_layers) if L < 0 else L for L in self.layers_to_train
            ]
            for L in resolved:
                if L < 0 or L >= num_hidden_layers:
                    raise IndexError(
                        f"layers_to_train index {L} out of range for sample with "
                        f"num_hidden_layers={num_hidden_layers}"
                    )
            hidden_states = torch.stack(
                [hidden_states_all_layers[L] for L in resolved], dim=1
            )  # [seq_len, len(layers_to_train), hidden_size]
        elif self.hidden_layer == "all":
            # Concatenate all layers along hidden dimension
            hidden_states = torch.cat(hidden_states_all_layers, dim=-1)  # [seq_len, hidden_size * num_layers]
        else:
            # Use specific layer (supports negative indexing)
            layer_idx = int(self.hidden_layer)
            hidden_states = hidden_states_all_layers[layer_idx]  # [seq_len, hidden_size]

        # Compute remaining token counts (same logic as in trainer)
        seq_len = len(full_sequence_ids)
        remaining_counts = torch.full((seq_len,), -100, dtype=torch.long)

        # Only set targets for completion tokens if sequence has EOS
        if has_eos and seq_len > prompt_length:
            # Remaining tokens = total_completion_tokens - tokens_generated_so_far - 1,
            # with tokens_generated_so_far = i - prompt_length (vectorized).
            tokens_so_far = torch.arange(seq_len - prompt_length, dtype=torch.long)
            remaining_counts[prompt_length:] = completion_length - tokens_so_far - 1

        # Create attention mask
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        return {
            "input_ids": full_sequence_ids,
            "attention_mask": attention_mask,
            "remaining_token_counts": remaining_counts,
            "hidden_states": hidden_states,  # [seq_len, hidden_size]
            "has_eos": has_eos,
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "prompt": prompt,
            "completion": completion,
        }


class HiddenStatesDatasetBuilder:
    """
    Builder for HiddenStatesDataset that follows the same interface as other dataset builders.
    """

    def __init__(
        self,
        hidden_states_dir: str,
        hidden_layer: str = "-1",
        layers_to_train=None,
        **kwargs,
    ):
        """
        Args:
            hidden_states_dir: Directory containing saved hidden states
            hidden_layer: Which layer to use ('-1' for last, 'all' for concatenate all).
                Ignored when `layers_to_train` is set.
            layers_to_train: Optional list of layer indices (e.g. [0, 8, 16, 24, 31])
                or the string "all" to expand to every layer. When set, the dataset
                returns a 3-D hidden_states tensor [seq_len, len(layers), hidden_size]
                so a single training run can sweep all listed layers.
            **kwargs: Additional arguments (ignored, for config compatibility)
        """
        self.hidden_states_dir = hidden_states_dir
        self.hidden_layer = hidden_layer
        # Normalize layers_to_train at the builder level. Hydra may hand us a
        # ListConfig or a literal None; we resolve both here.
        if layers_to_train is None:
            self.layers_to_train = None
        elif isinstance(layers_to_train, str):
            if layers_to_train.strip().lower() == "all":
                self.layers_to_train = "all"  # resolved per-sample inside the dataset
            else:
                raise ValueError(
                    f"layers_to_train must be a list of ints or the literal string 'all'; got {layers_to_train!r}"
                )
        else:
            self.layers_to_train = [int(L) for L in layers_to_train]
        # Ignore other kwargs (like num_train_samples, etc.) for config compatibility

    def get_dataset(self, split: str, tokenizer=None):
        """
        Get dataset for the given split.

        Args:
            split: 'train' or 'validation'
            tokenizer: Not used for hidden states dataset (for interface compatibility)

        Returns:
            HiddenStatesDataset
        """
        # Map 'validation' to 'eval' for consistency with extract_hidden_states.py
        actual_split = "train" if split == "train" else "eval"

        # Resolve "all" to a concrete list by peeking at the first sample's
        # metadata. Done here (not in the builder) so we don't load tensors
        # before the dataset is constructed.
        layers_resolved = self.layers_to_train
        if layers_resolved == "all":
            split_dir = Path(self.hidden_states_dir) / actual_split
            first = sorted(split_dir.glob("sample_*.json"))[:1]
            if not first:
                raise ValueError(
                    f"layers_to_train='all' but {split_dir} has no samples to inspect for layer count."
                )
            with open(first[0], "r") as f:
                meta = json.load(f)
            layers_resolved = list(range(int(meta["num_hidden_layers"])))

        return HiddenStatesDataset(
            hidden_states_dir=self.hidden_states_dir,
            split=actual_split,
            hidden_layer=self.hidden_layer,
            layers_to_train=layers_resolved,
        )
