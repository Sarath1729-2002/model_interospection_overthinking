"""
Utility functions for text generation and target computation.

This module contains shared logic used by both the trainer and callbacks
to avoid code duplication.
"""

from typing import Optional

import torch


def build_generation_kwargs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: dict,
    pad_token_id: int,
    eos_token_id: int,
    include_hidden_states: bool = False
) -> dict:
    """
    Build generation kwargs from configuration.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        generation_config: Dictionary containing generation parameters
        pad_token_id: Padding token ID
        eos_token_id: End-of-sequence token ID
        include_hidden_states: Whether to include hidden states in generation output

    Returns:
        Dictionary of kwargs ready for model.generate()
    """
    # Extract generation parameters from config
    max_new_tokens = generation_config.get('max_new_tokens', 500)
    do_sample = generation_config.get('do_sample', False)
    temperature = generation_config.get('temperature', 1.0)
    top_p = generation_config.get('top_p', 1.0)
    top_k = generation_config.get('top_k', 50)

    # Build base generation kwargs
    gen_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'pad_token_id': pad_token_id,
        'eos_token_id': eos_token_id,
    }

    # Add hidden states if requested
    if include_hidden_states:
        gen_kwargs['output_hidden_states'] = True
        gen_kwargs['return_dict_in_generate'] = True

    # Optional pass-through. When absent we leave generate() to inherit whatever
    # the checkpoint's generation_config.json says (unchanged historical
    # behaviour). Some checkpoints (e.g. Qwen2.5-Instruct) ship a non-trivial
    # repetition_penalty there, so a model config can pin it explicitly.
    repetition_penalty = generation_config.get('repetition_penalty', None)
    if repetition_penalty is not None:
        gen_kwargs['repetition_penalty'] = float(repetition_penalty)

    # Set sampling parameters based on do_sample
    if do_sample:
        gen_kwargs['temperature'] = temperature
        gen_kwargs['top_p'] = top_p
        gen_kwargs['top_k'] = top_k
    else:
        # Explicitly disable sampling parameters when using greedy decoding
        # This prevents warnings if the model's generation_config has these set
        gen_kwargs['temperature'] = None
        gen_kwargs['top_p'] = None
        gen_kwargs['top_k'] = None

    return gen_kwargs


def extract_completion_with_eos(
    generated_ids: torch.Tensor,
    prompt_length: int,
    eos_token_id: int,
    check_bos: bool = False,
    bos_token_id: Optional[int] = None,
    original_prompt_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, bool, int]:
    """
    Extract completion tokens from generated sequence and detect EOS.

    Args:
        generated_ids: Generated token IDs [seq_len]
        prompt_length: Length of the original prompt
        eos_token_id: End-of-sequence token ID
        check_bos: Whether to check if BOS was added by .generate()
        bos_token_id: Beginning-of-sequence token ID (required if check_bos=True)
        original_prompt_ids: Original prompt tokens (required if check_bos=True)

    Returns:
        completion_ids: Extracted completion tokens
        has_eos: Whether EOS token was found
        actual_prompt_len: Actual prompt length (may differ if BOS was added)
    """
    actual_prompt_len = prompt_length

    # Check if .generate() added a BOS token
    if check_bos and bos_token_id is not None:
        if len(generated_ids) > 0 and generated_ids[0] == bos_token_id:
            # Check if BOS was in the original prompt
            if original_prompt_ids is None or len(original_prompt_ids) == 0 or original_prompt_ids[0] != bos_token_id:
                # BOS was added by .generate(), so actual prompt is 1 token longer
                actual_prompt_len = prompt_length + 1

    # Extract completion starting from where prompt ends
    completion_ids = generated_ids[actual_prompt_len:]

    # Find EOS token or use all generated tokens
    eos_positions = (completion_ids == eos_token_id).nonzero(as_tuple=True)[0]
    has_eos = len(eos_positions) > 0

    if has_eos:
        # Truncate at first EOS (inclusive)
        completion_end = eos_positions[0].item() + 1
        completion_ids = completion_ids[:completion_end]

    return completion_ids, has_eos, actual_prompt_len


def compute_percentage_targets(
    sequence_length: int,
    prompt_length: int,
    has_eos: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute remaining token percentage targets for each position.

    Target values:
    - Prompt tokens: -100 (ignored)
    - Completion tokens when has_eos=True: remaining fraction (1.0 → 0.0)
    - Completion tokens when has_eos=False: -100 (ignored, truncated sequence)

    Args:
        sequence_length: Total length of prompt + completion
        prompt_length: Length of the prompt
        has_eos: Whether the sequence has an EOS token (not truncated)
        device: Device to create tensor on

    Returns:
        Tensor of shape [sequence_length] with float target values in [0, 1]
    """
    targets = torch.full((sequence_length,), -100.0, dtype=torch.float32, device=device)
    if not has_eos:
        return targets
    completion_length = sequence_length - prompt_length
    if completion_length <= 0:
        return targets
    # Vectorized form of: targets[j] = (sequence_length - j - 1) / max(completion_length - 1, 1)
    # for j in [prompt_length, sequence_length). Division is done in float64 and
    # rounded to float32 at the end, exactly as the original scalar loop did.
    remaining = torch.arange(completion_length - 1, -1, -1, dtype=torch.float64, device=device)
    targets[prompt_length:] = (remaining / max(completion_length - 1, 1)).to(torch.float32)
    return targets


def compute_soft_class_targets(
    sequence_length: int,
    prompt_length: int,
    has_eos: bool,
    device: torch.device,
    sigma: float = 0.15,
    num_bins: int = 3,
    anchors: Optional[list[float]] = None,
) -> torch.Tensor:
    """
    Compute soft N-class targets for each position using Gaussian smoothing.

    Anchors are evenly spaced in [0, 1] representing normalized completion progress.
    Each completion token gets a soft probability distribution over bins based on
    its position as a percentage of total completion, smoothed with a Gaussian kernel.

    Args:
        sequence_length: Total length of prompt + completion
        prompt_length: Length of the prompt
        has_eos: Whether the sequence has an EOS token (not truncated)
        device: Device to create tensor on
        sigma: Gaussian kernel width (smaller = sharper distributions)
        num_bins: Number of classification bins
        anchors: Custom anchor positions in [0, 1]. Defaults to linspace(0, 1, num_bins)

    Returns:
        Tensor of shape [sequence_length, num_bins] with -100 for masked positions
    """
    if anchors is None:
        anchors = torch.linspace(0, 1, num_bins, device=device).tolist()
    anchors = torch.tensor(anchors, device=device)
    targets = torch.full((sequence_length, num_bins), -100.0, dtype=torch.float32, device=device)
    if not has_eos:
        return targets
    completion_length = sequence_length - prompt_length
    if completion_length <= 0:
        return targets
    # Vectorized form of the per-position loop: one [completion_length, num_bins]
    # computation instead of completion_length * num_bins tiny tensor ops.
    # `pct` is computed in float64 then rounded to float32, matching how the
    # original scalar `pct` was combined with the float32 anchors.
    pct = (
        torch.arange(completion_length, dtype=torch.float64, device=device)
        / max(completion_length - 1, 1)
    ).to(torch.float32)
    class_probs = torch.exp(-0.5 * ((pct.unsqueeze(1) - anchors.unsqueeze(0)) / sigma) ** 2)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
    targets[prompt_length:] = class_probs
    return targets


def generate_and_extract_hidden_states(
    base_model,
    hidden_layer,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    generation_config: dict,
    pad_token_id: int,
    eos_token_id: int,
    bos_token_id: Optional[int] = None,
) -> dict:
    """Generate completions and extract hidden states for the full sequences.

    Designed to be called from a Trainer's training_step / prediction_step.
    Behaves identically to the body of CustomTrainer._generate_and_compute_targets
    up through the forward-pass + layer-select step, then re-pads everything to
    the same shape the offline `hidden_states_collator` produces so the result
    can be consumed by either CustomTrainer (single-probe online) or
    MultiProbeOfflineTrainer (multi-probe streaming).

    Args:
        base_model: the frozen base LLM (NOT the probe wrapper). Must support
            .generate() and forward with output_hidden_states=True.
        hidden_layer: int (with negative indexing) or the literal string "all"
            to concatenate all layers along the hidden dimension.
        prompt_input_ids: [B, S_padded] left-padded prompt token ids.
        prompt_attention_mask: [B, S_padded] left-padded mask.
        generation_config: dict consumed by build_generation_kwargs.
        pad_token_id, eos_token_id: tokenizer special-token ids.
        bos_token_id: optional, used to detect a BOS prepended by .generate().

    Returns a dict (right-padded across the batch):
        - input_ids:              [B, max_seq] long, right-padded with pad_token_id
        - attention_mask:         [B, max_seq] long, right-padded with 0
        - hidden_states:          [B, max_seq, H_eff] float, right-padded with 0
        - remaining_token_counts: [B, max_seq] long, right-padded / masked with -100
        - has_eos:                [B] bool
        - prompt_length:          [B] long, actual prompt lengths post-BOS
        - completion_length:      [B] long, completion token counts
    """
    batch_size = prompt_input_ids.shape[0]
    device = prompt_input_ids.device

    # Per-item prompt lengths (excluding left padding).
    prompt_lengths_pre = prompt_attention_mask.sum(dim=1)

    # Trim padding and re-pad to the max prompt length in the batch.
    prompt_input_ids_list = []
    prompt_attention_mask_list = []
    for i in range(batch_size):
        plen = prompt_lengths_pre[i].item()
        prompt_input_ids_list.append(prompt_input_ids[i, -plen:] if plen > 0 else prompt_input_ids[i, :0])
        prompt_attention_mask_list.append(prompt_attention_mask[i, -plen:] if plen > 0 else prompt_attention_mask[i, :0])

    max_prompt_len = int(prompt_lengths_pre.max().item())
    prompt_input_ids_trimmed = torch.full((batch_size, max_prompt_len), pad_token_id, dtype=torch.long, device=device)
    prompt_attention_mask_trimmed = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=device)
    for i in range(batch_size):
        plen = prompt_lengths_pre[i].item()
        pad_needed = max_prompt_len - plen
        if pad_needed > 0:
            prompt_input_ids_trimmed[i, pad_needed:] = prompt_input_ids_list[i]
            prompt_attention_mask_trimmed[i, pad_needed:] = prompt_attention_mask_list[i]
        else:
            prompt_input_ids_trimmed[i] = prompt_input_ids_list[i]
            prompt_attention_mask_trimmed[i] = prompt_attention_mask_list[i]

    # Generate completions (no gradients needed for the base model).
    gen_kwargs = build_generation_kwargs(
        input_ids=prompt_input_ids_trimmed,
        attention_mask=prompt_attention_mask_trimmed,
        generation_config=generation_config,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        include_hidden_states=False,  # we redo a clean forward below
    )
    with torch.no_grad():
        generated_ids = base_model.generate(**gen_kwargs)
        # generated_ids may be a ModelOutput or a plain tensor depending on flags.
        if hasattr(generated_ids, "sequences"):
            generated_ids = generated_ids.sequences

    # Extract per-item completion + actual prompt length (handling BOS that
    # .generate() may have prepended).
    full_sequences = []
    sequence_lengths = []
    actual_prompt_lengths: list[int] = []
    sequence_has_eos: list[bool] = []
    for i in range(batch_size):
        plen = prompt_lengths_pre[i].item()
        full_generated = generated_ids[i]
        original_prompt_ids = prompt_input_ids[i, -plen:] if plen > 0 else prompt_input_ids[i, :0]
        completion_ids, has_eos, actual_prompt_len = extract_completion_with_eos(
            generated_ids=full_generated,
            prompt_length=plen,
            eos_token_id=eos_token_id,
            check_bos=bos_token_id is not None,
            bos_token_id=bos_token_id,
            original_prompt_ids=original_prompt_ids,
        )
        actual_prompt_ids = full_generated[:actual_prompt_len]
        full_seq = torch.cat([actual_prompt_ids, completion_ids])
        full_sequences.append(full_seq)
        sequence_lengths.append(int(len(full_seq)))
        actual_prompt_lengths.append(int(actual_prompt_len))
        sequence_has_eos.append(bool(has_eos))

    # Run a clean forward pass on left-padded full sequences to obtain reliable
    # hidden states (the per-step .generate() hidden states have caching gotchas).
    max_seq_len = max(sequence_lengths)
    padded_seqs_left = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=device)
    attn_left = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    for i, (seq, slen) in enumerate(zip(full_sequences, sequence_lengths)):
        pad_needed = max_seq_len - slen
        padded_seqs_left[i, pad_needed:] = seq
        attn_left[i, pad_needed:] = 1

    # Run the forward pass under no_grad: the base model is frozen, so its
    # autograd graph is "empty" — keeping the per-layer attention / MLP
    # intermediates around would just waste VRAM. The probe heads later read
    # `batch_hidden_left` (requires_grad=False), and their own `head(...)` call
    # creates a fresh autograd graph from the head's weights, which is what
    # backward needs.
    with torch.no_grad():
        outputs = base_model(
            input_ids=padded_seqs_left,
            attention_mask=attn_left,
            output_hidden_states=True,
            return_dict=True,
        )
        if hidden_layer == "all":
            batch_hidden_left = torch.cat(outputs.hidden_states, dim=-1)
        else:
            batch_hidden_left = outputs.hidden_states[int(hidden_layer)]

    hidden_size_eff = batch_hidden_left.shape[-1]

    # Re-pack everything right-padded (matching the offline collator's layout).
    input_ids_right = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask_right = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    hidden_states_right = torch.zeros((batch_size, max_seq_len, hidden_size_eff), dtype=batch_hidden_left.dtype, device=device)
    remaining_counts = torch.full((batch_size, max_seq_len), -100, dtype=torch.long, device=device)

    for i, (seq, slen) in enumerate(zip(full_sequences, sequence_lengths)):
        # Strip left padding from the per-item hidden states.
        item_hidden = batch_hidden_left[i, -slen:]  # [slen, H_eff]

        input_ids_right[i, :slen] = seq
        attention_mask_right[i, :slen] = 1
        hidden_states_right[i, :slen] = item_hidden

        # Targets (-100 mask everywhere prompt or no-EOS).
        rc = compute_remaining_token_targets(
            sequence_length=slen,
            prompt_length=actual_prompt_lengths[i],
            has_eos=sequence_has_eos[i],
            device=device,
        )
        remaining_counts[i, :slen] = rc

    completion_lengths = [
        slen - aplen for slen, aplen in zip(sequence_lengths, actual_prompt_lengths)
    ]

    return {
        "input_ids": input_ids_right,
        "attention_mask": attention_mask_right,
        "hidden_states": hidden_states_right,
        "remaining_token_counts": remaining_counts,
        "has_eos": torch.tensor(sequence_has_eos, dtype=torch.bool, device=device),
        "prompt_length": torch.tensor(actual_prompt_lengths, dtype=torch.long, device=device),
        "completion_length": torch.tensor(completion_lengths, dtype=torch.long, device=device),
    }


def compute_remaining_token_targets(
    sequence_length: int,
    prompt_length: int,
    has_eos: bool,
    device: torch.device
) -> torch.Tensor:
    """
    Compute remaining token count targets for each position.

    Target values:
    - Prompt tokens: -100 (ignored)
    - Completion tokens when has_eos=True: remaining count (N-1, N-2, ..., 0)
    - Completion tokens when has_eos=False: -100 (ignored, truncated sequence)

    Args:
        sequence_length: Total length of prompt + completion
        prompt_length: Length of the prompt
        has_eos: Whether the sequence has an EOS token (not truncated)
        device: Device to create tensor on

    Returns:
        Tensor of shape [sequence_length] with target values
    """
    # Initialize all targets to -100 (ignore)
    remaining_counts = torch.full((sequence_length,), -100, dtype=torch.long, device=device)

    # Only compute targets when we know the true completion length (has EOS)
    # If sequence was cut off (no EOS), we only know LOWER BOUNDS,
    # which would bias the model to underestimate
    if has_eos and sequence_length > prompt_length:
        # Remaining count for each completion token: N-1, N-2, ..., 0
        remaining_counts[prompt_length:] = torch.arange(
            sequence_length - prompt_length - 1, -1, -1, dtype=torch.long, device=device
        )

    return remaining_counts
