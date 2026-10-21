"""
Unit tests for generation utilities.
"""

import torch
import pytest
from miol.model.generation_utils import (
    build_generation_kwargs,
    extract_completion_with_eos,
    compute_remaining_token_targets,
)


class TestBuildGenerationKwargs:
    """Tests for build_generation_kwargs function."""

    def test_basic_greedy_generation(self):
        """Test building kwargs for greedy decoding."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        generation_config = {
            'max_new_tokens': 50,
            'do_sample': False,
        }

        kwargs = build_generation_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            pad_token_id=0,
            eos_token_id=2,
            include_hidden_states=False
        )

        assert torch.equal(kwargs['input_ids'], input_ids)
        assert torch.equal(kwargs['attention_mask'], attention_mask)
        assert kwargs['max_new_tokens'] == 50
        assert kwargs['do_sample'] is False
        assert kwargs['pad_token_id'] == 0
        assert kwargs['eos_token_id'] == 2
        # Greedy decoding should have None sampling params
        assert kwargs['temperature'] is None
        assert kwargs['top_p'] is None
        assert kwargs['top_k'] is None
        # Should not include hidden states
        assert 'output_hidden_states' not in kwargs

    def test_sampling_generation(self):
        """Test building kwargs for sampling-based generation."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        generation_config = {
            'max_new_tokens': 100,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
        }

        kwargs = build_generation_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            pad_token_id=0,
            eos_token_id=2,
            include_hidden_states=False
        )

        assert kwargs['do_sample'] is True
        assert kwargs['temperature'] == 0.7
        assert kwargs['top_p'] == 0.9
        assert kwargs['top_k'] == 50

    def test_with_hidden_states(self):
        """Test that hidden states are included when requested."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        generation_config = {'max_new_tokens': 10, 'do_sample': False}

        kwargs = build_generation_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            pad_token_id=0,
            eos_token_id=2,
            include_hidden_states=True
        )

        assert kwargs['output_hidden_states'] is True
        assert kwargs['return_dict_in_generate'] is True

    def test_default_values(self):
        """Test that default values are used when not in config."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        generation_config = {}  # Empty config

        kwargs = build_generation_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            pad_token_id=0,
            eos_token_id=2,
        )

        assert kwargs['max_new_tokens'] == 500  # default
        assert kwargs['do_sample'] is False  # default
        assert kwargs['temperature'] is None  # greedy mode


class TestExtractCompletionWithEos:
    """Tests for extract_completion_with_eos function."""

    def test_completion_with_eos(self):
        """Test extracting completion when EOS is present."""
        # Prompt: [1, 2, 3], Completion: [4, 5, 2] (2 is EOS)
        generated_ids = torch.tensor([1, 2, 3, 4, 5, 2, 6, 7])
        prompt_length = 3
        eos_token_id = 2

        completion_ids, has_eos, actual_prompt_len = extract_completion_with_eos(
            generated_ids=generated_ids,
            prompt_length=prompt_length,
            eos_token_id=eos_token_id,
        )

        # Should extract [4, 5, 2] and truncate at EOS
        assert torch.equal(completion_ids, torch.tensor([4, 5, 2]))
        assert has_eos is True
        assert actual_prompt_len == 3

    def test_completion_without_eos(self):
        """Test extracting completion when no EOS is found (truncated)."""
        # Prompt: [1, 2, 3], Completion: [4, 5, 6] (no EOS)
        generated_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        prompt_length = 3
        eos_token_id = 2

        completion_ids, has_eos, actual_prompt_len = extract_completion_with_eos(
            generated_ids=generated_ids,
            prompt_length=prompt_length,
            eos_token_id=eos_token_id,
        )

        # Should extract [4, 5, 6] (all remaining tokens)
        assert torch.equal(completion_ids, torch.tensor([4, 5, 6]))
        assert has_eos is False
        assert actual_prompt_len == 3

    def test_bos_added_by_generate(self):
        """Test detecting when .generate() adds BOS token."""
        # Generated: [10, 1, 2, 3, 4, 5, 2] where 10 is BOS
        # Original prompt: [1, 2, 3] (no BOS)
        generated_ids = torch.tensor([10, 1, 2, 3, 4, 5, 2])
        prompt_length = 3
        eos_token_id = 2
        bos_token_id = 10
        original_prompt_ids = torch.tensor([1, 2, 3])

        completion_ids, has_eos, actual_prompt_len = extract_completion_with_eos(
            generated_ids=generated_ids,
            prompt_length=prompt_length,
            eos_token_id=eos_token_id,
            check_bos=True,
            bos_token_id=bos_token_id,
            original_prompt_ids=original_prompt_ids,
        )

        # Should detect BOS was added, so actual_prompt_len = 4
        # Completion starts at index 4: [4, 5, 2]
        assert torch.equal(completion_ids, torch.tensor([4, 5, 2]))
        assert has_eos is True
        assert actual_prompt_len == 4  # 3 + 1 for added BOS

    def test_bos_already_in_prompt(self):
        """Test when BOS is already in original prompt."""
        # Generated: [10, 1, 2, 3, 4, 5, 2] where 10 is BOS
        # Original prompt: [10, 1, 2, 3] (BOS already present)
        generated_ids = torch.tensor([10, 1, 2, 3, 4, 5, 2])
        prompt_length = 4
        eos_token_id = 2
        bos_token_id = 10
        original_prompt_ids = torch.tensor([10, 1, 2, 3])

        completion_ids, has_eos, actual_prompt_len = extract_completion_with_eos(
            generated_ids=generated_ids,
            prompt_length=prompt_length,
            eos_token_id=eos_token_id,
            check_bos=True,
            bos_token_id=bos_token_id,
            original_prompt_ids=original_prompt_ids,
        )

        # BOS was not added, so actual_prompt_len stays 4
        # Completion starts at index 4: [4, 5, 2]
        assert torch.equal(completion_ids, torch.tensor([4, 5, 2]))
        assert has_eos is True
        assert actual_prompt_len == 4


class TestComputeRemainingTokenTargets:
    """Tests for compute_remaining_token_targets function."""

    def test_targets_with_eos(self):
        """Test target computation when EOS is present."""
        sequence_length = 8  # 3 prompt + 5 completion
        prompt_length = 3
        has_eos = True
        device = torch.device('cpu')

        targets = compute_remaining_token_targets(
            sequence_length=sequence_length,
            prompt_length=prompt_length,
            has_eos=has_eos,
            device=device
        )

        # Prompt tokens: -100 (ignored)
        assert torch.equal(targets[:3], torch.tensor([-100, -100, -100]))
        # Completion tokens: 4, 3, 2, 1, 0
        assert torch.equal(targets[3:], torch.tensor([4, 3, 2, 1, 0]))

    def test_targets_without_eos_truncated(self):
        """Test target computation when sequence was truncated (no EOS)."""
        sequence_length = 8  # 3 prompt + 5 completion
        prompt_length = 3
        has_eos = False  # Truncated
        device = torch.device('cpu')

        targets = compute_remaining_token_targets(
            sequence_length=sequence_length,
            prompt_length=prompt_length,
            has_eos=has_eos,
            device=device
        )

        # ALL tokens should be -100 (ignored) when truncated
        assert torch.all(targets == -100)

    def test_zero_length_completion(self):
        """Test when completion has zero length (edge case)."""
        sequence_length = 3  # Only prompt
        prompt_length = 3
        has_eos = True
        device = torch.device('cpu')

        targets = compute_remaining_token_targets(
            sequence_length=sequence_length,
            prompt_length=prompt_length,
            has_eos=has_eos,
            device=device
        )

        # All should be -100 (only prompt)
        assert torch.all(targets == -100)

    def test_single_token_completion(self):
        """Test with single completion token."""
        sequence_length = 4  # 3 prompt + 1 completion
        prompt_length = 3
        has_eos = True
        device = torch.device('cpu')

        targets = compute_remaining_token_targets(
            sequence_length=sequence_length,
            prompt_length=prompt_length,
            has_eos=has_eos,
            device=device
        )

        # Prompt: -100, -100, -100; Completion: 0
        assert torch.equal(targets, torch.tensor([-100, -100, -100, 0]))
