"""
Dataset creation and loading utilities.
"""

import random
from datasets import Dataset, load_dataset


def get_dataset(data_source, split, num_samples, min_tokens=1, max_tokens=30,
                hf_dataset=None, hf_config=None, hf_prompt_col='query', hf_completion_col='passages',
                *, tokenizer, synthetic_system_prompt, synthetic_user_prompt_template):
    """
    Get dataset (synthetic or HuggingFace).

    Args:
        data_source: 'synthetic' or 'huggingface'
        split: 'train' or 'validation'
        num_samples: Number of samples
        min_tokens: Min tokens for synthetic
        max_tokens: Max tokens for synthetic
        hf_dataset: HuggingFace dataset name
        hf_config: HuggingFace config
        hf_prompt_col: Column with prompts
        hf_completion_col: Column with completions
        tokenizer: Tokenizer for applying chat templates (required for synthetic)
        synthetic_system_prompt: System prompt for chat template
        synthetic_user_prompt_template: User prompt template (use {n} for token count)

    Returns:
        Dataset with 'prompt' and 'completion' columns
    """
    if data_source == 'synthetic':
        # Validate required parameters
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided for synthetic data generation. "
                "Pass the tokenizer to get_dataset()."
            )
        if synthetic_system_prompt is None:
            raise ValueError(
                "synthetic_system_prompt must be specified in config for synthetic data. "
                "Add 'synthetic_system_prompt' to your data config (e.g., data/synthetic.yaml)"
            )
        if synthetic_user_prompt_template is None:
            raise ValueError(
                "synthetic_user_prompt_template must be specified in config for synthetic data. "
                "Add 'synthetic_user_prompt_template' to your data config (e.g., data/synthetic.yaml)"
            )

        # Generate one example per token length (deterministic, full coverage)
        total_available = max_tokens - min_tokens + 1

        # Handle num_samples
        if num_samples == -1:
            # Use all available token lengths
            selected_lengths = list(range(min_tokens, max_tokens + 1))
        elif num_samples > total_available:
            raise ValueError(
                f"Requested {num_samples} samples but only {total_available} unique token lengths "
                f"available (min_tokens={min_tokens}, max_tokens={max_tokens}). "
                f"Either reduce num_samples, increase the token range, or use -1 to get all available data."
            )
        elif num_samples < total_available:
            # Generate all possible token lengths
            all_token_lengths = list(range(min_tokens, max_tokens + 1))
            # Use fixed seed for reproducibility per split
            seed = 42 if split == 'train' else 43
            rng = random.Random(seed)
            selected_lengths = rng.sample(all_token_lengths, num_samples)
        else:
            # num_samples == total_available
            selected_lengths = list(range(min_tokens, max_tokens + 1))

        # Sort to maintain deterministic order
        selected_lengths.sort()

        def data_generator():
            for n in selected_lengths:
                # Build chat messages with system and user prompts
                messages = [
                    {"role": "system", "content": synthetic_system_prompt},
                    {"role": "user", "content": synthetic_user_prompt_template.format(n=n)}
                ]

                # Apply chat template to format the prompt properly
                # tokenize=False returns the formatted string, not token ids
                # add_generation_prompt=True adds the assistant prompt start tokens
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                yield {"prompt": prompt, "completion": "t " * n}

        return Dataset.from_generator(data_generator)

    elif data_source == 'huggingface':
        dataset = load_dataset(hf_dataset, hf_config, split=split)
        total_available = len(dataset)

        # Handle num_samples
        if num_samples == -1:
            # Use all available data
            pass
        elif num_samples is not None:
            if num_samples > total_available:
                raise ValueError(
                    f"Requested {num_samples} samples but only {total_available} available "
                    f"in {hf_dataset} (config={hf_config}, split={split}). "
                    f"Either reduce num_samples or use -1 to get all available data."
                )
            dataset = dataset.select(range(num_samples))

        def format_sample(ex):
            prompt = ex.get(hf_prompt_col, "")
            completion = ex.get(hf_completion_col, "")
            if isinstance(completion, list):
                completion = " ".join(str(c) for c in completion) if completion else ""
            return {"prompt": str(prompt), "completion": str(completion)}

        return dataset.map(format_sample, remove_columns=dataset.column_names)

    else:
        raise ValueError(f"Unknown data_source: {data_source}")
