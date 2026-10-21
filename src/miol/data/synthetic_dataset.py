"""
Synthetic dataset builder for token count prediction.
"""

import random
from datasets import Dataset


class SyntheticDatasetBuilder:
    """
    Builds synthetic datasets for token count prediction training.

    Generates one example per token length (deterministic, full coverage).
    Each example asks the model to generate exactly N tokens.
    """

    def __init__(
        self,
        num_train_samples: int,
        num_eval_samples: int,
        min_tokens: int = 1,
        max_tokens: int = 100,
        synthetic_system_prompt: str = None,
        synthetic_user_prompt_template: str = None,
        dataset_name: str = None,  # Optional alias for hf_dataset (for config clarity)
    ):
        """
        Args:
            num_train_samples: Number of training samples (-1 for all available)
            num_eval_samples: Number of eval samples (-1 for all available)
            min_tokens: Minimum number of tokens to generate
            max_tokens: Maximum number of tokens to generate
            synthetic_system_prompt: System prompt for chat template
            synthetic_user_prompt_template: User prompt template (use {n} for token count)
            dataset_name: Optional alias for the Hugging Face dataset
        """
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.synthetic_system_prompt = synthetic_system_prompt
        self.synthetic_user_prompt_template = synthetic_user_prompt_template

        # Validate required parameters
        if self.synthetic_system_prompt is None:
            raise ValueError(
                "synthetic_system_prompt must be specified in config for synthetic data. "
                "Add 'synthetic_system_prompt' to your data config (e.g., data/synthetic.yaml)"
            )
        if self.synthetic_user_prompt_template is None:
            raise ValueError(
                "synthetic_user_prompt_template must be specified in config for synthetic data. "
                "Add 'synthetic_user_prompt_template' to your data config (e.g., data/synthetic.yaml)"
            )

    def get_dataset(self, split: str, tokenizer) -> Dataset:
        """
        Generate synthetic dataset for the given split.

        Args:
            split: 'train' or 'validation'
            tokenizer: Tokenizer for applying chat templates

        Returns:
            Dataset with 'prompt' and 'completion' columns
        """
        # Validate required parameters
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided for synthetic data generation. "
                "Pass the tokenizer to get_dataset()."
            )

        # Determine number of samples for this split
        num_samples = self.num_train_samples if split == 'train' else self.num_eval_samples

        # Generate one example per token length (deterministic, full coverage)
        total_available = self.max_tokens - self.min_tokens + 1

        # Handle num_samples
        if num_samples == -1:
            # Use all available token lengths
            selected_lengths = list(range(self.min_tokens, self.max_tokens + 1))
        elif num_samples > total_available:
            raise ValueError(
                f"Requested {num_samples} samples but only {total_available} unique token lengths "
                f"available (min_tokens={self.min_tokens}, max_tokens={self.max_tokens}). "
                f"Either reduce num_samples, increase the token range, or use -1 to get all available data."
            )
        elif num_samples < total_available:
            # Generate all possible token lengths
            all_token_lengths = list(range(self.min_tokens, self.max_tokens + 1))
            # Use fixed seed for reproducibility per split
            seed = 42 if split == 'train' else 43
            rng = random.Random(seed)
            selected_lengths = rng.sample(all_token_lengths, num_samples)
        else:
            # num_samples == total_available
            selected_lengths = list(range(self.min_tokens, self.max_tokens + 1))

        # Sort to maintain deterministic order
        selected_lengths.sort()

        def data_generator():
            for n in selected_lengths:
                # Build chat messages with system and user prompts
                messages = [
                    {"role": "system", "content": self.synthetic_system_prompt},
                    {"role": "user", "content": self.synthetic_user_prompt_template.format(n=n)}
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
