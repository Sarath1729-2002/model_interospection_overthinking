"""
HuggingFace dataset builder for token count prediction.
"""

from datasets import Dataset, concatenate_datasets, load_dataset


class HuggingFaceDatasetBuilder:
    """
    Builds datasets from HuggingFace datasets for token count prediction training.
    Uses chat template API to format prompts with system and user messages.
    """

    def __init__(
        self,
        num_train_samples: int,
        num_eval_samples: int,
        hf_dataset: str,
        hf_config: str = None,
        hf_prompt_col: str = 'query',
        hf_completion_col: str = 'passages',
        hf_system_prompt: str = None,
        hf_eval_split: str = 'validation',
        hf_train_split: str = 'train',
        enable_thinking: bool = False,
        dataset_name: str = None,  # Optional alias for hf_dataset (for config clarity
    ):
        """
        Args:
            num_train_samples: Number of training samples (-1 for all available)
            num_eval_samples: Number of eval samples (-1 for all available)
            hf_dataset: HuggingFace dataset name
            hf_config: HuggingFace dataset config/subset name. Either a single
                string (e.g. "main"), or a list of strings (e.g.
                ["algebra", "geometry", ...]) — in which case each config is
                loaded and the resulting datasets are concatenated for both
                the train and eval splits.
            hf_prompt_col: Column name containing prompts (will be used as user message)
            hf_completion_col: Column name containing completions
            hf_system_prompt: System prompt to use in chat template
            hf_eval_split: Name of the eval split ('validation' or 'test')
            hf_train_split: Name of the train split (default 'train')
            enable_thinking: Whether to enable 'thinking mode' in chat template
                (only meaningful for tokenizers that support it, e.g. Qwen3)
        """
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config
        self.hf_prompt_col = hf_prompt_col
        self.hf_completion_col = hf_completion_col
        self.hf_system_prompt = hf_system_prompt
        self.hf_eval_split = hf_eval_split
        self.hf_train_split = hf_train_split
        self.enable_thinking = enable_thinking

        # Validate required parameters
        if self.hf_system_prompt is None:
            raise ValueError(
                "hf_system_prompt must be specified in config for HuggingFace datasets. "
                "Add 'hf_system_prompt' to your data config (e.g., data/msmarco.yaml)"
            )

    def get_dataset(self, split: str, tokenizer) -> Dataset:
        """
        Load HuggingFace dataset for the given split and apply chat template.

        Args:
            split: 'train' or 'validation'
            tokenizer: Tokenizer for applying chat templates

        Returns:
            Dataset with 'prompt' and 'completion' columns
        """
        # Validate required parameters
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided for HuggingFace data formatting. "
                "Pass the tokenizer to get_dataset()."
            )

        # Determine number of samples for this split
        num_samples = self.num_train_samples if split == 'train' else self.num_eval_samples

        # Map generic split name to actual dataset split name
        actual_split = self.hf_train_split if split == 'train' else self.hf_eval_split

        # Load dataset. `hf_config` may be a single string OR a list of config
        # names (e.g. the seven MATH subjects); in the list case we load each
        # one and concatenate. Hydra hands us a ListConfig for list literals;
        # we coerce to a plain list.
        cfg_value = self.hf_config
        if cfg_value is not None and not isinstance(cfg_value, str):
            try:
                cfg_list = list(cfg_value)
            except TypeError:
                cfg_list = None
        else:
            cfg_list = None

        if cfg_list is not None and len(cfg_list) > 1:
            sub_datasets = [
                load_dataset(self.hf_dataset, str(c), split=actual_split)
                for c in cfg_list
            ]
            dataset = concatenate_datasets(sub_datasets)
        elif cfg_list is not None and len(cfg_list) == 1:
            dataset = load_dataset(self.hf_dataset, str(cfg_list[0]), split=actual_split)
        else:
            dataset = load_dataset(self.hf_dataset, cfg_value, split=actual_split)
        total_available = len(dataset)

        # Handle num_samples. Conventions:
        #   num_samples == -1      -> use all available data (sentinel).
        #   num_samples >  0       -> first `num_samples` examples.
        #   num_samples <  -1      -> last  `|num_samples|` examples.
        if num_samples == -1:
            # Use all available data
            pass
        elif num_samples is not None and num_samples < -1:
            n = -num_samples
            if n > total_available:
                raise ValueError(
                    f"Requested last {n} samples but only {total_available} available "
                    f"in {self.hf_dataset} (config={self.hf_config}, split={split})."
                )
            dataset = dataset.select(range(total_available - n, total_available))
        elif num_samples is not None:
            if num_samples > total_available:
                raise ValueError(
                    f"Requested {num_samples} samples but only {total_available} available "
                    f"in {self.hf_dataset} (config={self.hf_config}, split={split}). "
                    f"Either reduce num_samples or use -1 to get all available data."
                )
            dataset = dataset.select(range(num_samples))

        def format_sample(ex):
            # Get user content from the specified column
            user_content = ex.get(self.hf_prompt_col, "")

            # Get completion and format if it's a list
            completion = ex.get(self.hf_completion_col, "")
            if isinstance(completion, list):
                completion = " ".join(str(c) for c in completion) if completion else ""

            # Build chat messages with system and user prompts
            messages = [
                {"role": "system", "content": self.hf_system_prompt},
                {"role": "user", "content": str(user_content)}
            ]

            # Apply chat template to format the prompt properly
            # tokenize=False returns the formatted string, not token ids
            # add_generation_prompt=True adds the assistant prompt start tokens
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            return {"prompt": prompt, "completion": str(completion)}

        return dataset.map(format_sample, remove_columns=dataset.column_names)
