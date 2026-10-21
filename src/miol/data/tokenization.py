"""
Tokenization utilities for token count prediction.
"""


def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize only the prompt (no completion).
    The model will generate its own completion during training.

    Args:
        examples: Dict with 'prompt' and 'completion' keys
        tokenizer: The tokenizer
        max_length: Maximum sequence length

    Returns:
        Dict with tokenized prompts only
    """
    batch_size = len(examples["prompt"])

    input_ids_list = []
    attention_mask_list = []

    for i in range(batch_size):
        prompt = examples["prompt"][i]

        # Tokenize only the prompt
        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=True,  # Add BOS if needed
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        prompt_ids = prompt_tokens["input_ids"]
        prompt_length = len(prompt_ids)

        # Pad to max_length with LEFT padding (required for decoder-only models)
        # Left padding ensures all prompts "end" at the same position for batch generation
        padding_length = max_length - prompt_length
        if padding_length > 0:
            input_ids = [tokenizer.pad_token_id] * padding_length + prompt_ids
            attention_mask = [0] * padding_length + [1] * prompt_length
        else:
            input_ids = prompt_ids
            attention_mask = [1] * prompt_length

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
    }
