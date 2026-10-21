"""
Inference script that loads a trained token count probe and displays
predictions on evaluation samples using the same formatting as the
training ExampleDisplay callback.
"""

import os
import random
from typing import Optional

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from miol.model import ModelWithTokenCountHead
from miol.model.callback import ExampleDisplayCallback


def is_main_process() -> bool:
    """Return True if running on the main process."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank in [-1, 0]


def print_main(*args, **kwargs):
    """Print only on the main process."""
    if is_main_process():
        print(*args, **kwargs)


def load_tokenizer(model_path: str, base_model_name: str):
    """Load tokenizer from saved probe directory or fall back to base model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_probe(cfg: DictConfig, model_path: str) -> tuple[ModelWithTokenCountHead, AutoTokenizer]:
    """Load base model + token count head for inference."""
    model_name = cfg.model.model_name
    tokenizer = load_tokenizer(model_path, model_name)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        device_map = "auto"
    else:
        model_dtype = torch.float32
        device_map = None

    print_main(f"Loading base model '{model_name}' with dtype {model_dtype} on device map {device_map}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        device_map=device_map,
    )

    for param in base_model.parameters():
        param.requires_grad = False

    print_main(f"Loading token count probe from '{model_path}'")
    model = ModelWithTokenCountHead.from_pretrained(
        model_path,
        base_model=base_model,
        hidden_layer=cfg.model.probe_hidden_layer,
    )
    model.eval()
    return model, tokenizer


def display_examples(model, tokenizer, eval_dataset, generation_cfg: dict, num_examples: int):
    """Reuse ExampleDisplay callback to print nicely formatted tables."""
    callback = ExampleDisplayCallback(
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        display_every_n_steps=1,
        generation_config=generation_cfg,
        num_examples=1,
    )

    # Force deterministic sampling for reproducibility
    rng_state = random.getstate()
    random.seed(42)

    for step in range(1, num_examples + 1):
        callback._display_predictions(model, step=step, global_step=step)

    random.setstate(rng_state)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print_main("=" * 80)
    print_main("TOKEN COUNT PROBE INFERENCE")
    print_main("=" * 80)
    print_main(OmegaConf.to_yaml(cfg))
    print_main("=" * 80 + "\n")

    model_path = to_absolute_path(cfg.inference.model_path)
    model, tokenizer = load_probe(cfg, model_path)

    dataset_builder = instantiate(cfg.data)
    split = cfg.inference.dataset_split
    try:
        eval_dataset = dataset_builder.get_dataset(split=split, tokenizer=tokenizer)
    except TypeError:
        # Some builders don't accept tokenizer
        eval_dataset = dataset_builder.get_dataset(split=split)

    generation_config = {
        "max_new_tokens": cfg.generation.max_new_tokens,
        "do_sample": cfg.generation.do_sample,
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "top_k": cfg.generation.top_k,
    }

    print_main(f"Loaded {len(eval_dataset)} samples from split '{split}'")
    print_main(f"Displaying {cfg.inference.num_examples} example(s)\n")

    display_examples(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        generation_cfg=generation_config,
        num_examples=cfg.inference.num_examples,
    )

    print_main("Inference complete!")


if __name__ == "__main__":
    main()
