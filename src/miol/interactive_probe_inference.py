"""
Interactive multi-probe inference.

Loads:
  1) a base causal LM,
  2) every probe head present in a `trained_probe/` directory (count, percentage,
     cls_K, prompt_only_count, baseline_*, ...),

then asks for a prompt, generates a response, and applies every probe at every
position. Reports:

  - per-token table for the canonical regression probe (count) with predicted
    remaining tokens, ground truth (= actual generation length minus position),
    absolute and percentage errors;
  - the prompt-only-count probe's headline T̂ at the prompt's last position
    (the "decodability before generation" number);
  - aggregate metrics for every probe — MAE / MAPE / SMAPE / NM score for
    regression, accuracy / Cohen's kappa for classification — using the same
    length-weighted convention as `evaluate_probes.py`.

The probe inference path delegates to `evaluate_probes._apply_heads_to_sample`
so this script and the offline eval are guaranteed to compute identical
numbers from the same hidden states.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from miol.evaluate_probes import _apply_heads_to_sample, load_probe_heads
from miol.model.probe_specs import PROBE_SPECS


# ----------------------------- Loading ---------------------------------------


def parse_hidden_layer(value: str) -> int | str:
    """Parse the --hidden-layer arg into either an int index or 'all'."""
    value = value.strip().lower()
    if value == "all":
        return "all"
    return int(value)


def load_tokenizer(probe_dir: str, base_model_name: str):
    """Prefer the tokenizer stashed alongside the probe; fall back to the base model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(probe_dir)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(base_model_name: str):
    """Load the frozen base causal LM in bf16 on GPU when available."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        device_map = "auto"
    else:
        model_dtype = torch.float32
        device_map = None

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=model_dtype,
        device_map=device_map,
    )
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    return base_model


def discover_probes(probe_dir: Path) -> list[str]:
    """Probes whose `<name>_head.pt` exists in the dir, in PROBE_SPECS order."""
    return [name for name in PROBE_SPECS if (probe_dir / f"{name}_head.pt").exists()]


def load_config_for_hidden_layer(probe_dir: Path, fallback: int | str) -> int | str:
    """Pick up `data.hidden_layer` from `<probe_dir>/config.yaml` when present."""
    cfg_path = probe_dir / "config.yaml"
    if not cfg_path.exists():
        return fallback
    cfg = OmegaConf.load(cfg_path)
    val = str(cfg.data.get("hidden_layer", fallback)).strip().lower()
    if val == "all":
        return "all"
    try:
        return int(val)
    except ValueError:
        return fallback


def build_cls_anchors(active_probes: list[str]) -> dict[str, torch.Tensor]:
    """Per-probe class anchors `linspace(0, 1, n_bins)` for each cls probe."""
    anchors: dict[str, torch.Tensor] = {}
    for name in active_probes:
        spec = PROBE_SPECS[name]
        if spec["type"] == "classification":
            anchors[name] = torch.linspace(0, 1, spec["output_dim"])
    return anchors


# ----------------------------- Hidden-state extraction -----------------------


def extract_hidden_states(outputs, hidden_layer: int | str) -> torch.Tensor:
    """Extract the per-token hidden-state tensor matching the train-time layout.

    HF returns `outputs.hidden_states` as a tuple of length `num_hidden_layers + 1`
    (input embeddings + per-layer outputs). Our offline `extract_hidden_states.py`
    saves only the per-layer outputs, so for `hidden_layer="all"` we concat
    indices 1..N to match. For numeric indices we pass through directly so the
    semantics match what `HiddenStatesDataset.__getitem__` returned at train time.
    Returns a 2-D tensor `[seq_len, hidden_size]` (batch dim squeezed).
    """
    if hidden_layer == "all":
        per_layer = list(outputs.hidden_states)[1:]  # drop input embeddings
        merged = torch.cat(per_layer, dim=-1)
    else:
        merged = outputs.hidden_states[int(hidden_layer)]
    return merged[0]  # [seq_len, hidden_size]


# ----------------------------- Generation + probe ----------------------------


def generate_and_apply(
    base_model,
    tokenizer,
    heads: dict[str, torch.nn.Linear],
    cls_anchors: dict[str, torch.Tensor],
    hidden_layer: int | str,
    prompt: str,
    *,
    system_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Generate, run a forward pass for hidden states, apply every probe head.

    Per-probe outputs (predictions, per-token errors, aggregates) come from
    `_apply_heads_to_sample`, the same helper that drives offline evaluation —
    so the numbers here match `evaluate_probes.py` to the bit when fed the same
    hidden states.
    """
    device = next(base_model.parameters()).device

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": str(prompt)})
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    tokenized = tokenizer(formatted, return_tensors="pt")
    prompt_ids = tokenized["input_ids"].to(device)
    prompt_attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        generated = base_model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        full_ids = generated[0]
        prompt_len = int(prompt_ids.shape[1])
        completion_ids = full_ids[prompt_len:]

        outputs = base_model(
            input_ids=full_ids.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = extract_hidden_states(outputs, hidden_layer)  # [seq_len, H]

    completion_length = int(completion_ids.shape[0])
    eos_id = tokenizer.eos_token_id
    eos_emitted = (
        eos_id is not None
        and bool((completion_ids == eos_id).any().item())
    )
    # We always have a concrete generation length, so we treat the realized
    # length as the ground truth $T$ regardless of whether the model emitted
    # EOS naturally or hit max_new_tokens. The flag is reported for context.
    has_eos = completion_length > 0

    if has_eos:
        count_targets = [float(completion_length - k - 1) for k in range(completion_length)]
        if completion_length > 1:
            pct_targets = [(completion_length - k - 1) / (completion_length - 1) for k in range(completion_length)]
        else:
            pct_targets = [0.0]
    else:
        count_targets = None
        pct_targets = None

    sample = {
        "input_ids": full_ids.cpu(),
        "hidden_states": hidden_states.cpu(),
        "prompt_length": prompt_len,
        "completion_length": completion_length,
        "has_eos": has_eos,
        "prompt": tokenizer.decode(full_ids[:prompt_len], skip_special_tokens=True),
        "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
    }

    probes_out = _apply_heads_to_sample(
        sample=sample,
        heads=heads,
        cls_anchors=cls_anchors,
        has_eos=has_eos,
        prompt_length=prompt_len,
        completion_length=completion_length,
        count_targets=count_targets,
        pct_targets=pct_targets,
    )

    tokens_decoded = [tokenizer.decode([tid]) for tid in full_ids.tolist()]

    return {
        "full_ids": full_ids.cpu(),
        "tokens_decoded": tokens_decoded,
        "prompt_text": sample["prompt"],
        "completion_text": sample["completion"],
        "prompt_length": prompt_len,
        "completion_length": completion_length,
        "has_eos": has_eos,
        "eos_emitted": eos_emitted,
        "count_targets": count_targets,
        "probes": probes_out,
    }


# ----------------------------- Display ---------------------------------------


def _trunc(s: str, n: int) -> str:
    return (s if len(s) <= n else s[: n - 1] + "…")


def _print_count_table(entry: dict[str, Any], result: dict[str, Any]) -> None:
    """Per-token table for the canonical count probe."""
    predictions = entry.get("predictions") or []
    mae_per_token = entry.get("mae_per_token") or []
    mape_per_token = entry.get("mape_per_token") or []
    count_targets = result["count_targets"] or []
    prompt_len = result["prompt_length"]
    full_ids = result["full_ids"].tolist()
    tokens_decoded = result["tokens_decoded"]

    print("\n" + "=" * 96)
    print("count probe — per-token predictions")
    print("=" * 96)
    print(
        f"{'idx':>4}  {'type':<6}  {'token':<22}  {'pred':>10}  {'gt':>6}  {'abs_err':>9}  {'pct_err':>9}"
    )
    print("-" * 96)
    seq_len = len(full_ids)
    for idx in range(seq_len):
        token_str = repr(tokens_decoded[idx])[:22]
        if idx < prompt_len:
            tag = "PROMPT"
            pred_str = "—"
            gt_str = "—"
            abs_str = "—"
            pct_str = "—"
        else:
            j = idx - prompt_len
            pred = predictions[j] if j < len(predictions) else None
            tgt = count_targets[j] if j < len(count_targets) else None
            ape = mae_per_token[j] if j < len(mae_per_token) else None
            mp = mape_per_token[j] if j < len(mape_per_token) else None
            tag = "COMPL"
            pred_str = f"{pred:.3f}" if isinstance(pred, (int, float)) else "—"
            gt_str = f"{int(tgt)}" if isinstance(tgt, (int, float)) else "—"
            abs_str = f"{ape:.3f}" if isinstance(ape, (int, float)) else "—"
            pct_str = f"{mp * 100:.2f}%" if isinstance(mp, (int, float)) else "—"
        print(f"{idx:>4}  {tag:<6}  {token_str:<22}  {pred_str:>10}  {gt_str:>6}  {abs_str:>9}  {pct_str:>9}")


def _print_aggregate(name: str, entry: dict[str, Any]) -> None:
    """Compact aggregate summary for one probe."""
    spec = PROBE_SPECS.get(name, {})
    kind = spec.get("type", "?")
    agg = entry.get("aggregate") or {}
    if not agg:
        print(f"  {name:<24} ({kind:<14}): aggregate unavailable (has_eos=False)")
        return

    def g(k: str) -> str:
        v = agg.get(k)
        if not isinstance(v, (int, float)):
            return "—"
        return f"{v:.4f}"

    if kind == "regression":
        line = (
            f"MAE={g('mae')}  MSE={g('mse')}  MAPE={g('mape')}  SMAPE={g('smape')}  "
            f"NM={g('nm_score')}  NM/tok={g('nm_score_per_token')}"
        )
    elif kind == "classification":
        line = (
            f"acc={g('accuracy')}  acc_last10={g('accuracy_last_10')}  "
            f"κ={g('cohen_kappa')}  derived_pct_mae={g('derived_pct_mae')}  "
            f"derived_pct_mse={g('derived_pct_mse')}  NM={g('nm_score')}"
        )
    else:
        line = " ".join(f"{k}={g(k)}" for k in agg)
    print(f"  {name:<24} ({kind:<14}): {line}")


def _print_prompt_only_summary(name: str, entry: dict[str, Any], completion_length: int) -> None:
    """Headline numbers for the prompt-only-count probe."""
    print("\n" + "=" * 96)
    print(f"{name} — length probe (predicts T from the prompt's last hidden state)")
    print("=" * 96)
    pe = entry.get("prompt_end_prediction")
    if isinstance(pe, (int, float)):
        err = abs(pe - completion_length) if completion_length > 0 else None
        err_str = f"{err:.3f}" if err is not None else "—"
        print(f"  prompt-end T̂ = {pe:.3f}   actual T = {completion_length}   |T̂ − T| = {err_str}")
    agg = entry.get("aggregate") or {}
    if agg:
        def g(k: str) -> str:
            v = agg.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
        print(
            f"  prompt-end MAE={g('prompt_mae')}  MSE={g('prompt_mse')}  "
            f"MAPE={g('prompt_mape')}  SMAPE={g('prompt_smape')}"
        )
        print(
            f"  static-decay (no info from generation): "
            f"MAE={g('static_decay_mae')}  MSE={g('static_decay_mse')}  "
            f"MAPE={g('static_decay_mape')}  NM_static={g('nm_score_static_decay')}"
        )
        print(
            f"  per-position implied-remaining: "
            f"MAE={g('completion_mae_remaining')}  MSE={g('completion_mse_remaining')}  "
            f"MAPE={g('completion_mape_remaining')}  NM={g('nm_score')}"
        )


def display_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print("Generated response")
    print("=" * 96)
    print(result["completion_text"])
    print(
        f"\nprompt_length={result['prompt_length']}  "
        f"completion_length={result['completion_length']}  "
        f"eos_emitted={result['eos_emitted']}"
    )

    probes = result["probes"]

    # Big per-token table for the count probe (the canonical regression head).
    if "count" in probes:
        _print_count_table(probes["count"], result)

    # Length-probe headline numbers.
    if "prompt_only_count" in probes:
        _print_prompt_only_summary(
            "prompt_only_count", probes["prompt_only_count"], result["completion_length"]
        )

    # Compact aggregate summary for every probe.
    print("\n" + "=" * 96)
    print("Aggregate metrics (length-weighted, matches evaluate_probes.py)")
    print("=" * 96)
    for name, entry in probes.items():
        if name == "prompt_only_count":
            continue  # already covered above
        _print_aggregate(name, entry)


# ----------------------------- CLI -------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive multi-probe inference: generate a response and "
                    "report every probe's per-token predictions and aggregates."
    )
    parser.add_argument("--base-model", type=str, required=True, help="Base model name/path")
    parser.add_argument("--probe-dir", type=str, required=True,
                        help="Path to a `trained_probe/` directory (contains *_head.pt + config.yaml)")
    parser.add_argument(
        "--hidden-layer",
        type=parse_hidden_layer,
        default=None,
        help="Override `data.hidden_layer` from the probe-dir's config.yaml. "
             "Accepts an int (e.g. -1, 0, 5) or 'all'. If omitted, the saved config wins.",
    )
    parser.add_argument(
        "--probes",
        type=str,
        nargs="+",
        default=None,
        help="Subset of probe names to load (e.g. 'count cls_5 prompt_only_count'). "
             "Defaults to every probe whose head is present in the probe-dir.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful math tutor that solves grade school math problems step by step.",
        help="System prompt prepended to every message. Pass an empty string to skip.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    if not probe_dir.is_dir():
        parser.error(f"--probe-dir does not exist or is not a directory: {probe_dir}")

    hidden_layer = (
        args.hidden_layer
        if args.hidden_layer is not None
        else load_config_for_hidden_layer(probe_dir, fallback="all")
    )

    available = discover_probes(probe_dir)
    if not available:
        parser.error(f"No `<name>_head.pt` files found in {probe_dir}; nothing to run.")
    if args.probes:
        unknown = [p for p in args.probes if p not in PROBE_SPECS]
        if unknown:
            parser.error(f"Unknown probe(s): {unknown}. Known: {sorted(PROBE_SPECS)}")
        missing = [p for p in args.probes if not (probe_dir / f"{p}_head.pt").exists()]
        if missing:
            parser.error(f"Probe head(s) not found in {probe_dir}: {missing}")
        active_probes = list(args.probes)
    else:
        active_probes = available

    print(f"Hidden layer: {hidden_layer}")
    print(f"Active probes ({len(active_probes)}): {active_probes}")

    tokenizer = load_tokenizer(str(probe_dir), args.base_model)
    base_model = load_base_model(args.base_model)

    # Determine the probe-input hidden size from one base-model forward on a
    # 1-token input. Skips assumptions about config naming across families.
    with torch.no_grad():
        device = next(base_model.parameters()).device
        probe_outputs = base_model(
            input_ids=torch.tensor([[tokenizer.eos_token_id or 0]], device=device),
            output_hidden_states=True,
            return_dict=True,
        )
        h_probe = extract_hidden_states(probe_outputs, hidden_layer)
        hidden_size = int(h_probe.shape[-1])
    print(f"Probe input hidden size: {hidden_size}")

    heads = load_probe_heads(probe_dir, active_probes, hidden_size)
    cls_anchors = build_cls_anchors(active_probes)

    print("\nModel and probes loaded. Enter a prompt (empty to exit).")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            print("Exiting.")
            break

        result = generate_and_apply(
            base_model=base_model,
            tokenizer=tokenizer,
            heads=heads,
            cls_anchors=cls_anchors,
            hidden_layer=hidden_layer,
            prompt=prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        display_result(result)


if __name__ == "__main__":
    main()
