"""
Script to extract and save hidden states from the base model for offline probe training.

This script:
1. Loads the base model (without probe) from Hydra config
2. Runs *batched* generation on the specified dataset
3. Runs one teacher-forced forward pass over the generated (prompt, completion)
   sequences and saves inputs, outputs, and per-layer hidden states to disk

Throughput notes (the on-disk format is unchanged, so train_offline.py,
evaluate_probes.py etc. need no changes):
  * Generation is batched (`extract.batch_size`, prompts sorted by length to keep
    padding low) instead of one prompt at a time.
  * `generate()` no longer requests hidden states: they were never used, the
    clean teacher-forced forward below is the source of truth.
  * The forward pass runs on the base transformer (no `lm_head`), right-padded,
    so real tokens get exactly the positions 0..n-1 they would get unbatched.
  * Tensors are copied to host and written by background threads, so GPU work
    overlaps disk I/O. A sample's `.json` is written last and atomically; it is
    the completion marker the resume logic relies on.

Usage:
    uv run python src/miol/extract_hidden_states.py
    uv run python src/miol/extract_hidden_states.py data=gsm8k output_dir=./hidden_states_gsm8k
    uv run python src/miol/extract_hidden_states.py data.num_train_samples=100
    uv run python src/miol/extract_hidden_states.py extract.batch_size=48
"""

import json
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from miol.model.generation_utils import (
    build_generation_kwargs,
    extract_completion_with_eos,
)

# Same truncation length the original per-sample tokenize_function used.
MAX_PROMPT_LENGTH = 512

_SAMPLE_JSON_RE = re.compile(r"^sample_(\d+)\.json$")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Seed before any data sampling or generation. With do_sample=true the
    # generated completions depend on the seed; downstream probe-training
    # multi-seed runs share these completions, so this is the *extraction*
    # seed and is independent from the probe-training seed.
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    print_main("=" * 80)
    print_main("EXTRACT HIDDEN STATES")
    print_main(f"SEED: {seed}")
    print_main("=" * 80)
    print_main(OmegaConf.to_yaml(cfg))
    print_main("=" * 80 + "\n")

    extract_cfg = cfg.get("extract", {})

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    dist_backend: Optional[str] = None
    if local_rank != -1 and not is_distributed():
        dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Ranks only synchronise at the very end, and with batched generation
        # they can finish far apart; the default 10 min collective timeout would
        # kill early finishers while they wait for the slowest rank.
        timeout_hours = float(extract_cfg.get("dist_timeout_hours", 12))
        dist.init_process_group(backend=dist_backend, timeout=timedelta(hours=timeout_hours))
    elif is_distributed():
        backend_obj = dist.get_backend()
        dist_backend = str(backend_obj)

    rank = get_rank()
    world_size = get_world_size()

    # Setup output directory
    output_dir = cfg.get("output_dir", "./outputs")
    hidden_states_dir = Path(output_dir) / "hidden_states"
    hidden_states_dir.mkdir(parents=True, exist_ok=True)

    print_main(f"Output directory: {hidden_states_dir}")

    # Load tokenizer
    model_name = cfg.model.model_name
    print_main(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Left padding for generation
    tokenizer.padding_side = 'left'

    # Load base model (without probe)
    print_main(f"Loading base model: {model_name}")

    if torch.cuda.is_available():
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    print_main(f"Using device: {device}, dtype: {model_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    # Load dataset
    dataset_builder = instantiate(cfg.data)
    dataset_type = cfg.data._target_.split('.')[-1]
    print_main(f"Loading data (builder: {dataset_type})...")

    # Extract both train and eval datasets
    train_dataset = dataset_builder.get_dataset(split='train', tokenizer=tokenizer)
    eval_dataset = dataset_builder.get_dataset(split='validation', tokenizer=tokenizer)

    print_main(f"Train dataset size: {len(train_dataset)}")
    print_main(f"Eval dataset size: {len(eval_dataset)}")

    # Get generation config from cfg and override with model-specific config if present
    generation_config = dict(cfg.get("generation", {}))
    if "generation_config" in cfg.model:
        generation_config.update(dict(cfg.model.get("generation_config", {})))
    print_main("\nProcessing generation config...")
    print_main(generation_config)

    # Whether to save the heavy `.safetensors` for samples whose generation did
    # not terminate naturally. The per-sample `.json` metadata is always kept.
    save_truncated = bool(extract_cfg.get("save_truncated", False))
    print_main(f"\nsave_truncated = {save_truncated} "
               f"({'all samples saved' if save_truncated else 'truncated samples skipped on disk; JSON kept'})")

    batching = dict(
        batch_size=int(extract_cfg.get("batch_size", 32)),
        forward_batch_size=int(extract_cfg.get("forward_batch_size", 16)),
        writer_threads=int(extract_cfg.get("writer_threads", 4)),
        max_inflight_batches=int(extract_cfg.get("max_inflight_batches", 2)),
    )
    print_main(f"Batching: {batching}")

    # Process train dataset
    print_main("\nProcessing train dataset...")
    train_saved, train_skipped, train_failed = process_dataset(
        dataset=train_dataset,
        split="train",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        output_dir=hidden_states_dir,
        device=device,
        rank=rank,
        world_size=world_size,
        save_truncated=save_truncated,
        **batching,
    )

    # Process eval dataset
    print_main("\nProcessing eval dataset...")
    eval_saved, eval_skipped, eval_failed = process_dataset(
        dataset=eval_dataset,
        split="eval",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        output_dir=hidden_states_dir,
        device=device,
        rank=rank,
        world_size=world_size,
        save_truncated=save_truncated,
        **batching,
    )

    total_train_saved = aggregate_count(train_saved, dist_backend, device)
    total_train_skipped = aggregate_count(train_skipped, dist_backend, device)
    total_train_failed = aggregate_count(train_failed, dist_backend, device)
    total_eval_saved = aggregate_count(eval_saved, dist_backend, device)
    total_eval_skipped = aggregate_count(eval_skipped, dist_backend, device)
    total_eval_failed = aggregate_count(eval_failed, dist_backend, device)

    train_total = total_train_saved + total_train_skipped + total_train_failed
    eval_total = total_eval_saved + total_eval_skipped + total_eval_failed

    def _pct(n, d):
        return (100.0 * n / d) if d > 0 else 0.0

    print_main(
        f"\nDone! Hidden states saved to: {hidden_states_dir}\n"
        f"  train: saved {total_train_saved} / processed {train_total}"
        f" — {total_train_skipped} truncated"
        f" ({_pct(total_train_skipped, train_total):.1f}%)"
        f", {total_train_failed} failed"
        f" ({_pct(total_train_failed, train_total):.1f}%)\n"
        f"  eval:  saved {total_eval_saved} / processed {eval_total}"
        f" — {total_eval_skipped} truncated"
        f" ({_pct(total_eval_skipped, eval_total):.1f}%)"
        f", {total_eval_failed} failed"
        f" ({_pct(total_eval_failed, eval_total):.1f}%)"
    )

    # Per-split extraction stats — useful when reading the cache months later
    # to remember why some indices are missing.
    if is_main_process():
        for split, saved, skipped, failed_n in (
            ("train", total_train_saved, total_train_skipped, total_train_failed),
            ("eval", total_eval_saved, total_eval_skipped, total_eval_failed),
        ):
            stats = {
                "split": split,
                "saved_with_tensors": int(saved),
                "skipped_truncated": int(skipped),
                "failed": int(failed_n),
                "total_processed": int(saved + skipped + failed_n),
                "save_truncated_flag": save_truncated,
            }
            stats_path = hidden_states_dir / split / "extraction_stats.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def split_generated_batch(
    generated_ids: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int,
    bos_token_id: Optional[int],
):
    """Split a left-padded batch of generate() outputs into per-sample pieces.

    `input_ids` / `attention_mask` are the left-padded prompts that were passed
    to generate(); `generated_ids` is its [B, S + new_tokens] output. For every
    row this strips the left padding and applies exactly the same
    `extract_completion_with_eos` logic the one-sample-at-a-time script used
    (truncate at the first EOS inclusive, detect has_eos, BOS handling).

    Returns a list of (actual_prompt_ids, completion_ids, has_eos, actual_prompt_len).
    """
    batch_size, padded_prompt_len = input_ids.shape
    prompt_lens = attention_mask.sum(dim=1).tolist()
    pieces = []
    for i in range(batch_size):
        plen = int(prompt_lens[i])
        row = generated_ids[i, padded_prompt_len - plen:]
        original_prompt_ids = input_ids[i, padded_prompt_len - plen:]
        completion_ids, has_eos, actual_prompt_len = extract_completion_with_eos(
            generated_ids=row,
            prompt_length=plen,
            eos_token_id=eos_token_id,
            check_bos=True,
            bos_token_id=bos_token_id,
            original_prompt_ids=original_prompt_ids,
        )
        pieces.append((row[:actual_prompt_len], completion_ids, bool(has_eos), int(actual_prompt_len)))
    return pieces


def _get_base_model(model):
    """Return the bare transformer (no lm_head) of a *ForCausalLM model.

    hidden_states from it are identical to those of the full model's forward,
    but the (very large for 150k-token vocabularies) logits are never built.
    """
    prefix = getattr(model, "base_model_prefix", "model")
    return getattr(model, prefix, model)


def _write_error_marker(split_dir: Path, idx: int, exc: BaseException):
    err_meta = {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "idx": int(idx),
        "skipped": True,
    }
    try:
        with open(split_dir / f"sample_{idx:06d}.json", "w") as f:
            json.dump(err_meta, f)
    except Exception:
        pass


def _write_sample(split_dir, idx, metadata, small_tensors, layer_buffers, row, seq_len, write_tensors):
    """Runs in a writer thread. Returns "saved" | "skipped" | "failed".

    Order matters for resumability: the safetensors file goes first (via a temp
    name + atomic rename) and the `.json` last, so an existing `.json` implies
    its tensors are complete.
    """
    base = split_dir / f"sample_{idx:06d}"
    try:
        if write_tensors:
            tensors = dict(small_tensors)
            for k, buf in enumerate(layer_buffers):
                # clone: detach from the shared [B, L, H] host buffer and make
                # the slice contiguous for safetensors.
                tensors[f"hidden_states_{k}"] = buf[row, :seq_len].clone()
            tmp = f"{base}.safetensors.tmp"
            save_file(tensors, tmp)
            os.replace(tmp, f"{base}.safetensors")
        tmp_json = f"{base}.json.tmp"
        with open(tmp_json, "w") as f:
            json.dump(metadata, f)
        os.replace(tmp_json, f"{base}.json")
        return "saved" if write_tensors else "skipped"
    except Exception as exc:  # disk full, quota, ...
        _write_error_marker(split_dir, idx, exc)
        print(f"sample_{idx:06d} write failed: {type(exc).__name__}: {exc}", flush=True)
        return "failed"


class _BatchExtractor:
    """Generates completions for batches of prompts and hands finished samples
    to background writer threads."""

    def __init__(
        self,
        model,
        tokenizer,
        generation_config,
        device,
        split_dir: Path,
        save_truncated: bool,
        forward_batch_size: int,
        writer_threads: int,
        max_inflight_batches: int,
    ):
        self.model = model
        self.base_model = _get_base_model(model)
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = device
        self.split_dir = split_dir
        self.save_truncated = save_truncated
        self.forward_batch_size = max(1, forward_batch_size)
        self.max_inflight = max(1, max_inflight_batches)
        self.pool = ThreadPoolExecutor(max_workers=max(1, writer_threads))
        self.inflight = deque()  # each entry: list[Future] for one forward sub-batch
        self.saved = 0
        self.skipped = 0
        self.failed = 0
        self.generated_tokens = 0
        self.done_samples = 0

    # -- bookkeeping ---------------------------------------------------------
    def _tally(self, status: str):
        if status == "saved":
            self.saved += 1
        elif status == "skipped":
            self.skipped += 1
        else:
            self.failed += 1

    def _drain_oldest(self):
        for fut in self.inflight.popleft():
            self._tally(fut.result())

    def drain_all(self):
        while self.inflight:
            self._drain_oldest()

    def close(self):
        self.drain_all()
        self.pool.shutdown(wait=True)

    def _free_cuda(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _fail(self, indices, exc):
        for idx in indices:
            _write_error_marker(self.split_dir, idx, exc)
            self.failed += 1
            self.done_samples += 1
            print(f"sample_{idx:06d} failed: {type(exc).__name__}: {exc}", flush=True)

    # -- work ----------------------------------------------------------------
    def process(self, indices, prompts, token_lists):
        """Process one batch. On any error the batch is split in halves and
        retried, so a bad/oversized sample is isolated down to batch size 1
        (where it gets an error-marker JSON, like the original per-sample loop)."""
        try:
            gen = self._generate(token_lists)
        except Exception as exc:
            self._free_cuda()
            if len(indices) == 1:
                self._fail(indices, exc)
                return
            mid = len(indices) // 2
            self.process(indices[:mid], prompts[:mid], token_lists[:mid])
            self.process(indices[mid:], prompts[mid:], token_lists[mid:])
            return

        input_ids, attention_mask, generated_ids = gen
        pieces = split_generated_batch(
            generated_ids,
            input_ids,
            attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        del generated_ids, input_ids, attention_mask
        self.generated_tokens += sum(len(p[1]) for p in pieces)

        # Group similar total lengths together for the forward pass (less padding).
        order = sorted(range(len(pieces)), key=lambda i: -(pieces[i][3] + len(pieces[i][1])))
        for start in range(0, len(order), self.forward_batch_size):
            sub = order[start:start + self.forward_batch_size]
            self._forward_and_submit(
                [indices[i] for i in sub],
                [prompts[i] for i in sub],
                [pieces[i] for i in sub],
            )

    def _generate(self, token_lists):
        pad_id = self.tokenizer.pad_token_id
        batch = len(token_lists)
        padded_len = max(len(t) for t in token_lists)
        input_ids = torch.full((batch, padded_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, padded_len), dtype=torch.long)
        for i, toks in enumerate(token_lists):
            input_ids[i, padded_len - len(toks):] = torch.tensor(toks, dtype=torch.long)
            attention_mask[i, padded_len - len(toks):] = 1
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            gen_kwargs = build_generation_kwargs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
                include_hidden_states=False,  # the teacher-forced forward below is the source of truth
            )
            generated = self.model.generate(**gen_kwargs)
            if hasattr(generated, "sequences"):
                generated = generated.sequences
        return input_ids, attention_mask, generated

    def _forward_and_submit(self, indices, prompts, pieces):
        try:
            self._forward_and_submit_impl(indices, prompts, pieces)
        except Exception as exc:
            self._free_cuda()
            if len(indices) == 1:
                self._fail(indices, exc)
                return
            mid = len(indices) // 2
            self._forward_and_submit(indices[:mid], prompts[:mid], pieces[:mid])
            self._forward_and_submit(indices[mid:], prompts[mid:], pieces[mid:])

    def _forward_and_submit_impl(self, indices, prompts, pieces):
        pad_id = self.tokenizer.pad_token_id
        fulls = [torch.cat([p[0], p[1]]) for p in pieces]
        lengths = [int(f.numel()) for f in fulls]
        max_len = max(lengths)

        # Right padding: real tokens sit at positions 0..n-1 exactly as in an
        # unbatched forward, and the causal mask hides the trailing pads.
        ids = torch.full((len(fulls), max_len), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(fulls), max_len), dtype=torch.long, device=self.device)
        for r, f in enumerate(fulls):
            ids[r, :f.numel()] = f
            mask[r, :f.numel()] = 1

        with torch.no_grad():
            outputs = self.base_model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            # One host copy per layer for the whole sub-batch.
            layer_buffers = [h.cpu() for h in outputs.hidden_states]
        del outputs, ids, mask
        num_layers = len(layer_buffers)

        futures = []
        for r, (idx, prompt, piece) in enumerate(zip(indices, prompts, pieces)):
            actual_prompt_ids, completion_ids, has_eos, actual_prompt_len = piece
            completion_text = self.tokenizer.decode(completion_ids.cpu(), skip_special_tokens=False)
            metadata = {
                "prompt": prompt,
                "completion": completion_text,
                "has_eos": bool(has_eos),
                "prompt_length": int(actual_prompt_len),
                "completion_length": int(completion_ids.numel()),
                "num_hidden_layers": num_layers,
            }
            small_tensors = {
                "prompt_ids": actual_prompt_ids.cpu().clone(),
                "completion_ids": completion_ids.cpu().clone(),
                "full_sequence_ids": fulls[r].cpu().clone(),
            }
            # Skip the heavy `.safetensors` for has_eos=False unless explicitly
            # requested; the trainer would loss-mask these positions anyway.
            write_tensors = bool(has_eos) or self.save_truncated
            futures.append(
                self.pool.submit(
                    _write_sample,
                    self.split_dir, idx, metadata, small_tensors,
                    layer_buffers, r, lengths[r], write_tensors,
                )
            )
        self.done_samples += len(indices)
        self.inflight.append(futures)
        # Bound host memory: at most `max_inflight` sub-batches of hidden states
        # are alive at once.
        while len(self.inflight) > self.max_inflight:
            self._drain_oldest()


def process_dataset(
    dataset,
    split,
    model,
    tokenizer,
    generation_config,
    output_dir,
    device,
    rank: int,
    world_size: int,
    save_truncated: bool = False,
    batch_size: int = 32,
    forward_batch_size: int = 16,
    writer_threads: int = 4,
    max_inflight_batches: int = 2,
):
    """Process a dataset and save hidden states.

    When `save_truncated=False` (default), samples whose generation did not
    terminate with EOS have their heavy `.safetensors` skipped on disk; only
    the small per-sample `.json` metadata is written so the truncation can
    still be audited after the fact. The trainer's `HiddenStatesDataset`
    discovers samples by globbing `sample_*.safetensors`, so skipped samples
    are automatically excluded from training without any reader-side change.

    The loop is resumable: indices whose `.json` already exists are skipped
    and counted against the existing tally. The `.json` is written last and
    atomically, so it also certifies that the sample's tensors are complete.
    Per-sample failures (CUDA OOM, model errors, etc.) are caught and recorded
    as an error-marker JSON so one bad sample doesn't kill the whole rank, and
    so a re-run picks up where the previous one left off. To retry an errored
    sample, delete its JSON before re-running.

    Work is sharded across ranks by stride (`range(rank, len, world_size)`);
    within a rank, prompts are processed longest-first in batches of
    `batch_size`.

    Returns:
        (saved, skipped, failed): saved is the number of samples written
        with a full safetensors tensor, skipped is the number of has_eos=False
        samples that wrote only the JSON metadata, failed is the number of
        samples that crashed and got an error-marker JSON.
    """
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    failed = 0

    # Resumability: a single directory scan instead of one stat per sample.
    existing = set()
    with os.scandir(split_dir) as it:
        for entry in it:
            m = _SAMPLE_JSON_RE.match(entry.name)
            if m:
                existing.add(int(m.group(1)))

    pending = []
    for idx in range(rank, len(dataset), world_size):
        if idx in existing:
            # Count against the existing tally. To retry an errored sample,
            # delete its JSON before re-running.
            try:
                with open(split_dir / f"sample_{idx:06d}.json") as f:
                    meta = json.load(f)
                if meta.get("error"):
                    failed += 1
                elif meta.get("has_eos", False):
                    saved += 1
                else:
                    skipped += 1
            except Exception:
                # Corrupt JSON — leave it; don't double-process.
                pass
        else:
            pending.append(idx)

    total_local = len(range(rank, len(dataset), world_size))
    print_main(f"[{split}] {total_local - len(pending)} already done on this rank, {len(pending)} to process")
    if not pending:
        return saved, skipped, failed

    # Tokenize every pending prompt once (same settings as the original
    # per-sample tokenize_function) and sort longest-first: keeps padding low
    # and surfaces worst-case memory use in the very first batch.
    prompts_all = [dataset[idx]["prompt"] for idx in pending]
    token_lists_all = tokenizer(
        prompts_all,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
        padding=False,
    )["input_ids"]
    order = sorted(range(len(pending)), key=lambda k: -len(token_lists_all[k]))

    extractor = _BatchExtractor(
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        device=device,
        split_dir=split_dir,
        save_truncated=save_truncated,
        forward_batch_size=forward_batch_size,
        writer_threads=writer_threads,
        max_inflight_batches=max_inflight_batches,
    )

    progress = tqdm(total=len(pending), desc=f"Extracting {split}") if is_main_process() else None
    start_time = time.perf_counter()
    try:
        for b in range(0, len(order), max(1, batch_size)):
            chunk = order[b:b + max(1, batch_size)]
            before = extractor.done_samples
            extractor.process(
                [pending[k] for k in chunk],
                [prompts_all[k] for k in chunk],
                [token_lists_all[k] for k in chunk],
            )
            if progress is not None:
                progress.update(extractor.done_samples - before)
            # A systematic failure (bad path, missing weights, OOM at batch
            # size 1) would otherwise litter the cache with error markers.
            if extractor.failed >= 20:
                extractor.drain_all()  # successes are tallied asynchronously
                if extractor.saved + extractor.skipped == 0:
                    raise RuntimeError(
                        f"[rank {rank}] the first {extractor.failed} samples all failed; aborting "
                        "instead of writing error markers for the whole split "
                        "(delete the sample_*.json error markers before re-running)."
                    )
    finally:
        extractor.close()
        if progress is not None:
            progress.close()

    elapsed = time.perf_counter() - start_time
    print(
        f"[rank {rank}] {split}: {len(pending)} samples in {elapsed:.0f}s "
        f"({len(pending) / max(elapsed, 1e-9):.2f} samples/s, "
        f"{extractor.generated_tokens / max(elapsed, 1e-9):.0f} generated tokens/s)",
        flush=True,
    )

    return saved + extractor.saved, skipped + extractor.skipped, failed + extractor.failed


def aggregate_count(local_count: int, backend: Optional[str], device: torch.device) -> int:
    if not is_distributed():
        return local_count

    reduce_device = device if backend == "nccl" else torch.device("cpu")
    tensor = torch.tensor(local_count, dtype=torch.long, device=reduce_device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


if __name__ == "__main__":
    main()
