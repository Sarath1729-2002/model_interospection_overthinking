"""
Tests for the batched hidden-state extraction and the vectorized target helpers.

Everything runs offline on CPU: the "model" is a tiny randomly-initialised
Qwen2 and the tokenizer is built locally, so no checkpoint or dataset download
is needed. Decoding is greedy so batched and unbatched runs are comparable.

The reference implementations below are verbatim copies of the code the
vectorized/batched versions replaced.
"""

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from miol.data.hidden_states_dataset import HiddenStatesDataset
from miol.extract_hidden_states import (
    _BatchExtractor,
    process_dataset,
    split_generated_batch,
)
from miol.model.generation_utils import (
    build_generation_kwargs,
    compute_percentage_targets,
    compute_remaining_token_targets,
    compute_soft_class_targets,
    extract_completion_with_eos,
)


# --------------------------------------------------------------------------
# Reference (original loop) implementations
# --------------------------------------------------------------------------

def ref_remaining(sequence_length, prompt_length, has_eos):
    rc = torch.full((sequence_length,), -100, dtype=torch.long)
    if has_eos:
        for j in range(prompt_length, sequence_length):
            rc[j] = sequence_length - j - 1
    return rc


def ref_percentage(sequence_length, prompt_length, has_eos):
    targets = torch.full((sequence_length,), -100.0, dtype=torch.float32)
    if not has_eos:
        return targets
    completion_length = sequence_length - prompt_length
    if completion_length <= 0:
        return targets
    for j in range(prompt_length, sequence_length):
        targets[j] = (sequence_length - j - 1) / max(completion_length - 1, 1)
    return targets


def ref_soft(sequence_length, prompt_length, has_eos, sigma=0.15, num_bins=3, anchors=None):
    if anchors is None:
        anchors = torch.linspace(0, 1, num_bins).tolist()
    anchors = torch.tensor(anchors)
    targets = torch.full((sequence_length, num_bins), -100.0, dtype=torch.float32)
    if not has_eos:
        return targets
    completion_length = sequence_length - prompt_length
    if completion_length <= 0:
        return targets
    for j in range(prompt_length, sequence_length):
        pct = (j - prompt_length) / max(completion_length - 1, 1)
        class_probs = torch.exp(-0.5 * ((pct - anchors) / sigma) ** 2)
        class_probs = class_probs / class_probs.sum()
        targets[j] = class_probs
    return targets


SHAPES = [
    # (sequence_length, prompt_length)
    (1, 0), (1, 1), (2, 1), (5, 5), (5, 6), (8, 3), (30, 7), (331, 124), (1100, 150), (64, 63),
]


@pytest.mark.parametrize("seq_len,prompt_len", SHAPES)
@pytest.mark.parametrize("has_eos", [True, False])
class TestVectorizedTargets:
    def test_remaining_counts_identical(self, seq_len, prompt_len, has_eos):
        new = compute_remaining_token_targets(seq_len, prompt_len, has_eos, torch.device("cpu"))
        assert torch.equal(new, ref_remaining(seq_len, prompt_len, has_eos))

    def test_percentage_identical(self, seq_len, prompt_len, has_eos):
        new = compute_percentage_targets(seq_len, prompt_len, has_eos, torch.device("cpu"))
        assert torch.equal(new, ref_percentage(seq_len, prompt_len, has_eos))

    @pytest.mark.parametrize("num_bins", [2, 3, 5, 7, 9])
    def test_soft_targets_match(self, seq_len, prompt_len, has_eos, num_bins):
        new = compute_soft_class_targets(
            seq_len, prompt_len, has_eos, torch.device("cpu"), sigma=0.15, num_bins=num_bins
        )
        ref = ref_soft(seq_len, prompt_len, has_eos, sigma=0.15, num_bins=num_bins)
        # masked entries (-100) must match exactly; probabilities to float32 precision
        assert torch.equal(new == -100.0, ref == -100.0)
        torch.testing.assert_close(new, ref, rtol=1e-5, atol=1e-7)


def test_soft_targets_custom_anchors_and_sigma():
    anchors = [0.0, 0.2, 0.9]
    new = compute_soft_class_targets(40, 10, True, torch.device("cpu"), sigma=0.3, num_bins=3, anchors=anchors)
    ref = ref_soft(40, 10, True, sigma=0.3, num_bins=3, anchors=anchors)
    torch.testing.assert_close(new, ref, rtol=1e-5, atol=1e-7)
    # every completion row is a probability distribution
    assert torch.allclose(new[10:].sum(dim=1), torch.ones(30), atol=1e-6)


def test_repetition_penalty_is_opt_in():
    ids = torch.tensor([[1, 2, 3]])
    base = dict(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        pad_token_id=0, eos_token_id=2,
    )
    without = build_generation_kwargs(generation_config={"do_sample": True}, **base)
    assert "repetition_penalty" not in without
    with_pen = build_generation_kwargs(generation_config={"do_sample": True, "repetition_penalty": 1.0}, **base)
    assert with_pen["repetition_penalty"] == 1.0


def test_hidden_states_dataset_remaining_counts_match_reference(tmp_path):
    from safetensors.torch import save_file

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    prompt_len, completion_len, n_layers, hidden = 7, 11, 3, 4
    seq_len = prompt_len + completion_len
    for name, has_eos in (("sample_000000", True), ("sample_000001", False)):
        tensors = {
            "prompt_ids": torch.arange(prompt_len),
            "completion_ids": torch.arange(completion_len),
            "full_sequence_ids": torch.arange(seq_len),
        }
        for k in range(n_layers):
            tensors[f"hidden_states_{k}"] = torch.randn(seq_len, hidden)
        save_file(tensors, str(split_dir / f"{name}.safetensors"))
        (split_dir / f"{name}.json").write_text(json.dumps({
            "prompt": "p", "completion": "c", "has_eos": has_eos,
            "prompt_length": prompt_len, "completion_length": completion_len,
            "num_hidden_layers": n_layers,
        }))
    ds = HiddenStatesDataset(str(tmp_path), split="train", hidden_layer="all")
    assert len(ds) == 2
    eos_item, trunc_item = ds[0], ds[1]
    assert torch.equal(eos_item["remaining_token_counts"], ref_remaining(seq_len, prompt_len, True))
    assert torch.equal(trunc_item["remaining_token_counts"], ref_remaining(seq_len, prompt_len, False))
    assert eos_item["hidden_states"].shape == (seq_len, n_layers * hidden)


# --------------------------------------------------------------------------
# split_generated_batch
# --------------------------------------------------------------------------

def test_split_generated_batch_strips_left_padding_and_truncates_at_eos():
    pad, eos = 0, 9
    # prompts: [5,6,7] and [8] -> left padded to length 3
    input_ids = torch.tensor([[5, 6, 7], [0, 0, 8]])
    attention_mask = torch.tensor([[1, 1, 1], [0, 0, 1]])
    generated = torch.tensor([
        [5, 6, 7, 1, 2, eos, pad, pad],     # eos after 3 tokens, then padding
        [0, 0, 8, 3, 4, 3, 4, 3],           # never emits eos (truncated)
    ])
    pieces = split_generated_batch(generated, input_ids, attention_mask, eos_token_id=eos, bos_token_id=None)
    p0, c0, e0, l0 = pieces[0]
    assert p0.tolist() == [5, 6, 7] and c0.tolist() == [1, 2, eos] and e0 and l0 == 3
    p1, c1, e1, l1 = pieces[1]
    assert p1.tolist() == [8] and c1.tolist() == [3, 4, 3, 4, 3] and not e1 and l1 == 1


# --------------------------------------------------------------------------
# End-to-end extraction vs. the original one-sample-at-a-time pipeline
# --------------------------------------------------------------------------

WORDS = [f"w{i}" for i in range(40)]


def make_tokenizer(eos_token="<eos>"):
    specials = ["<pad>", "<eos>", "<unk>"]
    vocab = {t: i for i, t in enumerate(specials + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token=eos_token, unk_token="<unk>"
    )
    fast.padding_side = "left"
    return fast


class ListDataset(list):
    """Minimal stand-in for the HF dataset: len() and ds[i]["prompt"]."""


def reference_extract(model, tokenizer, prompt, generation_config):
    """Verbatim logic of the original per-sample loop in process_dataset."""
    toks = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=512, padding=False)["input_ids"]
    ids = torch.tensor([toks])
    mask = torch.ones_like(ids)
    prompt_len = ids.shape[1]
    with torch.no_grad():
        gen_kwargs = build_generation_kwargs(
            input_ids=ids, attention_mask=mask, generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            include_hidden_states=True,
        )
        out = model.generate(**gen_kwargs)
        generated = out.sequences[0]
        completion_ids, has_eos, actual_prompt_len = extract_completion_with_eos(
            generated_ids=generated, prompt_length=prompt_len, eos_token_id=tokenizer.eos_token_id,
            check_bos=True, bos_token_id=tokenizer.bos_token_id, original_prompt_ids=ids[0],
        )
        full = torch.cat([generated[:actual_prompt_len], completion_ids])
        o = model(
            input_ids=full.unsqueeze(0), attention_mask=torch.ones((1, len(full)), dtype=torch.long),
            output_hidden_states=True, return_dict=True,
        )
    return completion_ids, bool(has_eos), actual_prompt_len, full, [h[0] for h in o.hidden_states]


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    vocab = len(WORDS) + 3
    cfg = Qwen2Config(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rms_norm_eps=1e-6, tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    # spread the logits out so greedy argmax is robust to float noise
    model.lm_head.weight.data *= 30

    g = torch.Generator().manual_seed(1)
    prompts = []
    for n in (2, 9, 4, 7, 3, 12, 5, 8, 6, 10, 2):
        prompts.append(" ".join(WORDS[i] for i in torch.randint(0, len(WORDS), (n,), generator=g).tolist()))

    gen_cfg = {"max_new_tokens": 24, "do_sample": False}

    # Pick an EOS word that the (deterministic) greedy continuations actually
    # contain for some prompts but not others, so both terminated and
    # truncated samples occur.
    plain = make_tokenizer()
    conts = []
    for p in prompts:
        ids = torch.tensor([plain(p)["input_ids"]])
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=24, do_sample=False, pad_token_id=0)
        conts.append(out[0, ids.shape[1]:].tolist())
    best, best_count = None, -1
    for tok_id in range(3, vocab):
        occurs = sum(1 for c in conts if tok_id in c[1:])
        if 0 < occurs < len(prompts) and occurs > best_count:
            best, best_count = tok_id, occurs
    assert best is not None, "could not find an EOS candidate that splits the samples"
    tokenizer = make_tokenizer(eos_token=plain.convert_ids_to_tokens(best))

    refs = [reference_extract(model, tokenizer, p, gen_cfg) for p in prompts]
    assert any(r[1] for r in refs) and not all(r[1] for r in refs)
    return {"model": model, "tokenizer": tokenizer, "prompts": prompts, "gen_cfg": gen_cfg, "refs": refs}


def run_extraction(world, out_dir, split="train", **kwargs):
    dataset = ListDataset({"prompt": p, "completion": ""} for p in world["prompts"])
    params = dict(
        batch_size=4, forward_batch_size=2, writer_threads=2, max_inflight_batches=1,
        rank=0, world_size=1, save_truncated=False,
    )
    params.update(kwargs)
    return process_dataset(
        dataset=dataset, split=split, model=world["model"], tokenizer=world["tokenizer"],
        generation_config=world["gen_cfg"], output_dir=out_dir, device=torch.device("cpu"), **params,
    )


def test_batched_extraction_matches_reference(world, tmp_path):
    refs = world["refs"]
    saved, skipped, failed = run_extraction(world, tmp_path)
    n_eos = sum(1 for r in refs if r[1])
    assert (saved, skipped, failed) == (n_eos, len(refs) - n_eos, 0)

    split_dir = tmp_path / "train"
    for idx, (completion_ids, has_eos, prompt_len, full, hs) in enumerate(refs):
        meta = json.loads((split_dir / f"sample_{idx:06d}.json").read_text())
        assert meta["has_eos"] == has_eos
        assert meta["prompt_length"] == prompt_len
        assert meta["completion_length"] == len(completion_ids)
        assert meta["num_hidden_layers"] == len(hs)
        assert meta["prompt"] == world["prompts"][idx]
        st = split_dir / f"sample_{idx:06d}.safetensors"
        assert st.exists() == has_eos          # truncated samples keep only the JSON
        if has_eos:
            t = load_file(str(st))
            assert torch.equal(t["completion_ids"], completion_ids)
            assert torch.equal(t["full_sequence_ids"], full)
            assert torch.equal(t["prompt_ids"], full[:prompt_len])
            for k, h in enumerate(hs):
                torch.testing.assert_close(t[f"hidden_states_{k}"], h, rtol=1e-4, atol=1e-4)
    # no stray temp files
    assert not list(split_dir.glob("*.tmp"))


def test_output_is_loadable_by_hidden_states_dataset(world, tmp_path):
    run_extraction(world, tmp_path)
    ds = HiddenStatesDataset(str(tmp_path), split="train", hidden_layer="all")
    n_eos = sum(1 for r in world["refs"] if r[1])
    assert len(ds) == n_eos
    item = ds[0]
    assert item["hidden_states"].shape[-1] == 32 * 4        # hidden_size * (3 layers + embeddings)
    assert item["has_eos"] is True
    assert item["remaining_token_counts"][-1] == 0           # decrements to 0 at EOS


def test_resume_skips_finished_samples(world, tmp_path, monkeypatch):
    first = run_extraction(world, tmp_path)

    def boom(*a, **k):
        raise AssertionError("generate() must not run again on a finished cache")

    monkeypatch.setattr(world["model"], "generate", boom)
    assert run_extraction(world, tmp_path) == first


def test_rank_sharding_covers_every_index_once(world, tmp_path):
    for rank in (0, 1):
        run_extraction(world, tmp_path, rank=rank, world_size=2)
    jsons = sorted(int(p.stem.split("_")[1]) for p in (tmp_path / "train").glob("sample_*.json"))
    assert jsons == list(range(len(world["prompts"])))


def test_oversized_batches_are_split_until_they_fit(world, tmp_path, monkeypatch):
    real_generate = world["model"].generate

    def picky_generate(**kwargs):
        if kwargs["input_ids"].shape[0] > 2:
            raise RuntimeError("CUDA out of memory (simulated)")
        return real_generate(**kwargs)

    monkeypatch.setattr(world["model"], "generate", picky_generate)
    saved, skipped, failed = run_extraction(world, tmp_path, batch_size=8)
    n_eos = sum(1 for r in world["refs"] if r[1])
    assert (saved, skipped, failed) == (n_eos, len(world["refs"]) - n_eos, 0)


def test_one_bad_sample_gets_error_marker_and_others_survive(world, tmp_path, monkeypatch):
    bad_prompt = world["prompts"][3]
    real_impl = _BatchExtractor._forward_and_submit_impl

    def flaky(self, indices, prompts, pieces):
        if bad_prompt in prompts:
            raise RuntimeError("simulated per-sample failure")
        return real_impl(self, indices, prompts, pieces)

    monkeypatch.setattr(_BatchExtractor, "_forward_and_submit_impl", flaky)
    saved, skipped, failed = run_extraction(world, tmp_path)
    assert failed == 1
    assert saved + skipped == len(world["prompts"]) - 1
    marker = json.loads((tmp_path / "train" / "sample_000003.json").read_text())
    assert marker["error_type"] == "RuntimeError" and marker["skipped"] is True
    assert not (tmp_path / "train" / "sample_000003.safetensors").exists()


def test_systematic_failure_aborts_instead_of_poisoning_the_cache(world, tmp_path, monkeypatch):
    big = ListDataset({"prompt": world["prompts"][0], "completion": ""} for _ in range(60))
    monkeypatch.setattr(
        _BatchExtractor, "_generate",
        lambda self, token_lists: (_ for _ in ()).throw(RuntimeError("weights missing")),
    )
    with pytest.raises(RuntimeError, match="all failed"):
        process_dataset(
            dataset=big, split="train", model=world["model"], tokenizer=world["tokenizer"],
            generation_config=world["gen_cfg"], output_dir=tmp_path, device=torch.device("cpu"),
            rank=0, world_size=1, batch_size=4,
        )
    # aborted early: nowhere near 60 markers were written
    assert len(list((tmp_path / "train").glob("sample_*.json"))) < 60
