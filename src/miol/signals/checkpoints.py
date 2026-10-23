"""Checkpoint tables: where inside each cached completion we cut, plus the exact labels that need no model.

For every usable cached sample (has_eos, tensors on disk) we build rows
    k = 0           the prompt end (nothing of the completion written yet; hidden state at prompt_length-1)
    k = 1 .. m-1    every blank-line boundary ("\\n\\n", also ".\\n\\n" etc.) - one whole step of the solution
    k = m           the last content token before <|im_end|>  (kind "final")
and, with mode="both", an extra checkpoint every `stride` tokens.
`pos` is the completion-token index of the checkpoint (-1 = prompt end); the hidden state of that checkpoint is
row  prompt_length + pos  of the cached tensors (row prompt_length - 1 for the prompt end).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from miol.signals.text_utils import (boundary_token_ids, parse_final_answer, parse_gold, prefix_features,
                                     question_from_prompt)


def checkpoint_positions(comp_ids: List[int], boundary_ids: set, eos_id: Optional[int], has_eos: bool,
                         mode: str = "lines", stride: int = 8, max_ckpt: int = 0) -> List[Tuple[int, str]]:
    n = len(comp_ids)
    last = n - 1
    if has_eos and n >= 1 and eos_id is not None and comp_ids[-1] == eos_id:
        last = n - 2
    kinds: Dict[int, str] = {-1: "prompt"}
    if last < 0:
        return sorted(kinds.items())
    if mode in ("lines", "both"):
        for p in range(last + 1):
            if comp_ids[p] in boundary_ids:
                kinds.setdefault(p, "line")
    if mode in ("stride", "both"):
        p = stride - 1
        while p < last:
            kinds.setdefault(p, "stride")
            p += stride
    kinds[last] = "final"
    out = sorted(kinds.items())
    if max_ckpt and len(out) > max_ckpt:            # long traces: keep the prompt end, the final checkpoint and an even spread between
        inner = out[1:-1]
        take = max_ckpt - 2
        idx = sorted({int(round(i * (len(inner) - 1) / max(take - 1, 1))) for i in range(take)}) if take > 0 else []
        out = [out[0]] + [inner[i] for i in idx] + [out[-1]]
    return out


def gsm8k_gold_map(hf_dataset: str = "openai/gsm8k", hf_config: str = "main") -> Dict[str, str]:
    """question text -> gold answer string, over train+test (looked up by question so sample ids never matter)."""
    from datasets import load_dataset
    out = {}
    for split in ("train", "test"):
        ds = load_dataset(hf_dataset, hf_config, split=split)
        for r in ds:
            g = parse_gold(r["answer"])
            if g is not None:
                out[r["question"].strip()] = g
    return out


def list_samples(split_dir: Path) -> List[int]:
    ids = []
    for p in Path(split_dir).glob("sample_*.json"):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except ValueError:
            pass
    return sorted(ids)


def build_split_tables(split_dir, split: str, tok, gold_map: Dict[str, str], eos_id: int, boundary_ids: set,
                       mode: str = "lines", stride: int = 8, limit: int = 0, subset_seed: int = 0, log=print):
    """-> (ckpt_df, traces_df).  Rows are ordered by (sample, k); `row` is the index into the states array."""
    from safetensors import safe_open
    split_dir = Path(split_dir)
    ids = list_samples(split_dir)
    if limit and limit < len(ids):
        perm = np.random.RandomState(subset_seed).permutation(len(ids))[:limit]
        ids = sorted(ids[i] for i in perm)
    rows, traces = [], []
    skipped = {"no_tensors": 0, "no_eos": 0, "no_gold": 0, "error": 0}
    for n_done, sid in enumerate(ids):
        base = split_dir / f"sample_{sid:06d}"
        try:
            meta = json.load(open(base.with_suffix(".json")))
            if "completion" not in meta:
                skipped["error"] += 1
                continue
            if not meta.get("has_eos", False):
                skipped["no_eos"] += 1
                continue
            if not base.with_suffix(".safetensors").exists():
                skipped["no_tensors"] += 1
                continue
            with safe_open(str(base.with_suffix(".safetensors")), "pt") as f:
                comp_ids = f.get_tensor("completion_ids").tolist()
            q = question_from_prompt(meta["prompt"])
            gold = gold_map.get(q) if q is not None else None
            if gold is None:
                skipped["no_gold"] += 1
                continue
            pl, n_tok = int(meta["prompt_length"]), int(meta["completion_length"])
            assert n_tok == len(comp_ids), (sid, n_tok, len(comp_ids))
            final = parse_final_answer(meta["completion"])
            cps = checkpoint_positions(comp_ids, boundary_ids, eos_id, True, mode, stride)
            for k, (pos, kind) in enumerate(cps):
                text = tok.decode(comp_ids[:pos + 1]) if pos >= 0 else ""
                feats = prefix_features(text, gold, final)
                rows.append({"split": split, "sample": sid, "k": k, "pos": pos, "kind": kind, "n_tok": n_tok,
                             "prompt_len": pl, "gold": gold, "final": final if final is not None else "",
                             "correct": int(final is not None and final == gold),
                             "tail": text[-200:].replace("\n", "⏎"), **feats})
            traces.append({"split": split, "sample": sid, "n_tok": n_tok, "prompt_len": pl, "gold": gold,
                           "final": final if final is not None else "", "final_ok": int(final is not None),
                           "correct": int(final is not None and final == gold), "n_ckpt": len(cps)})
        except Exception as exc:  # a bad sample must not kill the whole table
            skipped["error"] += 1
            log(f"  sample {sid}: {type(exc).__name__}: {exc}")
        if (n_done + 1) % 1000 == 0:
            log(f"  {split}: {n_done + 1}/{len(ids)} samples")
    ckpt = pd.DataFrame(rows)
    ckpt.insert(0, "row", np.arange(len(ckpt)))
    ckpt["rem"] = ckpt["n_tok"] - (ckpt["pos"] + 1)          # completion tokens still to be written
    log(f"{split}: {len(traces)} traces, {len(ckpt)} checkpoints; skipped {skipped}")
    return ckpt, pd.DataFrame(traces)
