"""Step 1 (GPU, one process per card): generate -> checkpoint states + token stats -> forced-answer scores.

    CUDA_VISIBLE_DEVICES=0 python cluster/msig_worker.py --run-dir $RUN_DIR --split test --shard 0 --nshards 4
Resumable per chunk (finished chunks are skipped).  Do not change --chunk-problems / --limit / --subset-seed between a run and its resume.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import io as sio  # noqa: E402
from miol.signals.mathdata import SYSTEM_PROMPT_MATH, read_problems  # noqa: E402
from miol.signals.scoring import DEFAULT_CUE  # noqa: E402
from miol.signals.worker import Worker, chunk_done, write_chunk  # noqa: E402


def select_problems(run_dir, split, limit, subset_seed):
    probs = read_problems(Path(run_dir) / f"problems_{split}.jsonl")
    if limit and limit < len(probs):
        pick = np.random.RandomState(subset_seed).permutation(len(probs))[:limit]
        probs = [probs[i] for i in sorted(pick)]
    return probs


def main(argv=None, tok=None, model=None, worker_hook=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--chunk-problems", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cue", default=DEFAULT_CUE)
    ap.add_argument("--system", default=SYSTEM_PROMPT_MATH)
    ap.add_argument("--gen-batch", type=int, default=0, help="prompts per generate() call (0 = the whole chunk, halved on OOM)")
    ap.add_argument("--max-ckpt", type=int, default=0, help="cap on checkpoints per trace (long-CoT models); 0 = every blank line")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20, help="0 disables top-k")
    ap.add_argument("--mode", default="lines")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    probs = select_problems(a.run_dir, a.split, a.limit, a.subset_seed)
    chunks = [probs[i:i + a.chunk_problems] for i in range(0, len(probs), a.chunk_problems)]
    mine = [c for c in range(len(chunks)) if c % a.nshards == a.shard]
    todo = [c for c in mine if not chunk_done(out, a.split, c)]
    print(f"[worker {a.split} s{a.shard}] {len(probs)} problems, {len(chunks)} chunks, mine {len(mine)}, to do {len(todo)}", flush=True)
    if not todo:
        return
    tok = tok or sio.load_tokenizer(a.model_id)
    model = model or sio.load_model(a.model_id, a.device)
    w = Worker(model, tok, a.device, a.system, a.cue, a.max_new, mode=a.mode, stride=a.stride, seed=a.seed,
               gen_kw=dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k), gen_batch=a.gen_batch, max_ckpt=a.max_ckpt,
               log=sio.log_to(f"[worker {a.split} s{a.shard}]"))
    if worker_hook:
        worker_hook(w)
    t0 = time.time()
    for n, c in enumerate(todo):
        ck, tr, states, ids = w.process_chunk(a.split, chunks[c], c)
        write_chunk(out, a.split, c, ck, tr, states, ids)
        el = (time.time() - t0) / 60
        print(f"[worker {a.split} s{a.shard}] chunk {c} saved ({n + 1}/{len(todo)}), {el:.1f} min, "
              f"eta {el / (n + 1) * (len(todo) - n - 1):.1f} min", flush=True)
    print(f"[worker {a.split} s{a.shard}] done", flush=True)


if __name__ == "__main__":
    main()
