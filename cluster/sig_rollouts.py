"""Stage (optional, GPU, sharded): empirical answer distribution by RESAMPLING the rest of the solution.

For a subset of traces and for every checkpoint, continue the written prefix M times (sampling T=0.7, top_p=0.8, top_k=20,
the model's own recommended settings) to the end, parse each final answer and record
    roll_H      entropy (nats) of the distribution of final answers over the M continuations
    roll_pgold  fraction of continuations that end on the gold answer
    roll_pfinal fraction that end on the trace's own final answer
This is the expensive, model-honest cousin of the forced-answer score a_k; its job is VALIDATION (does a_k track what the model
would really go on to answer?) and a second probe target.  The 'final' checkpoint is deterministic (no generation).

    CUDA_VISIBLE_DEVICES=0 python cluster/sig_rollouts.py --run-dir $RUN_DIR --split eval --shard 0 --nshards 4 --n-traces 600 --m 4
"""
import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import io as sio  # noqa: E402
from miol.signals.text_utils import parse_final_answer  # noqa: E402

NAN = float("nan")


def entropy(answers):
    c = Counter(answers)
    n = sum(c.values())
    return -sum((v / n) * math.log(v / n) for v in c.values())


@torch.no_grad()
def rollout_batch(model, tok, ctxs, pad_id, m, max_new, device, gen_kw):
    B, Lmax = len(ctxs), max(len(c) for c in ctxs)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    mask = torch.zeros((B, Lmax), dtype=torch.long)
    for r, s in enumerate(ctxs):
        ids[r, Lmax - len(s):] = torch.tensor(s)
        mask[r, Lmax - len(s):] = 1
    out = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new, do_sample=True,
                         num_return_sequences=m, pad_token_id=pad_id, repetition_penalty=1.0, **gen_kw)
    new = out[:, Lmax:].tolist()
    return [[tok.decode(new[r * m + j], skip_special_tokens=True) for j in range(m)] for r in range(B)]


def main(argv=None, tok=None, model=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-traces", type=int, default=600, help="random subset of traces that get rollouts (0 = all)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=4, help="continuations per checkpoint")
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--rows-per-batch", type=int, default=12, help="checkpoints per generate() call (x m sequences)")
    ap.add_argument("--chunk-samples", type=int, default=20)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    parts = out / "roll_parts"
    parts.mkdir(exist_ok=True)
    ck = sio.read_table(out / f"ckpt_{a.split}.csv")
    samples = sorted(ck["sample"].unique())
    if a.n_traces and a.n_traces < len(samples):
        pick = np.random.RandomState(a.subset_seed).permutation(len(samples))[:a.n_traces]
        samples = sorted(samples[i] for i in pick)
    mine = [s for i, s in enumerate(samples) if i % a.nshards == a.shard]
    tok = tok or sio.load_tokenizer(a.model_id)
    model = model or sio.load_model(a.model_id, a.device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    gen_kw = dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    from miol.signals.scoring import AnswerScorer  # only for the OOM-halving helper
    helper = AnswerScorer.__new__(AnswerScorer)
    split_dir = sio.hs_dir(a.run_dir, a.split)
    t0, n_done = time.time(), 0
    chunks = [mine[i:i + a.chunk_samples] for i in range(0, len(mine), a.chunk_samples)]
    for ci, sids in enumerate(chunks):
        part = parts / f"roll_{a.split}_s{a.shard}of{a.nshards}_c{ci:05d}.csv"
        if part.exists():
            continue
        rows, work = [], []                      # work: (ctx_ids, prefix_text, sample, k, gold, final)
        for sid in sids:
            from safetensors import safe_open
            with safe_open(str(split_dir / f"sample_{sid:06d}.safetensors"), "pt") as f:
                p_ids, c_ids = f.get_tensor("prompt_ids").tolist(), f.get_tensor("completion_ids").tolist()
            for r in ck[ck["sample"] == sid].itertuples(index=False):
                fin = r.final if isinstance(r.final, str) and r.final != "" else None
                if r.kind == "final":                                             # nothing left to write: deterministic
                    rows.append({"sample": sid, "k": r.k, "roll_H": 0.0, "roll_pgold": float(r.correct),
                                 "roll_pfinal": 1.0 if fin is not None else NAN})
                    continue
                ctx = p_ids + c_ids[:r.pos + 1]
                work.append((ctx, tok.decode(c_ids[:r.pos + 1]) if r.pos >= 0 else "", sid, int(r.k), str(r.gold), fin))
        work.sort(key=lambda w: len(w[0]))

        def run(idx):
            texts = rollout_batch(model, tok, [work[i][0] for i in idx], pad_id, a.m, a.max_new, a.device, gen_kw)
            res = []
            for i, conts in zip(idx, texts):
                _, pref, sid, k, gold, fin = work[i]
                ans = [parse_final_answer(pref + c) or "NONE" for c in conts]
                res.append({"sample": sid, "k": k, "roll_H": entropy(ans),
                            "roll_pgold": float(np.mean([x == gold for x in ans])),
                            "roll_pfinal": float(np.mean([x == fin for x in ans])) if fin is not None else NAN})
            return res

        idxs = list(range(len(work)))
        for b in range(0, len(idxs), a.rows_per_batch):
            rows += AnswerScorer._guard(helper, run, idxs[b:b + a.rows_per_batch])
        df = pd.DataFrame(rows).sort_values(["sample", "k"])
        df.to_csv(str(part) + ".tmp", index=False)
        Path(str(part) + ".tmp").replace(part)
        n_done += len(df)
        print(f"[roll {a.split} s{a.shard}] chunk {ci + 1}/{len(chunks)}  {n_done} rows  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[roll {a.split} s{a.shard}] done", flush=True)


if __name__ == "__main__":
    main()
