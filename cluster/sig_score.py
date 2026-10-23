"""Stage 2 (GPU, one process per GPU): score the answer distribution at every checkpoint.

    CUDA_VISIBLE_DEVICES=0 python cluster/sig_score.py --run-dir $RUN_DIR --split eval --shard 0 --nshards 4
(run_signals.sbatch starts one per GPU)    Resumable: finished chunks (<out>/score_parts/...) are skipped.

For each checkpoint:  a = log P(gold | prefix + cue),  f = log P(own final answer | prefix + cue)  (f := a when they are
the same string, NaN when the trace has no parsable answer),  forced = greedy answer,  ent1/p1max = confidence of the first
answer token.  The cue is appended after the written prefix, exactly as if the model were asked to answer right now.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import io as sio  # noqa: E402
from miol.signals.scoring import DEFAULT_CUE, AnswerScorer  # noqa: E402


def read_ids(split_dir, sid):
    from safetensors import safe_open
    with safe_open(str(Path(split_dir) / f"sample_{sid:06d}.safetensors"), "pt") as f:
        return f.get_tensor("prompt_ids").tolist(), f.get_tensor("completion_ids").tolist()


def score_samples(scorer: AnswerScorer, split_dir, ck: pd.DataFrame) -> pd.DataFrame:
    """ck: the checkpoint rows of a set of samples.  -> one row per checkpoint with the score columns."""
    ctxs, gold_a, final_a, keys = [], [], [], []
    cache = {}
    for sid, g in ck.groupby("sample", sort=False):
        p_ids, c_ids = read_ids(split_dir, int(sid))
        for r in g.itertuples(index=False):
            ctxs.append(scorer.context(p_ids, c_ids, int(r.pos)))
            gold_a.append(str(r.gold))
            final_a.append(str(r.final) if isinstance(r.final, str) and r.final != "" else None)
            keys.append((int(sid), int(r.k)))
    sg = scorer.score(ctxs, gold_a)
    # own-final scoring only where it differs from gold (otherwise f = a)
    need = [i for i in range(len(ctxs)) if final_a[i] is not None and final_a[i] != gold_a[i]]
    sf = scorer.score([ctxs[i] for i in need], [final_a[i] for i in need]) if need else []
    fmap = dict(zip(need, sf))
    forced = scorer.greedy(ctxs)
    rows = []
    for i, (sid, k) in enumerate(keys):
        a = sg[i]
        if final_a[i] is None:
            f = float("nan")
        elif final_a[i] == gold_a[i]:
            f = a["a"]
        else:
            f = fmap[i]["a"]
        rows.append({"sample": sid, "k": k, "a": a["a"], "a_digits": a["a_digits"], "term": a["term"],
                     "ent1": a["ent1"], "p1max": a["p1max"], "f": f, "forced": forced[i]})
    return pd.DataFrame(rows)


def main(argv=None, tok=None, model=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cue", default=DEFAULT_CUE)
    ap.add_argument("--chunk-samples", type=int, default=120)
    ap.add_argument("--token-budget", type=int, default=24000)
    ap.add_argument("--max-batch", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    parts = out / "score_parts"
    parts.mkdir(exist_ok=True)
    ck = sio.read_table(out / f"ckpt_{a.split}.csv")
    samples = sorted(ck["sample"].unique())
    mine = [s for i, s in enumerate(samples) if i % a.nshards == a.shard]
    tok = tok or sio.load_tokenizer(a.model_id)
    model = model or sio.load_model(a.model_id, a.device)
    scorer = AnswerScorer(model, tok, a.device, a.cue, token_budget=a.token_budget, max_batch=a.max_batch)
    chunks = [mine[i:i + a.chunk_samples] for i in range(0, len(mine), a.chunk_samples)]
    split_dir = sio.hs_dir(a.run_dir, a.split)
    t0, n_rows = time.time(), 0
    for ci, sids in enumerate(chunks):
        part = parts / f"score_{a.split}_s{a.shard}of{a.nshards}_c{ci:05d}.csv"
        if part.exists():
            continue
        sub = ck[ck["sample"].isin(set(sids))]
        df = score_samples(scorer, split_dir, sub)
        df.to_csv(str(part) + ".tmp", index=False)
        Path(str(part) + ".tmp").replace(part)
        n_rows += len(df)
        el = time.time() - t0
        print(f"[score {a.split} s{a.shard}] chunk {ci + 1}/{len(chunks)}  {n_rows} rows  {el / 60:.1f} min  "
              f"({n_rows / max(el, 1e-9):.1f} rows/s)", flush=True)
    print(f"[score {a.split} s{a.shard}] done", flush=True)


if __name__ == "__main__":
    main()
