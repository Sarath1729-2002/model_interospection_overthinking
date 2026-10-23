"""Stage 1 (CPU, minutes): build the checkpoint + trace tables from the cached completions.

    python cluster/sig_checkpoints.py --run-dir $RUN_DIR [--limit 300 --splits eval --out $RUN_DIR/signals_gate]

Writes  <out>/ckpt_<split>.csv  and  <out>/traces_<split>.csv.   Needs only the tokenizer and the cached .json/.safetensors
(it reads just the small id tensors, not the hidden states) and the GSM8K gold answers from the local HF cache.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import io as sio  # noqa: E402
from miol.signals.checkpoints import build_split_tables, gsm8k_gold_map  # noqa: E402
from miol.signals.text_utils import boundary_token_ids  # noqa: E402


def main(argv=None, tok=None, gold_map=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--splits", default="train,eval")
    ap.add_argument("--mode", default="lines", choices=["lines", "stride", "both"])
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples; N = a fixed random subset of N per split")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--hf-dataset", default="openai/gsm8k")
    ap.add_argument("--hf-config", default="main")
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    tok = tok or sio.load_tokenizer(a.model_id)
    gold_map = gold_map or gsm8k_gold_map(a.hf_dataset, a.hf_config)
    bset = boundary_token_ids(tok, len(tok))
    print(f"{len(bset)} vocabulary tokens contain a blank line; eos id {tok.eos_token_id}", flush=True)
    for split in a.splits.split(","):
        ckpt, traces = build_split_tables(sio.hs_dir(a.run_dir, split), split, tok, gold_map, tok.eos_token_id, bset,
                                          a.mode, a.stride, a.limit, a.subset_seed, log=sio.log_to(f"[ckpt {split}]"))
        sio.write_table(ckpt, out / f"ckpt_{split}.csv")
        sio.write_table(traces, out / f"traces_{split}.csv")
        print(f"[ckpt {split}] wrote {len(ckpt)} checkpoints for {len(traces)} traces "
              f"({len(ckpt) / max(len(traces), 1):.2f} per trace) -> {out}", flush=True)


if __name__ == "__main__":
    main()
