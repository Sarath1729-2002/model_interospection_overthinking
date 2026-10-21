"""Summarise an extracted hidden-states cache (reads only the small .json files).

    python cluster/audit_extraction.py <run_dir>/hidden_states

Use it to sanity-check an extraction before spending GPU hours on training:
the truncation rate (completions that never emitted EOS, i.e. hit the
max_new_tokens cutoff and are excluded from training) and the completion-length
distribution are the numbers to look at.
"""

import json
import statistics
import sys
from pathlib import Path


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def audit(split_dir: Path):
    n = errors = truncated = 0
    comp, prompt, missing_tensors = [], [], 0
    for jp in split_dir.glob("sample_*.json"):
        try:
            meta = json.loads(jp.read_text())
        except Exception:
            errors += 1
            continue
        n += 1
        if meta.get("error"):
            errors += 1
            continue
        if not meta.get("has_eos", False):
            truncated += 1
            continue
        comp.append(meta["completion_length"])
        prompt.append(meta["prompt_length"])
        if not jp.with_suffix(".safetensors").exists():
            missing_tensors += 1
    comp.sort()
    prompt.sort()
    print(f"[{split_dir.name}] samples={n}  usable(EOS)={len(comp)}  truncated={truncated} "
          f"({100 * truncated / max(n, 1):.2f}%)  errors={errors}  json-without-tensors={missing_tensors}")
    if comp:
        print(f"    completion tokens: mean={statistics.fmean(comp):.1f} median={statistics.median(comp)} "
              f"p95={pct(comp, 0.95)} max={comp[-1]}")
        print(f"    prompt tokens    : mean={statistics.fmean(prompt):.1f} max={prompt[-1]}")
    return len(comp), sum(comp) + sum(prompt)


def main():
    root = Path(sys.argv[1])
    total_tokens = 0
    for split in ("train", "eval"):
        d = root / split
        if not d.exists():
            print(f"[{split}] missing: {d}")
            continue
        _, toks = audit(d)
        total_tokens += toks
    print(f"total usable tokens ≈ {total_tokens:,}")


if __name__ == "__main__":
    main()
