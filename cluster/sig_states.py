"""Stage 3 (CPU or GPU node, I/O bound): gather the hidden state of every checkpoint into one array per split.

    python cluster/sig_states.py --run-dir $RUN_DIR --split train [--workers 8]
Writes <out>/states_<split>.npy  float16 [L, N, D]  (row n = checkpoint `row` of ckpt_<split>.csv) + states_<split>.json.
Values are clipped to +-6e4 (float16 range) and the number of clipped entries is reported.  Resumable per sample group is NOT
needed: the whole thing is a streaming copy (~minutes per 10^4 samples); a half-written file is detected via the .json marker.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import io as sio  # noqa: E402

FP16_MAX = 6.0e4


class SampleStates(Dataset):
    def __init__(self, split_dir, groups):
        self.split_dir, self.groups = Path(split_dir), groups      # list of (sample_id, [row...], [state row idx...])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        from safetensors import safe_open
        sid, rows, idx = self.groups[i]
        with safe_open(str(self.split_dir / f"sample_{sid:06d}.safetensors"), "pt") as f:
            keys = sorted((k for k in f.keys() if k.startswith("hidden_states_")), key=lambda s: int(s.split("_")[-1]))
            X = torch.stack([f.get_tensor(k)[idx] for k in keys], 0)                 # [L, n_k, D]
        clipped = int((X.abs() > FP16_MAX).sum())
        X = X.float().clamp_(-FP16_MAX, FP16_MAX).half()
        return rows, X, clipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    ck = sio.read_table(out / f"ckpt_{a.split}.csv")
    path, marker = out / f"states_{a.split}.npy", out / f"states_{a.split}.json"
    split_dir = sio.hs_dir(a.run_dir, a.split)
    groups = []
    for sid, g in ck.groupby("sample", sort=False):
        pl = int(g["prompt_len"].iloc[0])
        groups.append((int(sid), g["row"].tolist(), (pl + g["pos"].values).tolist()))      # pos=-1 -> row pl-1
    # layer count / width from the first sample
    from safetensors import safe_open
    with safe_open(str(split_dir / f"sample_{groups[0][0]:06d}.safetensors"), "pt") as f:
        keys = [k for k in f.keys() if k.startswith("hidden_states_")]
        D = f.get_tensor(keys[0]).shape[-1]
    Lyr, N = len(keys), len(ck)
    if marker.exists() and path.exists():
        print(f"[states {a.split}] already complete: {path}")
        return
    arr = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float16, shape=(Lyr, N, D))
    dl = DataLoader(SampleStates(split_dir, groups), batch_size=None, shuffle=False, num_workers=a.workers,
                    prefetch_factor=2 if a.workers else None)
    clipped = 0
    for i, (rows, X, c) in enumerate(dl):
        assert rows[-1] - rows[0] + 1 == len(rows)
        arr[:, rows[0]:rows[-1] + 1, :] = X.numpy()
        clipped += c
        if (i + 1) % 500 == 0:
            print(f"[states {a.split}] {i + 1}/{len(groups)} samples", flush=True)
    arr.flush()
    del arr
    marker.write_text(json.dumps({"layers": Lyr, "rows": N, "dim": D, "clipped_entries": clipped}))
    print(f"[states {a.split}] wrote {path}  shape ({Lyr}, {N}, {D})  clipped entries {clipped}", flush=True)


if __name__ == "__main__":
    main()
