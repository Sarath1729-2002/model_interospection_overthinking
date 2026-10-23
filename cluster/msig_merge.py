"""Step 2 (CPU, minutes): merge the chunks of a split, derive the labels, print the gate report.

    python cluster/msig_merge.py --run-dir $RUN_DIR [--out ...] [--splits train,test] [--tau 30]
Writes  ckpt_<split>.csv  traces_<split>.csv  ids_<split>.jsonl  states_<split>.npy (+.json)  labels_<split>.csv
        traces_labeled_<split>.csv  gate_report_<split>.txt
(split 'test' is also written under the name 'eval' because the analysis stages call the held-out split 'eval'.)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import io as sio  # noqa: E402
from miol.signals.derive import derive  # noqa: E402
from miol.signals.report import gate_report  # noqa: E402


def add_rollouts(out: Path, name: str, df: pd.DataFrame, log=print) -> pd.DataFrame:
    roll = sorted((out / "roll_parts").glob(f"roll_{name}_s*of*_c*.csv"))
    if not roll:
        return df
    rl = pd.concat([sio.read_table(f) for f in roll], ignore_index=True).drop_duplicates(["sample", "k"])
    log(f"[merge {name}] rollout columns merged for {rl['sample'].nunique()} traces")
    return df.merge(rl, on=["sample", "k"], how="left")


def relabel(out: Path, name: str, tau: int, log=print):
    """labels only (after the rollouts finished): re-derive from the existing ckpt/traces tables; states are left alone."""
    ck = sio.read_table(out / f"ckpt_{name}.csv")
    tr = sio.read_table(out / f"traces_{name}.csv")
    df, trl = derive(add_rollouts(out, name, ck.copy(), log), tau)
    assert np.array_equal(df["row"].values, ck["row"].values)
    trl = tr.drop(columns=[c for c in ("correct", "n_ckpt") if c in tr.columns]).merge(trl, on="sample", how="inner")
    sio.write_table(df, out / f"labels_{name}.csv")
    sio.write_table(trl, out / f"traces_labeled_{name}.csv")
    log(f"[merge {name}] labels rewritten ({len(df)} rows, {df.shape[1]} columns)")
    return df


def merge_split(out: Path, split: str, name: str, tau: int, log=print):
    d = out / "chunks" / split
    ids = sorted(int(p.stem[1:]) for p in d.glob("c*.done"))
    if not ids:
        log(f"[merge {split}] no finished chunks")
        return None
    ck_parts, tr_parts, shapes = [], [], []
    for c in ids:
        base = d / f"c{c:05d}"
        ck_parts.append(sio.read_table(Path(f"{base}_ckpt.csv")) if (Path(f"{base}_ckpt.csv").stat().st_size > 5) else pd.DataFrame())
        tr_parts.append(sio.read_table(Path(f"{base}_traces.csv")))
        shapes.append(np.load(f"{base}_states.npy", mmap_mode="r").shape)
    ck = pd.concat([p for p in ck_parts if len(p)], ignore_index=True).sort_values(["sample", "k"]).reset_index(drop=True)
    tr = pd.concat(tr_parts, ignore_index=True).sort_values("sample").reset_index(drop=True)
    # states in the same row order as the sorted table
    first = next(s for s in shapes if len(s) == 3 and s[1] > 0)
    L, D = first[0], first[2]
    N = len(ck)
    arr = np.lib.format.open_memmap(str(out / f"states_{name}.npy"), mode="w+", dtype=np.float16, shape=(L, N, D))
    # chunk-local row r of chunk c <-> (sample, k); fill by key
    key = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(ck["sample"].values, ck["k"].values))}
    for c, part in zip(ids, ck_parts):
        if not len(part):
            continue
        st = np.load(d / f"c{c:05d}_states.npy", mmap_mode="r")
        dest = np.array([key[(int(s), int(k))] for s, k in zip(part["sample"].values, part["k"].values)])
        order = np.argsort(dest)
        arr[:, dest[order], :] = np.asarray(st[:, order, :])
    arr.flush()
    del arr
    (out / f"states_{name}.json").write_text(json.dumps({"layers": L, "rows": N, "dim": D}))
    ck.insert(0, "row", np.arange(N))
    sio.write_table(ck, out / f"ckpt_{name}.csv")
    sio.write_table(tr, out / f"traces_{name}.csv")
    with open(out / f"ids_{name}.jsonl", "w") as f:
        for c in ids:
            f.write(open(d / f"c{c:05d}_ids.jsonl").read())
    df, trl = derive(add_rollouts(out, name, ck.copy(), log), tau)
    assert np.array_equal(df["row"].values, np.arange(N)), "derive must keep the (sample, k) order"
    trl = tr.drop(columns=[c for c in ("correct", "n_ckpt") if c in tr.columns]).merge(trl, on="sample", how="inner")
    sio.write_table(df, out / f"labels_{name}.csv")
    sio.write_table(trl, out / f"traces_labeled_{name}.csv")
    extras = [f"problems {len(tr)}  terminated {int(tr['has_eos'].sum())} ({100 * tr['has_eos'].mean():.1f}%)  "
              f"truncated {int((1 - tr['has_eos']).sum())}  no integer answer among terminated "
              f"{int(((tr['has_eos'] == 1) & (tr['final_ok'] == 0)).sum())}",
              f"accuracy over ALL problems {100 * tr['correct'].mean():.1f}%  (terminated only {100 * tr.loc[tr['has_eos'] == 1, 'correct'].mean():.1f}%)"]
    if "level" in tr and tr["level"].astype(str).str.len().gt(0).any():
        lv = tr.groupby("level")["correct"].agg(["mean", "size"])
        extras.append("accuracy by level: " + "  ".join(f"{i}: {100 * r['mean']:.0f}% (n={int(r['size'])})" for i, r in lv.iterrows()))
    txt = "\n".join(extras) + "\n" + gate_report(df, trl, name)
    (out / f"gate_report_{name}.txt").write_text(txt + "\n")
    log(txt)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--tau", type=int, default=30)
    ap.add_argument("--labels-only", action="store_true", help="re-derive labels (e.g. after the rollouts) without rebuilding states")
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    for split in a.splits.split(","):
        if a.labels_only:
            relabel(out, "eval" if split == "test" else split, a.tau)
            continue
        merge_split(out, split, "eval" if split == "test" else split, a.tau)


if __name__ == "__main__":
    main()
