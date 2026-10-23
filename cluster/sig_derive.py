"""Stage 2b (CPU, seconds): merge the score chunks into labels_<split>.csv (+ derived labels) and print the gate report.

    python cluster/sig_derive.py --run-dir $RUN_DIR [--out ...] [--splits train,eval] [--tau 30]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals import io as sio  # noqa: E402
from miol.signals.derive import derive  # noqa: E402
from miol.signals.report import gate_report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", default="train,eval")
    ap.add_argument("--tau", type=int, default=30, help="tokens written after convergence/commitment that count as a condition")
    a = ap.parse_args(argv)
    out = sio.out_dir(a.run_dir, a.out)
    for split in a.splits.split(","):
        ck = sio.read_table(out / f"ckpt_{split}.csv")
        files = sorted((out / "score_parts").glob(f"score_{split}_s*of*_c*.csv"))
        if not files:
            print(f"[derive {split}] no score parts found - skipped")
            continue
        sc = pd.concat([sio.read_table(f) for f in files], ignore_index=True)
        df = ck.merge(sc, on=["sample", "k"], how="left", validate="one_to_one", indicator=True)
        no_score = (df.pop("_merge") == "left_only").values
        miss = int(no_score.sum())
        if miss:
            bad = df.loc[no_score, "sample"].unique()
            print(f"[derive {split}] WARNING: {miss} checkpoints of {len(bad)} traces have no score yet "
                  f"(unfinished chunks) - those traces are dropped. Re-run the score stage, then this one.")
            df = df[~df["sample"].isin(set(bad))].reset_index(drop=True)
        roll = sorted((out / "roll_parts").glob(f"roll_{split}_s*of*_c*.csv"))
        if roll:
            rl = pd.concat([sio.read_table(f) for f in roll], ignore_index=True)
            df = df.merge(rl, on=["sample", "k"], how="left")
            print(f"[derive {split}] rollout columns merged for {rl['sample'].nunique()} traces")
        df["forced"] = df["forced"].fillna("").astype(str)
        df, tr = derive(df, a.tau)
        base = sio.read_table(out / f"traces_{split}.csv")
        tr = base.drop(columns=["correct", "n_ckpt"]).merge(tr, on="sample", how="inner")
        assert df["row"].is_monotonic_increasing
        sio.write_table(df, out / f"labels_{split}.csv")
        sio.write_table(tr, out / f"traces_labeled_{split}.csv")
        txt = gate_report(df, tr, split)
        (out / f"gate_report_{split}.txt").write_text(txt + "\n")
        print(txt, flush=True)


if __name__ == "__main__":
    main()
