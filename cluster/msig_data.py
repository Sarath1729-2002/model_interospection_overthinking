"""Step 0 (LOGIN node - needs internet): download MATH and write the integer-answer problem lists.

    python cluster/msig_data.py --run-dir $RUN_DIR
Writes <run>/problems_train.jsonl, problems_test.jsonl.  The compute nodes read only these files (no network needed).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from miol.signals.mathdata import build_problem_files, load_math_splits  # noqa: E402


def main(argv=None, loader=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    splits = load_math_splits(loader=loader)
    counts = build_problem_files(Path(a.run_dir), splits)
    print(f"problem lists written to {a.run_dir}: {counts}")


if __name__ == "__main__":
    main()
