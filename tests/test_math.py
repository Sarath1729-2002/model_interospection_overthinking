"""MATH pipeline: answer parsing, and the whole chain worker -> merge -> probes -> controls -> analyze -> transfer/steer/rollouts
on a tiny random Qwen2 with canned completions."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_signals import FakeTok, _load, tiny_model, tok  # noqa: F401,E402
from miol.signals import mathdata as md  # noqa: E402


class ChatTok(FakeTok):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        s = "".join(f"{m['role'][0].upper()}:{m['content']}\n" for m in messages)
        s = "".join(c for c in s if c in self.idx or c == "\n")
        return s + "A:"


@pytest.fixture(scope="module")
def ctok():
    return ChatTok()


def test_math_answers():
    assert md.clean_math_answer("12") == "12"
    assert md.clean_math_answer("\\$1,200") == "1200"
    assert md.clean_math_answer("45^\\circ") == "45"
    assert md.clean_math_answer("\\text{15 cm}") is None or md.clean_math_answer("\\text{15 cm}") == "15"
    assert md.clean_math_answer("\\frac{1}{2}") is None
    assert md.clean_math_answer("x+1") is None
    assert md.math_gold("so the answer is \\boxed{\\$7{,}000}.") == "7000"
    assert md.parse_final_math("a \\boxed{3} then \\boxed{12}") == "12"
    assert md.parse_final_math("no box") is None
    assert md.parse_final_math("\\boxed{\\frac{1}{2}}") is None


def _canned(tok):
    def gen(prompts, seed):
        out = []
        for i, p in enumerate(prompts):
            ans = (len(p) + i) % 5 + 3
            body = [f"First {i}+{ans}={i + ans}", "Check it again 1+1=2", "Hmm, wait 2+2=4" if i % 3 == 0 else "Fine 3+3=6",
                    f"So the answer is \\boxed{{{ans}}}."]
            txt = "\n\n".join(body)
            out.append((tok.encode(txt) + [tok.eos_token_id], True))
        return out
    return gen


def _problems(run, n_train, n_test):
    for split, n in (("train", n_train), ("test", n_test)):
        with open(Path(run) / f"problems_{split}.jsonl", "w") as f:
            for i in range(n):
                f.write(json.dumps({"idx": i, "problem": f"What is {i}+{i}?", "gold": str((i * 3) % 5 + 3), "level": f"Level {i % 5 + 1}",
                                    "type": "Algebra"}) + "\n")


def test_math_pipeline_tiny(tmp_path, ctok, tiny_model):
    run = str(tmp_path)
    sig = tmp_path / "signals"
    _problems(run, 36, 12)
    w = _load("msig_worker")
    hook = lambda wk: setattr(wk, "generate_fn", _canned(ctok))  # noqa: E731
    for split in ("test", "train"):
        for sh in (0, 1):
            w.main(["--run-dir", run, "--out", str(sig), "--split", split, "--shard", str(sh), "--nshards", "2", "--chunk-problems", "5",
                    "--device", "cpu", "--max-new", "40"], tok=ctok, model=tiny_model, worker_hook=hook)
    n_chunks = len(list((sig / "chunks" / "train").glob("*.done")))
    assert n_chunks == 8                                              # 36 problems / 5 per chunk
    before = sorted(p.name for p in (sig / "chunks" / "test").iterdir())
    w.main(["--run-dir", run, "--out", str(sig), "--split", "test", "--chunk-problems", "5", "--device", "cpu"], tok=ctok, model=tiny_model,
           worker_hook=hook)
    assert before == sorted(p.name for p in (sig / "chunks" / "test").iterdir())    # resumable

    m = _load("msig_merge")
    m.main(["--run-dir", run, "--out", str(sig), "--tau", "5"])
    lab = pd.read_csv(sig / "labels_train.csv", keep_default_na=False, na_values=["", "nan"], dtype={"gold": str, "final": str, "forced": str})
    need = {"a", "f", "forced", "d_gold", "pbow", "qbow", "ent_next", "a_next", "da", "loss_next", "gain_next", "loss_end", "y_end",
            "hedge_next"}
    assert need <= set(lab.columns), need - set(lab.columns)
    tl = pd.read_csv(sig / "traces_labeled_train.csv")
    assert {"cond_overthink_core", "cond_wobble", "cond_dither"} <= set(tl.columns)
    S = np.load(sig / "states_train.npy", mmap_mode="r")
    assert S.shape[0] == 3 and S.shape[1] == len(lab) and S.dtype == np.float16
    assert (sig / "gate_report_eval.txt").exists()
    # forecast labels are defined only where they should be
    assert lab["loss_next"].dropna().isin([0, 1]).all() and lab["a_next"].notna().sum() > 0

    p = _load("sig_probes")
    p.main(["--run-dir", run, "--out", str(sig), "--device", "cpu", "--val-fraction", "0.2", "--kernel-max-fit", "0", "--split-source", "labels",
            "--save-weights", "--no-mlp"])
    sc = pd.read_csv(sig / "probes" / "scores_test.csv")
    assert {"B1", "B4", "B5", "L1"} <= set(sc.model) and (sig / "probes" / "weights_layers.pt").exists()
    assert {"a_next", "da"} <= set(sc.target)

    c = _load("sig_controls")
    c.main(["--run-dir", run, "--out", str(sig), "--device", "cpu", "--layers", "1,2", "--val-fraction", "0.2", "--split-source", "labels",
            "--no-bq", "--min-rows", "10", "--boot", "20"])
    assert (sig / "analysis" / "controls.csv").exists()

    a = _load("sig_analyze")
    a.main(["--run-dir", run, "--out", str(sig), "--boot", "20"])
    assert (sig / "analysis" / "report.md").exists()

    t = _load("msig_transfer")
    t.main(["--math-out", str(sig), "--gsm-out", str(sig), "--layers", "1,2", "--device", "cpu"])
    tr = pd.read_csv(sig / "analysis" / "transfer.csv")
    assert len(tr) > 0 and np.isfinite(tr["R2 (MATH probe on GSM8K)"]).all()

    s = _load("msig_steer")
    nh = len(tiny_model.model.layers[0]._forward_hooks)
    df = s.main(["--run-dir", run, "--out", str(sig), "--layer", "1", "--n-traces", "6", "--per-trace", "2", "--alphas=-0.3,0.3",
                 "--n-random", "1", "--device", "cpu"], tok=ctok, model=tiny_model)
    assert {"none", "probe", "random0"} <= set(df.direction) and df.loc[df.alpha == 0, "d_a"].eq(0).all()
    # an unsteered rescoring is deterministic: the hook is off for the baseline and removed afterwards
    assert len(tiny_model.model.layers[0]._forward_hooks) == nh

    r = _load("sig_rollouts")
    for split, nt in (("eval", 4), ("train", 6)):
        r.main(["--run-dir", run, "--out", str(sig), "--split", split, "--data", "math", "--device", "cpu", "--n-traces", str(nt), "--m", "2",
                "--max-new", "6", "--chunk-samples", "3", "--rows-per-batch", "4"], tok=ctok, model=tiny_model)
    m.main(["--run-dir", run, "--out", str(sig), "--tau", "5", "--labels-only"])
    lab2 = pd.read_csv(sig / "labels_train.csv", keep_default_na=False, na_values=["", "nan"], dtype={"gold": str, "final": str, "forced": str})
    assert len(lab2) == len(lab) and lab2["roll_pgold"].notna().sum() > 0
