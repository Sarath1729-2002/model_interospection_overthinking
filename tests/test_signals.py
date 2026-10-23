"""Unit + end-to-end tests for the signals pipeline, on a character-level fake tokenizer and a tiny random Qwen2."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cluster" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(mod)
    return mod


from miol.signals import analysis, checkpoints, derive as dv, metrics, ridge, targets, text_utils as tu  # noqa: E402
from miol.signals.scoring import AnswerScorer  # noqa: E402

CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,=+-*/\\{}()$:#?!<>|_'\n")


class FakeTok:
    """char-level tokenizer with one extra token for a blank line, an EOS and a PAD."""

    def __init__(self):
        self.vocab = ["<pad>", "<eos>"] + CHARS + ["\n\n"]
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        self.eos_token_id, self.pad_token_id = 1, 0

    def __len__(self):
        return len(self.vocab)

    def encode(self, s, add_special_tokens=False):
        out, i = [], 0
        while i < len(s):
            if s[i:i + 2] == "\n\n":
                out.append(self.idx["\n\n"])
                i += 2
            else:
                out.append(self.idx[s[i]])
                i += 1
        return out

    def decode(self, ids, skip_special_tokens=False):
        parts = []
        for i in ids:
            t = self.vocab[int(i)]
            if t in ("<pad>", "<eos>"):
                if not skip_special_tokens:
                    parts.append("<|im_end|>" if t == "<eos>" else "")
                continue
            parts.append(t)
        return "".join(parts)


@pytest.fixture(scope="module")
def tok():
    return FakeTok()


@pytest.fixture(scope="module")
def tiny_model(tok):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=1024, bos_token_id=None, eos_token_id=1, pad_token_id=0,
                      tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg).eval()


# ------------------------------------------------------------------ text utils
def test_numbers_and_answers():
    assert tu.norm_number("1,000") == "1000" and tu.norm_number("18.00") == "18" and tu.norm_number("$5") == "5"
    assert tu.norm_number("abc") is None and tu.norm_number("-0") == "0"
    assert tu.parse_gold("blah <<2*3=6>> 6\n#### 1,072") == "1072"
    assert tu.parse_gold("no marker") is None
    assert tu.extract_boxed("so \\boxed{\\frac{1}{2}} and \\boxed{18}") == "18"
    assert tu.extract_boxed("\\boxed{a{b}c}") == "a{b}c"
    assert tu.parse_final_answer("x = 5\n\nTherefore the answer is \\(\\boxed{42}\\)<|im_end|>") == "42"
    assert tu.parse_final_answer("so 3 + 4 = 7\n\nThe total is 7 dollars.") == "7"
    assert tu.parse_final_answer("no digits here") is None
    assert tu.question_from_prompt("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n") == "What is 2+2?"


def test_prefix_features_and_bow():
    f = tu.prefix_features("a = 12\n\nwait, 5 + 7 = 12", "12", "13")
    assert f["gold_in"] == 1 and f["final_in"] == 0 and f["last_eq_gold"] == 1 and f["n_hedge"] == 1 and f["n_eq"] == 2
    assert tu.prefix_features("x 123", "12", None)["gold_in"] == 0          # whole numbers only, not substrings
    assert tu.hash_bow("a 12 = b") == tu.hash_bow("a 12 = b") and len(tu.hash_bow("hello world")) == 2


def test_boundary_and_digit_ids(tok):
    b = tu.boundary_token_ids(tok)
    assert b == {tok.idx["\n\n"]}
    assert tu.digit_token_ids(tok) == [tok.idx[str(d)] for d in range(10)]


# ------------------------------------------------------------------ checkpoints
def test_checkpoint_positions(tok):
    b = tu.boundary_token_ids(tok)
    comp = tok.encode("ab\n\ncd\n\nef") + [tok.eos_token_id]
    cps = checkpoints.checkpoint_positions(comp, b, tok.eos_token_id, True)
    pos = [p for p, _ in cps]
    kinds = [k for _, k in cps]
    assert pos == [-1, 2, 5, 7] and kinds == ["prompt", "line", "line", "final"]       # last content token = index 7, EOS excluded
    assert checkpoints.checkpoint_positions(comp, b, tok.eos_token_id, True, "stride", 3)[-1] == (7, "final")
    both = checkpoints.checkpoint_positions(comp, b, tok.eos_token_id, True, "both", 3)
    assert len({p for p, _ in both}) == len(both) and both[0][0] == -1
    ends_on_boundary = tok.encode("ab\n\n") + [tok.eos_token_id]
    c2 = checkpoints.checkpoint_positions(ends_on_boundary, b, tok.eos_token_id, True)
    assert c2 == [(-1, "prompt"), (2, "final")]            # the boundary IS the last content token: one checkpoint, kind final


# ------------------------------------------------------------------ derive
def _toy(forced, gold, final, n_tok=100, correct=None):
    n = len(forced)
    pos = np.linspace(-1, n_tok - 2, n).astype(int)
    pos[-1] = n_tok - 2
    return pd.DataFrame({"sample": 0, "k": np.arange(n), "pos": pos, "rem": n_tok - (pos + 1), "n_tok": n_tok, "gold": gold,
                         "final": final, "forced": forced, "correct": int(final == gold) if correct is None else correct,
                         "kind": ["prompt"] + ["line"] * (n - 2) + ["final"]})


def test_derive_overthinking_and_flips():
    df = _toy(["3", "7", "7", "9", "9", "9"], "9", "9", 100)          # converges at k=3 and stays
    d, tr = dv.derive(df, tau=10)
    t = tr.iloc[0]
    assert t.t_star == 3 and t.c_star == 3 and t.flips == 2 and t.cond_dither == 1
    assert d["conv_gold"].tolist() == [0, 0, 0, 1, 1, 1]
    assert d.loc[d.k == 3, "d_gold"].iloc[0] == 0 and d.loc[d.k == 5, "d_gold"].iloc[0] == d.pos.iloc[5] - d.pos.iloc[3]
    assert t.slack_gold == int(d.rem.iloc[3] - 1) and t.cond_overthink == 1 and t.cond_rationalize == 0


def test_derive_rationalization_and_lost():
    df = _toy(["5", "9", "5", "5", "5"], "9", "5", 100)               # had gold at k=1, lost it, ends on wrong 5
    d, tr = dv.derive(df, tau=10)
    t = tr.iloc[0]
    assert t.t_star == -1 and t.c_star == 2 and t.cond_lost == 1 and t.cond_rationalize == 1 and t.cond_overthink == 0
    assert d["committed_wrong"].tolist() == [0, 0, 1, 1, 1]
    assert np.isnan(d["d_gold"]).all() and d["d_commit"].iloc[2] == 0


def test_derive_never_converges():
    d, tr = dv.derive(_toy(["1", "2", "3", "4"], "9", "4"), tau=10)
    assert tr.iloc[0].t_star == -1 and tr.iloc[0].c_star == 3 and tr.iloc[0].slack_commit == 0 and tr.iloc[0].cond_rationalize == 0


# ------------------------------------------------------------------ metrics / ridge
def test_metrics_basic():
    rng = np.random.RandomState(0)
    y = rng.rand(300) > 0.5
    s = y + 0.8 * rng.randn(300)
    pos, neg = s[y], s[~y]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert abs(metrics.auroc(y, s) - brute) < 1e-9
    assert metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    g = np.repeat(np.arange(30), 10)
    yy = rng.randn(300)
    out = metrics.boot_reg(yy, yy + 0.1 * rng.randn(300), g, 200)
    assert out["mae"]["lo"] <= out["mae"]["est"] <= out["mae"]["hi"] and out["r2"]["est"] > 0.9
    d = metrics.boot_paired(yy, yy + 0.1 * rng.randn(300), yy + 1.0 * rng.randn(300), g, 200)
    assert d["delta_mae"]["hi"] < 0 and d["delta_r2"]["lo"] > 0


def test_fit_layer_handles_nan_and_recovers_signal():
    rng = np.random.RandomState(0)
    X = torch.tensor(rng.randn(400, 12) * 3 + 5, dtype=torch.float32)
    w = rng.randn(12)
    y1 = X.numpy() @ w + 2.0
    y2 = y1.copy()
    y2[::7] = np.nan
    lf = ridge.fit_layer(X, np.stack([y1, y2], 1), ratios=[1e-4, 1.0])
    for t, y in ((0, y1), (1, y2)):
        p = lf.predict(X, t)[:, 0].numpy()
        m = ~np.isnan(y)
        assert np.abs(p[m] - y[m]).max() < 0.05
    assert np.abs(lf.predict(X, 0)[:, 1].numpy() - y1).mean() > np.abs(lf.predict(X, 0)[:, 0].numpy() - y1).mean()   # more shrinkage fits worse


def test_kernel_allcat_matches_primal_ridge():
    rng = np.random.RandomState(1)
    n, nv, D, L = 120, 30, 8, 3
    layers = [torch.tensor(rng.randn(n + nv + nv, D) * (l + 1), dtype=torch.float32) for l in range(L)]
    w = rng.randn(L * D)
    cat = torch.cat(layers, 1)
    y = (cat.numpy() @ w + 0.3 * rng.randn(n + 2 * nv)).astype(np.float32)
    Y = y[:n, None]
    get = lambda l: {"fit": layers[l][:n], "val": layers[l][n:n + nv], "test": layers[l][n + nv:]}
    cand = ridge.kernel_allcat(get, L, Y, ratios=[0.1], log=lambda *_: None)
    # primal reference on the concatenation of per-layer standardised features, same lambda = ratio * n
    mu, sd = ridge.standardise_stats(cat[:n])
    Z = (cat - mu) / sd
    Zf = Z[:n].double()
    lam = 0.1 * n
    yc = torch.tensor(Y[:, 0] - Y[:, 0].mean(), dtype=torch.float64)
    W = torch.linalg.solve(Zf.T @ Zf + lam * torch.eye(L * D, dtype=torch.float64), Zf.T @ yc)
    ref = (Z[n:n + nv].double() @ W + Y[:, 0].mean()).numpy()
    assert np.abs(cand[0][0]["val"] - ref).max() < 2e-2


# ------------------------------------------------------------------ scorer (tiny random Qwen2)
def test_scorer_matches_manual_logprob_and_batching(tok, tiny_model):
    sc = AnswerScorer(tiny_model, tok, "cpu", token_budget=5000, max_batch=4)
    p = tok.encode("<|im_start|>user\nq<|im_end|>\n")
    c = tok.encode("x = 3\n\ny = 4\n\nz")
    ctxs = [sc.context(p, c, pos) for pos in (-1, 4, 10)]
    answers = ["18", "7", "-3"]
    res = sc.score(ctxs, answers)
    for ctx, ans, r in zip(ctxs, answers, res):
        ids = ctx + tok.encode(ans)
        with torch.no_grad():
            logp = torch.log_softmax(tiny_model(torch.tensor([ids])).logits[0].float(), -1)
        lp = sum(float(logp[len(ctx) - 1 + j, t]) for j, t in enumerate(tok.encode(ans)))
        p_digit = float(sum(logp[len(ids) - 1, tok.idx[str(d)]].exp() for d in range(10)))
        want = lp + np.log1p(-p_digit)
        assert r["a"] == pytest.approx(want, abs=1e-3)
    solo = [sc.score([ctx], [a])[0]["a"] for ctx, a in zip(ctxs, answers)]
    assert solo == pytest.approx([r["a"] for r in res], abs=1e-3)
    assert np.isnan(sc.score([ctxs[0]], ["1.5"])[0]["a"]) and np.isnan(sc.score([ctxs[0]], [None])[0]["a"])


def test_context_separator_and_greedy(tok, tiny_model):
    sc = AnswerScorer(tiny_model, tok, "cpu", token_budget=5000, max_batch=4)
    p = tok.encode("Q")
    c = tok.encode("ab\n\ncd")
    assert sc.context(p, c, 2)[len(p):len(p) + 3] == tok.encode("ab\n\n")                    # ends with a blank line: no extra separator
    assert sc.context(p, c, 4)[len(p) + 5:len(p) + 6] == tok.encode("\n\n")                  # mid-line: separator added
    assert sc.context(p, c, -1) == p + sc.cue_ids
    g = sc.greedy([sc.context(p, c, 2), sc.context(p, c, 4)])
    assert len(g) == 2 and all(isinstance(x, str) for x in g)
    single = sc.greedy([sc.context(p, c, 4)])
    assert single == [g[1]]


# ------------------------------------------------------------------ end to end on a synthetic cache
def _make_cache(root, tok, n_train, n_eval, L=3, D=16, seed=0):
    rng = np.random.RandomState(seed)
    gold_map = {}
    direction = rng.randn(D)
    for split, n in (("train", n_train), ("eval", n_eval)):
        d = root / "hidden_states" / split
        d.mkdir(parents=True)
        for i in range(n):
            a, b = int(rng.randint(2, 9)), int(rng.randint(2, 9))
            q = f"What is {a} plus {b}, call it case {split}{i}?"
            gold = str(a + b)
            gold_map[q] = gold
            ans = gold if rng.rand() < 0.7 else str(a + b + 1)
            steps = [f"First {a} + {b} = {a + b}.", f"Check again {a} + {b} = {a + b}.", f"So we get {ans}."]
            completion = "\n\n".join(steps) + f"\n\nTherefore the answer is \\(\\boxed{{{ans}}}\\)."
            prompt = f"<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            p_ids, c_ids = tok.encode(prompt), tok.encode(completion) + [tok.eos_token_id]
            T = len(p_ids) + len(c_ids)
            rem = np.concatenate([np.zeros(len(p_ids)), len(c_ids) - np.arange(len(c_ids)) - 1])
            rem[len(p_ids) - 1] = len(c_ids)                       # the prompt-end checkpoint has the whole completion left
            tensors = {f"hidden_states_{l}": torch.tensor(rng.randn(T, D), dtype=torch.float32) for l in range(L)}
            tensors["hidden_states_1"] = torch.tensor(rng.randn(T, D) * 0.1 + np.outer(rem, direction), dtype=torch.float32)
            tensors["prompt_ids"] = torch.tensor(p_ids)
            tensors["completion_ids"] = torch.tensor(c_ids)
            tensors["full_sequence_ids"] = torch.tensor(p_ids + c_ids)
            save_file(tensors, str(d / f"sample_{i:06d}.safetensors"))
            json.dump({"prompt": prompt, "completion": completion + "<|im_end|>", "prompt_length": len(p_ids),
                       "completion_length": len(c_ids), "has_eos": True, "num_hidden_layers": L},
                      open(d / f"sample_{i:06d}.json", "w"))
        # a truncated sample (no EOS) that must be skipped
        json.dump({"prompt": "x", "completion": "y", "prompt_length": 1, "completion_length": 1, "has_eos": False,
                   "num_hidden_layers": L}, open(d / f"sample_{n:06d}.json", "w"))
    return gold_map


def test_end_to_end_tiny(tmp_path, tok, tiny_model):
    gold_map = _make_cache(tmp_path, tok, n_train=40, n_eval=12)
    run = str(tmp_path)
    sig = tmp_path / "signals"

    c = _load("sig_checkpoints")
    c.main(["--run-dir", run, "--splits", "train,eval"], tok=tok, gold_map=gold_map)
    ck = pd.read_csv(sig / "ckpt_train.csv", keep_default_na=False)
    assert ck["sample"].nunique() == 40 and (ck.groupby("sample")["k"].apply(lambda x: list(x) == list(range(len(x))))).all()
    assert ck.groupby("sample")["kind"].apply(lambda x: x.iloc[0] == "prompt" and x.iloc[-1] == "final").all()
    assert (ck["rem"] >= 0).all() and ck.loc[ck.kind == "final", "rem"].eq(1).all()          # one token (EOS) left at the final checkpoint

    s = _load("sig_score")
    kw = dict(tok=tok, model=tiny_model)
    s.main(["--run-dir", run, "--split", "eval", "--device", "cpu", "--chunk-samples", "5"], **kw)
    s.main(["--run-dir", run, "--split", "train", "--device", "cpu", "--chunk-samples", "7", "--shard", "0", "--nshards", "2"], **kw)
    s.main(["--run-dir", run, "--split", "train", "--device", "cpu", "--chunk-samples", "7", "--shard", "1", "--nshards", "2"], **kw)
    before = sorted(p.name for p in (sig / "score_parts").glob("*.csv"))
    s.main(["--run-dir", run, "--split", "eval", "--device", "cpu", "--chunk-samples", "5"], **kw)            # resumable: no-op
    assert before == sorted(p.name for p in (sig / "score_parts").glob("*.csv"))

    r = _load("sig_rollouts")
    for split, nt in (("eval", 6), ("train", 15)):
        r.main(["--run-dir", run, "--split", split, "--device", "cpu", "--n-traces", str(nt), "--m", "2", "--max-new", "6",
                "--chunk-samples", "4", "--rows-per-batch", "4"], **kw)

    d = _load("sig_derive")
    d.main(["--run-dir", run, "--tau", "5"])
    lab = pd.read_csv(sig / "labels_train.csv", keep_default_na=False, na_values=["", "nan"], dtype={"gold": str, "final": str, "forced": str})
    assert {"a", "f", "forced", "eq_gold", "t_star", "d_gold", "roll_pgold"} <= set(lab.columns)
    assert lab["roll_pgold"].notna().sum() > 0 and lab["roll_pgold"].dropna().between(0, 1).all()
    fin = lab[lab.kind == "final"]
    assert (fin["roll_H"].dropna() == 0).all()
    assert (sig / "gate_report_eval.txt").exists() and "feasibility gate" in (sig / "gate_report_eval.txt").read_text()
    # f == a exactly when the trace's own answer is the gold answer; NaN-free where a final exists
    same = lab[(lab.final == lab.gold)]
    assert np.allclose(same["f"], same["a"], equal_nan=True)

    st = _load("sig_states")
    for split in ("train", "eval"):
        st.main(["--run-dir", run, "--split", split, "--workers", "0"])
    S = np.load(sig / "states_train.npy", mmap_mode="r")
    assert S.shape[0] == 3 and S.shape[1] == len(lab) and S.dtype == np.float16
    # a stored state really is the cached hidden state at prompt_len + pos
    from safetensors import safe_open
    row = lab.iloc[5]
    with safe_open(str(tmp_path / "hidden_states" / "train" / f"sample_{int(row['sample']):06d}.safetensors"), "pt") as f:
        want = f.get_tensor("hidden_states_2")[int(row["prompt_len"]) + int(row["pos"])].numpy()
    assert np.allclose(S[2, int(row["row"])], want, atol=2e-2)

    pr = _load("sig_probes")
    pr.main(["--run-dir", run, "--device", "cpu", "--val-fraction", "0.2", "--kernel-max-fit", "0"])
    meta = json.load(open(sig / "probes" / "meta.json"))
    assert {"B0", "B1", "B4", "L0", "L1", "L2", "ALL"} <= set(meta["models"]) and "a" in meta["targets"] and "r_pgold" in meta["targets"]
    sc = pd.read_csv(sig / "probes" / "scores_test.csv")
    r2 = lambda m, t: float(sc[(sc.model == m) & (sc.target == t)]["r2"].iloc[0])
    assert r2("L1", "rem") > 0.9 and r2("ALL", "rem") > 0.8          # the planted layer is found, directly and through the all-layer kernel
    assert r2("L0", "rem") < r2("L1", "rem")

    an = _load("sig_analyze")
    an.main(["--run-dir", run, "--boot", "30"])
    rep = (sig / "analysis" / "report.md").read_text()
    for sec in ("Headline", "Condition prevalence", "Probe vs baselines", "Early exit", "Condition monitors"):
        assert sec in rep
    assert (sig / "analysis" / "probe_vs_baselines.csv").exists()
