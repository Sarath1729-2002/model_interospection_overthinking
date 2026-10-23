"""Row-level and trace-level labels derived from the scored checkpoints (exact; no judge).

Per checkpoint:   eq_gold  = greedy forced answer == gold      eq_final = greedy forced answer == the trace's own final answer
Per trace:        t_star   = first checkpoint from which EVERY later forced answer equals gold     ("stable convergence")
                  c_star   = first checkpoint from which every later forced answer equals the trace's own final answer ("commitment")
                  slack    = completion tokens still written after that checkpoint (EOS excluded)
Named conditions (thresholds in tokens, `tau`):
   overthink    correct trace, converged to gold, then wrote >= tau more tokens
   rationalize  WRONG trace, committed to its wrong final answer, then wrote >= tau more tokens
   lost         wrong trace whose forced answer was gold at some checkpoint k >= 1 (it had the answer and lost it)
   dither       the forced answer changed at least twice along the trace
"""
import numpy as np
import pandas as pd


def _stable_from(flags: np.ndarray):
    """Smallest index i such that flags[i:] are all True; None if the last flag is False."""
    if len(flags) == 0 or not flags[-1]:
        return None
    i = len(flags) - 1
    while i > 0 and flags[i - 1]:
        i -= 1
    return i


def derive(df: pd.DataFrame, tau: int = 30):
    df = df.sort_values(["sample", "k"]).reset_index(drop=True).copy()
    forced = df["forced"].fillna("").astype(str)
    df["eq_gold"] = (forced == df["gold"].astype(str)).astype(int)
    fin = df["final"].fillna("").astype(str)
    df["eq_final"] = ((fin != "") & (forced == fin)).astype(int)
    n = len(df)
    t_k, c_k = np.full(n, -1), np.full(n, -1)
    d_gold, d_commit = np.full(n, np.nan), np.full(n, np.nan)
    conv_gold, commit = np.zeros(n, int), np.zeros(n, int)
    traces = []
    for sid, g in df.groupby("sample", sort=False):
        idx = g.index.values
        pos, rem = g["pos"].values, g["rem"].values
        eqg, eqf = g["eq_gold"].values.astype(bool), g["eq_final"].values.astype(bool)
        has_final = bool(fin[idx[0]] != "")
        correct = bool(g["correct"].iloc[0])
        ts = _stable_from(eqg)
        cs = _stable_from(eqf) if has_final else None
        seq = forced[idx].tolist()
        flips = int(sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b))
        if ts is not None:
            t_k[idx] = ts
            conv_gold[idx] = (np.arange(len(idx)) >= ts).astype(int)
            d_gold[idx] = pos - pos[ts]
        if cs is not None:
            c_k[idx] = cs
            commit[idx] = (np.arange(len(idx)) >= cs).astype(int)
            d_commit[idx] = pos - pos[cs]
        slack_g = int(max(rem[ts] - 1, 0)) if ts is not None else -1
        slack_c = int(max(rem[cs] - 1, 0)) if cs is not None else -1
        traces.append({
            "sample": sid, "n_ckpt": len(idx), "t_star": -1 if ts is None else ts, "c_star": -1 if cs is None else cs,
            "slack_gold": slack_g, "slack_commit": slack_c, "flips": flips, "correct": int(correct),
            "first_gold_k": int(np.argmax(eqg)) if eqg.any() else -1,
            "cond_overthink": int(correct and ts is not None and slack_g >= tau),
            "cond_rationalize": int((not correct) and cs is not None and slack_c >= tau),
            "cond_lost": int((not correct) and bool(eqg[1:].any())),
            "cond_dither": int(flips >= 2),
        })
    df["t_star"], df["c_star"] = t_k, c_k
    df["conv_gold"], df["commit"] = conv_gold, commit
    df["d_gold"], df["d_commit"] = d_gold, d_commit
    df["committed_wrong"] = (df["commit"].astype(bool) & (df["correct"] == 0)).astype(int)
    df["converged_right"] = (df["conv_gold"].astype(bool) & (df["correct"] == 1)).astype(int)
    return df, pd.DataFrame(traces)
