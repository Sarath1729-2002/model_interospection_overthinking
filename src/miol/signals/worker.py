"""One GPU worker: generate -> checkpoint states + token statistics -> forced-answer scores, chunk by chunk, resumable.

Nothing per-token is stored: only the hidden states at the checkpoints (all layers, float16), which is ~100x smaller than the full cache.

A chunk = `chunk_problems` problems.  Per chunk we write (atomically, `.done` last):
    c<id>_ckpt.csv    one row per checkpoint: text features, token statistics, forced-answer scores, forecasting columns
    c<id>_traces.csv  one row per problem (also the truncated / unfinished ones)
    c<id>_states.npy  float16 [layers, n_ckpt, dim]  in ckpt-row order
    c<id>_ids.jsonl   prompt / completion token ids (rollouts and audits need them)
"""
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from miol.signals.checkpoints import checkpoint_positions
from miol.signals.mathdata import SYSTEM_PROMPT_MATH, parse_final_math, prompt_text
from miol.signals.scoring import DEFAULT_CUE, AnswerScorer
from miol.signals.text_utils import HEDGE_RE, boundary_token_ids, hash_bow, prefix_features

FP16_MAX = 6.0e4
NAN = float("nan")


def _bow_str(text: str, dim: int = 2048) -> str:
    return " ".join(map(str, hash_bow(text, dim)))


class Worker:
    def __init__(self, model, tok, device, system: str = SYSTEM_PROMPT_MATH, cue: str = DEFAULT_CUE, max_new: int = 2048,
                 gen_kw: Optional[dict] = None, fwd_token_budget: int = 14000, score_token_budget: int = 24000,
                 score_max_batch: int = 48, mode: str = "lines", stride: int = 8, seed: int = 0, log=print):
        self.model, self.tok, self.device, self.system, self.max_new = model, tok, device, system, max_new
        self.gen_kw = gen_kw if gen_kw is not None else dict(temperature=0.7, top_p=0.8, top_k=20)
        self.fwd_token_budget, self.mode, self.stride, self.seed, self.log = fwd_token_budget, mode, stride, seed, log
        self.base = getattr(model, getattr(model, "base_model_prefix", "model"), model)
        self.head = model.get_output_embeddings()
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        gc = getattr(model, "generation_config", None)
        eos = set()
        if tok.eos_token_id is not None:
            eos.add(int(tok.eos_token_id))
        ge = getattr(gc, "eos_token_id", None)
        for e in (ge if isinstance(ge, (list, tuple)) else [ge]):
            if e is not None:
                eos.add(int(e))
        self.eos_ids = eos
        self.boundary = boundary_token_ids(tok, len(tok))
        self.scorer = AnswerScorer(model, tok, device, cue, token_budget=score_token_budget, max_batch=score_max_batch)
        self.generate_fn: Callable = self._generate           # tests replace this with canned completions

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def _generate(self, prompts: List[List[int]], seed: int):
        """-> list of (completion_ids, has_eos).  Completion includes its EOS token."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        res = [None] * len(prompts)

        def run(idx):
            Lmax = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), Lmax), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), Lmax), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, Lmax - len(prompts[i]):] = torch.tensor(prompts[i])
                mask[r, Lmax - len(prompts[i]):] = 1
            torch.manual_seed(seed + idx[0])
            out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), max_new_tokens=self.max_new,
                                      do_sample=True, pad_token_id=self.pad_id, repetition_penalty=1.0, **self.gen_kw)
            new = out[:, Lmax:].tolist()
            outs = []
            for row in new:
                cut = next((j for j, t in enumerate(row) if t in self.eos_ids), None)
                outs.append((row[:cut + 1], True) if cut is not None else (row, False))
            return outs

        for i, o in zip(order, self.scorer._guard(run, order)):
            res[i] = o
        return res

    # ------------------------------------------------------------------ checkpoint rows (no model needed)
    def trace_rows(self, sid: int, split: str, prob: dict, p_ids: List[int], c_ids: List[int]):
        tok = self.tok
        n = len(c_ids)
        text_noeos = tok.decode(c_ids[:-1])
        final = parse_final_math(text_noeos)
        gold = str(prob["gold"])
        correct = int(final is not None and final == gold)
        cps = checkpoint_positions(c_ids, self.boundary, c_ids[-1], True, self.mode, self.stride)
        m = len(cps) - 1
        qbow = _bow_str(prob["problem"])
        rows = []
        for k, (pos, kind) in enumerate(cps):
            text = tok.decode(c_ids[:pos + 1]) if pos >= 0 else ""
            nxt = cps[k + 1][0] if k < m else None
            seg_next = tok.decode(c_ids[pos + 1:nxt + 1]) if nxt is not None else ""
            rest = tok.decode(c_ids[pos + 1:-1]) if pos >= 0 else text_noeos
            rows.append({"split": split, "sample": sid, "k": k, "pos": pos, "kind": kind, "n_tok": n, "prompt_len": len(p_ids),
                         "gold": gold, "final": final if final is not None else "", "correct": correct, "rem": n - (pos + 1),
                         "tail": text[-200:].replace("\n", "⏎"), **prefix_features(text, gold, final),
                         "pbow": _bow_str(text), "qbow": qbow, "level": prob.get("level", ""), "type": prob.get("type", ""),
                         "hedge_next": float(len(HEDGE_RE.findall(seg_next))) if nxt is not None else NAN,
                         "hedge_rem": float(len(HEDGE_RE.findall(rest))),
                         "len_next": float(nxt - pos) if nxt is not None else NAN, "steps_rem": float(m - k)})
        return rows, final, correct

    # ------------------------------------------------------------------ forward pass: states + token statistics
    @torch.no_grad()
    def forward_states(self, items: List[dict]):
        """items: dict(prompt, comp, pos=[checkpoint positions]).  -> per item (states [L,n_ck,D] float16, H [n], S [n], clipped)."""
        order = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]) + len(items[i]["comp"]))
        out: List = [None] * len(items)
        batches, cur, mx = [], [], 0
        for i in order:
            L = len(items[i]["prompt"]) + len(items[i]["comp"])
            if cur and max(mx, L) * (len(cur) + 1) > self.fwd_token_budget:
                batches.append(cur)
                cur, mx = [], 0
            cur.append(i)
            mx = max(mx, L)
        if cur:
            batches.append(cur)

        def run(idx):
            seqs = [items[i]["prompt"] + items[i]["comp"] for i in idx]
            B, Lmax = len(idx), max(len(s) for s in seqs)
            ids = torch.full((B, Lmax), self.pad_id, dtype=torch.long)
            mask = torch.zeros((B, Lmax), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, :len(s)] = torch.tensor(s)
                mask[r, :len(s)] = 1
            o = self.base(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), output_hidden_states=True)
            hs, last = o.hidden_states, o.last_hidden_state
            res = []
            for r, i in enumerate(idx):
                it = items[i]
                pl, n = len(it["prompt"]), len(it["comp"])
                rows = torch.tensor([pl + p for p in it["pos"]], device=self.device)
                st = torch.stack([h[r].index_select(0, rows) for h in hs], 0).float()
                clipped = int((st.abs() > FP16_MAX).sum())
                st = st.clamp(-FP16_MAX, FP16_MAX).half().cpu().numpy()
                hl = last[r, pl - 1: pl + n - 1]
                tgt = torch.tensor(it["comp"], device=self.device)
                H, S = [], []
                for a in range(0, n, 512):
                    lp = torch.log_softmax(self.head(hl[a:a + 512]).float(), dim=-1)
                    H.append(-(lp.exp() * lp).sum(-1))
                    S.append(-lp.gather(1, tgt[a:a + 512, None]).squeeze(1))
                res.append((st, torch.cat(H).cpu().numpy(), torch.cat(S).cpu().numpy(), clipped))
            return res

        for b in batches:
            for i, o in zip(b, self.scorer._guard(run, b)):
                out[i] = o
        return out

    @staticmethod
    def token_stats(pos: List[int], H: np.ndarray, S: np.ndarray, n: int) -> Dict[str, List[float]]:
        """H[i], S[i] belong to completion position p = i-1 (the state that predicts token p+1)."""
        cH, cS = np.concatenate([[0.0], np.cumsum(H)]), np.concatenate([[0.0], np.cumsum(S)])

        def mean(c, a, b):                      # mean over positions p in [a, b] (inclusive); empty -> nan
            if b < a:
                return NAN
            return float((c[b + 2] - c[a + 1]) / (b - a + 1))
        m = len(pos) - 1
        out = {k: [] for k in ("ent_here", "ent_past", "ent_last", "surp_last", "surp_past", "ent_next", "surp_next", "ent_rem", "surp_rem")}
        for k, p in enumerate(pos):
            prev = pos[k - 1] if k > 0 else None
            nxt = pos[k + 1] if k < m else None
            out["ent_here"].append(float(H[p + 1]))
            out["ent_past"].append(0.0 if np.isnan(mean(cH, -1, p - 1)) else mean(cH, -1, p - 1))
            out["surp_past"].append(0.0 if np.isnan(mean(cS, -1, p - 1)) else mean(cS, -1, p - 1))
            out["ent_last"].append(0.0 if prev is None else mean(cH, prev, p - 1))
            out["surp_last"].append(0.0 if prev is None else mean(cS, prev, p - 1))
            out["ent_next"].append(NAN if nxt is None else mean(cH, p, nxt - 1))
            out["surp_next"].append(NAN if nxt is None else mean(cS, p, nxt - 1))
            out["ent_rem"].append(mean(cH, p, n - 2))
            out["surp_rem"].append(mean(cS, p, n - 2))
        return out

    # ------------------------------------------------------------------ one chunk
    def process_chunk(self, split: str, problems: List[dict], chunk_id: int):
        t0 = time.time()
        prompts = [self.tok.encode(prompt_text(self.tok, p["problem"], self.system), add_special_tokens=False) for p in problems]
        gens = self.generate_fn(prompts, self.seed + 1000 * chunk_id)
        t_gen = time.time() - t0
        rows, traces, items, owner, id_rows = [], [], [], [], []
        for prob, p_ids, (c_ids, has_eos) in zip(problems, prompts, gens):
            sid = int(prob["idx"])
            tr = {"split": split, "sample": sid, "n_tok": len(c_ids), "prompt_len": len(p_ids), "gold": str(prob["gold"]),
                  "has_eos": int(has_eos), "level": prob.get("level", ""), "type": prob.get("type", "")}
            if has_eos and len(c_ids) >= 2:
                r, final, correct = self.trace_rows(sid, split, prob, p_ids, c_ids)
                tr.update(final=final if final is not None else "", final_ok=int(final is not None), correct=correct, n_ckpt=len(r))
                rows.extend(r)
                items.append({"prompt": p_ids, "comp": c_ids, "pos": [x["pos"] for x in r]})
                owner.append((sid, len(r)))
            else:
                tr.update(final="", final_ok=0, correct=0, n_ckpt=0)
            traces.append(tr)
            id_rows.append({"sample": sid, "prompt_ids": p_ids, "completion_ids": c_ids, "has_eos": bool(has_eos)})
        ck = pd.DataFrame(rows)
        states = np.zeros((0, 0, 0), np.float16)
        clipped = 0
        t_fwd = t_sc = 0.0
        if len(ck):
            t1 = time.time()
            fw = self.forward_states(items)
            t_fwd = time.time() - t1
            stat_cols: Dict[str, List[float]] = {}
            sts = []
            for it, (st, H, S, c) in zip(items, fw):
                clipped += c
                sts.append(st)
                for key, vals in self.token_stats(it["pos"], H, S, len(it["comp"])).items():
                    stat_cols.setdefault(key, []).extend(vals)
            states = np.concatenate(sts, axis=1)
            for key, vals in stat_cols.items():
                ck[key] = vals
            t2 = time.time()
            ck = self.score(ck, {int(p["idx"]): (pr, c) for p, pr, (c, he) in zip(problems, prompts, gens) if he})
            t_sc = time.time() - t2
        self.log(f"chunk {chunk_id}: {len(problems)} problems, {len(ck)} checkpoints, gen {t_gen:.0f}s fwd {t_fwd:.0f}s score {t_sc:.0f}s, "
                 f"truncated {sum(1 for t in traces if not t['has_eos'])}, clipped {clipped}")
        return ck, pd.DataFrame(traces), states, id_rows

    def score(self, ck: pd.DataFrame, ids: Dict[int, tuple]) -> pd.DataFrame:
        ctxs, gold_a, final_a, keys = [], [], [], []
        for sid, g in ck.groupby("sample", sort=False):
            p_ids, c_ids = ids[int(sid)]
            for r in g.itertuples(index=False):
                ctxs.append(self.scorer.context(p_ids, c_ids, int(r.pos)))
                gold_a.append(str(r.gold))
                final_a.append(str(r.final) if isinstance(r.final, str) and r.final != "" else None)
                keys.append((int(sid), int(r.k)))
        sg = self.scorer.score(ctxs, gold_a)
        need = [i for i in range(len(ctxs)) if final_a[i] is not None and final_a[i] != gold_a[i]]
        sf = self.scorer.score([ctxs[i] for i in need], [final_a[i] for i in need]) if need else []
        fmap = dict(zip(need, sf))
        forced = self.scorer.greedy(ctxs)
        recs = []
        for i, (sid, k) in enumerate(keys):
            a = sg[i]
            f = NAN if final_a[i] is None else (a["a"] if final_a[i] == gold_a[i] else fmap[i]["a"])
            recs.append({"sample": sid, "k": k, "a": a["a"], "a_digits": a["a_digits"], "term": a["term"], "ent1": a["ent1"],
                         "p1max": a["p1max"], "f": f, "forced": forced[i]})
        return ck.merge(pd.DataFrame(recs), on=["sample", "k"], how="left", validate="one_to_one")


def write_chunk(out: Path, split: str, chunk_id: int, ck, traces, states, id_rows):
    d = Path(out) / "chunks" / split
    d.mkdir(parents=True, exist_ok=True)
    base = d / f"c{chunk_id:05d}"
    ck.to_csv(f"{base}_ckpt.csv", index=False)
    traces.to_csv(f"{base}_traces.csv", index=False)
    np.save(f"{base}_states.npy", states)
    with open(f"{base}_ids.jsonl", "w") as f:
        for r in id_rows:
            f.write(json.dumps(r) + "\n")
    Path(f"{base}.done").write_text(json.dumps({"checkpoints": int(len(ck)), "traces": int(len(traces))}))


def chunk_done(out: Path, split: str, chunk_id: int) -> bool:
    return (Path(out) / "chunks" / split / f"c{chunk_id:05d}.done").exists()
