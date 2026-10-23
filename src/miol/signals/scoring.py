"""Forced-answer scoring: at a checkpoint, append the cue and read the model's answer distribution.

For a context  [prompt][completion prefix][\\n\\n][cue = "Therefore, the final answer is \\(\\boxed{"]  we compute
    a  = log P(answer digits, then a non-digit)      for a given answer string (the gold answer, or the trace's own)
         = sum_i log P(d_i | ctx, d_<i)  +  log(1 - P(next token is a digit))
    forced = the greedy answer (<= max_new tokens, first number parsed)
    ent1, p1max = entropy and top probability of the first answer token (a confidence signal)
Everything is exact: no sampling, no parser except for the greedy string.
"""
import math
from typing import Dict, List, Optional

import torch

from miol.signals.text_utils import digit_token_ids, first_number

DEFAULT_CUE = "Therefore, the final answer is \\(\\boxed{"
NAN = float("nan")


class AnswerScorer:
    def __init__(self, model, tok, device, cue: str = DEFAULT_CUE, pad_id: Optional[int] = None,
                 token_budget: int = 24000, max_batch: int = 48, max_new: int = 10):
        self.model, self.tok, self.device = model, tok, device
        self.base = getattr(model, getattr(model, "base_model_prefix", "model"), model)
        self.head = model.get_output_embeddings()
        self.digit_ids = digit_token_ids(tok)
        m = tok.encode("-", add_special_tokens=False)
        self.minus_id = m[0] if len(m) == 1 else None
        self.cue_ids = tok.encode(cue, add_special_tokens=False)
        self.sep_ids = tok.encode("\n\n", add_special_tokens=False)
        self.pad_id = pad_id if pad_id is not None else (tok.pad_token_id if tok.pad_token_id is not None else 0)
        self.token_budget, self.max_batch, self.max_new = token_budget, max_batch, max_new
        self._digit_t = torch.tensor(self.digit_ids, device=device)

    # ---- contexts ---------------------------------------------------------------------------------------
    def context(self, prompt_ids: List[int], comp_ids: List[int], pos: int) -> List[int]:
        """prompt + completion[:pos+1] (+ blank-line separator unless it already ends with one) + cue."""
        ids = list(prompt_ids) + list(comp_ids[:pos + 1])
        if pos >= 0 and not self.tok.decode(comp_ids[pos:pos + 1]).endswith("\n\n"):
            ids += self.sep_ids
        return ids + self.cue_ids

    def encode_answer(self, s: Optional[str]) -> Optional[List[int]]:
        if not s:
            return None
        ids = self.tok.encode(s, add_special_tokens=False)
        ok = set(self.digit_ids) | ({self.minus_id} if self.minus_id is not None else set())
        return ids if ids and all(i in ok for i in ids) else None

    # ---- batching ---------------------------------------------------------------------------------------
    def _batches(self, order, length):
        cur, mx = [], 0
        for i in order:
            L = length(i)
            if cur and (len(cur) >= self.max_batch or max(mx, L) * (len(cur) + 1) > self.token_budget):
                yield cur
                cur, mx = [], 0
            cur.append(i)
            mx = max(mx, L)
        if cur:
            yield cur

    def _guard(self, fn, idx):
        """Run fn(idx); on CUDA OOM split the batch in halves."""
        try:
            return fn(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            mid = len(idx) // 2
            return self._guard(fn, idx[:mid]) + self._guard(fn, idx[mid:])

    # ---- teacher-forced answer log-probability ----------------------------------------------------------
    @torch.no_grad()
    def score(self, ctxs: List[List[int]], answers: List[Optional[str]]) -> List[Dict[str, float]]:
        ans_ids = [self.encode_answer(a) for a in answers]
        res: List[Optional[Dict[str, float]]] = [None] * len(ctxs)
        valid = [i for i in range(len(ctxs)) if ans_ids[i] is not None]
        for i in range(len(ctxs)):
            if ans_ids[i] is None:
                res[i] = {"a": NAN, "a_digits": NAN, "term": NAN, "ent1": NAN, "p1max": NAN}
        order = sorted(valid, key=lambda i: len(ctxs[i]) + len(ans_ids[i]))

        def run(idx):
            seqs = [ctxs[i] + ans_ids[i] for i in idx]
            B, Lmax = len(idx), max(len(s) for s in seqs)
            ids = torch.full((B, Lmax), self.pad_id, dtype=torch.long)
            mask = torch.zeros((B, Lmax), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, :len(s)] = torch.tensor(s)
                mask[r, :len(s)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            h = self.base(input_ids=ids, attention_mask=mask).last_hidden_state              # [B, L, H]
            P = max(len(ans_ids[i]) for i in idx) + 1
            gather = torch.zeros((B, P), dtype=torch.long)
            tgt = torch.zeros((B, P), dtype=torch.long)
            vmask = torch.zeros((B, P))
            m_len = torch.zeros(B, dtype=torch.long)
            for r, i in enumerate(idx):
                c, a = len(ctxs[i]), ans_ids[i]
                m_len[r] = len(a)
                for j in range(len(a) + 1):
                    gather[r, j] = c - 1 + j
                for j in range(len(a)):
                    tgt[r, j] = a[j]
                    vmask[r, j] = 1.0
                gather[r, len(a) + 1:] = c - 1 + len(a)
            gather, tgt, vmask, m_len = gather.to(self.device), tgt.to(self.device), vmask.to(self.device), m_len.to(self.device)
            hg = h.gather(1, gather[:, :, None].expand(-1, -1, h.shape[-1]))                # [B, P, H]
            logp = torch.log_softmax(self.head(hg).float(), dim=-1)                          # [B, P, V]
            lp = logp.gather(2, tgt[:, :, None]).squeeze(-1) * vmask
            a_digits = lp.sum(1)
            at_end = logp[torch.arange(B, device=self.device), m_len]                        # [B, V]
            s = torch.logsumexp(at_end[:, self._digit_t], dim=-1)
            term = torch.log1p(-torch.exp(s).clamp(max=1 - 1e-6))
            p0 = logp[:, 0]
            ent1 = -(p0.exp() * p0).sum(-1)
            p1max = p0.exp().max(-1).values
            out = []
            for r in range(B):
                out.append({"a": float(a_digits[r] + term[r]), "a_digits": float(a_digits[r]),
                            "term": float(term[r]), "ent1": float(ent1[r]), "p1max": float(p1max[r])})
            return out

        for batch in self._batches(order, lambda i: len(ctxs[i]) + len(ans_ids[i])):
            for i, o in zip(batch, self._guard(run, batch)):
                res[i] = o
        return res

    # ---- greedy forced answer ---------------------------------------------------------------------------
    @torch.no_grad()
    def greedy(self, ctxs: List[List[int]]) -> List[str]:
        res = [""] * len(ctxs)
        order = sorted(range(len(ctxs)), key=lambda i: len(ctxs[i]))

        def run(idx):
            B, Lmax = len(idx), max(len(ctxs[i]) for i in idx)
            ids = torch.full((B, Lmax), self.pad_id, dtype=torch.long)
            mask = torch.zeros((B, Lmax), dtype=torch.long)
            for r, i in enumerate(idx):
                s = ctxs[i]
                ids[r, Lmax - len(s):] = torch.tensor(s)           # left padding for generation
                mask[r, Lmax - len(s):] = 1
            out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                      max_new_tokens=self.max_new, do_sample=False, pad_token_id=self.pad_id,
                                      temperature=None, top_p=None, top_k=None,
                                      repetition_penalty=1.0)   # Qwen's checkpoint ships 1.05, which would bias digits
            new = out[:, Lmax:]
            return [first_number(self.tok.decode(new[r].tolist(), skip_special_tokens=True)) or "" for r in range(B)]

        for batch in self._batches(order, lambda i: len(ctxs[i])):
            for i, o in zip(batch, self._guard(run, batch)):
                res[i] = o
        return res
