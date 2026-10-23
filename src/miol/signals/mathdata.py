"""MATH (Hendrycks) problems with INTEGER answers: loading, exact answer parsing, prompt building.

Why integers only: the forced-answer score  a = log P(gold answer | prefix + cue)  is exact only when the gold answer has one surface form.
Fractions, radicals and expressions have many (\\frac / \\dfrac / a/b), which would make `a` a noisy under-estimate.  Integers also keep
'correct' a plain string comparison with no equivalence checker.
"""
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from miol.signals.text_utils import extract_boxed, norm_number, strip_special

SYSTEM_PROMPT_MATH = "Please reason step by step, and put your final answer within \\boxed{}."
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
# (dataset, config-or-None) tried in order; the first one that loads wins.  Compute nodes are offline: run this on the login node.
CANDIDATES = [("EleutherAI/hendrycks_math", "subjects"), ("DigitalLearningGmbH/MATH-lighteval", None),
              ("nlile/hendrycks-MATH-benchmark", None), ("qwedsacf/competition_math", None)]


def clean_math_answer(s: Optional[str]) -> Optional[str]:
    """'\\$1{,}000' -> '1000', '45^\\circ' -> '45', '12\\text{ cm}' -> '12', '20\\%' -> '20'.  None unless the WHOLE content is an integer."""
    if s is None:
        return None
    s = re.sub(r"\\text\{[^{}]*\}", "", s)
    for a, b in (("\\$", ""), ("$", ""), ("\\!", ""), ("\\,", ""), ("\\ ", ""), ("{,}", ","), ("\\%", ""), ("%", ""),
                 ("^\\circ", ""), ("^{\\circ}", ""), ("\\circ", ""), ("\\left", ""), ("\\right", ""), ("\\dots", "")):
        s = s.replace(a, b)
    s = s.strip().rstrip(".").strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    if not re.fullmatch(r"-?\d+(\.0+)?", s) or len(s.lstrip("-")) > 12:
        return None
    return norm_number(s)


def math_gold(solution: str) -> Optional[str]:
    return clean_math_answer(extract_boxed(solution))


def parse_final_math(completion_text: str) -> Optional[str]:
    """The trace's OWN final answer: the last \\boxed{} content, if it is an integer (else None = wrong / no usable answer)."""
    return clean_math_answer(extract_boxed(strip_special(completion_text)))


def _norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {c: c for c in df.columns}
    for want, alts in (("problem", ["problem", "question"]), ("solution", ["solution", "answer_text"]),
                       ("level", ["level"]), ("type", ["type", "subject"])):
        for a in alts:
            if a in df.columns:
                ren[a] = want
                break
    df = df.rename(columns=ren)
    for c in ("level", "type"):
        if c not in df.columns:
            df[c] = ""
    return df[["problem", "solution", "level", "type"]]


def load_math_splits(candidates=CANDIDATES, loader: Optional[Callable] = None, log=print) -> Dict[str, pd.DataFrame]:
    """-> {'train': df, 'test': df} with columns problem, solution, level, type (all MATH problems, unfiltered)."""
    if loader is None:
        from datasets import load_dataset as loader
    last = None
    for name, cfg in candidates:
        try:
            out = {}
            if cfg == "subjects":
                for split in ("train", "test"):
                    parts = [pd.DataFrame(loader(name, s, split=split)).assign(type=s) for s in SUBJECTS]
                    out[split] = _norm_columns(pd.concat(parts, ignore_index=True))
            else:
                ds = loader(name, cfg) if cfg else loader(name)
                if hasattr(ds, "keys") and "train" in ds and "test" in ds:
                    out = {s: _norm_columns(pd.DataFrame(ds[s])) for s in ("train", "test")}
                else:                                            # single split: deterministic 60/40 split
                    allp = _norm_columns(pd.DataFrame(ds["train"] if hasattr(ds, "keys") else ds))
                    perm = allp.sample(frac=1.0, random_state=0).reset_index(drop=True)
                    k = int(0.6 * len(perm))
                    out = {"train": perm.iloc[:k].reset_index(drop=True), "test": perm.iloc[k:].reset_index(drop=True)}
            log(f"loaded MATH from {name}: train {len(out['train'])}, test {len(out['test'])}")
            return out
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last = exc
            log(f"  {name}: {type(exc).__name__}: {str(exc)[:120]}")
    raise RuntimeError(f"no MATH dataset could be loaded (last error: {last}). Download one on the login node first.")


def build_problem_files(out_dir: Path, splits: Dict[str, pd.DataFrame], log=print) -> Dict[str, int]:
    """Keep integer-answer problems; write <out_dir>/problems_{train,test}.jsonl with {idx, problem, gold, level, type}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, df in splits.items():
        rows = []
        for r in df.itertuples(index=False):
            g = math_gold(r.solution)
            if g is not None:
                rows.append({"problem": r.problem.strip(), "gold": g, "level": str(r.level), "type": str(r.type)})
        for i, r in enumerate(rows):
            r["idx"] = i
        with open(out_dir / f"problems_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts[split] = len(rows)
        log(f"{split}: kept {len(rows)} of {len(df)} problems with an integer answer")
    return counts


def read_problems(path: Path) -> List[dict]:
    return [json.loads(l) for l in open(path)]


USER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def prompt_text(tok, problem: str, system: str = SYSTEM_PROMPT_MATH) -> str:
    """system != '': Qwen-style (system message + problem).  system == '': R1-style (no system message; the same instruction goes
    into the user turn, and the prompt is guaranteed to end inside an open <think> block)."""
    if system:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": problem}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    msgs = [{"role": "user", "content": problem + "\n" + USER_INSTRUCTION}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "<think>" not in text[-40:]:
        text += "<think>\n"
    return text
