"""Pure-python helpers: answer parsing, checkpoint-boundary detection, text features.  No torch needed."""
import re
import zlib
from typing import Iterable, List, Optional

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
HEDGE_RE = re.compile(
    r"\b(wait|but|however|alternatively|actually|hmm|check|verify|double[- ]check|recheck|re-?evaluate|mistake)\b", re.I)
SPECIAL_TAILS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
QUESTION_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def norm_number(raw: Optional[str]) -> Optional[str]:
    """'1,000' -> '1000', '18.00' -> '18', '$5' -> '5', '-0' -> '0'.  None if it is not a number."""
    if raw is None:
        return None
    s = raw.strip().replace(",", "").replace("$", "").rstrip(".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", "-", ""):
        s = "0"
    return s


def numbers_in(text: str) -> List[str]:
    out = []
    for m in NUM_RE.finditer(text):
        n = norm_number(m.group(0))
        if n is not None:
            out.append(n)
    return out


def first_number(text: str) -> Optional[str]:
    ns = numbers_in(text)
    return ns[0] if ns else None


def last_number(text: str) -> Optional[str]:
    ns = numbers_in(text)
    return ns[-1] if ns else None


def parse_gold(answer_field: str) -> Optional[str]:
    """GSM8K 'answer' column: reasoning, then '#### 72'."""
    if "####" not in answer_field:
        return None
    return first_number(answer_field.split("####")[-1])


def extract_boxed(text: str) -> Optional[str]:
    """Content of the LAST \\boxed{...} (brace-matched), or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def strip_special(text: str) -> str:
    for t in SPECIAL_TAILS:
        text = text.replace(t, "")
    return text


def parse_final_answer(completion_text: str) -> Optional[str]:
    """The trace's OWN final answer: last \\boxed{} number, else the last number of the last paragraph,
    else the last number anywhere.  Deterministic; coverage and agreement with gold are reported by the gate."""
    text = strip_special(completion_text)
    boxed = extract_boxed(text)
    if boxed is not None:
        n = first_number(boxed)
        if n is not None:
            return n
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paras:
        n = last_number(paras[-1])
        if n is not None:
            return n
    return last_number(text)


def question_from_prompt(prompt_text: str) -> Optional[str]:
    m = QUESTION_RE.search(prompt_text)
    return m.group(1).strip() if m else None


def boundary_token_ids(tok, vocab_size: Optional[int] = None, needle: str = "\n\n") -> set:
    """Ids of every vocabulary token whose decoded text contains `needle` (e.g. '\\n\\n', '.\\n\\n')."""
    n = vocab_size if vocab_size is not None else len(tok)
    ids = set()
    for i in range(n):
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if needle in s:
            ids.add(i)
    return ids


def digit_token_ids(tok) -> List[int]:
    """Ids of '0'..'9'; asserts the tokenizer splits numbers into single digits (true for Qwen2.5)."""
    ids = []
    for d in range(10):
        e = tok.encode(str(d), add_special_tokens=False)
        if len(e) != 1:
            raise ValueError(f"digit {d} is not a single token ({e}); scoring assumes single-digit tokenisation")
        ids.append(e[0])
    e = tok.encode("0123456789", add_special_tokens=False)
    if e != ids:
        raise ValueError("number strings are not tokenised digit by digit; scoring assumes single-digit tokenisation")
    return ids


def prefix_features(prefix_text: str, gold: Optional[str], final: Optional[str]) -> dict:
    """Cheap, causal text features of the completion written so far (the 'is it already visible' baselines).
    gold_in / final_in compare against numbers as whole tokens, not substrings."""
    ns = numbers_in(prefix_text)
    nset = set(ns)
    last = ns[-1] if ns else None
    return {
        "n_numbers": len(ns),
        "n_eq": prefix_text.count("="),
        "n_lines": prefix_text.count("\n"),
        "n_hedge": len(HEDGE_RE.findall(prefix_text)),
        "has_boxed": int("\\boxed" in prefix_text),
        "gold_in": int(gold is not None and gold in nset),
        "final_in": int(final is not None and final in nset),
        "last_eq_gold": int(gold is not None and last == gold),
        "last_eq_final": int(final is not None and last == final),
    }


_WORD = re.compile(r"[a-z]+|\d+|[=+\-*/%]")


def hash_bow(text: str, dim: int = 2048) -> List[int]:
    """Deterministic hashed bag-of-words indices (crc32, not Python's randomised hash)."""
    return sorted({zlib.crc32(w.encode()) % dim for w in _WORD.findall(text.lower())})
