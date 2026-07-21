"""Post-inference format verification.

Returns a 0..1 score for how well a (cleaned) answer meets the structural
expectations of its category and any explicit constraint in the prompt. The
router escalates to Fireworks when this score is low, catching cases where the
local model is confident in its words but wrong in its format.
"""
from __future__ import annotations

import re


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def _bullets(text: str) -> list[str]:
    return [l for l in text.splitlines() if re.match(r"^\s*([-*•]|\d+[.)])\s+", l)]


def _word_to_int(w: str):
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    return words.get(w, int(w) if w.isdigit() else None)


def score(answer: str, category: str, prompt: str) -> float:
    a = (answer or "").strip()
    if not a:
        return 0.0
    p = (prompt or "").lower()

    if category == "text_summarization":
        mb = re.search(r"(one|two|three|four|five|\d+)\s+bullet", p)
        if mb:
            n = _word_to_int(mb.group(1))
            b = _bullets(a)
            is_max = bool(re.search(r"(no more than|at most|up to|maximum|or fewer)", p))
            ok_count = (n is not None and (len(b) <= n if is_max else len(b) == n))
            wl = re.search(r"no longer than\s+(\d+)\s+words", p)
            ok_len = True
            if wl:
                limit = int(wl.group(1))
                ok_len = all(
                    len(re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", x).split()) <= limit
                    for x in b
                )
            return 1.0 if (ok_count and ok_len) else 0.3
        ms = re.search(r"(one|two|three|four|five|\d+)\s+sentence", p)
        if ms:
            n = _word_to_int(ms.group(1))
            is_max = bool(re.search(r"(no more than|at most|up to|maximum|or fewer)", p))
            return 1.0 if (n is not None and (_sentences(a) <= n if is_max else _sentences(a) == n)) else 0.3
        return 0.9  # free-form summary: accept if non-empty

    if category == "named_entity_recognition":
        labels = ("PERSON", "ORGANIZATION", "ORG", "LOCATION", "LOC", "DATE",
                  "MISC")
        has_labels = sum(l in a.upper() for l in labels)
        return 1.0 if (has_labels >= 1 or "-" in a) else 0.3

    if category == "sentiment_classification":
        has_label = re.search(r"\b(positive|negative|neutral|mixed)\b", a, re.I)
        has_reason = len(a.split()) >= 3 or "reason:" in a.lower()
        return 1.0 if (has_label and has_reason) else 0.3

    if category == "mathematical_reasoning":
        # A concrete numeric result should appear.
        return 1.0 if re.search(r"\d", a) else 0.2

    if category in ("code_generation", "code_debugging"):
        looks_like_code = bool(re.search(r"[(){}=;:]|def |return|class |=>", a))
        return 0.9 if looks_like_code else 0.4

    # factual_knowledge, logical_reasoning: length-based sanity.
    return 0.85 if len(a.split()) >= 4 else 0.3
