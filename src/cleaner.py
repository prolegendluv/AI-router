"""Output cleaning — strip model chatter the accuracy grader penalises.

Removes reasoning blocks (<think>...</think>), leading preambles ("Here are the
entities:", "Sure, ...", "Below is ..."), and stray markdown fences on
non-code answers. Also lightly normalises summarisation outputs to the exact
sentence / bullet count requested, so a correct answer isn't docked purely for
format. Applied to BOTH local and Fireworks answers.
"""
from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)

# Preambles to drop when they start the answer.
_PREAMBLE = re.compile(
    r"^\s*(sure[,!.]?|certainly[,!.]?|of course[,!.]?|here(?:'s| is| are)"
    r"[^\n:]*:?|below is[^\n:]*:?|the answer is[:]?)\s*",
    re.I,
)
_CODE_CATS = {"code_generation", "code_debugging"}


def _strip_think(text: str) -> str:
    # Remove a complete <think>...</think> block.
    text = _THINK.sub("", text)
    # If only a closing tag remains, keep what follows it (the real answer).
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # Drop any dangling think tags but KEEP the surrounding text, so an
    # unclosed <think> never nukes the whole answer to empty.
    text = re.sub(r"</?think>", "", text, flags=re.I)
    return text


def _strip_fences(text: str, category: str) -> str:
    # Remove ```lang ... ``` wrappers around prose or single code block outputs
    # so automated execution benchmarks (exec / compile) don't hit SyntaxErrors.
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    if category in _CODE_CATS:
        return text
    return stripped


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _is_bullet(line: str) -> bool:
    return bool(re.match(r"^\s*([-*•]|\d+[.)])\s+", line))


def _limit_summary(text: str, prompt: str) -> str:
    """If the prompt fixes a sentence or bullet count, trim to it."""
    p = prompt.lower()

    # Bullet-count constraint, e.g. "exactly three bullet points".
    m = re.search(r"(exactly\s+)?(one|two|three|four|five|\d+)\s+bullet", p)
    if m:
        n = _word_to_int(m.group(2))
        lines = text.splitlines()
        bullets = [l for l in lines if _is_bullet(l)]
        if n and len(bullets) > n:
            keep, count = [], 0
            for l in lines:
                if _is_bullet(l):
                    if count >= n:
                        continue
                    count += 1
                keep.append(l)
            text = "\n".join(keep).strip()
        return text

    # Sentence-count constraint, e.g. "in exactly two sentences".
    m = re.search(r"(exactly\s+)?(one|two|three|four|five|\d+)\s+sentence", p)
    if m:
        n = _word_to_int(m.group(2))
        sents = _sentences(text)
        if n and len(sents) > n:
            text = " ".join(sents[:n])
    return text


def _word_to_int(w: str):
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    if w in words:
        return words[w]
    try:
        return int(w)
    except ValueError:
        return None


def clean(text: str, category: str, prompt: str = "") -> str:
    if not text:
        return ""
    out = _strip_think(text)
    out = _strip_fences(out, category)
    out = _PREAMBLE.sub("", out).strip()
    # Drop a trailing sign-off line some models add.
    out = re.sub(r"\n+\s*(let me know if.*|hope this helps.*)$", "", out, flags=re.I).strip()
    if category == "text_summarization" and prompt:
        out = _limit_summary(out, prompt)
    if category == "named_entity_recognition":
        lines = []
        for l in out.splitlines():
            l_clean = re.sub(r"^\s*([-*•]|\d+[.)])\s*", "", l).strip()
            if l_clean and "-" in l_clean:
                lines.append(l_clean)
            elif l_clean:
                lines.append(l_clean)
        out = "\n".join(lines)
    if category == "mathematical_reasoning":
        ans = re.findall(
            r"(?im)^\s*(?:[#*`]+\s*)?(?:final\s+)?answer\s*(?:[#*`]+\s*)?[:\-]\s*(.+?)\s*$",
            out,
        )
        if ans:
            val = ans[-1].strip().strip("*`")
            val = re.sub(r"^\\boxed\{(.+?)\}$", r"\1", val).strip()
            out = "Answer: " + val.strip()
    return out.strip()
