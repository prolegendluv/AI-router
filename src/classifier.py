"""Zero-cost heuristic classification of an incoming prompt.

Produces:
  - category:   one of the 8 Track 1 categories (drives the system prompt)
  - freshness:  True if the task needs current/future real-world info
  - complexity: True if the task looks hard enough to warrant escalation

All rule-based, so it adds no tokens and negligible latency.
"""
from __future__ import annotations

import re

# --- Freshness indicators: things a static local model cannot know ---
# These key on the *request intent*, not on quoted data the task asks us to
# process. Quoted passages are stripped before matching (see detect_freshness).
FRESHNESS_PATTERNS = [
    r"\b(latest|current(ly)?|today|tonight|tomorrow|yesterday|right now|as of now)\b",
    r"\b(release date|when will|when is .* (coming|releasing|out)|upcoming|"
    r"is .* out yet)\b",
    r"\b(just (launched|announced|released)|newly released|this (week|month|year))\b",
    r"\b(live score|stock price|share price|crypto price|exchange rate|"
    r"market cap)\b",
    r"\b(weather|forecast|election results?|breaking news|who won (the|last))\b",
    r"\b(20(2[6-9]|[3-9][0-9]))\b",  # explicit years 2026+
]
_FRESH_RE = [re.compile(p, re.I) for p in FRESHNESS_PATTERNS]

# Quoted spans are DATA to process (NER/summary/sentiment), not a request for
# current info — strip them before freshness/complexity intent detection.
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"|'''.*?'''|\"\"\".*?\"\"\"", re.S)


def _instruction_only(text: str) -> str:
    return _QUOTED.sub(" ", text or "")

# --- Complexity indicators: escalate-worthy difficulty ---
COMPLEXITY_PATTERNS = [
    r"\bmulti[- ]file\b",
    r"\b(entire|whole) (code ?base|repo(sitory)?|project)\b",
    r"\b(design|architect) (a|the|an) (system|service|distributed)\b",
    r"\b(olympiad|imo|putnam|competition math|prove that)\b",
    r"\b(dynamic programming|np-hard|graph algorithm|concurrency|race condition)\b",
]
_COMPLEX_RE = [re.compile(p, re.I) for p in COMPLEXITY_PATTERNS]

# --- Category keyword signals ---
CODE_HINT = re.compile(
    r"\b(function|def |class |return|for loop|python|javascript|java|c\+\+|"
    r"algorithm|implement|snippet|code|bug|error|exception|compile|```)\b",
    re.I,
)
DEBUG_HINT = re.compile(
    r"\b(bug|fix|debug\w*|error|exception|traceback|corrected? (version|implementation)|does ?n.?t work)\b|"
    r"\bwhy (does|is|did).*\b(fail|crash|wrong|error)\b",
    re.I,
)
MATH_HINT = re.compile(
    r"(\d+\s*%|\bpercent(age)?\b|\bhow (many|much)\b|\bcalculate\w*\b|\bsum\b|"
    r"\btotal\b|\bcost\b|\baverage\b|\bremain\w*\b|[-+*/=]\s*\d|\bcups?\b|\bunits?\b)",
    re.I,
)
SENTIMENT_HINT = re.compile(
    r"\b(sentiment|positive.*negative.*neutral)\b|"
    r"\bclassify.*\b(review|tweet|comment|feedback|sentiment)\b",
    re.I,
)
SUMMARY_HINT = re.compile(
    r"\b(summari[sz]\w*|summary|condenses?|condensing)\b|"
    r"\b(in|exactly|at most|no more than)\s+(one|two|three|four|five|\d+)\s+(sentences?|bullets?|bullet points?)\b",
    re.I,
)
NER_HINT = re.compile(
    r"\b(named entit\w*|PERSON|ORGANIZATION|LOCATION|DATE)\b|"
    r"\b(extract|identify|find|list).*\b(entit\w*|person|organization|location|date)",
    re.I,
)
LOGIC_HINT = re.compile(
    r"\b(puzzle|logic\w*|deduc\w*|if .* then|constraint|riddle|arrange|seating|knights? and knaves)\b|"
    r"\bwho (is|sits|owns|likes)\b",
    re.I,
)


def detect_freshness(text: str) -> bool:
    scope = _instruction_only(text)
    return any(r.search(scope) for r in _FRESH_RE)


def detect_complexity(text: str) -> bool:
    if any(r.search(text) for r in _COMPLEX_RE):
        return True
    # Very long code-oriented prompts are also complexity signals.
    if CODE_HINT.search(text) and len(text) > 1200:
        return True
    return False


def classify_category(text: str) -> str:
    """Order matters: check the most format-specific categories first."""
    if NER_HINT.search(text):
        return "named_entity_recognition"
    if SENTIMENT_HINT.search(text):
        return "sentiment_classification"
    if SUMMARY_HINT.search(text):
        return "text_summarization"
    if DEBUG_HINT.search(text) and CODE_HINT.search(text):
        return "code_debugging"
    if LOGIC_HINT.search(text):
        return "logical_reasoning"
    if CODE_HINT.search(text):
        # Distinguish "write" (generation) vs. "fix" (debugging handled above).
        return "code_generation"
    if MATH_HINT.search(text):
        return "mathematical_reasoning"
    return "factual_knowledge"


def analyze(prompt: str) -> dict:
    text = prompt or ""
    return {
        "category": classify_category(text),
        "freshness": detect_freshness(text),
        "complexity": detect_complexity(text),
    }
