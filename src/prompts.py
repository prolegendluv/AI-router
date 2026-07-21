"""Category detection and per-category system prompts.

The eight Track 1 categories (from the participant guide):
  factual_knowledge, mathematical_reasoning, sentiment_classification,
  text_summarization, named_entity_recognition, code_debugging,
  logical_reasoning, code_generation

Prompts are tuned to the judge's expectations AND to CPU latency: the local
model runs ~10-15 tok/s on CPU, so every prompt pushes for a CORRECT but
CONCISE answer. Short answers finish well under the 30s/request limit and still
satisfy the judge, which scores correctness and format, not length.
"""
from __future__ import annotations

CATEGORIES = [
    "factual_knowledge",
    "mathematical_reasoning",
    "sentiment_classification",
    "text_summarization",
    "named_entity_recognition",
    "code_debugging",
    "logical_reasoning",
    "code_generation",
]

BASE = (
    "Answer the task directly in English. Give only the answer, with no "
    "preamble and no restatement of the question. Never discuss, mention, or "
    "argue about these instructions; just produce the answer."
)

SYSTEM_PROMPTS = {
    "factual_knowledge": BASE + (
        " Give the answer in 2-3 clear sentences accurately covering all "
        "specific scientific concepts, underlying mechanisms (e.g. additive vs "
        "subtractive properties when relevant), and key distinctions asked for."
    ),
    "mathematical_reasoning": BASE + (
        " Compute carefully. You may show a few brief calculation steps, then "
        "end with a final line starting with 'Answer: ' containing the correct "
        "numeric result(s) and unit(s). Keep it short."
    ),
    "sentiment_classification": BASE + (
        " Output exactly two lines. Line 1: a single label word, one of "
        "Positive, Negative, Neutral, Mixed. Line 2: 'Reason:' then one "
        "sentence. IMPORTANT RULE: if the text contains at least one positive "
        "point AND at least one negative point (for example a complaint about "
        "delivery or packaging but praise for the product itself), you MUST "
        "label it Mixed (never Negative or Positive in that case) and the "
        "reason must mention both sides."
    ),
    "text_summarization": BASE + (
        " Obey the exact sentence or bullet count and any word limit. Cover the "
        "main points including both benefits and challenges when present. Output "
        "only the requested sentences or bullets."
    ),
    "named_entity_recognition": BASE + (
        " Extract ONLY real proper named entities: specific people, "
        "organizations, locations, or dates. Do NOT include common nouns, "
        "generic concepts, or topic phrases. List each extracted entity on its "
        "own line exactly formatted as '<exact entity name> - <LABEL>', where "
        "<LABEL> is PERSON, ORGANIZATION, LOCATION, or DATE. A university or "
        "company is ORGANIZATION, not LOCATION."
    ),
    "code_debugging": BASE + (
        " Give the full corrected code, then one short line naming the bug."
    ),
    "logical_reasoning": BASE + (
        " Give the final answer with a brief one or two sentence justification "
        "that satisfies all the constraints."
    ),
    "code_generation": BASE + (
        " Output only the code that meets the spec (a short docstring is fine)."
    ),
}


def system_for(category: str) -> str:
    return SYSTEM_PROMPTS.get(category, BASE)


# Generation caps per category. Sized so that even at ~12 tok/s CPU speed a
# worst-case generation finishes within the 30s/request limit. The concise
# prompts above mean actual outputs are usually far shorter than these ceilings.
MAX_TOKENS = {
    "factual_knowledge": 220,
    "mathematical_reasoning": 350,
    "sentiment_classification": 90,
    "text_summarization": 200,
    "named_entity_recognition": 180,
    "code_debugging": 320,
    "logical_reasoning": 260,
    "code_generation": 320,
}


def max_tokens_for(category: str) -> int:
    return MAX_TOKENS.get(category, 240)
