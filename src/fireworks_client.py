"""Fireworks safety-net client (OpenAI-compatible, stdlib only).

ALL Fireworks calls go through FIREWORKS_BASE_URL with the harness-injected key.
We use urllib rather than the openai SDK to avoid dependency/version breakage
(e.g. the httpx 'proxies' incompatibility) and keep the image dependency-free.
Tokens returned here are the only tokens that count toward the Track 1 score, so
this path keeps prompts tight and picks the cheapest sufficient model.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional

from . import config, prompts

# Rough size extraction from model IDs like ".../llama-v3p1-8b-instruct".
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.I)
_SMALL_HINTS = re.compile(r"\b(mini|small|lite|tiny|flash|8b|7b|3b|1b)\b", re.I)
_LARGE_HINTS = re.compile(r"\b(70b|72b|405b|k2|large|max|ultra|opus)\b", re.I)


class FireworksResult:
    def __init__(self, text: str, tokens: int, model: str):
        self.text = text
        self.tokens = tokens  # prompt + completion tokens (COUNT toward score)
        self.model = model


def _inferred_size(model_id: str) -> float:
    m = _SIZE_RE.search(model_id)
    if m:
        return float(m.group(1))
    if _SMALL_HINTS.search(model_id):
        return 8.0
    if _LARGE_HINTS.search(model_id):
        return 1000.0
    return 50.0


def rank_models() -> list[str]:
    """Ascending by inferred cost (smallest/cheapest first)."""
    return sorted(config.ALLOWED_MODELS, key=_inferred_size)


def choose_model(complex_task: bool) -> Optional[str]:
    models = rank_models()
    if not models:
        return None
    # Always select the most capable/powerful model available (e.g. Kimi / 405B / 70B)
    # whenever we fall back or escalate to Fireworks so we maximize accuracy.
    return models[-1]


def generate(prompt: str, category: str, complex_task: bool = False) -> FireworksResult:
    model = choose_model(complex_task)
    if not model:
        raise RuntimeError("No ALLOWED_MODELS available for Fireworks fallback.")

    url = config.FIREWORKS_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompts.system_for(category)},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": prompts.max_tokens_for(category),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=config.REQUEST_BUDGET_S) as r:
            out = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} for model {model}: {body}") from e

    choice = (out.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content", "") or "").strip()
    usage = out.get("usage") or {}
    tokens = int(usage.get("total_tokens", 0) or 0)
    return FireworksResult(text=text, tokens=tokens, model=model)
