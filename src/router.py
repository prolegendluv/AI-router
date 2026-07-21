"""Local-first router with a Fireworks safety net.

Strategy (ranking is by Fireworks tokens ascending, gated by an accuracy floor):
  1. Static, in-scope tasks  -> answer locally with Gemma E4B (0 Fireworks tokens).
  2. Fresh-knowledge tasks    -> go straight to Fireworks (local can't know these).
  3. Low local confidence OR
     bad output format         -> escalate that single task to the cheapest
                                  sufficient Fireworks model.

Every answer (local or remote) is cleaned (strip <think>/preambles/fences) and
format-verified before it is returned, so we don't lose points to chatter or a
wrong sentence/bullet count.
"""
from __future__ import annotations

from . import (classifier, cleaner, config, fireworks_client, local_model,
               verifier)

# Escalate when the verifier score is below this (format looks wrong).
_FORMAT_FLOOR = 0.6


class Decision:
    def __init__(self, answer, route, category, fireworks_tokens, model, note=""):
        self.answer = answer
        self.route = route                    # "local" | "fireworks" | "fallback"
        self.category = category
        self.fireworks_tokens = fireworks_tokens
        self.model = model
        self.note = note


def _finalize(text: str, category: str, prompt: str) -> str:
    return cleaner.clean(text, category, prompt)


def _local_available() -> bool:
    try:
        return local_model.is_ready()
    except Exception:
        return False


def route(prompt: str) -> Decision:
    info = classifier.analyze(prompt)
    category = info["category"]
    fresh = info["freshness"]
    complex_task = info["complexity"]
    fw_ok = config.fireworks_enabled()
    local_ok = _local_available()

    if config.FIREWORKS_MODE == "always" and fw_ok:
        return _fireworks(prompt, category, complex_task, "mode=always")

    if fresh and fw_ok:
        return _fireworks(prompt, category, complex_task, "freshness")

    # Categories the local model is unreliable at -> cheapest Fireworks model.
    if category in config.FORCE_FIREWORKS_CATEGORIES and fw_ok:
        return _fireworks(prompt, category, complex_task, "forced_category")

    if local_ok:
        try:
            res = local_model.generate(prompt, category)
        except Exception as e:
            if fw_ok:
                return _fireworks(prompt, category, complex_task, f"local_error:{e}")
            return Decision("", "local", category, 0, "local", f"local_error:{e}")

        answer = _finalize(res.text, category, prompt)
        fmt = verifier.score(answer, category, prompt)
        needs_help = (
            res.confidence < config.CONFIDENCE_THRESHOLD
            or fmt < _FORMAT_FLOOR
            or (complex_task and res.confidence < 0.7)
        )
        if needs_help and fw_ok:
            fw = _fireworks(prompt, category, complex_task,
                            f"escalate(conf={res.confidence:.2f},fmt={fmt:.2f})")
            # Keep the local answer if Fireworks returned nothing usable.
            if not fw.answer and answer:
                return Decision(answer, "local", category, fw.fireworks_tokens,
                                fw.model, "escalate_empty_fw")
            return fw
        return Decision(answer, "local", category, 0, "local",
                        f"conf={res.confidence:.2f},fmt={fmt:.2f}")

    if fw_ok:
        return _fireworks(prompt, category, complex_task, "no_local")
    return Decision("", "fallback", category, 0, "none", "no_backend")


def _fireworks(prompt, category, complex_task, note) -> Decision:
    try:
        fw = fireworks_client.generate(prompt, category, complex_task)
        answer = _finalize(fw.text, category, prompt)
        return Decision(answer, "fireworks", category, fw.tokens, fw.model, note)
    except Exception as e:
        return Decision("", "fireworks", category, 0, "error", f"fw_error:{e}")
