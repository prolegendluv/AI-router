"""Runtime configuration.

All Fireworks values are read purely from the environment (the harness injects
them at evaluation time). Local-model settings are baked into the image.
"""
from __future__ import annotations

import glob
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# ---- Fireworks (safety net) ----
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "").strip()
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "").strip()
ALLOWED_MODELS = [
    m.strip()
    for m in os.environ.get("ALLOWED_MODELS", "").split(",")
    if m.strip()
]

# escalate: use Fireworks only when the local model is low-confidence (default).
# never:    pure-local, guarantees zero Fireworks tokens.
# always:   route everything to Fireworks (debug only).
FIREWORKS_MODE = os.environ.get("FIREWORKS_MODE", "escalate").strip().lower()
CONFIDENCE_THRESHOLD = _float("CONFIDENCE_THRESHOLD", 0.45)

# Categories where a small local model is unreliable (confident-wrong answers
# our heuristics can't catch), so route them straight to the cheapest Fireworks
# model to protect the accuracy gate. Only fires when Fireworks is enabled, so
# pure-local testing (FIREWORKS_MODE=never) still runs everything locally.
FORCE_FIREWORKS_CATEGORIES = {
    c.strip() for c in os.environ.get(
        "FORCE_FIREWORKS_CATEGORIES",
        "mathematical_reasoning,logical_reasoning",
    ).split(",") if c.strip()
}

# ---- Local model (served by llama-server) ----
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models").strip()
MODEL_FILE = os.environ.get("MODEL_FILE", "").strip()
# On the grading VM (2 vCPU / 4GB RAM), auto-threading (0) spawns 8-16 host threads,
# causing severe CPU context thrashing and dropping inference speed to 2 tokens/sec.
# Setting LLAMA_THREADS=2 locks exactly 1 thread per vCPU for maximum speed.
LLAMA_THREADS = _int("LLAMA_THREADS", 2)
LLAMA_CTX = _int("LLAMA_CTX", 2048)
N_GPU_LAYERS = _int("N_GPU_LAYERS", 999)          # offload all if a GPU exists
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server").strip()
LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1").strip()
LLAMA_PORT = _int("LLAMA_PORT", 8080)
# Gemma 4 is a reasoning model. We run with the model's own chat template
# (--jinja) and a zero reasoning budget so it answers directly instead of
# burning the whole token budget (and the 30s limit) on a <think> block.
USE_JINJA = os.environ.get("USE_JINJA", "1").strip() not in ("0", "false", "False", "")
REASONING_BUDGET = _int("REASONING_BUDGET", 0)
# How long to wait for the server to load the model and report healthy.
SERVER_START_TIMEOUT_S = _float("SERVER_START_TIMEOUT_S", 50.0)

# I/O paths (spec-mandated).
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json").strip()
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json").strip()

# Global wall-clock budget (seconds). Overall watchdog is 9m30s (570s).
TOTAL_BUDGET_S = _float("TOTAL_BUDGET_S", 565.0)
# Per-request soft budget (remote call limit is 25s, hard timeout is 28s/30s).
REQUEST_BUDGET_S = _float("REQUEST_BUDGET_S", 25.0)
# Local model generation timeout. If local generation takes >22s on 2 vCPUs,
# abort and fail over to Fireworks immediately so Fireworks still has 5s+ left
# before hitting the 28s per-task hard timeout.
LOCAL_TIMEOUT_S = _float("LOCAL_TIMEOUT_S", 22.0)


def resolve_gguf() -> str | None:
    """Locate the bundled GGUF. Explicit MODEL_FILE wins, else auto-discover."""
    if MODEL_FILE:
        cand = MODEL_FILE if os.path.isabs(MODEL_FILE) else os.path.join(MODEL_DIR, MODEL_FILE)
        return cand if os.path.exists(cand) else None
    matches = sorted(glob.glob(os.path.join(MODEL_DIR, "*.gguf")))
    return matches[0] if matches else None


def local_base_url() -> str:
    return f"http://{LLAMA_HOST}:{LLAMA_PORT}"


def fireworks_enabled() -> bool:
    return (
        FIREWORKS_MODE != "never"
        and bool(FIREWORKS_API_KEY)
        and bool(FIREWORKS_BASE_URL)
        and bool(ALLOWED_MODELS)
    )
