"""Local Gemma 4 E4B inference via `llama-server` (built from llama.cpp source).

We run the model through llama.cpp's own server rather than the Python binding,
so support for the gemma4 architecture depends only on the llama.cpp build in
the image (recent) — not on whether llama-cpp-python has caught up. Python talks
to the server over its local OpenAI-compatible HTTP endpoint using only the
standard library. Local tokens are free and never count toward the score.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import config, prompts

_PROC = None            # llama-server subprocess
_SERVER_READY = False
_SERVER_FAILED = False   # set once startup fails so we stop retrying (fast fallback)

_LOW_CONF_MARKERS = re.compile(
    r"\b(i'?m not sure|i am not sure|i don'?t know|i cannot|i can'?t (help|"
    r"answer|determine)|as an ai|unable to|no information|not enough "
    r"information|it depends|unclear)\b",
    re.I,
)


class LocalResult:
    def __init__(self, text: str, confidence: float, tokens: int):
        self.text = text
        self.confidence = confidence
        self.tokens = tokens  # local tokens (FREE)


def is_ready() -> bool:
    if _SERVER_FAILED:
        return False
    try:
        return config.resolve_gguf() is not None
    except Exception:
        return False


def _http_get(path: str, timeout: float):
    url = config.local_base_url() + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def _http_post(path: str, payload: dict, timeout: float):
    url = config.local_base_url() + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _health_ok() -> bool:
    try:
        status, _ = _http_get("/health", timeout=2.0)
        return status == 200
    except Exception:
        return False


def start_server() -> bool:
    """Launch llama-server and wait until the model is loaded and healthy.

    On any failure the server is marked permanently failed so callers fall back
    to Fireworks immediately instead of re-waiting the startup timeout per task.
    """
    global _PROC, _SERVER_READY, _SERVER_FAILED
    if _SERVER_FAILED:
        raise RuntimeError("llama-server previously failed to start.")
    if _SERVER_READY and _health_ok():
        return True
    if _health_ok():  # a server is already running (e.g. provided externally)
        _SERVER_READY = True
        return True

    try:
        gguf = config.resolve_gguf()
        if not gguf:
            raise FileNotFoundError(
                f"No GGUF found. Put a *.gguf in {config.MODEL_DIR} or set MODEL_FILE."
            )

        cmd = [
            config.LLAMA_SERVER_BIN,
            "-m", gguf,
            "--host", config.LLAMA_HOST,
            "--port", str(config.LLAMA_PORT),
            "-c", str(config.LLAMA_CTX),
            "-ngl", str(config.N_GPU_LAYERS),
            "-b", "512",
            "-ub", "512",
            "-fa", "on",
            "--no-webui",
        ]
        # Use the model's own chat template and give reasoning a zero budget so
        # Gemma 4 answers directly instead of exhausting the token cap thinking.
        if config.USE_JINJA:
            cmd += ["--jinja", "--reasoning-budget", str(config.REASONING_BUDGET)]
        if config.LLAMA_THREADS:
            cmd += ["-t", str(config.LLAMA_THREADS)]

        print(f"[llama-server] starting: {' '.join(cmd)}", file=sys.stderr)
        _PROC = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=sys.stderr)

        deadline = time.time() + config.SERVER_START_TIMEOUT_S
        while time.time() < deadline:
            if _PROC.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {_PROC.returncode}). "
                    "Check the model architecture is supported by the build."
                )
            if _health_ok():
                _SERVER_READY = True
                print("[llama-server] ready", file=sys.stderr)
                return True
            time.sleep(1.0)
        raise TimeoutError("llama-server did not become healthy in time.")
    except Exception:
        _SERVER_FAILED = True
        shutdown()
        raise


def shutdown() -> None:
    global _PROC
    if _PROC and _PROC.poll() is None:
        try:
            _PROC.terminate()
            _PROC.wait(timeout=5)
        except Exception:
            try:
                _PROC.kill()
            except Exception:
                pass
    _PROC = None


def _confidence(choice: dict, text: str) -> float:
    conf = None
    try:
        lp = choice.get("logprobs") or {}
        toks = lp.get("content") or []
        vals = [t.get("logprob") for t in toks if t.get("logprob") is not None]
        if vals:
            conf = math.exp(sum(vals) / len(vals))
    except Exception:
        conf = None
    if conf is None:
        conf = 0.8
    stripped = (text or "").strip()
    if not stripped:
        return 0.0
    if len(stripped) < 3:
        conf = min(conf, 0.2)
    if _LOW_CONF_MARKERS.search(stripped):
        conf = min(conf, 0.35)
    return max(0.0, min(1.0, conf))


def generate(prompt: str, category: str) -> LocalResult:
    start_server()
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": prompts.system_for(category)},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": prompts.max_tokens_for(category),
        "logprobs": True,
        "top_logprobs": 1,
        "cache_prompt": True,
    }
    out = _http_post("/v1/chat/completions", payload,
                     timeout=config.LOCAL_TIMEOUT_S)
    choice = (out.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    # Prefer the final answer; fall back to reasoning so we never emit empty
    # when the model spent its budget in a <think> block.
    text = content or reasoning
    print(f"[gen] cat={category} finish={choice.get('finish_reason')} "
          f"content_len={len(content)} reasoning_len={len(reasoning)}",
          file=sys.stderr)
    usage = out.get("usage") or {}
    tokens = int(usage.get("completion_tokens", 0) or 0)
    return LocalResult(text=text, confidence=_confidence(choice, text),
                       tokens=tokens)
