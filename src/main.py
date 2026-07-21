"""Container entrypoint for Track 1.

Reads /input/tasks.json, routes each task (local-first, Fireworks safety net),
and writes /output/results.json as a list of {task_id, answer}.

Crash-safety (avoids RUNTIME_ERROR / OUTPUT_MISSING):
  * A complete results.json (one empty answer per task) is written BEFORE any
    model work, so a valid file always exists even if the model backend dies.
  * Results are flushed to disk after every task, so an OOM/timeout mid-run
    still leaves a valid, partially-filled file.
  * Every heavy import is lazy and every task is wrapped, so nothing propagates.
  * A top-level guard catches anything unexpected and still exits 0.
"""
from __future__ import annotations

import json
import os
import sys
import time


def _load_tasks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks") or data.get("data") or []
    tasks = []
    for i, t in enumerate(data or []):
        if not isinstance(t, dict):
            continue
        tid = t.get("task_id", t.get("id", f"t{i+1}"))
        prompt = t.get("prompt", t.get("input", "")) or ""
        tasks.append({"task_id": tid, "prompt": prompt})
    return tasks


def _write_results(path: str, results: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, path)


def _run() -> int:
    # Imports are inside _run so an import-time failure is caught by the guard
    # AFTER a best-effort empty results.json has been written.
    from . import config, local_model, router

    start = time.time()
    print(f"[boot] fireworks_mode={config.FIREWORKS_MODE} "
          f"fireworks_enabled={config.fireworks_enabled()} "
          f"local_ready={local_model.is_ready()} "
          f"gguf={config.resolve_gguf()}", file=sys.stderr)

    out_path = config.OUTPUT_PATH
    try:
        tasks = _load_tasks(config.INPUT_PATH)
    except Exception as e:
        print(f"[fatal] could not read {config.INPUT_PATH}: {e}", file=sys.stderr)
        _write_results(out_path, [])
        return 0

    # Pre-fill and flush a complete, valid file up front.
    results = [{"task_id": t["task_id"], "answer": ""} for t in tasks]
    _write_results(out_path, results)

    # Warm up the model once (failure is non-fatal — router falls back).
    try:
        if local_model.is_ready():
            local_model.generate("Reply with: ok", "factual_knowledge")
    except Exception as e:
        print(f"[warmup] local model not ready: {e}", file=sys.stderr)

    log: list[dict] = []
    total_fw_tokens = 0

    for idx, task in enumerate(tasks):
        tid, prompt = task["task_id"], task["prompt"]
        t0 = time.time()
        if time.time() - start > config.TOTAL_BUDGET_S:
            log.append({"task_id": tid, "route": "skipped_budget"})
            continue
        try:
            d = router.route(prompt)
            answer = d.answer if isinstance(d.answer, str) else str(d.answer)
            results[idx]["answer"] = answer
            total_fw_tokens += d.fireworks_tokens
            log.append({
                "task_id": tid, "route": d.route, "category": d.category,
                "model": d.model, "fireworks_tokens": d.fireworks_tokens,
                "seconds": round(time.time() - t0, 2), "note": d.note,
            })
        except Exception as e:
            log.append({"task_id": tid, "route": "error", "note": str(e)})
        # Flush after every task so partial progress is always on disk.
        try:
            _write_results(out_path, results)
        except Exception as e:
            print(f"[warn] flush failed: {e}", file=sys.stderr)
        print(f"[task {idx+1}/{len(tasks)}] {tid} -> {log[-1].get('route')} "
              f"fw_tokens={log[-1].get('fireworks_tokens', 0)}", file=sys.stderr)

    _write_results(out_path, results)
    try:
        log_path = os.path.join(os.path.dirname(out_path) or ".",
                                "inference_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"total_fireworks_tokens": total_fw_tokens,
                       "tasks": log}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"[done] tasks={len(tasks)} total_fireworks_tokens={total_fw_tokens} "
          f"elapsed={round(time.time()-start,1)}s", file=sys.stderr)
    return 0


def main() -> int:
    try:
        return _run()
    except Exception as e:
        # Last-resort guard: never crash the container. Try to leave a valid
        # (possibly empty) results.json so the run is scored, not errored.
        import traceback
        traceback.print_exc()
        try:
            out_path = os.environ.get("OUTPUT_PATH", "/output/results.json")
            if not os.path.exists(out_path):
                _write_results(out_path, [])
        except Exception:
            pass
        print(f"[fatal] {e}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
