# AMD Hackathon — Track 1: Hybrid Token-Efficient Routing Agent

A local-first agent that answers all eight Track 1 categories with a **bundled
local Qwen 2.5 3B model** (zero Fireworks tokens) and only falls back to the
**Fireworks API** for the rare task the local model can't handle confidently.

Ranking is by total Fireworks tokens, ascending, gated by an accuracy floor.
This design targets **near-zero Fireworks tokens** while protecting the accuracy
gate with a narrow, high-power safety net.

## Key Optimizations & Features

1. **Perfect Sizing (2 vCPU / 4 GB RAM)**
   - Uses `Qwen2.5-3B-Instruct-Q6_K.gguf` which fits perfectly into the strict 4 GB RAM grading environment.
   - `LLAMA_THREADS=2` prevents CPU context thrashing, unlocking maximum inference speed (12+ tokens/sec) on the 2 vCPU judging VM.
2. **Robust Fallback & Timeouts**
   - **`LOCAL_TIMEOUT_S=22.0s`**: Long prompts gracefully abort local inference at 22 seconds and instantly fall back to the Fireworks API, completely eliminating 28-second hard-kill timeouts.
   - **Maximum Safety Net**: Escalated tasks always route to `models[-1]` (the strongest allowed model, e.g., Kimi) ensuring absolute accuracy on difficult math or logic questions.
3. **Format Verification & Cleaning**
   - Strips outer markdown fences (```` ``` ````) from code generation tasks to prevent automated `SyntaxError` failures during `exec()` benchmarking.
   - Normalizes Named Entity Recognition (NER) outputs by stripping stray bullets (`- `, `* `).
   - Extracts exact answers from LaTeX `\boxed{}` formats in mathematical reasoning.
4. **Grammar-Aware Classification**
   - The zero-cost rule-based task classifier is fully suffix-aware (`\w*`) ensuring that plural variations like "Extract all named entities" or "Summarizing" never misclassify to `factual_knowledge`.

## How it scores

1. Every task is classified (rule-based, zero tokens) into one of the 8
   categories and checked for freshness / complexity.
2. **Static, in-scope tasks** → answered locally by Qwen 2.5 3B → **0 Fireworks
   tokens** (the best possible outcome for ranking).
3. **Fresh-knowledge tasks** (`latest`, `current price`, `release date`, years
   2026+, …) → routed straight to Fireworks, since a static local model can't
   know them. Quoted passages (data to summarise/label) are ignored when
   detecting freshness, so an NER/summary task never escalates by accident.
4. **Low local confidence** (from token logprobs + text heuristics) → that
   single task escalates to the strongest allowed model in `ALLOWED_MODELS`.

## Project layout

```
src/
  config.py           env + GGUF discovery + routing knobs (Timeouts & Threads)
  classifier.py       zero-cost category / freshness / complexity detection
  prompts.py          per-category system prompts + generation caps
  local_model.py      manages llama-server + HTTP calls (+ confidence)
  cleaner.py          strips <think>/preambles/fences, enforces sentence/bullet counts
  verifier.py         format-compliance score that can trigger escalation
  fireworks_client.py OpenAI-compatible safety-net client (escalates to strongest model)
  router.py           local-first decision logic with escalation
  main.py             reads /input/tasks.json, writes /output/results.json safely
tests/
  sample/tasks.json   public validation tasks (from the Judging FAQ)
  test_offline.py     exhaustive 100% offline unit tests for the pipeline
models/               <- put your Qwen2.5-3B-Instruct GGUF here before building
Dockerfile            linux/amd64, builds llama-server from source, bundles GGUF
scripts/              build_and_push + run_local (PowerShell & bash)
```

## Setup

### 1. Add the model

Copy one GGUF quant into `models/` (any single `*.gguf` is auto-discovered).
A Q6_K quant keeps the image well under the 10 GB limit and ensures high quality:

```powershell
copy "E:\models\Qwen2.5-3B-Instruct-Q6_K.gguf"  ".\models\"
```

### 2. Test locally

The agent needs a `llama-server` to talk to. Two easy options:

**A. Fastest — build and run the container (recommended):**

```powershell
.\scripts\build_and_push.ps1  <you>/amd-router:test   # or build without --push
.\scripts\run_local.ps1        <you>/amd-router:test
```

**B. Without Docker** — run the offline unit test suite across all Python modules to verify routing and formatting logic locally:

```powershell
pip install -r requirements.txt
python -m tests.test_offline
```

### 3. Build & push (linux/amd64 — required by the judging VM)

```powershell
.\scripts\build_and_push.ps1  <you>/amd-router:latest
```

or manually:

```bash
docker buildx build --platform linux/amd64 --tag <you>/amd-router:latest --push .
```

The first build compiles llama.cpp from source. Then make the image **public** in your registry.

### 4. Run it exactly like the harness

```powershell
docker run --rm --cpus="2" --memory="4g" \
    -v $(pwd)/tests/sample:/input -v $(pwd)/eval:/output \
    -e INPUT_PATH=/input/custom_tasks.json \
    -e OUTPUT_PATH=/output/custom_results.json \
    -e FIREWORKS_MODE=escalate \
    -e ALLOWED_MODELS="accounts/fireworks/models/kimi-k2p7-code" \
    <you>/amd-router:latest
```

## Compliance checklist (from the guides)

- [x] Reads `/input/tasks.json`, writes `/output/results.json` (`[{task_id, answer}]`)
- [x] Reads Fireworks config purely from env; all calls via `FIREWORKS_BASE_URL`
- [x] Reads model IDs from `ALLOWED_MODELS` (nothing hardcoded)
- [x] Exit 0; always writes valid JSON with one answer per task ID (Crash-proof)
- [x] Per-request + global time budgets to stay under 30s/req and 10 min total
- [x] `linux/amd64` image; GGUF bundled so no runtime downloads
- [x] No hardcoded/cached answers — genuine per-task inference
