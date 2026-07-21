# AMD Hackathon Track 1 — Handoff / Status

## What this is
A token-efficient routing agent. Local model answers what it can (0 Fireworks
tokens); a few categories route to Fireworks. Ranking = fewest Fireworks tokens
among submissions that pass an accuracy gate.

Reads `/input/tasks.json`, writes `/output/results.json` (`[{task_id, answer}]`).

## Architecture (all in `src/`)
- `main.py` — entrypoint; reads tasks, warms model, routes each, writes results.
  Crash-safe: writes a complete results.json up front, flushes after each task,
  always exits 0.
- `local_model.py` — starts `llama-server` (built from source in the image) and
  calls it over local HTTP. Runs with `--jinja --reasoning-budget 0`.
- `fireworks_client.py` — OpenAI-compatible calls via stdlib urllib (NO openai
  SDK — that caused a `proxies` crash). Picks cheapest model from ALLOWED_MODELS.
- `router.py` — local-first; escalates on freshness, on forced categories, or on
  low local confidence / bad format score.
- `classifier.py` — zero-cost category + freshness + complexity detection.
- `prompts.py` — per-category system prompts + token caps.
- `cleaner.py` — strips think/preamble/fences, enforces summary counts, extracts
  math `Answer:` line.
- `verifier.py` — format-compliance score that can trigger escalation.

## Current model
`Qwen2.5-3B-Instruct-Q6_K.gguf` in `models/` (non-reasoning, ~2.5GB). Any single
`*.gguf` there is auto-discovered. Swap freely.

## Verified working (local test, `FIREWORKS_MODE=never`)
Language categories are clean and correct at 0 tokens: factual (T01b/T01c),
sentiment (T03/T03b → Mixed + both sides), summarization (T04/T04b), NER (T05).
Run ~40s for 10 tasks (~18 tok/s CPU, no GPU in container).

Escalate test (`FIREWORKS_MODE=escalate` + real key + Kimi): routing confirmed —
math → Fireworks (`kimi-k2p7-code`), everything else local (0 tokens). Total Fireworks tokens: **719** (down from 844).
- `T02` clean output: `Answer: 1,672 units` (313 tokens)
- `T02b` clean output: `Answer: 1.875 cups of sugar; total cost $4.50` (406 tokens)

## Just-fixed & Verified (FINAL BUILD PUSHED)
The Kimi math answers previously rambled due to prompt conflicts (`BASE` instruction vs math instruction vs single/multi-quantity `Answer:` format). Fixed and verified:
- `prompts.py`: `BASE` no longer forbids showing work or causes instruction arguments. `mathematical_reasoning` prompt explicitly requests ending with a line starting with `'Answer: '` containing numeric result(s) and unit(s). Max tokens set to 350.
- `cleaner.py`: Robust regex extracts `Answer:` / `**Answer:**` / `### Answer:` / `Final Answer:` line and strips trailing markdown/bold wrappers.
- Removed non-ASCII em-dashes from prompts to prevent mojibake.
- Docker image built for `linux/amd64` and pushed to Docker Hub: `uvlb/amd-router:latest` (digest `sha256:f7bece3534ba463186c91031fed82050986177a39089e63a794223bc92faa2ed`).

## Known weak spots / risks
- Local 3B is weak on math (that's WHY math+logic force-route to Fireworks via
  `FORCE_FIREWORKS_CATEGORIES`). Confident-wrong local answers won't self-escalate.
- T01 (RGB "why") and T04b (missing org-response bullet) are borderline on the
  judge — a 7B local model or escalating factual would help if the gate is tight.
- Container has NO GPU → CPU speed. Fine for ~10-30 tasks; a very large hidden
  set + a big model could approach the 10-min total (budget guard emits empties).

## Config knobs (env; harness injects the Fireworks ones)
- `FIREWORKS_MODE` = escalate (submission) | never (pure-local test) | always
- `FORCE_FIREWORKS_CATEGORIES` = "mathematical_reasoning,logical_reasoning"
  (set "" for pure-local zero-token gamble; add code_* for more gate safety)
- `CONFIDENCE_THRESHOLD` (0.45), `REQUEST_BUDGET_S` (28), `LLAMA_CTX` (4096),
  `N_GPU_LAYERS` (999), `MODEL_FILE`, `TOTAL_BUDGET_S` (540)

## Commands
Build (linux/amd64, compiles llama.cpp — cached after first build):
```
docker buildx build --platform linux/amd64 -t uvlb/amd-router:latest --load .
```
Pure-local test (0 tokens):
```
.\scripts\run_local.ps1 uvlb/amd-router:latest
```
Escalate test (real key + model; key never written to disk):
```
.\scripts\run_local_fw.ps1 -ImageTag uvlb/amd-router:latest -ApiKey "fw_..." -Models "accounts/fireworks/models/kimi-k2p7-code"
```
Push + submit:
```
docker push uvlb/amd-router:latest
```
Then make the Docker Hub repo PUBLIC and submit the image tag (`uvlb/amd-router:latest`).

## Next steps (User Action Required)
1. **Submit**: Ensure `uvlb/amd-router:latest` repository is set to PUBLIC on Docker Hub, then submit `uvlb/amd-router:latest` in `escalate` mode.
2. **Rotate API Key**: Immediately rotate the Fireworks API key (`fw_MNn...`) on the Fireworks dashboard since it was used during testing and appears in local chat/transcript history.
3. **Optional Tuning**: If the accuracy gate turns out tight on evaluation, consider escalating `code_debugging` / `code_generation` via `FORCE_FIREWORKS_CATEGORIES`.

## Compliance (met)
Reads env for FIREWORKS_API_KEY/BASE_URL/ALLOWED_MODELS; all Fireworks calls via
FIREWORKS_BASE_URL; no hardcoded model IDs; no .env in image; exit 0; valid JSON
per task; linux/amd64; image <10GB; no runtime downloads.
