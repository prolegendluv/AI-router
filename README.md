# AMD Developer Hackathon: Track 1 - General-Purpose AI Agent
## Hybrid Token-Efficient Routing Agent (E4B Local + Minimax 3 Fireworks API)

This repository contains a production-grade **Hybrid Token-Efficient Routing Agent** built for **Track 1** of the AMD Developer Hackathon.

---

## 🎯 Architecture & Strategy

The hackathon scoring mechanism is twofold:
1. **Accuracy Gate**: Evaluated by an LLM-Judge across 8 capability categories. Submissions below threshold are disqualified.
2. **Token Efficiency**: Ranked ascending by total Fireworks tokens recorded by the judging proxy. **Fewer tokens = higher rank.**

💡 **The Winning Strategy (`E4B(local)` + `Minimax 3 (Fireworks API)`)**:
Since **local inference uses zero Fireworks tokens** (`0 cost towards ranking`), our routing layer dynamically classifies every query and calculates its complexity:
- **`E4B (Local Model)` [0 Fireworks Tokens]**: Handled entirely locally inside the container for structured, simpler, or domain-specific tasks where a local SLM comfortably passes the accuracy gate (e.g., Sentiment Classification, Named Entity Recognition, basic Summarization, direct Factual Knowledge).
- **`Minimax 3` / Fireworks API Models**: Reserved exclusively for high-complexity tasks where maximum reasoning depth is mandatory to pass the accuracy gate (e.g., Multi-step Mathematical Reasoning, Constraint-based Logical/Deductive Puzzles, and Complex Code Generation/Debugging).

---

## 🧠 Capability Categories & Routing Logic

| # | Category | Description | Primary Route | Why? |
|---|---|---|---|---|
| 1 | **Factual Knowledge** | Concepts, definitions, how things work | `E4B (Local)` or `Fireworks (Small)` | Local/cheapest model for direct facts; complex nuanced explanations to Minimax 3 |
| 2 | **Mathematical Reasoning** | Arithmetic, percentages, word problems | `Minimax 3 (Fireworks)` | Multi-step calculation requires high precision |
| 3 | **Sentiment Classification** | Labelling sentiment and justifying | `E4B (Local)` | Near 100% accuracy on local SLM using **0 tokens** |
| 4 | **Text Summarisation** | Condensing passages with constraints | `E4B (Local)` | Excellent local performance for length & formatting constraints |
| 5 | **Named Entity Recognition** | Extracting & labelling (person, org, etc.) | `E4B (Local)` | Exact entity extraction works locally with **0 tokens** |
| 6 | **Code Debugging** | Identifying bugs & fixing implementations | `Minimax 3 (Fireworks)` | Code syntax & subtle logic bugs need strong code models |
| 7 | **Logical / Deductive Reasoning** | Constraint-based puzzles | `Minimax 3 (Fireworks)` | Highest difficulty; requires powerful reasoning capabilities |
| 8 | **Code Generation** | Writing well-structured functions from spec | `Minimax 3 (Fireworks)` | Ensures clean syntax and boundary handling |

---

## 🛠️ Project Structure

```
amd_hackathon_track1/
├── Dockerfile                  # Multi-stage Docker build (--platform linux/amd64)
├── requirements.txt            # Python dependencies
├── main.py                     # Container entrypoint (/input/tasks.json -> /output/results.json)
├── router.py                   # Zero-token local task classifier & complexity estimator
├── local_engine.py             # E4B (Local inference handler / zero Fireworks tokens)
├── fireworks_client.py         # Compliant Fireworks API client (uses FIREWORKS_BASE_URL)
├── evaluator.py                # Local evaluation harness (checks accuracy + token savings)
├── eval_dataset.json           # Benchmark tasks across all 8 categories
└── test_run.py                 # Quick end-to-end verification script
```

---

## ⚙️ Environment Variables (Evaluated at Runtime)

As required by the harness, our agent reads all configuration from environment variables without hardcoding:
- `FIREWORKS_API_KEY`: API key injected by harness.
- `FIREWORKS_BASE_URL`: Base URL for all API calls (all API requests are strictly routed here).
- `ALLOWED_MODELS`: Comma-separated list of permitted model IDs published on launch day (e.g., `accounts/fireworks/models/minimax-01,accounts/fireworks/models/llama-v3p1-8b-instruct`).

---

## 🚀 Running Locally & Evaluation

### 1. Run Local Evaluation Harness
```bash
python evaluator.py
```
This tests the agent against the 8 benchmark categories, displays latency, outputs classification accuracy, and tracks exact Fireworks token consumption.

### 2. Test Container Entrypoint (`main.py`)
```bash
# Prepare sample input
mkdir -p /input /output
python test_run.py
```

### 3. Build & Test Docker Container (`linux/amd64`)
```bash
docker buildx build --platform linux/amd64 -t amd-hackathon-track1:latest .
docker run --rm \
  -e FIREWORKS_API_KEY="your_key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="accounts/fireworks/models/minimax-01,accounts/fireworks/models/llama-v3p1-8b-instruct" \
  -v $(pwd)/sample_input:/input \
  -v $(pwd)/sample_output:/output \
  amd-hackathon-track1:latest
```
