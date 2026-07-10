# BhaluRouter v3.0: Hybrid E4B Capability-Based Routing Engine

This document provides the complete architectural breakdown and technical specification of the **BhaluRouter v3.0** implementation developed for **AMD Hackathon Track 1 (`Hybrid E4B Engine`)**.

---

## 1. System Architecture Overview

Instead of classifying user queries purely by semantic topic and routing everything to remote cloud endpoints, `BhaluRouter` implements a **multi-stage capability estimation pipeline**. It maximizes on-device / local AI inference using **Local Gemma 4B (`google/gemma-4-e4b`)** and deterministic rule engines (`0 cloud tokens`, `$0.00 cost`), while dynamically escalating only complex, multi-step reasoning tasks to specialized remote cloud models (**Fireworks API `minimax-m3`**).

```mermaid
flowchart TD
    Prompt[User Prompt / Query] --> Stage1[Stage 1: Deterministic Engine\n0 ms | 0 Tokens]
    
    Stage1 -- "Arithmetic, Unit Conversion,\nExact Formulas, JSON Check" --> DetAns[⚡ Deterministic Answer\n0 Tokens | Cost: $0.00]
    Stage1 -- "Unresolved" --> Stage2[Stage 2: Capability & Domain Router\nML Structural Feature Analysis]
    
    Stage2 -- "Local Domain:\nFactual Q&A, Math Reasoning,\nSentiment, Summarization, NER" --> LocalGemma[🐻 Local Gemma 4B E4B\nLM Studio Port 1234\n0 Cloud Tokens | Cost: $0.00]
    
    Stage2 -- "Expert Cloud Domain:\nCode Debugging, Logic Puzzles,\nCode Generation" --> Stage3[Stage 3: Cloud Escalation Engine]
    
    Stage3 --> Fireworks[🔥 Fireworks API\nModel: minimax-m3\nCloud Inference | High-Tier Reasoning]
```

---

## 2. Multi-Stage Routing Pipeline (`router.py`)

### Tier 0 — Stage 1: Deterministic Rule Engine (`0 ms`, `0 tokens`)
Before invoking any LLM, `check_deterministic_layer(prompt)` evaluates whether the query can be resolved via exact computational algorithms:
- **Trivial Arithmetic (`python_math_ast`)**: Evaluates numeric expressions (`3+9`, `sqrt(144)`, algebraic calculations) using AST/regex parsing.
- **Unit & Temperature Conversions (`unit_converter`)**: Instant unit conversions (`miles <-> km`, `°C <-> °F`, `kg <-> lbs`).
- **Kinship & Family Relationships (`kinship_rule_engine`)**: Solves deductive family logic locally (`"who is the brother of my mother..."`).
- **JSON Syntax Validation (`json_validator`)**: Validates, parses, and pretty-prints JSON strings locally.
- **Encyclopedic & Scientific Lookups (`encyclopedic_ast`)**: Instant lookups for scientific constants and facts (e.g., speed of light, value of $\pi$, distance from Earth to Sun, world and Indian state capitals like Bihar $\rightarrow$ Patna).

### Tier 1 — Stage 2: Local E4B Capability & Domain Engine (`0 cloud tokens`)
When natural language generation is required, `estimate_local_capability(prompt, category)` checks the task's domain (`classify_task_category`) and structural complexity features (word count, reasoning depth, code complexity markers):
- **Strictly Routed to Local Gemma (`google/gemma-4-e4b` @ Port `1234`)**:
  - `factual_knowledge` (General knowledge Q&A, historical explanations, science)
  - `mathematical_reasoning` (Calculus explanations, word problems, algebraic steps)
  - `sentiment_classification` (Tone analysis, product reviews, emotional classification)
  - `text_summarisation` (Summarizing articles, comparative multi-paragraph overviews)
  - `named_entity_recognition` (Extracting persons, organizations, locations)
- **Execution & Timeout Guarantees**:
  - Connects directly to **LM Studio (`http://127.0.0.1:1234/v1/chat/completions`)** with a **180-second timeout**, ensuring comprehensive local outputs complete without timing out or incurring cloud costs.
  - If the local inference server is temporarily busy or offline, our built-in zero-token synthesis engine (`_generate_fallback_local_synthesis`) guarantees an instant local response for these domains.

### Tier 2 — Stage 3: Remote Cloud Escalation Engine (`Fireworks API`)
When the prompt requires high-tier multi-step constraint solving or expert code synthesis (`is_capable = False`), `router.py` cleanly escalates the task to the cloud:
- **Strictly Routed to Fireworks (`accounts/fireworks/models/minimax-m3`)**:
  - `code_debugging` (Traceback diagnosis, memory leaks, concurrency race conditions, subtle logic bugs)
  - `logical_deductive_reasoning` (Knights and knaves puzzles, multi-step deductive constraints, Sudoku grids)
  - `code_generation` (Implementing complex data structures, full-stack Next.js/React SaaS architectures, LRU caches)

---

## 3. Core System Modules & File Structure

```text
C:\Users\Admin\.gemini\antigravity\scratch\amd_hackathon_track1\
│
├── router.py               # Core CapabilityRouter & ML TF-IDF/Structural feature classifiers
├── local_engine.py         # LocalEngine managing LM Studio (Port 1234) & Ollama connections + deterministic solvers
├── fireworks_client.py     # Fireworks API HTTP client (minimax-m3) with token & cost tracking
├── backend_server.py       # Asynchronous API Server running on Port 8000 (/api/route & /api/health)
│
└── frontend/               # Interactive Web Dashboard (Running on Port 3000)
    ├── package.json
    └── src/
        └── components/
            └── DashboardView.tsx   # Live routing Decision Cards, badges, latency charts & history table
```

---

## 4. API Specification (`backend_server.py`)

### `POST /api/route`
Accepts a user prompt and optional category override, runs the multi-stage capability router, and executes the target engine.

#### Request Body
```json
{
  "prompt": "summarize the benefits of modern Zen 4 architectures",
  "category": "Text Summarisation"
}
```

#### Response Body (`Stage 2: Local Gemma Example`)
```json
{
  "route": "local_llm",
  "model": "google/gemma-4-e4b",
  "confidence": 0.98,
  "complexity": "medium",
  "tokens": 0,
  "latency": 31451,
  "cost": 0.00,
  "reason": "Domain (`text_summarisation`) successfully verified for high-accuracy local execution via Gemma 4B E4B Engine (`0 tokens used`).",
  "token_savings": 242,
  "answer": "# Benefits of AMD's Zen 4 Architecture\n\nZen 4 is AMD's CPU architecture..."
}
```

#### Response Body (`Stage 3: Fireworks Cloud Example`)
```json
{
  "route": "remote_fireworks",
  "model": "accounts/fireworks/models/minimax-m3",
  "confidence": 0.99,
  "complexity": "expert",
  "tokens": 2384,
  "latency": 11485,
  "cost": 0.0238,
  "reason": "Domain (`code_generation`) mapped directly to Fireworks API (`minimax-m3`) for high-tier complex code debugging, logic puzzles, or code generation.",
  "token_savings": 0,
  "answer": "# Binary Search with Overflow Protection and Infinite Loop Detection..."
}
```

---

## 5. Frontend Dashboard (`DashboardView.tsx`)

The dashboard running on **Port `3000`** renders visual badges and telemetry for every routed task:
- ⚡ **`Deterministic Rule (0 tokens)`** — Green badge (`0 ms`, `$0.00`)
- 🐻 **`Local Gemma (0 tokens)`** — Teal/Cyan badge (`http://127.0.0.1:1234`, `$0.00`)
- 🔥 **`Fireworks Expert (minimax-m3)`** — Orange/Red badge (`Cloud API`, actual token cost shown)

It includes a live **Benchmark Execution Suite** that tests the routing rules in real-time and graphs latency and cumulative token savings across all queries.

---

## 6. Setup & Execution Commands

### 1. Launch Local Inference Server
Open **LM Studio**, load `google/gemma-4-e4b`, navigate to the `<->` (Server) tab, and click **Start Server** (`Port 1234`).

### 2. Launch BhaluRouter Backend API (`Port 8000`)
```powershell
cd C:\Users\Admin\.gemini\antigravity\scratch\amd_hackathon_track1
$env:BACKEND_PORT="8000"
$env:FIREWORKS_API_KEY="your_fireworks_api_key"
python backend_server.py
```

### 3. Launch Frontend Web Dashboard (`Port 3000`)
```powershell
cd C:\Users\Admin\.gemini\antigravity\scratch\amd_hackathon_track1\frontend
npm run dev -- -p 3000
```
