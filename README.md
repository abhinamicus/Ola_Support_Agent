# Ola Domain Support Agent Capstone

This repository completes the **Ola (Business Operations / Customer Support)** track. The dataset is deterministic with `seed=20260829`, category weights `{Billing: 0.24, Technical Issue: 0.22, Account Access: 0.18, Product Defect: 0.16, General Inquiry: 0.20}`, status weights `{Open: 0.22, In Progress: 0.24, Escalated: 0.12, Resolved: 0.24, Closed: 0.18}`, and `resolution_time_hours` sampled uniformly from `2.0` to `144.0` hours because that spans quick chat resolutions through multi-day escalations without making routine tickets unrealistically long.

`MOCK_LLM` is the default execution path and requires zero API keys. `CREWAI_DISABLE_TELEMETRY=true` must be set before any CrewAI execution; the implementation sets it in code and you should also keep it in your shell environment when running locally.

In `MOCK_LLM` mode, the system does not call an external generative model. Instead, it returns a bounded excerpt of the retrieved, approved knowledge-base context. This keeps the graded execution deterministic, local, and free of API keys.

The main CrewAI support flow performs retrieval, optional ticket lookup,
and response composition. The separate AutoGen review demonstration
shows bounded approval and revision behavior using two review agents.

## Repository Layout

- `dataset.py`: deterministic support ticket generator and validator.
- `knowledge_base/`: 12 policy documents covering all required support topics.
- `src/ola_support_agent/rag.py`: chunking, retrieval, threshold calibration, grounded generation, and cache.
- `src/ola_support_agent/tickets.py`: ticket lookup plus designed escalation score.
- `src/ola_support_agent/crew.py`: CrewAI orchestration, input guardrails, memory handling and budget cap.
- `src/ola_support_agent/review.py`: post-draft review verdict stage.
- `src/ola_support_agent/api.py`: FastAPI HTTP and WebSocket endpoints.
- `src/ola_support_agent/demo.py`: end-to-end evidence generator.
- `work/requests.jsonl`: structured request logs generated at runtime.
- `work/demo_results.json`: generated evidence bundle after the demo script runs.

## Run

```bash
python -m pip install -e .
set CREWAI_DISABLE_TELEMETRY=true
set PYTHONPATH=src
python dataset.py
python -m ola_support_agent.demo
uvicorn ola_support_agent.api:app --reload
```

## Part 1

`dataset.py` generates 48 records and validates:

- every given category appears at least 3 times
- every given status appears at least once
- `escalated=True` lands between 10% and 30%

The knowledge base contains 12 authored documents. The RAG layer implements fixed-size overlapping chunks and sentence chunks, then evaluates both strategies on a common query set. The grounded answer path refuses unsupported questions using an empirically calibrated threshold from in-scope and out-of-scope top-1 retrieval scores.

## Part 2

`check_support_ticket_status(record_id)` returns `status`, `resolution_time_hours`, and an escalation score:

```text
escalation_score = 0.6 * escalated_flag + 0.4 * (days_since_created / 30)
```

The recommended escalation threshold is based on the 80th percentile of `days_since_created` from the generated dataset:

```text
threshold = 0.6 + 0.4 * (p80_days_since_created / 30)
```

The orchestration layer includes:

- retrieval flow
- ticket lookup flow
- session memory
- structured Pydantic outputs
- phone-number masking
- prompt-injection blocking
- output-side grounded fallback

Least autonomy is enforced by design: only the lookup path calls `check_support_ticket_status`, and no retrieval or composition path is wired to invoke that tool directly.

## Part 3

FastAPI exposes:

- `POST /ask`
- `POST /add-document`
- `GET /health`
- `WS /ws/chat`

Structured logging writes one JSON-Lines entry per request with a trace ID and masked query text. The WebSocket handler catches `WebSocketDisconnect` so one client disconnect does not crash the service.

## Part 4

The separate AutoGen review demonstration returns a structured verdict:

```json
{
  "approved": true,
  "final_answer": "grounded answer text",
  "reason": "why the draft was approved or revised"
}
```

The runtime budget cap rejects oversized requests before processing. Grounded generation also uses an in-memory cache keyed by normalized query text, and the demo script records a repeated-query cache hit.

## Evidence

After running `python -m ola_support_agent.demo`, inspect:

- `work/demo_results.json`
- `work/requests.jsonl`

The demo output includes:

- dataset statistics
- threshold calibration values
- chunking precision/recall rows
- multi-turn memory transcript
- fresh-session reset transcript
- guardrail firing cases
- lookup-tool example
- out-of-scope fallback
- review approval and revision verdicts
- budget rejection evidence
- cache-hit evidence
- 15-query evaluation table with averages

## Risk Classification

This system is **Medium risk**. It influences customer support operations and ticket handling, so errors can affect customer experience and internal workflow, but it does not make medical, hiring, lending, or similarly high-impact regulated decisions.

For zero-network execution in PowerShell:

$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:CREWAI_DISABLE_TELEMETRY="true"
$env:PYTHONPATH=".;src"

pytest -q
python main.py




