"""Redis Streams trace-ingestion protocol + pure helpers (TODO Phase 11).

Stream contract: each message carries field ``payload`` = JSON object:
  {"type": "step",    "run_id": "...", "template_id": "...",
   "event": {"step_id", "kind", "output", "retrieved_docs",
             "predecessors", "tool_invocation"}}
  {"type": "run_end", "run_id": "..."}

POST /v1/ingest validates against this contract and XADDs to the stream;
``api/worker.py`` consumes via a consumer group, buffers steps per run, and on
``run_end`` runs the full analysis. ``parse_event`` is pure so the contract is
unit-testable without a live Redis.
"""

from __future__ import annotations

import json
from typing import Any

VALID_KINDS = frozenset({"plan", "retrieve", "reason", "tool_call", "synthesize"})


class StreamEventError(ValueError):
    """Malformed stream event — dead-lettered by the worker, 422 by the API."""


def parse_event(raw: bytes | str) -> dict[str, Any]:
    """Validate one stream payload; returns the decoded event or raises."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise StreamEventError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StreamEventError("payload must be a JSON object")
    if data.get("type") not in ("step", "run_end"):
        raise StreamEventError("event.type must be 'step' or 'run_end'")
    if not isinstance(data.get("run_id"), str) or not data["run_id"]:
        raise StreamEventError("event.run_id must be a non-empty string")
    if data["type"] == "step":
        ev = data.get("event")
        if not isinstance(ev, dict) or not ev.get("step_id"):
            raise StreamEventError("step events need event.step_id")
        if ev.get("kind", "reason") not in VALID_KINDS:
            raise StreamEventError(f"event.kind must be one of {sorted(VALID_KINDS)}")
        docs = ev.get("retrieved_docs", [])
        if not isinstance(docs, list) or any(
            not isinstance(d, dict) or "doc_id" not in d or "content" not in d for d in docs
        ):
            raise StreamEventError("event.retrieved_docs entries need doc_id and content")
    return data


def new_run_buffer(template_id: str | None) -> dict[str, Any]:
    return {"template_id": template_id or "adhoc", "steps": []}
