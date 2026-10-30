"""In-band blocking decision endpoint (TODO Phase 12 leftover).

POST /v1/decision/{run_id}/{step_id} — the agent calls this synchronously with
a CANDIDATE step (not yet committed) before executing it. The endpoint scores
the step against the run's current graph (H → α → H_cascade → CHRS, Eqs. 3-10)
and projects ONE CUSUM update from the run's accumulated S_n.

Semantics (deliberate, documented):
- Decisions are EPHEMERAL projections: the pooled per-run detector is read but
  never advanced (AGENTS.md §2.5 — /analyze remains the authoritative scorer;
  repeated decisions cannot double-count S_n).
- SE never runs here: decisions are hot-path and the candidate is not in S*
  (budget-constrained monitoring, AGENTS.md §2.4) — cheap fidelity only.
- mode "notify_only" always returns block=false (scores still returned, so the
  agent can log them); mode "block" enforces the gates below.

Block gates (mode=block): CUSUM projection s_n_projected > h_control_limit,
and optionally an absolute CHRS gate blocking.threshold_tau for single-step
risk that hasn't accumulated CUSUM state yet.

H and α are cached per run (both are immutable per step/edge) and invalidated
by a (nodes, edges) size signature; the per-decision recompute is the Eq. 5
topological pass plus one α estimate for each new candidate edge.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.detection.cusum import CusumDetector
from core.graph.pipeline_graph import Document, PipelineGraph, Step
from core.scoring.cascade_score import compute_h_cascade
from core.scoring.faithfulness import hallucination_score
from core.scoring.risk_combiner import chrs as chrs_fn

router = APIRouter(prefix="/v1")


class DecisionIn(BaseModel):
    kind: str = Field(default="synthesize", pattern="^(plan|retrieve|reason|tool_call|synthesize)$")
    output: str = ""
    retrieved_docs: list[dict[str, str]] = []
    predecessors: list[str] = []


def _run_cache(request: Request, run_id: str, graph: PipelineGraph, engine: Any) -> dict[str, Any]:
    """Per-run {H, alpha} cache keyed by a graph-size signature."""
    cache: dict[str, Any] | None = getattr(request.app.state, "decision_cache", None)
    if cache is None:
        cache = {}
        request.app.state.decision_cache = cache
    sig = (len(graph.nodes), len(graph.edges))
    entry: dict[str, Any] | None = cache.get(run_id)
    if entry is None or entry["sig"] != sig:
        h_values = {
            nid: hallucination_score(step.output, step.retrieved_docs, engine.embedder)
            for nid, step in graph.nodes.items()
        }
        entry = {"sig": sig, "H": h_values, "alpha": engine._estimate_all_alpha(graph)}
        cache[run_id] = entry
    return entry


@router.post("/decision/{run_id}/{step_id}")
def decide_step(run_id: str, step_id: str, payload: DecisionIn, request: Request) -> dict[str, Any]:
    state = request.app.state
    graph: PipelineGraph | None = state.run_graphs.get(run_id)
    if graph is None:
        raise HTTPException(404, f"unknown run {run_id}")
    if step_id in graph.nodes:
        raise HTTPException(
            409, f"step {step_id!r} already exists in run {run_id}; decisions score candidate steps"
        )
    missing = [p for p in payload.predecessors if p not in graph.nodes]
    if missing:
        raise HTTPException(400, f"unknown predecessors {missing} in run {run_id}")

    engine = state.engine
    cfg = state.config.get("blocking", {})
    mode = cfg.get("mode", "notify_only")
    tau = cfg.get("threshold_tau")

    docs = [
        Document(doc_id=d.get("doc_id", f"{step_id}_d{i}"), content=d.get("content", ""))
        for i, d in enumerate(payload.retrieved_docs)
    ]
    candidate = Step(
        id=step_id,
        kind=payload.kind,  # type: ignore[arg-type]  # Field pattern constrains to StepKind
        output=payload.output,
        retrieved_docs=docs,
        predecessors=list(payload.predecessors),
    )

    entry = _run_cache(request, run_id, graph, engine)

    # Extended graph = current run + candidate (candidates are never committed
    # here; /analyze via the normal ingest path remains the commit point).
    probe = PipelineGraph(pipeline_template_id=graph.pipeline_template_id)
    for nid in graph.topo_order():
        probe.add_step(dataclasses.replace(graph.nodes[nid]))
    probe.add_step(candidate)

    h_ext = dict(entry["H"])
    h_ext[step_id] = hallucination_score(payload.output, docs, engine.embedder)
    alpha_ext = dict(entry["alpha"])
    for pred in payload.predecessors:
        alpha_ext[(pred, step_id)] = engine._estimate_alpha(graph.nodes[pred], candidate)

    numeric_H = {nid: (h if h is not None else 0.0) for nid, h in h_ext.items()}
    alpha_plain = {edge: a.value for edge, a in alpha_ext.items()}
    h_cascade = compute_h_cascade(probe, alpha_plain, numeric_H)
    # SE = 0.0: cheap fidelity on the decision path (see module docstring).
    chrs_candidate = chrs_fn(h_cascade[step_id], 0.0, engine.lambda_)

    # Ephemeral CUSUM projection: read the run's accumulated S_n, never advance it.
    detector: CusumDetector = engine.detector_pool.get_or_create(run_id, engine.cusum_params)
    s_projected = max(0.0, detector.S + chrs_candidate - detector.kappa)

    reasons: list[str] = []
    if mode == "block":
        if s_projected > detector.h:
            reasons.append("cusum_control_limit")
        if tau is not None and chrs_candidate > float(tau):
            reasons.append("chrs_threshold")

    return {
        "run_id": run_id,
        "step_id": step_id,
        "mode": mode,
        "block": bool(reasons),
        "reasons": reasons,
        "H": h_ext[step_id],
        "h_cascade": h_cascade[step_id],
        "chrs": chrs_candidate,
        "s_n_current": detector.S,
        "s_n_projected": s_projected,
        "h_control_limit": detector.h,
    }
