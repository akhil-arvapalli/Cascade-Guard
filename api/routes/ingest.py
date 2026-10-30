"""Trace ingestion + analysis endpoints (TODO Phase 11)."""

from __future__ import annotations

import threading
import uuid
from typing import Any
from typing import Any as _Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.engine import Alert, AnalysisResult
from core.graph.pipeline_graph import Document, PipelineGraph, Step
from storage.repository import AlertRecord

router = APIRouter(prefix="/v1")


class StepIn(BaseModel):
    step_id: str
    kind: str = Field(pattern="^(plan|retrieve|reason|tool_call|synthesize)$")
    output: str = ""
    retrieved_docs: list[dict[str, str]] = []
    predecessors: list[str] = []
    tool_invocation: dict[str, Any] | None = None


class RunIn(BaseModel):
    run_id: str | None = None
    pipeline_template_id: str = "adhoc"


def _alert_json(a: Alert) -> dict:
    return {
        "triggered_at_step": a.triggered_at_step,
        "s_n_value": round(a.s_n_value, 6),
        "root_causes": [
            {"step_id": rc.step_id, "share": round(rc.share, 4), "hops": rc.path_length}
            for rc in a.root_causes
        ],
    }


@router.post("/runs")
def create_run(payload: RunIn, request: Request) -> dict[str, str]:
    run_id = payload.run_id or str(uuid.uuid4())
    request.app.state.repository.create_run(run_id, payload.pipeline_template_id)
    request.app.state.run_graphs[run_id] = PipelineGraph(payload.pipeline_template_id)
    return {"run_id": run_id, "status": "running"}


@router.post("/runs/{run_id}/steps")
def add_step(run_id: str, step: StepIn, request: Request) -> dict[str, _Any]:
    if run_id not in request.app.state.run_graphs:
        raise HTTPException(404, f"unknown run {run_id}")
    graph = request.app.state.run_graphs[run_id]
    docs = [Document(doc_id=d["doc_id"], content=d["content"]) for d in step.retrieved_docs]
    try:
        graph.add_step(
            Step(
                id=step.step_id,
                kind=step.kind,  # type: ignore[arg-type]  # regex-validated above
                output=step.output,
                retrieved_docs=docs,
                predecessors=step.predecessors,
                tool_invocation=step.tool_invocation,
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"run_id": run_id, "step_id": step.step_id, "steps_recorded": len(graph.nodes)}


@router.post("/runs/{run_id}/analyze")
def analyze_run(run_id: str, request: Request) -> dict[str, _Any]:
    """Full-pipeline analysis: scores, Theorem 1 check, CUSUM, alert dispatch."""
    graph = request.app.state.run_graphs.get(run_id)
    if graph is None or not graph.nodes:
        raise HTTPException(404, f"unknown or empty run {run_id}")
    engine = request.app.state.engine
    result = engine.analyze_run(graph, run_id=run_id)
    request.app.state.results[run_id] = result
    _persist(request, run_id, graph, result)

    threading.Thread(
        target=request.app.state.dispatcher.dispatch,
        args=(result.alerts,),
        daemon=True,
    ).start()

    return {
        "run_id": run_id,
        "analyze_ms": round(result.elapsed_ms, 3),
        "alerted": result.alerted,
        "alerts": [_alert_json(a) for a in result.alerts],
        "scores": {
            nid: {
                "H": result.H[nid],
                "h_cascade": round(result.h_cascade[nid], 6),
                "chaf": round(result.chaf[nid], 6),
                "chrs": round(result.chrs[nid], 6),
                "fidelity": result.monitored_fidelity[nid],
                "se": result.semantic_entropy_values.get(nid),
            }
            for nid in graph.topo_order()
        },
    }


def _persist(request: Request, run_id: str, graph: PipelineGraph, result: AnalysisResult) -> None:
    repo = request.app.state.repository
    for node_id in graph.nodes:
        step = graph.nodes[node_id]
        repo.save_step(
            run_id,
            step,
            {
                "h": result.H.get(node_id),
                "h_cascade": result.h_cascade.get(node_id),
                "chaf": result.chaf.get(node_id),
                "chrs": result.chrs.get(node_id),
                "se": result.semantic_entropy_values.get(node_id),
                "fidelity": result.monitored_fidelity.get(node_id, "cheap"),
            },
        )
    for (src, dst), alpha_res in result.alpha_edges.items():
        repo.save_edge(run_id, src, dst, alpha_res.value, alpha_res.method)
    for alert in result.alerts:
        repo.insert_alert(
            AlertRecord(
                alert_id=alert.alert_id,
                run_id=alert.run_id,
                triggered_at_step=alert.triggered_at_step,
                s_n_value=alert.s_n_value,
                root_cause_step=(alert.root_causes[0].step_id if alert.root_causes else None),
            )
        )


@router.post("/cheap_check/{run_id}/{step_id}")
def cheap_check(run_id: str, step_id: str, request: Request) -> dict[str, _Any]:
    """Sub-12ms cached lookup of a step's last-computed CHRS."""
    result = request.app.state.results.get(run_id)
    if result is None or step_id not in result.chrs:
        raise HTTPException(404, "no analysis available for this run/step")
    return {
        "run_id": run_id,
        "step_id": step_id,
        "chrs": result.chrs[step_id],
        "fidelity": result.monitored_fidelity[step_id],
    }
