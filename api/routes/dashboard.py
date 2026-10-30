"""Dashboard read endpoints (TODO Phase 13).

Expose per-run graph + scores + alerts so a frontend can render the DAG
visualization, alert drill-down, and aggregate metrics.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from core.engine import AnalysisResult
from core.graph.pipeline_graph import PipelineGraph

router = APIRouter(prefix="/v1")


def _node_json(node_id: str, graph: PipelineGraph, result: AnalysisResult) -> dict[str, Any]:
    step = graph.nodes[node_id]
    return {
        "id": node_id,
        "kind": step.kind,
        "output": step.output,
        "retrieved_docs": [{"doc_id": d.doc_id, "content": d.content} for d in step.retrieved_docs],
        "predecessors": list(step.predecessors),
        "tool_invocation": step.tool_invocation,
        "timestamp": step.timestamp,
        "depth": graph.depth(node_id),
        "H": result.H.get(node_id),
        "h_cascade": result.h_cascade.get(node_id),
        "chaf": result.chaf.get(node_id),
        "chrs": result.chrs.get(node_id),
        "se": result.semantic_entropy_values.get(node_id),
        "fidelity": result.monitored_fidelity.get(node_id, "cheap"),
    }


def _edge_json(src: str, dst: str, result: AnalysisResult) -> dict[str, Any]:
    ares = result.alpha_edges.get((src, dst))
    return {
        "src": src,
        "dst": dst,
        "alpha": ares.value if ares else None,
        "alpha_method": ares.method if ares else None,
    }


@router.get("/runs")
def list_runs(request: Request) -> dict[str, Any]:
    repo = request.app.state.repository
    runs = []
    for run_id, record in repo._runs.items():  # noqa: SLF001 — in-memory impl
        result = request.app.state.results.get(run_id)
        runs.append(
            {
                "run_id": run_id,
                "pipeline_template_id": record.get("pipeline_template_id"),
                "started_at": record.get("started_at"),
                "status": record.get("status"),
                "step_count": len(
                    request.app.state.run_graphs.get(run_id, type("G", (), {"nodes": {}})()).nodes
                ),
                "alerted": bool(result and result.alerted),
                "analyze_ms": round(result.elapsed_ms, 3) if result else None,
            }
        )
    runs.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return {"count": len(runs), "runs": runs}


@router.get("/runs/{run_id}")
def get_run_detail(run_id: str, request: Request) -> dict[str, Any]:
    graph = request.app.state.run_graphs.get(run_id)
    if graph is None:
        raise HTTPException(404, f"unknown run {run_id}")
    result = request.app.state.results.get(run_id)
    record = request.app.state.repository.get_run(run_id)
    alerts = request.app.state.repository.list_alerts(run_id)

    nodes = [_node_json(nid, graph, result) for nid in graph.topo_order()] if result else []
    edges = [_edge_json(s, d, result) for (s, d) in graph.edges] if result else []

    return {
        "run_id": run_id,
        "run_record": record,
        "alerted": bool(result and result.alerted),
        "analyze_ms": round(result.elapsed_ms, 3) if result else None,
        "nodes": nodes,
        "edges": edges,
        "alerts": [
            {
                "alert_id": a.alert_id,
                "triggered_at_step": a.triggered_at_step,
                "s_n_value": a.s_n_value,
                "root_cause_step": a.root_cause_step,
                "created_at": a.created_at,
            }
            for a in alerts
        ],
        "scores_summary": {
            "max_chrs": max((n["chrs"] for n in nodes if n["chrs"] is not None), default=None),
            "max_chaf": max((n["chaf"] for n in nodes if n["chaf"] is not None), default=None),
            "monitored_count": sum(1 for n in nodes if n["fidelity"] == "full"),
            "total_steps": len(nodes),
        },
    }


@router.get("/alerts")
def list_all_alerts(request: Request, run_id: str | None = None) -> dict[str, Any]:
    records = request.app.state.repository.list_alerts(run_id)
    return {
        "count": len(records),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "run_id": a.run_id,
                "triggered_at_step": a.triggered_at_step,
                "s_n_value": a.s_n_value,
                "root_cause_step": a.root_cause_step,
                "created_at": a.created_at,
            }
            for a in records
        ],
    }
