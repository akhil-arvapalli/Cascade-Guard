"""Dashboard read endpoints (TODO Phase 13).

Expose per-run graph + scores + alerts so a frontend can render the DAG
visualization, alert drill-down, and aggregate metrics.
"""

from __future__ import annotations

import json
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
    for record in repo.list_runs():
        run_id = str(record.get("run_id"))
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

    # Attach contribution-ranked root causes from the analysis result (the
    # persisted AlertRecord only keeps the single top root_cause_step).
    rc_by_step: dict[str, list[dict[str, Any]]] = {}
    if result:
        for al in result.alerts:
            rc_by_step[al.triggered_at_step] = [
                {"step_id": rc.step_id, "share": rc.share, "path_length": rc.path_length}
                for rc in al.root_causes
            ]

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
                "root_causes": rc_by_step.get(a.triggered_at_step, []),
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


# --- Phase 14 eval summary (the 150-trace HaluEval sample) ------------------


def _eval_cache(request: Request) -> dict[str, Any]:
    cache = getattr(request.app.state, "eval_cache", None)
    if cache is None:
        cache = {}
        request.app.state.eval_cache = cache
    return cache


@router.get("/eval/summary")
def eval_summary(request: Request) -> dict[str, Any]:
    """Aggregate results of the Phase 14 sample-dataset evaluation.

    Reads eval/results/ artifacts (written by eval/run_detection_eval.py and
    the figure scripts); 404 if the eval has not been run in this checkout.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    results = repo_root / "eval" / "results"
    try:
        det = json.loads((results / "detection_eval.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(
            404, "eval results not found; run python -m eval.run_detection_eval"
        ) from exc
    stats = {}
    try:
        stats = json.loads((results / "figs" / "fig_stats.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    return {
        "dataset": det.get("dataset", {"name": "halu_eval_rag_sample", "records": 150}),
        "lambda_sweep": det.get("lambda_sweep", []),
        "head_to_head": det.get("head_to_head", []),
        "theorem1": det.get("theorem1", det.get("chaf_check", {})),
        "latency": stats,
        "figures": [
            f"/dashboard/figs/{n}"
            for n in (
                "fig_latency.png",
                "fig_chaf_depth.png",
                "fig_h_distribution.png",
                "fig_lambda_sweep.png",
            )
        ],
    }


@router.get("/eval/samples")
def eval_samples(request: Request) -> dict[str, Any]:
    """Per-sample verdicts for all 150 labeled traces (computed once, cached)."""
    cache = _eval_cache(request)
    if "samples" in cache:
        return {
            "kappa": cache["kappa"],
            "count": len(cache["samples"]),
            "samples": cache["samples"],
        }
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    data = repo_root / "eval" / "datasets" / "data" / "halu_eval_rag_sample.json"
    if not data.is_file():
        raise HTTPException(404, "sample dataset not found")
    from eval.datasets.loaders import load_halu_eval_rag_sample

    engine = request.app.state.engine
    kappa = engine.cusum_params.mu0 + engine.cusum_params.delta / 2
    rows = []
    for i, labeled in enumerate(load_halu_eval_rag_sample(data)):
        truth = bool(next(iter(labeled.labels.values()), False))
        result = engine.analyze_run(labeled.graph)
        max_chrs = max(result.chrs.values()) if result.chrs else 0.0
        predicted = max_chrs > kappa
        rows.append(
            {
                "sample": f"sample-{i}",
                "truth": "hallucinated" if truth else "faithful",
                "max_chrs": round(max_chrs, 4),
                "predicted": "hallucinated" if predicted else "faithful",
                "match": predicted == truth,
            }
        )
    cache["samples"] = rows
    cache["kappa"] = round(kappa, 4)
    return {"kappa": cache["kappa"], "count": len(rows), "samples": rows}
