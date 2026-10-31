"""Calibration + placement refresh endpoints (README §4.7/§4.8)."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.detection.cusum import CusumConfigError
from storage.calibration_store import DEFAULT_PATH, CalibrationStore

router = APIRouter(prefix="/v1")


class CalibrateIn(BaseModel):
    clean_chrs_values: list[float]


@router.post("/calibration/cusum")
def calibrate_cusum(payload: CalibrateIn, request: Request) -> dict[str, float]:
    engine = request.app.state.engine
    try:
        mu0, sigma = engine.calibrate(payload.clean_chrs_values)
    except CusumConfigError as exc:
        raise HTTPException(422, str(exc)) from exc

    # Persist so recalibration survives restarts. The path is config-gated
    # (tests isolate via tmp paths); empty string disables persistence.
    store_path = request.app.state.config.get("cusum", {}).get(
        "calibration_store_path", DEFAULT_PATH
    )
    if store_path:
        CalibrationStore(store_path).save(
            mu0, sigma, extra={"n_samples": len(payload.clean_chrs_values)}
        )
        request.app.state.repository.save_calibration(mu0, sigma, updated_at=time.time())
    return {"mu0": mu0, "sigma": sigma}


class RefreshIn(BaseModel):
    template_id: str
    use_history: bool = False
    use_measured_costs: bool = False


@router.get("/placement/templates")
def list_templates(request: Request) -> dict[str, object]:
    templates = sorted({g.pipeline_template_id for g in request.app.state.run_graphs.values()})
    return {"templates": templates}


@router.post("/placement/refresh")
def refresh_placement(payload: RefreshIn, request: Request) -> dict[str, object]:
    # run_graphs is keyed by RUN id; resolve the template's graph by attribute
    # (the v1 lookup by template_id only worked when the ids coincided — bug
    # found by tests/test_placement_v2.py).
    graph = next(
        (
            g
            for g in request.app.state.run_graphs.values()
            if g.pipeline_template_id == payload.template_id
        ),
        None,
    )
    if graph is None or not graph.nodes:
        raise HTTPException(404, f"unknown or empty template graph {payload.template_id}")
    history = None
    if payload.use_history:
        # Trace-simulated risk (Phase 8): every analyzed run of this template
        # contributes its (graph, AnalysisResult) to the risk model.
        history = [
            (g, request.app.state.results[rid])
            for rid, g in request.app.state.run_graphs.items()
            if rid in request.app.state.results and g.pipeline_template_id == payload.template_id
        ]
    s_star = request.app.state.engine.refresh_placement(
        payload.template_id,
        graph,
        history=history,
        use_measured_costs=payload.use_measured_costs,
    )
    return {
        "template_id": payload.template_id,
        "S_star": sorted(s_star),
        "history_traces": len(history) if history else 0,
        "measured_costs": bool(payload.use_measured_costs),
    }
