"""CascadeGuard Core Engine service — FastAPI app (TODO Phase 11).

Wires: trace ingestion -> engine analysis -> persistence -> alert dispatch.
Default composition uses the in-memory repository; set storage.backend=postgres
with CASCADEGUARD_PG_DSN for the real store.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path as _Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Allow running as `uvicorn api.main:app` from repo root without install.
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.alerts import AlertDispatcher, build_sink  # noqa: E402
from core.embeddings import load_embedder  # noqa: E402
from core.engine import CascadeGuardEngine  # noqa: E402
from core.graph.pipeline_graph import PipelineGraph  # noqa: E402, F401
from storage.repository import InMemoryTraceRepository  # noqa: E402


def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = path or os.environ.get("CASCADEGUARD_CONFIG", "configs/default.yaml")
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_engine(config: dict[str, Any]) -> CascadeGuardEngine:
    embedder = load_embedder(config.get("embeddings", {}))

    # Lightweight Track B entailment surrogate: embedding cosine similarity between
    # source output and target output. Wired by default so the dashboard/API is
    # demo-able with zero heavy-model downloads. Swap in a real NLI model
    # (microsoft/deberta-large-mnli) for production per configs/default.yaml.
    def entail_fn(source: str, target: str) -> float:
        a = embedder(source)
        b = embedder(target)
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0

    engine = CascadeGuardEngine(
        config=config,
        embedder=embedder,
        entail_fn=entail_fn,
        # generate_fn / counterfactual_stack: wire real models here when standing up
        # Track B counterfactual / semantic-entropy backends (DECISIONS D-002/D-003).
    )
    _load_persisted_calibration(engine, config)
    return engine


def _load_persisted_calibration(engine: CascadeGuardEngine, config: dict[str, Any]) -> None:
    """Boot-time reload of persisted CUSUM calibration (data/calibration.json).

    Without this the engine falls back to un-calibrated placeholder μ0/σ and
    the detector false-alarms on clean traces (smoke test, 2026-10-30).
    """
    from storage.calibration_store import DEFAULT_PATH, CalibrationStore

    path = config.get("cusum", {}).get("calibration_store_path", DEFAULT_PATH)
    cal = CalibrationStore(path).load()
    if cal is not None:
        engine.cusum_params.mu0 = cal["mu0"]
        engine.cusum_params.sigma = cal["sigma"]


def create_app(
    config: dict[str, Any] | None = None,
    repository: object | None = None,
    dispatcher: object | None = None,
) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="CascadeGuard Core Engine", version="0.1.0")

    app.state.config = cfg
    app.state.repository = repository or InMemoryTraceRepository()
    sinks = [build_sink(s) for s in cfg.get("alerts", {}).get("sinks", [])]
    app.state.dispatcher = dispatcher or AlertDispatcher(sinks)
    app.state.engine = build_engine(cfg)
    app.state.run_graphs = {}
    app.state.results = {}

    from api.routes.alerts import router as alerts_router
    from api.routes.calibration import router as calibration_router
    from api.routes.dashboard import router as dashboard_router
    from api.routes.ingest import router as ingest_router

    app.include_router(ingest_router)
    app.include_router(alerts_router)
    app.include_router(calibration_router)
    app.include_router(dashboard_router)

    # Serve the Phase 13 dashboard UI (built with Vite into dashboard/dist).
    _DASHBOARD_DIST = _REPO_ROOT / "dashboard" / "dist"
    if _DASHBOARD_DIST.is_dir():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(_DASHBOARD_DIST), html=True),
            name="dashboard",
        )
    else:

        @app.get("/dashboard", include_in_schema=False)
        def dashboard_not_built() -> dict[str, str]:
            return {
                "detail": ("Dashboard not built. Run `cd dashboard && npm run build` then restart.")
            }

    return app


app = create_app()
