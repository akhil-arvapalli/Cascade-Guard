"""CascadeGuard Core Engine service — FastAPI app (TODO Phase 11).

Wires: trace ingestion -> engine analysis -> persistence -> alert dispatch.
Default composition uses the in-memory repository; set storage.backend=postgres
with CASCADEGUARD_PG_DSN for the real store.

Phase 15 hardening slice lives here too: static API-key auth middleware, JSON
structured logging with request timing, and GET /health.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path as _Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

# Allow running as `uvicorn api.main:app` from repo root without install.
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.alerts import AlertDispatcher, build_sink  # noqa: E402
from core.embeddings import load_embedder  # noqa: E402
from core.engine import CascadeGuardEngine  # noqa: E402
from core.graph.pipeline_graph import PipelineGraph  # noqa: E402, F401
from storage.repository import InMemoryTraceRepository, make_postgres_repository  # noqa: E402

_LOGGER_NAME = "cascadeguard"

# Standard LogRecord attributes — anything else on the record is caller `extra`.
_STD_LOG_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, message + extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(config: dict[str, Any]) -> logging.Logger:
    """Configure the 'cascadeguard' logger (JSON lines by default).

    config.logging: {json: bool (default true), level: str (default INFO)}.
    Only the app logger is configured — the root logger is never touched, so
    embedding this app into other processes/uvicorn setups stays predictable.
    """
    log_cfg = config.get("logging", {})
    logger = logging.getLogger(_LOGGER_NAME)
    level_name = str(log_cfg.get("level", "INFO")).upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    already_json = any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers)
    if log_cfg.get("json", True) and not already_json:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Logs method/path/status/latency(ms) at INFO on the cascadeguard logger."""

    def __init__(self, app: ASGIApp, logger: logging.Logger) -> None:
        super().__init__(app)
        self._log = logger

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": 500,
                    "ms": round(elapsed_ms, 3),
                },
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round(elapsed_ms, 3),
            },
        )
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Static API-key gate (Phase 15): header value must match auth.api_key.

    Comparison is constant-time (secrets.compare_digest). Exempt paths:
    /docs, /openapi.json, /health — everything else, including /v1/* and the
    mounted dashboard, requires the configured header.
    """

    _EXEMPT_PATHS = frozenset({"/docs", "/openapi.json", "/health"})

    def __init__(self, app: ASGIApp, api_key: str, header: str) -> None:
        super().__init__(app)
        self._api_key = api_key
        self._header = header.lower()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path not in self._EXEMPT_PATHS:
            provided = request.headers.get(self._header, "")
            if not secrets.compare_digest(provided.encode("utf-8"), self._api_key.encode("utf-8")):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)


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


def _build_repository(cfg: dict[str, Any]) -> Any:
    """Build the configured storage backend ('memory' default, 'postgres' optional)."""
    storage_cfg = cfg.get("storage", {})
    backend = str(storage_cfg.get("backend", "memory"))
    if backend != "postgres":
        return InMemoryTraceRepository()
    dsn_env_var = str(storage_cfg.get("dsn_env_var", "CASCADEGUARD_PG_DSN"))
    dsn = os.environ.get(dsn_env_var, "")
    if not dsn:
        raise RuntimeError(
            f"storage.backend=postgres requires the DSN env var {dsn_env_var!r} "
            "(e.g. postgresql://cascadeguard:cascadeguard@localhost:5432/cascadeguard)"
        )
    try:
        return make_postgres_repository(dsn=dsn)
    except ImportError as exc:
        raise RuntimeError(
            "storage backend 'postgres' requires pip install cascadeguard[postgres] "
            "(psycopg + pgvector); set storage.backend=memory or install the extra"
        ) from exc


def create_app(
    config: dict[str, Any] | None = None,
    repository: object | None = None,
    dispatcher: object | None = None,
) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="CascadeGuard Core Engine", version="0.1.0")

    app.state.config = cfg
    app.state.repository = repository or _build_repository(cfg)
    sinks = [build_sink(s) for s in cfg.get("alerts", {}).get("sinks", [])]
    app.state.dispatcher = dispatcher or AlertDispatcher(sinks)
    app.state.engine = build_engine(cfg)
    app.state.run_graphs = {}
    app.state.results = {}

    # --- Phase 15: health endpoint -------------------------------------------
    from storage.calibration_store import DEFAULT_PATH, CalibrationStore

    cal_path = cfg.get("cusum", {}).get("calibration_store_path", DEFAULT_PATH)
    storage_backend = str(cfg.get("storage", {}).get("backend", "memory"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        # `calibrated` = persisted CUSUM calibration exists (file store); the
        # JSON store is the source of truth because engine.cusum_params is also
        # mutated at boot from it (and by POST /v1/calibration/cusum).
        calibrated = bool(CalibrationStore(cal_path).load()) if cal_path else False
        return {"status": "ok", "storage_backend": storage_backend, "calibrated": calibrated}

    # --- Phase 15: auth + request-timing middleware ---------------------------
    # Middleware added LAST wraps OUTERMOST, so timing wraps auth responses too
    # (every request gets a latency line, including 401s).
    auth_cfg = cfg.get("auth", {})
    if auth_cfg.get("enabled", False):
        api_key = str(auth_cfg.get("api_key", ""))
        if not api_key:
            raise RuntimeError("auth.enabled=true requires a non-empty auth.api_key")
        app.add_middleware(
            ApiKeyMiddleware,
            api_key=api_key,
            header=str(auth_cfg.get("header", "X-API-Key")),
        )
    logger = setup_logging(cfg)
    app.add_middleware(RequestTimingMiddleware, logger=logger)

    from api.routes.alerts import router as alerts_router
    from api.routes.calibration import router as calibration_router
    from api.routes.dashboard import router as dashboard_router
    from api.routes.decision import router as decision_router
    from api.routes.ingest import router as ingest_router

    app.include_router(ingest_router)
    app.include_router(alerts_router)
    app.include_router(calibration_router)
    app.include_router(decision_router)
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
