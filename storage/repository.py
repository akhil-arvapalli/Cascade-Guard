"""Trace repository layer (README SS5).

InMemoryTraceRepository is the v1 default so the whole system runs without
Postgres; PostgresTraceRepository is driver-guarded for real deployments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from core.graph.pipeline_graph import Step


@dataclass
class AlertRecord:
    alert_id: str
    run_id: str
    triggered_at_step: str
    s_n_value: float
    root_cause_step: str | None = None
    created_at: float = 0.0


class TraceRepository(Protocol):
    """Storage surface the engine/API need; SQL schema is README SS5."""

    def create_run(self, run_id: str, template_id: str) -> None: ...
    def complete_run(self, run_id: str, status: str) -> None: ...
    def save_step(self, run_id: str, step: Step, scores: dict[str, float | None]) -> None: ...
    def save_edge(
        self, run_id: str, src: str, dst: str, alpha_value: float, alpha_method: str
    ) -> None: ...
    def save_cusum_state(self, run_id: str, state: dict[str, float]) -> None: ...
    def save_calibration(self, mu0: float, sigma: float, updated_at: float) -> None: ...
    def get_calibration(self) -> dict[str, float] | None: ...
    def insert_alert(self, alert: AlertRecord) -> None: ...
    def list_alerts(self, run_id: str | None = None) -> list[AlertRecord]: ...
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    def list_runs(self) -> list[dict[str, Any]]: ...


def _now() -> float:
    return time.time()


class InMemoryTraceRepository:
    """Dict-backed store mirroring the README SS5 table shape one-to-one."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._steps: dict[str, tuple[str, Step, dict]] = {}
        self._edges: dict[tuple[str, str], dict] = {}
        self._cusum: dict[str, dict[str, float]] = {}
        self._calibration: dict[str, float] | None = None
        self._alerts: list[AlertRecord] = []

    def create_run(self, run_id: str, template_id: str) -> None:
        self._runs[run_id] = {
            "run_id": run_id,
            "pipeline_template_id": template_id,
            "started_at": _now(),
            "completed_at": None,
            "status": "running",
        }

    def complete_run(self, run_id: str, status: str) -> None:
        if run := self._runs.get(run_id):
            run["completed_at"] = _now()
            run["status"] = status

    def save_step(self, run_id: str, step: Step, scores: dict[str, float | None]) -> None:
        # scores keys map 1:1 onto `steps` columns of README SS5.
        self._steps[step.id] = (run_id, step, dict(scores))

    def save_edge(
        self,
        run_id: str,
        src: str,
        dst: str,
        alpha_value: float,
        alpha_method: str,
    ) -> None:
        self._edges[(src, dst)] = {
            "alpha": alpha_value,
            "alpha_method": alpha_method,
        }

    def save_cusum_state(self, run_id: str, state: dict[str, float]) -> None:
        self._cusum[run_id] = dict(state)

    def save_calibration(self, mu0: float, sigma: float, updated_at: float) -> None:
        self._calibration = {"mu0": mu0, "sigma": sigma, "updated_at": updated_at}

    def get_calibration(self) -> dict[str, float] | None:
        return dict(self._calibration) if self._calibration else None

    def insert_alert(self, alert: AlertRecord) -> None:
        self._alerts.append(alert)

    def list_alerts(self, run_id: str | None = None) -> list[AlertRecord]:
        if run_id is None:
            return list(self._alerts)
        return [a for a in self._alerts if a.run_id == run_id]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        """All run records, newest first (started_at desc)."""
        runs = [dict(record) for record in self._runs.values()]
        runs.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
        return runs


def make_postgres_repository(dsn: str | None = None) -> Any:
    """Driver-guarded factory: fails loudly if [postgres] extra not installed.

    `dsn` is passed through to PostgresTraceRepository (falls back to
    $CASCADEGUARD_PG_DSN there when omitted).
    """
    try:
        import psycopg  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "storage backend 'postgres' requires pip install cascadeguard[postgres] "
            "(psycopg + pgvector); set storage.backend=memory or install the extra"
        ) from exc
    from storage.postgres_impl import PostgresTraceRepository

    return PostgresTraceRepository(dsn=dsn)
