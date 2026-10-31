"""Postgres-backed TraceRepository (psycopg 3, connection-per-operation).

Implements the full TraceRepository protocol against storage/schema.sql:
pipeline_runs, steps, step_edges, cusum_states, alerts, calibration.

v1 keeps persistence deliberately simple — one connection per operation via
psycopg.connect(dsn) with an explicit commit/rollback — swap for a
psycopg_pool pool when concurrency demands it. The DSN comes from the
constructor argument or, failing that, $CASCADEGUARD_PG_DSN (loud error if
neither is set). Ops note: libpq has no default connect timeout — append
`?connect_timeout=5` (or similar) to the DSN so an unreachable host fails
fast instead of hanging the request path.

Schema fidelity notes (do not "fix" one impl to match the other):
- steps are keyed by a GLOBAL step_id PRIMARY KEY per schema.sql, whereas
  InMemoryTraceRepository keys them by step.id only — both are per spec.
- scores dict keys map onto columns: h->h_score, h_cascade->h_cascade,
  chaf->chaf, chrs->chrs, se->semantic_entropy, fidelity->monitored_fidelity.
- save_cusum_state expects CusumDetector.state keys verbatim: s_n, kappa,
  h_control_limit (core/detection/cusum.py).
- insert_alert is idempotent on alert_id (ON CONFLICT DO NOTHING) so
  re-analyzing a run cannot duplicate alerts in the SQL store.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg

from core.graph.pipeline_graph import Step
from storage.repository import AlertRecord

DSN_ENV_VAR = "CASCADEGUARD_PG_DSN"

# scores-dict key -> steps column (README SS5); "fidelity" carries the
# monitored_fidelity label ('cheap'|'full'), the rest are floats.
_SCORE_COLUMN_MAP: dict[str, str] = {
    "h": "h_score",
    "h_cascade": "h_cascade",
    "chaf": "chaf",
    "chrs": "chrs",
    "se": "semantic_entropy",
    "fidelity": "monitored_fidelity",
}

_RUN_QUERY = (
    "SELECT run_id::text, pipeline_template_id, "
    "EXTRACT(EPOCH FROM started_at), EXTRACT(EPOCH FROM completed_at), status "
    "FROM pipeline_runs"
)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """Schema run/alert ids are UUIDs; coerce loudly (raises on junk)."""
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _run_row_to_dict(row: Any) -> dict[str, Any]:
    """Map a _RUN_QUERY row onto the InMemory run-record shape."""
    return {
        "run_id": str(row[0]),
        "pipeline_template_id": str(row[1]),
        "started_at": float(row[2]),
        "completed_at": float(row[3]) if row[3] is not None else None,
        "status": str(row[4]) if row[4] is not None else None,
    }


class PostgresTraceRepository:
    """TraceRepository implementation on Postgres (storage/schema.sql)."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or os.environ.get(DSN_ENV_VAR, "")
        if not self._dsn:
            raise RuntimeError(
                "PostgresTraceRepository requires a DSN: pass dsn= or set "
                f"${DSN_ENV_VAR} (e.g. postgresql://cascadeguard:...@localhost:5432/...)"
            )

    # -- connection handling -------------------------------------------------

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """Connection-per-operation: commit on success, rollback+close always."""
        conn = psycopg.connect(self._dsn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -- runs ----------------------------------------------------------------

    def create_run(self, run_id: str, template_id: str) -> None:
        sql = (
            "INSERT INTO pipeline_runs (run_id, pipeline_template_id, status) "
            "VALUES (%s, %s, 'running')"
        )
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [_as_uuid(run_id), template_id])

    def complete_run(self, run_id: str, status: str) -> None:
        sql = "UPDATE pipeline_runs SET status = %s, completed_at = now() WHERE run_id = %s"
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [status, _as_uuid(run_id)])
        # Unknown run_id: silent no-op, mirroring InMemoryTraceRepository.

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(f"{_RUN_QUERY} WHERE run_id = %s", [_as_uuid(run_id)])
            row = cur.fetchone()
        return _run_row_to_dict(row) if row is not None else None

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(f"{_RUN_QUERY} ORDER BY started_at DESC")
            rows = cur.fetchall()
        return [_run_row_to_dict(row) for row in rows]

    # -- steps / edges -------------------------------------------------------

    def save_step(self, run_id: str, step: Step, scores: dict[str, float | None]) -> None:
        """UPSERT one step row; scores keys map 1:1 onto `steps` columns."""
        columns = ["step_id", "run_id", "kind", "output", "retrieved_doc_ids", "tool_call"]
        values: list[Any] = [
            step.id,
            _as_uuid(run_id),
            step.kind,
            step.output,
            [doc.doc_id for doc in step.retrieved_docs],
            step.tool_invocation,
        ]
        for key, column in _SCORE_COLUMN_MAP.items():
            if key in scores:
                columns.append(column)
                values.append(scores[key])
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
        sql = (
            f"INSERT INTO steps ({', '.join(columns)}) "
            f"VALUES ({', '.join(['%s'] * len(columns))}) "
            f"ON CONFLICT (step_id) DO UPDATE SET {updates}"
        )
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, values)

    def save_edge(
        self,
        run_id: str,
        src: str,
        dst: str,
        alpha_value: float,
        alpha_method: str,
    ) -> None:
        sql = (
            "INSERT INTO step_edges (run_id, from_step, to_step, alpha, alpha_method) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (from_step, to_step) DO UPDATE SET "
            "run_id = EXCLUDED.run_id, alpha = EXCLUDED.alpha, "
            "alpha_method = EXCLUDED.alpha_method"
        )
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [_as_uuid(run_id), src, dst, alpha_value, alpha_method])

    # -- cusum / calibration --------------------------------------------------

    def save_cusum_state(self, run_id: str, state: dict[str, float]) -> None:
        """UPSERT the per-run CUSUM state (CusumDetector.state keys verbatim)."""
        sql = (
            "INSERT INTO cusum_states (run_id, s_n, kappa, h_control_limit, updated_at) "
            "VALUES (%s, %s, %s, %s, now()) "
            "ON CONFLICT (run_id) DO UPDATE SET "
            "s_n = EXCLUDED.s_n, kappa = EXCLUDED.kappa, "
            "h_control_limit = EXCLUDED.h_control_limit, updated_at = now()"
        )
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                [
                    _as_uuid(run_id),
                    state["s_n"],
                    state["kappa"],
                    state["h_control_limit"],
                ],
            )

    def save_calibration(self, mu0: float, sigma: float, updated_at: float) -> None:
        """Single-row UPSERT (calibration.id is a constant TRUE per schema)."""
        sql = (
            "INSERT INTO calibration (id, mu0, sigma, updated_at) "
            "VALUES (TRUE, %s, %s, to_timestamp(%s)) "
            "ON CONFLICT (id) DO UPDATE SET "
            "mu0 = EXCLUDED.mu0, sigma = EXCLUDED.sigma, updated_at = EXCLUDED.updated_at"
        )
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, [mu0, sigma, updated_at])

    def get_calibration(self) -> dict[str, float] | None:
        sql = "SELECT mu0, sigma, EXTRACT(EPOCH FROM updated_at) FROM calibration WHERE id"
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        if row is None:
            return None
        return {"mu0": float(row[0]), "sigma": float(row[1]), "updated_at": float(row[2])}

    # -- alerts ----------------------------------------------------------------

    def insert_alert(self, alert: AlertRecord) -> None:
        sql = (
            "INSERT INTO alerts (alert_id, run_id, triggered_at_step, s_n_value, "
            "root_cause_step, created_at) "
            "VALUES (%s, %s, %s, %s, %s, COALESCE(to_timestamp(%s), now())) "
            "ON CONFLICT (alert_id) DO NOTHING"
        )
        # created_at <= 0 means "unset" on the record — let the DB default stamp it.
        created_at = alert.created_at if alert.created_at > 0 else None
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                [
                    _as_uuid(alert.alert_id),
                    _as_uuid(alert.run_id),
                    alert.triggered_at_step,
                    alert.s_n_value,
                    alert.root_cause_step,
                    created_at,
                ],
            )

    def list_alerts(self, run_id: str | None = None) -> list[AlertRecord]:
        sql = (
            "SELECT alert_id::text, run_id::text, triggered_at_step, s_n_value, "
            "root_cause_step, EXTRACT(EPOCH FROM created_at) FROM alerts"
        )
        params: list[Any] = []
        if run_id is not None:
            sql += " WHERE run_id = %s"
            params.append(_as_uuid(run_id))
        sql += " ORDER BY created_at, alert_id"
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            AlertRecord(
                alert_id=str(row[0]),
                run_id=str(row[1]),
                triggered_at_step=row[2],
                s_n_value=float(row[3]) if row[3] is not None else 0.0,
                root_cause_step=row[4],
                created_at=float(row[5]),
            )
            for row in rows
        ]
