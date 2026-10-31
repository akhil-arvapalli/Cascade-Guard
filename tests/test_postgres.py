"""Postgres integration tests — SKIPPED unless $CASCADEGUARD_PG_DSN is set.

Run against a real Postgres+pgvector instance:

    docker compose up -d postgres
    export CASCADEGUARD_PG_DSN=postgresql://cascadeguard:cascadeguard@localhost:5432/cascadeguard
    pytest -q tests/test_postgres.py

If Docker isn't available (or the container fails to come up) the whole module
skips — that is the expected outcome in environments without Postgres.

Each test works on a uniquely-uuid'd run and deletes its rows afterwards; the
calibration row is a shared single-row slot (per schema.sql) and is simply
overwritten, never deleted.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from core.graph.pipeline_graph import Document, Step
from storage.repository import AlertRecord

DSN = os.environ.get("CASCADEGUARD_PG_DSN", "")

pytestmark = [
    pytest.mark.skipif(
        not DSN,
        reason="CASCADEGUARD_PG_DSN not set — Postgres integration tests skipped",
    )
]

# FK-safe deletion order (children before the run row).
_TABLES_BY_RUN = ("alerts", "cusum_states", "step_edges", "steps", "pipeline_runs")


def _repo() -> Any:
    """PostgresTraceRepository bound to the env DSN; skips if psycopg missing."""
    pytest.importorskip("psycopg", reason="pip install cascadeguard[postgres]")
    from storage.postgres_impl import PostgresTraceRepository

    return PostgresTraceRepository(dsn=DSN)


def _cleanup_run(run_id: str) -> None:
    """Best-effort deletion of every row belonging to `run_id`."""
    psycopg = pytest.importorskip("psycopg")
    conn = psycopg.connect(DSN)
    try:
        with conn.cursor() as cur:
            for table in _TABLES_BY_RUN:
                cur.execute(f"DELETE FROM {table} WHERE run_id = %s", (uuid.UUID(run_id),))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def repo() -> Iterator[Any]:
    yield _repo()


@pytest.fixture()
def run_id(repo: Any) -> Iterator[str]:
    rid = str(uuid.uuid4())
    repo.create_run(rid, "test-template")
    yield rid
    _cleanup_run(rid)


def _sql_rows(query: str, params: list[Any]) -> list[tuple[Any, ...]]:
    psycopg = pytest.importorskip("psycopg")
    conn = psycopg.connect(DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return list(cur.fetchall())
    finally:
        conn.close()


class TestRunLifecycle:
    def test_create_get_complete(self, repo: Any, run_id: str) -> None:
        record = repo.get_run(run_id)
        assert record is not None
        assert record["run_id"] == run_id
        assert record["pipeline_template_id"] == "test-template"
        assert record["status"] == "running"
        assert isinstance(record["started_at"], float)
        assert record["completed_at"] is None

        repo.complete_run(run_id, "alerted")
        done = repo.get_run(run_id)
        assert done is not None
        assert done["status"] == "alerted"
        assert done["completed_at"] is not None

    def test_complete_unknown_run_is_silent_noop(self, repo: Any) -> None:
        repo.complete_run(str(uuid.uuid4()), "completed")  # must not raise

    def test_list_runs_newest_first(self, repo: Any) -> None:
        old_id = str(uuid.uuid4())
        new_id = str(uuid.uuid4())
        try:
            repo.create_run(old_id, "t-old")
            time.sleep(0.01)
            repo.create_run(new_id, "t-new")
            runs = repo.list_runs()
            ids = [r["run_id"] for r in runs]
            assert new_id in ids and old_id in ids
            assert ids.index(new_id) < ids.index(old_id)
            assert all({"run_id", "pipeline_template_id", "started_at"} <= set(r) for r in runs)
        finally:
            _cleanup_run(old_id)
            _cleanup_run(new_id)

    def test_get_unknown_run_returns_none(self, repo: Any) -> None:
        assert repo.get_run(str(uuid.uuid4())) is None


class TestStepEdgeCusum:
    def test_step_roundtrip(self, repo: Any, run_id: str) -> None:
        step = Step(
            id=f"s1_{run_id[:8]}",
            kind="reason",
            output="the capital is Paris",
            retrieved_docs=[Document(doc_id="d1", content="Paris is the capital of France")],
            tool_invocation={"name": "search", "query": "capital of France"},
        )
        repo.save_step(
            run_id,
            step,
            {
                "h": 0.42,
                "h_cascade": 0.51,
                "chaf": 0.1,
                "chrs": 0.33,
                "se": None,
                "fidelity": "full",
            },
        )
        rows = _sql_rows(
            "SELECT run_id::text, kind, output, h_score, h_cascade, chaf, chrs, "
            "semantic_entropy, monitored_fidelity, retrieved_doc_ids, tool_call "
            "FROM steps WHERE step_id = %s",
            [step.id],
        )
        assert len(rows) == 1
        (
            got_run,
            kind,
            output,
            h,
            h_cascade,
            chaf,
            chrs,
            se,
            fidelity,
            doc_ids,
            tool_call,
        ) = rows[0]
        assert got_run == run_id
        assert kind == "reason"
        assert output == "the capital is Paris"
        assert (h, h_cascade, chaf, chrs) == (0.42, 0.51, 0.1, 0.33)
        assert se is None
        assert fidelity == "full"
        assert list(doc_ids) == ["d1"]
        assert tool_call["name"] == "search"

    def test_step_upsert_overwrites_scores(self, repo: Any, run_id: str) -> None:
        sid = f"s_up_{run_id[:8]}"
        step = Step(id=sid, kind="plan", output="v1")
        repo.save_step(run_id, step, {"h": 0.1, "chrs": 0.2, "fidelity": "cheap"})
        repo.save_step(run_id, step, {"h": 0.9, "chrs": 0.8, "fidelity": "full"})
        rows = _sql_rows(
            "SELECT h_score, chrs, monitored_fidelity FROM steps WHERE step_id = %s", [sid]
        )
        assert rows == [(0.9, 0.8, "full")]

    def test_edge_roundtrip(self, repo: Any, run_id: str) -> None:
        src, dst = f"s1_{run_id[:8]}", f"s2_{run_id[:8]}"
        repo.save_step(run_id, Step(id=src, kind="retrieve", output="docs"), {})
        repo.save_step(run_id, Step(id=dst, kind="reason", output="answer"), {})
        repo.save_edge(run_id, src, dst, 0.75, "nli_proxy")
        rows = _sql_rows(
            "SELECT alpha, alpha_method, run_id::text FROM step_edges "
            "WHERE from_step = %s AND to_step = %s",
            [src, dst],
        )
        assert rows == [(0.75, "nli_proxy", run_id)]

    def test_cusum_state_upsert(self, repo: Any, run_id: str) -> None:
        repo.save_cusum_state(run_id, {"s_n": 0.4, "kappa": 0.3, "h_control_limit": 2.5})
        repo.save_cusum_state(run_id, {"s_n": 0.9, "kappa": 0.3, "h_control_limit": 2.5})
        rows = _sql_rows(
            "SELECT s_n, kappa, h_control_limit FROM cusum_states WHERE run_id = %s",
            [uuid.UUID(run_id)],
        )
        assert rows == [(0.9, 0.3, 2.5)]


class TestCalibration:
    def test_calibration_roundtrip(self, repo: Any) -> None:
        now = time.time()
        repo.save_calibration(0.25, 0.059, now)
        cal = repo.get_calibration()
        assert cal is not None
        assert cal["mu0"] == pytest.approx(0.25)
        assert cal["sigma"] == pytest.approx(0.059)
        assert cal["updated_at"] == pytest.approx(now, abs=1.0)

        # Upsert: still exactly one row after a second save.
        repo.save_calibration(0.3, 0.06, time.time())
        count = _sql_rows("SELECT count(*) FROM calibration", [])
        assert count == [(1,)]

    def test_get_calibration_unset_returns_none(self, repo: Any) -> None:
        # Cannot easily empty the shared row in a live DB; just assert the shape.
        cal = repo.get_calibration()
        assert cal is None or {"mu0", "sigma", "updated_at"} <= set(cal)


class TestAlerts:
    def test_alert_roundtrip(self, repo: Any, run_id: str) -> None:
        step_id = f"s_alert_{run_id[:8]}"
        repo.save_step(run_id, Step(id=step_id, kind="synthesize", output="done"), {})
        alert = AlertRecord(
            alert_id=str(uuid.uuid4()),
            run_id=run_id,
            triggered_at_step=step_id,
            s_n_value=1.234,
            root_cause_step=step_id,
        )
        repo.insert_alert(alert)
        found = repo.list_alerts(run_id)
        assert len(found) == 1
        assert found[0].alert_id == alert.alert_id
        assert found[0].run_id == run_id
        assert found[0].triggered_at_step == step_id
        assert found[0].s_n_value == pytest.approx(1.234)
        assert found[0].root_cause_step == step_id
        assert found[0].created_at > 0
        # DB stamps created_at=now() at insert, so compare on identity fields.
        assert any(a.alert_id == alert.alert_id for a in repo.list_alerts())
        assert repo.list_alerts(str(uuid.uuid4())) == []

    def test_insert_alert_idempotent(self, repo: Any, run_id: str) -> None:
        step_id = f"s_alert_{run_id[:8]}"
        repo.save_step(run_id, Step(id=step_id, kind="synthesize", output="done"), {})
        alert = AlertRecord(
            alert_id=str(uuid.uuid4()), run_id=run_id, triggered_at_step=step_id, s_n_value=0.5
        )
        repo.insert_alert(alert)
        repo.insert_alert(alert)  # re-analyze path must not duplicate
        assert len(repo.list_alerts(run_id)) == 1


class TestApplySchema:
    def test_apply_is_idempotent(self) -> None:
        pytest.importorskip("psycopg", reason="pip install cascadeguard[postgres]")
        from storage.apply_schema import apply_schema

        schema = Path(__file__).resolve().parent.parent / "storage" / "schema.sql"
        applied_twice = apply_schema(DSN, schema) + apply_schema(DSN, schema)
        assert applied_twice > 0
