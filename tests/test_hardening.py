"""Phase 15 hardening slice: auth middleware, /health, dashboard list_runs shape."""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import _LOGGER_NAME, JsonFormatter, create_app, load_config
from core.alerts import AlertDispatcher, LogSink
from core.graph.pipeline_graph import PipelineGraph, Step
from storage.calibration_store import CalibrationStore
from storage.repository import InMemoryTraceRepository


def _config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Default config with an isolated calibration store path + overrides."""
    cfg = load_config()
    cfg.setdefault("cusum", {})["calibration_store_path"] = str(tmp_path / "calibration.json")
    cfg.update(overrides)
    return cfg


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    app = create_app(
        config=_config(tmp_path, **overrides),
        dispatcher=AlertDispatcher(sinks=[LogSink()]),
    )
    return TestClient(app)


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    yield _client(tmp_path)


class TestHealth:
    def test_shape_and_defaults(self, client: TestClient) -> None:
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert set(body.keys()) == {"status", "storage_backend", "calibrated"}
        assert body["status"] == "ok"
        assert body["storage_backend"] == "memory"
        assert body["calibrated"] is False  # tmp calibration path has no file

    def test_calibrated_flips_with_persisted_file(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        assert client.get("/health").json()["calibrated"] is False
        CalibrationStore(str(tmp_path / "calibration.json")).save(0.25, 0.059)
        assert client.get("/health").json()["calibrated"] is True

    def test_reports_postgres_backend_label(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        cfg["storage"] = {"backend": "postgres", "dsn_env_var": "CASCADEGUARD_PG_DSN"}
        app = create_app(
            config=cfg, repository=InMemoryTraceRepository(), dispatcher=AlertDispatcher([])
        )
        body = TestClient(app).get("/health").json()
        assert body["storage_backend"] == "postgres"


class TestAuthDisabled:
    def test_open_access_by_default(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/alerts").status_code == 200
        assert client.get("/openapi.json").status_code == 200


class TestAuthEnabled:
    @pytest.fixture()
    def auth_client(self, tmp_path: Path) -> Iterator[TestClient]:
        yield _client(
            tmp_path,
            auth={"enabled": True, "api_key": "secret-key", "header": "X-API-Key"},
        )

    def test_missing_header_is_401_json(self, auth_client: TestClient) -> None:
        res = auth_client.get("/v1/alerts")
        assert res.status_code == 401
        assert res.json()["detail"] == "unauthorized"

    def test_wrong_key_is_401(self, auth_client: TestClient) -> None:
        res = auth_client.get("/v1/alerts", headers={"X-API-Key": "wrong"})
        assert res.status_code == 401

    def test_right_key_is_200_on_health(self, auth_client: TestClient) -> None:
        assert auth_client.get("/health").status_code == 200

    def test_right_key_reaches_api(self, auth_client: TestClient) -> None:
        res = auth_client.get("/v1/alerts", headers={"X-API-Key": "secret-key"})
        assert res.status_code == 200

    def test_docs_and_openapi_exempt(self, auth_client: TestClient) -> None:
        assert auth_client.get("/docs").status_code == 200
        assert auth_client.get("/openapi.json").status_code == 200

    def test_custom_header_name(self, tmp_path: Path) -> None:
        client = _client(
            tmp_path,
            auth={"enabled": True, "api_key": "k2", "header": "X-Custom-Key"},
        )
        assert client.get("/v1/alerts", headers={"X-API-Key": "k2"}).status_code == 401
        assert client.get("/v1/alerts", headers={"X-Custom-Key": "k2"}).status_code == 200

    def test_enabled_without_api_key_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="auth.api_key"):
            create_app(
                config=_config(tmp_path, auth={"enabled": True, "api_key": ""}),
                dispatcher=AlertDispatcher([]),
            )


class TestPostgresWiring:
    def test_missing_dsn_env_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _config(tmp_path)
        cfg["storage"] = {"backend": "postgres", "dsn_env_var": "CASCADEGUARD_PG_DSN"}
        monkeypatch.delenv("CASCADEGUARD_PG_DSN", raising=False)
        with pytest.raises(RuntimeError, match="CASCADEGUARD_PG_DSN"):
            create_app(config=cfg, dispatcher=AlertDispatcher([]))

    def test_postgres_backend_builds_lazy_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Construction opens no connection (lazy, per-op); only the driver
        # import is validated here. Skipped when psycopg is not installed.
        pytest.importorskip("psycopg", reason="pip install cascadeguard[postgres]")
        from storage.postgres_impl import PostgresTraceRepository

        cfg = _config(tmp_path)
        cfg["storage"] = {"backend": "postgres", "dsn_env_var": "CASCADEGUARD_PG_DSN"}
        monkeypatch.setenv(
            "CASCADEGUARD_PG_DSN", "postgresql://u:p@localhost:1/db?connect_timeout=1"
        )
        app = create_app(config=cfg, dispatcher=AlertDispatcher([]))
        assert isinstance(app.state.repository, PostgresTraceRepository)


class TestListRunsView:
    def test_shape_and_ordering(self, client: TestClient) -> None:
        # Seed directly through the repository + run_graphs state.
        repo: InMemoryTraceRepository = client.app.state.repository  # type: ignore[attr-defined]
        repo.create_run("run-old", "template-a")
        time.sleep(0.01)
        repo.create_run("run-new", "template-b")

        graph = PipelineGraph("template-a")
        graph.add_step(Step(id="s1", kind="plan", output="plan"))
        graph.add_step(Step(id="s2", kind="reason", output="reason", predecessors=["s1"]))
        client.app.state.run_graphs["run-old"] = graph  # type: ignore[attr-defined]

        res = client.get("/v1/runs")
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 2
        assert [r["run_id"] for r in body["runs"]] == ["run-new", "run-old"]  # newest first

        expected_keys = {
            "run_id",
            "pipeline_template_id",
            "started_at",
            "status",
            "step_count",
            "alerted",
            "analyze_ms",
        }
        for row in body["runs"]:
            assert set(row.keys()) == expected_keys
        newest_row, oldest_row = body["runs"]
        # run-old has a seeded 2-step graph; run-new has none registered yet.
        assert newest_row["run_id"] == "run-new"
        assert newest_row["step_count"] == 0
        assert oldest_row["step_count"] == 2
        assert oldest_row["pipeline_template_id"] == "template-a"
        assert oldest_row["status"] == "running"
        assert oldest_row["alerted"] is False
        assert oldest_row["analyze_ms"] is None

    def test_empty_store(self, client: TestClient) -> None:
        body = client.get("/v1/runs").json()
        assert body == {"count": 0, "runs": []}


class TestJsonLogging:
    def test_request_timing_line_is_json(self, tmp_path: Path) -> None:
        # The app's StreamHandler binds stdout at first create_app (module
        # import), which pytest has already replaced — so capture by rebinding
        # the handler's stream rather than via capsys/capfd.
        logger = logging.getLogger(_LOGGER_NAME)
        handler = next(h for h in logger.handlers if isinstance(h.formatter, JsonFormatter))
        buffer = io.StringIO()
        original_stream = handler.stream
        handler.stream = buffer
        try:
            client = _client(tmp_path)
            client.get("/health")
        finally:
            handler.stream = original_stream
        lines = [ln for ln in buffer.getvalue().splitlines() if ln.startswith("{")]
        assert lines, "expected a JSON request-timing line"
        last = json.loads(lines[-1])
        assert last["message"] == "request"
        assert last["level"] == "INFO"
        assert last["logger"] == _LOGGER_NAME
        assert last["method"] == "GET"
        assert last["path"] == "/health"
        assert last["status"] == 200
        assert isinstance(last["ms"], (int, float))

    def test_no_duplicate_handlers_on_rebuild(self, tmp_path: Path) -> None:
        from api.main import _LOGGER_NAME, JsonFormatter

        _client(tmp_path)
        _client(tmp_path)
        handlers = logging.getLogger(_LOGGER_NAME).handlers
        json_handlers = [h for h in handlers if isinstance(h.formatter, JsonFormatter)]
        assert len(json_handlers) == 1  # create_app twice must not stack handlers
