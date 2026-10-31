"""Redis Streams ingestion + async worker (TODO Phase 11) and lifecycle status.

The stream contract (api/streaming.py) and the worker flow are tested with
in-memory fakes — no live Redis required; the DSN/Redis integration path is
exercised only when a real Redis is configured.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.streaming import StreamEventError, parse_event
from api.worker import TraceWorker
from eval.synthetic import make_paper_fig1_trace
from storage.repository import InMemoryTraceRepository

EIFFEL_DOC = "The Eiffel Tower is a landmark in Paris, France, built in 1889."


def _client() -> TestClient:
    app = create_app(config={"cusum": {"delta": 0.03}}, repository=InMemoryTraceRepository())
    return TestClient(app)


def _trace_events(run_id: str, graph, template_id: str) -> list[dict]:
    events = []
    for nid in graph.topo_order():
        step = graph.nodes[nid]
        events.append(
            {
                "type": "step",
                "run_id": run_id,
                "template_id": template_id,
                "event": {
                    "step_id": nid,
                    "kind": step.kind,
                    "output": step.output,
                    "retrieved_docs": [
                        {"doc_id": d.doc_id, "content": d.content} for d in step.retrieved_docs
                    ],
                    "predecessors": list(step.predecessors),
                },
            }
        )
    events.append({"type": "run_end", "run_id": run_id})
    return events


class TestParseEvent:
    def test_valid_step_and_run_end(self) -> None:
        step = {
            "type": "step",
            "run_id": "r1",
            "template_id": "t",
            "event": {"step_id": "s1", "kind": "retrieve", "output": "x"},
        }
        assert parse_event(json.dumps(step)) == step
        assert parse_event(json.dumps({"type": "run_end", "run_id": "r1"}))["type"] == "run_end"

    def test_malformed_payloads_raise(self) -> None:
        bad = [
            b"not json",
            b"[]",
            b'{"type": "other", "run_id": "r"}',
            b'{"type": "step"}',
            b'{"type": "step", "run_id": "", "event": {"step_id": "s"}}',
            b'{"type": "step", "run_id": "r", "event": {}}',
            b'{"type": "step", "run_id": "r", "event": {"step_id": "s", "kind": "bogus"}}',
            b'{"type": "step", "run_id": "r", "event": {"step_id": "s", "retrieved_docs": [{}]}}',
        ]
        for raw in bad:
            with pytest.raises(StreamEventError):
                parse_event(raw)


class _FakeRedis:
    def __init__(self) -> None:
        self.added: list[tuple[str, dict]] = []

    def xadd(self, stream: str, fields: dict) -> str:
        self.added.append((stream, fields))
        return f"1-{len(self.added)}"

    def ping(self) -> bool:
        return True


class TestIngestEndpoint:
    def test_disabled_returns_503(self) -> None:
        client = _client()
        res = client.post("/v1/ingest", json={"events": [{"type": "run_end", "run_id": "r"}]})
        assert res.status_code == 503
        assert "disabled" in res.json()["detail"]

    def test_publishes_valid_events(self) -> None:
        app = create_app(config={"cusum": {"delta": 0.03}}, repository=InMemoryTraceRepository())
        fake = _FakeRedis()
        app.state.redis = fake  # injected async wiring (as main.py does for redis_stream)
        client = TestClient(app)
        events = _trace_events("r-pub", make_paper_fig1_trace(corrupted=False), "fig1")
        res = client.post("/v1/ingest", json={"events": events})
        assert res.status_code == 200
        assert res.json() == {"published": len(events), "stream": "cascadeguard:trace_events"}
        assert len(fake.added) == len(events)

    def test_contract_violation_is_422(self) -> None:
        app = create_app(config={"cusum": {"delta": 0.03}}, repository=InMemoryTraceRepository())
        app.state.redis = _FakeRedis()
        client = TestClient(app)
        res = client.post(
            "/v1/ingest",
            json={"events": [{"type": "step", "run_id": "r", "event": {"output": "x"}}]},
        )
        assert res.status_code == 422
        assert "step_id" in res.json()["detail"]


class TestWorkerFlow:
    def _worker(self) -> tuple[TestClient, TraceWorker]:
        app = create_app(config={"cusum": {"delta": 0.03}}, repository=InMemoryTraceRepository())
        return TestClient(app), TraceWorker(app)

    def test_clean_run_completes_without_alert(self) -> None:
        """Uses a shallow 2-step echo trace: the fig1-clean trace sits in the
        documented CHRS tail (AGENTS.md §3) and is expected to alert."""
        client, worker = self._worker()
        events = [
            {
                "type": "step",
                "run_id": "r-clean",
                "template_id": "rag_clean",
                "event": {
                    "step_id": "s1",
                    "kind": "retrieve",
                    "output": EIFFEL_DOC,
                    "retrieved_docs": [{"doc_id": "d1", "content": EIFFEL_DOC}],
                },
            },
            {
                "type": "step",
                "run_id": "r-clean",
                "template_id": "rag_clean",
                "event": {
                    "step_id": "s2",
                    "kind": "synthesize",
                    "output": EIFFEL_DOC,
                    "predecessors": ["s1"],
                },
            },
            {"type": "run_end", "run_id": "r-clean"},
        ]
        for ev in events:
            worker.handle_event(ev)
        state = client.app.state
        assert "r-clean" in state.results
        assert state.repository.get_run("r-clean")["status"] == "completed"
        assert not state.results["r-clean"].alerted
        # worker path GCs the run's CUSUM detector (V2-3 lifecycle end)
        assert "r-clean" not in state.engine.detector_pool._detectors

    def test_corrupt_cascade_alerts_and_marks_alerted(self) -> None:
        client, worker = self._worker()
        for ev in _trace_events("r-casc", make_paper_fig1_trace(corrupted=True), "fig1"):
            worker.handle_event(ev)
        state = client.app.state
        assert state.repository.get_run("r-casc")["status"] == "alerted"
        assert state.results["r-casc"].alerted
        assert state.repository.list_alerts("r-casc")

    def test_run_end_without_steps_is_skipped(self) -> None:
        _client, worker = self._worker()
        worker.handle_event({"type": "run_end", "run_id": "ghost"})
        assert "ghost" not in worker.buffers

    def test_malformed_message_dead_letters_not_retried(self) -> None:
        _client, worker = self._worker()
        assert worker.process_message("1-1", {b"payload": b"not json"}) is True
        # unhandled exceptions (engine failures) stay pending for redelivery
        assert worker.process_message("1-2", {}) is True  # missing payload -> dead letter


class TestInlineLifecycle:
    def test_analyze_marks_run_completed_or_alerted(self) -> None:
        client = _client()
        state = client.app.state
        # clean run -> completed
        r1 = client.post("/v1/runs", json={"pipeline_template_id": "clean"}).json()["run_id"]
        client.post(
            f"/v1/runs/{r1}/steps",
            json={
                "step_id": "s1",
                "kind": "retrieve",
                "output": EIFFEL_DOC,
                "retrieved_docs": [{"doc_id": "d1", "content": EIFFEL_DOC}],
            },
        )
        client.post(f"/v1/runs/{r1}/analyze")
        assert state.repository.get_run(r1)["status"] == "completed"
        # cascading run -> alerted
        r2 = client.post("/v1/runs", json={"pipeline_template_id": "fig1"}).json()["run_id"]
        graph = make_paper_fig1_trace(corrupted=True)
        for nid in graph.topo_order():
            step = graph.nodes[nid]
            client.post(
                f"/v1/runs/{r2}/steps",
                json={
                    "step_id": nid,
                    "kind": step.kind,
                    "output": step.output,
                    "retrieved_docs": [
                        {"doc_id": d.doc_id, "content": d.content} for d in step.retrieved_docs
                    ],
                    "predecessors": list(step.predecessors),
                },
            )
        client.post(f"/v1/runs/{r2}/analyze")
        assert state.repository.get_run(r2)["status"] == "alerted"
