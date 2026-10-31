"""Async trace-ingestion worker (TODO Phase 11): Redis Streams consumer.

    python -m api.worker

Requires ``ingestion.mode: redis_stream`` in config (and a reachable Redis via
``CASCADEGUARD_REDIS_URL``) — the same wiring that enables ``POST /v1/ingest``.

Behavior: consumer group on the trace stream; step events are buffered per
run; on ``run_end`` the graph is rebuilt, the full analysis runs (same engine
as the API), results persist to the configured repository, alerts dispatch,
the run is marked completed/alerted, and the run's CUSUM detector is
garbage-collected (the worker path is where the V2-3 lifecycle actually ends —
the inline API path keeps the detector pooled for post-analysis blocking
decisions). Malformed payloads are acked + deleted with a log line so one bad
event can't wedge the stream; processing failures are left unacked for
redelivery. Scale by starting more workers with distinct CASCADEGUARD_CONSUMER
values (consumer-group semantics distribute messages).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from api.main import create_app, load_config
from api.routes.ingest import _persist  # noqa: SLF001 — same-package reuse
from api.streaming import StreamEventError, new_run_buffer, parse_event

log = logging.getLogger("cascadeguard.worker")


class TraceWorker:
    """Buffers stream events per run; analyzes on run_end. Testable without Redis."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.buffers: dict[str, dict[str, Any]] = {}

    def handle_event(self, event: dict[str, Any]) -> None:
        from core.graph.pipeline_graph import Document, PipelineGraph, Step

        run_id = event["run_id"]
        if event["type"] == "step":
            step_buf = self.buffers.setdefault(run_id, new_run_buffer(event.get("template_id")))
            step_buf["steps"].append(event["event"])
            return

        buf = self.buffers.pop(run_id) if run_id in self.buffers else None
        if buf is None or not buf["steps"]:
            log.warning("run_end for run %s with no buffered steps; skipping", run_id)
            return
        graph = PipelineGraph(buf["template_id"])
        for ev in buf["steps"]:
            docs = [
                Document(doc_id=d["doc_id"], content=d["content"])
                for d in ev.get("retrieved_docs", [])
            ]
            graph.add_step(
                Step(
                    id=ev["step_id"],
                    kind=ev["kind"],
                    output=ev.get("output", ""),
                    retrieved_docs=docs,
                    predecessors=ev.get("predecessors", []),
                    tool_invocation=ev.get("tool_invocation"),
                )
            )

        state = self.app.state
        if state.repository.get_run(run_id) is None:
            state.repository.create_run(run_id, buf["template_id"])
        state.run_graphs[run_id] = graph
        result = state.engine.analyze_run(graph, run_id=run_id)
        state.results[run_id] = result
        _persist(state, run_id, graph, result)
        state.dispatcher.dispatch(result.alerts)
        status = "alerted" if result.alerted else "completed"
        state.repository.complete_run(run_id, status)
        state.engine.detector_pool.complete(run_id)
        log.info("analyzed run %s: status=%s alerts=%d", run_id, status, len(result.alerts))

    def process_message(self, msg_id: Any, fields: dict[Any, Any]) -> bool:
        """Consume one stream message; True = ack + delete, False = redeliver."""
        try:
            raw = fields.get(b"payload") or fields.get("payload")
            if raw is None:
                # an envelope without a payload can never succeed — dead-letter
                # rather than redelivering forever
                raise StreamEventError("message has no payload field")
            self.handle_event(parse_event(raw))
            return True
        except StreamEventError as exc:
            log.error("dead-lettering message %s: %s", msg_id, exc)
            return True  # consumed: malformed payloads never wedge the stream
        except Exception:
            log.exception("message %s failed; left pending for redelivery", msg_id)
            return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = create_app(load_config())
    cfg = app.state.config.get("ingestion", {})
    client = getattr(app.state, "redis", None)
    if cfg.get("mode") != "redis_stream" or client is None:
        print(
            "worker requires ingestion.mode=redis_stream with a reachable Redis "
            "(CASCADEGUARD_REDIS_URL); see configs/default.yaml"
        )
        return 1

    stream = cfg.get("stream_name", "cascadeguard:trace_events")
    group = cfg.get("consumer_group", "cascadeguard-workers")
    consumer = os.environ.get("CASCADEGUARD_CONSUMER", f"worker-{os.getpid()}")
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    worker = TraceWorker(app)
    log.info("worker %s consuming stream=%s group=%s", consumer, stream, group)
    while True:
        batches = client.xreadgroup(group, consumer, {stream: ">"}, count=20, block=2000)
        for _stream_name, entries in batches or []:
            for msg_id, fields in entries:
                if worker.process_message(msg_id, fields):
                    client.xack(stream, group, msg_id)
                    client.xdel(stream, msg_id)


if __name__ == "__main__":
    raise SystemExit(main())
