"""In-band blocking decision endpoint (Phase 12).

Covers the documented semantics: decisions are ephemeral CUSUM projections
(the pooled per-run S_n is read but never advanced), SE never runs on the
decision path, and mode=notify_only never blocks.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app
from storage.repository import InMemoryTraceRepository

EIFFEL_DOC = "The Eiffel Tower is a landmark in Paris, France, built in 1889."
FABRICATED = "Brandenburg Gate overlooks Berlin central plaza fountains nearby."

# Test-local CUSUM params (documented placeholders elsewhere): small enough that
# two fabricated steps accumulate past the control limit in test_cusum_state_blocks.
CUSUM_CFG = {"mu0": 0.02, "sigma": 0.03, "delta": 0.02}


def _client(config_overrides: dict) -> TestClient:
    config = {"cusum": CUSUM_CFG, **config_overrides}
    app = create_app(config=config, repository=InMemoryTraceRepository())
    return TestClient(app)


def _seed_echo_run(client: TestClient, steps: list[tuple[str, str, list[str]]]) -> str:
    """steps: (step_id, output, predecessors). Every step retrieves the same doc."""
    res = client.post("/v1/runs", json={"pipeline_template_id": "decision_test"})
    run_id = res.json()["run_id"]
    for step_id, output, preds in steps:
        client.post(
            f"/v1/runs/{run_id}/steps",
            json={
                "step_id": step_id,
                "kind": "reason" if preds else "retrieve",
                "output": output,
                "retrieved_docs": [{"doc_id": f"{step_id}_d0", "content": EIFFEL_DOC}],
                "predecessors": preds,
            },
        )
    return run_id


class TestDecisionSemantics:
    def test_notify_only_never_blocks(self) -> None:
        client = _client({"blocking": {"mode": "notify_only", "threshold_tau": None}})
        run_id = _seed_echo_run(client, [("s1", EIFFEL_DOC, []), ("s2", EIFFEL_DOC, ["s1"])])
        res = client.post(
            f"/v1/decision/{run_id}/s3",
            json={
                "kind": "synthesize",
                "output": FABRICATED,
                "retrieved_docs": [{"doc_id": "s3_d0", "content": EIFFEL_DOC}],
                "predecessors": ["s2"],
            },
        )
        body = res.json()
        assert res.status_code == 200
        assert body["mode"] == "notify_only"
        assert body["block"] is False
        assert body["H"] is not None and body["H"] > 0.4  # fabricated output scores high...

    def test_block_mode_with_tau_gates_fabricated_step(self) -> None:
        client = _client({"blocking": {"mode": "block", "threshold_tau": 0.3}})
        run_id = _seed_echo_run(client, [("s1", EIFFEL_DOC, []), ("s2", EIFFEL_DOC, ["s1"])])
        res = client.post(
            f"/v1/decision/{run_id}/s3",
            json={
                "kind": "synthesize",
                "output": FABRICATED,
                "retrieved_docs": [{"doc_id": "s3_d0", "content": EIFFEL_DOC}],
                "predecessors": ["s2"],
            },
        )
        body = res.json()
        assert body["block"] is True
        assert "chrs_threshold" in body["reasons"]
        # identical-echo candidates score CHRS ~0 and pass the same gate
        res_clean = client.post(
            f"/v1/decision/{run_id}/s4",
            json={
                "kind": "synthesize",
                "output": EIFFEL_DOC,
                "retrieved_docs": [{"doc_id": "x", "content": EIFFEL_DOC}],
                "predecessors": ["s2"],
            },
        )
        assert res_clean.json()["block"] is False

    def test_decisions_do_not_advance_pooled_s_n(self) -> None:
        client = _client({"blocking": {"mode": "block", "threshold_tau": None}})
        run_id = _seed_echo_run(client, [("s1", EIFFEL_DOC, [])])
        first = client.post(
            f"/v1/decision/{run_id}/s2",
            json={"kind": "reason", "output": FABRICATED, "predecessors": ["s1"]},
        ).json()
        second = client.post(
            f"/v1/decision/{run_id}/s3",
            json={"kind": "reason", "output": FABRICATED, "predecessors": ["s1"]},
        ).json()
        # ephemeral projection: the run's accumulated S_n is unchanged across decisions
        assert first["s_n_current"] == second["s_n_current"]

    def test_cusum_state_blocks_after_accumulated_run(self) -> None:
        client = _client({"blocking": {"mode": "block", "threshold_tau": None}})
        run_id = _seed_echo_run(
            client,
            [
                ("s1", EIFFEL_DOC, []),
                ("s2", FABRICATED, ["s1"]),
                ("s3", "Berlin central plaza Brandenburg fountains overlook gate.", ["s2"]),
            ],
        )
        # accumulate S_n via the authoritative analysis pass (may or may not alert)
        client.post(f"/v1/runs/{run_id}/analyze")
        res = client.post(
            f"/v1/decision/{run_id}/s4",
            json={"kind": "synthesize", "output": FABRICATED, "predecessors": ["s3"]},
        )
        body = res.json()
        assert body["s_n_current"] > 0.0, "analysis must have accumulated S_n"
        assert body["block"] is True
        assert "cusum_control_limit" in body["reasons"]


class TestDecisionValidation:
    def test_unknown_run_404(self) -> None:
        client = _client({})
        res = client.post("/v1/decision/nope/s1", json={"output": "x"})
        assert res.status_code == 404

    def test_existing_step_409(self) -> None:
        client = _client({})
        run_id = _seed_echo_run(client, [("s1", EIFFEL_DOC, [])])
        res = client.post(f"/v1/decision/{run_id}/s1", json={"output": "x"})
        assert res.status_code == 409

    def test_unknown_predecessor_400(self) -> None:
        client = _client({})
        run_id = _seed_echo_run(client, [("s1", EIFFEL_DOC, [])])
        res = client.post(
            f"/v1/decision/{run_id}/s2", json={"output": "x", "predecessors": ["ghost"]}
        )
        assert res.status_code == 400
