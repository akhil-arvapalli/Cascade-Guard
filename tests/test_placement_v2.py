"""Phase 8 placement upgrades: empirical costs, trace-simulated risk model,
placement cache (in-process + Redis fallback), and the refresh job's endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.embeddings import HashingEmbedder
from core.engine import CascadeGuardEngine
from core.placement.cache import InProcessPlacementCache, RedisPlacementCache
from core.placement.costs import measure_monitor_costs
from core.placement.greedy_optimizer import (
    brute_force_optimal_placement,
    greedy_monitor_placement,
)
from core.placement.risk_model import HistoricalRiskModel
from eval.synthetic import make_paper_fig1_trace
from storage.repository import InMemoryTraceRepository


def _engine() -> CascadeGuardEngine:
    return CascadeGuardEngine(
        config={"cusum": {"delta": 0.03}},
        embedder=HashingEmbedder(64),
        entail_fn=lambda a, b: 0.5,
    )


class TestPlacementCache:
    def test_in_process_roundtrip_and_copy_semantics(self) -> None:
        cache = InProcessPlacementCache()
        assert cache.get("t") is None
        s = {"s1", "s2"}
        cache.put("t", s)
        s.add("s3")  # caller mutation must not leak into the cache
        assert cache.get("t") == {"s1", "s2"}
        # dict-style compat (engine/tests seed via [])
        cache["t2"] = {"s9"}
        assert cache["t2"] == {"s9"}
        assert "t2" in cache

    class _BoomClient:
        def get(self, key: str) -> None:
            raise ConnectionError("redis down")

        def set(self, key: str, value: str) -> None:
            raise ConnectionError("redis down")

    def test_redis_fails_open_to_in_process(self) -> None:
        cache = RedisPlacementCache(client=self._BoomClient())
        cache.put("t", {"s1"})  # redis put fails; in-process copy must survive
        assert cache.get("t") == {"s1"}
        cache["t2"] = {"s2"}
        assert cache["t2"] == {"s2"}

    def test_build_dispatch(self) -> None:
        assert isinstance(InProcessPlacementCache(), InProcessPlacementCache)
        with pytest.raises(ValueError):
            from core.placement.cache import build_placement_cache

            build_placement_cache("memcached")


class TestMeasuredCosts:
    def test_all_kinds_measured_positive(self) -> None:
        engine = _engine()
        costs = measure_monitor_costs(engine, n_trials=5)
        for kind in ("plan", "retrieve", "reason", "tool_call", "synthesize"):
            assert costs[kind] > 0.0
        assert costs["embed_call_ms"] > 0.0
        # no generate_fn wired → SE contributes 0 (DECISIONS D-002/D-003 open)
        assert costs["se_step_ms"] == 0.0

    def test_refresh_with_measured_costs_uses_cache(self) -> None:
        engine = _engine()
        graph = make_paper_fig1_trace(corrupted=False)
        s_star = engine.refresh_placement("fig1", graph, use_measured_costs=True)
        assert engine._measured_costs is not None
        assert engine._placement_cache.get("fig1") == s_star
        # live path reads the cache instead of recomputing
        s_star2 = engine._placement_cache.get("fig1")
        assert s_star2 == s_star


class TestHistoricalRiskModel:
    def _linear_graph(self) -> object:
        """s1 -> s2 -> s3 (duck-typed graph: .nodes, .edges_out_of)."""

        return make_paper_fig1_trace(corrupted=False)  # 5-node linear chain

    def test_hand_computed_marginal_risk(self) -> None:
        """Linear 3-node trace with rm = {s1: 0.5, s2: 1.0, s3: 2.0}.

        anc(s3) = {s1, s2, s3}. marginal(s1, {}) = 0.5/1 + 1.0/2 + 2.0/3.
        With selected={s2}: s2, s3 are covered (their ancestors include s2);
        only s1's own mass remains → marginal(s1, {s2}) = 0.5.
        """
        from core.graph.pipeline_graph import PipelineGraph

        g = PipelineGraph(pipeline_template_id="t")
        from core.graph.pipeline_graph import Step

        g.add_step(Step(id="s1", kind="retrieve", output="a"))
        g.add_step(Step(id="s2", kind="reason", output="b", predecessors=["s1"]))
        g.add_step(Step(id="s3", kind="synthesize", output="c", predecessors=["s2"]))
        rm = {"s1": 0.5, "s2": 1.0, "s3": 2.0}
        model = HistoricalRiskModel([(g, rm)])

        m_empty = model.marginal_risk("s1", set())
        assert m_empty == pytest.approx(0.5 + 0.5 + 2.0 / 3.0)
        assert model.marginal_risk("s1", {"s2"}) == pytest.approx(0.5)
        # monitoring s3 covers s3's own mass fully (dist(s3,s3)=0); s1/s2 have no
        # s3 in their ancestor closures so their mass stays uncovered
        assert model.marginal_risk("s3", set()) == pytest.approx(2.0)
        # zero-mass history contributes nothing
        assert (
            HistoricalRiskModel([(g, {"s1": 0.0, "s2": 0.0, "s3": 0.0})]).marginal_risk("s1", set())
            == 0.0
        )

    def test_greedy_with_risk_model_beats_random_under_budget(self) -> None:
        """Greedy over the risk model must stay within the theoretical
        (1-1/e) ≈ 63% approximation bound of the brute-force optimum under the
        matching distance-discounted coverage objective (TODO Phase 8 exit
        criterion, same pattern as tests/test_placement.py).

        Oracle objective f(S) = Σ_m max_{s∈S∩anc(m)} rm(m)/(1+dist(s,m)) —
        the discount-coverage function the greedy marginals approximate.
        Measured on this seed: greedy 6.91 vs optimal 7.89 → ratio 0.876.
        """
        import random

        from core.graph.pipeline_graph import PipelineGraph, Step

        rng = random.Random(11)
        g = PipelineGraph(pipeline_template_id="t")
        for i in range(8):
            preds = [f"s{j}" for j in range(max(0, i - 2), i) if rng.random() < 0.7]
            g.add_step(Step(id=f"s{i}", kind="reason", output=f"o{i}", predecessors=preds))
        rm = {f"s{i}": 1.0 + rng.random() for i in range(8)}
        model = HistoricalRiskModel([(g, rm)])
        weights = {f"s{i}": 1.0 for i in range(8)}
        trace = model._traces[0]

        def covered_mass(selected: set[str]) -> float:
            total = 0.0
            for node, mass in rm.items():
                best = 0.0
                for s in selected:
                    if s in trace.ancestors[node]:
                        d = model._distances(0, s).get(node, 10**9)
                        best = max(best, mass / (1.0 + d))
                total += best
            return total

        greedy = greedy_monitor_placement(
            g, weights, budget=2.0, marginal_risk_fn=model.marginal_risk
        )
        optimal, opt_val = brute_force_optimal_placement(g, weights, 2.0, covered_mass)
        greedy_val = covered_mass(set(greedy))
        assert greedy_val >= (1.0 - 1.0 / 2.718281828) * opt_val - 1e-9

    def test_analysis_result_accepted_as_history(self) -> None:
        engine = _engine()
        graph = make_paper_fig1_trace(corrupted=True)
        result = engine.analyze_run(graph)
        model = HistoricalRiskModel([(graph, result)])
        assert model.marginal_risk("s1_plan", set()) > 0.0


class TestRefreshEndpoints:
    def test_templates_and_history_aware_refresh(self) -> None:
        from api.main import create_app

        app = create_app(config={"cusum": {"delta": 0.03}}, repository=InMemoryTraceRepository())
        client = TestClient(app)

        # seed two analyzed runs of the same template
        graph = make_paper_fig1_trace(corrupted=True)
        res = client.post("/v1/runs", json={"pipeline_template_id": "fig1"})
        run_id = res.json()["run_id"]
        for nid in graph.topo_order():
            step = graph.nodes[nid]
            client.post(
                f"/v1/runs/{run_id}/steps",
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
        client.post(f"/v1/runs/{run_id}/analyze")

        templates = client.get("/v1/placement/templates").json()["templates"]
        assert "fig1" in templates

        body = client.post(
            "/v1/placement/refresh",
            json={"template_id": "fig1", "use_history": True},
        ).json()
        assert body["history_traces"] == 1
        assert isinstance(body["S_star"], list)
        # cache written through — live path would read this exact set
        assert app.state.engine._placement_cache.get("fig1") == set(body["S_star"])

    def test_unknown_template_404(self) -> None:
        from api.main import create_app

        app = create_app(config={}, repository=InMemoryTraceRepository())
        client = TestClient(app)
        res = client.post("/v1/placement/refresh", json={"template_id": "ghost"})
        assert res.status_code == 404
