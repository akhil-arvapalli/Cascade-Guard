"""Calibration persistence + recalibration pool semantics.

Covers the false-alarm root cause found in the 2026-10-30 smoke test: pooled
CusumDetectors snapshot κ/h at construction, so recalibrating without clearing
the pool leaves stale (placeholder) limits in force.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.calibration import calibrate_from_clean_traces, harvest_clean_chrs
from core.detection.cusum import CusumDetectorPool, CusumParams
from core.embeddings import HashingEmbedder
from core.engine import CascadeGuardEngine
from eval.synthetic import make_paper_fig1_trace, make_random_dag
from storage.calibration_store import CalibrationStore
from storage.repository import InMemoryTraceRepository


def _engine() -> CascadeGuardEngine:
    # delta=0.03 mirrors configs/default.yaml (in-regime: |Δ|=δ/(2σ)≈0.25 for the
    # calibrated σ≈0.059; δ=0.1 was tuned for the pre-calibration placeholder σ).
    return CascadeGuardEngine(
        config={"cusum": {"delta": 0.03}},
        embedder=HashingEmbedder(64),
        entail_fn=lambda a, b: 0.5,
    )


class TestPoolClear:
    def test_clear_drops_all_detectors(self) -> None:
        pool = CusumDetectorPool()
        params = CusumParams(mu0=0.3, sigma=0.1, delta=0.05)
        pool.get_or_create("run-1", params)
        pool.get_or_create("run-2", params)
        pool.clear()
        # a cleared pool constructs a FRESH detector (S back to 0)
        fresh = pool.get_or_create("run-1", params)
        assert fresh.S == 0.0

    def test_engine_calibrate_clears_pool(self) -> None:
        engine = _engine()
        d1 = engine.detector_pool.get_or_create("run-x", engine.cusum_params)
        d1.update(0.9)  # accumulate state under placeholder limits
        assert d1.S > 0
        mu0, sigma = engine.calibrate([0.05 + i * 0.001 for i in range(40)])
        assert sigma > 0
        d2 = engine.detector_pool.get_or_create("run-x", engine.cusum_params)
        assert d2 is not d1
        assert d2.S == 0.0
        # new limits are rebuilt from the calibrated params, not the old ones
        assert d2.kappa == pytest.approx(mu0 + engine.cusum_params.delta / 2)


class TestCalibrationStore:
    def test_roundtrip(self, tmp_path) -> None:
        store = CalibrationStore(tmp_path / "cal.json")
        assert store.load() is None  # missing file reads as un-calibrated
        store.save(0.12, 0.03, extra={"n_samples": 42})
        assert store.load() == {"mu0": 0.12, "sigma": 0.03}

    def test_corrupt_or_invalid_file_returns_none(self, tmp_path) -> None:
        path = tmp_path / "cal.json"
        path.write_text("{not json", encoding="utf-8")
        assert CalibrationStore(path).load() is None
        path.write_text('{"mu0": "x", "sigma": 1}', encoding="utf-8")
        assert CalibrationStore(path).load() is None
        path.write_text('{"mu0": 0.1, "sigma": 0}', encoding="utf-8")
        assert CalibrationStore(path).load() is None


class TestCleanTraceCalibration:
    def test_harvest_yields_30_plus_in_range_samples(self) -> None:
        engine = _engine()
        values = harvest_clean_chrs(engine, n_dags=6)
        assert len(values) >= 30
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_calibrate_from_clean_traces_sane_params(self) -> None:
        engine = _engine()
        mu0, sigma, n = calibrate_from_clean_traces(engine, n_dags=6)
        assert n >= 30
        assert 0.0 < mu0 < 1.0
        assert sigma > 0

    def test_calibrated_detector_false_alarm_and_detection_rates(self) -> None:
        """Regression for the smoke-test finding: with calibrated μ0/σ the clean
        population must fire rarely, and a genuinely cascading trace must fire.

        Known tail behavior (documented, not asserted to zero): the fig1-clean
        trace sits in the CHRS tail — the hashing placeholder embedder inflates
        H on faithful paraphrase echoes and echo-α is high, so H_cascade
        accumulates with depth (AGENTS.md §3, D-003/D-005). Resolves with V2-1.
        """
        engine = _engine()
        calibrate_from_clean_traces(engine, n_dags=6)
        # fresh, out-of-sample clean traces (calibration used seeds 0-5 + fig1)
        fires = 0
        n = 10
        for seed in range(100, 100 + n):
            graph = make_random_dag(seed=seed, corrupted_node=None)[0]
            if engine.analyze_run(graph).alerted:
                fires += 1
        assert fires <= 1, f"clean-population false-alarm rate too high: {fires}/{n}"
        # detection side: a real cascade fires
        assert engine.analyze_run(make_paper_fig1_trace(corrupted=True)).alerted


class TestCalibrationEndpoint:
    def test_endpoint_persists_to_store_and_repo(self, tmp_path) -> None:
        from api.main import create_app

        store_path = tmp_path / "cal.json"
        config = {"cusum": {"calibration_store_path": str(store_path)}}
        app = create_app(config=config, repository=InMemoryTraceRepository())
        client = TestClient(app)

        res = client.post(
            "/v1/calibration/cusum",
            json={"clean_chrs_values": [0.10 + i * 0.001 for i in range(40)]},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["sigma"] > 0

        loaded = CalibrationStore(store_path).load()
        assert loaded is not None
        assert loaded["mu0"] == pytest.approx(body["mu0"])
        assert loaded["sigma"] == pytest.approx(body["sigma"])
        assert app.state.repository.get_calibration() is not None

    def test_endpoint_rejects_insufficient_samples(self) -> None:
        from api.main import create_app

        app = create_app(config={}, repository=InMemoryTraceRepository())
        client = TestClient(app)
        res = client.post("/v1/calibration/cusum", json={"clean_chrs_values": [0.1] * 5})
        assert res.status_code == 422
        assert ">=30" in res.json()["detail"]
