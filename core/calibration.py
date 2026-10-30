"""Clean-trace calibration harness (README §4.7 — μ0/σ from labeled-clean runs).

``calibrate_from_clean_traces`` replays clean synthetic traces through the
engine to harvest the CHRS distribution under "no hallucination", then runs
the CUSUM calibration (Eq. 12 inputs). The engine passed in must be wired the
same way as the deployment (same embedder/entailment hooks), because CHRS is
a function of those hooks — calibrating on a different wiring produces μ0/σ
that don't match production CHRS.
"""

from __future__ import annotations

import warnings

from core.engine import CascadeGuardEngine
from eval.synthetic import make_paper_fig1_trace, make_random_dag


def harvest_clean_chrs(engine: CascadeGuardEngine, n_dags: int = 10) -> list[float]:
    """Analyze clean (corrupted=False) traces and collect every CHRS value."""
    values: list[float] = []
    traces = [make_paper_fig1_trace(corrupted=False)]
    traces += [make_random_dag(seed=seed, corrupted_node=None)[0] for seed in range(n_dags)]
    for graph in traces:
        result = engine.analyze_run(graph)
        values.extend(result.chrs.values())
    return values


def calibrate_from_clean_traces(
    engine: CascadeGuardEngine, n_dags: int = 10
) -> tuple[float, float, int]:
    """Calibrate μ0/σ from clean traces; returns (mu0, sigma, n_samples).

    Emits a UserWarning when |Δ| = δ/(2σ) exceeds 0.5 — outside that regime
    the Eq. 12 approximation degrades (AGENTS.md §3): h is overestimated and
    the detector effectively stops firing. The fix is lowering ``cusum.delta``
    in config, not touching the math.
    """
    values = harvest_clean_chrs(engine, n_dags)
    if len(values) < 30:
        raise ValueError(
            f"clean-trace harvest produced only {len(values)} CHRS samples; increase n_dags"
        )
    mu0, sigma = engine.calibrate(values)
    abs_delta = engine.cusum_params.delta / (2.0 * sigma)
    if abs_delta > 0.5:
        warnings.warn(
            f"|Δ| = δ/(2σ) = {abs_delta:.2f} > 0.5 — Eq. 12 regime degraded: h is "
            "overestimated and the detector may stop firing. Lower cusum.delta in config.",
            UserWarning,
            stacklevel=2,
        )
    return mu0, sigma, len(values)
