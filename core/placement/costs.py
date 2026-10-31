"""Empirical per-step-kind monitoring costs (TODO Phase 8).

The placeholder ratios in configs/default.yaml (LLM call ~= 1.0, embed ~= 0.05)
are replaced by MEASURED per-call latencies of the actual monitor primitives on
the deployed engine hooks:

- cheap fidelity: one embedding call per step (H, Eq. 3-4) plus one entailment
  call per incoming edge (α, Track B surrogate)
- full fidelity: adds semantic entropy — K generate calls + O(K²) entailment
  clustering calls (0 while generate_fn is unwired, per DECISIONS D-002/D-003)

Units are milliseconds measured live; only RELATIVE ratios reach the greedy
optimizer (Δrisk/cost), so numbers are meaningful only for the same wiring they
were measured on. configs/default.yaml ratios remain the documented fallback
and the deterministic path for tests (measure_costs=false).
"""

from __future__ import annotations

import time
from typing import Any


def measure_monitor_costs(engine: Any, n_trials: int = 30) -> dict[str, float]:
    """Measure monitor-primitive latencies; returns per-kind monitor costs (ms)."""
    embedder = engine.embedder
    entail_fn = engine.entail_fn
    generate_fn = engine.generate_fn
    sample = "The Eiffel Tower is a landmark in Paris, France, built in 1889."
    other = "Photosynthesis converts sunlight into chemical energy in plants."

    t0 = time.perf_counter()
    for _ in range(n_trials):
        embedder(sample)
    embed_ms = (time.perf_counter() - t0) * 1000.0 / n_trials

    entail_ms = 0.0
    if entail_fn is not None:
        t0 = time.perf_counter()
        for _ in range(n_trials):
            entail_fn(sample, other)
        entail_ms = (time.perf_counter() - t0) * 1000.0 / n_trials

    se_ms = 0.0
    if generate_fn is not None:
        k = int(engine.config.get("semantic_entropy", {}).get("K", 10))
        t0 = time.perf_counter()
        for _ in range(n_trials):
            generate_fn(sample)
        gen_ms = (time.perf_counter() - t0) * 1000.0 / n_trials
        se_ms = k * gen_ms + k * k * entail_ms  # K samples + O(K²) cluster calls

    cheap_ms = embed_ms + entail_ms  # one H + one α (≈1 predecessor) per step
    full_ms = cheap_ms + se_ms
    return {
        "embed_call_ms": round(embed_ms, 4),
        "entail_call_ms": round(entail_ms, 4),
        "se_step_ms": round(se_ms, 4),
        # kind-level monitor costs consumed by greedy_monitor_placement
        "plan": cheap_ms,
        "retrieve": cheap_ms,
        "reason": cheap_ms,
        "tool_call": full_ms,
        "synthesize": full_ms,
    }
