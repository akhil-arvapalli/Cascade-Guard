"""CUSUM Detector — Eq. 11 (statistic) and Eq. 12 (ARL0 control limit).

AGENTS.md §2.5: detector state S_n is PER-RUN (keyed by run_id), never global —
use CusumDetectorPool, one instance per active pipeline execution.
μ0/σ are never hardcoded: they come from `calibrate()` on labeled-clean traces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class CusumConfigError(ValueError):
    pass


@dataclass
class CusumParams:
    mu0: float  # baseline CHRS mean under "no hallucination" (from calibrate())
    sigma: float  # baseline CHRS stdev (from calibrate())
    delta: float  # minimum detectable CHRS shift
    arl0_target: float = 500  # → <0.2% false alarms per execution


def arl0_approximation(delta_shift: float, h: float) -> float:
    """Eq. 12: ARL0 ≈ (e^{−2Δh} + 2Δh − 1) / (2Δ²), Δ = (μ0−κ)/σ.

    With κ = μ0+δ/2 the standardized gap Δ is negative, making e^{−2Δh}
    grow with h — ARL0 is monotonically increasing in h as required.
    """
    if abs(delta_shift) < 1e-9:
        raise CusumConfigError("degenerate CUSUM: |Δ|≈0 (check delta vs sigma)")
    return (math.exp(-2 * delta_shift * h) + 2 * delta_shift * h - 1) / (2 * delta_shift**2)


def solve_control_limit(delta_shift: float, arl0_target: float) -> float:
    """Numerically invert Eq. 12 for h given the ARL0 target (bisection).

    arl0_approximation is monotone increasing in h>0 with f(0⁺)=0, so bisection
    converges; the closed form doesn't invert cleanly (README §4.7).
    """
    if arl0_target <= 1:
        raise CusumConfigError("arl0_target must exceed 1")
    lo, hi = 1e-6, 1.0
    # Exponential growth guard: cap the upper bracket before exp() can overflow.
    while arl0_approximation(delta_shift, hi) < arl0_target:
        hi *= 2.0
        if -2 * delta_shift * hi > 650:
            raise CusumConfigError("ARL0 target unreachable without exp overflow")
    for _ in range(200):
        mid = (lo + hi) / 2
        if arl0_approximation(delta_shift, mid) < arl0_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class CusumDetector:
    """One-sided CUSUM over CHRS values for ONE pipeline run (Eq. 11)."""

    def __init__(self, params: CusumParams) -> None:
        if params.sigma <= 0 or params.delta <= 0:
            raise CusumConfigError("sigma and delta must be positive")
        self.params = params
        self.kappa = params.mu0 + params.delta / 2  # Eq. 11 reference value κ
        delta_shift = (params.mu0 - self.kappa) / params.sigma  # Δ = (μ0−κ)/σ (negative)
        # Eq. 12 solves h in SIGMA-STANDARDIZED units; S_n accumulates raw CHRS,
        # so rescale before comparing (S_n > h fires).
        self.h = solve_control_limit(delta_shift, params.arl0_target) * params.sigma
        self.S: float = 0.0
        self.alerted: bool = False

    def update(self, chrs_value: float) -> bool:
        """Eq. 11 update; returns True on the step this detector first alerts."""
        self.S = max(0.0, self.S + chrs_value - self.kappa)
        fired = self.S > self.h
        if fired:
            self.alerted = True
        return fired

    @property
    def state(self) -> dict[str, float]:
        return {"s_n": self.S, "kappa": self.kappa, "h_control_limit": self.h}


def calibrate(clean_chrs_values: list[float]) -> tuple[float, float]:
    """Empirical μ0, σ from labeled-clean traces — never guess these."""
    if len(clean_chrs_values) < 30:
        raise CusumConfigError(
            f"calibration needs >=30 clean CHRS samples, got {len(clean_chrs_values)}"
        )
    arr = np.asarray(clean_chrs_values, dtype=float)
    sigma = float(arr.std(ddof=1))
    if sigma <= 0:
        raise CusumConfigError("clean-trace CHRS has zero variance; cannot calibrate σ")
    return float(arr.mean()), sigma


class CusumDetectorPool:
    """Per-run detector registry (AGENTS.md §2.5). State never crosses runs."""

    def __init__(self) -> None:
        self._detectors: dict[str, CusumDetector] = {}

    def get_or_create(self, run_id: str, params: CusumParams) -> CusumDetector:
        if run_id not in self._detectors:
            self._detectors[run_id] = CusumDetector(params)
        return self._detectors[run_id]

    def restore(self, run_id: str, detector: CusumDetector) -> None:
        self._detectors[run_id] = detector

    def complete(self, run_id: str) -> None:
        """Garbage-collect state on pipeline completion."""
        self._detectors.pop(run_id, None)

    def clear(self) -> None:
        """Drop every pooled detector.

        Required after recalibration: CusumDetector snapshots κ and the control
        limit h at construction, so pooled detectors would keep operating on
        stale limits (root cause of the false-alarm-on-clean-trace smoke-test
        finding, 2026-10-30).
        """
        self._detectors.clear()
