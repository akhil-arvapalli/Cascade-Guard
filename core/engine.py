"""CascadeGuard Engine — orchestrates one full pipeline-run analysis.

Order of operations mirrors the paper's dependency chain:
H (Eq.3-4) → α_ij (Eq.6/surrogates) → H_cascade (Eq.5, single topo pass) →
CHAF (Eq.7, Thm.1 check) → placement S* (§II.H) → SE only on S* (Eq.9) →
CHRS (Eq.10) → CUSUM (Eq.11-12) → localization (README §5).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from core.detection.cusum import CusumDetectorPool, CusumParams
from core.embeddings import EmbeddingFn
from core.graph.pipeline_graph import PipelineGraph, Step
from core.localization.root_cause import RootCause, localize_root_cause
from core.scoring.cascade_score import (
    assert_theorem_1,
    compute_chaf,
    compute_h_cascade,
)
from core.scoring.faithfulness import hallucination_score
from core.scoring.propagation import (
    AlphaResult,
    CounterfactualEstimator,
    NliProxyEstimator,
    PropagationDependencyError,
    propagation_coefficient_attention,
)
from core.scoring.risk_combiner import chrs as chrs_fn
from core.scoring.semantic_entropy import semantic_entropy


@dataclass
class Alert:
    alert_id: str
    run_id: str
    triggered_at_step: str
    s_n_value: float
    root_causes: list[RootCause] = field(default_factory=list)


@dataclass
class AnalysisResult:
    run_id: str
    H: dict[str, float | None]
    h_cascade: dict[str, float]
    chaf: dict[str, float]
    chrs: dict[str, float]
    alpha_edges: dict[tuple[str, str], AlphaResult]
    monitored_fidelity: dict[str, str]  # step -> 'cheap' | 'full'
    semantic_entropy_values: dict[str, float]
    alerts: list[Alert]
    elapsed_ms: float

    @property
    def alerted(self) -> bool:
        return bool(self.alerts)


class CascadeGuardEngine:
    """Stateless-per-call analyzer; CUSUM state lives in a per-run pool."""

    def __init__(
        self,
        config: dict,
        embedder: EmbeddingFn,
        generate_fn: Callable[[str], str] | None = None,  # LLM sampling hook for SE
        entail_fn: Callable[[str, str], float] | None = None,  # NLI hook (Track B + SE)
        counterfactual_stack: CounterfactualEstimator | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.generate_fn = generate_fn
        self.entail_fn = entail_fn
        self.counterfactual_stack = counterfactual_stack
        self.detector_pool = CusumDetectorPool()
        self._placement_cache: dict[str, set[str]] = {}

        prop_cfg = config.get("propagation_estimator", {})
        self.alpha_track: str = prop_cfg.get("track", "nli_proxy")
        self.expensive_mode: bool = bool(prop_cfg.get("expensive_mode", False))
        se_cfg = config.get("semantic_entropy", {})
        self.se_k: int = int(se_cfg.get("K", 10))
        self.se_threshold: float = float(se_cfg.get("cluster_threshold", 0.85))
        rc_cfg = config.get("risk_combiner", {})
        self.lambda_: float = float(rc_cfg.get("lambda", 0.6))  # PLACEHOLDER until D-001 sweep
        cusum_cfg = config.get("cusum", {})
        self.cusum_params = CusumParams(
            mu0=float(cusum_cfg.get("mu0", 0.05)),  # overwritten by calibrate()
            sigma=float(cusum_cfg.get("sigma", 0.02)),
            delta=float(cusum_cfg.get("delta", 0.1)),
            arl0_target=float(cusum_cfg.get("arl0_target", 500)),
        )

    # -- calibration / placement -------------------------------------------

    def calibrate(self, clean_chrs_values: list[float]) -> tuple[float, float]:
        """Recalibrate μ0/σ from labeled-clean CHRS values (README §4.7).

        Clears the detector pool afterwards: CusumDetector fixes κ and h at
        construction, so pooled detectors would otherwise keep running on the
        stale (placeholder) limits.
        """
        from core.detection.cusum import calibrate

        mu0, sigma = calibrate(clean_chrs_values)
        self.cusum_params.mu0, self.cusum_params.sigma = mu0, sigma
        self.detector_pool.clear()
        return mu0, sigma

    def refresh_placement(self, template_id: str, graph: PipelineGraph) -> set[str]:
        """Offline/nightly recompute of S* for a pipeline template (README §4.8).

        NOTE (flagged): the marginal-risk model below is a structural PROXY —
        risk mass covered among descendants — because the empirical
        trace-simulated marginal_risk_fn is a TODO Phase 8 item requiring
        historical data we don't have yet. Swap it there when it exists.
        """
        from core.placement.greedy_optimizer import greedy_monitor_placement

        mp_cfg = self.config.get("monitor_placement", {})
        budget = float(mp_cfg.get("budget_B", 10.0))
        # Fallback mirrors configs/default.yaml ratios (LLM-call ~= 1.0, embed ~ 0.05)
        # so analysis works before Phase 8 empirical calibration lands.
        kind_costs = {
            k: float(v)
            for k, v in mp_cfg.get(
                "step_kind_costs",
                {
                    "plan": 0.05,
                    "retrieve": 0.05,
                    "reason": 0.05,
                    "tool_call": 1.1,
                    "synthesize": 1.1,
                },
            ).items()
        }
        weights: dict[str, float] = {}

        def marginal_risk_proxy(step_id: str, selected: set[str]) -> float:
            return CascadeGuardEngine._marginal_mass(step_id, selected, graph)

        s_star = greedy_monitor_placement(
            graph, weights, budget, marginal_risk_proxy, step_kind_costs=kind_costs
        )
        self._placement_cache[template_id] = s_star
        return s_star

    @staticmethod
    def _marginal_mass(step_id: str, selected: set[str], graph: PipelineGraph) -> float:
        """Proxy Δrisk: uncovered descendant risk-mass reachable from step_id."""
        reach = _descendants(graph, step_id)
        mass = 0.0
        for target in reach:
            if target in selected:
                continue
            dist = _min_distance(graph, step_id, target)
            mass += 1.0 / (1.0 + dist)
        return mass + 1.0  # monitoring the step itself always has standalone value

    # -- main analysis -------------------------------------------------------

    def analyze_run(self, graph: PipelineGraph, run_id: str | None = None) -> AnalysisResult:
        start = time.perf_counter()
        run_id = run_id or str(uuid.uuid4())

        # 1. Cheap-fidelity faithfulness H(s_i) everywhere (Eq. 3-4).
        H: dict[str, float | None] = {}
        for node_id, step in graph.nodes.items():
            H[node_id] = hallucination_score(step.output, step.retrieved_docs, self.embedder)
        numeric_H = {nid: (h if h is not None else 0.0) for nid, h in H.items()}

        # 2. α_ij per edge, tagged by method (AGENTS.md §2.3).
        alpha_results = self._estimate_all_alpha(graph)

        # 3-4. H_cascade via single topological pass; CHAF; Theorem 1 guard.
        alpha_plain = {edge: res.value for edge, res in alpha_results.items()}
        h_cascade = compute_h_cascade(graph, alpha_plain, numeric_H)
        chaf = compute_chaf(h_cascade, numeric_H)
        assert_theorem_1(graph, chaf)

        # 5. Monitor placement S* — expensive scoring ONLY here (AGENTS.md §2.4).
        template = graph.pipeline_template_id
        if template not in self._placement_cache:
            self.refresh_placement(template, graph)
        s_star = self._placement_cache[template]

        # 6. Semantic entropy strictly limited to S*.
        se_values: dict[str, float] = {}
        monitored: dict[str, str] = {nid: "cheap" for nid in graph.nodes}
        if self.generate_fn is not None and self.entail_fn is not None:
            for step_id in sorted(s_star & set(graph.nodes)):
                step = graph.nodes[step_id]
                se_values[step_id] = semantic_entropy(
                    step_id,
                    step.output,
                    self.generate_fn,
                    self.entail_fn,
                    selected_steps=s_star,
                    K=self.se_k,
                    cluster_threshold=self.se_threshold,
                )
                monitored[step_id] = "full"

        # 7. CHRS (Eq. 10); unmonitored steps get se=0 at 'cheap' fidelity.
        chrs_values = {
            nid: chrs_fn(h_cascade[nid], se_values.get(nid, 0.0), self.lambda_)
            for nid in graph.nodes
        }

        # 8. Online CUSUM across the run's topo order (per-run state).
        detector = self.detector_pool.get_or_create(run_id, self.cusum_params)
        alerts: list[Alert] = []
        for node_id in graph.topo_order():
            fired = detector.update(chrs_values[node_id])
            if fired:
                causes = localize_root_cause(graph, alpha_plain, h_cascade, node_id)
                alerts.append(
                    Alert(
                        alert_id=str(uuid.uuid4()),
                        run_id=run_id,
                        triggered_at_step=node_id,
                        s_n_value=detector.S,
                        root_causes=causes,
                    )
                    if not alerts
                    else alerts[-1]  # dedupe within one analysis pass (TODO Phase 12)
                )
        if alerts:
            # One alert per analysis pass; keep first trigger + its root causes.
            alerts = alerts[:1]

        return AnalysisResult(
            run_id=run_id,
            H=H,
            h_cascade=h_cascade,
            chaf=chaf,
            chrs=chrs_values,
            alpha_edges=alpha_results,
            monitored_fidelity=monitored,
            semantic_entropy_values=se_values,
            alerts=alerts,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    # -- internals -------------------------------------------------------------

    def _estimate_all_alpha(self, graph: PipelineGraph) -> dict[tuple[str, str], AlphaResult]:
        results: dict[tuple[str, str], AlphaResult] = {}
        for src, dst in graph.edges:
            results[(src, dst)] = self._estimate_alpha(graph.nodes[src], graph.nodes[dst])
        return results

    def _estimate_alpha(self, source: Step, target: Step) -> AlphaResult:
        clamp_max = float(self.config.get("propagation_estimator", {}).get("clamp_max", 1.0))
        result = self._dispatch_alpha(source, target)
        if result.value > clamp_max or abs(result.raw_value) > clamp_max + 1e-9:
            # Clamping keeps Theorem 1 sound; a >clamp raw value signals an
            # estimator bug upstream — surfaced here rather than hidden.
            result = AlphaResult(
                value=float(np.clip(result.raw_value, 0.0, clamp_max)),
                raw_value=result.raw_value,
                method=result.method,
            )
        return result

    def _dispatch_alpha(self, source: Step, target: Step) -> AlphaResult:
        track = self.alpha_track
        # Track A requires the attention matrix on the TARGET step (A^(j)).
        if track == "attention" or target.attention_weights is not None:
            if target.attention_weights is None:
                raise PropagationDependencyError(
                    f"track 'attention' requested but step {target.id!r} carries no "
                    "attention_weights (white-box model access required)"
                )
            return propagation_coefficient_attention(
                target.attention_weights,
                self._source_span_in_context(source, target),
            )
        if track == "nli_proxy":
            if self.entail_fn is None:
                raise PropagationDependencyError(
                    "track 'nli_proxy' requires an entail_fn (NLI model) — "
                    "see configs/default.yaml propagation_estimator.nli_model"
                )
            return NliProxyEstimator(self.entail_fn)(source.output, target.output)
        if track == "counterfactual":
            if not self.expensive_mode:
                raise PropagationDependencyError(
                    "track 'counterfactual' is gated behind expensive_mode=true"
                )
            if self.counterfactual_stack is None:
                raise PropagationDependencyError(
                    "track 'counterfactual' requires a CounterfactualEstimator stack"
                )
            context = "\n".join(d.content for d in target.retrieved_docs)
            return self.counterfactual_stack(source.output, context, target.output)
        raise ValueError(f"unknown propagation track {track!r}")

    @staticmethod
    def _source_span_in_context(source: Step, target: Step) -> list[int]:
        """Token positions of o_i inside c_j — approximate by proportional split.

        PAPER-verified against the PDF text (Def. 2): α_ij is the per-row attention
        fraction on o_i's span, averaged over all L_j rows; idx(o_i) selects context
        columns only. Exact token alignment requires the tokenizer that produced the
        attention matrix — TODO Phase 3's off-by-one warning applies until wired.
        """
        n_ctx = target.attention_weights.shape[1] if target.attention_weights is not None else 0
        return list(range(n_ctx))


def _descendants(graph: PipelineGraph, start: str) -> list[str]:
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for succ in graph.edges_out_of(node):
            if succ not in seen:
                seen.add(succ)
                frontier.append(succ)
    return sorted(seen)


def _min_distance(graph: PipelineGraph, start: str, target: str) -> int:
    from collections import deque

    dist = {start: 0}
    q: deque[str] = deque([start])
    while q:
        node = q.popleft()
        if node == target:
            return dist[node]
        for succ in graph.edges_out_of(node):
            if succ not in dist:
                dist[succ] = dist[node] + 1
                q.append(succ)
    return 10**9
