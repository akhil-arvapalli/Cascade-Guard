"""Trace-simulated marginal_risk_fn (TODO Phase 8).

Replaces the structural descendant-mass PROXY in
``CascadeGuardEngine.refresh_placement`` when historical analyses exist.

Model: per historical trace, each node carries risk mass rm(m) = H_cascade(m)
(α-weighted accumulated risk — exactly the quantity CHRS monitors). A monitor
set covers node m in trace t if any monitor sits on m or one of m's ancestors
(detection at an ancestor surfaces the cascade before it reaches m). The
marginal value of candidate s given the selected set is the additional covered
mass, distance-discounted:

    Δ(s, selected) = Σ_t Σ_{m uncovered} [s ∈ anc_t(m)] · rm_t(m) / (1 + dist_t(s, m))

Structure still comes from the graphs, but the risk MASS now comes from real
measured H_cascade instead of the proxy's uniform 1.0 — traces that actually
carried hallucinated content dominate placement, clean traces contribute
little. History entries are (graph, h_cascade-map-or-AnalysisResult); plain
dicts are accepted so tests and future storage-backed histories need no result
object.
"""

from __future__ import annotations

from collections import deque
from typing import Any


def _ancestors(graph: Any, start: str) -> frozenset[str]:
    """Ancestor closure of start, INCLUDING start itself."""
    seen: set[str] = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for pred in graph.nodes[node].predecessors:
            if pred not in seen:
                seen.add(pred)
                frontier.append(pred)
    return frozenset(seen)


def _bfs_dist(graph: Any, start: str) -> dict[str, int]:
    dist = {start: 0}
    q: deque[str] = deque([start])
    while q:
        node = q.popleft()
        for succ in graph.edges_out_of(node):
            if succ not in dist:
                dist[succ] = dist[node] + 1
                q.append(succ)
    return dist


class _TraceRisk:
    __slots__ = ("graph", "rm", "ancestors")

    def __init__(self, graph: Any, rm: dict[str, float]) -> None:
        self.graph = graph
        self.rm = rm
        self.ancestors = {nid: _ancestors(graph, nid) for nid in graph.nodes}


class HistoricalRiskModel:
    """Marginal-risk function over historical traces, memoized per selected set."""

    def __init__(self, history: list[tuple[Any, Any]]) -> None:
        self._traces: list[_TraceRisk] = []
        for graph, result in history:
            rm_raw: dict[str, Any]
            if isinstance(result, dict):
                rm_raw = result
            else:
                rm_raw = getattr(result, "h_cascade", {})
            rm = {nid: float(v) for nid, v in rm_raw.items()}
            self._traces.append(_TraceRisk(graph, rm))
        self._dist_cache: dict[tuple[int, str], dict[str, int]] = {}
        self._cover_cache: dict[frozenset[str], list[set[str]]] = {}

    def marginal_risk(self, step_id: str, selected: set[str]) -> float:
        covered = self._covered(frozenset(selected))
        total = 0.0
        for ti, trace in enumerate(self._traces):
            if step_id not in trace.ancestors:
                continue
            dist = self._distances(ti, step_id)
            for node, mass in trace.rm.items():
                if mass <= 0.0 or node in covered[ti]:
                    continue
                if step_id in trace.ancestors[node]:
                    total += mass / (1.0 + dist.get(node, 10**9))
        return total

    def _covered(self, selected: frozenset[str]) -> list[set[str]]:
        """Per-trace set of nodes whose ancestors include any selected monitor."""
        cached = self._cover_cache.get(selected)
        if cached is not None:
            return cached
        covered: list[set[str]] = []
        for trace in self._traces:
            covered.append({n for n, anc in trace.ancestors.items() if anc & selected})
        self._cover_cache[selected] = covered
        return covered

    def _distances(self, trace_idx: int, step_id: str) -> dict[str, int]:
        key = (trace_idx, step_id)
        dist = self._dist_cache.get(key)
        if dist is None:
            dist = _bfs_dist(self._traces[trace_idx].graph, step_id)
            self._dist_cache[key] = dist
        return dist
