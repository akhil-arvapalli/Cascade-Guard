"""Single-step H baseline (TODO Phase 14) - the Theorem 1 comparison point.

This is EXACTLY the "single-step detector" Theorem 1 contrasts CascadeGuard
against: each step is judged in isolation from its own H(s_i) (Eq. 4 alone,
1 - cosine(output, retrieved context)), with NO H_cascade propagation and no
pipeline awareness. Any step whose own H exceeds the threshold is flagged; a
downstream step inherits nothing from an upstream hallucination. Wrappers.py
stubs (RAGAS/SelfCheckGPT) stay as-is - those are library wraps, this one is an
in-repo reimplementation because no published package exposes this exact
per-step Eq.4 detector, and the comparison must not be faked.
"""

from __future__ import annotations

from core.embeddings import EmbeddingFn
from core.graph.pipeline_graph import PipelineGraph
from core.scoring.faithfulness import hallucination_score


def single_step_h_baseline(
    graph: PipelineGraph, embedder: EmbeddingFn, threshold: float = 0.5
) -> dict[str, bool]:
    """Flag each step purely on its own H(s_i) (Eq. 4), no propagation.

    Returns {step_id: flagged}. Steps with NO retrieved context have undefined
    H (hallucination_score returns None - "not applicable", never 0), so they
    are never flagged: there is nothing for the step to be unfaithful *to*.
    The default threshold=0.5 is the starting point; run_detection_eval tunes
    it on the tuning split for best F1 before any held-out comparison.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    flags: dict[str, bool] = {}
    for step_id, step in graph.nodes.items():
        h = hallucination_score(step.output, step.retrieved_docs, embedder)
        flags[step_id] = h is not None and h > threshold
    return flags
