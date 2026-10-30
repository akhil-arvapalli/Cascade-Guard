"""Synthetic trace generators for validation without external datasets.

The paper's Fig. 1 shape: s1 Plan -> s2 Retrieve -> s3 Reason(*) -> s4 Tool Call
-> s5 Synthesize, where a hallucination at s3 propagates downstream.
"""

from __future__ import annotations

import random

from core.graph.pipeline_graph import Document, PipelineGraph, Step, StepKind


def _step(
    step_id: str,
    kind: StepKind,
    output: str,
    docs_content: list[str] | None = None,
    preds: list[str] | None = None,
) -> Step:
    return Step(
        id=step_id,
        kind=kind,
        output=output,
        retrieved_docs=[
            Document(doc_id=f"{step_id}_d{i}", content=c) for i, c in enumerate(docs_content or [])
        ],
        predecessors=preds or [],
    )


def make_paper_fig1_trace(corrupted: bool = True) -> PipelineGraph:
    """Fig. 1 pipeline. corrupted=True plants a hallucination at s3 (Reason)."""
    g = PipelineGraph("paper_fig1")

    good_doc_2 = "The Eiffel Tower is located in Paris, France."
    # Clean reasoning echoes the source; corrupted reasoning fabricates a
    # location with vocabulary disjoint from every retrieved document, so even
    # a purely lexical embedder scores H(s3) high on the corrupted variant.
    good_out_3 = (
        "According to the retrieved document, " "the Eiffel Tower is located in Paris, France."
    )
    bad_out_3 = "Brandenburg Gate overlooks Berlin central plaza fountains nearby."

    if corrupted:
        out_3 = bad_out_3
        doc_4 = "Reasoning transcript captured: Brandenburg Gate overlooks Berlin central plaza."
        # s4 consumes the corrupted o3 -> propagation edge by construction
    else:
        out_3 = good_out_3
        doc_4 = f"Reasoning transcript: {good_out_3}"

    g.add_step(_step("s1_plan", "plan", "Plan: find where the Eiffel Tower is."))
    g.add_step(
        _step(
            "s2_retrieve",
            "retrieve",
            output="Retrieved: " + good_doc_2,
            docs_content=[good_doc_2],
            preds=["s1_plan"],
        )
    )
    g.add_step(_step("s3_reason", "reason", out_3, [good_doc_2], ["s2_retrieve"]))
    g.add_step(
        _step(
            "s4_tool",
            "tool_call",
            output="Tool query built from reasoning: " + ("Berlin" if corrupted else "Paris"),
            docs_content=[doc_4],
            preds=["s3_reason"],
        )
    )
    final_city = "Berlin" if corrupted else "Paris"
    g.add_step(
        _step(
            "s5_synth",
            "synthesize",
            output=f"The answer is: the tower is in {final_city}. Based on: "
            + ("the tower is in Berlin" if corrupted else "the tower is in Paris"),
            docs_content=[doc_4, "Final answer context: " + out_3],
            preds=["s4_tool"],
        )
    )
    return g


def make_random_dag(
    seed: int, n_nodes: int = 8, p_edge: float = 0.25, corrupted_node: str | None = None
) -> tuple[PipelineGraph, dict]:
    """Random layered DAG with seeded rng; returns (graph, alpha_truth).

    Every node gets a small retrieval set echoing its own output (low H) except
    `corrupted_node`, whose output shares nothing with its docs (high H).
    """
    rng = random.Random(seed)
    g = PipelineGraph(f"random_dag_{seed}")
    node_ids = [f"s{i}" for i in range(n_nodes)]

    for i, nid in enumerate(node_ids):
        corrupted = nid == corrupted_node
        output = (
            "unrelated gibberish zqx jkw vbp about nothing real here"
            if corrupted
            else f"Step {i} reports factual findings consistent with source material {i}."
        )
        alt = "Clean source document entirely different words appear here"
        doc_text = output if not corrupted else alt
        g.add_step(
            Step(
                id=nid,
                kind=("plan" if i == 0 else "reason"),
                output=output,
                retrieved_docs=[Document(doc_id=nid + "_d0", content=doc_text)],
                predecessors=(
                    [p for p in node_ids[: max(i, 1)] if rng.random() < p_edge] if i > 0 else []
                ),
            )
        )

    # ensure connectivity: link each node to one earlier node if it has none
    order = g.topo_order()
    for idx, nid in enumerate(order):
        if not graph_has_pred(g, nid) and idx > 0:
            parent = order[rng.randrange(idx)]
            g.nodes[nid].predecessors.append(parent)
            g.add_edge(parent, nid)
    return g, {}


def graph_has_pred(g: PipelineGraph, nid: str) -> bool:
    return bool(g.edges_into(nid))
