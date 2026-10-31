"""Dataset loaders (TODO Phase 14).

Each loader converts a benchmark dataset into PipelineGraph traces with
ground-truth labels. Loaders that need data still missing from disk raise with
download instructions rather than pretending to run - per AGENTS.md, stubs must
be loud, never silently faked.

HaluEval-RAG is REAL: `load_halu_eval_rag_sample` reads the committed,
provenance-stamped sample slice at eval/datasets/data/halu_eval_rag_sample.json
(150 stratified records; see eval/datasets/make_halu_sample.py for how it was
cut and the header's provenance_note for why it comes from upstream's
qa_data.json rather than the retired halu_eval_rag.json path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.graph.pipeline_graph import Document, PipelineGraph, Step

REQUIRED_NOTE = (
    "Download {name} and place it at {{path}}; see README section 6 step 1. "
    "Loader implementation lands with TODO Phase 14 once the raw format is in hand."
)

SAMPLE_TEMPLATE_ID = "halu_eval_rag_sample"


class DatasetNotAvailable(FileNotFoundError):
    pass


class MalformedSample(ValueError):
    """Sample file exists but does not match the committed header format."""


@dataclass
class LabeledTrace:
    """One converted dataset record: a PipelineGraph plus per-step ground truth."""

    graph: PipelineGraph
    labels: dict[str, bool] = field(default_factory=dict)


def load_halu_eval_rag_sample(path: Path) -> list[LabeledTrace]:
    """Load the committed HaluEval-RAG sample slice into 3-step labeled traces.

    Per record (knowledge, question, answer, sampled_answer):
      s1 retrieve  - output=knowledge, retrieved_docs=[knowledge]
      s2 reason    - output=question, predecessors=[s1]
      s3 synthesize- output=answer, predecessors=[s2], retrieved_docs=[knowledge]
    label {s3: answer_is_hallucinated} where the answer is right_answer
    (label False) or hallucinated_answer (label True) per the header's
    'sampled_answer' field, which every record of the committed sample carries;
    a record without it raises MalformedSample (loud, never guessed).
    """
    path = Path(path)
    if not path.exists():
        raise DatasetNotAvailable(
            f"HaluEval-RAG sample not found at {path}; run "
            "eval/datasets/make_halu_sample.py against a downloaded raw file "
            "(see its docstring for the upstream-path provenance note)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "records" not in payload:
        raise MalformedSample(
            f"{path}: expected the committed header format "
            '{"source", ..., "records": [...]} — not a bare record list'
        )
    traces: list[LabeledTrace] = []
    for idx, record in enumerate(payload["records"]):
        knowledge, question = str(record["knowledge"]), str(record["question"])
        sampled = record.get("sampled_answer")
        if sampled == "right_answer":
            answer, hallucinated = str(record["right_answer"]), False
        elif sampled == "hallucinated_answer":
            answer, hallucinated = str(record["hallucinated_answer"]), True
        else:
            raise MalformedSample(
                f"{path}: record {idx} has no valid 'sampled_answer' field "
                f"(got {sampled!r}); the committed sample always carries it"
            )
        traces.append(_build_trace(knowledge, question, answer, hallucinated))
    return traces


def _build_trace(knowledge: str, question: str, answer: str, hallucinated: bool) -> LabeledTrace:
    graph = PipelineGraph(SAMPLE_TEMPLATE_ID)
    knowledge_doc = Document(doc_id="s1_knowledge", content=knowledge)
    graph.add_step(Step(id="s1", kind="retrieve", output=knowledge, retrieved_docs=[knowledge_doc]))
    graph.add_step(Step(id="s2", kind="reason", output=question, predecessors=["s1"]))
    graph.add_step(
        Step(
            id="s3",
            kind="synthesize",
            output=answer,
            retrieved_docs=[knowledge_doc],
            predecessors=["s2"],
        )
    )
    return LabeledTrace(graph=graph, labels={"s3": hallucinated})


def load_ragtruth(path: Path):
    raise DatasetNotAvailable(REQUIRED_NOTE.format(name="RAGTruth", path=path))


def load_agenthallu(path: Path):
    raise DatasetNotAvailable(REQUIRED_NOTE.format(name="AgentHallu", path=path))
