"""Build the committed HaluEval-RAG sample slice (TODO Phase 14).

Downloads are done OUTSIDE this repo (raw file stays in a temp dir); this script
turns a downloaded raw HaluEval QA/RAG file into the committed, stratified
sample at eval/datasets/data/halu_eval_rag_sample.json so the sample itself is
reproducible and its provenance is inspectable.

PROVENANCE NOTE (loud, per AGENTS.md): the task spec pointed at
`data/halu_eval_rag.json` on RUCAIBox/HaluEval, but upstream no longer ships a
file under that name — the path 404s on both `master` and `main` (verified
2026-10-31, GitHub contents API for data/ lists only dialogue/general/qa/
summarization_data.json). The official `data/qa_data.json` (10,000 records,
HotpotQA-based) carries the exact record schema the RAG file is documented
with: {"knowledge", "question", "right_answer", "hallucinated_answer"}. This
sample is therefore cut from the official qa_data.json and says so in the
header — it is NOT the silent-substitute kind of fallback.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REQUIRED_FIELDS = ("knowledge", "question", "right_answer", "hallucinated_answer")

SOURCE = "RUCAIBox/HaluEval data/qa_data.json (HaluEval QA / RAG split, 10,000 records)"
URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
LICENSE = "see repo (CC BY-NC-SA)"
PROVENANCE_NOTE = (
    "Requested path data/halu_eval_rag.json returns 404 on both master and main "
    "(verified 2026-10-31): upstream restructured data/ and ships this same "
    "HotpotQA-based 10K QA/RAG split as qa_data.json with the identical record "
    "schema. Every record carries BOTH right_answer and hallucinated_answer, so "
    "stratification is by which answer field the trace uses (field 'sampled_answer')."
)


def _load_records(raw_path: Path) -> list[dict]:
    """Accept either a single JSON array or JSON Lines (upstream ships JSONL)."""
    text = raw_path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(loaded, list):
        raise ValueError(f"{raw_path}: expected a JSON array or JSONL records")
    return loaded


def build_sample(
    raw_path: Path,
    out_path: Path,
    n_faithful: int = 75,
    n_hallucinated: int = 75,
    seed: int = 20261031,
) -> dict:
    records = _load_records(raw_path)
    bad = [
        i
        for i, r in enumerate(records)
        if not all(str(r.get(f, "")).strip() for f in REQUIRED_FIELDS)
    ]
    if bad:
        raise ValueError(
            f"{raw_path}: {len(bad)} record(s) missing/empty required fields, first at idx {bad[0]}"
        )

    # Stratified 75/75: every record carries both answer fields, so the two
    # classes are drawn from disjoint record sets (no record used twice) with a
    # single seeded shuffle — fully reproducible.
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    if n_faithful + n_hallucinated > len(order):
        raise ValueError("requested sample larger than source dataset")
    sampled = [dict(records[i], sampled_answer="right_answer") for i in order[:n_faithful]] + [
        dict(records[i], sampled_answer="hallucinated_answer")
        for i in order[n_faithful : n_faithful + n_hallucinated]
    ]

    header = {
        "source": SOURCE,
        "url": URL,
        "license": LICENSE,
        "sampled": f"2026-10-31, stratified {n_faithful}/{n_hallucinated} "
        f"(seed={seed}, disjoint records, 'sampled_answer' marks the trace answer field)",
        "provenance_note": PROVENANCE_NOTE,
        "records": sampled,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(header, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return header


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut the committed HaluEval-RAG sample slice")
    parser.add_argument(
        "--raw", type=Path, required=True, help="downloaded raw HaluEval QA/RAG json (temp dir)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/datasets/data/halu_eval_rag_sample.json"),
    )
    parser.add_argument("--n-faithful", type=int, default=75)
    parser.add_argument("--n-hallucinated", type=int, default=75)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    header = build_sample(args.raw, args.out, args.n_faithful, args.n_hallucinated, args.seed)
    n_faithful = sum(1 for r in header["records"] if r["sampled_answer"] == "right_answer")
    n_hall = sum(1 for r in header["records"] if r["sampled_answer"] == "hallucinated_answer")
    size_kb = args.out.stat().st_size / 1024
    print(
        f"wrote {args.out}: {len(header['records'])} records "
        f"({n_faithful} right / {n_hall} hallucinated), {size_kb:.0f} KB"
    )


if __name__ == "__main__":
    main()
