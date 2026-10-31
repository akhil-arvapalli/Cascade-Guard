# DECISIONS.md

A lightweight decision log (ADR-style). Every time a "hard stop" question from `AGENT_OPERATING_PROMPT.md` gets resolved — or any other choice gets made that a future session/agent/interviewer would reasonably ask "why did you do it this way?" — it gets an entry here. Newest entries at the top.

Don't skip this because a decision feels obvious in the moment. It won't feel obvious in three months, and "why bge-base and not e5" is exactly the kind of question that comes up in a technical interview about this project.

---

## Entry template

```
## D-00X: [short title]

**Date:** YYYY-MM-DD
**Status:** proposed | decided | superseded by D-00Y
**Decided by:** [Akhil / agent name+session / joint]

**Context**
What question needed answering, and why it mattered.

**Options considered**
1. Option A — pros/cons
2. Option B — pros/cons
3. (etc.)

**Decision**
What was chosen.

**Why**
The actual reasoning — not just "seemed better," the specific tradeoff that tipped it.

**Consequences**
What this locks in, what it makes harder later, what would need to change if this is revisited.
```

---

## Log

*(Entries below added during the 2026-08-25 v1 build session.)*

### D-005: v1 embedding default is a dependency-free hashing embedder
**Date:** 2026-08-25 **Status:** proposed (revisit with real benchmarks) **Decided by:** agent session (v1 build)

**Context** — Every scoring path needs embeddings; the production choice (bge vs e5) is still open (D-003) and requires benchmarking infrastructure that doesn't exist yet.
**Options considered** — 1. Block all code on standing up vLLM/ONNX + downloading a model. 2. Default `HashingEmbedder` (deterministic char-trigram bag, zero deps), keep `SentenceTransformerEmbedder` behind the `[embeddings]` extra, switch via one config key.
**Decision** — Option 2 (`configs/default.yaml: embeddings.backend`).
**Why** — Keeps the whole system runnable and testable today without downloads; the embedder is an injectable everywhere in `core/`, so swapping backends later changes zero scoring code. Honest about being lexical-only, not semantic.
**Consequences** — H(s_i) numbers on real data are NOT meaningful until the production backend lands; eval claims must not be made on the hashing backend.

### D-006: Track A span alignment deferred; Eq. 6 normalization corrected mid-build
**Date:** 2026-08-25 **Status:** decided for v1 **Decided by:** agent session (v1 build)

**Context** — Eq. 6 needs idx(o_i): o_i's token positions inside c_j. Exact alignment requires the tokenizer that produced the attention matrix.
**Decision** — Track A implements Eq. 6's arithmetic EXACTLY as verified against the paper text (per-row fraction averaged over output rows); span alignment currently attributes the full context width, marked PAPER-CHECK/TODO in `core/engine.py::_source_span_in_context`.
**Why** — Math fidelity first (the original draft had the wrong normalization — global ratio instead of per-row mean; caught and fixed against the PDF). Alignment is plumbing that only matters once a white-box model actually runs; it does not change any equation.
**Consequences** — Track A/B correlation (Phase 3 exit criterion) is meaningless until alignment is wired; flagged in AGENTS.md §3 so nobody trusts premature correlation numbers.

### D-007: In-memory repository is the v1 default storage backend
**Date:** 2026-08-25 **Status:** proposed **Decided by:** agent session (v1 build)

**Context** — README §5 specifies Postgres+pgvector; running Postgres in this dev environment wasn't verifiable during the build.
**Decision** — `storage.backend=memory` default (schema-shaped dict store); SQL schema delivered verbatim in `storage/schema.sql`, wired into docker-compose initdb; Postgres repository behind the `[postgres]` extra.
**Consequences** — Persistence is per-process until Postgres is stood up; no silent divergence — both implement the same TraceRepository protocol.

### D-001..D-004 remain UNDECIDED (unchanged)
λ sweep, α track production default, embedding model, Kafka migration — all still require the evidence described below. *(2026-10-31: D-001 now carries a PROVISIONAL sample-scale value — see its entry; it is not a production decision.)*

### D-001: λ (lambda) value for CHRS risk combiner — **PROVISIONAL (sample-scale, 2026-10-31)**
Per README §6 / TODO Phase 14, this requires an actual cross-validation sweep on held-out data, not a guess. Log the winning value and the sweep results (or a link to the eval report) here once resolved.

**2026-10-31 entry (Phase 14 small-scale eval, agent session) — PROVISIONAL, sample-scale:**
- **Winning λ = 0.2**, picked by grid sweep (0.0–1.0 step 0.1) maximizing detection F1 on the tuning split (106 traces, stratified, seed 20261031) of the committed HaluEval-RAG sample (`eval/datasets/data/halu_eval_rag_sample.json`), with the fixed alert rule max CHRS > κ = μ0 + δ/2 from `data/calibration.json`.
- **Tuning F1 = 0.6667 (precision 0.5, recall 1.0, FAR 1.0); held-out identical (n=44).** Crucially, λ ∈ [0.2, 1.0] sits on an IDENTICAL F1 plateau — λ is effectively **not identifiable** on this slice; 0.2 is just the lowest grid point reaching the plateau (deterministic tie-break).
- **Why degenerate:** with SE=0 and the 3-step HaluEval trace (upstream H ≡ 0 by construction), CHRS = λ·H(s3), so the sweep is a threshold sweep in disguise; and the hashing embedder (D-005) makes per-step H INVERTED on this slice (mean H 0.46 on hallucinated vs 0.72 on right — short gold entities vs knowledge-copied hallucination sentences), so the F1-optimal operating point is "alert on everything".
- **Consequences:** `configs/default.yaml` is deliberately left at the placeholder λ=0.6 — 0.2 must NOT be baked in from this run. Re-sweep with a semantic embedder + real entailment/SE (V2-1) on richer traces; only that can break the plateau.
- Evidence: `eval/results/detection_eval.json`, EVAL_REPORT.md §3/§4/§8, `eval/run_detection_eval.py`.

### D-002: Production default track for α_ij (propagation coefficient) — **NOT YET DECIDED**
Track A (attention-based, exact, needs self-hosted open-weight model) vs. Track B (NLI/counterfactual surrogate, works with closed APIs). This is a real architecture decision — depends on whether the target deployment can host its own model or is stuck behind a closed API. Log the reasoning and the Track A/B correlation data from Phase 3 here once decided.

### D-003: Embedding model choice — **NOT YET DECIDED**
README §7 suggests bge-base-en-v1.5 or e5-base as candidates. Needs actual latency/quality benchmarking (Phase 0/2) before locking in. Log the benchmark numbers and final choice here.

### D-004: Streaming transport — Redis Streams vs Kafka — **PROVISIONALLY Redis Streams**
README §7 suggests starting with Redis Streams for simplicity and moving to Kafka only if volume demands it. Log here if/when that migration actually happens, and what throughput number triggered it.
