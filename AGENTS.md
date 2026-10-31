# AGENTS.md

This file is the shared context for any AI agent (Claude, Claude Code, Cursor, Copilot, etc.) working on this repository. Read this before making changes. If you're an agent picking up work here, treat this as more authoritative than your own assumptions about what a "hallucination detection framework" should look like — this project implements a specific paper with specific formulas, and deviating from them silently is the single easiest way to quietly break the project.

---

## 1. What this project is

CascadeGuard is a real-time, pipeline-aware hallucination detection framework for multi-step agentic RAG systems. It is the implementation of a research paper (`CascadeGuard_Research_Paper.pdf`, included in the repo — read it before touching scoring/detection code) written by the project owner as a final-year academic research contribution.

**Full spec:** `README.md` (architecture, module design, math-to-code mapping) and `TODO.md` (phased build plan with exit criteria) are the primary references. Do not re-derive architecture from scratch — it's already specified. If something in README/TODO seems wrong or you want to deviate from it, say so explicitly and ask, rather than silently implementing something different.

**Owner:** Akhil (Arvapalli Venkata Sai Akhil) — final-year B.Tech CSE (AI/ML), building this as his primary research/academic showcase project, alongside active job applications where this project is referenced directly. Precision and correctness matter more than speed here — this isn't a throwaway prototype.

---

## 2. Non-negotiable invariants

These come directly from the paper's theorems. Any agent working on scoring/detection code must respect them:

1. **`H_cascade` must be computed via a single topological traversal** of the pipeline DAG (Eq. 5), not recomputed recursively per-node without memoization, and not computed in an arbitrary node order. Wrong order = wrong scores, silently.
2. **`CHAF(s_j) ≤ depth(s_j) + 1` must hold for every node** (Theorem 1). This is a runtime-testable invariant — if it's ever violated in tests, the bug is in `α_ij` estimation (likely an unclamped value >1), not in the theorem. Never "fix" a violation by loosening the assertion.
3. **`α_ij` (propagation coefficient) has two legitimate estimation tracks** — attention-based (Track A, exact per Eq. 6, requires white-box model access) and surrogate (Track B, NLI/counterfactual, for closed-model APIs). Every computed `α_ij` must be tagged with which track produced it (`alpha_method` field). Never blend or silently substitute one for the other without labeling it.
4. **Semantic entropy (`SE`, Eq. 9) is expensive** (K extra LLM calls) and must only run on steps selected by the Monitor Placement Optimizer's `S*`, never unconditionally on every step. If you're tempted to "just run it everywhere to be safe," don't — that defeats the entire point of the budget-constrained placement (§II.H of the paper).
5. **CUSUM detector state (`S_n`) is per-pipeline-run, not global.** One detector instance per `run_id`. Don't share state across concurrent runs.
6. **Config values (`λ`, `ARL0` target, budget `B`, etc.) live in `configs/default.yaml` and must never be hardcoded inline.** `λ` in particular is meant to be tuned via cross-validation (README §6), not guessed — if it's still a placeholder, say so rather than treating it as final.

---

## 3. Current project state

> **Agents: update this section as work progresses.** This is the single most important part of this file for cross-agent continuity — if you finish a phase or make a significant decision, write it here before ending your session, in plain language, so the next agent (possibly a different model/tool) doesn't have to reconstruct it from git log archaeology.

- **Phase:** v1 built (2026-08-25, agent session): research core Phases 1–9 implemented + tested; SDK/API/alerts skeletons (Phases 10–12); Phase 13 dashboard **DONE** (React + FastAPI read endpoints, DAG visualization, node inspector, root-cause path highlighting, score tables, run/alert listing); Phase 14 eval and Phase 15 hardening remain open
- **2026-10-31 morning session (ZCode):** Phase 8 COMPLETE (`43c3305`: core/placement/costs.py measured monitor costs, risk_model.py trace-simulated marginal risk from historical H_cascade, cache.py PlacementCache in-process+Redis fail-open, live path reads cached S*, jobs/refresh.py; also fixed /v1/placement/refresh looking up template_id in run-id-keyed run_graphs). demo.py one-command executable (`11f4995`, `956d2e4`: clean run + Fig.1 cascade + balanced HaluEval slice; --serve seeds the dashboard). Phase 14 eval DONE at sample scale (`f34b7cd`): 150-record HaluEval-RAG slice from data/qa_data.json (upstream halu_eval_rag.json was renamed/404), single-step baseline, lambda sweep (PLATEAU 0.2-1.0 all F1=0.6667 - config lambda 0.6 placeholder UNCHANGED, D-001 provisional), head-to-head lands at degenerate all-alert point because hashing-embedder H is INVERTED on this slice (hallucinated = sentence-form knowledge copies -> low trigram distance); Theorem 1: 0 violations across 150 traces. EVAL_REPORT.md filled with limitations; honest framing required. V2-3 + Phase 15 slice (`0318c37`): storage/postgres_impl.py full protocol (DSN-gated tests skip without Docker), calibration table + apply_schema.py, auth middleware (config-gated, default off), JSON logging + request timing, GET /health, jobs/retention.py, RUNBOOK.md, repo.list_runs(). KNOWN GAPS: complete_run/pool.complete not wired on analyze (runs stay 'running' - RUNBOOK section 5); Redis Streams ingest + worker (Phase 11) still open; Slide deck (user-made) reviewed + 2 factual edits (vertex-cut -> greedy 1-1/e) delivered as MP-1 Batch-102 PRC-2_updated.pptx. Suite: 108 passed + 13 DSN-gated skipped; ruff/black/mypy clean.
- **2026-10-30 late session (ZCode):** CUSUM calibration loop closed (Phase 0/V2-4 partial): `scripts/calibrate.py` + `core/calibration.py` harvest clean-trace CHRS → `data/calibration.json`, reloaded at boot; `engine.calibrate()` now clears the detector pool (pooled detectors snapshot κ/h — root cause of stale-limit false alarms); `cusum.delta` 0.1→0.03 (in-regime |Δ|≈0.25 for calibrated σ≈0.059). Dashboard: fixed broken `/v1` fetch paths (UI showed no data before), dark-terminal-brutalist restyle, server-ranked root-cause wiring. Phase 12 **DONE**: in-band blocking via `POST /v1/decision/{run_id}/{step_id}` (ephemeral CUSUM projection — pooled S_n never advanced by decisions; optional `blocking.threshold_tau` absolute gate; notify_only default). Phase 5 SE unit tests verified as already existing (stale checkbox fixed). Load-test harness at `eval/loadtest/load_test.py` (Phase 11 item). Suite: 81 tests.
- **Last completed milestone:** v1 code-complete — every paper equation Eq. 1–13 has a real implementation verified against the paper PDF text, plus a passing test suite including hand-computed cases, the Theorem 1 invariant test, ARL0 simulation, and greedy-vs-brute-force placement validation; Phase 13 dashboard built and served from FastAPI static files at /dashboard
- **Flagged approximations in v1 (all deliberate, all marked in code):**
  - **Eq. 12 (ARL0) is an APPROXIMATION whose accuracy degrades as |Δ|=(κ−μ0)/σ grows.** Empirically validated at |Δ|=0.25: empirical ARL ≈ 1.6–2× target, i.e. CONSERVATIVE (real false-alarm rate lower than the <0.2% claim). At extreme gaps (δ≫σ) the formula massively overestimates h and the detector effectively never fires — keep δ within ~σ/2 of σ when configuring. See `tests/test_cusum.py` regime note.
  - **Eq. 12's h comes out in σ-standardized units** and is rescaled by σ before comparing against raw-S_n (`core/detection/cusum.py`) — this was a real bug caught by simulation during the build.
  - Track A token-span alignment (`idx(o_i)` in c_j) currently attributes the FULL context width — needs tokenizer-level alignment before Track A/B correlation means anything (TODO Phase 3)
  - Placement `marginal_risk_fn` is a structural descendant-mass PROXY; the trace-simulated version needs historical data (TODO Phase 8)
  - Embedding default is the dependency-free `HashingEmbedder` so everything runs/tests without downloads; production backend choice still open (D-003)
  - NLI/counterfactual/SE model hooks are injectables (`entail_fn`, `generate_fn`) — no heavy models wired by default; wire them in `api/main.py::build_engine`
  - CUSUM μ0/σ defaults in config are placeholders until `calibrate()` runs on clean traces
- **Open decisions not yet made:**
  - `λ` value (currently placeholder 0.6 — needs cross-validation per README §6, TODO Phase 14)
  - Which `α_ij` track is the default for production (Track A needs a self-hosted open-weight model; decide whether that's actually in scope or if Track B is the realistic default)
  - Embedding model final choice (README suggests bge-base-en-v1.5 or e5-base — not yet benchmarked)
- **Known blockers:** external datasets (HaluEval-RAG, RAGTruth, AgentHallu) not downloaded — Phase 14 blocked on that

---

## 4. Repo structure

See `README.md` §8 for the full intended structure. Short version:

```
sdk/            — in-process instrumentation (LangChain/LlamaIndex integrations)
core/           — the actual math: graph, scoring, detection, placement, localization
storage/        — Postgres schema + repository layer
api/            — FastAPI service
eval/           — dataset loaders, baseline implementations, eval harness
dashboard/      — DAG visualization + metrics UI
tests/          — includes the Theorem 1 invariant test, keep this passing always
configs/        — default.yaml, all tunable knobs
```

---

## 5. Conventions

- **Language:** Python for `sdk/`, `core/`, `api/`, `eval/`. Consider Go only for the CUSUM hot path if latency profiling later shows it's needed — don't preemptively rewrite in Go without a measured reason.
- **Style:** `black` + `ruff` for formatting/linting, `mypy` for type-checking. Type hints are expected on all `core/` functions — this is scoring/math code, type errors here are correctness bugs.
- **Testing:** every function implementing a paper equation (Eq. 3–13) needs a unit test that checks it against a hand-computed example, not just "runs without crashing." See TODO.md phase exit criteria for what "done" means per component.
- **Docstrings:** functions implementing paper equations should reference the equation number in their docstring (e.g. `"""Implements Eq. 5 (H_cascade)."""`) so any agent reading the code later can trace it back to the paper without guessing.
- **Commits:** reference the TODO.md phase/checkbox being worked on where relevant, so progress is traceable across sessions and across different agents picking up the work.

---

## 6. What NOT to do

- Don't invent a simplified hallucination-detection approach "for now" and call it CascadeGuard. If a full implementation of a component (e.g. attention-based `α_ij`) is out of scope for a given session, say so explicitly and stub it clearly (`raise NotImplementedError("Track A not yet implemented — see AGENTS.md")`), don't quietly fake it with something else.
- Don't skip the Theorem 1 invariant test to make CI green faster.
- Don't apply semantic entropy or the expensive counterfactual `α_ij` track to every step "to be thorough" — this contradicts the budget-constrained design that's a core contribution of the paper.
- Don't change the math (Eq. 3–13) without flagging it loudly — if you think a formula should be implemented differently than the paper states, that's a conversation to have with Akhil, not a silent judgment call.
- Don't remove or weaken the `alpha_method` tagging — it's what makes eval ablations (Track A vs B) possible later.

---

## 7. Where to look first

- New to the project? Read `README.md` in full before writing code.
- Picking up mid-build? Read §3 above (Current project state), then check `TODO.md` for the active phase's unchecked items.
- Touching scoring/detection math? Re-read the relevant section of `CascadeGuard_Research_Paper.pdf` (Eq. references are in README §4) before implementing — don't implement from memory of "how hallucination detection usually works."
