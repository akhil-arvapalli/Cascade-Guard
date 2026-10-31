# EVAL_REPORT.md

Results of running CascadeGuard's evaluation plan (README §6, TODO Phase 14) against the paper's claims. Filled in as Phase 14 actually runs, with real numbers only. Sections not yet executed say so explicitly rather than carrying stale placeholders — a half-honest eval report is worse than an incomplete one.

**Status (2026-10-31): Phase 14 partially executed — a small-scale real-dataset eval on a HaluEval-RAG sample slice (150 records) plus the synthetic-check suite. Track A/B ablation and placement-approximation quality remain open.**

---

## Honesty frame for this run (read before any table below)

This run is a **harness + pipeline-aware-vs-single-step comparison**, NOT the paper's full claim. Three deliberate placeholder substitutions dominate the numbers:

1. **Embedder:** `HashingEmbedder` (char-trigram bag, dependency-free — DECISIONS D-005). H(s_i) = 1 − cosine(output, retrieved context) is pure lexical overlap, not semantic faithfulness.
2. **Entailment:** α_ij via a cosine-similarity surrogate replicated from `api/main.build_engine` — not a real NLI model (DECISIONS D-002 open).
3. **SE = 0 everywhere:** no `generate_fn` wired, so CHRS(s_i) = λ·H_cascade(s_i) exactly (Eq. 10 with the SE term zeroed).

---

## What the paper claims (baseline to reproduce)

| Claim | Value |
|---|---|
| Reduction in undetected cascading failures vs. best single-step baseline | 34.7% |
| Latency overhead | 12 ms per pipeline step |
| Datasets | HaluEval-RAG, RAGTruth, AgentHallu |
| Target false-alarm rate (from ARL0=500, Eq. 12) | < 0.2% per pipeline execution |
| Monitor placement approximation guarantee | (1 − 1/e) ≈ 63% of optimal |

---

## 1. Dataset preparation status

| Dataset | Loader implemented | # traces converted | Ground-truth labels available |
|---|---|---|---|
| HaluEval-RAG | Yes — `eval/datasets/loaders.py::load_halu_eval_rag_sample` | 150 (committed sample slice) | Yes — `{s3: answer_is_hallucinated}` |
| RAGTruth | No — loud stub raises with download instructions | 0 | — |
| AgentHallu | No — loud stub raises with download instructions | 0 | — |

**HaluEval-RAG provenance (loud deviation from the original plan, per AGENTS.md):** the planned upstream path `data/halu_eval_rag.json` on RUCAIBox/HaluEval returns **404 on both `master` and `main`** (verified 2026-10-31 via the GitHub contents API — `data/` now ships only dialogue/general/qa/summarization_data.json). The official **`data/qa_data.json`** is the same 10,000-record HotpotQA-based QA/RAG split with the identical record schema (`knowledge`, `question`, `right_answer`, `hallucinated_answer`), and the sample was cut from it. Nothing was hand-curated or faked; the deviation is stamped in the sample header (`provenance_note`) and in `eval/datasets/make_halu_sample.py`.

- Committed file: `eval/datasets/data/halu_eval_rag_sample.json` (107 KB), header records source, URL, license (CC BY-NC-SA per the upstream repo), sampling date, and per-record `sampled_answer`.
- Sampling: stratified 75 faithful (trace answer = `right_answer`) / 75 hallucinated (trace answer = `hallucinated_answer`), disjoint records, seeded shuffle (seed 20261031), reproducible via `eval/datasets/make_halu_sample.py`.
- Trace conversion (per record): s1 retrieve (output + doc = knowledge) → s2 reason (output = question) → s3 synthesize (output = answer, doc = knowledge); label on s3. `pipeline_template_id = "halu_eval_rag_sample"`.
- Label-quality quirk: every upstream record carries BOTH answer fields, so stratification is by which field the trace uses — verified by key presence on all 10,000 records before sampling.
- Note: raw upstream text contains mojibake artifacts (e.g. `1844û1846`); kept verbatim — the eval must see the data as distributed.

---

## 2. Baselines implemented

| Baseline | Status | Source |
|---|---|---|
| Single-step H detector (Theorem 1's comparison point) | Implemented — `eval/baselines/single_step.py::single_step_h_baseline` (Eq. 4 alone, no propagation; threshold tuned on the tuning split) | in-repo reimplementation (no published package exposes this exact per-step Eq. 4 detector) |
| RAGAS | Not run — loud import-guard stub remains (`ragas` not installed); no numbers produced | OSS wrap planned, never silently reimplemented |
| SelfCheckGPT | Not run — loud stub remains (no maintained pip package) | OSS wrap planned |
| RAGTruth / FActScore / ARES | Not reimplemented — cited from published numbers | paper references [4],[6],[7] |

---

## 3. λ tuning (TODO Phase 14, decision → DECISIONS.md D-001)

- **Search method:** grid sweep, λ ∈ {0.0, 0.1, …, 1.0} (11 points), engine `risk_combiner.lambda` set per point.
- **Alert rule (fixed across λ, honest):** a trace alerts iff max CHRS over its steps > κ = μ0 + δ/2 (mirrors `CusumDetector`'s Eq. 11 κ, strict `>`). κ = 0.025975 from `data/calibration.json` (μ0=0.010975, σ=0.058770, δ=0.03; calibrated on synthetic-clean CHRS, not this dataset). Because CHRS = λ·H_cascade (SE=0), a fixed κ acts as a per-λ shifted threshold — the effective cutoff in H_cascade units is κ/λ. Documented, not hidden.
- **Validation split:** deterministic stratified 70/30 (seed 20261031): tuning 106 (53 pos / 53 neg), held-out 44 (22 pos / 22 neg).
- **Optimization target:** F1 on detection (trace label = hallucinated answer).
- **Winning λ: 0.2 — PROVISIONAL, and effectively NOT IDENTIFIABLE on this slice:** λ ∈ [0.2, 1.0] all sit on an identical F1 plateau (0.6667); the tie-break rule (best F1 → recall → precision → lowest λ) picks 0.2. Do **not** read this as "0.2 is better than 0.6"; on a non-degenerate rerun the plateau should break. `configs/default.yaml` is unchanged (placeholder λ=0.6 stands until a real-model sweep).
- **Full sweep results** (tuning split, n=106):

| λ | precision | recall (detection) | F1 | false-alarm rate | mean analyze latency (ms) |
|---|---|---|---|---|---|
| 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.76 |
| 0.1 | 0.4904 | 0.9623 | 0.6497 | 1.0000 | 2.71 |
| **0.2** | **0.5000** | **1.0000** | **0.6667** | **1.0000** | **2.80** |
| 0.3–1.0 | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 2.69–2.90 |

Raw sweep + per-threshold baseline curve: `eval/results/detection_eval.json`.

**Why the sweep is flat and why FAR = 1.0:** see §8 — the per-step H signal on this slice is inverted under the flag-if-H>threshold semantics, so the F1-optimal operating point degenerates to "alert on every trace" (F1 0.6667 = flag-all on a balanced split). Only λ=0.1 (effective cutoff κ/0.1 = 0.26) and λ=0.0 (never alerts) break from the plateau.

---

## 4. Headline results — CascadeGuard vs. single-step baseline (held-out, n=44)

The only baseline actually run is the single-step H detector (Eq. 4 alone, no propagation — exactly the detector Theorem 1 contrasts against). RAGAS/SelfCheckGPT were not run (see §2). **Per-dataset breakdown beyond HaluEval-RAG is not possible yet** — the other two datasets are not loaded.

| Metric | Single-step H baseline | CascadeGuard (λ=0.2) | Paper's claim |
|---|---|---|---|
| Cascading failure detection rate (recall) | 1.0000 | 1.0000 | — |
| Reduction vs. best single-step baseline | — | **0.0%** (paper: 34.7%) | 34.7% |
| False alarm rate (empirical) | 1.0000 | 1.0000 | <0.2% (target) |
| Precision | 0.5000 | 0.5000 | — |
| F1 | 0.6667 | 0.6667 | — |
| Decision threshold | 0.00 (tuned on tuning split) | κ = 0.0260 (fixed, calibrated) | — |
| Latency per trace, mean / p95 (ms) | 1.64 / 2.55 | 2.69 / 4.01 | 12 ms/step budget |

Confusion matrices are identical (TP=22, FP=22, TN=0, FN=0): both detectors sit at the degenerate all-alert point, so **this table shows the harness works and the machinery runs — it does not show a pipeline-aware advantage.** Two structural reasons, both honest:

1. **The trace shape has zero upstream risk by construction.** s1's output equals its only retrieved document (H(s1)=0) and s2 carries no retrieved context (H undefined → 0 in the cascade), so H_cascade(s3) = H(s3) + α·H_cascade(s2) = H(s3). Cascade propagation has nothing to amplify on this slice; the two detectors differ only in thresholding granularity.
2. **The raw H signal is inverted under the flag-if-H>threshold semantics** (see §8) — a lexical-embedder artifact, not a property of the framework.

**Latency:** ~0.9 ms/step (2.69 ms per 3-step trace, wall-clock around `analyze_run`, placement + CUSUM pool included) — inside the 12 ms/step claim, but the hashing embedder is orders of magnitude cheaper than any real embedding/NLI backend, so this says nothing about production latency.

---

## 5. Theorem 1 empirical validation (CHAF ≤ depth+1)

- Ran on: **all 150 real HaluEval-RAG traces** at the chosen λ (0.2), plus the standing 50-graph synthetic sweep (`eval/run_eval.py`, still passing).
- Violations found: **0 / 0** (the engine's runtime `assert_theorem_1` would abort the run otherwise).
- Max CHAF/(depth+1) ratio observed: **1.0000** (attained at s1, where CHAF=1.0 against bound depth+1=1).
- Raw: `eval/results/detection_eval.json → chaf_check`.

---

## 6. Track A vs Track B ablation (α_ij estimation)

Not run in this phase: Track A needs a white-box model's attention matrices, which are not wired, so only the Track B nli_proxy surrogate executed and there is nothing to correlate it against. Remains open → DECISIONS.md D-002.

---

## 7. Monitor placement approximation quality

Not run in this phase beyond the existing unit-level greedy-vs-brute-force test on synthetic DAGs (in the pytest suite). The eval harness exercised placement only incidentally (every trace shares one template, so S* is computed once per engine). Remains open.

---

## 8. Honest gap analysis

| Metric | Paper | Reproduced | Gap | Likely cause |
|---|---|---|---|---|
| Detection improvement vs single-step baseline | 34.7% | 0.0% (undetected-rate reduction) | −34.7 pp | Degenerate all-alert operating point (see below); zero-upstream-risk trace shape; no real NLI/SE models |
| Latency overhead | 12 ms/step | ~0.9 ms/step | better than claimed — but not comparable | HashingEmbedder ≈ free vs the paper's real embedding/NLI backends |

**The finding that matters most: the raw signal is inverted on this slice.** Per-step H on the synthesize step (all 150 traces): mean H = **0.4603 ± 0.1217 on hallucinated answers vs 0.7179 ± 0.1623 on right answers** (gap **−0.2576**). Under Eq. 4's semantics (high H = unfaithful = hallucinated) the detector should score hallucinated answers HIGHER — it scores them LOWER. Mechanism: gold answers are short entity strings (mean 13 chars, e.g. "Athletic Bilbao") whose trigrams are a tiny fraction of a ~200–1500-char knowledge passage, so bag-of-trigrams cosine is length-dominated and low; hallucinated answers are sentence-form (mean 66 chars) heavily copied from the same knowledge passage, so their overlap is high. H here mostly measures **answer length + verbatim reuse**, not faithfulness. Consequences: the F1-optimal threshold flags everything (F1 0.6667 = flag-all), and a tuned-threshold baseline cannot beat that either.

This is a **placeholder-embedder artifact, not evidence against the framework** — D-005 already forbids trusting H numbers from the hashing backend. It does mean this run's detection metrics are uninterpretable as model quality.

**What would change with V2-1 real models (bge/e5-class embedder, real NLI for α, SE wired):** (1) cosine faithfulness stops being length-dominated — semantic similarity of a short correct entity to its supporting passage is genuinely high; (2) α_ij becomes a real entailment probability instead of lexical cosine; (3) SE gives CHRS a second, decorrelated signal so λ actually trades two meaningful terms — making the sweep identifiable instead of a plateau; (4) richer trace shapes (real retrieve→reason→synthesize transcripts with non-zero upstream H) would let H_cascade's propagation term matter at all; (5) latency must be re-measured — the 12 ms/step budget has only been tested against a near-free embedder.

---

## 9. Summary

The Phase 14 harness is real end-to-end: a committed, provenance-stamped 150-record HaluEval-RAG slice (cut from the official upstream `qa_data.json` after `halu_eval_rag.json` 404'd — documented, not silently substituted), a real loader into labeled `PipelineGraph` traces, a λ grid sweep with a fixed calibrated-κ alert rule, a tuned single-step head-to-head, the Theorem 1 CHAF check on real data (150 traces, 0 violations, max ratio 1.0), and per-trace latency (~2.7 ms/trace ≈ 0.9 ms/step). On the substance: with the placeholder hashing embedder the per-step H signal is inverted (length/verbatim confound), both detectors degenerate to flag-everything (F1 0.6667, FAR 1.0, 0% improvement over the single-step baseline vs the paper's 34.7%), and λ is not identifiable on the plateau — so the most defensible one-sentence summary is: **"the pipeline runs on real data with the Theorem 1 invariant intact and honest instrumentation in place, but no detection-quality claim is supported until a semantic embedder and real entailment/SE models are wired (V2-1)."** The winning λ = 0.2 is logged in DECISIONS.md D-001 as provisional, sample-scale, and effectively a tie across λ ∈ [0.2, 1.0].
