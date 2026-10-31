# CascadeGuard — v2 Plan (Optimizations)

v1 proved the math is implemented faithfully and the pipeline works end-to-end on
synthetic traces. v2 makes it **real**: real models instead of injectable fakes,
real latencies instead of assumed ones, real data instead of synthetic, and the
learning loops (calibration, placement refresh) actually closing.

Order matters: **V2-1 unblocks V2-5** (no ablations without real models);
**V2-3 should land before V2-4** (the learning loop needs stored history).
V2-2 can proceed in parallel with everything once V2-1's embedder lands.

---

## V2-1 — Real model backends (replaces every injectable fake)

The v1 system runs on a hashing embedder and refuses to run α/SE without wired
models (`PropagationDependencyError` by design). This phase swaps in the real
ones and closes DECISIONS D-002/D-003/D-005.

- [ ] **Embedding backend bake-off**: bge-base-en-v1.5 vs e5-base served locally.
      Metrics: detection quality delta on synthetic + a HaluEval-RAG sample,
      single-call and batch-of-32 latency, memory footprint. Log winner in
      DECISIONS.md D-003 and flip `embeddings.backend`.
- [ ] **NLI model wiring**: `microsoft/deberta-large-mnli` (or comparable)
      behind `entail_fn`; measure per-call latency — this number becomes the
      calibrated Track B monitoring cost (feeds V2-4).
- [ ] **LLM sampling hook** for semantic entropy K-samples and counterfactual
      re-runs: OpenAI-compatible client adapter + self-hosted option; document
      cost-per-monitored-step with real numbers.
- [ ] **Track A attention path**: self-hosted open-weight model via HF
      Transformers `output_attentions=True` (vLLM if supported); implement
      **tokenizer-exact `idx(o_i)` span alignment** in
      `core/engine.py::_source_span_in_context`, replacing the
      full-context-width approximation. This is the highest-risk item in v2 —
      off-by-one here silently corrupts every α_ij (TODO Phase 3 warning), so:
  - [ ] Unit test alignment against hand-tokenized examples before any scoring runs
  - [ ] Property test: α ∈ [0,1] and Theorem 1 still holds on model-generated traces

**Exit criterion:** `analyze_run` on a real LangChain agent trace produces
scores from real models; span-alignment tests pass; D-003/D-005 closed.

---

## V2-2 — Latency engineering (the 12 ms budget, measured not assumed)

The paper claims 12 ms/step overhead. Nothing in v1 has been measured under
load — this phase makes the number real and defensible.

- [ ] **Embedding inference server**: ONNX Runtime (optimum) or vLLM embedding
      endpoint in docker-compose; replace in-process model loading for the API path.
- [ ] **Batched scoring**: score all steps of a run in one embedding batch
      instead of per-step calls (faithfulness is embarrassingly batchable).
- [ ] **Async ingestion**: move `/analyze` work into a worker consuming Redis
      Streams (README §7 start-simple path); API returns 202 + run status
      endpoint. Keep the synchronous path behind a config flag for small deploys.
- [ ] **Redis-backed caches**: cheap-check lookups and precomputed S* read
      through Redis with in-process fallback (completes TODO Phases 8/11 items).
- [ ] **Load test harness** (locust or k6 script committed under `eval/loadtest/`):
      p50/p95/p99 for cheap-check and analyze endpoints at realistic concurrency;
      report split by cheap-vs-full fidelity steps.
- [ ] **Latency regression gate** in CI with generous bounds so gross
      regressions fail loudly.

**Exit criterion:** published p50/p95 table showing cheap-check < 12 ms at
target concurrency; async path survives a burst test without dropping traces.

---

## V2-3 — Data plane hardening

v1 ships the Postgres schema but runs on the in-memory store. This phase makes
persistence real and closes Phase 0's never-executed exit criterion.

- [ ] **Postgres activation**: `docker-compose up` verified end-to-end (schema
      auto-applies); switch `storage.backend=postgres`; integration tests run
      against the real store in CI (service container).
- [ ] **pgvector storage** of output embeddings (schema column exists, nothing
      writes it yet); use for eval-side similar-trace inspection.
- [ ] **CUSUM state durability**: persist per-step S_n updates (not just final),
      restore a detector mid-run after restart; round-trip test.
- [x] **Lifecycle wiring**: runs marked completed/alerted on inline `/analyze`;
      worker path (`api/worker.py`) calls `pool.complete(run_id)` on run_end (the
      inline path keeps the detector pooled for post-analysis blocking decisions
      — documented trade-off; orphan cleanup = pending-message redelivery).
- [ ] **In-band blocking mode** (TODO Phase 12 leftover): decision endpoint the
      agent calls synchronously — "block this step?" — with configurable
      threshold policy separate from notify-only alerting.
- [ ] **Failure modes** (early Phase 15 scope): defined degraded behavior when
      embedding server is down (fail-open vs fail-closed flag), LLM timeout on
      SE (skip + mark step degraded), Postgres unreachable (buffer + flush).

**Exit criterion:** kill -9 the engine mid-run, restart, and the run resumes
with correct CUSUM state; chaos test passes without silent loss.

---

## V2-4 — Learning loop (make the offline jobs real)

Three v1 components are structurally correct but fed placeholders. This phase
closes their data loops.

- [ ] **Empirical `marginal_risk_fn`**: replay stored historical traces through
      the scorer with/without candidate monitors; replace
      `_marginal_mass` structural proxy in `core/engine.py`. Validate greedy
      still within (1−1/e) using the new objective.
- [ ] **Cost calibration**: replace placeholder `step_kind_costs` ratios with
      measured per-kind monitor costs from V2-1/V2-2 latency tables.
- [ ] **Placement refresh job**: scheduled (cron container or APScheduler)
      nightly recompute of S* per template from drifted traffic; writes to
      Redis; alert when S* changes materially (placement drift signal).
- [ ] **Calibration automation**: one command that pulls clean-labeled traces,
      runs `calibrate()`, persists μ0/σ, and hot-reloads detectors — no manual
      config edits.
- [ ] *(Research stretch, timeboxed)*: distilled α surrogate — train a small
      model on (o_i, c_j, attention-α) triples harvested from the Track A
      deployment; evaluate whether it beats NLI-proxy on correlation without
      needing white-box access. Flag results in DECISIONS.md D-002 either way.

**Exit criterion:** S*, μ0/σ, and costs all regenerate from stored data via
scheduled jobs; no placeholder numbers remain in configs/default.yaml.

---

## V2-5 — Evaluation completion (TODO Phase 14, the paper-proof phase)

Everything above exists to make this phase honest. The deliverable is a filled
EVAL_REPORT.md that survives interview scrutiny.

- [ ] **Dataset loaders**: HaluEval-RAG, RAGTruth, AgentHallu → PipelineGraph
      traces with per-step ground-truth labels (loaders currently raise loudly
      with instructions — replace with real converters + download scripts).
- [ ] **Baselines head-to-head**: RAGAS + SelfCheckGPT running locally on the
      same traces (wrappers exist; make them execute).
- [ ] **λ tuning sweep**: grid then Bayesian on held-out split optimizing
      detection F1; log winning value + sweep curve in DECISIONS.md D-001.
- [ ] **Headline metrics**: cascading-failure detection rate vs baselines,
      false-alarm rate vs Eq. 12 target, per-step latency overhead — compared
      against the paper's 34.7% / 12 ms claims with gap analysis.
- [ ] **CHAF-vs-depth empirical validation** across the full dataset (bound
      currently validated only on synthetic DAGs).
- [ ] **Track A vs B ablation**: correlation between tracks on open-model
      traces (only meaningful AFTER V2-1 span alignment lands).

**Exit criterion:** EVAL_REPORT.md contains real numbers for every metric in
README §6 plus honest discussion of over/under-performance vs the paper.

---

## Explicit non-goals for v2

- Dashboard/Grafana UI (Phase 13) — defer until numbers exist to display; the
  API + alerts carry operational value alone.
- Kafka migration — Redis Streams until measured throughput demands it (D-004).
- Multi-tenancy, auth hardening, PII retention policy (rest of Phase 15) —
  next major version, after the system earns production trust with real evals.
