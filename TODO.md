# CascadeGuard — Build TODO

Phased implementation plan. Each phase has a concrete exit criterion — don't move to the next phase until the current one's criterion is actually met, since later phases (CUSUM, monitor placement) depend on earlier ones (H_cascade, CHRS) being numerically correct, not just "written."

---

## Phase 0 — Project setup

- [x] Initialize repo with structure from README §8
- [x] Set up `configs/default.yaml` (README §9) with placeholder values
- [x] Set up Postgres + pgvector locally (docker-compose) using schema in README §5
- [x] Set up Redis (streams + cache) locally via docker-compose
- [ ] Pick and stand up a self-hosted embedding model (bge-base-en-v1.5 or e5-base) behind a thin local inference server (ONNX or vLLM) — confirm p50 latency for a single embed call, this number matters later for the 12ms/step budget
- [x] CI skeleton: lint (ruff/black), type-check (mypy), test runner (pytest)
- [ ] Decide logging/observability baseline (structured logs + OpenTelemetry traces from day one — retrofitting tracing later is painful)

**Exit criterion:** `docker-compose up` brings up Postgres, Redis, and the embedding server; a smoke test can embed a string and write a row to `steps`.

---

## Phase 1 — Pipeline Graph (§4.1)

- [x] Implement `Step` and `PipelineGraph` classes (README §4.1)
- [x] Implement `topo_order()` — Kahn's algorithm, must raise clearly on cycles (a DAG violation should be a loud error, not silently ignored, since Eq. 5's evaluation order depends on it)
- [x] Implement `depth(node_id)` — longest path from any source node
- [x] Implement static-template edge construction (for LangGraph-style pipelines with an explicit graph definition) — this path should be exact, not heuristic
- [x] Implement dynamic edge inference heuristic for free-form ReAct-style loops:
  - [x] n-gram overlap check between `o_i` and `c_j`
  - [x] embedding-similarity fallback when overlap is low (paraphrased reuse)
  - [x] configurable similarity threshold
- [x] Unit tests: linear chain, branching DAG (fan-out/fan-in), single-node pipeline, cycle-detection failure case
- [x] Unit test: verify `depth()` matches hand-computed values on 5+ synthetic DAGs

**Exit criterion:** can ingest a real LangChain/LangGraph trace and produce a correct `PipelineGraph` with accurate edges and depths, verified against 3 manually-annotated example traces.

---

## Phase 2 — Faithfulness scoring: `F(s_i)`, `H(s_i)` (§4.2)

- [x] Implement `faithfulness()` and `hallucination_score()` per Eq. 3–4
- [x] Implement retrieval-set concatenation + chunk-averaging for long `R(s_i)` (don't silently truncate)
- [x] Wire into `PipelineGraph` so every `Step` gets an `H(s_i)` on ingestion
- [ ] Benchmark embedding call latency at realistic batch sizes — this is the first real data point toward the 12ms/step target
- [x] Unit tests against hand-constructed cases: near-identical output/context (H≈0), completely unrelated output/context (H≈1)

**Exit criterion:** `H(s_i)` computed correctly and fast enough to not dominate the latency budget on its own.

---

## Phase 3 — Propagation coefficient: `α_ij` (§4.3)

This phase has two independent tracks — do not treat Track B as "good enough, skip Track A." Track A is what lets you validate Track B later.

- [ ] **Track A (white-box/attention):**
  - [ ] Stand up a self-hosted open-weight model (e.g. Llama/Mistral-class) via HF Transformers with `output_attentions=True`
  - [x] Implement `propagation_coefficient_attention()` exactly per Eq. 6
  - [ ] Verify token-index alignment between `o_i` token positions and `c_j` token positions — this is the easiest place to introduce a silent off-by-one bug that corrupts every downstream score
  - [ ] Unit test: construct a synthetic case where `s_j`'s output obviously and heavily quotes `s_i`'s output — assert `α_ij` comes out high; construct a case with no relation — assert it comes out low
- [ ] **Track B (black-box surrogate):**
  - [x] Implement NLI-entailment-based proxy (pick and wire an NLI model)
  - [x] Implement counterfactual-perturbation proxy (mask `o_i`, re-run `s_j`, measure output divergence)
  - [x] Add `expensive_mode` config gate for the counterfactual variant
  - [x] Tag every computed `α_ij` with its `alpha_method` (README §5 schema) — never leave this unlabeled
- [ ] **Cross-validation between tracks:** on the self-hosted model (where both tracks are available), run both and compute correlation between Track A and each Track B method — this number is a load-bearing part of the eval story, not a nice-to-have

**Exit criterion:** both tracks produce `α_ij ∈ [0,1]`, clamped and validated; correlation analysis between tracks completed and documented.

---

## Phase 4 — Cascade Score Engine: `H_cascade`, `CHAF` (§4.4)

- [x] Implement `compute_h_cascade()` via single topological pass (Eq. 5)
- [x] Implement `compute_chaf()` (Eq. 7)
- [x] Implement the **Theorem 1 invariant check** as an assertable test utility: `CHAF(s_j) <= depth(s_j) + 1` for every node — run this on every synthetic and real trace in the test suite
- [x] Performance test: confirm `O(|V|+|E|)` scaling empirically on synthetic DAGs up to realistic production sizes (e.g. 50, 200, 1000 steps)
- [x] Unit tests: reproduce the paper's Fig. 1 example by hand (hallucination at `s_3`, amplified at `s_4`, `s_5`) and confirm the engine's output matches the expected qualitative pattern

**Exit criterion:** Theorem 1 invariant holds with zero violations across the full test suite; a deliberately-corrupted-early-step synthetic trace shows the expected amplification in `H_cascade` at downstream nodes.

---

## Phase 5 — Semantic Entropy: `SE(s_i)` (§4.5)

- [x] Implement K-sample generation against the target LLM (temperature > 0)
- [x] Implement semantic clustering via bidirectional NLI entailment (not naive embedding k-means — match the paper's definition)
- [x] Implement `semantic_entropy()` per Eq. 9
- [x] Cost-guard: make `K` and the "only run when selected by placement optimizer" rule enforced at the call site, not just documented
- [ ] Unit tests: fully consistent samples → SE≈0; maximally diverse samples → SE near `log(K)`

**Exit criterion:** SE computed correctly, and demonstrably *not* invoked on unmonitored steps in an integration test.

---

## Phase 6 — Risk Combiner: `CHRS` (§4.6)

- [x] Implement `chrs()` per Eq. 10
- [x] Wire `λ` to config (README §9) — do not hardcode
- [x] Integration test: full pipeline from raw trace → `H(s_i)` → `H_cascade` → `SE` (where monitored) → `CHRS`, end to end on a synthetic multi-step trace

**Exit criterion:** a synthetic trace produces a correct, inspectable `CHRS` value at every node, with intermediate values (H, H_cascade, CHAF, SE) all persisted for debugging.

---

## Phase 7 — CUSUM Detector (§4.7)

- [x] Implement `CusumDetector` class with `update()` per Eq. 11
- [x] Implement `_solve_control_limit()` — numerically invert Eq. 12 (bisection is fine) to get `h` from a target `ARL0`
- [x] Implement `calibrate()`: given a labeled-clean historical trace set, compute `mu0`, `sigma` empirically — do not hardcode these
- [x] Implement per-run detector state management (keyed by `run_id`, persisted to `cusum_states` table, garbage-collected on run completion)
- [x] Simulation test: generate synthetic CHRS streams from `N(mu0, sigma^2)` (no shift) and confirm empirical `ARL0` roughly matches the target from Eq. 12 over many simulated runs
- [x] Simulation test: inject a mean shift (simulating a real cascading hallucination) and confirm the detector fires within a reasonable number of steps, and measure detection delay

**Exit criterion:** empirical false-alarm rate from simulation is within a reasonable tolerance of the theoretical `ARL0` target; shift-detection simulation shows the detector reliably catches injected anomalies.

---

## Phase 8 — Monitor Placement Optimizer (§4.8)

- [x] Implement `greedy_monitor_placement()` per the paper's submodular greedy approach
- [ ] Implement per-step-kind cost calibration (`weights[s]`) from empirical latency/cost measurements taken in Phases 2–5, not guesses
- [ ] Implement `marginal_risk_fn` using historical trace simulation (with/without monitor `s`)
- [ ] Implement the offline/nightly refresh job that recomputes `S*` per pipeline template
- [x] Validation: on small synthetic DAGs (≤12 nodes) where brute-force optimal `S*` is computable, compare greedy vs. brute-force and confirm the empirical approximation ratio is consistent with the theoretical `(1-1/e) ≈ 63%` bound
- [ ] Wire the live request path to look up precomputed `S*` from cache (Redis) rather than recomputing per-request

**Exit criterion:** greedy placement demonstrably within the theoretical approximation bound on validation DAGs; live lookup path confirmed sub-millisecond.

---

## Phase 9 — Root-cause localization

- [x] Implement backward walk from an alerted node, ranking predecessors by `α_ij · H_cascade(s_i)` contribution (README §5)
- [x] Return top-N contributing upstream steps with their contribution share
- [x] Unit test against the Phase 4 Fig.1-style synthetic trace — confirm `s_3` is correctly identified as the root cause of an alert fired at `s_5`

**Exit criterion:** on synthetic multi-hop cascades, root-cause localization correctly points at the true origin step, not just the alerting step.

---

## Phase 10 — SDK / instrumentation

- [x] Implement `StepRecorder` (in-process, async, non-blocking emission)
- [x] Implement LangChain integration (callback handler that emits `Step` events)
- [x] Implement LlamaIndex integration
- [x] Implement a minimal manual/generic integration path for custom agent loops that don't use either framework
- [ ] Confirm SDK overhead on the agent's critical path is negligible (measure p50/p99 added latency from instrumentation alone, separate from any detection compute)

**Exit criterion:** a real LangChain agent, wrapped with the SDK, produces correctly-populated traces in the trace store with no observable behavior change to the agent itself.

---

## Phase 11 — Core Engine service + API

- [x] Stand up FastAPI app (`api/main.py`)
- [ ] Implement trace ingestion endpoint (`/ingest`) consuming from Redis Streams
- [x] Implement synchronous "cheap check" endpoint the agent can call inline (cache-backed, must hit the <12ms/step target)
- [ ] Implement async full-pipeline analysis worker (consumes queue, runs full scoring pipeline, writes to Postgres)
- [x] Implement `/calibration` endpoints (trigger CUSUM calibration, trigger monitor-placement refresh)
- [x] Implement `/alerts` endpoints (list, acknowledge, webhook registration)
- [ ] Load test: confirm the sync cheap-check path holds the latency budget under realistic concurrent load

**Exit criterion:** end-to-end path — agent step → SDK → queue → engine → stored score → (if applicable) alert fired — works under load-tested concurrency with the sync path meeting the 12ms budget.

---

## Phase 12 — Alerting

- [x] Implement pluggable alert sink interface
- [x] Implement Slack webhook sink
- [x] Implement generic webhook sink
- [ ] Implement (optional) in-band blocking mode — pipeline pauses/short-circuits on alert rather than only notifying after the fact
- [x] Test: alert fires exactly once per run (no duplicate/spam alerts from repeated CUSUM checks on the same run)

**Exit criterion:** alerts reliably reach at least one real sink (Slack) in an integration test, with root-cause info attached.

---

## Phase 13 — Dashboard

- [x] Grafana dashboards: false-alarm rate over time, detection latency distribution, per-pipeline-template CHAF distribution, monitor placement cost vs. budget utilization *(deferred to Phase 15/Postgres; metrics available via API)*
- [x] Custom DAG visualization view: render a single run's `PipelineGraph` with per-node `H`, `H_cascade`, `CHRS` overlaid, root-cause path highlighted when an alert exists *(React + Vite, served at /dashboard)*
- [ ] Basic auth/access control on the dashboard

**Exit criterion:** an engineer can open a specific alerted `run_id` and visually see which step was the root cause and how the score propagated.

---

## Phase 14 — Evaluation (README §6)

- [ ] Write dataset loaders for HaluEval-RAG, RAGTruth, AgentHallu → `PipelineGraph` trace format with ground-truth labels
- [ ] Implement/wrap RAGAS baseline
- [ ] Implement/wrap SelfCheckGPT baseline
- [ ] Run λ tuning sweep on held-out validation split (grid or Bayesian search), select final `λ`
- [ ] Run full evaluation: CascadeGuard vs. baselines on detection rate, false-alarm rate, latency, across all three datasets
- [ ] Run the CHAF-vs-depth empirical validation (Theorem 1 bound check) across the full dataset, not just synthetic tests
- [ ] Run Track A vs Track B ablation, report correlation and any detection-quality gap
- [ ] Write up results — compare against the paper's claimed 34.7% reduction / 12ms overhead and document any gap plus likely causes

**Exit criterion:** a written results report with numbers for every metric in README §6, including honest discussion of where the implementation over/under-performs the paper's claims and why.

---

## Phase 15 — Hardening / productionization

- [ ] Failure-mode handling: what happens if the embedding server is down, if the LLM call for semantic entropy times out, if Postgres is unreachable — every external dependency needs a defined degraded-mode behavior, not a silent crash
- [ ] Backpressure handling on the ingestion queue under burst load
- [ ] Data retention policy for `steps`/`step_edges` (traces are potentially large and sensitive — define retention and PII handling explicitly)
- [ ] Security review: trace data may contain sensitive user content (menu orders, PII, etc. depending on deployment) — access control on trace store and dashboard
- [ ] Documentation pass: make sure README config knobs (§9) all have accurate, tested defaults
- [ ] Runbook: what an on-call engineer does when an alert fires (root-cause view → triage → escalate/resolve)

**Exit criterion:** system survives a chaos test (kill embedding server, kill Postgres, burst 10x normal load) without silent data loss or crash, and degrades in a documented, predictable way.

---

## Suggested sequencing note

Phases 1–9 are the research-faithful core and should be built and validated in order — each depends on the previous being numerically correct (H_cascade depends on H and α; CUSUM depends on CHRS; placement depends on calibrated costs from earlier phases). Phases 10–13 (SDK, service, alerting, dashboard) can be parallelized once Phase 9 is stable, since they're mostly plumbing around an already-correct core. Phase 14 (evaluation) is the phase that actually proves the system matches the paper's claims — don't treat it as a final formality; budget real time for it, since λ tuning and the Track A/B ablation will likely surface issues worth fixing before Phase 15.
