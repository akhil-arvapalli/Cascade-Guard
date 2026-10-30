# CascadeGuard — In-Depth Project Analysis

> A complete read of the repo: what it is, the project idea, system design, backend
> architecture, data flow, diagrams, current state, how to run it, and what's left.
> Local reference doc only — not committed, not pushed.

---

## 1. What it is

**CascadeGuard** is a Python implementation of a research paper: *"A Real-Time
Pipeline-Aware Hallucination Detection Framework for Multi-Step Agentic RAG
Systems"* (Arvapalli Venkata Sai Akhil). It detects hallucinations that
**propagate through multi-step AI agent pipelines** — a failure mode that
single-step detectors are mathematically blind to.

- **Framework:** given a multi-step agent run (a DAG of steps: plan, retrieve,
  reason, tool_call, synthesize), it computes a per-step **cascade risk score**
  and runs a real-time online detector (CUSUM) that fires an alert when risk
  crosses a threshold — with root-cause localization pointing at the *origin* of
  the hallucination, not just the step where it surfaced.
- **Key claim:** it catches cascading failures that single-step detectors miss,
  and it does this within a latency budget by only spending expensive compute
  (semantic entropy, attention analysis) on a budget-selected subset of steps.

---

## 2. Project idea / the problem it solves

### The failure mode: cascading hallucination

Single-step hallucination detectors (RAGAS, RAGTruth, SelfCheckGPT, FActScore,
ARES) check whether a step's *output* is faithful to *that step's own retrieved
context*. They are blind to this:

1. Step `s_i` produces a hallucination (output not grounded in its context).
2. Step `s_j` consumes `s_i`'s output as **part of its own context**.
3. `s_j`'s output looks perfectly faithful **to its own (corrupted) context**.

So a single-step detector reports a clean bill of health on an output that is
actually built on a false premise several steps upstream. **The lie got laundered
through the pipeline.**

### Why it matters (Theorem 1)

The paper proves this isn't a minor edge case:

> **Theorem 1:** For a pipeline of depth `k`, single-step detectors can
> underestimate true hallucination risk by up to `(k+1)×`.

This is why the framework is *pipeline-aware*: risk must be **propagated along
the DAG edges**, weighted by how strongly each downstream step leans on each
upstream step's output (`α_ij`, the propagation coefficient).

### The design constraints (what makes it hard)

1. **Real-time:** detection must run within a latency budget (target 12ms/step),
   not after the fact.
2. **Bounded false alarms:** use CUSUM with a formal `ARL0` (average run length
   to false alarm) target — e.g. `ARL0=500` → <0.2% false-alarm rate.
3. **Compute-constrained:** the most accurate signals (attention weights,
   semantic entropy's K extra LLM calls) are expensive. **Don't run them
   everywhere** — pick which steps to monitor under a budget (NP-hard → greedy
   `(1−1/e)` approximation).

---

## 3. System design & backend architecture

```
   Agent (LangChain / LlamaIndex / custom loop)
        │   record_step() per step (trace events: output, docs, tool calls, attention*)
        ▼
┌───────────────────────────────┐      POST /v1/runs, /runs/{id}/steps,
│   SDK  (sdk/)                  │      /runs/{id}/analyze
│   StepRecorder                 ├───────────────────────────────────────┐
│   - builds the PipelineGraph   │                                       ▼
│   - infers edges heuristically │              ┌─────────────────────────────────────┐
│     for dynamic agents         │              │  FastAPI Core Engine (api/)           │
└───────────────────────────────┘              │  CascadeGuardEngine.analyze_run()      │
                                               └──────────────────┬──────────────────────┘
                                                                  │
   THE MATH PIPELINE (core/, in execution order):                 ▼
   ┌──────────────────────┐   H(s_i) = 1 − cos(o_i, R(s_i))      Eq. 3–4
   │ 1. Faithfulness      │   cheap, per-step                       core/scoring/faithfulness.py
   ├──────────────────────┤
   │ 2. Propagation  α_ij │   per edge, 3 tracks (tagged method):    core/scoring/propagation.py
   │                      │   A: attention  (exact Eq. 6, open-weight model)
   │                      │   B: NLI proxy / counterfactual (closed APIs)
   ├──────────────────────┤
   │ 3. Cascade Score     │   H_cascade(s_j) = H(s_j) + Σ α_ij·H_cascade(s_i)
   │    + CHAF            │   via ONE topological pass; CHAF = H_c/H   Eq. 5, 7
   │                      │   + Theorem 1 guard: CHAF ≤ depth+1        core/scoring/cascade_score.py
   ├──────────────────────┤
   │ 4. Placement S*      │   greedy Δrisk/cost under budget B (NP-hard, 1−1/e)
   │                      │   picks WHICH steps get expensive monitoring
   │                      │                                           core/placement/greedy_optimizer.py
   ├──────────────────────┤
   │ 5. Semantic Entropy  │   SE(s_i): K LLM samples clustered by
   │                      │   bidirectional NLI — ONLY on S*           core/scoring/semantic_entropy.py
   │                      │   (raises UnmonitoredStepError otherwise)
   ├──────────────────────┤
   │ 6. Risk Combiner     │   CHRS = λ·H_cascade + (1−λ)·SE           Eq. 10  core/scoring/risk_combiner.py
   ├──────────────────────┤
   │ 7. CUSUM Detector    │   online, S_n accumulates; alert if S_n > h
   │                      │   per-run state (pool keyed by run_id)     Eq. 11–12  core/detection/cusum.py
   ├──────────────────────┤
   │ 8. Localization      │   backward attribution to find the ORIGIN  core/localization/root_cause.py
   └──────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Storage (storage/) — TraceRepository protocol                │
   │  - InMemoryTraceRepository (v1 default)                      │
   │  - PostgresTraceRepository (schema.sql, [postgres] extra)    │
   └──────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Alerting (core/alerts.py) — pluggable sinks                 │
   │  LogSink / WebhookSink / SlackSink, fire-once-per-run        │
   └──────────────────────────────────────────────────────────────┘
```

### Module map (file → responsibility)

| Module | File | Responsibility |
|---|---|---|
| Engine | `core/engine.py` | Orchestrates the whole pipeline per run; `analyze_run()` |
| Graph | `core/graph/pipeline_graph.py` | `Step`, `PipelineGraph`, Kahn topo sort, `depth()`, dynamic edge inference |
| Faithfulness | `core/scoring/faithfulness.py` | Eq. 3–4: `faithfulness`, `hallucination_score` |
| Propagation | `core/scoring/propagation.py` | Eq. 6: `propagation_coefficient_attention`, NLI/counterfactual surrogates |
| Cascade score | `core/scoring/cascade_score.py` | Eq. 5 (`compute_h_cascade`), Eq. 7 (`compute_chaf`), Theorem 1 check |
| Semantic entropy | `core/scoring/semantic_entropy.py` | Eq. 9, bidirectional-NLI clustering, S* budget guard |
| Risk combiner | `core/scoring/risk_combiner.py` | Eq. 10: `chrs()` |
| CUSUM | `core/detection/cusum.py` | Eq. 11–12: detector, control-limit inversion, `calibrate()`, per-run pool |
| Placement | `core/placement/greedy_optimizer.py` | Eq. 13: greedy S*, brute-force validation oracle |
| Localization | `core/localization/root_cause.py` | Backward attribution via dynamic programming |
| Embeddings | `core/embeddings.py` | `HashingEmbedder` (default), `SentenceTransformerEmbedder` |
| Alerts | `core/alerts.py` | Pluggable sinks + fire-once dispatcher |
| API | `api/main.py`, `api/routes/*` | FastAPI: ingest, analyze, cheap-check, alerts, calibration |
| SDK | `sdk/step_recorder.py`, `sdk/integrations/*` | In-process instrumentation |
| Storage | `storage/repository.py`, `storage/schema.sql` | Trace repository + Postgres schema |
| Eval | `eval/run_eval.py`, `eval/synthetic.py`, `eval/baselines/` | Harness + synthetic traces + baselines |

---

## 4. Data flow / math pipeline (request lifecycle)

### The full analysis path (`CascadeGuardEngine.analyze_run`, core/engine.py:162)

```
            ┌─────────────────────────────────────────────────────────────┐
            │  INPUT: PipelineGraph (DAG of steps) for one run             │
            └─────────────────────────────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼──────────────────────────────────────┐
   │ 1. H(s_i)  — faithfulness everywhere (cheap embedding)   [Eq. 3–4]     │
   │    H = 1 − cos(embed(output), embed(retrieved_context))                │
   └────────────────────────────────┬──────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼──────────────────────────────────────┐
   │ 2. α_ij  — per edge, tagged by track              [Eq. 6 / surrogates] │
   │    track: attention | nli_proxy | counterfactual (config)              │
   │    clamped to [0, clamp_max] — keeps Theorem 1 sound                   │
   └────────────────────────────────┬──────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼──────────────────────────────────────┐
   │ 3. H_cascade  — single topological pass             [Eq. 5]            │
   │    H_c(s_j) = H(s_j) + Σ_{(i→j)} α_ij · H_c(s_i)                       │
   │ 4. CHAF = H_c / H  + Theorem 1 guard (CHAF ≤ depth+1)  [Eq. 7, Thm 1] │
   └────────────────────────────────┬──────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼──────────────────────────────────────┐
   │ 5. Placement S* — greedy under budget B  (offline, cached per template)│
   │    decides which steps get 'full' vs 'cheap' fidelity       [§II.H]    │
   └────────────────────────────────┬──────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼──────────────────────────────────────┐
   │ 6. SE(s_i) — semantic entropy ONLY on steps ∈ S*        [Eq. 9]        │
   │    K samples → bidirectional-NLI clusters → entropy                     │
   └────────────────────────────────┬──────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼──────────────────────────────────────┐
   │ 7. CHRS = λ·H_cascade + (1−λ)·SE                [Eq. 10]                │
   │    (cheap steps: SE = 0)                                                │
   └────────────────────────────────┬──────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼──────────────────────────────────────┐
   │ 8. CUSUM online update per step in topo order      [Eq. 11–12]          │
   │    S_n = max(0, S_n + CHRS − κ); alert if S_n > h                       │
   └────────────────────────────────┬──────────────────────────────────────┘
                                    │
   ┌────────────────────────────────▼──────────────────────────────────────┐
   │ 9. On alert → localize root cause (backward attribution)               │
   │    → dispatch to sinks (once per run)                                   │
   └─────────────────────────────────────────────────────────────────────────┘
```

### Online (real-time) vs offline (periodic)

- **Per-request hot path:** H → α → H_cascade → CHAF → CHRS → CUSUM. Uses cached
  placement `S*`. Cheap.
- **Offline / periodic (nightly):** placement refresh (recompute `S*`),
  CUSUM calibration (`μ0`, `σ` from clean traces), λ tuning (cross-validation).

---

## 5. Data model (trace store)

From `storage/schema.sql` (Postgres + pgvector):

```sql
pipeline_runs  ( run_id PK, pipeline_template_id, started_at, completed_at, status )
steps          ( step_id PK, run_id FK, kind, output, output_embedding VECTOR(768),
                 retrieved_doc_ids TEXT[], tool_call JSONB,
                 h_score, h_cascade, chaf, semantic_entropy, chrs,
                 monitored_fidelity 'cheap'|'full', created_at )
step_edges     ( run_id FK, from_step FK, to_step FK, alpha, alpha_method,
                 PK (from_step, to_step) )
cusum_states   ( run_id PK FK, s_n, kappa, h_control_limit, updated_at )
alerts         ( alert_id PK, run_id FK, triggered_at_step FK, s_n_value,
                 root_cause_step, created_at )
```

Key design point: every `step` row records **which fidelity** it was monitored at
(`cheap` vs `full`), and every edge records **which α track** produced it
(`attention`/`nli_proxy`/`counterfactual`) — so evaluation can separate the
populations and ablation results stay honest.

---

## 6. Current state (what's real vs. fake)

**v1 is code-complete and tested, but deliberately runs on fakes.**

| Real / verified | Placeholder / fake |
|---|---|
| All math Eq. 1–13 implemented + hand-computed unit tests | Embeddings = `HashingEmbedder` (char-trigrams, **not semantic**) |
| Graph engine, cascade scoring, Theorem 1 runtime guard | NLI/LLM hooks = injectable stubs (`entail_fn=lambda a,b: 0.5`) |
| CUSUM with ARL0 simulation, greedy placement + brute-force oracle, root-cause DP | Storage = in-memory dicts (Postgres schema written, unused) |
| FastAPI, SDK, alert sinks, eval harness | λ = 0.6 is a **placeholder** (D-001 undecided) |
| Test suite incl. Theorem 1 invariant, ARL0 sim, placement-ratio test | No real dataset eval (Phase 14 blocked) — `EVAL_REPORT.md` all PENDING |

### Documented approximations (AGENTS.md §3, DECISIONS.md)

- **Eq. 12 (ARL0)** is an approximation; empirically conservative (real false
  alarms lower than the <0.2% claim). Keep `δ ~ σ/2` — at extreme gaps the
  detector effectively never fires.
- **h is in σ-standardized units** and rescaled by σ before comparing to raw S_n
  (was a real bug caught by simulation).
- **Track A span alignment** (`idx(o_i)` in c_j) currently attributes the full
  context width — needs tokenizer-exact alignment (TODO Phase 3 / V2-1).
- **Placement marginal_risk_fn** is a structural descendant-mass **proxy**, not
  the trace-simulated version (needs historical data, TODO Phase 8).
- **μ0/σ** CUSUM defaults are placeholders until `calibrate()` runs on clean
  traces.

### Decision log status (DECISIONS.md)

| Decision | Status |
|---|---|
| D-001 λ (risk combiner weight) | **Undecided** — needs cross-validation sweep |
| D-002 α track production default | **Undecided** — needs Track A/B correlation data |
| D-003 embedding model (bge vs e5) | **Undecided** — needs benchmark |
| D-004 streaming transport | Provisionally Redis Streams |
| D-005 hashing embedder default | Proposed (revisit with benchmarks) |
| D-006 Track A span alignment deferred | Decided for v1 |
| D-007 in-memory storage default | Proposed |

---

## 7. How to run it

### Prerequisites
- Python 3.10+ (CI runs 3.10 and 3.12)
- Docker (for Postgres/Redis; optional — the app runs on in-memory storage by default)

### Install
```bash
# from repo root
pip install -e ".[dev]"          # core + dev (pytest, ruff, black, mypy)
# optional extras:
pip install -e ".[embeddings]"   # real sentence-transformers embedder
pip install -e ".[postgres]"     # Postgres + pgvector repository
pip install -e ".[nli]"          # transformers/torch for NLI + Track B
pip install -e ".[langchain]"    # LangChain integration
pip install -e ".[llamaindex]"   # LlamaIndex integration
```

### Local infra (optional)
```bash
docker compose up -d        # Postgres (pgvector) + Redis
# config .env from .env.example, then set storage.backend=postgres in configs/default.yaml
```

### Run the tests / lint / type-check (matches CI)
```bash
pytest -q                   # run the test suite
ruff check .                # lint
black --check core sdk api storage eval tests   # format check
mypy                        # type-check
```

### Run the API server
```bash
uvicorn api.main:app --reload
```
Default endpoints (`api/routes/ingest.py`, `alerts.py`, `calibration.py`):

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/runs` | Create a run (get `run_id`) |
| POST | `/v1/runs/{run_id}/steps` | Add a step (kind, output, retrieved_docs, predecessors) |
| POST | `/v1/runs/{run_id}/analyze` | Full pipeline analysis → scores, CUSUM, alert dispatch |
| POST | `/v1/cheap_check/{run_id}/{step_id}` | Sub-12ms cached CHRS lookup |
| GET | `/v1/runs/{run_id}` | Run record + alert status |
| GET | `/v1/alerts` | List alerts (optionally filtered by run) |
| POST | `/v1/calibration/cusum` | Calibrate CUSUM μ0/σ from clean CHRS values |
| POST | `/v1/placement/refresh` | Recompute monitor placement S* for a template |

### Run the eval harness
```bash
python -m eval.run_eval      # or: python eval/run_eval.py
```
Runs the checks that need no external dataset/model: Theorem 1 sweep across
random DAGs, CUSUM ARL0 simulation vs. Eq. 12 target, and the paper's Fig. 1
amplification demo. External-dataset eval (HaluEval-RAG, RAGTruth, AgentHallu)
is blocked until datasets are fetched.

### SDK usage sketch
```python
from sdk.step_recorder import StepRecorder
from core.embeddings import HashingEmbedder
from core.engine import CascadeGuardEngine

rec = StepRecorder(template_id="my_pipeline")
rec.record_step("s1", "retrieve", output="...", retrieved_docs=[...])
rec.record_step("s2", "reason", output="...", predecessors=["s1"])
graph = rec.finalize(infer_edges_with_embeddings=HashingEmbedder())

engine = CascadeGuardEngine(config={}, embedder=HashingEmbedder(64))
result = engine.analyze_run(graph)          # .H, .h_cascade, .chaf, .chrs, .alerts
```

---

## 8. Roadmap / what's left

### TODO.md — remaining open phases
- **Phase 0:** self-hosted embedding server; observability/OpenTelemetry baseline
- **Phase 3:** Track A token alignment + Track A/B correlation; Track B model wiring
- **Phase 5:** SE unit tests (fully-consistent → SE≈0, diverse → SE≈log K)
- **Phase 8:** per-step-kind cost calibration (empirical, not guesses)
- **Phase 10–12 (partial):** full framework adapters, alerting hardening
- **Phase 13:** dashboard (untouched)
- **Phase 14:** dataset eval (blocked on downloads), λ tuning
- **Phase 15:** hardening

### V2_PLAN.md — "make it real"
- **V2-1:** real model backends (embedder bake-off, NLI wiring, LLM sampling hook, Track A attention path + tokenizer-exact alignment)
- **V2-2:** latency engineering (embedding inference server, batched scoring, async ingestion via Redis Streams, Redis caches, load-test harness) — makes the 12ms claim measured, not assumed
- **V2-3:** data-plane hardening (Postgres activation, pgvector writes, CUSUM state durability + restart recovery, in-band blocking mode, failure modes)
- **V2-4:** learning loop (make calibration/placement refresh real, close λ)
- **V2-5:** real eval + ablations

### The "in-depth project idea" in one line
Turn a research paper's formally-proven *pipeline-aware hallucination detection*
into a production-grade, latency-bounded, budget-aware observability layer that
any multi-step agent (open- or closed-model) can plug into — validated end-to-end
against the paper's 34.7% / 12ms claims on real datasets.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| `P = (G, L, R)` | An agentic RAG pipeline: DAG `G`, LLM `L`, retriever `R` |
| `s_i` | A pipeline step (node); kinds: plan, retrieve, reason, tool_call, synthesize |
| `o_i` | Step `s_i`'s output |
| `R(s_i)` | The set of documents `s_i` retrieved |
| `c_i` | Step `s_i`'s context = `R(s_i) ∪ {o_j | (s_j, s_i) ∈ E}` (Eq. 1) |
| `F(s_i)` | Faithfulness = cos(output, context) (Eq. 3) |
| `H(s_i)` | Hallucination score = `1 − F(s_i)` (Eq. 4) |
| `α_ij` | Propagation coefficient: how strongly `s_j` leans on `s_i`'s output (Eq. 6 / surrogates) |
| `H_cascade(s_j)` | Cascading hallucination score = own H + α-weighted upstream H_cascade (Eq. 5) |
| `CHAF(s_j)` | Cascade amplification factor = `H_cascade / H` (Eq. 7); bounded by `depth+1` (Thm. 1) |
| `SE(s_i)` | Semantic entropy (Eq. 9); K-sample diversity of the model's answers |
| `CHRS(s_i)` | Combined risk = `λ·H_cascade + (1−λ)·SE` (Eq. 10) |
| `S*` | Budget-constrained monitor placement set (NP-hard, greedy 1−1/e) |
| CUSUM | Online change-detection statistic, `S_n` (Eq. 11); control limit `h` (Eq. 12) |
| `ARL0` | Average run length to false alarm; target 500 → <0.2% false alarms |
| `μ0, σ` | Baseline CHRS mean/stdev under "no hallucination" (from `calibrate()`) |
