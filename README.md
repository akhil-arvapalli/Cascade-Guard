# CascadeGuard

**A real-time, pipeline-aware hallucination detection framework for multi-step agentic RAG systems.**

CascadeGuard implements the framework from *"CascadeGuard: A Real-Time Pipeline-Aware Hallucination Detection Framework for Multi-Step Agentic RAG Systems"* (Arvapalli Venkata Sai Akhil, CMR College of Engineering and Technology). This document is the system design and build spec — it translates every formal contribution in the paper (Hcascade, CHAF, CHRS, CUSUM detection, budget-constrained monitor placement) into a concrete, buildable software architecture.

---

## 1. What problem this solves

Single-step hallucination detectors (RAGAS, RAGTruth, SelfCheckGPT, FActScore, ARES) check whether a step's output is faithful to *that step's own retrieved context*. They are architecturally blind to a specific failure mode: a hallucination produced at step `s_i` gets consumed as context by a downstream step `s_j`, and `s_j`'s output looks perfectly faithful *to its own (corrupted) context* — so single-step detectors report a clean bill of health on an output that is actually built on a false premise several steps upstream.

The paper proves this isn't a minor edge case: **Theorem 1** shows that for a pipeline of depth `k`, single-step detectors can underestimate true hallucination risk by up to `(k+1)×`. CascadeGuard exists to close that gap — in real time, within a production latency budget, with a formal bound on false-alarm rate, and with provably near-optimal placement of detectors under a compute budget.

---

## 2. Core concepts → system components

Every module in this system exists because a specific piece of the paper's math requires it. This is the map:

| Paper concept | Formula | System component |
|---|---|---|
| Agentic RAG pipeline | `P = (G, L, R)`, DAG `G=(V,E)` | **Pipeline Graph Builder** |
| Single-step faithfulness | `F(s_i)`, `H(s_i) = 1 - F(s_i)` | **Faithfulness Scorer** |
| Propagation coefficient | `α_ij` (cross-attention) | **Propagation Estimator** |
| Cascading Hallucination Score | `H_cascade(s_j)` | **Cascade Score Engine** |
| Amplification factor | `CHAF(s_j)` | **Cascade Score Engine** (derived metric) |
| Semantic entropy | `SE(s_i)` | **Semantic Entropy Module** |
| Combined risk | `CHRS(s_i)` | **Risk Combiner** |
| Sequential online detection | CUSUM `S_n`, `ARL_0` | **CUSUM Detector** |
| Budget-constrained placement | `S*` (NP-hard, greedy `1-1/e`) | **Monitor Placement Optimizer** |

---

## 3. High-level architecture

```
                         ┌─────────────────────────────────────────┐
                         │           Instrumented Agent             │
                         │  (LangChain / LlamaIndex / custom loop)  │
                         └───────────────────┬───────────────────────┘
                                              │ trace events (step start/end,
                                              │ retrieved docs, tool calls,
                                              │ output tokens, attention[optional])
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │         CascadeGuard SDK (in-process)     │
                         │  - StepRecorder                           │
                         │  - GraphBuilder (incremental DAG)         │
                         └───────────────────┬───────────────────────┘
                                              │ gRPC / async queue (Kafka or Redis Streams)
                                              ▼
        ┌─────────────────────────────────────────────────────────────────────┐
        │                        CascadeGuard Core Engine                       │
        │                                                                       │
        │  ┌────────────────┐   ┌────────────────────┐   ┌──────────────────┐  │
        │  │ Faithfulness   │   │ Propagation         │   │ Semantic Entropy │  │
        │  │ Scorer  F(s_i) │   │ Estimator  α_ij     │   │ Module  SE(s_i)  │  │
        │  └───────┬────────┘   └──────────┬──────────┘   └────────┬─────────┘  │
        │          │                       │                       │            │
        │          └──────────┬────────────┴───────────┬───────────┘            │
        │                     ▼                         ▼                       │
        │          ┌────────────────────┐   ┌────────────────────────┐          │
        │          │ Cascade Score      │   │ Risk Combiner           │          │
        │          │ Engine  H_cascade, │──▶│ CHRS = λH_c+(1-λ)SE     │          │
        │          │ CHAF (topo sort)   │   └───────────┬─────────────┘          │
        │          └────────────────────┘               ▼                       │
        │                                    ┌────────────────────────┐         │
        │                                    │ CUSUM Detector          │         │
        │                                    │ S_n, alert if S_n > h   │         │
        │                                    └───────────┬─────────────┘         │
        │                                                 ▼                       │
        │                                    ┌────────────────────────┐         │
        │                                    │ Monitor Placement       │         │
        │                                    │ Optimizer (greedy, S*)  │         │
        │                                    └────────────────────────┘         │
        └───────────────────────────────┬───────────────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
     ┌────────────────┐        ┌─────────────────┐        ┌──────────────────┐
     │ Trace Store     │        │ Alert Bus        │        │ Metrics/Dashboard │
     │ (Postgres +     │        │ (webhook, Slack, │        │ (Grafana/Prometheus│
     │ pgvector or     │        │ PagerDuty, or    │        │  + custom UI)     │
     │ Timescale)      │        │ in-band block)    │        │                   │
     └────────────────┘        └─────────────────┘        └──────────────────┘
```

**Deployment shape:** the SDK runs in-process with the agent (low overhead, no network hop for step capture). The Core Engine runs as a separate service so model/embedding compute doesn't block the agent's critical path — the agent gets a fast synchronous "should I block this step?" check (cache-backed, sub-12ms) plus an async full pipeline analysis that updates the DAG-level score after the fact.

---

## 4. Component design

### 4.1 Pipeline Graph Builder

Builds `G = (V, E)` incrementally as the agent executes. Each step `s_i` is a node; edges represent data-flow (`(s_i, s_j) ∈ E` iff `s_j`'s context `c_j` includes `s_i`'s output, per Eq. 1).

```python
class Step:
    id: str
    kind: Literal["plan", "retrieve", "reason", "tool_call", "synthesize"]
    output: str                    # o_i
    retrieved_docs: list[Document]  # R(s_i)
    predecessors: list[str]         # incoming edges
    tool_invocation: ToolCall | None
    attention_weights: np.ndarray | None  # A^(j), if accessible
    timestamp: float

class PipelineGraph:
    nodes: dict[str, Step]
    edges: dict[str, list[str]]     # adjacency list

    def add_step(self, step: Step) -> None: ...
    def topo_order(self) -> list[str]:
        """Kahn's algorithm — required for O(|V|+|E|) H_cascade evaluation (paper §II.C)."""
    def depth(self, node_id: str) -> int:
        """Longest path from any source to node_id — needed for CHAF bound (Theorem 1)."""
```

Edge construction rule (implements Eq. 1: `c_i = R(s_i) ∪ {o_j | (s_j, s_i) ∈ E}`):
- Static agents (fixed DAG templates, e.g. LangGraph): edges come directly from the graph definition.
- Dynamic agents (free-form ReAct loops): infer edges by checking whether `o_j` (or a paraphrase/embedding-similar span of it) appears in `c_i`. Use a lightweight containment/similarity heuristic (n-gram overlap + embedding cosine > threshold) rather than exact string match, since agents summarize/reformat prior outputs.

### 4.2 Faithfulness Scorer — `F(s_i)`, `H(s_i)`

Implements Eq. 3–4 directly:

```python
def faithfulness(o_i: str, r_i: str, embed_fn) -> float:
    v_o, v_r = embed_fn(o_i), embed_fn(r_i)
    return cosine_similarity(v_o, v_r)   # F(s_i)

def hallucination_score(o_i: str, r_i: str, embed_fn) -> float:
    return 1 - faithfulness(o_i, r_i, embed_fn)   # H(s_i) ∈ [0,1]
```

- `embed_fn`: production default = a sentence-embedding model served locally (e.g. a bge/e5-class model) for latency control — do not round-trip to an external embeddings API on the hot path.
- `r_i` is the concatenation of `R(s_i)` (Eq. 3) — cap concatenation length and chunk-average embeddings for long retrieval sets rather than truncating silently.

### 4.3 Propagation Estimator — `α_ij`

This is the hardest component to build faithfully because Eq. 6 requires the **cross-attention matrix** `A^(j)` of step `s_j` — i.e., white-box access to the generating model's attention weights. This works cleanly for self-hosted open-weight models (vLLM/HF with `output_attentions=True`) but **not** for closed API models (OpenAI, Anthropic, etc.), which don't expose attention.

Two implementation tracks, both must exist:

**Track A — White-box (exact, per Eq. 6):**
```python
def propagation_coefficient_attention(A_j: np.ndarray, output_token_idx: list[int],
                                        context_token_idx: list[int]) -> float:
    # A_j: [L_j, |c_j|] cross-attention matrix for step s_j
    L_j = A_j.shape[0]
    num = A_j[:, output_token_idx].sum()
    den = A_j.sum()
    return num / (L_j * den) if den > 0 else 0.0   # implements Eq. 6
```

**Track B — Black-box surrogate (for closed models):**
Since attention isn't available, approximate `α_ij` with a calibrated proxy that should be validated to correlate with Track A wherever both are measurable (during offline evaluation on open models):
- **NLI-entailment weight**: run an NLI model between `o_i` and `o_j`; use entailment probability as a proxy for "how much `s_j`'s output leans on `s_i`'s claim."
- **Counterfactual perturbation weight**: re-run `s_j` with `o_i` masked/replaced by a null placeholder in `c_j`, measure output divergence (embedding distance) between original and counterfactual `o_j`. Larger divergence ⇒ higher `α_ij`. More expensive (extra LLM call) but model-agnostic and doesn't require attention access — this should be the default for closed-model deployments, gated behind an `expensive_mode` flag since it adds latency.

Document this tradeoff prominently in code comments and configuration — silently substituting a proxy for the paper's exact formula without flagging it would misrepresent the guarantees.

### 4.4 Cascade Score Engine — `H_cascade`, `CHAF`

Direct implementation of Eq. 5, evaluated via single topological pass (paper's claimed `O(|V|+|E|)`):

```python
def compute_h_cascade(graph: PipelineGraph, alpha: dict[tuple[str,str], float],
                       H: dict[str, float]) -> dict[str, float]:
    h_cascade = {}
    for node_id in graph.topo_order():          # topological order guarantees
        total = H[node_id]                        # predecessors are resolved first
        for pred_id in graph.edges_into(node_id):
            total += alpha[(pred_id, node_id)] * h_cascade[pred_id]
        h_cascade[node_id] = total
    return h_cascade

def compute_chaf(h_cascade: dict[str, float], H: dict[str, float]) -> dict[str, float]:
    return {n: h_cascade[n] / H[n] if H[n] > 0 else 1.0 for n in h_cascade}
```

- Runtime guard: assert `CHAF(s_j) <= depth(s_j) + 1` (Theorem 1) in test/staging environments — a violation means `α_ij` estimation is broken (e.g. an unclamped value above 1), not that the theorem is wrong. This is a genuinely useful invariant test, not busywork.

### 4.5 Semantic Entropy Module — `SE(s_i)`

Implements Eq. 9 (Farquhar et al. semantic entropy, as cited in the paper):

```python
def semantic_entropy(prompt: str, llm, embed_fn, K: int = 10,
                      cluster_threshold: float = 0.85) -> float:
    samples = [llm.generate(prompt, temperature=1.0) for _ in range(K)]
    clusters = cluster_by_meaning(samples, embed_fn, cluster_threshold)  # bidirectional
                                                                          # entailment or
                                                                          # embedding clustering
    probs = [len(c) / K for c in clusters]
    return -sum(p * math.log(p) for p in probs if p > 0)
```

- `K` (sample count) is a latency/accuracy knob — this is the most expensive component (K extra LLM calls per monitored step). It should only run on steps selected by the Monitor Placement Optimizer (§4.7), never on every step unconditionally — that's the entire point of §II.H in the paper.
- Clustering by meaning: use bidirectional NLI entailment (cluster `a,b` together iff each entails the other) as the paper's semantic-equivalence-class method implies, not naive embedding k-means, to stay faithful to Farquhar et al.'s definition.

### 4.6 Risk Combiner — `CHRS`

```python
def chrs(h_cascade_i: float, se_i: float, lam: float) -> float:
    return lam * h_cascade_i + (1 - lam) * se_i   # Eq. 10
```

`λ` is a config value tuned via cross-validation (paper §II.F) — see §6 (Evaluation Plan) for how to actually tune it rather than guessing a value.

### 4.7 CUSUM Detector

Implements Eq. 11–12 as a genuine **online/streaming** algorithm — state must persist across pipeline steps within an execution, not be recomputed from scratch:

```python
class CusumDetector:
    def __init__(self, mu0: float, sigma: float, delta: float, arl0_target: float = 500):
        self.kappa = mu0 + delta / 2
        self.h = self._solve_control_limit(mu0, sigma, delta, arl0_target)  # invert Eq. 12
        self.S = 0.0

    def update(self, chrs_value: float) -> bool:
        self.S = max(0.0, self.S + chrs_value - self.kappa)   # Eq. 11
        return self.S > self.h                                 # alert condition

    def _solve_control_limit(self, mu0, sigma, delta, arl0_target) -> float:
        # Eq. 12: ARL0 ≈ (e^{-2Δh} + 2Δh - 1) / (2Δ²), Δ = (μ0-κ)/σ
        # Solve numerically for h given target ARL0 (bisection or Newton's method —
        # closed form doesn't invert cleanly).
        ...
```

- `mu0` and `sigma` (baseline CHRS mean/stdev under "no hallucination") must be estimated from a calibration run on known-clean traces — **do not hardcode**; expose a `calibrate()` routine that computes them from a labeled clean dataset (see §6).
- One `CusumDetector` instance per active pipeline execution (state is per-run, not global) — a process-wide detector pool keyed by `run_id`, garbage-collected on pipeline completion.
- Target `ARL0 = 500` reproduces the paper's <0.2% false-alarm rate per execution — expose this as a config knob, since different deployments may want a stricter/looser tradeoff against detection latency.

### 4.8 Monitor Placement Optimizer — budget-constrained `S*`

Implements the paper's §II.H: since full-fidelity monitoring (attention-based `α`, K-sample semantic entropy) at every step is expensive, decide *which* steps get expensive monitoring under budget `B`. This is NP-hard (Theorem 2); use the greedy submodular approximation:

```python
def greedy_monitor_placement(graph: PipelineGraph, weights: dict[str, float],
                              budget: float, marginal_risk_fn) -> set[str]:
    """(1 - 1/e) ≈ 63% approximation per submodularity (paper §II.H)."""
    S, remaining_budget = set(), budget
    candidates = set(graph.nodes)
    while candidates:
        best, best_ratio = None, -1
        for s in candidates:
            if weights[s] > remaining_budget:
                continue
            ratio = marginal_risk_fn(s, S) / weights[s]   # Δrisk(s)/w(s)
            if ratio > best_ratio:
                best, best_ratio = s, ratio
        if best is None:
            break
        S.add(best)
        remaining_budget -= weights[best]
        candidates.remove(best)
    return S
```

- `weights[s]`: per-step monitoring cost — calibrate empirically per step *kind* (a `tool_call` monitor with counterfactual perturbation costs one extra LLM call; a cheap embedding-only check is near-free). Don't assume uniform cost.
- `marginal_risk_fn(s, S)`: expected reduction in `H_cascade(s_n)` (sink node) from adding monitor `s` to set `S` — computable by simulating with/without `s`'s full-fidelity score on historical traces (offline), refreshed periodically (e.g. nightly) rather than recomputed live.
- This runs **offline/periodically** (e.g. once per pipeline template, or nightly on drifted traffic patterns) — not per-request. Per-request, the system just looks up the precomputed `S*` for that pipeline template and applies cheap-vs-expensive scoring accordingly.

---

## 5. Data model

```sql
-- Trace store schema (Postgres, pgvector extension for embeddings)

CREATE TABLE pipeline_runs (
    run_id UUID PRIMARY KEY,
    pipeline_template_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT CHECK (status IN ('running','completed','failed','alerted'))
);

CREATE TABLE steps (
    step_id TEXT PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(run_id),
    kind TEXT NOT NULL,
    output TEXT,
    output_embedding VECTOR(768),
    retrieved_doc_ids TEXT[],
    tool_call JSONB,
    h_score FLOAT,             -- H(s_i)
    h_cascade FLOAT,           -- H_cascade(s_i)
    chaf FLOAT,                -- CHAF(s_i)
    semantic_entropy FLOAT,    -- SE(s_i)
    chrs FLOAT,                -- CHRS(s_i)
    monitored_fidelity TEXT CHECK (monitored_fidelity IN ('cheap','full')),
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE step_edges (
    run_id UUID REFERENCES pipeline_runs(run_id),
    from_step TEXT REFERENCES steps(step_id),
    to_step TEXT REFERENCES steps(step_id),
    alpha FLOAT,                -- α_ij
    alpha_method TEXT CHECK (alpha_method IN ('attention','nli_proxy','counterfactual')),
    PRIMARY KEY (from_step, to_step)
);

CREATE TABLE cusum_states (
    run_id UUID PRIMARY KEY REFERENCES pipeline_runs(run_id),
    s_n FLOAT NOT NULL,
    kappa FLOAT NOT NULL,
    h_control_limit FLOAT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE alerts (
    alert_id UUID PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(run_id),
    triggered_at_step TEXT REFERENCES steps(step_id),
    s_n_value FLOAT,
    root_cause_step TEXT,       -- localized via highest H_cascade contributor upstream
    created_at TIMESTAMPTZ NOT NULL
);
```

Root-cause localization (a practical necessity the paper doesn't fully spec, but the system needs): when an alert fires at `s_j`, walk backward through predecessors ranked by `α_ij · H_cascade(s_i)` contribution to `H_cascade(s_j)` (each summand in Eq. 5) and surface the top contributor(s) as the likely origin — this is what makes an alert *actionable* instead of just a number.

---

## Quickstart — run it in 3 commands

```bash
pip install -e ".[dev]"
python scripts/calibrate.py     # CUSUM μ0/σ from clean traces → data/calibration.json (once; reloaded every boot)
python demo.py --serve          # seeded clean + cascading runs → http://127.0.0.1:8000/dashboard
```

Daily workflow commands (all runnable without external services; Postgres/Redis optional via `docker compose up -d`):

```bash
python demo.py [--serve] [--report demo_output.md]   # executable demo: clean run + Fig.1 cascade + dataset slice
uvicorn api.main:app                                 # API + dashboard (dist built via: cd dashboard && npm run build)
pytest -q                                            # full test suite (incl. Theorem 1 invariant, ARL0 simulation)
ruff check . && black --check core sdk api storage eval tests && mypy   # quality gates (CI parity)

# operations
python -m eval.run_eval                              # Theorem-1 sweep, ARL0 simulation, Fig.1 amplification
python -m eval.loadtest.load_test --scenario cheap_check --concurrency 10 --requests 100
python jobs/refresh.py --base-url http://127.0.0.1:8000 --loop 86400 --use-history --use-measured-costs

# placement / calibration / blocking endpoints
curl -X POST localhost:8000/v1/placement/refresh -H "Content-Type: application/json" \
     -d '{"template_id": "paper_fig1", "use_history": true}'
curl -X POST localhost:8000/v1/decision/<run_id>/<step_id> -H "Content-Type: application/json" \
     -d '{"output": "...", "retrieved_docs": [{"doc_id": "d1", "content": "..."}], "predecessors": ["s1"]}'
```

In-band blocking (Phase 12): `POST /v1/decision/{run_id}/{step_id}` returns `block: true/false` by projecting one CUSUM step from the run's accumulated S_n without advancing it — set `blocking.mode: "block"` (+ optional `threshold_tau`) in `configs/default.yaml` to enforce.

Evaluation (Phase 14): `python -m eval.run_detection_eval` runs the head-to-head vs. the single-step baseline on the committed HaluEval-RAG sample slice and writes `eval/results/detection_eval.json`; results live in `EVAL_REPORT.md`.

---

## 6. Evaluation plan (reproducing the paper's claims)

The paper claims **34.7% reduction in undetected cascading failures** vs. the best single-step baseline, at **12ms latency overhead per step**, evaluated on **HaluEval-RAG, RAGTruth, and AgentHallu**. To validate the implementation:

1. **Dataset preparation**: obtain HaluEval-RAG, RAGTruth, AgentHallu; convert each to the `PipelineGraph` trace format (§5) with ground-truth hallucination labels per step and per-pipeline outcome labels.
2. **Baselines to implement for comparison**: RAGAS, SelfCheckGPT (both are self-contained enough to reimplement or wrap existing OSS implementations) — RAGTruth/FActScore/ARES can be cited from published numbers if reimplementation cost is too high, but at minimum RAGAS + SelfCheckGPT should run head-to-head locally.
3. **Metrics**:
   - Cascading failure detection rate (recall on pipeline-level ground truth) — CascadeGuard vs. baselines.
   - False alarm rate — empirical `ARL0`, compare to the theoretical target from Eq. 12.
   - Per-step latency overhead (p50/p95/p99), separated by cheap-monitor vs. full-fidelity-monitor steps.
   - CHAF distribution vs. pipeline depth — empirically validate Theorem 1's bound holds (`CHAF ≤ depth+1`) across the dataset; this is a strong correctness check on the `α_ij` estimator.
4. **λ tuning**: grid search or Bayesian optimization over `λ ∈ [0,1]` on a held-out validation split, optimizing detection F1.
5. **Ablations worth running**: Track A (attention) vs Track B (NLI proxy / counterfactual) for `α_ij` — quantify how much detection quality degrades on the black-box track, since most production deployments (closed-model APIs) will be forced onto it.

---

## 7. Tech stack

| Layer | Choice | Why |
|---|---|---|
| SDK / instrumentation | Python, async | Matches LangChain/LlamaIndex ecosystem; async needed for non-blocking trace emission |
| Streaming transport | Redis Streams (start) → Kafka (scale) | Redis Streams is enough for early-stage throughput and much simpler ops; swap to Kafka only when volume demands it |
| Core engine | Python (FastAPI + asyncio workers), or Go for the CUSUM hot path if latency budget gets tight | FastAPI for fast iteration; keep CUSUM update itself trivially cheap regardless of language |
| Embeddings | Self-hosted bge-base/e5-base via vLLM or ONNX runtime | Avoid external API round-trip on the latency-critical faithfulness scoring path |
| Trace store | PostgreSQL + pgvector | One database for structured trace data and embeddings; avoids a separate vector DB early on |
| Cache (cheap-path lookups) | Redis | Sub-ms lookups for precomputed monitor placement `S*`, calibration constants |
| Alerting | Webhook/Slack/PagerDuty connectors | Keep pluggable — don't hardcode one destination |
| Dashboard | Grafana (metrics) + a small custom UI (trace/DAG visualization, root-cause view) | Grafana for time-series ops metrics; DAG visualization needs a custom view (Grafana can't render per-run graphs well) |
| Attention access (Track A) | vLLM / HF Transformers with `output_attentions=True`, self-hosted open-weight models | Required for exact Eq. 6; not available for closed APIs |

---

## 8. Repository structure

```
cascadeguard/
├── sdk/                          # in-process instrumentation
│   ├── step_recorder.py
│   ├── graph_builder.py
│   └── integrations/
│       ├── langchain.py
│       └── llamaindex.py
├── core/
│   ├── graph/
│   │   └── pipeline_graph.py     # §4.1
│   ├── scoring/
│   │   ├── faithfulness.py       # §4.2 — F(s_i), H(s_i)
│   │   ├── propagation.py        # §4.3 — α_ij (attention + surrogate tracks)
│   │   ├── cascade_score.py      # §4.4 — H_cascade, CHAF
│   │   ├── semantic_entropy.py   # §4.5 — SE(s_i)
│   │   └── risk_combiner.py      # §4.6 — CHRS
│   ├── detection/
│   │   └── cusum.py              # §4.7
│   ├── placement/
│   │   └── greedy_optimizer.py   # §4.8
│   └── localization/
│       └── root_cause.py         # §5 root-cause walk
├── storage/
│   ├── schema.sql                # §5
│   └── repository.py
├── api/
│   ├── main.py                   # FastAPI app
│   └── routes/
│       ├── ingest.py             # trace ingestion endpoints
│       ├── alerts.py
│       └── calibration.py
├── eval/
│   ├── datasets/                 # HaluEval-RAG, RAGTruth, AgentHallu loaders
│   ├── baselines/                # RAGAS, SelfCheckGPT wrappers
│   └── run_eval.py               # §6
├── dashboard/                    # trace/DAG visualization UI
├── tests/
│   ├── test_cascade_score.py     # includes CHAF ≤ depth+1 invariant test
│   ├── test_cusum.py
│   └── test_placement.py
├── configs/
│   └── default.yaml              # λ, ARL0 target, budget B, embedding model, etc.
└── README.md
```

---

## 9. Key configuration knobs

```yaml
risk_combiner:
  lambda: 0.6            # tuned via §6 step 4, not guessed

cusum:
  arl0_target: 500        # → <0.2% false alarm rate per Eq. 12
  delta: 0.1               # minimum detectable CHRS shift

monitor_placement:
  budget_B: 10.0            # compute-cost units per pipeline execution
  refresh_schedule: nightly

propagation_estimator:
  track: "counterfactual"   # "attention" | "nli_proxy" | "counterfactual"
  expensive_mode: false      # gates counterfactual re-execution

semantic_entropy:
  K: 10                       # samples; only applied to steps in S*
  cluster_threshold: 0.85

embeddings:
  model: "bge-base-en-v1.5"
  self_hosted: true
```

---

## 10. Known limitations to design around, not paper over

- **Attention-based `α_ij` (Eq. 6) requires white-box model access.** Most production agentic systems sit on top of closed APIs. The system must default gracefully to the black-box track and make that degradation visible in metrics/dashboards — never silently blend the two without labeling which was used (see `alpha_method` column in §5).
- **NP-hardness (Theorem 2)** means the placement optimizer is fundamentally approximate. Budget planning should treat `S*` as "good," not "optimal," and the eval harness should track the empirical gap vs. brute-force on small synthetic DAGs to sanity-check the `(1-1/e)` bound is being realized in practice.
- **Edge inference for dynamic (non-templated) agents** is a heuristic (§4.1), not guaranteed correct — mislabeled edges directly corrupt `H_cascade` propagation. This is the single highest-leverage place to invest testing effort.
- **Semantic entropy's `K` extra LLM calls per monitored step** is real added cost and latency — this is precisely why §4.8 (budget-constrained placement) exists; don't apply SE unconditionally to every step regardless of what the optimizer says, even if it seems "more correct" to do so everywhere.

---

## 11. References

Full reference list matches the paper: Lewis et al. (RAG, NeurIPS 2020), Yao et al. (ReAct, ICLR 2023), Es et al. (RAGAS, EACL 2024), Niu et al. (RAGTruth, ACL 2024), Manakul et al. (SelfCheckGPT, EMNLP 2023), Min et al. (FActScore, EMNLP 2023), Saad-Falcon et al. (ARES, NAACL 2024), Farquhar et al. (Semantic Entropy, Nature 2024), Garey & Johnson (NP-hardness), Nemhauser/Wolsey/Fisher (submodular approximation).
