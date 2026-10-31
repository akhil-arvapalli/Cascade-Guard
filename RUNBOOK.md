# CascadeGuard On-Call Runbook

Operational runbook for the CascadeGuard Core Engine service (`uvicorn api.main:app`).
Config knobs live in `configs/default.yaml` — never hardcode them elsewhere.

---

## 1. When an alert fires

An alert means the CUSUM detector's running statistic `S_n` crossed the
control limit `h` for a pipeline run — CHRS (hallucination risk) has shifted
up persistently, and the root-cause localization has ranked the steps that
contributed most.

Triage path:

1. **Dashboard** (`/dashboard` in the browser, or `GET /v1/runs`):
   find the run — `alerted: true`, status `alerted`. The run detail view shows
   the DAG, per-step scores, and the alert's root-cause path highlighted.
2. **Root cause** (`GET /v1/runs/{run_id}` → `alerts[].root_causes`):
   contribution-ranked steps (share + path length). The top share is where the
   corrupted content most likely entered. Check that step's `retrieved_docs`
   first — retrieval poisoning is the common case; a high `chaf` on a
   deep node suggests propagation of an upstream error.
3. **CUSUM `S_n`** (`alerts[].s_n_value`): how far past the control limit the
   run's statistic climbed. One marginal exceedance is worth a look; repeated
   alerts across runs means the baseline drifted — recalibrate (Section 2).
4. **Cheap check** (`POST /v1/cheap_check/{run_id}/{step_id}`): sub-12ms cached
   lookup of a step's last CHRS + monitor fidelity. Use it to confirm a
   specific step's risk without re-running analysis.

If `blocking.mode: block` (default `notify_only`), the same verdict also gates
`POST /v1/decision/{run_id}/{step_id}` — see the decision endpoint docs.

## 2. Recalibrating CUSUM (μ0/σ)

Symptoms: false alarms on clean traffic, or alerts never firing.

```bash
python scripts/calibrate.py --config configs/default.yaml --dags 10
```

This harvests CHRS from clean synthetic traces, writes `data/calibration.json`
(path from `cusum.calibration_store_path`), and is reloaded at every boot
(`api.main.build_engine`). Also persisted to the trace repository
(`POST /v1/calibration/cusum` does the same in-band). Keep `cusum.delta`
within ~σ/2 of the calibrated σ — see the regime note in
`configs/default.yaml` and `tests/test_cusum.py`.

## 3. Refreshing monitor placement (S\*)

Nightly (or after embedding-backend changes):

```bash
python jobs/refresh.py --base-url http://127.0.0.1:8000 --use-history
```

Recomputes the budget-constrained monitor set S\* per discovered pipeline
template so semantic entropy only runs where it pays (paper §II.H).

## 4. Retention (purge old runs)

```bash
python jobs/retention.py --days 30
```

Deletes pipeline runs whose `completed_at` is older than N days (plus their
steps, edges, CUSUM state, and alerts, in FK-safe order, in one transaction).
Refuses to run without an explicit `--days`. DSN from `--dsn` or
`$CASCADEGUARD_PG_DSN`. Runs still in `running` status are never purged.

## 5. Failure notes

- **In-memory store resets on restart.** Default `storage.backend: memory`
  keeps runs/graphs/alerts in process memory only — restart wipes traces and
  run graphs. Calibration survives via `data/calibration.json`.
- **Postgres activation:**
  1. `docker compose up -d postgres` (pgvector/pgvector:pg16; fresh volumes
     get `storage/schema.sql` applied automatically via initdb).
  2. Set `storage.backend: postgres` in config.
  3. `export CASCADEGUARD_PG_DSN=postgresql://cascadeguard:cascadeguard@localhost:5432/cascadeguard`
     (name configurable via `storage.dsn_env_var`). The API fails loudly at
     boot if the variable is unset. Consider appending `?connect_timeout=5` —
     libpq has no default connect timeout, so an unreachable host would hang
     requests instead of erroring.
  4. Pre-existing volume/managed instance only:
     `python storage/apply_schema.py` (idempotent; fresh volumes skip this).
  5. `pip install cascadeguard[postgres]` if psycopg is missing (boot raises
     a RuntimeError naming this extra).
- **Postgres runs stay `running` in the DB.** Run-completion wiring
  (`repo.complete_run(...)` / detector `pool.complete`) is not yet called on
  the analyze path — the API's `run_graphs`/`results` state tracks liveness in
  process, but persisted rows keep `status='running'` and `completed_at=NULL`.
  Retention therefore never purges them; wire completion in the ingest path
  (TODO Phase 15 follow-up) before relying on DB-side status.
- **Auth misconfiguration fails loudly**: `auth.enabled: true` with an empty
  `auth.api_key` refuses to boot.
- **Load test**: `eval/loadtest/load_test.py` (p50 budget 12ms/step).

## 6. Auth (API key)

`config auth: {enabled, api_key, header}` (all default off/empty/`X-API-Key`).
When enabled, every path except `/docs`, `/openapi.json`, `/health` requires
the configured header with the exact key (constant-time compare); anything
else gets `401 {"detail": "unauthorized"}`. Recommended deployment:

```yaml
auth:
  enabled: true
  api_key: "${CASCADEGUARD_API_KEY}"   # substitute at deploy time — do not commit secrets
  header: "X-API-Key"
```

Keep the key out of the repo: render it from a secret manager / env at deploy
time (a config preprocessor or `CASCADEGUARD_CONFIG` pointing at a
non-committed file both work).

## 7. Logging

`cascadeguard` logger, JSON lines on stdout by default (`logging.json: true`,
`logging.level: INFO`): timestamp, level, logger, message + extras. Every
request logs `method`/`path`/`status`/`ms` at INFO. The root logger is never
reconfigured, so uvicorn's own logging is untouched.

## Run lifecycle

See AGENTS.md 2026-10-31 entry: runs marked completed/alerted on analyze; worker path GCs detectors; async ingestion via POST /v1/ingest + python -m api.worker.
