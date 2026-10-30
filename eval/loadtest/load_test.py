"""Phase 11 load-test harness for the CascadeGuard API (no new dependencies).

Drives a RUNNING server (default http://127.0.0.1:8000) with httpx.AsyncClient.

Scenarios (--scenario):
  cheap_check    Seeds --seed-runs runs via the API (retrieve + synthesize + analyze),
                 then measures repeated POST /v1/cheap_check/{run_id}/{step_id} lookups
                 against the pre-seeded runs. Seeding latency is NOT measured.
  full_pipeline  Per iteration: POST /v1/runs -> --steps-per-run POST steps -> POST
                 analyze. All three endpoints are measured per request.

Reports per-endpoint count/errors/p50/p95/p99/mean/max in milliseconds and exits 1
when the measured error rate exceeds 5%.

Usage:
  python -m eval.loadtest.load_test --base-url http://127.0.0.1:8767 \
      --scenario cheap_check --concurrency 10 --requests 100
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
ERROR_RATE_FAIL_THRESHOLD = 0.05

# Endpoint labels used as stats keys and report rows.
LABEL_CHEAP_CHECK = "POST /v1/cheap_check/{run_id}/{step_id}"
LABEL_RUNS = "POST /v1/runs"
LABEL_STEPS = "POST /v1/runs/{run_id}/steps"
LABEL_ANALYZE = "POST /v1/runs/{run_id}/analyze"

# Deterministic-ish seeding fixture: Eiffel Tower / Paris strings.
_RETRIEVED_DOCS: list[dict[str, str]] = [
    {
        "doc_id": "doc-eiffel",
        "content": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."
        ),
    },
    {
        "doc_id": "doc-paris",
        "content": "Paris is the capital and largest city of France, standing on the Seine river.",
    },
]
_SEED_RETRIEVE_STEP = "s1_retrieve"
_SEED_SYNTH_STEP = "s2_synth"

# Short per-invocation suffix keeps run ids unique across harness restarts.
_RUN_SUFFIX = uuid.uuid4().hex[:8]


@dataclass
class EndpointStats:
    """Latency samples and error count for one measured endpoint."""

    label: str
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0

    def record(self, latency_ms: float, ok: bool) -> None:
        if ok:
            self.latencies_ms.append(latency_ms)
        else:
            self.errors += 1

    @property
    def count(self) -> int:
        return len(self.latencies_ms) + self.errors


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile over an ascending-sorted sample (0.0 when empty)."""
    if not sorted_values:
        return 0.0
    rank = math.ceil(pct / 100.0 * len(sorted_values))
    index = min(len(sorted_values) - 1, max(rank - 1, 0))
    return sorted_values[index]


async def _timed_request(
    client: httpx.AsyncClient,
    stats: dict[str, EndpointStats],
    label: str,
    path: str,
    json_body: dict[str, object] | None = None,
) -> httpx.Response | None:
    """Issue one measured request; returns the response or None on transport failure."""
    started = time.perf_counter()
    try:
        response = await client.post(path, json=json_body)
    except httpx.HTTPError:
        stats[label].record((time.perf_counter() - started) * 1000.0, ok=False)
        return None
    stats[label].record((time.perf_counter() - started) * 1000.0, ok=response.status_code < 400)
    return response


async def _seed_cheap_check_runs(client: httpx.AsyncClient, num_runs: int) -> list[tuple[str, str]]:
    """Create num_runs fully-analyzed runs (retrieve + synthesize) via the API; unmeasured."""
    retrieve_body: dict[str, object] = {
        "step_id": _SEED_RETRIEVE_STEP,
        "kind": "retrieve",
        "output": "Retrieved 2 documents about the Eiffel Tower in Paris.",
        "retrieved_docs": _RETRIEVED_DOCS,
        "predecessors": [],
    }
    synth_body: dict[str, object] = {
        "step_id": _SEED_SYNTH_STEP,
        "kind": "synthesize",
        "output": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "predecessors": [_SEED_RETRIEVE_STEP],
    }
    pairs: list[tuple[str, str]] = []
    for index in range(num_runs):
        run_id = f"lt-seed-{_RUN_SUFFIX}-{index:03d}"
        (await client.post("/v1/runs", json={"run_id": run_id})).raise_for_status()
        (await client.post(f"/v1/runs/{run_id}/steps", json=retrieve_body)).raise_for_status()
        (await client.post(f"/v1/runs/{run_id}/steps", json=synth_body)).raise_for_status()
        (await client.post(f"/v1/runs/{run_id}/analyze")).raise_for_status()
        pairs.append((run_id, _SEED_SYNTH_STEP))
    return pairs


def _pipeline_step_chain(steps_per_run: int) -> list[tuple[str, str]]:
    """Deterministic step chain (id, kind): plan -> retrieve... -> synthesize."""
    chain: list[tuple[str, str]] = []
    for index in range(steps_per_run):
        if steps_per_run == 1:
            kind = "synthesize"
        elif index == 0:
            kind = "plan"
        elif index == steps_per_run - 1:
            kind = "synthesize"
        else:
            kind = "retrieve"
        chain.append((f"s{index:02d}_{kind}", kind))
    return chain


def _pipeline_step_body(
    iteration: int, step_id: str, kind: str, predecessor: str | None
) -> dict[str, object]:
    return {
        "step_id": step_id,
        "kind": kind,
        "output": f"The Eiffel Tower stands in Paris, France (iteration {iteration}, {step_id}).",
        "retrieved_docs": _RETRIEVED_DOCS if kind == "retrieve" else [],
        "predecessors": [predecessor] if predecessor else [],
    }


async def _full_pipeline_iteration(
    client: httpx.AsyncClient,
    stats: dict[str, EndpointStats],
    iteration: int,
    steps_per_run: int,
) -> None:
    """One measured iteration: create run, add steps, analyze. Stops early on failure."""
    run_id = f"lt-fp-{_RUN_SUFFIX}-{iteration:06d}"
    created = await _timed_request(client, stats, LABEL_RUNS, "/v1/runs", {"run_id": run_id})
    if created is None or created.status_code >= 400:
        return
    predecessor: str | None = None
    for step_id, kind in _pipeline_step_chain(steps_per_run):
        body = _pipeline_step_body(iteration, step_id, kind, predecessor)
        response = await _timed_request(
            client, stats, LABEL_STEPS, f"/v1/runs/{run_id}/steps", body
        )
        if response is None or response.status_code >= 400:
            return
        predecessor = step_id
    await _timed_request(client, stats, LABEL_ANALYZE, f"/v1/runs/{run_id}/analyze")


async def _run_pool(
    worker_count: int, total: int, worker: Callable[[int], Awaitable[None]]
) -> None:
    """Run `total` work items across worker_count coroutines pulling from a shared queue."""
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(total):
        queue.put_nowait(index)

    async def runner() -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await worker(index)

    await asyncio.gather(*(runner() for _ in range(min(worker_count, total))))


async def run(args: argparse.Namespace) -> int:
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    stats = {label: EndpointStats(label=label) for label in _scenario_labels(args.scenario)}
    seed_pairs: list[tuple[str, str]] = []

    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=args.timeout, limits=limits
    ) as client:
        if args.scenario == "cheap_check":
            seed_started = time.perf_counter()
            seed_pairs = await _seed_cheap_check_runs(client, args.seed_runs)
            print(
                f"seeded {len(seed_pairs)} runs in {time.perf_counter() - seed_started:.2f}s "
                "(seeding is not measured)"
            )

        print(
            f"scenario={args.scenario} base_url={args.base_url} "
            f"concurrency={args.concurrency} requests={args.requests}"
        )
        started = time.perf_counter()

        async def cheap_check_worker(index: int) -> None:
            run_id, step_id = seed_pairs[index % len(seed_pairs)]
            await _timed_request(
                client, stats, LABEL_CHEAP_CHECK, f"/v1/cheap_check/{run_id}/{step_id}"
            )

        async def full_pipeline_worker(index: int) -> None:
            await _full_pipeline_iteration(client, stats, index, args.steps_per_run)

        worker = cheap_check_worker if args.scenario == "cheap_check" else full_pipeline_worker
        await _run_pool(args.concurrency, args.requests, worker)
        wall_seconds = time.perf_counter() - started

    _print_report(stats, wall_seconds)
    return _exit_code(stats)


def _scenario_labels(scenario: str) -> list[str]:
    if scenario == "cheap_check":
        return [LABEL_CHEAP_CHECK]
    return [LABEL_RUNS, LABEL_STEPS, LABEL_ANALYZE]


def _print_report(stats: dict[str, EndpointStats], wall_seconds: float) -> None:
    print("\nmeasured latency by endpoint (ms)")
    header = (
        f"{'endpoint':<40} {'count':>7} {'errors':>7} "
        f"{'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'max':>8}"
    )
    print(header)
    print("-" * len(header))
    for endpoint in stats.values():
        values = sorted(endpoint.latencies_ms)
        mean_ms = statistics.fmean(values) if values else 0.0
        print(
            f"{endpoint.label:<40.40} {endpoint.count:>7} {endpoint.errors:>7} "
            f"{_percentile(values, 50):>8.2f} {_percentile(values, 95):>8.2f} "
            f"{_percentile(values, 99):>8.2f} {mean_ms:>8.2f} "
            f"{values[-1] if values else 0.0:>8.2f}"
        )
    total = sum(endpoint.count for endpoint in stats.values())
    errors = sum(endpoint.errors for endpoint in stats.values())
    rate = errors / total if total else 0.0
    verdict = "FAIL" if rate > ERROR_RATE_FAIL_THRESHOLD else "PASS"
    print(
        f"\n{total} measured requests in {wall_seconds:.2f}s, "
        f"{errors} errors ({rate:.2%}) -> {verdict} "
        f"(fail threshold {ERROR_RATE_FAIL_THRESHOLD:.0%})"
    )


def _exit_code(stats: dict[str, EndpointStats]) -> int:
    total = sum(endpoint.count for endpoint in stats.values())
    errors = sum(endpoint.errors for endpoint in stats.values())
    if total == 0:
        return 1
    return 1 if errors / total > ERROR_RATE_FAIL_THRESHOLD else 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CascadeGuard Phase 11 API load-test harness.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="running server base URL")
    parser.add_argument(
        "--scenario",
        choices=("cheap_check", "full_pipeline"),
        default="cheap_check",
        help="workload to run (default: cheap_check)",
    )
    parser.add_argument("--concurrency", type=int, default=20, help="parallel workers")
    parser.add_argument("--requests", type=int, default=200, help="iterations to measure")
    parser.add_argument(
        "--steps-per-run", type=int, default=3, help="steps per run (full_pipeline only)"
    )
    parser.add_argument(
        "--seed-runs", type=int, default=5, help="runs to seed via the API (cheap_check only)"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout seconds")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.requests < 1:
        parser.error("--requests must be >= 1")
    if args.steps_per_run < 1:
        parser.error("--steps-per-run must be >= 1")
    if args.seed_runs < 1:
        parser.error("--seed-runs must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(run(args))
    except httpx.HTTPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
