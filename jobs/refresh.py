"""Offline/nightly placement refresh job (TODO Phase 8).

Usage:
    python jobs/refresh.py --base-url http://127.0.0.1:8000
    python jobs/refresh.py --base-url http://127.0.0.1:8000 --loop 86400 \
        --use-history --use-measured-costs

Discovers pipeline templates from the running deployment
(GET /v1/placement/templates), then refreshes each template's S*
(POST /v1/placement/refresh) so the live request path keeps reading a
precomputed monitor set from the placement cache. HTTP-based on purpose: the
job works against any deployment regardless of storage backend, and history
for the trace-simulated risk model is assembled server-side from analyzed
runs. --loop N reruns every N seconds (a cron container or APScheduler work
equally well; D-004 transport and scheduling revisit later).
"""

from __future__ import annotations

import argparse
import time

import httpx


def refresh_once(
    base_url: str,
    use_history: bool,
    use_measured_costs: bool,
    timeout_s: float = 30.0,
) -> int:
    with httpx.Client(timeout=timeout_s) as client:
        templates: list[str] = client.get(f"{base_url}/v1/placement/templates").json()["templates"]
        if not templates:
            print("no templates discovered; nothing to refresh")
            return 0
        failures = 0
        for template_id in templates:
            res = client.post(
                f"{base_url}/v1/placement/refresh",
                json={
                    "template_id": template_id,
                    "use_history": use_history,
                    "use_measured_costs": use_measured_costs,
                },
            )
            if res.status_code != 200:
                failures += 1
                print(f"FAIL {template_id}: HTTP {res.status_code} {res.text[:200]}")
                continue
            body = res.json()
            print(
                f"refreshed {template_id}: S*={body['S_star']} "
                f"(history={body['history_traces']} traces, "
                f"measured_costs={body['measured_costs']})"
            )
        return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly monitor-placement refresh.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--loop", type=int, default=0, help="rerun every N seconds (0 = once)")
    parser.add_argument("--use-history", action="store_true", help="trace-simulated risk model")
    parser.add_argument("--use-measured-costs", action="store_true", help="live-measured costs")
    args = parser.parse_args()

    if args.loop <= 0:
        return refresh_once(args.base_url, args.use_history, args.use_measured_costs)

    while True:
        try:
            refresh_once(args.base_url, args.use_history, args.use_measured_costs)
        except httpx.HTTPError as exc:
            print(f"refresh failed ({exc}); retrying next cycle")
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
