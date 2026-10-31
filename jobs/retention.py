"""Trace-store retention job (TODO Phase 15): purge runs older than N days.

Usage:
    python jobs/retention.py --days 30
    python jobs/retention.py --days 30 --dsn-env CASCADEGUARD_PG_DSN
    python jobs/retention.py --days 7 --dsn postgresql://...   # explicit DSN

Pure psycopg (no HTTP): deletes, in FK-safe order, the alerts, CUSUM state,
step edges, steps and finally the pipeline runs whose completed_at is older
than the cutoff. Runs still in 'running' status (completed_at NULL) are never
touched. The whole purge is one transaction. --days is REQUIRED — the job
refuses to run without an explicit age cutoff.

DSN comes from --dsn or the --dsn-env variable (default $CASCADEGUARD_PG_DSN).
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import psycopg

DSN_ENV_VAR = "CASCADEGUARD_PG_DSN"

# Children first, then the run row itself — matches schema.sql FK order.
_CHILD_TABLES = ("alerts", "cusum_states", "step_edges", "steps")


def purge_older_than(days: int, dsn: str) -> dict[str, int]:
    """Delete completed runs older than `days`; returns per-table delete counts."""
    conn: Any = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT run_id FROM pipeline_runs "
                "WHERE completed_at IS NOT NULL "
                "AND completed_at < now() - make_interval(days => %s)",
                (days,),
            )
            old_ids = [row[0] for row in cur.fetchall()]
            counts: dict[str, int] = {table: 0 for table in _CHILD_TABLES}
            counts["runs"] = 0
            if not old_ids:
                conn.commit()
                return counts
            for table in _CHILD_TABLES:
                cur.execute(f"DELETE FROM {table} WHERE run_id = ANY(%s)", (old_ids,))
                counts[table] = cur.rowcount
            cur.execute("DELETE FROM pipeline_runs WHERE run_id = ANY(%s)", (old_ids,))
            counts["runs"] = cur.rowcount
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delete trace-store runs older than N days.")
    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="age cutoff in days (required; runs with completed_at older than this are purged)",
    )
    parser.add_argument("--dsn", default=None, help="Postgres DSN (default: $<dsn-env>)")
    parser.add_argument(
        "--dsn-env",
        default=DSN_ENV_VAR,
        help="env var holding the DSN (default CASCADEGUARD_PG_DSN)",
    )
    args = parser.parse_args(argv)

    if args.days < 0:
        parser.error("--days must be >= 0")
    dsn = args.dsn or os.environ.get(args.dsn_env, "")
    if not dsn:
        parser.error(f"no DSN supplied: pass --dsn or set ${args.dsn_env}")

    counts = purge_older_than(args.days, dsn)
    total = sum(counts.values())
    print(f"retention: purged {total} row(s) older than {args.days} day(s): {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
