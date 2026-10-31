"""Apply storage/schema.sql to an existing Postgres database.

Fresh docker-compose volumes do NOT need this script — the schema is applied
automatically via the docker-entrypoint-initdb.d mount (see docker-compose.yml).
Use this against a pre-existing volume or a managed instance:

    python storage/apply_schema.py [--dsn postgresql://...] [--schema storage/schema.sql]

DSN defaults to $CASCADEGUARD_PG_DSN. Statements are split naively on ';'
— fine for this schema, which contains no semicolons inside string literals
or function bodies. The script is idempotent (schema.sql uses IF NOT EXISTS).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import psycopg

DSN_ENV_VAR = "CASCADEGUARD_PG_DSN"
DEFAULT_SCHEMA = Path(__file__).with_name("schema.sql")


def apply_schema(dsn: str, schema_path: Path = DEFAULT_SCHEMA) -> int:
    """Execute every statement in schema.sql; returns the statement count."""
    statements = [s.strip() for s in schema_path.read_text(encoding="utf-8").split(";")]
    applied = 0
    conn: Any = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            for statement in statements:
                if not statement:
                    continue
                cur.execute(statement)
                applied += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply storage/schema.sql to Postgres.")
    parser.add_argument("--dsn", default=None, help="Postgres DSN (default: $CASCADEGUARD_PG_DSN)")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="path to schema.sql")
    args = parser.parse_args(argv)

    dsn = args.dsn or os.environ.get(DSN_ENV_VAR, "")
    if not dsn:
        parser.error(f"no DSN supplied: pass --dsn or set ${DSN_ENV_VAR}")
    applied = apply_schema(dsn, Path(args.schema))
    print(f"applied {applied} statement(s) from {args.schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
