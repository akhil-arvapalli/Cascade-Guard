"""Placement cache (S* lookup) — TODO Phase 8.

The live request path must READ a precomputed S* (sub-millisecond) instead of
recomputing the greedy placement per request. InProcessPlacementCache is the
default and is dict-compatible so code (and tests) that seed
``engine._placement_cache[template] = {...}`` keep working. RedisPlacementCache
shares S* across processes/workers; every Redis failure degrades to the
in-process fallback rather than failing the analysis path (fail-open).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class PlacementCache(Protocol):
    def get(self, template_id: str) -> set[str] | None: ...
    def put(self, template_id: str, s_star: set[str]) -> None: ...


class InProcessPlacementCache:
    """Process-local dict cache; get() returns copies so callers can't mutate."""

    def __init__(self) -> None:
        self._data: dict[str, set[str]] = {}

    def get(self, template_id: str) -> set[str] | None:
        s = self._data.get(template_id)
        return set(s) if s is not None else None

    def put(self, template_id: str, s_star: set[str]) -> None:
        self._data[template_id] = set(s_star)

    # dict-style compat: engine/tests seed and read via [] and `in`
    def __getitem__(self, template_id: str) -> set[str]:
        return self._data[template_id]

    def __setitem__(self, template_id: str, s_star: set[str]) -> None:
        self.put(template_id, s_star)

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._data


class RedisPlacementCache:
    """S* shared via Redis under ``cascadeguard:placement:<template_id>``.

    Fails open: on any Redis error the bundled in-process cache answers instead,
    and a warning is logged — placement lookups never depend on Redis being up.
    """

    _PREFIX = "cascadeguard:placement:"

    def __init__(self, url: str | None = None, client: Any = None) -> None:
        if client is not None:  # test seam
            self._client = client
        else:
            import redis  # loud ImportError → pip install cascadeguard[redis]

            self._client = redis.Redis.from_url(
                url or os.environ.get("CASCADEGUARD_REDIS_URL", "redis://localhost:6379/0"),
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
        self._fallback = InProcessPlacementCache()

    def get(self, template_id: str) -> set[str] | None:
        try:
            raw = self._client.get(self._PREFIX + template_id)
            if raw:
                return set(json.loads(raw))
        except Exception:
            logger.warning("placement cache: redis get failed; in-process fallback", exc_info=True)
        return self._fallback.get(template_id)

    def put(self, template_id: str, s_star: set[str]) -> None:
        self._fallback.put(template_id, s_star)  # always keep a local copy
        try:
            self._client.set(self._PREFIX + template_id, json.dumps(sorted(s_star)))
        except Exception:
            logger.warning("placement cache: redis put failed; kept in-process copy", exc_info=True)

    def __contains__(self, template_id: str) -> bool:
        return self.get(template_id) is not None

    def __getitem__(self, template_id: str) -> set[str]:
        s = self.get(template_id)
        if s is None:
            raise KeyError(template_id)
        return s

    def __setitem__(self, template_id: str, s_star: set[str]) -> None:
        self.put(template_id, s_star)


def build_placement_cache(backend: str, redis_url: str | None = None) -> PlacementCache:
    if backend == "in_process":
        return InProcessPlacementCache()
    if backend == "redis":
        return RedisPlacementCache(url=redis_url)
    raise ValueError(f"unknown placement cache backend {backend!r} (in_process | redis)")
