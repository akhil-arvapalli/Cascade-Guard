"""JSON-file persistence for CUSUM calibration (μ0, σ).

The in-memory repository dies with the process, so the default deployment
persists calibration to a small JSON file (config: cusum.calibration_store_path,
default ``data/calibration.json``) that ``api.main.build_engine`` reloads at
boot. V2-3 will move this into the Postgres repository alongside the trace
store; the read/write surface here is deliberately tiny so that swap is local.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_PATH = "data/calibration.json"


class CalibrationStore:
    """Load/save μ0/σ as JSON; missing or corrupt files read as 'not calibrated'."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)

    def save(self, mu0: float, sigma: float, extra: dict[str, object] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"mu0": mu0, "sigma": sigma, "updated_at": time.time()}
        if extra:
            payload.update(extra)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> dict[str, float] | None:
        """Returns {'mu0', 'sigma'} or None when absent/corrupt/invalid.

        Corrupt data is treated as un-calibrated (engine falls back to config
        placeholders) rather than crashing the boot path — a calibration file
        is reproducible via scripts/calibrate.py, not irreplaceable state.
        """
        if not self.path.is_file():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        mu0, sigma = data.get("mu0"), data.get("sigma")
        if not isinstance(mu0, (int, float)) or not isinstance(sigma, (int, float)):
            return None
        if sigma <= 0:
            return None
        return {"mu0": float(mu0), "sigma": float(sigma)}
