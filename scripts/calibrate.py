"""One-command CUSUM calibration: clean traces → μ0/σ → data/calibration.json.

Usage: python scripts/calibrate.py [--config configs/default.yaml] [--dags 10]

Builds the SAME engine as the API (api.main.build_engine) so the harvested
CHRS distribution matches deployment, calibrates, persists via
CalibrationStore, and prints a summary. Run once before first serve;
build_engine reloads the persisted values on every boot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.main import build_engine, load_config  # noqa: E402
from core.calibration import calibrate_from_clean_traces  # noqa: E402
from core.detection.cusum import CusumParams  # noqa: E402
from storage.calibration_store import DEFAULT_PATH, CalibrationStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate CUSUM μ0/σ from clean traces.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dags", type=int, default=10, help="clean random DAGs to harvest")
    parser.add_argument(
        "--store-path",
        default=None,
        help=f"override calibration JSON path (default: {DEFAULT_PATH})",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    engine = build_engine(config)
    mu0, sigma, n = calibrate_from_clean_traces(engine, n_dags=args.dags)

    params: CusumParams = engine.cusum_params
    store = CalibrationStore(
        args.store_path or config.get("cusum", {}).get("calibration_store_path", DEFAULT_PATH)
    )
    store.save(
        mu0,
        sigma,
        extra={"n_samples": n, "delta": params.delta, "arl0_target": params.arl0_target},
    )

    abs_delta = params.delta / (2.0 * sigma)
    print(f"calibrated μ0={mu0:.6f} σ={sigma:.6f} from {n} clean CHRS samples")
    print(f"κ={params.mu0 + params.delta / 2:.6f}  |Δ|=δ/(2σ)={abs_delta:.3f}")
    if abs_delta > 0.5:
        print("WARNING: |Δ| > 0.5 — Eq. 12 regime degraded; consider lowering cusum.delta.")
    print(f"persisted → {store.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
