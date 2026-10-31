"""CascadeGuard — one-command executable demo.

    python demo.py            # run the demo, print the summary
    python demo.py --serve    # ... then serve the dashboard with the seeded runs
    python demo.py --report demo_output.md

What it does:
  1. calibrates CUSUM on clean traces if data/calibration.json is missing
  2. pushes a CLEAN rag run and a real CASCADING-hallucination run (the paper's
     Fig.1 trace) through the full API path (ingest -> analyze -> alerts)
  3. if the Phase 14 sample dataset is present, scores a few labeled
     HaluEval-RAG traces against their ground truth
  4. prints a summary table and, with --serve, serves the dashboard at
     http://127.0.0.1:8000/dashboard with the seeded runs (Ctrl+C to stop)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LINE = "─" * 78


def _hr(title: str) -> None:
    print(f"\n{LINE}\n  {title}\n{LINE}")


def _ensure_calibrated(app: Any) -> None:
    from storage.calibration_store import DEFAULT_PATH, CalibrationStore

    if CalibrationStore(DEFAULT_PATH).load() is not None:
        return
    print("no calibration found — running clean-trace CUSUM calibration first...")
    from core.calibration import calibrate_from_clean_traces

    mu0, sigma, n = calibrate_from_clean_traces(app.state.engine)
    CalibrationStore(DEFAULT_PATH).save(mu0, sigma, extra={"n_samples": n})
    print(f"  calibrated μ0={mu0:.4f} σ={sigma:.4f} from {n} samples → {DEFAULT_PATH}")


def _push_trace(client: Any, template_id: str, graph: Any) -> dict[str, Any]:
    """Ingest a PipelineGraph through the full API path and analyze it."""
    run = client.post("/v1/runs", json={"pipeline_template_id": template_id}).json()
    run_id = run["run_id"]
    for nid in graph.topo_order():
        step = graph.nodes[nid]
        client.post(
            f"/v1/runs/{run_id}/steps",
            json={
                "step_id": nid,
                "kind": step.kind,
                "output": step.output,
                "retrieved_docs": [
                    {"doc_id": d.doc_id, "content": d.content} for d in step.retrieved_docs
                ],
                "predecessors": list(step.predecessors),
            },
        )
    return client.post(f"/v1/runs/{run_id}/analyze").json()


def _print_run_summary(title: str, detail: dict[str, Any]) -> None:
    s = detail["scores_summary"]
    print(f"\n  {title}")
    print(
        f"  steps={s['total_steps']}  max_CHRS={_fmt(s['max_chrs'])}  "
        f"max_CHAF={_fmt(s['max_chaf'])}  analyze={_fmt(detail['analyze_ms'])}ms"
    )
    for node in detail["nodes"]:
        chrs = _fmt(node["chrs"], 4)
        marker = (
            "  ← ALERTED"
            if node["id"]
            == (detail["alerts"][0]["triggered_at_step"] if detail["alerts"] else None)
            else ""
        )
        print(
            f"    {node['id']:<12} {node['kind']:<11} H={_fmt(node['H'])}  "
            f"H_c={_fmt(node['h_cascade'])}  CHRS={chrs}  [{node['fidelity']}]{marker}"
        )
    if detail["alerts"]:
        a = detail["alerts"][0]
        rcs = a.get("root_causes") or []
        rc = (
            ", ".join(f"{rc['step_id']} ({rc['share'] * 100:.0f}%)" for rc in rcs[:2])
            or a.get("root_cause_step")
            or "unknown"
        )
        print(
            f"  ⚠ ALERT: S_n={a['s_n_value']:.4f} at '{a['triggered_at_step']}' — root cause: {rc}"
        )
    else:
        print("  ✓ no alert (detector stayed under the calibrated control limit)")


def _fmt(v: Any, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def _sample_dataset_section(client: Any, max_traces: int = 6) -> None:
    try:
        from eval.datasets.loaders import load_halu_eval_rag_sample
    except ImportError:
        print("  (dataset loader not available — skipping)")
        return
    data_path = _REPO_ROOT / "eval" / "datasets" / "data" / "halu_eval_rag_sample.json"
    if not data_path.is_file():
        print(f"  (sample dataset not present at {data_path} — skipping)")
        return

    traces = load_halu_eval_rag_sample(data_path)
    engine = client.app.state.engine
    kappa = engine.cusum_params.mu0 + engine.cusum_params.delta / 2
    print(f"  alert rule: max CHRS > κ = {kappa:.4f} (calibrated)\n")

    rows: list[tuple[str, str, float, str]] = []
    picked = traces[:max_traces]
    for i, labeled in enumerate(picked):
        labels = labeled.labels
        truth = next(iter(labels.values())) if labels else False
        result = engine.analyze_run(labeled.graph)
        max_chrs = max(result.chrs.values()) if result.chrs else 0.0
        predicted = max_chrs > kappa
        outcome = "✓" if predicted == truth else "✗ MISS"
        rows.append(
            (
                f"sample-{i}",
                "hallucinated" if truth else "faithful",
                max_chrs,
                outcome,
            )
        )

    print("  trace        ground-truth    max CHRS   verdict")
    for name, truth, chrs, outcome in rows:
        print(f"  {name:<12} {truth:<15} {chrs:.4f}     {outcome}")
    hits = sum(1 for r in rows if r[3].startswith("✓"))
    print(
        f"\n  {hits}/{len(rows)} verdicts match ground truth (small demo slice — "
        "see EVAL_REPORT.md for the full run)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CascadeGuard executable demo.")
    parser.add_argument("--serve", action="store_true", help="serve the dashboard after seeding")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--report", type=Path, default=None, help="write a markdown report")
    args = parser.parse_args()

    from fastapi.testclient import TestClient

    from api.main import create_app, load_config
    from core.graph.pipeline_graph import Document, PipelineGraph, Step
    from eval.synthetic import make_paper_fig1_trace

    app = create_app(config=load_config())
    client = TestClient(app)
    _ensure_calibrated(app)

    print(f"\n{LINE}\n  CascadeGuard demo — pipeline-aware hallucination detection\n{LINE}")

    # -- scenario 1: a clean, faithful run --------------------------------
    _hr("Scenario 1 — faithful run (should NOT alert)")
    clean = PipelineGraph(pipeline_template_id="demo_rag_clean")
    doc = (
        "Photosynthesis is the process by which plants convert sunlight, water and CO2 "
        "into chemical energy."
    )
    clean.add_step(
        Step(
            id="s1_retrieve",
            kind="retrieve",
            output="Photosynthesis converts sunlight into chemical energy in plants.",
            retrieved_docs=[Document(doc_id="d1", content=doc)],
        )
    )
    clean.add_step(
        Step(
            id="s2_reason",
            kind="reason",
            output="Plants convert sunlight into chemical energy via photosynthesis.",
            predecessors=["s1_retrieve"],
        )
    )
    clean.add_step(
        Step(
            id="s3_synthesize",
            kind="synthesize",
            output="Plants use photosynthesis to turn sunlight into chemical energy.",
            predecessors=["s2_reason"],
        )
    )
    r1 = _push_trace(client, "demo_rag_clean", clean)
    d1 = client.get(f"/v1/runs/{r1['run_id']}").json()
    _print_run_summary("CLEAN RUN", d1)

    # -- scenario 2: the paper's Fig.1 cascading hallucination ------------
    _hr("Scenario 2 — cascading hallucination (paper Fig.1, corrupted step 3)")
    corrupted = make_paper_fig1_trace(corrupted=True)
    r2 = _push_trace(client, "paper_fig1", corrupted)
    d2 = client.get(f"/v1/runs/{r2['run_id']}").json()
    _print_run_summary("CASCADING RUN", d2)

    # -- scenario 3: labeled sample dataset (when present) ----------------
    _hr("Scenario 3 — HaluEval-RAG sample traces vs ground truth")
    _sample_dataset_section(client)

    runs = client.get("/v1/runs").json()
    _hr(f"Dashboard ready — {runs['count']} seeded runs")
    if args.report:
        _write_report(args.report, d1, d2)
        print(f"report written → {args.report}")
    if args.serve:
        import uvicorn

        print(f"serving on http://127.0.0.1:{args.port}/dashboard  (Ctrl+C to stop)")
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    else:
        print("re-run with --serve to browse the seeded runs in the dashboard.")
    return 0


def _write_report(path: Path, d1: dict[str, Any], d2: dict[str, Any]) -> None:
    def table(title: str, d: dict[str, Any]) -> str:
        s = d["scores_summary"]
        lines = [
            f"## {title}",
            "",
            f"- steps: {s['total_steps']}, max CHRS: {s['max_chrs']}, max CHAF: {s['max_chaf']}",
            f"- alerted: {d['alerted']}, analyze: {d['analyze_ms']} ms",
            "",
        ]
        lines.append("| step | kind | H | H_cascade | CHRS | fidelity |")
        lines.append("|---|---|---|---|---|---|")
        for n in d["nodes"]:
            row = (
                f"| {n['id']} | {n['kind']} | {n['H']} | {n['h_cascade']} | {n['chrs']} |"
                f" {n['fidelity']} |"
            )
            lines.append(row)
        if d["alerts"]:
            a = d["alerts"][0]
            rcs = a.get("root_causes") or []
            rc = ", ".join(
                f"{rc['step_id']} ({rc['share'] * 100:.0f}%)" for rc in rcs[:3]
            ) or a.get("root_cause_step")
            lines += [
                "",
                f"**ALERT** at `{a['triggered_at_step']}` (S_n={a['s_n_value']:.4f})"
                f" — root cause: {rc}",
            ]
        return "\n".join(lines) + "\n"

    path.write_text(
        "# CascadeGuard demo output\n\n"
        f"_generated {time.strftime('%Y-%m-%d %H:%M')}_\n\n"
        + table("Scenario 1 — faithful run", d1)
        + "\n"
        + table("Scenario 2 — cascading hallucination (paper Fig.1)", d2)
        + "\nRun `python demo.py --serve` to reproduce in the dashboard.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
