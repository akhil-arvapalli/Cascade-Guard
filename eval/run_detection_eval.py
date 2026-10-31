"""Detection eval on a real dataset slice (TODO Phase 14) - HaluEval-RAG sample.

What this runs:
  1. Loads the committed 150-record HaluEval-RAG sample; deterministic
     stratified 70/30 split into tuning / held-out.
  2. Lambda grid sweep (0.0..1.0 step 0.1) for CascadeGuard on the tuning
     split with a FIXED alert rule: a trace alerts when max CHRS over its
     steps > kappa = mu0 + delta/2 (mirrors CusumDetector's kappa, Eq. 11).
     mu0/sigma/delta come from data/calibration.json when present, else
     mu0=0.05 / sigma=0.02 / delta=0.03 defaults. CHRS varies with lambda
     (CHRS = lambda*H_cascade because SE=0), so a fixed kappa acts as a
     per-lambda shifted threshold (effective cutoff kappa/lambda in
     H_cascade units) - documented, not hidden.
  3. Head-to-head on held-out at the best-F1 lambda: CascadeGuard (max-CHRS
     alert, trace label = hallucinated answer) vs single_step_h_baseline
     (alert if ANY step flagged; threshold tuned on the tuning split for
     best F1). Metrics: detection rate (recall), false-alarm rate on
     right-answer traces, precision, F1, mean per-trace analyze latency.
  4. Per-step H-vs-label breakdown (mean H on hallucinated vs right
     synthesize steps - shows whether the raw signal exists) and the
     CHAF <= depth+1 (Theorem 1) check across all traces.

HONESTY FRAME (AGENTS.md): this run uses the HashingEmbedder (char-trigram
lexical placeholder, D-005), a cosine-similarity entail surrogate replicated
from api/main.build_engine (not a real NLI model, D-002 open), and SE=0 (no
generate_fn wired). Absolute numbers are a small-sample harness result and a
pipeline-aware-vs-single-step comparison - NOT the paper's full claim.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from core.embeddings import EmbeddingFn, HashingEmbedder
from core.engine import AnalysisResult, CascadeGuardEngine
from core.scoring.faithfulness import hallucination_score
from eval.baselines.single_step import single_step_h_baseline
from eval.datasets.loaders import LabeledTrace, load_halu_eval_rag_sample

DEFAULT_SAMPLE = Path("eval/datasets/data/halu_eval_rag_sample.json")
DEFAULT_CALIBRATION = Path("data/calibration.json")
DEFAULT_OUTPUT = Path("eval/results/detection_eval.json")

# CUSUM defaults when no persisted calibration exists (mirrors engine fallbacks;
# delta=0.03 keeps |Delta| in the empirically validated Eq. 12 regime).
DEFAULT_MU0, DEFAULT_SIGMA, DEFAULT_DELTA = 0.05, 0.02, 0.03
LAMBDA_GRID = [round(i / 10, 1) for i in range(11)]
BASELINE_THRESHOLD_GRID = [round(i * 0.05, 2) for i in range(21)]  # 0.00..1.00 step 0.05


def make_embedder(dim: int) -> HashingEmbedder:
    return HashingEmbedder(dim)


def make_entail_fn(embedder: EmbeddingFn):
    """Cosine-similarity entail surrogate, replicated locally from
    api/main.build_engine (importing api would drag FastAPI into the eval
    path). This is NOT a real NLI model - DECISIONS D-002 stays open."""

    def entail_fn(source: str, target: str) -> float:
        a = embedder(source)
        b = embedder(target)
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0

    return entail_fn


def load_cusum_params(path: Path) -> dict:
    """kappa inputs: persisted calibration if present, else loud defaults."""
    if path.exists():
        cal = json.loads(path.read_text(encoding="utf-8"))
        params = {
            "mu0": float(cal.get("mu0", DEFAULT_MU0)),
            "sigma": float(cal.get("sigma", DEFAULT_SIGMA)),
            "delta": float(cal.get("delta", DEFAULT_DELTA)),
        }
        params["source"] = f"{path} (calibrated on synthetic-clean CHRS)"
    else:
        params = {"mu0": DEFAULT_MU0, "sigma": DEFAULT_SIGMA, "delta": DEFAULT_DELTA}
        params["source"] = f"defaults - no calibration file at {path}"
    params["kappa"] = params["mu0"] + params["delta"] / 2  # CusumDetector kappa (Eq. 11)
    return params


def make_engine(lam: float, embedder: EmbeddingFn, entail_fn, cusum: dict) -> CascadeGuardEngine:
    return CascadeGuardEngine(
        config={
            "risk_combiner": {"lambda": lam},
            "propagation_estimator": {"track": "nli_proxy", "clamp_max": 1.0},
            "cusum": {
                "mu0": cusum["mu0"],
                "sigma": cusum["sigma"],
                "delta": cusum["delta"],
                "arl0_target": 500.0,
            },
        },
        embedder=embedder,
        entail_fn=entail_fn,
    )


def stratified_split(
    traces: list[LabeledTrace], train_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic 70/30 split, stratified on the s3 label for class balance."""
    rng = random.Random(seed)
    pos = [i for i, t in enumerate(traces) if t.labels["s3"]]
    neg = [i for i, t in enumerate(traces) if not t.labels["s3"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = int(len(pos) * train_frac + 0.5)
    n_neg = int(len(neg) * train_frac + 0.5)
    return sorted(pos[:n_pos] + neg[:n_neg]), sorted(pos[n_pos:] + neg[n_neg:])


def binary_metrics(labels: list[bool], preds: list[bool]) -> dict:
    tp = sum(1 for y, p in zip(labels, preds, strict=True) if y and p)
    fp = sum(1 for y, p in zip(labels, preds, strict=True) if not y and p)
    tn = sum(1 for y, p in zip(labels, preds, strict=True) if not y and not p)
    fn = sum(1 for y, p in zip(labels, preds, strict=True) if y and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0  # false-alarm rate on right-answer traces
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarm_rate": round(far, 4),
    }


def run_cascade(
    traces: list[LabeledTrace],
    idxs: list[int],
    lam: float,
    embedder: EmbeddingFn,
    entail_fn,
    cusum: dict,
    kappa: float,
) -> tuple[list[bool], list[bool], list[float], list[AnalysisResult]]:
    """Analyze each trace with CascadeGuard at `lam`; alert = max CHRS > kappa.

    Latency is wall-clock around analyze_run (includes placement + CUSUM pool
    setup on each trace's first analysis - honest end-to-end engine cost).
    """
    engine = make_engine(lam, embedder, entail_fn, cusum)
    labels: list[bool] = []
    preds: list[bool] = []
    latencies: list[float] = []
    results: list[AnalysisResult] = []
    for i in idxs:
        graph = traces[i].graph
        start = time.perf_counter()
        result = engine.analyze_run(graph)
        latencies.append((time.perf_counter() - start) * 1000.0)
        labels.append(traces[i].labels["s3"])
        preds.append(max(result.chrs.values()) > kappa)
        results.append(result)
    return labels, preds, latencies, results


def sweep_lambda(
    traces: list[LabeledTrace],
    tuning: list[int],
    embedder: EmbeddingFn,
    entail_fn,
    cusum: dict,
    kappa: float,
) -> list[dict]:
    curve = []
    for lam in LAMBDA_GRID:
        labels, preds, latencies, _ = run_cascade(
            traces, tuning, lam, embedder, entail_fn, cusum, kappa
        )
        entry = {"lambda": lam, "kappa": round(kappa, 6)}
        entry.update(binary_metrics(labels, preds))
        entry["mean_latency_ms"] = round(float(np.mean(latencies)), 3)
        curve.append(entry)
    return curve


def pick_best(curve: list[dict]) -> dict:
    """Best F1; ties broken by recall, then precision, then the LOWEST lambda
    in the grid (deterministic)."""
    return max(curve, key=lambda e: (e["f1"], e["recall"], e["precision"], -e["lambda"]))


def tune_baseline_threshold(
    traces: list[LabeledTrace],
    tuning: list[int],
    embedder: EmbeddingFn,
) -> tuple[float, list[dict]]:
    """Trace alerts if ANY step's own H exceeds the threshold; best F1 on the
    tuning split. H does not depend on the threshold, but each grid point runs
    the real single_step_h_baseline code path (no precomputed shortcuts)."""
    curve = []
    for threshold in BASELINE_THRESHOLD_GRID:
        labels: list[bool] = []
        preds: list[bool] = []
        for i in tuning:
            flags = single_step_h_baseline(traces[i].graph, embedder, threshold)
            labels.append(traces[i].labels["s3"])
            preds.append(any(flags.values()))
        entry = {"threshold": threshold}
        entry.update(binary_metrics(labels, preds))
        curve.append(entry)
    best = max(curve, key=lambda e: (e["f1"], e["recall"], e["precision"], -e["threshold"]))
    return best["threshold"], curve


def per_step_h_breakdown(traces: list[LabeledTrace], embedder: EmbeddingFn) -> dict:
    """Mean H per step, split by s3 label - shows whether raw signal exists.

    H(s2) is undefined for every trace (reason step carries no retrieved
    context): reported as a count, never as 0 (AGENTS.md - None is not 0).
    """
    by_label: dict[bool, list[float]] = {True: [], False: []}
    s1_h: list[float] = []
    for trace in traces:
        s3_h = hallucination_score(
            trace.graph.nodes["s3"].output, trace.graph.nodes["s3"].retrieved_docs, embedder
        )
        by_label[trace.labels["s3"]].append(s3_h if s3_h is not None else 0.0)
        s1_h.append(
            hallucination_score(
                trace.graph.nodes["s1"].output, trace.graph.nodes["s1"].retrieved_docs, embedder
            )
        )

    def _stats(values: list[float]) -> dict:
        return {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "n": len(values),
        }

    gap = float(np.mean(by_label[True]) - np.mean(by_label[False]))
    return {
        "s3_synthesize": {
            "hallucinated": _stats(by_label[True]),
            "right": _stats(by_label[False]),
            "gap_mean": round(gap, 4),
            "interpretation": (
                (
                    "H(hallucinated) < H(right): under the flag-if-H>threshold semantics the "
                    "raw signal is INVERTED on this slice - a length/verbatim-reuse artifact "
                    "of the hashing embedder (short gold-entity answers vs sentence-form "
                    "hallucinations copied from the knowledge). See caveats."
                )
                if gap < 0
                else "H(hallucinated) > H(right): signal points the expected direction."
            ),
        },
        "s1_retrieve": _stats(s1_h),
        "s2_reason": {
            "undefined_no_context": len(traces),
            "note": (
                "H undefined (no retrieved docs) for every trace - "
                "treated as not applicable, never 0"
            ),
        },
    }


def chaf_depth_check(
    traces: list[LabeledTrace],
    lam: float,
    embedder: EmbeddingFn,
    entail_fn,
    cusum: dict,
) -> dict:
    """CHAF <= depth+1 (Theorem 1) across every trace at the chosen lambda."""
    engine = make_engine(lam, embedder, entail_fn, cusum)
    violations = 0
    max_ratio = 0.0
    for trace in traces:
        result = engine.analyze_run(trace.graph)
        for nid, chaf_value in result.chaf.items():
            bound = trace.graph.depth(nid) + 1
            max_ratio = max(max_ratio, chaf_value / bound)
            if chaf_value > bound + 1e-9:
                violations += 1
    return {
        "traces": len(traces),
        "violations": violations,
        "max_chaf_over_bound": round(max_ratio, 4),
        "lambda_used": lam,
    }


def print_summary(payload: dict) -> None:
    meta = payload["meta"]
    print("== CascadeGuard detection eval - HaluEval-RAG sample (150 records) ==")
    print(
        f"split: tuning {meta['split']['tuning']['n']} "
        f"({meta['split']['tuning']['pos']} pos / {meta['split']['tuning']['neg']} neg) | "
        f"held-out {meta['split']['held_out']['n']} "
        f"({meta['split']['held_out']['pos']} pos / {meta['split']['held_out']['neg']} neg) | "
        f"kappa={payload['alert_rule']['kappa']:.4f} [{payload['alert_rule']['kappa_source']}]"
    )
    print("lambda sweep (tuning split, alert = max CHRS > kappa):")
    print("  lambda  precision  recall    F1     FAR    lat_ms")
    for e in payload["lambda_sweep"]:
        marker = "  <-- chosen" if e["lambda"] == payload["chosen_lambda"] else ""
        print(
            f"  {e['lambda']:.1f}    {e['precision']:.4f}   {e['recall']:.4f}  "
            f"{e['f1']:.4f}  {e['false_alarm_rate']:.4f}  {e['mean_latency_ms']:.2f}{marker}"
        )
    h2h = payload["head_to_head"]
    print(f"head-to-head (held-out, n={h2h['held_out_n']}):")
    print("  detector               thresh  precision  recall  FAR      F1    lat_ms")
    for name in ("cascadeguard", "single_step_h_baseline"):
        m = h2h[name]
        print(
            f"  {name:<22} {m['threshold']:.4f}  {m['precision']:.4f}    "
            f"{m['recall']:.4f}  {m['false_alarm_rate']:.4f}  {m['f1']:.4f}"
            f"  {m['mean_latency_ms']:.2f}"
        )
    b = payload["per_step_H_breakdown"]["s3_synthesize"]
    print(
        "per-step H (synthesize, all 150): "
        f"hallucinated={b['hallucinated']['mean']:.4f} vs right={b['right']['mean']:.4f} "
        f"(gap {b['gap_mean']:+.4f})"
    )
    chk = payload["chaf_check"]
    print(
        f"CHAF <= depth+1: {chk['traces']} traces, {chk['violations']} violations, "
        f"max ratio {chk['max_chaf_over_bound']:.4f}"
    )
    print(f"wrote {payload['meta']['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection eval on HaluEval-RAG sample (Phase 14)")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--dim", type=int, default=256, help="HashingEmbedder dimensionality")
    args = parser.parse_args()

    traces = load_halu_eval_rag_sample(args.sample)
    embedder = make_embedder(args.dim)
    entail_fn = make_entail_fn(embedder)
    cusum = load_cusum_params(args.calibration)
    kappa = cusum["kappa"]

    tuning, held_out = stratified_split(traces, args.train_frac, args.seed)

    # --- lambda sweep on the tuning split -----------------------------------
    curve = sweep_lambda(traces, tuning, embedder, entail_fn, cusum, kappa)
    best = pick_best(curve)
    chosen_lambda = best["lambda"]

    # --- head-to-head on held-out -------------------------------------------
    cg_labels, cg_preds, cg_latencies, _ = run_cascade(
        traces, held_out, chosen_lambda, embedder, entail_fn, cusum, kappa
    )
    cg_metrics = binary_metrics(cg_labels, cg_preds)
    cg_metrics["threshold"] = round(kappa, 6)
    cg_metrics["mean_latency_ms"] = round(float(np.mean(cg_latencies)), 3)
    cg_metrics["p95_latency_ms"] = round(float(np.percentile(cg_latencies, 95)), 3)

    base_threshold, base_curve = tune_baseline_threshold(traces, tuning, embedder)
    base_labels: list[bool] = []
    base_preds: list[bool] = []
    base_latencies: list[float] = []
    for i in held_out:
        start = time.perf_counter()
        flags = single_step_h_baseline(traces[i].graph, embedder, base_threshold)
        base_latencies.append((time.perf_counter() - start) * 1000.0)
        base_labels.append(traces[i].labels["s3"])
        base_preds.append(any(flags.values()))
    base_metrics = binary_metrics(base_labels, base_preds)
    base_metrics["threshold"] = base_threshold
    base_metrics["mean_latency_ms"] = round(float(np.mean(base_latencies)), 3)
    base_metrics["p95_latency_ms"] = round(float(np.percentile(base_latencies, 95)), 3)

    # undetected-rate reduction (paper claims 34.7% vs best single-step baseline)
    base_undetected = 1 - base_metrics["recall"]
    cg_undetected = 1 - cg_metrics["recall"]
    reduction = (base_undetected - cg_undetected) / base_undetected if base_undetected > 0 else 0.0

    breakdown = per_step_h_breakdown(traces, embedder)
    chaf = chaf_depth_check(traces, chosen_lambda, embedder, entail_fn, cusum)

    payload = {
        "meta": {
            "date": "2026-10-31",
            "phase": "TODO Phase 14 (small-scale real-dataset eval)",
            "sample": {
                "path": str(args.sample),
                "source": "RUCAIBox/HaluEval data/qa_data.json (HaluEval QA / RAG split)",
                "url": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
                "license": "see repo (CC BY-NC-SA)",
                "n_records": len(traces),
                "note": (
                    "Upstream no longer ships data/halu_eval_rag.json (404 on master and "
                    "main, verified 2026-10-31); qa_data.json is the same official "
                    "10K-record HotpotQA-based split with the identical schema. See the "
                    "sample header provenance_note and eval/datasets/make_halu_sample.py."
                ),
            },
            "split": {
                "seed": args.seed,
                "train_frac": args.train_frac,
                "tuning": {
                    "n": len(tuning),
                    "pos": sum(traces[i].labels["s3"] for i in tuning),
                    "neg": sum(not traces[i].labels["s3"] for i in tuning),
                },
                "held_out": {
                    "n": len(held_out),
                    "pos": sum(traces[i].labels["s3"] for i in held_out),
                    "neg": sum(not traces[i].labels["s3"] for i in held_out),
                },
            },
            "engine": {
                "embedder": (
                    f"HashingEmbedder(dim={args.dim}) - lexical char-trigram placeholder (D-005)"
                ),
                "entail_fn": (
                    "cosine similarity surrogate (replicated from api/main.build_engine; "
                    "D-002 open)"
                ),
                "semantic_entropy": 0,
                "alpha_track": "nli_proxy",
                "trace_shape": "s1 retrieve -> s2 reason -> s3 synthesize; label on s3",
            },
            "output_path": str(args.output),
        },
        "alert_rule": {
            "cascade": "trace alerts iff max CHRS over steps > kappa",
            "kappa": round(kappa, 6),
            "kappa_formula": "mu0 + delta/2 (mirrors CusumDetector, Eq. 11)",
            "kappa_source": cusum["source"],
            "comparison": "strict > (same strictness as CusumDetector.update)",
            "note": (
                "CHRS = lambda*H_cascade (SE=0), so a FIXED kappa acts as a per-lambda "
                "shifted threshold: the effective cutoff in H_cascade units is kappa/lambda."
            ),
        },
        "lambda_sweep": curve,
        "chosen_lambda": chosen_lambda,
        "chosen_lambda_metrics_tuning": best,
        "baseline_threshold_tuning": {"best_threshold": base_threshold, "curve": base_curve},
        "head_to_head": {
            "held_out_n": len(held_out),
            "held_out_pos": sum(cg_labels),
            "held_out_neg": sum(not x for x in cg_labels),
            "cascadeguard": cg_metrics,
            "single_step_h_baseline": base_metrics,
            "undetected_rate_reduction_vs_baseline": round(reduction, 4),
        },
        "per_step_H_breakdown": breakdown,
        "chaf_check": chaf,
        "caveats": [
            "HONESTY FRAME: hashing char-trigram embedder (D-005) measures lexical overlap, "
            "not semantic faithfulness; absolute numbers are modest and NOT the paper's claim.",
            "entail_fn is a cosine surrogate, not a real NLI model (D-002 open); SE=0 "
            "(no generate_fn wired), so CHRS = lambda*H_cascade exactly.",
            "Trace construction gives zero upstream risk: s1 echoes its only document "
            "(H=0) and s2 has no retrieved context (H undefined -> 0 in the cascade), so "
            "H_cascade(s3) == H(s3) and the pipeline-aware pathway adds NO signal on this "
            "slice. The head-to-head validates the harness and thresholding, not the "
            "paper's cascade-amplification advantage.",
            "H on this slice partially confounds answer LENGTH with hallucination: gold "
            "answers are short entities while hallucinated answers are sentences, which "
            "inflates the H gap under a lexical embedder.",
            "Sample-scale: 150 of 10,000 records (106 tuning / 44 held-out); wide "
            "confidence intervals, numbers are provisional.",
            "kappa comes from calibration on synthetic-clean CHRS, not this dataset's "
            "clean traces; per-lambda thresholding is fixed rather than recalibrated.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print_summary(payload)

    if chaf["violations"] > 0:
        print("FAIL: Theorem 1 violated on real data - alpha estimation bug, do not loosen")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
