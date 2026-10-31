import { useEffect, useState } from 'react';

interface SampleRow {
  sample: string;
  truth: string;
  max_chrs: number;
  predicted: string;
  match: boolean;
}

export function EvalPanel() {
  const [summary, setSummary] = useState<any>(null);
  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [kappa, setKappa] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    try {
      const s = await fetch('/v1/eval/summary').then((r) => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      });
      setSummary(s);
      const v = await fetch('/v1/eval/samples').then((r) => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      });
      setSamples(v.samples || []);
      setKappa(v.kappa ?? null);
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  if (err) return <div className="empty">Eval results unavailable: {err}</div>;
  if (!summary) return <div className="loading">Loading evaluation</div>;

  const latency = summary.latency || {};
  const t1 = summary.theorem1 || {};
  const violations = t1.violations ?? t1.violation_count ?? 0;
  const hallucinated = samples.filter((s) => s.truth === 'hallucinated');
  const caught = hallucinated.filter((s) => s.predicted === 'hallucinated').length;

  return (
    <section>
      <h2>Eval Sample — 150-trace HaluEval-RAG slice (75 faithful / 75 hallucinated)</h2>

      <div className="metrics">
        <div className="metric">
          <div className="label">Traces scored</div>
          <div className="value">{samples.length || '—'}</div>
        </div>
        <div className="metric">
          <div className="label">Cascades caught</div>
          <div className="value">
            {hallucinated.length ? `${caught}/${hallucinated.length}` : '—'}
          </div>
        </div>
        <div className="metric">
          <div className="label">Theorem 1 violations</div>
          <div className="value">{violations}</div>
        </div>
        <div className="metric">
          <div className="label">Analyze p50 / budget</div>
          <div className="value">
            {latency.p50 != null ? `${Number(latency.p50).toFixed(1)} / 12 ms` : '—'}
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Measured figures</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
          {(summary.figures || []).map((src: string, i: number) => (
            <figure key={i} style={{ margin: 0 }}>
              <img src={src} alt={`eval figure ${i + 1}`} style={{ width: 430, border: '1px solid #2c2c29' }} />
            </figure>
          ))}
        </div>
      </div>

      {kappa != null && (
        <div className="card">
          <h3>Alert rule</h3>
          <p className="text-xs">
            A trace alerts when max CHRS &gt; κ = {kappa} (calibrated μ0 + δ/2). Recall on cascades is
            1.00; the false-alarm side is elevated because the placeholder hashing embedder inverts the
            H signal on sentence-form answers — documented in EVAL_REPORT.md, fixed by V2-1 (real embedder).
          </p>
        </div>
      )}

      <div className="card">
        <h3>All 150 sample verdicts</h3>
        <div style={{ maxHeight: 420, overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Sample</th>
                <th>Ground truth</th>
                <th>Max CHRS</th>
                <th>Predicted</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {samples.map((s) => (
                <tr key={s.sample}>
                  <td><span className="mono">{s.sample}</span></td>
                  <td><span className={`badge ${s.truth === 'hallucinated' ? 'alert' : 'ok'}`}>{s.truth}</span></td>
                  <td>{s.max_chrs.toFixed(4)}</td>
                  <td><span className={`badge ${s.predicted === 'hallucinated' ? 'alert' : 'ok'}`}>{s.predicted}</span></td>
                  <td>{s.match ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
