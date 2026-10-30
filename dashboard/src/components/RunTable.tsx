import { useEffect, useState } from 'react';
import type { RunSummary } from '../api';
import { api } from '../api';

export function RunTable({ onSelect }: { onSelect: (runId: string) => void }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    try {
      const data = await api.listRuns();
      setRuns(data.runs || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <div className="loading">Loading runs…</div>;
  if (runs.length === 0) return <div className="empty">No runs yet. Create one via <code>POST /v1/runs</code>.</div>;

  return (
    <div className="card">
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Template</th>
            <th>Steps</th>
            <th>Status</th>
            <th>Alerted</th>
            <th>Analyze (ms)</th>
            <th>Started</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id} className="clickable" onClick={() => onSelect(r.run_id)}>
              <td><span className="mono">{r.run_id}</span></td>
              <td>{r.pipeline_template_id || '—'}</td>
              <td>{r.step_count ?? '—'}</td>
              <td>
                <span className="status">
                  <span className={`dot ${r.alerted ? 'alert' : 'ok'}`}></span>
                  {r.alerted ? 'ALERTED' : 'ok'}
                </span>
              </td>
              <td>
                <span className={`badge ${r.alerted ? 'alert' : 'ok'}`}>
                  {r.alerted ? 'YES' : 'no'}
                </span>
              </td>
              <td>{r.analyze_ms ?? '—'}</td>
              <td>{r.started_at ? new Date(r.started_at * 1000).toLocaleString() : '—'}</td>
              <td>
                <span className={`badge ${r.alerted ? 'alert' : 'ok'}`}>
                  {r.alerted ? 'alert' : 'clean'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
