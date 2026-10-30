import { useEffect, useState } from 'react';
import type { AlertRecord } from '../api';
import { api } from '../api';

export function AlertTable({ runId }: { runId?: string }) {
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    load();
  }, [runId]);

  async function load() {
    setLoading(true);
    try {
      const data = await api.listAlerts(runId);
      setAlerts(data.alerts || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <div className="loading">Loading alerts…</div>;
  if (alerts.length === 0) return <div className="empty">No alerts.</div>;

  return (
    <div className="card">
      <table>
        <thead>
          <tr>
            <th>Alert ID</th>
            <th>Run ID</th>
            <th>Triggered At Step</th>
            <th>S_n Value</th>
            <th>Root Cause</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((a) => (
            <tr key={a.alert_id} className="clickable" onClick={() => runId && window.dispatchEvent(new CustomEvent('viewRun', { detail: a.run_id }))}>
              <td><span className="mono">{a.alert_id}</span></td>
              <td><span className="mono">{a.run_id}</span></td>
              <td><span className="mono">{a.triggered_at_step}</span></td>
              <td>{(a.s_n_value ?? 0).toFixed(4)}</td>
              <td><span className="mono">{a.root_cause_step || '—'}</span></td>
              <td>{a.created_at ? new Date(a.created_at * 1000).toLocaleString() : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
