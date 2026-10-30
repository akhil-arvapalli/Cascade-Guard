import { useCallback, useEffect, useState } from 'react';
import type { RunDetail } from './api';
import { api } from './api';
import { RunTable } from './components/RunTable';
import { AlertTable } from './components/AlertTable';
import { DAG } from './components/DAG';
import { NodeInspector } from './components/NodeInspector';

type View = 'runs' | 'alerts' | 'runDetail';

export default function App() {
  const [view, setView] = useState<View>('runs');
  const [runId, setRunId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [activeTab, setActiveTabState] = useState<'dag' | 'scores' | 'trace'>('dag');

  const showRuns = useCallback(() => {
    setView('runs');
    setRunId(null);
    setDetail(null);
    setSelectedNodeId(null);
  }, []);

  const showAlerts = useCallback(() => {
    setView('alerts');
    setRunId(null);
    setDetail(null);
    setSelectedNodeId(null);
  }, []);

  const showRun = useCallback((id: string) => {
    setView('runDetail');
    setRunId(id);
    setSelectedNodeId(null);
    loadRun(id);
  }, []);

  const loadRun = useCallback(async (id: string) => {
    setLoading(true);
    try {
      const data = await api.getRun(id);
      setDetail(data);
    } catch (e) {
      alert('Error loading run: ' + (e as Error).message);
      showRuns();
    } finally {
      setLoading(false);
    }
  }, [showRuns]);

  const refreshRun = useCallback(() => {
    if (runId) loadRun(runId);
  }, [runId, loadRun]);

  const onViewRun = useCallback(
    (e: Event) => {
      const ce = e as CustomEvent;
      showRun(ce.detail as string);
    },
    [showRun],
  );

  useEffect(() => {
    window.addEventListener('viewRun', onViewRun as EventListener);
    return () => window.removeEventListener('viewRun', onViewRun as EventListener);
  }, [onViewRun]);

  const alertedStep = detail?.alerts?.[0]?.triggered_at_step ?? null;

  // Compute root-cause path by walking predecessors from the alerted step.
  let rootCauseSteps: string[] = [];
  if (alertedStep && detail) {
    const predsById = new Map<string, string[]>();
    detail.nodes.forEach((n) => predsById.set(n.id, n.predecessors));
    const path: string[] = [];
    let cur: string | null = alertedStep;
    const visited = new Set<string>();
    while (cur && !visited.has(cur)) {
      visited.add(cur);
      path.push(cur);
      const ps: string[] = predsById.get(cur as string) || [];
      cur = ps.length > 0 ? ps[ps.length - 1] : null;
    }
    rootCauseSteps = path;
  }

  const highlightedEdge = selectedNodeId
    ? detail?.edges.find((e) => e.src === selectedNodeId || e.dst === selectedNodeId)
    : undefined;

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="mark">CG</div>
          <div>
            CascadeGuard
            <span style={{ fontWeight: 400, fontSize: '0.85rem', color: '#94a3b8', marginLeft: '6px' }}>
              Dashboard
            </span>
          </div>
        </div>
        <nav className="app-nav">
          <a href="#" className={view === 'runs' ? 'active' : ''} onClick={(e) => { e.preventDefault(); showRuns(); }}>
            Runs
          </a>
          <a href="#" className={view === 'alerts' ? 'active' : ''} onClick={(e) => { e.preventDefault(); showAlerts(); }}>
            Alerts
          </a>
          <a href="/docs" target="_blank" rel="noopener noreferrer">API Docs</a>
        </nav>
      </header>

      <div className="container">
        <div id="tooltip" className="tooltip" />

        {view === 'runs' && (
          <section>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
              <h2>Pipeline Runs</h2>
              <button className="btn primary" onClick={() => showRuns()}>
                ↻ Refresh
              </button>
            </div>
            <RunTable onSelect={showRun} />
          </section>
        )}

        {view === 'alerts' && (
          <section>
            <h2>Alerts</h2>
            <AlertTable />
          </section>
        )}

        {view === 'runDetail' && detail && (
          <section>
            <div className="run-header">
              <div className="run-info">
                <h1>
                  Run <span style={{ color: '#63b3ed' }}>{detail.run_id}</span>
                </h1>
                <div className="meta">
                  Template: <span className="mono">{detail.run_record.pipeline_template_id || 'adhoc'}</span>
                  &nbsp;|&nbsp; Status: {detail.run_record.status || '—'}
                  &nbsp;|&nbsp; Analyze: {detail.analyze_ms ?? '—'} ms
                  &nbsp;|&nbsp; Steps: {detail.scores_summary.total_steps}
                </div>
              </div>
              <div className="run-actions">
                <button className="btn" onClick={showRuns}>← Back to runs</button>
                <button className="btn primary" onClick={refreshRun}>↻ Refresh</button>
                <button className="btn small" onClick={exportRun}>Export JSON</button>
              </div>
            </div>

            {detail.alerted && detail.alerts.length > 0 && (
              <div className="alert-banner">
                <div className="icon">⚠</div>
                <div className="msg">
                  <h4>
                    Hallucination cascade detected at step &quot;{detail.alerts[0].triggered_at_step}&quot;
                  </h4>
                  <p>
                    CUSUM S_n = {(detail.alerts[0].s_n_value ?? 0).toFixed(4)}. Root cause:
                    {detail.alerts[0].root_cause_step || 'unknown'}.
                  </p>
                </div>
              </div>
            )}

            <div className="metrics">
              <div className="metric">
                <div className="label">Max CHRS</div>
                <div className="value">{detail.scores_summary.max_chrs != null ? detail.scores_summary.max_chrs.toFixed(4) : '—'}</div>
              </div>
              <div className="metric">
                <div className="label">Max CHAF</div>
                <div className="value">{detail.scores_summary.max_chaf != null ? detail.scores_summary.max_chaf.toFixed(4) : '—'}</div>
              </div>
              <div className="metric">
                <div className="label">Total Steps</div>
                <div className="value">{detail.scores_summary.total_steps}</div>
              </div>
              <div className="metric">
                <div className="label">Monitored (full)</div>
                <div className="value">{detail.scores_summary.monitored_count}</div>
              </div>
              <div className="metric">
                <div className="label">Alerted</div>
                <div className="value">{detail.alerted ? 'YES' : 'no'}</div>
              </div>
              <div className="metric">
                <div className="label">Analyze (ms)</div>
                <div className="value">{detail.analyze_ms ?? '—'}</div>
              </div>
            </div>

            <div className="tabs">
              <button className="tab active" onClick={() => setActiveTabState('dag')}>
                DAG Visualization
              </button>
              <button className="tab" onClick={() => setActiveTabState('scores')}>
                Scores Table
              </button>
              <button className="tab" onClick={() => setActiveTabState('trace')}>
                Trace / Steps
              </button>
            </div>

            <div className={`tabpanel ${activeTab === 'dag' ? 'active' : 'hidden'}`}>
              <div className="card">
                <div className="dag-wrapper">
                  <DAG
                    nodes={detail.nodes}
                    edges={detail.edges}
                    alertedStep={alertedStep}
                    rootCauseSteps={rootCauseSteps}
                    highlightedEdge={highlightedEdge ? { src: highlightedEdge.src, dst: highlightedEdge.dst } : null}
                    onNodeSelect={setSelectedNodeId}
                    selectedNodeId={selectedNodeId}
                  />
                </div>
              </div>
              <div className="card" style={{ marginTop: 0 }}>
                <h3>Node Inspector</h3>
                <p className="text-xs">Click a node in the DAG to inspect its scores and root-cause path.</p>
                <NodeInspector
                  node={selectedNodeId ? detail.nodes.find((n) => n.id === selectedNodeId) ?? null : null}
                  rootCauseSteps={rootCauseSteps}
                />
              </div>
            </div>

            <div className={`tabpanel ${activeTab === 'scores' ? 'active' : 'hidden'}`}>
              <div className="card">
                <table>
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>Kind</th>
                      <th>Depth</th>
                      <th>H (faithfulness)</th>
                      <th>H_cascade</th>
                      <th>CHAF</th>
                      <th>CHRS</th>
                      <th>SE</th>
                      <th>Fidelity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.nodes.map((n) => {
                      const chrs = n.chrs != null ? n.chrs.toFixed(4) : '—';
                      const chaf = n.chaf != null ? n.chaf.toFixed(4) : '—';
                      const hc = n.h_cascade != null ? n.h_cascade.toFixed(4) : '—';
                      const se = n.se != null ? n.se.toFixed(4) : '—';
                      const H = n.H != null ? n.H.toFixed(4) : '—';
                      const cls = n.chrs != null && n.chrs >= 0.6 ? 'text-red' : n.chrs != null && n.chrs >= 0.3 ? 'text-yellow' : 'text-green';
                      return (
                        <tr key={n.id} onClick={() => setSelectedNodeId(n.id)} className="clickable" style={{ cursor: 'pointer' }}>
                          <td><span className="mono">{n.id}</span></td>
                          <td>{n.kind}</td>
                          <td>{n.depth}</td>
                          <td>{H}</td>
                          <td className={cls}>{hc}</td>
                          <td className={cls}>{chaf}</td>
                          <td className={cls}>{chrs}</td>
                          <td>{se}</td>
                          <td><span className={`badge ${n.fidelity}`}>{n.fidelity}</span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className={`tabpanel ${activeTab === 'trace' ? 'active' : 'hidden'}`}>
              <div className="card">
                <table>
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>Kind</th>
                      <th>Predecessors</th>
                      <th>Output (truncated)</th>
                      <th>Retrieved Docs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.nodes.map((n) => {
                      const output = (n.output || '').slice(0, 120) + (n.output && n.output.length > 120 ? '…' : '');
                      return (
                        <tr key={n.id}>
                          <td><span className="mono">{n.id}</span></td>
                          <td>{n.kind}</td>
                          <td><span className="mono">{n.predecessors.join(', ') || '—'}</span></td>
                          <td>{escapeHtml(output)}</td>
                          <td>{n.retrieved_docs.length}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {loading && view === 'runDetail' && !detail && <div className="loading">Loading run…</div>}
      </div>
    </div>
  );

  function escapeHtml(s: string) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function exportRun() {
    if (!detail) return;
    const blob = new Blob([JSON.stringify(detail, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cascadeguard-run-${detail.run_id}.json`;
    a.click();
  }
}
