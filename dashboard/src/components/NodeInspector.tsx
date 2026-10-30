import type { StepNode } from '../api';

interface NodeInspectorProps {
  node: StepNode | null;
  rootCauseSteps: string[];
}

export function NodeInspector({ node, rootCauseSteps }: NodeInspectorProps) {
  if (!node) {
    return (
      <div className="empty">Select a node in the DAG to inspect its scores and root-cause path.</div>
    );
  }

  const chrs = node.chrs != null ? node.chrs.toFixed(4) : '—';
  const chaf = node.chaf != null ? node.chaf.toFixed(4) : '—';
  const hc = node.h_cascade != null ? nH(node.h_cascade) : '—';
  const se = node.se != null ? node.se.toFixed(4) : '—';
  const H = node.H != null ? node.H.toFixed(4) : 'n/a';
  const cls = node.chrs != null && node.chrs >= 0.6 ? 'text-red' : node.chrs != null && node.chrs >= 0.3 ? 'text-yellow' : 'text-green';
  const isRootCause = rootCauseSteps.includes(node.id);

  return (
    <div className="node-detail">
      <h4>{node.id}</h4>
      <div className="kv">
        <div className="k">Kind</div>
        <div className="v">{node.kind}</div>
        <div className="k">Depth</div>
        <div className="v">{node.depth}</div>
        <div className="k">H (faithfulness)</div>
        <div className="v" title={node.H == null ? 'no retrieval context for this step (Eq. 3 needs R(s_i))' : undefined}>{H}</div>
        <div className="k">H_cascade</div>
        <div className="v">{hc}</div>
        <div className="k">CHAF</div>
        <div className="v">{chaf}</div>
        <div className={`k v ${cls}`}>CHRS</div>
        <div className={`v ${cls}`}>{chrs}</div>
        <div className="k">Semantic Entropy</div>
        <div className="v">{se}</div>
        <div className="k">Fidelity</div>
        <div className="v"><span className={`badge ${node.fidelity}`}>{node.fidelity}</span></div>
        <div className="k">Predecessors</div>
        <div className="v mono">{node.predecessors.join(', ') || '—'}</div>
        <div className="k">Root Cause Path</div>
        <div className="v">{isRootCause ? 'YES — on path to alerted step' : 'no'}</div>
      </div>
      <div className="section-label">Output</div>
      <div className="mono output-box">{escapeHtml(node.output || '')}</div>
      {node.retrieved_docs && node.retrieved_docs.length > 0 && (
        <>
          <div className="section-label">Retrieved Docs</div>
          {node.retrieved_docs.map((d, i) => (
            <div key={i} className="doc-row">
              <span className="mono">{d.doc_id}:</span> {escapeHtml(d.content)}
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function nH(v: number | null): string {
  return v != null ? v.toFixed(4) : '—';
}

function escapeHtml(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
