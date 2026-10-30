import { useEffect, useRef, useState } from 'react';
import type { Edge, StepNode } from '../api';

interface DAGProps {
  nodes: StepNode[];
  edges: Edge[];
  alertedStep: string | null;
  rootCauseSteps: string[];
  highlightedEdge: { src: string; dst: string } | null;
  onNodeSelect: (id: string) => void;
  selectedNodeId: string | null;
}

const NODE_W = 240;
const NODE_H = 92;

export function DAG({
  nodes,
  edges,
  alertedStep,
  rootCauseSteps,
  highlightedEdge,
  onNodeSelect,
  selectedNodeId,
}: DAGProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dims, setDims] = useState({ w: 1000, h: 650 });

  useEffect(() => {
    const svg = svgRef.current;
    if (svg) setDims({ w: svg.clientWidth || 1000, h: svg.clientHeight || 650 });
  }, []);

  // Layout by depth layering
  const byDepth = new Map<number, StepNode[]>();
  nodes.forEach((n) => {
    const d = n.depth != null ? n.depth : 0;
    const arr = byDepth.get(d) || [];
    arr.push(n);
    byDepth.set(d, arr);
  });
  const depths = Array.from(byDepth.keys()).sort((a, b) => a - b);

  const rootSet = new Set(rootCauseSteps);
  const pathEdges = new Set<string>();
  const sortedRoot = [...rootSet];
  for (let i = 0; i < sortedRoot.length - 1; i++) {
    pathEdges.add(`${sortedRoot[i]}->${sortedRoot[i + 1]}`);
  }

  const pos = new Map<string, { x: number; y: number }>();
  depths.forEach((d, di) => {
    const layer = byDepth.get(d)!;
    const layerW = dims.w - 120;
    const gap = layerW / Math.max(layer.length + 1, 1);
    layer.forEach((n, i) => {
      pos.set(n.id, { x: 60 + (i + 1) * gap - NODE_W / 2, y: 60 + di * ((dims.h - 80) / Math.max(depths.length, 1)) });
    });
  });

  const scoreClass = (v: number | null) => {
    if (v == null) return 'low';
    if (v >= 0.6) return 'high';
    if (v >= 0.3) return 'med';
    return 'low';
  };

  return (
    <div className="dag-wrapper" style={{ height: dims.h }}>
      <svg ref={svgRef} className="dag" width={dims.w} height={Math.max(dims.h, depths.length * 120 + 120)}>
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#2c2c29" />
          </marker>
        </defs>
        <g id="dag-edges">
          {edges.map((e) => {
            const s = pos.get(e.src);
            const d = pos.get(e.dst);
            if (!s || !d) return null;
            const isPath = pathEdges.has(`${e.src}->${e.dst}`);
            const isHighlight =
              highlightedEdge && highlightedEdge.src === e.src && highlightedEdge.dst === e.dst;
            const cls = (isPath || isHighlight ? 'edge highlight' : 'edge dim') + (isPath ? ' path' : '');
            const alpha = e.alpha != null ? e.alpha.toFixed(3) : '—';
            const method = e.alpha_method || '—';
            return (
              <g key={`${e.src}-${e.dst}`}>
                <line
                  x1={s.x + NODE_W / 2}
                  y1={s.y + NODE_H}
                  x2={d.x + NODE_W / 2}
                  y2={d.y}
                  className={cls}
                  markerEnd="url(#arrowhead)"
                  data-src={e.src}
                  data-dst={e.dst}
                />
                <text
                  x={(s.x + d.x) / 2}
                  y={(s.y + d.y) / 2 - 10}
                  textAnchor="middle"
                  fontSize="9"
                  className={isPath ? 'edge-alpha path' : 'edge-alpha'}
                  pointerEvents="none"
                >
                  α={alpha}
                </text>
                <title>{`${e.src} → ${e.dst}  α=${alpha} (${method})`}</title>
              </g>
            );
          })}
        </g>
        <g id="dag-nodes">
          {nodes.map((n) => {
            const p = pos.get(n.id);
            if (!p) return null;
            const isAlerted = n.id === alertedStep;
            const isRoot = rootSet.has(n.id);
            const chrs = n.chrs != null ? n.chrs : 0;
            const cls = scoreClass(n.chrs);
            const chaf = n.chaf != null ? n.chaf.toFixed(3) : '—';
            const hc = n.h_cascade != null ? n.h_cascade.toFixed(3) : '—';
            const se = n.se != null ? n.se.toFixed(3) : '—';
            return (
              <g
                key={n.id}
                className={`node ${isAlerted ? 'alerted' : ''} ${isRoot ? 'root-cause' : ''} ${selectedNodeId === n.id ? 'selected' : ''}`}
                transform={`translate(${p.x},${p.y})`}
                onClick={(ev) => {
                  ev.stopPropagation();
                  onNodeSelect(n.id);
                }}
                onMouseEnter={(ev: React.MouseEvent) => {
                  const tt = document.getElementById('dag-tooltip');
                  if (tt) {
                    tt.classList.add('visible');
                    tt.style.left = (ev.clientX + 12) + 'px';
                    tt.style.top = (ev.clientY + 12) + 'px';
                    tt.innerHTML = `
                      <div><strong>${n.id}</strong> (${n.kind}, depth ${n.depth})</div>
                      <div>H_cascade: ${hc} • CHAF: ${chaf} • CHRS: ${chrs.toFixed(4)}</div>
                      <div>SE: ${se} • ${n.fidelity}</div>
                    `;
                  }
                }}
                onMouseLeave={() => {
                  const tt = document.getElementById('dag-tooltip');
                  if (tt) tt.classList.remove('visible');
                }}
              >
                <rect width={NODE_W} height={NODE_H} />
                <text x="10" y="22" fontSize="10" className="node-id">{n.id}</text>
                <text x="10" y="38" fontSize="10" className="node-meta">{`${n.kind} • depth ${n.depth}`}</text>
                <text x="10" y="56" fontSize="13" className={`chrs-val ${cls}`}>
                  CHRS {chrs.toFixed(3)}
                </text>
                <text x="10" y="72" fontSize="9" className="node-meta">
                  {`H_c=${hc} CHAF=${chaf} SE=${se} • ${n.fidelity}`}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      <div id="dag-tooltip" className="tooltip" />
    </div>
  );
}
