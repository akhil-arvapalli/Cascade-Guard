export const API_BASE = '';

async function json(path: string, opts?: RequestInit) {
  const res = await fetch(API_BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) throw new Error(`API ${res.status} ${await res.text()}`);
  return res.json();
}

export interface StepNode {
  id: string;
  kind: string;
  output: string;
  retrieved_docs: { doc_id: string; content: string }[];
  predecessors: string[];
  tool_invocation: any;
  timestamp: number;
  depth: number;
  H: number | null;
  h_cascade: number | null;
  chaf: number | null;
  chrs: number | null;
  se: number | null;
  fidelity: string;
}

export interface Edge {
  src: string;
  dst: string;
  alpha: number | null;
  alpha_method: string | null;
}

export interface AlertRecord {
  alert_id: string;
  run_id: string;
  triggered_at_step: string;
  s_n_value: number;
  root_cause_step: string | null;
  created_at: number;
}

export interface RunSummary {
  run_id: string;
  pipeline_template_id: string | null;
  started_at: number | null;
  status: string;
  step_count: number;
  alerted: boolean;
  analyze_ms: number | null;
}

export interface RunDetail {
  run_id: string;
  run_record: any;
  alerted: boolean;
  analyze_ms: number | null;
  nodes: StepNode[];
  edges: Edge[];
  alerts: AlertRecord[];
  scores_summary: {
    max_chrs: number | null;
    max_chaf: number | null;
    monitored_count: number;
    total_steps: number;
  };
}

export const api = {
  listRuns: () => json('/runs') as Promise<{ runs: RunSummary[] }>,
  getRun: (runId: string) => json(`/runs/${runId}`) as Promise<RunDetail>,
  listAlerts: (runId?: string) =>
    json(`/alerts${runId ? `?run_id=${encodeURIComponent(runId)}` : ''}`) as Promise<{ alerts: AlertRecord[] }>,
};
