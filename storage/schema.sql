-- CascadeGuard trace store schema.
-- Faithful transcription of README §5; applied automatically via docker-compose initdb.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id UUID PRIMARY KEY,
    pipeline_template_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT CHECK (status IN ('running','completed','failed','alerted'))
);

CREATE TABLE IF NOT EXISTS steps (
    step_id TEXT PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(run_id),
    kind TEXT NOT NULL,
    output TEXT,
    output_embedding VECTOR(768),
    retrieved_doc_ids TEXT[],
    tool_call JSONB,
    h_score FLOAT,
    h_cascade FLOAT,
    chaf FLOAT,
    chrs FLOAT,
    semantic_entropy FLOAT,
    monitored_fidelity TEXT CHECK (monitored_fidelity IN ('cheap','full')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS step_edges (
    run_id UUID REFERENCES pipeline_runs(run_id),
    from_step TEXT REFERENCES steps(step_id),
    to_step TEXT REFERENCES steps(step_id),
    alpha FLOAT,
    alpha_method TEXT CHECK (alpha_method IN ('attention','nli_proxy','counterfactual')),
    PRIMARY KEY (from_step, to_step)
);

CREATE TABLE IF NOT EXISTS cusum_states (
    run_id UUID PRIMARY KEY REFERENCES pipeline_runs(run_id),
    s_n FLOAT NOT NULL,
    kappa FLOAT NOT NULL,
    h_control_limit FLOAT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id UUID PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(run_id),
    triggered_at_step TEXT REFERENCES steps(step_id),
    s_n_value FLOAT,
    root_cause_step TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Single-row CUSUM calibration store (V2-3): `id` is a constant TRUE so the
-- table can never hold more than one calibration (CHECK enforces it).
CREATE TABLE IF NOT EXISTS calibration (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    mu0 DOUBLE PRECISION NOT NULL,
    sigma DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
