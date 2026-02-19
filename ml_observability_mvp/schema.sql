-- ML Observability MVP Schema

-- 1. Inference Events
-- Stores raw inference telemetry. 
-- In a real prod system, 'features_json' might store sketches, but here we allow raw for MVP demo.
CREATE TABLE IF NOT EXISTS inference_events (
    ts TIMESTAMPTZ NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    request_id TEXT, -- Nullable, but recommended for joining labels
    pred_type TEXT CHECK (pred_type IN ('classification', 'regression', 'other')),
    latency_ms INT,
    features_json JSONB, -- Stores {"feature_name": value, ...} or summary blob
    y_pred_num DOUBLE PRECISION,
    y_pred_text TEXT,
    segment_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inference_model_ts ON inference_events (model_id, model_version, ts);
CREATE INDEX IF NOT EXISTS idx_inference_request_id ON inference_events (request_id);


-- 2. Label Events
-- Stores delayed ground truth.
CREATE TABLE IF NOT EXISTS label_events (
    ts_label TIMESTAMPTZ NOT NULL,
    model_id TEXT NOT NULL, -- To speed up lookups if request_id isn't unique globally (though it should be)
    request_id TEXT NOT NULL,
    y_true_num DOUBLE PRECISION,
    y_true_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_labels_request_id ON label_events (request_id);
-- Compound index might help if partitioning by model later
CREATE INDEX IF NOT EXISTS idx_labels_model_ts ON label_events (model_id, ts_label);


-- 3. Daily Metrics
-- Stores pre-computed health metrics (Drift, Reliability, Performance).
CREATE TABLE IF NOT EXISTS metrics_daily (
    day DATE NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    metric_name TEXT NOT NULL, -- e.g. 'psi__age', 'p95_latency_ms'
    metric_value DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (day, model_id, model_version, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metrics_daily (model_id, model_version, day);
