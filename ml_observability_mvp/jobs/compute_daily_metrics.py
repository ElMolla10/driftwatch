import os
import datetime
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Tuple, Dict, Any

# Configurations
DATABASE_URL = os.getenv("DATABASE_URL")
DRIFT_Features = os.getenv("MODEL_DRIFT_FEATURES", "").split(",")

def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL env var not set")
    return psycopg2.connect(DATABASE_URL)

def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) for a single feature.
    Uses quantile binning from expected (baseline) to apply to actual.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define bins based on expected distribution (Baselne)
    try:
        # qcut handles quantiles. duplicates='drop' merges bins if many values are same (e.g. 0)
        _, bins = pd.qcut(expected, q=buckets, retbins=True, duplicates='drop')
    except ValueError:
        # Fallback if too few unique values
        return 0.0
    
    # Force bins to cover -inf/inf to include all new data
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Bin counts
    expected_counts = pd.cut(expected, bins=bins).value_counts(sort=False)
    actual_counts = pd.cut(actual, bins=bins).value_counts(sort=False)

    # Convert to proportions
    expected_percents = (expected_counts / len(expected)).replace(0, 0.0001) # Avoid div/0 later
    actual_percents = (actual_counts / len(actual)).replace(0, 0.0001)

    # PSI formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = psi_values.sum()
    
    return float(psi)

def run_job(target_day_str: str = None):
    """
    Main job entrypoint.
    target_day_str: 'YYYY-MM-DD'
    """
    target_day = datetime.date.today()
    if target_day_str:
        target_day = datetime.datetime.strptime(target_day_str, "%Y-%m-%d").date()
    
    print(f"Running Metrics Job for {target_day}...")
    
    conn = get_db_connection()
    
    # 1. Get Models active on target day
    # We query raw events. In a bigger system, we'd query a registry or manifest.
    with conn.cursor() as cur:
        query = "SELECT DISTINCT model_id, model_version FROM inference_events WHERE ts::date = %s"
        cur.execute(query, (target_day,))
        models = cur.fetchall()

    for model_id, model_version in models:
        print(f"Processing Model: {model_id} v{model_version}")
        process_model_metrics(conn, model_id, model_version, target_day)
        
    conn.close()
    print("Job Complete.")

def process_model_metrics(conn, model_id, model_version, target_day):
    metrics = []

    # --- A. Data Fetching ---
    
    # Current Day Data
    # Note: We are fetching Raw JSON features. 
    query_current = """
        SELECT features_json, latency_ms, pred_type, request_id, y_pred_num
        FROM inference_events
        WHERE model_id = %s AND model_version = %s AND ts::date = %s
    """
    df_current = pd.read_sql(query_current, conn, params=(model_id, model_version, target_day))
    
    if len(df_current) == 0:
        print("  No data for current day. Skipping.")
        return

    # Baseline Data (First 14 days of this model version)
    # We define baseline as "earliest 14 days available for this model version" 
    # OR simpler: "Window [TargetDay - 14, TargetDay - 1]"? 
    # The prompt says: "Baseline window default: first 14 days of events for the model/version"
    
    # Find start date of this model
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts)::date FROM inference_events WHERE model_id=%s AND model_version=%s", (model_id, model_version))
        min_date = cur.fetchone()[0]
    
    # Actually fetch baseline data. Limit 100k rows to avoid exploding memory in this MVP script.
    # Baseline window = [min_date, min_date + 14 days]
    baseline_end = min_date + datetime.timedelta(days=14)
    
    query_baseline = """
        SELECT features_json
        FROM inference_events
        WHERE model_id = %s AND model_version = %s 
          AND ts::date >= %s AND ts::date < %s
        LIMIT 100000
    """
    df_baseline = pd.read_sql(query_baseline, conn, params=(model_id, model_version, min_date, baseline_end))

    # --- B. Reliability Metrics ---
    count_inferences = len(df_current)
    p50_latency = float(df_current['latency_ms'].median()) if not df_current['latency_ms'].isnull().all() else 0.0
    p95_latency = float(df_current['latency_ms'].quantile(0.95)) if not df_current['latency_ms'].isnull().all() else 0.0
    
    metrics.append(('count_inferences', count_inferences))
    metrics.append(('p50_latency_ms', p50_latency))
    metrics.append(('p95_latency_ms', p95_latency))

    # --- C. Drift Metrics (PSI) ---
    # Need to expand JSON features into columns
    # Assuming flat JSON like {"age": 25, "income": 50000}
    
    # Skip PSI if sample size too small
    if len(df_baseline) < 200 or len(df_current) < 200:
        print(f"  Skipping Drift: Not enough samples (Baseline: {len(df_baseline)}, Current: {len(df_current)})")
    else:
        # Parsing JSON columns
        # In a real pipeline, we'd do this more efficiently or rely on summary sketches.
        pdf_cur_features = pd.json_normalize(df_current['features_json'])
        pdf_base_features = pd.json_normalize(df_baseline['features_json'])

        # Filter to numeric features only (as per constraints)
        # If DRIFT_Features env var is set, use that whitelist. Else verify numeric.
        drift_feats_config = [f.strip() for f in DRIFT_Features if f.strip()]
        
        feature_candidates = drift_feats_config if drift_feats_config else pdf_cur_features.columns
        
        for feat in feature_candidates:
            if feat not in pdf_cur_features.columns or feat not in pdf_base_features.columns:
                continue
            
            # Ensure numeric
            if not pd.api.types.is_numeric_dtype(pdf_cur_features[feat]) or not pd.api.types.is_numeric_dtype(pdf_base_features[feat]):
                continue
                
            psi_val = compute_psi(pdf_base_features[feat].dropna(), pdf_cur_features[feat].dropna())
            metrics.append((f'psi__{feat}', psi_val))


    # --- D. Performance Metrics (Conditional) ---
    # Join labels
    # We query label_events and join in memory for the MVP script or do separate lookup.
    # Better to join in SQL.
    query_perf = """
        SELECT i.y_pred_num, l.y_true_num, i.pred_type
        FROM inference_events i
        JOIN label_events l ON i.request_id = l.request_id
        WHERE i.model_id = %s AND i.model_version = %s AND i.ts::date = %s
    """
    df_perf = pd.read_sql(query_perf, conn, params=(model_id, model_version, target_day))
    
    if len(df_perf) >= 200:
        # Determine task type from first row (assuming consistent)
        pred_type = df_perf['pred_type'].iloc[0]
        
        if pred_type == 'regression':
            mae = (df_perf['y_true_num'] - df_perf['y_pred_num']).abs().mean()
            metrics.append(('mae', float(mae)))
        elif pred_type == 'classification':
            # F1 score for binary classification (threshold 0.5)
            # Assuming y_pred_num is probability of positive class (1)
            y_pred_binary = (df_perf['y_pred_num'] >= 0.5).astype(int)
            y_true_binary = (df_perf['y_true_num']).astype(int)
            
            tp = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
            fp = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
            fn = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics.append(('f1', f1))
    else:
        print(f"  Skipping Perf: Not enough labels ({len(df_perf)})")

    # --- E. Store Metrics ---
    if metrics:
        with conn.cursor() as cur:
            # Upsert
            upsert_sql = """
                INSERT INTO metrics_daily (day, model_id, model_version, metric_name, metric_value)
                VALUES %s
                ON CONFLICT (day, model_id, model_version, metric_name) 
                DO UPDATE SET metric_value = EXCLUDED.metric_value, created_at = NOW()
            """
            data_tuples = [(target_day, model_id, model_version, name, val) for name, val in metrics]
            execute_values(cur, upsert_sql, data_tuples)
        conn.commit()
        print(f"  Saved {len(metrics)} metrics.")

if __name__ == "__main__":
    import sys
    day_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_job(day_arg)
