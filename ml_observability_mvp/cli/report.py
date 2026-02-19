import os
import click
import psycopg2
import pandas as pd
from tabulate import tabulate
import datetime

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL env var not set")
    return psycopg2.connect(DATABASE_URL)

@click.command()
@click.option('--model_id', required=True, help='Model ID')
@click.option('--model_version', required=True, help='Model Version')
@click.option('--days', default=7, help='Number of days to look back')
def report(model_id, model_version, days):
    """
    Generate ML Health Report for a model.
    """
    conn = get_db_connection()
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)
    
    click.echo(f"\n=== ML Health Report: {model_id} v{model_version} ===")
    click.echo(f"Period: {start_date} to {end_date}\n")

    # Fetch Metrics
    query = """
        SELECT day, metric_name, metric_value 
        FROM metrics_daily 
        WHERE model_id = %s AND model_version = %s 
          AND day >= %s AND day <= %s
        ORDER BY day ASC
    """
    df = pd.read_sql(query, conn, params=(model_id, model_version, start_date, end_date))
    conn.close()
    
    if len(df) == 0:
        click.echo("No metrics found for this period.")
        return

    # Pivot for easy access
    # metric_name becomes columns
    df_pivot = df.pivot(index='day', columns='metric_name', values='metric_value')
    
    # --- 1. Latency Trend ---
    click.echo("--- Reliability (Latency) ---")
    latency_cols = ['p50_latency_ms', 'p95_latency_ms', 'count_inferences']
    cols_to_show = [c for c in latency_cols if c in df_pivot.columns]
    if cols_to_show:
        print(tabulate(df_pivot[cols_to_show].tail(days), headers='keys', tablefmt='simple'))
    else:
        click.echo("No latency metrics available.")
    click.echo("")

    # --- 2. Drift (PSI) ---
    click.echo("--- Top Drift (PSI) ---")
    # Filter columns starting with psi__
    psi_cols = [c for c in df_pivot.columns if c.startswith('psi__')]
    
    if psi_cols:
        # Get latest day values
        latest_day = df_pivot.index.max()
        latest_psi = df_pivot.loc[latest_day, psi_cols].sort_values(ascending=False).head(5)
        
        drift_data = []
        for col, val in latest_psi.items():
            feature = col.replace("psi__", "")
            severity = "OK"
            if val > 0.3: severity = "CRITICAL"
            elif val > 0.2: severity = "WARN"
            drift_data.append([feature, f"{val:.4f}", severity])
            
        print(tabulate(drift_data, headers=["Feature", "PSI", "Severity"], tablefmt='simple'))
    else:
        click.echo("No PSI metrics available.")
    click.echo("")
    
    # --- 3. Optional Performance ---
    click.echo("--- Performance ---")
    perf_cols = [c for c in ['mae', 'f1'] if c in df_pivot.columns]
    if perf_cols:
        print(tabulate(df_pivot[perf_cols].tail(days), headers='keys', tablefmt='simple'))
    else:
        click.echo("No performance metrics available (require labels).")
    click.echo("")

    # --- 4. Health Summary ---
    click.echo("--- Health Summary (Latest Day) ---")
    latest_day = df_pivot.index.max()
    row = df_pivot.loc[latest_day]
    
    issues = []
    
    # PSI Check
    for col in psi_cols:
        val = row.get(col, 0)
        if val > 0.3:
            issues.append(f"CRITICAL: {col} = {val:.3f} (> 0.3)")
        elif val > 0.2:
            issues.append(f"WARN: {col} = {val:.3f} (> 0.2)")
            
    # Latency Check
    # Rule: p95 > 2x median(p95 last 7 days) -> CRITICAL
    # Rule: p95 > 1.5x median(p95 last 7 days) -> WARN
    if 'p95_latency_ms' in df_pivot.columns:
        current_p95 = row['p95_latency_ms']
        # Calculate median of p95 over last 7 days EXCLUDING today (if possible, or typically including)
        # Prompt says "median(p95 over last 7 days)". Usually means baseline history. 
        # Let's take the rolling window of the data we pulled.
        history_p95 = df_pivot['p95_latency_ms'].median() # Simple median of current view
        
        if current_p95 > 2 * history_p95 and history_p95 > 0:
             issues.append(f"CRITICAL: p95 latency {current_p95} is > 2x baseline {history_p95}")
        elif current_p95 > 1.5 * history_p95 and history_p95 > 0:
             issues.append(f"WARN: p95 latency {current_p95} is > 1.5x baseline {history_p95}")

    if not issues:
        click.echo("STATUS: OK")
    else:
        # Determine max severity
        status = "WARN"
        if any("CRITICAL" in i for i in issues):
            status = "CRITICAL"
        
        click.echo(f"STATUS: {status}")
        for i in issues:
            click.echo(f"  - {i}")

if __name__ == '__main__':
    report()
