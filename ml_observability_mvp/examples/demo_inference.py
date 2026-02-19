import datetime
import random
import numpy as np
from ml_observability_mvp.sdk import MLOpsClient
from ml_observability_mvp.jobs.compute_daily_metrics import run_job

def generate_traffic(client, model_id, date, count=200, drift=False):
    print(f"Generating {count} events for {date} (Drift={drift})...")
    
    for _ in range(count):
        # Base features
        age = int(np.random.normal(35, 10))
        income = float(np.random.normal(50000, 15000))
        credit_score = int(np.random.normal(700, 50))
        
        # Inject Drift if requested
        if drift:
            # Shift age significantly
            age = int(np.random.normal(55, 10))
            # Shift income
            income = float(np.random.normal(75000, 20000))

        features = {
            "age": age,
            "income": income,
            "credit_score": credit_score
        }
        
        # Create Timestamp for that specific date
        # Random hour/minute
        dt = datetime.datetime.combine(date, datetime.datetime.min.time()) + \
             datetime.timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        
        # Log Inference
        client.log_inference(
            model_id=model_id,
            model_version="1.0.0",
            features=features,
            pred_type="classification",
            y_pred_num=random.random(), # Dummy probability
            latency_ms=random.randint(15, 120), # Realistic latency
            timestamp=dt,
            privacy_mode="raw" # Explicitly use raw for MVP demo
        )

def run_demo():
    client = MLOpsClient()
    model_id = "demo_credit_risk"
    
    today = datetime.date.today()
    
    # 1. Generate Baseline (Past 14 days)
    # 200 events per day
    for i in range(14, 0, -1):
        day = today - datetime.timedelta(days=i)
        generate_traffic(client, model_id, day, count=250, drift=False)
        
    # 2. Generate Today (With Drift)
    generate_traffic(client, model_id, today, count=300, drift=True)
    
    print("Traffic generation complete.")
    
    # 3. Running Metrics Job
    print("Triggering Metrics Job...")
    # Need to run job for all days to fill history? 
    # Or just today? The job computes baseline from history on the fly.
    # But we need 'metrics_daily' populated for the 'Latency Trend' chart in CLI.
    # Let's run it for the last 7 days at least to populate the trend graph.
    
    for i in range(7, -1, -1):
        d = today - datetime.timedelta(days=i)
        run_job(d.strftime("%Y-%m-%d"))

    print("\nDemo Complete!")
    print(f"Run report: python -m cli.report --model_id {model_id} --model_version 1.0.0 --days 7")

if __name__ == "__main__":
    run_demo()
