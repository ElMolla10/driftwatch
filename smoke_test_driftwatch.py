
import os
import time
import logging
import psycopg2
from datetime import datetime
from app.driftwatch_client import DriftWatchClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_smoke_test():
    db_url = os.getenv("DRIFTWATCH_DATABASE_URL")
    if not db_url:
        logger.error("DRIFTWATCH_DATABASE_URL not set. Skipping smoke test.")
        return

    logger.info("Initializing DriftWatch Client for Smoke Test...")
    dw = DriftWatchClient()
    
    if not dw.enabled:
        logger.error("DriftWatch disabled in client init. Check env vars.")
        return

    # 1. Log 5 dummy events
    logger.info("Logging 5 smoke test events...")
    for i in range(5):
        dw.log_inference(
            model_id="smoke_test_model",
            model_version="v0.0.1",
            ts=datetime.now(),
            pred_type="regression",
            y_pred_num=0.5,
            y_pred_text=None,
            latency_ms=10 + i,
            features_json={
                "sigma20_pct": 0.42,
                "price_change_pct": 0.01,
                "null_check": None,
                "nan_check": float('nan') # Should become None
            },
            segment_json={
                "env": "smoke_test",
                "sym": "TEST"
            },
            request_id=f"smoke-{i}"
        )

    # 2. Flush
    logger.info("Flushing events...")
    dw.flush()
    time.sleep(1) # Allow DB commit time

    # 3. Verify in DB
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Count check
                cur.execute("SELECT count(*) FROM inference_events WHERE model_id = 'smoke_test_model'")
                count = cur.fetchone()[0]
                logger.info(f"DB Row Count for 'smoke_test_model': {count}")
                
                if count < 5:
                    raise Exception(f"Expected at least 5 rows, found {count}")

                # JSON Value check (Critical anti-NaN check)
                cur.execute("SELECT features_json FROM inference_events WHERE request_id = 'smoke-0'")
                feats = cur.fetchone()[0]
                sigma = feats.get("sigma20_pct")
                nan_check = feats.get("nan_check")

                logger.info(f"Row 0 features: {feats}")
                
                if not isinstance(sigma, (int, float)):
                    raise Exception(f"sigma20_pct is not numeric: {type(sigma)} ({sigma})")
                
                if nan_check is not None:
                    raise Exception(f"nan_check should be null, got: {nan_check}")

        logger.info("✅ SUCCESS: Smoke test passed. Data inserted, counted, and types verified.")

    except Exception as e:
        logger.error(f"❌ FAILED: Verification failed: {e}")
        raise

if __name__ == "__main__":
    run_smoke_test()
