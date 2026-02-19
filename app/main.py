
import os
import time
import datetime
import random
import logging
import pytz
import math

from app.driftwatch_client import DriftWatchClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock objects/constants to simulate engine environment
TZ_NY = pytz.timezone("America/New_York")
class MockAPI:
    def __init__(self):
        self.base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

def now_ny():
    return datetime.datetime.now(TZ_NY)

def predict_block_signal(api, sym):
    # Simulating signal generation
    # MUST return 'feat_last' as per requirements
    
    # Generate random features
    feat_last = {
        "sigma20_pct": random.uniform(0.1, 0.5),
        "price_change_pct": random.uniform(-0.02, 0.02),
        "ret5": random.uniform(-0.01, 0.01),
        "vol20": random.uniform(0.1, 0.4),
        "MACDh_12_26_9": random.uniform(-0.5, 0.5),
        "BBM_20_2.0": random.uniform(0.9, 1.1),
        "price_change_pct_lag2": random.uniform(-0.02, 0.02),
        "price_change_pct_lag3": random.uniform(-0.02, 0.02),
        "MACDh_12_26_9_lag2": random.uniform(-0.5, 0.5),
        "MACDh_12_26_9_lag3": random.uniform(-0.5, 0.5),
        "BBM_20_2.0_lag2": random.uniform(0.9, 1.1),
        "BBM_20_2.0_lag3": random.uniform(0.9, 1.1),
    }

    return {
        "pred_pct": random.uniform(-0.01, 0.01),
        "p_up": random.random(),
        "vol_est": feat_last["sigma20_pct"], # Legacy field
        "per_model": {"xgb": 0.5, "elastic": 0.5},
        "feat_last": feat_last # NEW: Required for DriftWatch
    }

def target_position_from_pred(pred, band, ema, sym, state):
    # Mock target calc
    return round(pred * 10, 2) # Arbitrary logic

def submit_target(api, sym, target, eq, px, ledger=None):
    # Mock submission
    pass

def run_session(api):
    # 1. Init DriftWatch Client
    dw = DriftWatchClient()
    
    symbols = ["AAPL", "SPY"] # Tiny scope for mock
    
    try:
        # Simulate a few blocks
        for block_i in range(2): 
            block_start = now_ny()
            logger.info(f"Starting Block {block_i} at {block_start}")
            
            for sym in symbols:
                # 2. Start Timer
                t0 = time.perf_counter()
                
                # 3. Predict (returns feat_last)
                sig = predict_block_signal(api, sym)
                
                # Mock state variables
                px = 150.0 + random.uniform(-1, 1)
                
                # 4. Target Calc
                target_frac = target_position_from_pred(sig["pred_pct"], None, None, sym, None)
                
                # 5. Stop Timer
                ms = (time.perf_counter() - t0) * 1000.0
                if ms <= 0:
                    latency_ms = 0
                else:
                    latency_ms = max(1, int(round(ms)))
                
                # 6. Log Inference
                # Features Mapping
                feats = sig["feat_last"]
                features_json = {
                    "pred_pct": sig["pred_pct"],
                    "p_up": sig["p_up"],
                    "sigma20_pct": feats["sigma20_pct"],
                    "price_change_pct": feats["price_change_pct"],
                    "ret5": feats["ret5"],
                    "vol20": feats["vol20"],
                    "MACDh_12_26_9": feats["MACDh_12_26_9"],
                    "BBM_20_2.0": feats["BBM_20_2.0"],
                    "price_change_pct_lag2": feats.get("price_change_pct_lag2"),
                    "price_change_pct_lag3": feats.get("price_change_pct_lag3"),
                    "MACDh_12_26_9_lag2": feats.get("MACDh_12_26_9_lag2"),
                    "MACDh_12_26_9_lag3": feats.get("MACDh_12_26_9_lag3"),
                    "BBM_20_2.0_lag2": feats.get("BBM_20_2.0_lag2"),
                    "BBM_20_2.0_lag3": feats.get("BBM_20_2.0_lag3"),
                    "px": px,
                    "target_frac": target_frac
                }
                
                # Segment Mapping
                base_url = api.base_url
                env = "paper" if "paper" in base_url.lower() else "live"
                
                segment_json = {
                    "sym": sym,
                    "timeframe": "1h",
                    "env": env,
                    "session": "regular",
                    "block_id": block_start.strftime("%Y-%m-%dT%H:%M:%S%z")
                }
                
                dw.log_inference(
                    model_id="trading_ensemble_ret_1h",
                    model_version="mock-v1",
                    ts=now_ny(),
                    pred_type="regression",
                    y_pred_num=sig["pred_pct"],
                    y_pred_text=None,
                    latency_ms=latency_ms,
                    features_json=features_json,
                    segment_json=segment_json
                )
                
                submit_target(api, sym, target_frac, 100000, px)
                
            # 7. Block Flush
            dw.flush()
            time.sleep(1) # Mock block duration
            
    finally:
        # 8. Cleanup
        dw.close()

if __name__ == "__main__":
    mock_api = MockAPI()
    run_session(mock_api)
