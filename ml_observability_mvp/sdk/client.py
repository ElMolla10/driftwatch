import os
import json
import datetime
import uuid
from typing import Dict, Optional, Any, Union
import psycopg2
from psycopg2.extras import Json

class MLOpsClient:
    def __init__(self, database_url: Optional[str] = None):
        self.dsn = database_url or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise ValueError("DATABASE_URL must be set in environment or passed to constructor")
        
    def _get_conn(self):
        return psycopg2.connect(self.dsn)

    def log_inference(
        self,
        model_id: str,
        model_version: str,
        features: Dict[str, Any],
        pred_type: str,
        y_pred_num: Optional[float] = None,
        y_pred_text: Optional[str] = None,
        latency_ms: Optional[int] = 0,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        segment: Optional[Dict[str, Any]] = None,
        privacy_mode: str = "summary"
    ) -> str:
        """
        Log an inference event.
        
        Args:
            privacy_mode: 'summary' or 'raw'. 
                          For this MVP, we store the features dict as JSONB regardless,
                          but in a real system 'summary' would sketch it first.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
            
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
        
        # In a real impl, if privacy_mode == 'summary', we would bin/hash features here.
        # For the MVP, we pass features directly to features_json.
        features_json = features 

        sql = """
        INSERT INTO inference_events 
        (ts, model_id, model_version, request_id, pred_type, latency_ms, features_json, y_pred_num, y_pred_text, segment_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    ts,
                    model_id,
                    model_version,
                    request_id,
                    pred_type,
                    latency_ms,
                    Json(features_json),
                    y_pred_num,
                    y_pred_text,
                    Json(segment) if segment else None
                ))
            conn.commit()
            
        return request_id

    def log_label(
        self,
        model_id: str,
        request_id: str,
        y_true_num: Optional[float] = None,
        y_true_text: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None
    ):
        """
        Log a delayed label.
        """
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc)
        
        sql = """
        INSERT INTO label_events
        (ts_label, model_id, request_id, y_true_num, y_true_text)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    ts,
                    model_id,
                    request_id,
                    y_true_num,
                    y_true_text
                ))
            conn.commit()

    def check_schema(self) -> bool:
        """
        Validates that the database schema exists.
        Returns True if 'inference_events' table is found.
        """
        sql = "SELECT to_regclass('public.inference_events')"
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                result = cur.fetchone()
                return result and result[0] == 'inference_events'

    def get_inference_count(self) -> int:
        """
        Returns the total number of inference events logged.
        """
        sql = "SELECT count(*) FROM inference_events"
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone()[0]


