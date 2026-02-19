# ML Observability MVP

A minimal, global ML observability system using Python and PostgreSQL.

## Prerequisites

- Docker
- Python 3.9+

## Runbook

### 1. Start Postgres
```bash
docker run --name pg-obs -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres
```

### 2. Apply Schema
```bash
export DATABASE_URL=postgresql://postgres:password@localhost:5432/postgres
# Wait a few seconds for PG to start if you just ran docker run
psql $DATABASE_URL -f schema.sql
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Run Demo
Generates 14 days of baseline data + 1 day of traffic with injected drift.
```bash
# This will also run the metrics job automatically in the script
python examples/demo_inference.py
```

### 5. Generate Report
```bash
python -m cli.report --days 7
```

## Architecture
- **SDK**: `sdk/client.py` - Ingests inference and label events.
- **Database**: Postgres (3 tables: `inference_events`, `label_events`, `metrics_daily`).
- **Jobs**: `jobs/compute_daily_metrics.py` - Computes PSI, Reliability, and Performance.
- **CLI**: `cli/report.py` - Displays health status.
