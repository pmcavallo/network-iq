# Network IQ — Responsible AI for Telco Performance

**Goal:** Ship a Verizon‑aligned MVP that turns network telemetry into *faster incident detection (MTTD↓)*, *better customer experience (NPS proxy↑)*, and *leaner cost/GB* — with responsible AI baked in.

## Why this matters (business outcomes)
- **MTTD↓:** Detect congestion/outages earlier via KPI trends & anomalies.
- **NPS proxy↑:** Fewer dropped/slow sessions; clearer impact storytelling.
- **Cost/GB↓:** Smarter capacity offload & parameter tuning guidance.

## What’s in this repo (MVP scope)
- `/docs` — PRD v1, KPI dictionary, model card skeleton, course notes
- `/src` — `ingest/` PySpark CSV→Parquet pipeline, utilities
- `/data` — `raw/` sample dataset, `curated/parquet/` outputs (git‑ignored)
- `/notebooks` — EDA and quick experiments
- `streamlit_app.py` — lightweight EDA dashboard (stub)

## Quickstart
1. Create and activate a virtual environment (optional)  
   **One‑liner (Windows/PowerShell):** `python -m venv .venv && .\.venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt`  
   **One‑liner (macOS/Linux):** `python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt`

2. Convert the sample CSV to Parquet with PySpark  
   **One‑liner:** `python src/ingest/spark_ingest.py --input data/raw/sample_cells.csv --output data/curated/parquet`

3. (Optional) Run the EDA stub  
   **One‑liner:** `streamlit run streamlit_app.py`

## Next 2 Sprints (at a glance)
- **Sprint 1:** PRD v1, KPI glossary, EDA notebook, CSV→Parquet pipeline
- **Sprint 2:** Baselines (XGBoost congestion, ARIMA/Prophet trend), SHAP + PSI

> This repo is structured for clean, interview‑ready artifacts (PRD, KPIs, model card) plus runnable code.
>
> ## Quickstart (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
