# Product Requirements Document (PRD) — Network IQ (MVP)

## 1) Problem & Users
- **Problem:** Operations teams discover cell‑level performance issues too late, hurting CX and inflating costs.
- **Primary users:** NOC engineers, data scientists, reliability leads.
- **Secondary users:** Product/Customer Care needing plain‑English insights.

## 2) Objectives & Key Results (OKRs)
- **O1: Reduce mean time to detect (MTTD)** by **≥30%** within 90 days of pilot.
- **O2: Improve NPS proxy** (drop‑rate + throughput composite) by **≥5%** in pilot zones.
- **O3: Decrease cost/GB** via targeted offload decisions in top 10 hotspots.

## 3) KPIs & Definitions (see KPI Dictionary)
- Latency, Throughput, Jitter, Packet Loss/Drop Rate, **RSRP/RSRQ/SINR**.

## 4) MVP Scope
- PySpark **CSV→Parquet** pipeline, EDA notebook & dashboard stub.
- Baselines: **XGBoost (congestion)** and **ARIMA/Prophet (KPI trend)**.
- **Anomaly signals** (residuals/spectral) with alert thresholds.
- **Responsible AI:** model card v1, bias & drift checks; privacy notes.

## 5) Data
- **Synthetic/public**: cell_id, timestamp, geo (approx), RSRP/RSRQ/SINR, throughput, latency, jitter, drop_rate.
- Partitions: date, cell_id.

## 6) Guardrails & Constraints
- Clear **privacy stance** for any real data; minimize PII (none in MVP).
- **Cost controls:** batch scoring first; logging & sampling for LLMs.
- **Reproducibility:** MLflow runs; pinned requirements.

## 7) Milestones
- **Week 1–2:** Repo, KPI dictionary, EDA, ingest pipeline.
- **Week 3–4:** Baselines + anomaly signals.
- **Week 5–6:** Prescriptive heuristics; graph features (neighbor load).
- **Week 7–8:** Model card v1; drift/bias & cost notes; demo video.

## 8) Risks & Mitigations
- **Data mismatch:** start synthetic; add adapters for real schemas.
- **Overfitting to proxies:** use holdout cells/time; stability (PSI/KS).
- **LLM hallucinations:** retrieval‑augmented explainer + eval set.