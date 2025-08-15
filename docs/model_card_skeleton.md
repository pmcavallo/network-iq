# Model Card — Network IQ (Skeleton)

**Model purpose:** Predict/flag cell‑level congestion & anomalies.  
**Intended users:** NOC engineers, SREs, DS partners.  
**Training data:** Synthetic/public telemetry; no PII.  
**Target(s):** Congestion probability; anomaly score.

## Performance (to fill later)
- Metrics: AUC/PR, F1@k, lift, stability (PSI), calibration.
- Segments: urban/rural, band/tech mix, time of day.

## Risks & Limitations
- Synthetic→real gap; proxy choices.  
- Data drift; seasonal effects.

## Responsible AI
- Bias checks (urban vs rural), privacy notes, monitoring plan.