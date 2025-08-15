# Network KPI Dictionary (MVP)

| KPI | Unit | Typical Range | Definition / Notes | Why it matters |
|-----|------|----------------|--------------------|----------------|
| Latency | ms | 10–200 | One‑way or round‑trip delay; here use round‑trip estimate from ping/HTTP timing. | High latency degrades QoE (lag). |
| Throughput | Mbps | 0–1000+ | Effective data rate per user or cell segment. | Core CX measure; ties to cost/GB and revenue. |
| Jitter | ms | 0–50 | Variation in latency; std dev or P95–P50 spread per window. | VoIP/streaming sensitive. |
| Packet Loss / Drop Rate | % | 0–5%+ | Share of failed packets/sessions. | Directly hurts NPS proxy. |
| RSRP | dBm | −120 to −70 | 4G/5G Reference Signal Received Power (signal strength). | Coverage quality; correlates with throughput. |
| RSRQ | dB | −20 to −3 | 4G/5G Reference Signal Received Quality. | Interference/load indicator. |
| SINR | dB | −10 to 30 | Signal‑to‑Interference‑plus‑Noise Ratio. | Capacity and reliability predictor. |

## Computation Notes
- Use **hourly** and **5‑min** windows; aggregate by `cell_id` and geo tile.
- Compute **P50/P95** for latency & jitter; **mean/median** for throughput.
- Derive **NPS proxy** = w1·(Throughput z) − w2·(Latency z) − w3·(DropRate z). Tune weights later.
- Keep raw & engineered columns (suffix `_p50`, `_p95`, `_mean`).

## Data Quality Checks
- Drop impossible values (e.g., latency <= 0ms, SINR > 40dB).
- Winsorize extreme tails for visualization (document!).