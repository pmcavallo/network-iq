import os, re
import pandas as pd
import streamlit as st
st.session_state.setdefault("min_prob", 0.0);
st.session_state.setdefault("map_scale", 1.0);
import altair as alt
import numpy as np

st.title("Network IQ: Incident Risk Monitor")

# --- Brief intro right under the title ---
st.markdown(
    "Predict next-hour incident risk from recent radio KPIs, explain top drivers, "
    "and visualize at-risk cells on a map. Use the filters above to change scope; "
    "KPIs and predictions update accordingly."
)

# Compact data dictionary (toggle)
with st.expander("Data dictionary (short)"):
    st.markdown("""
- **timestamp**: measurement time (UTC).
- **cell_id**: unique cell identifier.
- **lat, lon**: cell coordinates (deg).
- **rsrp_dbm, rsrq_db, sinr_db**: radio link quality metrics.
- **throughput_mbps**: downlink throughput (Mbps).
- **latency_ms, jitter_ms**: latency statistics (ms).
- **drop_rate**: packet drop rate (%).
- **tech, band**: RAT and spectrum band (e.g., 5G B66).
- **pred_prob**: model-estimated probability of an incident in the next hour.
""")


parquet_dir = "data/curated/parquet"

# ===== Data Contract: summarize and render =====

import numpy as np  # ensure this is near your other imports

def _data_contract_summary(df: pd.DataFrame) -> dict:
    """Return a JSON-serializable summary of schema + quick quality checks."""
    cols = {c.lower(): c for c in df.columns}
    pick = lambda *opts: next((cols[o] for o in opts if o in cols), None)

    cell = pick("cell_id", "cell", "cellid", "id")
    ts   = pick("timestamp", "ts", "datetime", "date_time", "time")
    thr  = pick("throughput_mbps", "throughput")
    latm = pick("latency_ms", "latency")
    drp  = pick("drop_rate", "drop_percent")
    lat  = pick("lat", "latitude")
    lon  = pick("lon", "lng", "longitude")
    prob = pick("pred_prob", "pred_proba", "proba", "prob", "score", "y_hat",
                "y_pred_proba", "predicted_probability", "probability")

    out = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "required_present": {
            "cell_id": bool(cell),
            "timestamp": bool(ts),
            "throughput_mbps": bool(thr),
            "latency_ms": bool(latm),
            "drop_rate": bool(drp),
        },
        "null_counts": {},
        "lat_lon_invalid": None,
        "pred_prob_present": bool(prob),
        "bad_pred_prob": None,
        "pred_prob_min": None,
        "pred_prob_max": None,
    }

    if cell: out["null_counts"]["cell_id"] = int(df[cell].isna().sum())
    if ts:   out["null_counts"]["timestamp"] = int(df[ts].isna().sum())
    if thr:  out["null_counts"]["throughput_mbps"] = int(df[thr].isna().sum())
    if latm: out["null_counts"]["latency_ms"] = int(df[latm].isna().sum())
    if drp:  out["null_counts"]["drop_rate"] = int(df[drp].isna().sum())

    if lat and lon:
        la = pd.to_numeric(df[lat], errors="coerce")
        lo = pd.to_numeric(df[lon], errors="coerce")
        out["lat_lon_invalid"] = int(((la < -90) | (la > 90) | (lo < -180) | (lo > 180)).sum())

    if prob:
        p = pd.to_numeric(df[prob], errors="coerce")
        p_valid = p.dropna()
        out.update({
            "bad_pred_prob": int(p.isna().sum() + ((p < 0) | (p > 1)).sum()),
            "pred_prob_min": float(p_valid.min()) if not p_valid.empty else None,
            "pred_prob_max": float(p_valid.max()) if not p_valid.empty else None,
        })
    return out

def _render_data_contract(df: pd.DataFrame, title: str = "Data Quality Check"):
    """Streamlit widget: concise UX + optional advanced JSON."""
    s = _data_contract_summary(df)

    # Pass criteria
    req_missing = [k for k, v in s["required_present"].items() if not v]
    null_total = sum(s["null_counts"].values()) if s["null_counts"] else 0
    geo_bad = s["lat_lon_invalid"] or 0
    pred_bad = (s["bad_pred_prob"] or 0) if s["pred_prob_present"] else 0

    passed = (not req_missing) and (null_total == 0) and (geo_bad == 0) and (
        (not s["pred_prob_present"]) or (pred_bad == 0)
    )

    # Header + status
    st.subheader(title)
    st.markdown(
        ("✅ **Contract passed**" if passed else "⚠️ **Contract has issues**")
        + f" · rows: **{s['rows']}**"
    )

    # Compact KPI row (friendlier than raw JSON)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing required cols", len(req_missing))
    c2.metric("Nulls (key cols)", null_total)
    c3.metric("Invalid lat/lon", geo_bad)

    # Pred prob metric + caption so '0' isn't confusing
    c4.metric("Pred prob issues", pred_bad)
    if not s["pred_prob_present"]:
        c4.caption("not present in this dataset")
    else:
        # show range only if we have valid numbers
        if s["pred_prob_min"] is not None and s["pred_prob_max"] is not None:
            c4.caption(f"range {s['pred_prob_min']:.2f} – {s['pred_prob_max']:.2f}")

    # Human-readable issues (only when needed)
    if not passed:
        issues = []
        if req_missing:
            issues.append("Missing required columns: **" + ", ".join(req_missing) + "**")
        for k, v in s["null_counts"].items():
            if v:
                issues.append(f"**{k}** has **{v}** nulls")
        if geo_bad:
            issues.append(f"Invalid lat/lon rows: **{geo_bad}**")
        if s["pred_prob_present"] and pred_bad:
            issues.append(f"`pred_prob` invalid entries: **{pred_bad}** (NaN or out of [0,1])")

        st.warning("• " + "\n• ".join(issues))

    # Advanced details for power users
    
    with st.expander("Details"):
    # Left: Required columns as a simple table
        c1, c2 = st.columns(2)

        req_tbl = pd.DataFrame(
            [{"Column": k, "Present": ("✅" if v else "❌")} for k, v in s["required_present"].items()]
        )
        c1.markdown("**Required columns**")
        c1.table(req_tbl)

        # Right: Null counts for key columns
        null_tbl = pd.DataFrame(
            [{"Column": k, "Nulls": v} for k, v in s["null_counts"].items()]
        )
        c2.markdown("**Null counts (key cols)**")
        c2.table(null_tbl)

        st.divider()

        # Compact QA readouts
        d1, d2 = st.columns(2)
        d1.metric("Invalid lat/lon rows", s["lat_lon_invalid"] or 0)

        if s["pred_prob_present"]:
            d2.metric("Pred prob issues", pred_bad)
            if s["pred_prob_min"] is not None and s["pred_prob_max"] is not None:
                d2.caption(f"range {s['pred_prob_min']:.2f} – {s['pred_prob_max']:.2f}")
        else:
            d2.caption("`pred_prob` not present in this dataset")

    # Optional: keep raw JSON for debugging, but hide it behind a second expander
    with st.expander("Developer JSON"):
        st.json(s)

# Build df if it doesn't already exist, then render the data-quality panel once.
from pathlib import Path

if "df" not in locals():
    dataset_obj = locals().get("dataset", None)

    if dataset_obj is not None:
        df = dataset_obj.to_table().to_pandas()
    else:
        # Fallback to local CSVs (prefers the sample if present)
        sample = Path("data/raw/sample_cells.csv")
        csvs = [sample] if sample.exists() else list(Path("data").rglob("*.csv"))
        if not csvs:
            st.error("No dataset or CSV files found. Add data/raw/sample_cells.csv or configure a dataset.")
            st.stop()
        df = pd.concat([pd.read_csv(p, low_memory=False) for p in csvs], ignore_index=True)

_render_data_contract(df, title="Data Quality Check")


# ===== /Data Contract =====

# ---------- Load Spark-partitioned dataset ----------
paths = [os.path.join(root, f)
         for root, _, files in os.walk(parquet_dir)
         for f in files if f.endswith(".parquet")]
if not paths:
    st.info("No Parquet files found under data/curated/parquet.")
    st.stop()

df = None
try:
    import pyarrow.dataset as ds
    dataset = ds.dataset(parquet_dir, format="parquet", partitioning="hive")
    df = dataset.to_table().to_pandas()
except Exception:
    dfs, pat = [], re.compile(r"[\\/]{1}date=([^\\/]+)[\\/].*?[\\/]cell_id=([^\\/]+)[\\/]")
    for p in paths[:24]:
        try:
            pdf = pd.read_parquet(p)
            m = pat.search(p)
            if m:
                pdf["date"] = m.group(1)
                pdf["cell_id"] = m.group(2)
            dfs.append(pdf)
        except Exception:
            pass
    if dfs:
        df = pd.concat(dfs, ignore_index=True)

if df is None or df.empty:
    st.warning("Found Parquet files but couldn't load data. Check pyarrow install.")
    st.stop()

# Ensure timestamp/hour
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
if "hour" not in df.columns and "timestamp" in df.columns:
    df["hour"] = df["timestamp"].dt.hour

# ---------- Filters ----------
cell_opt = "ALL"
if "cell_id" in df.columns:
    cell_opt = st.selectbox("Cell filter", ["ALL"] + sorted(df["cell_id"].dropna().unique().tolist()), key="cell")

date_opt = "ALL"
if "date" in df.columns:
    date_opt = st.selectbox("Date", ["ALL"] + sorted(df["date"].dropna().unique().tolist()), key="date")

view = df.copy()
if cell_opt != "ALL" and "cell_id" in view.columns:
    view = view[view["cell_id"] == cell_opt]
if date_opt != "ALL" and "date" in view.columns:
    view = view[view["date"] == date_opt]

# ---------- KPI cards ----------
# --- KPI helper: NPS-style % (Promoters − Detractors) ---
def _nps_proxy_pct(df):
    # Use the same filtered DataFrame you feed into the KPI/table (e.g., view_df)
    prom = (df["throughput_mbps"] >= 150) & (df["latency_ms"] <= 50) & (df["drop_rate"] <= 1.0)
    det  = (df["throughput_mbps"] < 100) | (df["latency_ms"] > 80) | (df["drop_rate"] > 2.0)
    return 100.0 * (prom.mean() - det.mean())

def _fmt_pct0(x: float) -> str:
    # Avoid showing -0.00; round to whole-number percent
    return f"{0.0 if abs(x) < 0.5 else round(x, 0):+.0f}"

def fmt(v, nd=1):
    try: return f"{float(v):.{nd}f}"
    except: return "—"

c1, c2, c3, c4 = st.columns(4)
# --- KPI cards (with concise captions) ---
# assumes: import pandas as pd, and 'view' is your filtered DataFrame

# Avg Throughput
c1.metric("Avg Throughput (Mbps)", fmt(view.get("throughput_mbps", pd.Series()).mean()))
c1.caption("Average of throughput_mbps over the current filter (higher is better).")

# P95 Latency
c2.metric("P95 Latency (ms)", fmt(view.get("latency_ms", pd.Series()).quantile(0.95)))
c2.caption("95th percentile of latency_ms for the current filter (lower is better).")

# Drop Rate
c3.metric("Drop Rate (%)", fmt(view.get("drop_rate", pd.Series()).mean(), 2))
c3.caption("Average drop_rate percent for the current filter (lower is better).")

# NPS Proxy (%)  =  %Promoters − %Detractors
try:
    # Use the same filtered frame you use for the other KPIs
    tmp = view[["throughput_mbps", "latency_ms", "drop_rate"]].astype(float)

    # Promoters / Detractors thresholds (tweak later if you wish)
    prom = (tmp["throughput_mbps"] >= 150) & (tmp["latency_ms"] <= 50) & (tmp["drop_rate"] <= 1.0)
    det  = (tmp["throughput_mbps"] < 100) | (tmp["latency_ms"] > 80) | (tmp["drop_rate"] > 2.0)

    # %Promoters − %Detractors
    nps_pct = 100.0 * (prom.mean() - det.mean())

    # Avoid showing -0.00 and keep it clean for a KPI
    nps_pct = 0.0 if abs(nps_pct) < 0.5 else round(nps_pct, 0)

    c4.metric("NPS Proxy (%)", f"{nps_pct:+.0f}")
    c4.caption("Promoter: ≥150 Mbps, ≤50 ms, ≤1.0% drop. Detractor: <100 Mbps or >80 ms or >2.0% drop.")
except Exception:
    c4.metric("NPS Proxy (%)", "—")
    c4.caption("Promoter: ≥150 Mbps, ≤50 ms, ≤1.0% drop. Detractor: <100 Mbps or >80 ms or >2.0% drop.")

st.dataframe(view.sort_values("timestamp").head(100) if "timestamp" in view.columns else view.head(100))

# --- one-time app init (per browser session) ---
if "app_init" not in st.session_state:
    st.session_state.update({
        "min_prob": 0.0,   # show all pins by default
        "map_scale": 1.0,  # default pin scale
        "app_init": True
    })


# ---------- Hourly chart with quantile clipping + risk points ----------
if "hour" in view.columns:
    metric = st.selectbox("Metric", ["throughput_mbps","latency_ms","drop_rate"], index=0, key="metric_main")
    hourly = view.groupby("hour")[metric].mean().reset_index()
    hourly["hour"] = hourly["hour"].astype(int)

    # --- Smoothing (moving average) ---
    smooth = st.checkbox("Smooth (moving avg)", value=True, key="smooth_main")
    win = st.slider("Smoothing window (hours)", 1, 5, 3, key="smooth_win_main") if smooth else 1
    y_field = metric
    if smooth and len(hourly) >= win:
        hourly[f"{metric}_sm"] = hourly[metric].rolling(win, center=True, min_periods=1).mean()
        y_field = f"{metric}_sm"

    qmin, qmax = st.slider("Y-axis clip (quantiles)", 0.0, 1.0, (0.05, 0.95), 0.01, key="clip")
    lo = float(hourly[metric].quantile(qmin))
    hi = float(hourly[metric].quantile(qmax))
    if hi <= lo: hi = lo + 1.0

    # Risk rule
    # Choose how strict the risk rule is (quantile)
    risk_q = st.slider("Risk threshold (quantile)", 0.80, 0.99, 0.95, 0.01, key="risk_q_main")

    low_is_bad = metric in ["latency_ms","drop_rate"]
    if low_is_bad:
        thr = float(hourly[metric].quantile(risk_q))  # higher is risky
        hourly["risk"] = (hourly[metric] >= thr)
        risk_label = f">= P{int(risk_q*100)} {metric}"
    else:
        thr = float(hourly[metric].quantile(1.0-risk_q))  # lower is risky
        hourly["risk"] = (hourly[metric] <= thr)
        risk_label = f"<= P{int((1.0-risk_q)*100)} {metric}"

    base = (
        alt.Chart(hourly).mark_line().encode(
            x=alt.X("hour:Q", axis=alt.Axis(title="hour")),
            y=alt.Y(f"{y_field}:Q", scale=alt.Scale(domain=(lo, hi)),
                    title=metric.replace("_"," ").title()),
            tooltip=["hour", alt.Tooltip(f"{metric}:Q", format=".2f")])
        .properties(height=300)
    )

    pts = (
        alt.Chart(hourly).mark_point(filled=True, size=80).encode(
            x="hour:Q", y=f"{metric}:Q",
            color=alt.condition("datum.risk", alt.value("#d62728"), alt.value("#1f77b4"), legend=None),
            tooltip=["hour", alt.Tooltip(f"{metric}:Q", format=".2f"), alt.Tooltip("risk:N", title="Risky?")])
    )
    rule = alt.Chart(pd.DataFrame({"y":[thr]})).mark_rule(strokeDash=[6,4]).encode(y="y:Q")
    st.caption(f"Risk rule: {'higher' if low_is_bad else 'lower'} values are risky ({risk_label}).")
    st.altair_chart(base + pts + rule, use_container_width=True)
    risk_count = int(hourly["risk"].sum())
    total_hours = int(hourly["hour"].nunique())
    if total_hours and total_hours > 0:
        st.info(f"{risk_count}/{total_hours} hours flagged as risky ({risk_count/total_hours:.0%}).")
    else:
        st.info("0/0 hours flagged as risky (no hourly data in view).")



# ---- Cell vs Network comparison (only when a cell is selected) ----
if cell_opt != "ALL" and {"cell_id"}.issubset(df.columns):
    st.subheader("Cell vs Network (same date filter)")

    comp_metric = st.selectbox(
        "Compare metric",
        ["throughput_mbps", "latency_ms", "drop_rate"],
        index=0,
        key="metric_cmp",
    )
    cmp_mode = st.radio(
        "View",
        ["Hour-of-day pattern", "Timeline (hourly)"],
        index=0,
        horizontal=True,
        key="cmp_mode",
    )

    # Network baseline respects the date filter
    net = df.copy()
    if date_opt != "ALL" and "date" in net.columns:
        net = net[net["date"] == date_opt]

    use_hour = (cmp_mode == "Hour-of-day pattern") or ("timestamp" not in view.columns)

    if use_hour:
        # 0..23 shape
        cell_grp = (view.groupby("hour")[comp_metric].mean()
                        .reset_index().rename(columns={comp_metric: "Cell"}))
        net_grp  = (net.groupby("hour")[comp_metric].mean()
                        .reset_index().rename(columns={comp_metric: "Network"}))
        merged = pd.merge(net_grp, cell_grp, on="hour", how="left")
        x_col, x_field, x_title, x_tooltip = "hour", "hour:Q", "hour", alt.Tooltip("hour:Q", title="hour")
    else:
        # Real hourly timeline
        v2, n2 = view.copy(), net.copy()
        v2["ts"] = pd.to_datetime(v2["timestamp"]).dt.floor("H")
        n2["ts"] = pd.to_datetime(n2["timestamp"]).dt.floor("H")
        cell_grp = (v2.groupby("ts")[comp_metric].mean()
                        .reset_index().rename(columns={comp_metric: "Cell"}))
        net_grp  = (n2.groupby("ts")[comp_metric].mean()
                        .reset_index().rename(columns={comp_metric: "Network"}))
        merged = pd.merge(net_grp, cell_grp, on="ts", how="left")
        x_col, x_field, x_title, x_tooltip = "ts", "ts:T", "time", alt.Tooltip("ts:T", title="time")

    # Comparison Y-axis clip (same control style as main chart)
    vals = pd.concat([merged["Network"], merged["Cell"]], ignore_index=True).dropna()
    qmin2, qmax2 = st.slider("Comparison Y-axis clip (quantiles)", 0.0, 1.0, (0.05, 0.95), 0.01, key="cmp_clip")
    lo2 = float(vals.quantile(qmin2)); hi2 = float(vals.quantile(qmax2))
    if hi2 <= lo2: hi2 = lo2 + 1.0

    comp = merged.melt(x_col, var_name="series", value_name="value")
    comp_chart = (
        alt.Chart(comp)
        .mark_line(point=True)
        .encode(
            x=alt.X(x_field, title=x_title),
            y=alt.Y("value:Q", scale=alt.Scale(domain=(lo2, hi2)),
                    title=comp_metric.replace("_"," ").title()),
            color="series:N",
            tooltip=[x_tooltip, "series:N", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(comp_chart, use_container_width=True)

    # ---------- Delta: Cell − Network ----------
    import numpy as np
    merged["Delta"] = merged["Cell"] - merged["Network"]

    # >>> Insight chips (paste lives here) <<<
    d_mean = float(merged["Delta"].mean())
    d_max  = float(merged["Delta"].abs().max())
    d_p95  = float(merged["Delta"].abs().quantile(0.95))
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Δ",   f"{d_mean:+.2f}")
    c2.metric("Max |Δ|", f"{d_max:.2f}")
    c3.metric("P95 |Δ|", f"{d_p95:.2f}")

    # Robust delta scaling with toggle
    delta_mode = st.radio(
        "Delta axis mode",
        ["Auto (tight)", "Symmetric (quantile)"],
        index=1,
        horizontal=True,
        key="delta_mode",
    )
    abs_vals = merged["Delta"].abs().dropna()
    if delta_mode == "Auto (tight)":
        M = float(abs_vals.max()); M = (M if np.isfinite(M) and M > 0 else 1.0) * 1.05
    else:
        dq = st.slider("Delta Y-axis max (abs, quantile)", 0.50, 1.00, 0.90, 0.01, key="delta_clip")
        qv = float(abs_vals.quantile(dq)) if not abs_vals.empty else 0.0
        M = qv if np.isfinite(qv) and qv > 0 else max(float(abs_vals.max()), 1.0)

    zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888", strokeDash=[4, 4]).encode(y="y:Q")

    delta_chart = (
        alt.Chart(merged)
        .mark_bar(size=10)
        .encode(
            x=alt.X(x_field, title=x_title),
            y=alt.Y("Delta:Q", scale=alt.Scale(domain=(-M, M), clamp=True),
                    title=f"Cell − Network ({comp_metric.replace('_',' ')})"),
            color=alt.condition("datum.Delta > 0", alt.value("#1f77b4"), alt.value("#d62728"), legend=None),
            tooltip=[x_tooltip, alt.Tooltip("Delta:Q", format=".2f")],
        )
        .properties(height=180)
    )
    st.altair_chart(delta_chart + zero_rule, use_container_width=True)



# -------- Top Hotspots --------
st.subheader("Top Hotspots")
g = df if date_opt == "ALL" else df[df["date"] == date_opt]

hot_choice = st.selectbox("Rank by", ["Highest P95 Latency","Lowest Avg Throughput","Highest Drop Rate"], key="hot")

if hot_choice == "Highest P95 Latency" and "latency_ms" in g.columns:
    hotspots_agg = (
        g.groupby("cell_id")["latency_ms"].quantile(0.95)
         .sort_values(ascending=False).head(10).reset_index(name="p95_latency_ms")
    )
elif hot_choice == "Lowest Avg Throughput" and "throughput_mbps" in g.columns:
    hotspots_agg = (
        g.groupby("cell_id")["throughput_mbps"].mean()
         .sort_values().head(10).reset_index(name="avg_throughput_mbps")
    )
else:
    hotspots_agg = (
        g.groupby("cell_id")["drop_rate"].mean()
         .sort_values(ascending=False).head(10).reset_index(name="avg_drop_rate")
    )

st.dataframe(hotspots_agg)

st.subheader("Top Anomalies (z-score vs network baseline)")
# Build baseline by hour-of-day for the current date filter
scope = df if date_opt == "ALL" else df[df["date"] == date_opt]
if {"cell_id","hour", "throughput_mbps","latency_ms","drop_rate"}.issubset(scope.columns):
    metric_a = st.selectbox("Anomaly metric", ["throughput_mbps","latency_ms","drop_rate"], index=1, key="anom_metric")
    # --- Quick explanation: anomalies + how it connects to Incident summary ---
    st.markdown(
        """
    **What you're seeing**

    - **Top Anomalies** ranks cells by the **largest absolute z-score** of the selected metric **by hour-of-day** within your current filters.
    - **z-score** = (value − network’s hour-of-day mean) / (hour-of-day std). Larger magnitude ⇒ more unusual.  
    Sign shows direction (positive = above baseline, negative = below).  
    - Rule of thumb: z≈2 is notable; z≥3 is severe.

    **How it connects to Incident summary**
    - The **Incident summary** below uses the **same metric** and flags **cell-hours** where that metric is **≥ the chosen quantile**  
    (e.g., 0.95 = top 5% of hours in the current scope), then lists those hours with the metric value and its z-score for download.
    """
    )
    base = (scope.groupby("hour")[metric_a].agg(["mean","std"]).reset_index()
                  .rename(columns={"mean":"base_mean","std":"base_std"}))
    # Join baseline back and compute z per row, then aggregate per cell
    tmp = scope.merge(base, on="hour", how="left").copy()
    tmp["base_std"] = tmp["base_std"].replace(0, 1e-9)
    tmp["z"] = (tmp[metric_a] - tmp["base_mean"]) / tmp["base_std"]
    # For throughput, negative z is bad; for latency/drop, positive z is bad
    sign = -1.0 if metric_a == "throughput_mbps" else 1.0
    cell_z = (tmp.assign(z_bad=sign*tmp["z"])
                 .groupby("cell_id")["z_bad"]
                 .max()  # worst hour
                 .reset_index()
                 .rename(columns={"z_bad":"max_anom_z"}))
    top_anom = cell_z.sort_values("max_anom_z", ascending=False).head(10)
    st.dataframe(top_anom)
else:
    st.info("Not enough columns to compute anomalies.")

# ---- Incident Summary (export) ----
st.subheader("Incident summary (CSV)")
inc_metric = st.selectbox(
    "Incident metric",
    ["throughput_mbps", "latency_ms", "drop_rate"],
    index=1,
    key="inc_metric",
)
# Short caption for Incident summary behavior
st.caption(
    "Incidents = cell-hours where the selected metric ≥ the chosen quantile on the current scope. "
    "Rows are sorted by anomaly z; use **Download** to export."
)

# Work on current date scope (keeps your Date filter consistent)
scope = df if date_opt == "ALL" else df[df["date"] == date_opt]
scope = scope.copy()
if "timestamp" in scope.columns:
    scope["timestamp"] = pd.to_datetime(scope["timestamp"], errors="coerce")
    scope["ts"] = scope["timestamp"].dt.floor("h")
else:
    st.info("No timestamps found to build an incident summary.")
    scope["ts"] = pd.NaT

# Aggregate to cell-hour
grp_cols = ["cell_id", "ts"]
agg = (
    scope.groupby(grp_cols, dropna=True)[inc_metric]
    .mean()
    .reset_index()
    .rename(columns={inc_metric: "value"})
)
agg = agg.dropna(subset=["value"])

# Risk rule: high is bad for latency/drop; low is bad for throughput
inc_q = st.slider("Risk threshold (quantile)", 0.80, 0.99, 0.95, 0.01, key="inc_q")
if inc_metric in ["latency_ms", "drop_rate"]:
    thr = float(scope[inc_metric].quantile(inc_q))
    agg["risky"] = agg["value"] >= thr
    rule_desc = f"≥ P{int(inc_q*100)} {inc_metric}"
else:
    thr = float(scope[inc_metric].quantile(1.0 - inc_q))
    agg["risky"] = agg["value"] <= thr
    rule_desc = f"≤ P{int((1.0 - inc_q)*100)} {inc_metric}"

# Baseline by hour-of-day → z-score vs network
agg["hour"] = agg["ts"].dt.hour
base = (
    scope.groupby("hour")[inc_metric]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "base_mean", "std": "base_std"})
)
agg = agg.merge(base, on="hour", how="left")
agg["base_std"] = agg["base_std"].replace(0, 1e-9)
agg["z"] = (agg["value"] - agg["base_mean"]) / agg["base_std"]
# For throughput, negative z means worse (slower), flip the sign
if inc_metric == "throughput_mbps":
    agg["z_bad"] = -agg["z"]
else:
    agg["z_bad"] = agg["z"]

# Final incidents table (top 50 by severity)
incidents = (
    agg.loc[agg["risky"], ["cell_id", "ts", "value", "z_bad"]]
    .sort_values("z_bad", ascending=False)
    .head(50)
    .rename(columns={"ts": "hour_ts", "value": inc_metric, "z_bad": "anom_z"})
)

st.caption(f"Risk rule: {rule_desc}. Showing top {len(incidents)} incidents by anomaly z.")
st.dataframe(incidents.head(10))

def _ensure_cols(df, needed=("cell_id",)):
    """Return df where `needed` are real columns (not index) and exist."""
    import pandas as pd
    if df is None or isinstance(df, list):
        return pd.DataFrame(columns=list(needed))
    out = df.copy()

    # Bring needed names out of the index, if present there
    idx_names = []
    if hasattr(out.index, "names"):
        idx_names = [n for n in out.index.names if n is not None]
    elif out.index.name is not None:
        idx_names = [out.index.name]
    if any(n in needed for n in idx_names):
        out = out.reset_index()

    # Keep column names as strings (matches your original)
    out.columns = [str(c) for c in out.columns]

    # NEW: create any missing required columns (idempotent; no duplicates)
    for c in needed:
        if c not in out.columns:
            out[c] = pd.NA

    return out


# ======================= PREDICT NEXT-HOUR INCIDENTS =======================
# Uses the same df/date filters defined above. Train the model first:
#   python scripts/train_next_hour.py --parquet_dir data/curated/parquet --metric latency_ms --q 0.95 --roll 3

import os as _os
import pandas as _pd
import joblib as _joblib

st.markdown("---")
st.subheader("Predict next-hour incidents")

MODEL_PATH = "data/models/next_hour_congestion.joblib"

def _build_latest_features(df, label_metric="latency_ms", roll=3):
    # Ensure hourly timestamp column "ts"
    ts_col = None
    for c in ["timestamp","ts","Datetime","datetime","date_time"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("No usable timestamp column found (need one of timestamp/ts/Datetime).")
    use = df.copy()
    use["ts"] = _pd.to_datetime(use[ts_col], errors="coerce").dt.floor("h")
    if "cell_id" not in use.columns:
        raise ValueError("Missing required 'cell_id' column.")

    use = use.dropna(subset=["ts","cell_id"]).sort_values(["cell_id","ts"])
    use["hour"] = use["ts"].dt.hour

    # Network hour-of-day baseline + z-gap for label_metric
    if label_metric not in use.columns:
        raise ValueError(f"Metric '{label_metric}' not found in data.")
    base = (use.groupby("hour")[label_metric]
              .agg(["mean","std"]).reset_index()
              .rename(columns={"mean":"b_mean","std":"b_std"}))
    use = use.merge(base, on="hour", how="left")
    use["b_std"] = use["b_std"].replace(0, 1e-9)
    use["z_bad"] = (use[label_metric] - use["b_mean"]) / use["b_std"]  # high latency is bad

    # Rolling features (3h) for all expected metrics that exist
    def _add_roll(g, cols, w=3):
        out = g.copy()
        for c in cols:
            r = g[c].rolling(w, min_periods=1)
            out[f"{c}_rmean"] = r.mean()
            out[f"{c}_rstd"]  = r.std().fillna(0.0)
            out[f"{c}_rdiff"] = g[c].diff().fillna(0.0)
        return out

    base_feats = ["throughput_mbps","latency_ms","drop_rate","jitter_ms","rsrp_dbm","rsrq_db","sinr_db"]
    feats = [c for c in base_feats if c in use.columns]
    if not feats:
        raise ValueError("No expected feature columns found for inference.")

    use = use.groupby("cell_id", group_keys=False).apply(_add_roll, feats, roll, include_groups=True)
    use["dow"] = use["ts"].dt.dayofweek
    use["is_weekend"] = (use["dow"]>=5).astype(int)

    last_ts = use["ts"].max()
    latest = use[use["ts"] == last_ts].copy()
    return latest

def _score_latest(latest, bundle):
    num = [c for c in bundle.get("num_cols", []) if c in latest.columns]
    cat = [c for c in bundle.get("cat_cols", []) if c in latest.columns]
    X = latest[num + cat].copy()
    if X.empty:
        raise ValueError("No overlap between model features and current data columns.")
    p = bundle["model"].predict_proba(X)[:,1]
    latest = latest.assign(pred_prob=p)
    return latest

if _os.path.exists(MODEL_PATH):
    try:
        bundle = _joblib.load(MODEL_PATH)
        st.success("Model loaded.")

        # Respect your Date filter
        _inference_df = df if (("date" in df.columns) and date_opt == "ALL") else (df if "date" not in df.columns else df[df["date"] == date_opt])

        latest = _build_latest_features(
            _inference_df,
            label_metric=bundle.get("label_rule", {}).get("metric", "latency_ms"),
            roll=3,
        )

        # Score and normalize so cell_id/ts are columns (not index)
        # Score
        scored = _score_latest(latest, bundle)
        # --- Ensure identifiers survive (some pandas/apply paths drop or hide them) ---
        try:
            import pandas as _pd  # in case we're outside the earlier try
            # Align by index length to be safe
            _s = scored.reset_index(drop=True)
            _l = latest.reset_index(drop=True)
            L = min(len(_s), len(_l))

            # Attach/repair cell_id
            if ("cell_id" not in _s.columns) or _s["cell_id"].isna().all():
                if "cell_id" in _l.columns and L > 0:
                    _s.loc[:L-1, "cell_id"] = _l.loc[:L-1, "cell_id"]

            # Attach/repair ts (or derive from timestamp)
            if ("ts" not in _s.columns) or _s["ts"].isna().all():
                if "ts" in _l.columns and L > 0:
                    _s.loc[:L-1, "ts"] = _l.loc[:L-1, "ts"]
                elif "timestamp" in _l.columns and L > 0:
                    _s.loc[:L-1, "ts"] = _pd.to_datetime(_l.loc[:L-1, "timestamp"], errors="coerce").dt.floor("h")

            scored = _s
        except Exception:
            pass

        # Standardize expected column names
        if "pred_prob" not in scored.columns:
            for _cand in ("pred_proba","proba","prob","score","y_proba","y_pred_proba","p"):
                if _cand in scored.columns:
                    scored = scored.rename(columns={_cand: "pred_prob"})
                    break

        # --- Normalize column names so downstream UI is stable ---
        rename_map = {}
        if "pred_prob" not in scored.columns:
            for cand in ["pred_proba","proba","prob","score","y_proba","y_pred_proba","p"]:
                if cand in scored.columns:
                    rename_map[cand] = "pred_prob"
                    break
        if "cell_id" not in scored.columns:
            for cand in ["cell","cellid","cell_id_x","id"]:
                if cand in scored.columns:
                    rename_map[cand] = "cell_id"
                    break
        if "ts" not in scored.columns:
            for cand in ["timestamp","time","datetime","date_time","ts_x"]:
                if cand in scored.columns:
                    rename_map[cand] = "ts"
                    break
        if rename_map:
            scored = scored.rename(columns=rename_map)

        # Ensure required columns exist (idempotent; no duplicate insert)
        scored = _ensure_cols(scored, needed=("cell_id","ts","pred_prob"))
        # Sort only if column is present
        if "pred_prob" in scored.columns:
            scored = scored.sort_values("pred_prob", ascending=False)


        # --- Controls: Top-N and probability threshold ---
        topN = st.slider("Top N to display", 5, 50, 20)

        # Default min prob to 0.00 on first run; user can change it afterward
        st.session_state.setdefault("min_prob", 0.0)
        c1, c2 = st.columns([1, 0.22])
        with c1:
            min_prob = st.slider("Minimum probability (optional)", 0.0, 1.0, step=0.01, key="min_prob")
        with c2:
            if st.button("Reset to 0", key="reset_min_prob", use_container_width=True):
                st.session_state["min_prob"] = 0.0
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

        # Filter rows for display (single pass; no duplicate globals assignment)
        min_prob = float(st.session_state.get("min_prob", 0.0))

        # Ensure required columns exist once (safe = no duplicate insert errors)
        req = ("cell_id", "ts", "pred_prob")
        _scored_norm = _ensure_cols(scored, needed=req)

        # Optional columns if present
        opt = [c for c in ("throughput_mbps", "latency_ms", "drop_rate") if c in _scored_norm.columns]
        cols = list(req) + opt

        mask = _scored_norm["pred_prob"].fillna(0).ge(min_prob)
        view_rows = _scored_norm.loc[mask, cols].head(topN).copy()

        # Make available to other panels just once
        globals()["view_rows"] = view_rows


        cols_show = ["cell_id","ts","pred_prob"]
        for c in ["throughput_mbps","latency_ms","drop_rate"]:
            if c in view_rows.columns:
                cols_show.append(c)

        if {"cell_id","ts","pred_prob"}.issubset(view_rows.columns) and not view_rows.empty:
            st.dataframe(view_rows[cols_show])
            st.caption("Top predicted next-hour risks (per cell) for the latest hour in scope.")
            st.download_button(
                "Download predictions (CSV)",
                data=view_rows[["cell_id","ts","pred_prob"]].to_csv(index=False),
                file_name="pred_next_hour.csv",
                mime="text/csv",
            )
            # --- How to read this section (compact notes) ---
            with st.expander("Notes (how to read predictions)"):
                st.markdown("""
            - **Top N** shows the highest-risk **cells** for the **latest hour in scope** (sorted by `pred_prob`).
            - **Minimum probability** only **filters the table**; it does **not** change the model.
            - **pred_prob** = model-estimated chance that the cell will have an incident **in the next hour**.
            - **AUC** reflects ranking quality overall; **AP** (average precision) is more informative on **imbalanced** data.
            - Use **Download predictions (CSV)** for hand-offs or audit.
            """)
        else:
            st.info("No prediction rows available yet for this scope.")

                # --- Model metadata (if available) ---
        meta = {
            "trained_at": bundle.get("trained_at"),
            "rule_metric": bundle.get("label_rule", {}).get("metric"),
            "rule_q": bundle.get("label_rule", {}).get("q"),
        }
        m = bundle.get("metrics")
        if m:
            st.caption(f"Model AUC={m.get('auc'):.3f}  AP={m.get('ap'):.3f}  positives={m.get('positives')}/{m.get('n')}  (trained {meta['trained_at']})")
        else:
            st.caption(f"Model trained {meta['trained_at']}; rule={meta['rule_metric']} @ q={meta['rule_q']}.")

        # --- Feature signals (logistic coefficients) ---
        try:
            import numpy as _np, pandas as _pd
            _pipe = bundle["model"]
            _clf = _pipe.named_steps["clf"]
            _pre = _pipe.named_steps["pre"]
            names = _pre.get_feature_names_out()
            coef = _clf.coef_.ravel()
            imp = _pd.DataFrame({"feature": names, "coef": coef, "abs": _np.abs(coef)}).sort_values("abs", ascending=False).head(15)
            st.markdown("**Top directional features** (log-odds coefficient):")
            st.caption(
                "Positive coefficient → higher incident risk; negative → protective. Magnitude shows strength. "
                "Names: `*_mean` rolling mean, `*_rstd` rolling volatility, `*_rdiff` last-hour change, "
                "`num_*` numeric (standardized), `cat_*` one-hot categories."
            )
            st.dataframe(imp[["feature","coef"]])
        except Exception as e:
            st.caption(f"Feature view unavailable: {e}")

        # Full scored preview (first 20)
        cols_show2 = ["cell_id","ts","pred_prob"]
        for c in ["throughput_mbps","latency_ms","drop_rate"]:
            if c in scored.columns:
                cols_show2.append(c)

        if {"cell_id","ts","pred_prob"}.issubset(scored.columns) and not scored.empty:
            st.dataframe(scored[cols_show2].head(20))
        else:
            st.caption("Scored preview unavailable (missing cell_id/ts).")

        st.caption(f"Loaded rule: {bundle.get('label_rule')}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.info("No model found yet. Train one via: python scripts/train_next_hour.py --parquet_dir data/curated/parquet")

# ==========================================================================
# ==========================================================================

# Export
st.download_button(
    "Download incident summary (CSV)",
    data=incidents.to_csv(index=False),
    file_name="incident_summary.csv",
    mime="text/csv",
    key="incident_csv",
)

# ---------- Single download button ----------
st.download_button("Download current view (CSV)", data=view.to_csv(index=False),
                   file_name="eda_view.csv", mime="text/csv", key="download_view")

# === Map — Top-K predicted cells (append-only, deduped, suffix-proof, normalized sizes) ===
# === Map — Top-K predicted cells (robust, sized pins, no duplicate titles) ===

    
def _render_topk_map():
    import os, numpy as _np, pandas as _pd, streamlit as st

    st.subheader("Predicted Risk Map")

    # 1) Choose predictions robustly
    if "view_rows" in globals() and hasattr(view_rows, "dropna"):
        _pred = _ensure_cols(view_rows, needed=("cell_id", "pred_prob"))
    elif "scored" in globals() and hasattr(scored, "dropna"):
        _topN = int(globals().get("topN", 20))
        _pred = (
            _ensure_cols(scored, needed=("cell_id", "pred_prob"))
            .sort_values("pred_prob", ascending=False)
            .head(_topN)
        )
    else:
        st.caption("Map highlight unavailable: no predictions yet.")
        return

    if not {"cell_id", "pred_prob"}.issubset(_pred.columns) or _pred.empty:
        st.caption("Map highlight unavailable: predictions missing cell_id/pred_prob.")
        return

    _pred = _pred[["cell_id", "pred_prob"]].dropna()

    # 2) Load cell coordinates
    _cells_path = "data/raw/sample_cells.csv"
    if not os.path.exists(_cells_path):
        st.caption("Map highlight: data/raw/sample_cells.csv not found.")
        return

    _cells = _pd.read_csv(_cells_path)

    _latk = next((c for c in ("lat", "latitude", "Lat", "LAT") if c in _cells.columns), None)
    _lonk = next((c for c in ("lon", "lng", "longitude", "Lon", "LON") if c in _cells.columns), None)
    if not (_latk and _lonk):
        st.caption(f"Map highlight: need lat/lon columns. Have: {list(_cells.columns)}")
        return

    # One coord per cell_id (prefer most recent if timestamp exists)
    if "timestamp" in _cells.columns:
        _cells["__ts__"] = _pd.to_datetime(_cells["timestamp"], errors="coerce")
        _cells_u = _cells.sort_values("__ts__").drop_duplicates("cell_id", keep="last")
    else:
        _cells_u = _cells.drop_duplicates("cell_id", keep="first")

    # 3) Merge predictions with coords and normalize names/dtypes
    _m = _pred.merge(_cells_u[["cell_id", _latk, _lonk]], on="cell_id", how="left")
    _plot = (
        _m.rename(columns={_latk: "lat", _lonk: "lon"})
          .dropna(subset=["lat", "lon"])
          .drop_duplicates("cell_id")
          .copy()
    )
    _plot["lat"] = _pd.to_numeric(_plot["lat"], errors="coerce")
    _plot["lon"] = _pd.to_numeric(_plot["lon"], errors="coerce")
    _plot["pred_prob"] = _pd.to_numeric(_plot["pred_prob"], errors="coerce")
    _plot = _plot.dropna(subset=["lat", "lon", "pred_prob"])

    st.caption(f"Map: matched {_plot['cell_id'].nunique()} of {_pred['cell_id'].nunique()} unique cells using [lat, lon]")
    if _plot.empty:
        st.caption("Map highlight: no coordinates for current Top-K cell_ids.")
        return

    # 4) Fixed, small marker scaling (meters → visible pixels via radius_scale)
    p = _plot["pred_prob"].clip(0, 1).astype(float)
    p_min, p_max = float(p.min()), float(p.max())
    z = _np.ones(len(p)) if p_max == p_min else (p - p_min) / (p_max - p_min)

    # Base "size" (arbitrary units) that we will scale in deck.gl
    MIN_U, MAX_U = 3, 10
    _plot["__size"] = (MIN_U + (MAX_U - MIN_U) * z).astype(float)
    _plot["__fill"] = _plot["pred_prob"].apply(lambda s: [255, int(170 * (1 - float(s))), 0, 185])
    # Pretty string for tooltip (deck.gl can't use {:.2f} inline)
    _plot["pred_prob_s"] = _plot["pred_prob"].astype(float).map(lambda v: f"{v:.2f}")

    st.caption(
        "Legend — circle size scales with relative predicted risk among cells in view "
        "(bigger = higher), and color encodes risk from amber → red as probability increases. "
        "Hover a circle for cell ID and predicted probability."
    )

    try:
        import pydeck as pdk

        view = pdk.ViewState(
            latitude=float(_plot["lat"].mean()),
            longitude=float(_plot["lon"].mean()),
            zoom=10,
        )

        # Feed dataframe directly and use robust accessors
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=_plot,
            get_position=["lon", "lat"],     # robust across pydeck versions
            get_radius="__size",             # in "units" below
            radius_scale=80,                 # <- makes markers clearly visible at zoom~10
            radius_min_pixels=6,             # clamp to ensure you always see pins
            radius_max_pixels=80,
            get_fill_color="__fill",
            pickable=True,
            auto_highlight=True,
        )
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                tooltip={"text": "{cell_id}\nProb: {pred_prob_s}"},
            )
        )

    except Exception as _e:
        st.caption("Map renderer fallback: " + type(_e).__name__)
        st.map(_plot[["lat", "lon"]])


# ================= AI INTERPRETATION & CELL EXPLAINER (BETA) =================
import os, json, hashlib
import streamlit as st
import pandas as pd
import joblib as _joblib

# ---------- Context collectors ----------
def _collect_ai_context():
    ctx = {
        "filters": {
            "date": globals().get("date_opt", "ALL"),
            "cell": globals().get("cell_opt", "ALL"),
            "metric_main": st.session_state.get("metric_main", None)
        },
        "kpis": {},
        "risk_summary": {},
        "hotspots": [],
        "anomalies": [],
        "predictions": [],
        "model_meta": {}
    }
    try:
        v = globals().get("view", None)
        if v is not None and not v.empty:
            ctx["kpis"] = {
                "avg_throughput_mbps": float(v.get("throughput_mbps", pd.Series(dtype=float)).mean()),
                "p95_latency_ms": float(v.get("latency_ms", pd.Series(dtype=float)).quantile(0.95)),
                "avg_drop_rate": float(v.get("drop_rate", pd.Series(dtype=float)).mean())
            }
    except Exception:
        pass
    try:
        ctx["risk_summary"] = {
            "rule": globals().get("risk_label", None),
            "risky_hours": int(globals().get("risk_count", 0)),
            "total_hours": int(globals().get("total_hours", 0))
        }
    except Exception:
        pass
    try:
        _hot = globals().get("hotspots_agg", None)
        if _hot is not None and not _hot.empty and "cell_id" in _hot.columns:
            ctx["hotspots"] = _hot.head(10).to_dict("records")
    except Exception:
        pass

    try:
        _top_anom = globals().get("top_anom", None)
        if _top_anom is not None and not _top_anom.empty and "cell_id" in _top_anom.columns:
            ctx["anomalies"] = _top_anom.head(10).to_dict("records")
    except Exception:
        pass
    try:
        _preds = globals().get("view_rows", None)
        if _preds is not None and not _preds.empty and {"cell_id","pred_prob"}.issubset(_preds.columns):
            ctx["predictions"] = _preds.head(20)[["cell_id","pred_prob"]].to_dict("records")
    except Exception:
        pass
    try:
        _bundle = globals().get("bundle", {})
        ctx["model_meta"] = {
            "trained_at": _bundle.get("trained_at"),
            "rule": _bundle.get("label_rule"),
            "metrics": _bundle.get("metrics")
        }
    except Exception:
        pass
    return ctx

def _collect_cell_context(cell_id):
    cell_id = None if cell_id in (None, "", "ALL") else cell_id
    ctx = {"cell_id": cell_id, "kpis": {}, "row": {}, "anomaly_row": {}, "hotspot_row": {}}

    try:
        v = globals().get("view", None)
        if cell_id and v is not None and not v.empty and "cell_id" in v.columns:
            _vrow = v[v["cell_id"].astype(str) == str(cell_id)]
            if not _vrow.empty:
                ctx["row"] = _vrow.head(1).to_dict("records")[0]
            # Aggregate stats for comparison
            ctx["kpis"] = {
                "global_avg_throughput_mbps": float(v.get("throughput_mbps", pd.Series(dtype=float)).mean()),
                "global_p95_latency_ms": float(v.get("latency_ms", pd.Series(dtype=float)).quantile(0.95)),
                "global_avg_drop_rate": float(v.get("drop_rate", pd.Series(dtype=float)).mean())
            }
    except Exception:
        pass
    try:
        _preds = globals().get("view_rows", None)
        if cell_id and _preds is not None and not _preds.empty:
            _prow = _preds[_preds["cell_id"].astype(str) == str(cell_id)]
            if not _prow.empty:
                ctx["row"]["pred_prob"] = float(_prow.head(1)["pred_prob"].iloc[0])
    except Exception:
        pass
    try:
        _top_anom = globals().get("top_anom", None)
        if cell_id and _top_anom is not None and not _top_anom.empty:
            _arow = _top_anom[_top_anom["cell_id"].astype(str) == str(cell_id)]
            if not _arow.empty:
                ctx["anomaly_row"] = _arow.head(1).to_dict("records")[0]
    except Exception:
        pass
    try:
        _hot = globals().get("hotspots_agg", None)
        if cell_id and _hot is not None and not _hot.empty:
            _hrow = _hot[_hot["cell_id"].astype(str) == str(cell_id)]
            if not _hrow.empty:
                ctx["hotspot_row"] = _hrow.head(1).to_dict("records")[0]
    except Exception:
        pass

    return ctx

def _dashboard_hash(ctx: dict) -> str:
    return hashlib.sha256(json.dumps(ctx, sort_keys=True, default=str).encode("utf-8")).hexdigest()

# ---------- LLM callers ----------
def _ai_call(provider: str, model_name: str, system_msg: str, user_payload: dict) -> str:
    user_msg = "Use ONLY the JSON below; do not invent numbers.\n```json\n" + json.dumps(user_payload, default=str) + "\n```"
    if provider == "gemini":
        try:
            import google.generativeai as genai
            if not os.getenv("GOOGLE_API_KEY"): return "_AI disabled: GOOGLE_API_KEY not set._"
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(model_name or "gemini-1.5-flash")
            resp = model.generate_content([system_msg, user_msg])
            return resp.text or "_No response returned._"
        except Exception as e:
            return f"_Gemini error: {e}_"
    elif provider == "openai":
        try:
            from openai import OpenAI
            if not os.getenv("OPENAI_API_KEY"): return "_AI disabled: OPENAI_API_KEY not set._"
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model_name or "gpt-4o-mini",
                messages=[{"role":"system","content":system_msg},
                          {"role":"user","content":user_msg}],
                temperature=0.2,
                max_tokens=700
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"_OpenAI error: {e}_"
    return "_Unknown provider._"

@st.cache_data(show_spinner=False, ttl=600)
def _ai_interpret_cached(ctx: dict, provider: str, model_name: str) -> str:
    system_msg = (
        "You are a senior telecom reliability analyst. "
        "Write a concise briefing using ONLY provided data. "
        "Sections: 1) Executive Summary (<=4 bullets), 2) Key Drivers, "
        "3) At-Risk Cells (top items), 4) Recommended Actions, 5) Caveats/Data Quality."
    )
    return _ai_call(provider, model_name, system_msg, ctx)

@st.cache_data(show_spinner=False, ttl=600)
def _ai_explain_cell_cached(ctx: dict, provider: str, model_name: str) -> str:
    system_msg = (
        "You are a senior RF/network engineer. Explain the selected cell using ONLY the JSON. "
        "Cover: current health, anomalies/drivers, how it compares to global KPIs if provided, "
        "and targeted next actions (fast fixes vs deeper work). Keep it under 12 bullets."
    )
    return _ai_call(provider, model_name, system_msg, ctx)

# ---------- UI ----------
st.subheader("AI Interpretation (beta)")
with st.expander("Generate an executive summary, actions, and per-cell explanations"):
    # Provider & model
    provider = st.selectbox("Provider", ["gemini", "openai"], index=0, help="Gemini by default")
    model_name = st.text_input(
        "Model name",
        value="gemini-1.5-flash" if provider=="gemini" else "gpt-4o-mini",
        help="Use a small/fast model for short briefings"
    )
    st.session_state["ai_provider"] = provider
    st.session_state["ai_model_name"] = model_name
    
# Runtime API key inputs and run controls (safe: no prefilled secrets shown)
if provider == "gemini":
    if not os.getenv("GOOGLE_API_KEY"):
        gem_key = st.text_input("Gemini API key", type="password")
        if gem_key:
            os.environ["GOOGLE_API_KEY"] = gem_key
    else:
        st.caption("Using Gemini key from environment.")
else:
    if not os.getenv("OPENAI_API_KEY"):
        oa_key = st.text_input("OpenAI API key", type="password")
        if oa_key:
            os.environ["OPENAI_API_KEY"] = oa_key
    else:
        st.caption("Using OpenAI key from environment.")

# Cost note + auto-run
st.caption("💡 In a production app, summaries usually auto-refresh when filters change so users always see a fresh briefing. Here we **default to manual** to control token costs while iterating. Toggle auto-run if you want hands-free updates.")

ai_auto = st.checkbox("Auto-generate when filters change (uses tokens)", value=st.session_state.get("ai_auto_run", False))
st.session_state["ai_auto_run"] = ai_auto

# Collect current dashboard context
ctx = _collect_ai_context()
hsh = _dashboard_hash(ctx)
last_hsh = st.session_state.get("ai_last_hash")

# Manual generate
run_clicked = st.button("Generate AI Briefing", type="primary")

# Auto-run trigger
should_autorun = bool(ai_auto and hsh != last_hsh)
do_run = run_clicked or should_autorun

st.session_state.setdefault("ai_runs", 0)
max_runs = 10

if do_run:
    if st.session_state["ai_runs"] >= max_runs:
        st.warning("AI run limit reached for this session. Toggle auto-run off or increase the limit.")
    else:
        with st.spinner("Thinking..."):
            md = _ai_interpret_cached(ctx, provider, model_name)
        st.session_state["ai_last_hash"] = hsh
        st.session_state["ai_last_md"] = md
        st.session_state["ai_runs"] += 1

# Show last result if present
_last = st.session_state.get("ai_last_md")
if _last:
    _low = _last.lower()
    if ("ai disabled" in _low) or ("error:" in _low):
        st.error(_last)
    else:
        st.markdown(_last)
    st.download_button(
        "Download briefing (Markdown)",
        data=_last.encode("utf-8"),
        file_name="ai_interpretation.md",
        mime="text/markdown"
    )

# Debug view of JSON sent to LLM (and quick scored sanity)
if st.checkbox("🔍 Show LLM input (debug)", value=False):
    dbg_ctx = _collect_ai_context()
    st.code(json.dumps(dbg_ctx, indent=2)[:6000], language="json")
    try:
        st.caption(f"debug: scored cols={list(scored.columns)}  head={scored[['cell_id','ts','pred_prob']].head(3).to_dict('records')}")
    except Exception:
        st.caption("debug: scored not available yet.")

    # ------------- Explain this cell -------------
    st.markdown("---")
    preds_df = globals().get("view_rows", pd.DataFrame())
    options = list(pd.Series(preds_df["cell_id"].astype(str).unique()).head(50)) if not preds_df.empty and "cell_id" in preds_df.columns else []
    selected = st.selectbox("Pick a cell to explain (from current predictions)", options) if options else ""
    manual_cell = st.text_input("…or type a cell_id manually", value=selected or "")

    explain_clicked = st.button("Explain selected cell")
    if explain_clicked or (ai_auto and (manual_cell or selected) and hsh != last_hsh):
        cctx = _collect_cell_context(manual_cell or selected)
        with st.spinner("Explaining cell…"):
            emd = _ai_explain_cell_cached(cctx, provider, model_name)
        st.markdown(emd)
        st.download_button(
            "Download cell explanation (Markdown)",
            data=emd.encode("utf-8"),
            file_name=f"cell_{(manual_cell or selected or 'NA')}_explanation.md",
            mime="text/markdown"
        )

    
def _render_prediction_explain_actions(max_rows: int = 30):
    import pandas as pd
    df = globals().get("view_rows", pd.DataFrame())
    if df is None or df.empty:
        st.info("No predictions available for the current filters.")
        return

    st.caption("Click **Explain** to get a focused, actionable write-up for a single cell. "
               "In a production app we’d auto-refresh explanations with filter changes; here we keep it manual to control token costs.")

    provider = st.session_state.get("ai_provider", "gemini")
    model_name = st.session_state.get(
        "ai_model_name",
        "gemini-1.5-flash" if provider == "gemini" else "gpt-4o-mini"
    )

    # Header
    st.markdown("#### Quick Explain (top rows)")
    header = st.columns([2,2,2,2,2,2])
    header[0].markdown("**cell_id**")
    header[1].markdown("**pred_prob**")
    header[2].markdown("**latency_ms**")
    header[3].markdown("**throughput_mbps**")
    header[4].markdown("**drop_rate**")
    header[5].markdown("**Action**")

    subset = df.head(max_rows)
    for idx, row in subset.iterrows():
        c = st.columns([2,2,2,2,2,2])
        cell = str(row.get("cell_id", ""))
        # safe formatting
        def f(x, fmt):
            return (fmt.format(float(x)) if pd.notnull(x) else "—")
        c[0].write(cell)
        c[1].write(f(row.get("pred_prob"), "{:.3f}"))
        c[2].write(f(row.get("latency_ms"), "{:.1f}"))
        c[3].write(f(row.get("throughput_mbps"), "{:.1f}"))
        c[4].write(f(row.get("drop_rate"), "{:.3f}"))

        if c[5].button("Explain", key=f"explain_{cell}_{idx}"):
            with st.spinner(f"Explaining cell {cell}…"):
                cctx = _collect_cell_context(cell)           # from the AI block we added
                emd = _ai_explain_cell_cached(cctx, provider, model_name)
            st.markdown(emd)
            st.download_button(
                "Download cell explanation (Markdown)",
                data=emd.encode("utf-8"),
                file_name=f"cell_{cell}_explanation.md",
                mime="text/markdown"
            )

# ======================= /AI INTERPRETATION BLOCK =======================

# (keep the existing map call after this)
_render_prediction_explain_actions(max_rows=30)
_render_topk_map()


# --- Deployment diagnostics (sidebar) ---
# import os
# import streamlit as st

# def _mask(v: str) -> str:
#     if not v:
#         return "not set"
#     return (v[:4] + "…" + v[-4:]) if len(v) >= 8 else "set"

# with st.sidebar.expander("Deployment diagnostics"):
#     g_key = os.getenv("GOOGLE_API_KEY")
#     o_key = os.getenv("OPENAI_API_KEY")
#     active = "GOOGLE" if g_key else ("OPENAI" if o_key else None)

#     st.write("**Env vars**")
#     st.write(f"GOOGLE_API_KEY: {_mask(g_key)}")
#     st.write(f"OPENAI_API_KEY: {_mask(o_key)}")
#     st.caption("Only one should be set. If both are set, the app may behave unpredictably.")

#     if st.button("Run provider test"):
#         try:
#             if active == "GOOGLE":
#                 import google.generativeai as genai
#                 genai.configure(api_key=g_key)
#                 model = genai.GenerativeModel("gemini-1.5-flash")
#                 resp = model.generate_content(
#                     "Reply with just: OK",
#                     generation_config={"max_output_tokens": 2}
#                 )
#                 st.success(f"Google GenAI OK: {resp.text.strip() if hasattr(resp, 'text') else 'OK'}")
#             elif active == "OPENAI":
#                 # Only runs if you set OPENAI_API_KEY instead of GOOGLE_API_KEY
#                 from openai import OpenAI
#                 client = OpenAI(api_key=o_key)
#                 msg = client.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=[{"role":"user", "content":"Reply with just: OK"}],
#                     max_tokens=2
#                 )
#                 st.success(f"OpenAI OK: {msg.choices[0].message.content.strip()}")
#             else:
#                 st.error("No provider key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in Render → Environment.")
#         except Exception as e:
#             st.error(f"Provider test failed: {type(e).__name__}: {e}")
#             st.caption("Check the key value, project access, or model name. Also see Render logs.")


