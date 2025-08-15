import os, re
import pandas as pd
import streamlit as st
import altair as alt

st.title("Network IQ — EDA (MVP Stub)")

parquet_dir = "data/curated/parquet"

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
def fmt(v, nd=1):
    try: return f"{float(v):.{nd}f}"
    except: return "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Throughput (Mbps)", fmt(view.get("throughput_mbps", pd.Series()).mean()))
c2.metric("P95 Latency (ms)", fmt(view.get("latency_ms", pd.Series()).quantile(0.95)))
c3.metric("Drop Rate (%)", fmt(view.get("drop_rate", pd.Series()).mean(), 2))

try:
    tmp = view[["throughput_mbps","latency_ms","drop_rate"]].astype(float)
    z = (tmp - tmp.mean()) / (tmp.std(ddof=0) + 1e-9)
    nps_proxy = (1.0*z["throughput_mbps"] - 0.8*z["latency_ms"] - 1.2*z["drop_rate"]).mean()
    c4.metric("NPS Proxy (z)", f"{nps_proxy:.2f}")
except Exception:
    c4.metric("NPS Proxy (z)", "—")

st.dataframe(view.sort_values("timestamp").head(100))

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

    if low_is_bad:
        thr = float(hourly[metric].quantile(0.95))
        hourly["risk"] = (hourly[metric] >= thr)
        risk_label = f">= P95 {metric}"
    else:
        thr = float(hourly[metric].quantile(0.05))
        hourly["risk"] = (hourly[metric] <= thr)
        risk_label = f"<= P05 {metric}"

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
    st.info(f"{risk_count}/{total_hours} hours flagged as risky ({risk_count/total_hours:.0%}).")

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



# ---------- Top Hotspots ----------
st.subheader("Top Hotspots")
g = df if date_opt == "ALL" else df[df["date"] == date_opt]
hot_choice = st.selectbox("Rank by", ["Highest P95 Latency","Lowest Avg Throughput","Highest Drop Rate"], key="hot")
if hot_choice == "Highest P95 Latency" and "latency_ms" in g.columns:
    agg = g.groupby("cell_id")["latency_ms"].quantile(0.95).sort_values(ascending=False).head(10).reset_index(name="p95_latency_ms")
elif hot_choice == "Lowest Avg Throughput" and "throughput_mbps" in g.columns:
    agg = g.groupby("cell_id")["throughput_mbps"].mean().sort_values().head(10).reset_index(name="avg_throughput_mbps")
else:
    agg = g.groupby("cell_id")["drop_rate"].mean().sort_values(ascending=False).head(10).reset_index(name="avg_drop_rate")
st.dataframe(agg)

st.subheader("Top Anomalies (z-score vs network baseline)")
# Build baseline by hour-of-day for the current date filter
scope = df if date_opt == "ALL" else df[df["date"] == date_opt]
if {"cell_id","hour", "throughput_mbps","latency_ms","drop_rate"}.issubset(scope.columns):
    metric_a = st.selectbox("Anomaly metric", ["throughput_mbps","latency_ms","drop_rate"], index=1, key="anom_metric")
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

# Work on current date scope (keeps your Date filter consistent)
scope = df if date_opt == "ALL" else df[df["date"] == date_opt]
scope = scope.copy()
if "timestamp" in scope.columns:
    scope["timestamp"] = pd.to_datetime(scope["timestamp"], errors="coerce")
    scope["ts"] = scope["timestamp"].dt.floor("H")
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


