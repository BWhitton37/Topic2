"""
ECO 5012B — Coursework Project, Topic 2
Section 3: Interactive Application and Research Extension

Run with:  streamlit run st_app.py   (from this "app/" directory)

NOTE: this sandbox has no internet access, so `streamlit`/`plotly` could
not be pip-installed or executed here to test-run this file. It is written
against the standard, stable Streamlit/Plotly APIs and uses the same fitted
coefficients validated in analysis.py -- but please run it locally to
confirm before submission, and let me know if anything errors.
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Trade Tension & Market Volatility", layout="wide")

# --- Load fitted model + historical data (produced by ../code/analysis.py) ---
with open("../data/processed/model_coefficients.json") as f:
    coefs = json.load(f)

hist = pd.read_csv("../data/processed/trade_tension_data.csv", parse_dates=["Time"])

st.title("Trade Tensions and Financial Market Volatility")
st.caption(
    "Reference: Ferrari Minesso, Pagliari & Kurcz (2022), 'Do words hurt more than "
    "actions? The impact of trade tensions on financial markets', JAE 37(6). "
    "Data: S&P 500 and trade-policy uncertainty (EPUTRADE) via FRED."
)

st.markdown(
    f"My fitted model: squared return today = a constant + gamma \u00d7 trade policy "
    f"uncertainty today + a bit of yesterday's squared return + a bit of general policy "
    f"uncertainty. Estimated on {len(hist)} daily S&P 500 observations (2016-2019). "
    f"gamma_hat = {coefs['gamma_TPU']:.5f} (p-value = {coefs['gamma_p_value']:.4f}) - "
    f"positive and statistically significant, meaning higher trade-policy uncertainty "
    f"is associated with higher next-day volatility."
)

# =========================================================================
# Step 3.2 — interactive slider + dynamic metric
# =========================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Set trade-policy uncertainty index (TPU)")
    tpu_input = st.slider(
        "Trade Policy Uncertainty Index (TPU_t) — higher values mean more "
        "trade-related uncertainty in the news that month",
        min_value=float(hist["TPU"].min()), max_value=float(hist["TPU"].max()),
        value=float(hist["TPU"].median()), step=10.0,
    )

    # =====================================================================
    # Step 3.3 — asymmetric "words vs actions" state, via st.radio
    # =====================================================================
    st.subheader("Sentiment regime")
    regime = st.radio(
        "Select sentiment state (adjusts the trade-tension impact coefficient γ "
        "by ±50%, reflecting the paper's 'words hurt more than actions' asymmetry):",
        options=["Low sentiment (calmer than usual)", "Normal sentiment", "High sentiment (heightened tension)"],
        index=1,
    )

    if regime.startswith("Low"):
        gamma_adj = coefs["gamma_TPU"] * 0.5   # dampened impact
    elif regime.startswith("High"):
        gamma_adj = coefs["gamma_TPU"] * 1.5   # amplified impact
    else:
        gamma_adj = coefs["gamma_TPU"]

# --- One-period-ahead forecast using the last observed lagged volatility & EPU ---
last_ret_sq = float(hist["ret_sq"].iloc[-1])
last_epu = float(hist["EPU"].iloc[-1])

forecast = (
    coefs["const"]
    + gamma_adj * tpu_input
    + coefs["beta_ret_sq_lag1"] * last_ret_sq
    + coefs["beta_EPU"] * last_epu
)
forecast = max(0.0, forecast)  # variance/squared-return forecast cannot be negative

baseline_forecast = (
    coefs["const"]
    + coefs["gamma_TPU"] * float(hist["TPU"].median())   # neutral = typical/median TPU level
    + coefs["beta_ret_sq_lag1"] * last_ret_sq
    + coefs["beta_EPU"] * last_epu
)

with col2:
    st.subheader("Live forecast")
    st.metric(
        label="One-period-ahead forecast: squared S&P 500 return (%²)",
        value=f"{forecast:.3f}",
        delta=f"{forecast - baseline_forecast:+.3f} vs. neutral-TPU baseline",
    )
    st.caption(f"Effective γ used: {gamma_adj:.3f} ({regime})")

# =========================================================================
# Step 3.4 — final visualisation: historical series + live projection
# =========================================================================
st.subheader("Historical volatility with live one-period-ahead projection")

next_time = hist["Time"].iloc[-1] + pd.Timedelta(days=1)
plot_hist = hist.tail(150)  # last ~150 trading days for a readable chart

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot_hist["Time"], y=plot_hist["ret_sq"],
    mode="lines", name="Historical squared return", line=dict(color="steelblue"),
))
fig.add_trace(go.Scatter(
    x=[plot_hist["Time"].iloc[-1], next_time],
    y=[plot_hist["ret_sq"].iloc[-1], forecast],
    mode="lines+markers", name="Live forecast", line=dict(color="firebrick", dash="dash"),
    marker=dict(size=10),
))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Squared daily return (%²)",
    title="S&P 500 realised volatility: history + live nowcast",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Research extension: what this app illustrates"):
    st.markdown(
        "The central finding of Ferrari Minesso et al. (2022) is that trade-related "
        "**rhetoric** ('words') moves markets faster and, in some specifications, more "
        "strongly than realised **policy actions**. This app operationalises that "
        "asymmetry directly: the same TPU shock is scaled by a different γ depending on "
        "the selected sentiment regime, so a user can see how an identical piece of "
        "trade news would be expected to move volatility differently depending on the "
        "prevailing tension environment. A natural extension (not built here, for scope "
        "reasons) would be to fit *separate* γ's for 'words'-type events (statements, "
        "tweets) versus 'actions'-type events (tariffs actually imposed) rather than "
        "approximating the asymmetry with a fixed ±50% multiplier."
    )
