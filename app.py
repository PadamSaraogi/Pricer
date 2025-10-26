import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from bs_core import OptionInput, bs_prices, bs_greeks, implied_vol, moneyness_tags

st.set_page_config(page_title="Option Pricer — Black–Scholes", page_icon="📈", layout="wide")
st.title("📈 Black–Scholes Option Pricer (with Greeks & IV)")

with st.expander("What's new?", expanded=True):
    st.markdown("""
    **New features**
    - **Implied Volatility (IV) solver**: input market price → get σ.
    - **Bid/Ask support**: auto-compute mid and **IV(mid)**.
    - **Moneyness panel**: S/K, log-moneyness, d₁, and tags (ITM/ATM/OTM).
    - Core logic refactored into `bs_core.py` with simple tests.
    """)

# --- Sidebar Inputs ---
st.sidebar.header("Inputs")
colA, colB = st.sidebar.columns(2)
S0 = colA.number_input("Spot price (S₀)", min_value=0.01, value=100.0, step=1.0)
K  = colB.number_input("Strike (K)", min_value=0.01, value=105.0, step=1.0)

col1, col2 = st.sidebar.columns(2)
r  = col1.number_input("Risk-free r (annual, %)", value=5.0, step=0.25, format="%.2f")/100.0
sigma = col2.number_input("Volatility σ (annual, %)", min_value=0.01, value=20.0, step=0.5, format="%.2f")/100.0

col3, col4 = st.sidebar.columns(2)
T  = col3.number_input("Time to expiry T (years)", min_value=0.00274, value=0.5, step=0.01, help="0.00274 ≈ 1 day")
q  = col4.number_input("Dividend yield q (annual, %)", value=0.0, step=0.25, format="%.2f")/100.0

opt_type = st.sidebar.radio("Option type for calculations", ["Call", "Put"], horizontal=True)
mode = st.sidebar.radio("Mode", ["Inputs → Price", "Market Price → Implied Vol"], horizontal=False)

# Optional market data
st.sidebar.divider()
st.sidebar.caption("Optional: enter market quotes (any currency)")
colm1, colm2, colm3 = st.sidebar.columns(3)
bid = colm1.number_input("Bid", min_value=0.0, value=0.0, step=0.1)
ask = colm2.number_input("Ask", min_value=0.0, value=0.0, step=0.1)
mid = None
if bid > 0 and ask > 0 and ask >= bid:
    mid = 0.5 * (bid + ask)

# Optional single market price for IV mode
mkt_price = st.sidebar.number_input("Single market price (for IV mode)", min_value=0.0, value=0.0, step=0.1)

inp = OptionInput(S0=S0, K=K, r=r, sigma=sigma, T=T, q=q)

# --- Compute & Display ---
try:
    call, put, d1, d2 = bs_prices(inp)
    G = bs_greeks(inp)
except Exception as e:
    st.error(f"Input error: {e}")
    st.stop()

# Moneyness panel
tag_info = moneyness_tags(S0, K, d1)
with st.container():
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Call price", f"{call:,.4f}")
    m2.metric("Put price", f"{put:,.4f}")
    m3.metric("d₁", f"{d1:,.4f}")
    m4.metric("d₂", f"{d2:,.4f}")

st.subheader("Moneyness & Tags")
mtc1, mtc2, mtc3, mtc4 = st.columns(4)
mtc1.metric("S/K", tag_info["S_over_K"]) 
mtc2.metric("log-moneyness", tag_info["log_moneyness"]) 
mtc3.metric("d₁", tag_info["d1"]) 
mtc4.metric("Tag", tag_info["tag"]) 

# Greeks table
st.subheader("Greeks")
greek_df = pd.DataFrame({
    "Greek": ["Delta", "Gamma", "Vega (per 1% σ)", "Theta/day", "Theta/year", "Rho"],
    "Call": [G["delta"]["call"], G["gamma"], G["vega_per_1pct"], G["theta_per_day"]["call"], G["theta_per_year"]["call"], G["rho"]["call"]],
    "Put":  [G["delta"]["put"],  G["gamma"], G["vega_per_1pct"], G["theta_per_day"]["put"],  G["theta_per_year"]["put"],  G["rho"]["put"]]
})
st.dataframe(greek_df.style.format({"Call": "{:.6f}", "Put": "{:.6f}"}), use_container_width=True)

# --- IV Mode ---
if mode == "Market Price → Implied Vol":
    st.subheader("Implied Volatility (from market price)")
    if mkt_price <= 0 and mid is None:
        st.info("Enter a Single market price, or Bid/Ask to compute IV(mid).")
    coliv1, coliv2 = st.columns(2)
    if mkt_price > 0:
        iv = implied_vol(mkt_price, inp, opt_type.lower())
        if iv is None:
            coliv1.error("No valid IV found in [1e-6, 5.0]. Check inputs/price.")
        else:
            coliv1.metric("IV (from single price)", f"{iv*100:.3f}%")
    if mid is not None:
        iv_mid = implied_vol(mid, inp, opt_type.lower())
        if iv_mid is None:
            coliv2.error("No valid IV for Bid/Ask mid.")
        else:
            coliv2.metric("IV (from Bid/Ask mid)", f"{iv_mid*100:.3f}%")

# --- Charts ---
st.subheader("Charts")
S_min = max(0.01, S0 * 0.4)
S_max = S0 * 1.6
S_grid = np.linspace(S_min, S_max, 200)

# Payoff at expiry
payoff_call = np.maximum(S_grid - K, 0.0)
payoff_put  = np.maximum(K - S_grid, 0.0)
fig1, ax1 = plt.subplots()
if opt_type == "Call":
    ax1.plot(S_grid, payoff_call, label="Call payoff at expiry")
else:
    ax1.plot(S_grid, payoff_put, label="Put payoff at expiry")
ax1.axhline(0, linewidth=1)
ax1.set_xlabel("Underlying price at expiry (S_T)")
ax1.set_ylabel("Payoff")
ax1.set_title(f"{opt_type} payoff at expiry (not PV)")
ax1.legend()
st.pyplot(fig1, use_container_width=True)

# BS value vs spot today
call_vals, put_vals = [], []
for s in S_grid:
    _in = OptionInput(S0=float(s), K=K, r=r, sigma=sigma, T=T, q=q)
    c, p, *_ = bs_prices(_in)
    call_vals.append(c)
    put_vals.append(p)
fig2, ax2 = plt.subplots()
if opt_type == "Call":
    ax2.plot(S_grid, call_vals, label="Call value (today)")
    ax2.scatter([S0], [call], marker="o")
else:
    ax2.plot(S_grid, put_vals, label="Put value (today)")
    ax2.scatter([S0], [put], marker="o")
ax2.set_xlabel("Spot price S₀")
ax2.set_ylabel("Option value today")
ax2.set_title(f"{opt_type} value vs spot (Black–Scholes)")
ax2.legend()
st.pyplot(fig2, use_container_width=True)

# Scenario table
st.subheader("Scenario table (spot ladder)")
spots = np.linspace(S0 * 0.7, S0 * 1.3, 11)
rows = []
for s in spots:
    _in = OptionInput(S0=float(s), K=K, r=r, sigma=sigma, T=T, q=q)
    c, p, *_ = bs_prices(_in)
    rows.append({"Spot": s, "Call": c, "Put": p})

df_scn = pd.DataFrame(rows)
st.dataframe(df_scn.style.format({"Spot": "{:.4f}", "Call": "{:.4f}", "Put": "{:.4f}"}), use_container_width=True)

csv = df_scn.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="option_pricer_scenarios.csv", mime="text/csv")

st.caption("European options, Black–Scholes, continuous compounding, optional dividend yield q. IV solved via bracketed bisection/secant on [1e-6, 5.0].")
