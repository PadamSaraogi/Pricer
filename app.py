import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Core Black–Scholes functions
# -----------------------------

def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x: float) -> float:
    """Standard normal CDF using error function (no external deps)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class OptionInput:
    S0: float      # Spot price
    K: float       # Strike
    r: float       # Risk-free rate (annual, cont. comp.)
    sigma: float   # Volatility (annualized)
    T: float       # Time to expiry (years)
    q: float = 0.0 # Dividend yield (annual, cont. comp.)


def _d1_d2(inp: OptionInput) -> Tuple[float, float]:
    S0, K, r, sigma, T, q = inp.S0, inp.K, inp.r, inp.sigma, inp.T, inp.q
    if min(S0, K, sigma, T) <= 0:
        raise ValueError("S0, K, sigma, and T must be positive.")
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_prices(inp: OptionInput) -> Tuple[float, float, float, float]:
    d1, d2 = _d1_d2(inp)
    S0, K, r, T, q = inp.S0, inp.K, inp.r, inp.T, inp.q
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    call = S0 * disc_q * _Phi(d1) - K * disc_r * _Phi(d2)
    put  = K * disc_r * _Phi(-d2) - S0 * disc_q * _Phi(-d1)
    return call, put, d1, d2


def bs_greeks(inp: OptionInput) -> Dict:
    S0, K, r, sigma, T, q = inp.S0, inp.K, inp.r, inp.sigma, inp.T, inp.q
    d1, d2 = _d1_d2(inp)
    nd1 = _phi(d1)
    Nd1 = _Phi(d1)
    Nd2 = _Phi(d2)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    # Delta with dividends (q): multiply S term by e^{-qT}
    delta_call = disc_q * Nd1
    delta_put  = disc_q * (Nd1 - 1.0)

    gamma = disc_q * nd1 / (S0 * sigma * math.sqrt(T))
    vega  = S0 * disc_q * nd1 * math.sqrt(T)             # per 100% vol

    theta_call = -(S0 * disc_q * nd1 * sigma) / (2.0 * math.sqrt(T)) \
                 - r * K * disc_r * Nd2 + q * S0 * disc_q * Nd1
    theta_put  = -(S0 * disc_q * nd1 * sigma) / (2.0 * math.sqrt(T)) \
                 + r * K * disc_r * _Phi(-d2) - q * S0 * disc_q * _Phi(-d1)

    rho_call   = K * T * disc_r * Nd2
    rho_put    = -K * T * disc_r * _Phi(-d2)

    return {
        "delta": {"call": delta_call, "put": delta_put},
        "gamma": gamma,
        "vega_per_1": vega,
        "vega_per_1pct": vega / 100.0,
        "theta_per_year": {"call": theta_call, "put": theta_put},
        "theta_per_day": {"call": theta_call / 365.0, "put": theta_put / 365.0},
        "rho": {"call": rho_call, "put": rho_put},
        "d1": d1, "d2": d2
    }


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Option Pricer — Black–Scholes", page_icon="📈", layout="wide")

st.title("📈 Black–Scholes Option Pricer (with Greeks)")
with st.expander("What is this?", expanded=False):
    st.write(
        """
        This app prices **European call/put options** using the **Black–Scholes** model and shows key **Greeks**.
        It supports a continuous **dividend yield (q)**. Prices and Greeks update instantly as you change inputs.
        """
    )

# Sidebar — inputs
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

opt_type = st.sidebar.radio("Option type for charts/metrics", ["Call", "Put"], horizontal=True)

inp = OptionInput(S0=S0, K=K, r=r, sigma=sigma, T=T, q=q)

# Compute prices and greeks
try:
    call, put, d1, d2 = bs_prices(inp)
    G = bs_greeks(inp)
except Exception as e:
    st.error(f"Input error: {e}")
    st.stop()

# Top metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Call price", f"{call:,.4f}")
m2.metric("Put price", f"{put:,.4f}")
m3.metric("d₁", f"{d1:,.4f}")
m4.metric("d₂", f"{d2:,.4f}")

# Greeks table
st.subheader("Greeks")

greek_df = pd.DataFrame({
    "Greek": ["Delta", "Gamma", "Vega (per 1% σ)", "Theta/day", "Theta/year", "Rho"],
    "Call": [
        G["delta"]["call"],
        G["gamma"],
        G["vega_per_1pct"],
        G["theta_per_day"]["call"],
        G["theta_per_year"]["call"],
        G["rho"]["call"]
    ],
    "Put": [
        G["delta"]["put"],
        G["gamma"],
        G["vega_per_1pct"],
        G["theta_per_day"]["put"],
        G["theta_per_year"]["put"],
        G["rho"]["put"]
    ]
})

st.dataframe(greek_df.style.format({"Call": "{:.6f}", "Put": "{:.6f}"}), use_container_width=True)

# -----------------------------
# Charts: (1) Payoff at expiry, (2) BS value vs spot today
# -----------------------------

st.subheader("Charts")

# Range for S now / at expiry
S_min = max(0.01, S0 * 0.4)
S_max = S0 * 1.6
S_grid = np.linspace(S_min, S_max, 200)

# (1) Payoff at expiry (T->0) — piecewise payoff
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

# (2) BS Theoretical value today vs current spot range (holding other params)
call_vals = []
put_vals = []
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

# -----------------------------
# Scenario table & download
# -----------------------------

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

# Footer
st.caption("Built for quick demos and education. Model: European options, Black–Scholes, continuous compounding, optional dividend yield q.")

