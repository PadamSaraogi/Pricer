# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
from io import BytesIO

from bs_core import (
    OptionInput,
    bs_prices,
    bs_greeks,
    implied_vol,
    moneyness_tags,
)
from american_binomial import crr_price_european, crr_price_american
from vol_surface import compute_chain_iv, prepare_surface_arrays
from bonds import (
    price_bond,
    ytm_from_price,
    macaulay_duration,
    modified_duration,
    convexity_numeric,
    price_bond_dates,
    ytm_from_price_dates,
    macaulay_duration_dates,
    modified_duration_dates,
)
from yield_curve import YieldCurve
from swaps import par_swap_rate, swap_pv, dv01
from daycount import year_fraction, DayCount, BizConv
from date_utils import parse_date
from forwards import forward_price, forward_price_from_curve
from black76 import (
    black76_price,
    black76_greeks,
    implied_vol_b76,
    black76_price_df,
    black76_greeks_df,
)
from dividends import CashDividend, spot_adjusted_for_dividends
from reports import build_options_report_pdf, build_metrics_csv

# ---------------------------
# Cached helpers
# ---------------------------
@st.cache_data(show_spinner=False, ttl=300)
def cached_bootstrap_curve(
    deposits, bonds, valuation_date: date | None = None, day_count: str = "ACT/365F"
):
    from yield_curve import bootstrap_curve as _bootstrap_curve
    return _bootstrap_curve(deposits, bonds, valuation_date=valuation_date, day_count=day_count)


@st.cache_data(show_spinner=False, ttl=300)
def cached_compute_chain_iv(
    raw_df: pd.DataFrame,
    S0: float,
    r: float,
    q: float,
    default_T: float,
    default_type: str,
    sigma_seed: float,
) -> pd.DataFrame:
    from vol_surface import compute_chain_iv as _compute_chain_iv
    return _compute_chain_iv(
        raw_df, S0=S0, r=r, q=q, default_T=default_T, default_type=default_type, sigma_seed=sigma_seed
    )


st.set_page_config(page_title="Option & Fixed-Income Analytics", page_icon="📊", layout="wide")
st.title("📊 Option & Fixed-Income Analytics Dashboard")

# ========== Shared UI helper for Options inputs (inside tabs) ==========
def render_options_inputs(defaults_key: str = "opt_defaults"):
    """
    Renders the full set of options inputs INSIDE the current tab (no sidebar).
    Returns: dict with
      S0, K, r, sigma, T, q, opt_type, dc, val_date, S_for_options
    (S_for_options may be adjusted for discrete dividends.)
    """
    from datetime import date as _date
    import pandas as _pd
    from daycount import DayCount as _DC, year_fraction as _yf
    from date_utils import parse_date as _pdate
    from dividends import CashDividend as _CashDiv, spot_adjusted_for_dividends as _spot_adj

    if defaults_key not in st.session_state:
        st.session_state[defaults_key] = {
            "use_dates": False,
            "S0": 100.0,
            "K": 105.0,
            "r": 0.05,
            "sigma": 0.20,
            "T": 0.5,
            "q": 0.0,
            "opt_type": "Call",
            "dc": "ACT/365F",
            "val_date": str(_date.today()),
            "expiry": str(_date.today()),
            "use_disc_div": False,
            "div_dc": "ACT/365F",
            "div_rows": _pd.DataFrame(
                {
                    "Pay Date (YYYY-MM-DD)": [str(_date.today())],
                    "Amount": [1.00],
                }
            ),
        }
    s = st.session_state[defaults_key]
    key = lambda name: f"{defaults_key}__{name}"

    with st.expander("⚙️ Options Inputs", expanded=True):
        use_dates = st.toggle(
            "Use Dates for maturities",
            value=s["use_dates"],
            help="Compute T from dates using day-counts.",
            key=key("use_dates_toggle"),
        )
        s["use_dates"] = use_dates

        colA, colB = st.columns(2)
        S0 = colA.number_input("Spot price (S₀)",
                               min_value=0.01, value=float(s["S0"]), step=1.0,
                               key=key("S0"))
        K = colB.number_input("Strike (K)",
                              min_value=0.01, value=float(s["K"]), step=1.0,
                              key=key("K"))
        s["S0"], s["K"] = S0, K

        col1, col2 = st.columns(2)
        r = col1.number_input("Risk-free r (annual, %, cont.)",
                              value=float(s["r"]*100), step=0.25, format="%.2f",
                              key=key("r_pct")) / 100.0
        sigma = col2.number_input("Volatility σ (annual, %)",
                                  min_value=0.01, value=float(s["sigma"]*100), step=0.5, format="%.2f",
                                  key=key("sigma_pct")) / 100.0
        s["r"], s["sigma"] = r, sigma

        if use_dates:
            dc: _DC = st.selectbox(
                "Day-count (for T)",
                ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"],
                index=["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"].index(s["dc"]),
                key=key("dc_select"),
            )
            val_date_str = st.text_input("Valuation date (YYYY-MM-DD)",
                                         value=s["val_date"], key=key("val_date"))
            expiry_str = st.text_input("Expiry date (YYYY-MM-DD)",
                                       value=s["expiry"], key=key("expiry"))
            try:
                val_date = _pdate(val_date_str)
                exp_date = _pdate(expiry_str)
                T_eff = _yf(val_date, exp_date, dc)
            except Exception:
                val_date = _date.today()
                T_eff = 0.0
            s["dc"], s["val_date"], s["expiry"] = dc, val_date_str, expiry_str
        else:
            dc: _DC = "ACT/365F"
            val_date = _date.today()
            T_eff = st.number_input("Time to expiry T (years)",
                                    min_value=0.00274, value=float(s["T"]), step=0.01,
                                    help="0.00274 ≈ 1 day.",
                                    key=key("T_years"))
            s["T"] = T_eff

        q = st.number_input("Dividend yield q (annual, %, cont.)",
                            value=float(s["q"]*100), step=0.25, format="%.2f",
                            key=key("q_pct")) / 100.0
        s["q"] = q

        opt_type = st.radio(
            "Option type",
            ["Call", "Put"],
            horizontal=True,
            index=0 if s["opt_type"] == "Call" else 1,
            key=key("opt_type"),
        )
        s["opt_type"] = opt_type

    S_eff = S0
    use_disc_div = st.checkbox(
        "Use discrete CASH dividends (adjust spot)",
        value=s["use_disc_div"],
        help="If enabled, we adjust S₀ for PV of future cash dividends.",
        key=key("use_disc_div"),
    )
    s["use_disc_div"] = use_disc_div

    div_dc: _DC = st.selectbox(
        "Day-count for dividends",
        ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"],
        index=["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"].index(s["div_dc"]),
        disabled=not use_disc_div,
        key=key("div_dc"),
    )
    s["div_dc"] = div_dc

    if use_disc_div:
        st.caption("Add future cash dividends (per share) that occur before option expiry.")
        div_df = st.data_editor(
            s["div_rows"],
            num_rows="dynamic",
            use_container_width=True,
            key=key("disc_div_tbl"),
        )
        s["div_rows"] = div_df
        dividends_list = []
        for _, row in div_df.iterrows():
            try:
                dpay = parse_date(str(row["Pay Date (YYYY-MM-DD)"]))
                amt = float(row["Amount"])
                if amt > 0 and dpay > val_date:
                    dividends_list.append(_CashDiv(pay_date=dpay, amount=amt))
            except Exception:
                pass
        if dividends_list:
            S_eff_calc = _spot_adj(S0, dividends_list, r_cont=r, valuation_date=val_date, dc=div_dc)
            pv_div = S0 - S_eff_calc
            S_eff = S_eff_calc
            st.info(f"S₀,eff = S₀ − PV(dividends) = {S0:.4f} − {pv_div:.4f} = {S_eff:.4f}")
        else:
            S_eff = S0

    return {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T_eff,
        "q": q,
        "opt_type": opt_type,
        "dc": dc,
        "val_date": val_date,
        "S_for_options": S_eff,
    }


# ===================== TABS =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Option Pricer",
        "American (CRR)",
        "Chain & Surface",
        "Bonds",
        "Yield Curve",
        "Swaps",
        "Futures & Black-76",
        "Reports & Export",
    ]
)

# -------- Helper: compact beginner-friendly summary for Tab 1 --------
def summarize_option_output(opts, call, put, d1, d2, G, tag_info, mkt_price, iv_single=None, iv_mid=None):
    S = opts["S_for_options"]
    K = opts["K"]
    opt = opts["opt_type"]
    lines = []

    # 1) Moneyness
    tag = tag_info.get("tag", "")
    if "ATM" in tag:
        lines.append(f"{opt} is **ATM** — most of the value is time value.")
    elif "ITM" in tag:
        lines.append(f"{opt} is **ITM** — it already has intrinsic value.")
    elif "OTM" in tag:
        lines.append(f"{opt} is **OTM** — payoff only if price crosses the strike.")
    else:
        lines.append(f"Moneyness tag: {tag} — double-check S₀ and K if this looks odd.")

    # 2) Price context
    lines.append(f"Model prices: Call ≈ {call:.2f}, Put ≈ {put:.2f} using your inputs.")

    # 3) Sensitivity: Delta + Gamma
    delta = G["delta"]["call"] if opt.lower() == "call" else G["delta"]["put"]
    gamma = G["gamma"]

    if abs(delta) < 0.3:
        sens = "low"
    elif abs(delta) < 0.7:
        sens = "moderate"
    else:
        sens = "high"

    speed = "fast" if gamma > 0.05 else "stable"
    lines.append(
        f"Delta ≈ {delta:.2f}, Gamma ≈ {gamma:.3f} → {sens} sensitivity; delta changes {speed} if the price moves."
    )

    # 4) Time decay + vol risk: Theta + Vega
    theta = G["theta_per_day"]["call"] if opt.lower() == "call" else G["theta_per_day"]["put"]
    vega = G["vega_per_1pct"]

    decay = f"loses about {abs(theta):.4f} per day" if theta < 0 else "has minimal daily time decay"
    if vega > 0.5:
        vol = "very sensitive to volatility"
    elif vega > 0.1:
        vol = "moderately sensitive to volatility"
    else:
        vol = "not very sensitive to volatility"

    lines.append(
        f"Theta ≈ {theta:.4f}/day → it {decay}. Vega ≈ {vega:.3f} → it is {vol}."
    )

    # 5) Market vs model (only if user gives a price)
    model_px = call if opt.lower() == "call" else put
    if mkt_price and mkt_price > 0 and model_px > 0:
        diff = (mkt_price - model_px) / model_px
        if abs(diff) < 0.05:
            lines.append("Market price is **close** to the model price.")
        elif diff > 0:
            lines.append("Market price is **above** model → market is pricing in more volatility/risk.")
        else:
            lines.append("Market price is **below** model → model suggests it may be cheap vs these inputs.")

    # 6) Implied volatility level
    iv = iv_single or iv_mid
    if iv is not None:
        if iv < 0.15:
            lvl = "low"
        elif iv < 0.40:
            lvl = "normal"
        else:
            lvl = "high"
        lines.append(f"Implied volatility ≈ {iv * 100:.1f}% → a **{lvl}** volatility regime.")

    return lines[:5]


# ===== TAB 1: Option Pricer (BSM) =====
with tab1:
    opts = render_options_inputs("opt_inputs_tab1")
    st.subheader("European Options — Black–Scholes")

    colm1, colm2, _ = st.columns(3)
    bid = colm1.number_input("Bid", min_value=0.0, value=0.0, step=0.1, key="bid1")
    ask = colm2.number_input("Ask", min_value=0.0, value=0.0, step=0.1, key="ask1")
    mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0 and ask >= bid) else None
    mkt_price = st.number_input(
        "Single market price (for IV)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="mkt1",
    )

    inp = OptionInput(
        S0=opts["S_for_options"],
        K=opts["K"],
        r=opts["r"],
        sigma=opts["sigma"],
        T=opts["T"],
        q=opts["q"],
    )

    try:
        call, put, d1, d2 = bs_prices(inp)
        G = bs_greeks(inp)
    except Exception as e:
        st.error(f"Input error: {e}")
        st.stop()

    tag_info = moneyness_tags(opts["S_for_options"], opts["K"], d1)

    # Top-level metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Call price", f"{call:,.4f}")
    m2.metric("Put price", f"{put:,.4f}")
    m3.metric("S/K", tag_info["S_over_K"])
    m4.metric("Moneyness tag", tag_info["tag"])

    # Moneyness details
    with st.expander("Moneyness details", expanded=False):
        mtc1, mtc2, mtc3 = st.columns(3)
        mtc1.metric("log-moneyness", tag_info["log_moneyness"])
        mtc2.metric("d₁", tag_info["d1"])
        mtc3.metric("d₂", f"{d2:.4f}")

    # Core Greeks
    st.markdown("### Greeks (core)")
    greek_df_basic = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega (per 1% σ)", "Theta/day"],
            "Call": [
                G["delta"]["call"],
                G["gamma"],
                G["vega_per_1pct"],
                G["theta_per_day"]["call"],
            ],
            "Put": [
                G["delta"]["put"],
                G["gamma"],
                G["vega_per_1pct"],
                G["theta_per_day"]["put"],
            ],
        }
    )
    st.dataframe(
        greek_df_basic.style.format({"Call": "{:.6f}", "Put": "{:.6f}"}),
        use_container_width=True,
    )

    # More Greeks
    with st.expander("More Greeks (annual Theta, Rho)", expanded=False):
        greek_df_adv = pd.DataFrame(
            {
                "Greek": ["Theta/year", "Rho"],
                "Call": [
                    G["theta_per_year"]["call"],
                    G["rho"]["call"],
                ],
                "Put": [
                    G["theta_per_year"]["put"],
                    G["rho"]["put"],
                ],
            }
        )
        st.dataframe(
            greek_df_adv.style.format({"Call": "{:.6f}", "Put": "{:.6f}"}),
            use_container_width=True,
        )

    # Implied vol
    st.markdown("### Implied Volatility (from single market price)")
    coliv1, coliv2 = st.columns(2)
    otype = opts["opt_type"].lower()
    iv = None
    iv_mid = None

    if mkt_price > 0:
        iv = implied_vol(mkt_price, inp, otype)
        if iv is not None:
            coliv1.metric("IV (single price)", f"{iv * 100:.3f}%")
        else:
            coliv1.metric("IV (single price)", "No root")

    if mid is not None:
        iv_mid = implied_vol(mid, inp, otype)
        if iv_mid is not None:
            coliv2.metric("IV (Bid/Ask mid)", f"{iv_mid * 100:.3f}%")
        else:
            coliv2.metric("IV (Bid/Ask mid)", "No root")

    # 🔰 Beginner-friendly, output-based analysis
    analysis_lines = summarize_option_output(
        opts=opts,
        call=call,
        put=put,
        d1=d1,
        d2=d2,
        G=G,
        tag_info=tag_info,
        mkt_price=mkt_price,
        iv_single=iv,
        iv_mid=iv_mid,
    )
    with st.expander("notes & interpretation", expanded=True):
        for ln in analysis_lines:
            st.markdown(f"- {ln}")

    # Payoff chart
    st.markdown("### Payoff at expiry")
    S_min = max(0.01, opts["S_for_options"] * 0.4)
    S_max = opts["S_for_options"] * 1.6
    S_grid = np.linspace(S_min, S_max, 200)
    payoff = (
        np.maximum(S_grid - opts["K"], 0.0)
        if opts["opt_type"].lower() == "call"
        else np.maximum(opts["K"] - S_grid, 0.0)
    )

    fig1, ax1 = plt.subplots()
    ax1.plot(S_grid, payoff, label=f"{opts['opt_type']} payoff at expiry")
    ax1.axvline(opts["S_for_options"], linestyle="--", linewidth=1, label="Current spot")
    ax1.axhline(0, linewidth=1)
    ax1.set_xlabel("Underlying price at expiry (S_T)")
    ax1.set_ylabel("Payoff")
    ax1.set_title(f"{opts['opt_type']} payoff at expiry (not discounted)")
    ax1.legend()
    st.pyplot(fig1, use_container_width=True)

    # Value vs spot chart
    st.markdown("### Option value today vs spot")
    call_vals, put_vals = [], []
    for s_ in S_grid:
        _in = OptionInput(
            S0=float(s_),
            K=opts["K"],
            r=opts["r"],
            sigma=opts["sigma"],
            T=opts["T"],
            q=opts["q"],
        )
        c_, p_, *_ = bs_prices(_in)
        call_vals.append(c_)
        put_vals.append(p_)

    fig2, ax2 = plt.subplots()
    if opts["opt_type"].lower() == "call":
        ax2.plot(S_grid, call_vals, label="Call value (today)")
        ax2.scatter([opts["S_for_options"]], [call], marker="o")
    else:
        ax2.plot(S_grid, put_vals, label="Put value (today)")
        ax2.scatter([opts["S_for_options"]], [put], marker="o")
    ax2.axvline(opts["S_for_options"], linestyle="--", linewidth=1, label="Current spot")
    ax2.set_xlabel("Spot (S₀,eff)")
    ax2.set_ylabel("Option value today")
    ax2.set_title(
        f"{opts['opt_type']} value vs effective spot (Black–Scholes, today’s price)"
    )
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)

# -------- Helper: compact summary for American vs European (CRR) --------
def summarize_crr_output(opt_type: str, euro_tree: float, amer_tree: float, bs_euro: float, steps: int):
    lines: list[str] = []

    # 1) CRR European vs BS
    diff_abs = euro_tree - bs_euro
    diff_rel = diff_abs / bs_euro if bs_euro != 0 else 0.0
    if abs(diff_rel) < 0.02:
        lines.append(
            f"CRR European ≈ {euro_tree:.4f} is **very close** to Black–Scholes {bs_euro:.4f} → the tree with {steps} steps is well aligned with the continuous model."
        )
    elif abs(diff_rel) < 0.05:
        lines.append(
            f"CRR European ≈ {euro_tree:.4f} vs Black–Scholes {bs_euro:.4f} → small deviation (~{diff_rel*100:.1f}%), acceptable but you can increase steps for a tighter match."
        )
    else:
        lines.append(
            f"CRR European ≈ {euro_tree:.4f} vs Black–Scholes {bs_euro:.4f} → larger deviation (~{diff_rel*100:.1f}%), consider using more steps or checking inputs."
        )

    # 2) Early-exercise premium
    ee_prem = amer_tree - bs_euro
    if abs(ee_prem) < 1e-6 or abs(ee_prem / bs_euro) < 0.005:
        lines.append(
            f"American {opt_type} ≈ {amer_tree:.4f} is almost the same as European → **early exercise adds almost no value** here."
        )
    else:
        lines.append(
            f"American {opt_type} ≈ {amer_tree:.4f} is higher than European by about {ee_prem:.4f} → **early exercise flexibility is worth something** in this setup."
        )

    # 3) Relative premium size
    if bs_euro > 0:
        prem_pct = ee_prem / bs_euro * 100
        lines.append(
            f"Early-exercise premium is about {prem_pct:.2f}% of the European price (vs Black–Scholes)."
        )

    # 4) Steps guidance
    if abs(diff_rel) > 0.05:
        lines.append(
            f"Because CRR vs BS differ noticeably, increasing steps above {steps} can reduce discretisation error."
        )
    else:
        lines.append(
            f"{steps} steps already give a reasonable approximation; increasing further mainly improves smoothness, not intuition."
        )

    return lines[:4]

# ===== TAB 2: American (CRR) =====
with tab2:
    opts = render_options_inputs("opt_inputs_tab2")
    st.subheader("American vs European (CRR Binomial)")
    st.caption("Discrete dividends are applied via S₀,eff in the underlying start node.")

    steps = st.slider(
        "CRR steps (accuracy vs speed)",
        min_value=25,
        max_value=1000,
        value=200,
        step=25,
        key="crr_steps",
    )

    inp = OptionInput(
        S0=opts["S_for_options"],
        K=opts["K"],
        r=opts["r"],
        sigma=opts["sigma"],
        T=opts["T"],
        q=opts["q"],
    )
    otype = opts["opt_type"].lower()

    # Tree prices
    euro_tree = crr_price_european(inp, otype, steps)
    amer_tree = crr_price_american(inp, otype, steps)

    # Continuous-time European benchmark
    bs_call, bs_put, _, _ = bs_prices(inp)
    bs_euro = bs_call if otype == "call" else bs_put

    c1, c2, c3 = st.columns(3)
    c1.metric("CRR European", f"{euro_tree:,.6f}")
    c2.metric("BS European", f"{bs_euro:,.6f}", delta=f"{(euro_tree - bs_euro):+.6f}")
    c3.metric("CRR American", f"{amer_tree:,.6f}")

    early_ex_premium = amer_tree - bs_euro
    st.caption(f"Early-exercise premium vs BS: {early_ex_premium:+.6f}.")

    # 🔰 Beginner-friendly, output-based analysis
    analysis_lines_crr = summarize_crr_output(
        opt_type=opts["opt_type"],
        euro_tree=euro_tree,
        amer_tree=amer_tree,
        bs_euro=bs_euro,
        steps=steps,
    )
    with st.expander("notes & interpretation (CRR vs BS)", expanded=True):
        for ln in analysis_lines_crr:
            st.markdown(f"- {ln}")

# ===== TAB 3: Chain & Surface =====
with tab3:
    import math
    from datetime import datetime
    from typing import Optional

    import numpy as np
    import pandas as pd
    import streamlit as st
    import altair as alt

    # ==============================
    # Black–Scholes + IV utilities
    # ==============================

    def _norm_cdf(x: float) -> float:
        """Standard normal CDF using math.erf (no SciPy)."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_price(
        S: float,
        K: float,
        r: float,
        q: float,
        T: float,
        sigma: float,
        option_type: str,
    ) -> float:
        """
        Black–Scholes price for a European call/put on index/stock with cont. dividend.
        option_type: 'C' or 'P' (case-insensitive).
        """
        option_type = option_type.upper()
        if T <= 0 or sigma <= 0:
            # fallback: intrinsic only
            if option_type == "C":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)

        # forward
        fwd = S * math.exp((r - q) * T)
        vol_sqrtT = sigma * math.sqrt(T)
        try:
            d1 = (math.log(fwd / K) + 0.5 * sigma * sigma * T) / vol_sqrtT
        except ValueError:
            # log domain errors, etc.
            return float("nan")
        d2 = d1 - vol_sqrtT

        if option_type == "C":
            # discounted forward call
            return math.exp(-r * T) * (fwd * _norm_cdf(d1) - K * _norm_cdf(d2))
        else:
            # put via call–put parity
            call = math.exp(-r * T) * (fwd * _norm_cdf(d1) - K * _norm_cdf(d2))
            return call - math.exp(-r * T) * (fwd - K)

    def implied_vol_bs(
        market_price: float,
        S: float,
        K: float,
        r: float,
        q: float,
        T: float,
        option_type: str,
        sigma_low: float = 1e-4,
        sigma_high: float = 10.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> float:
        """
        Robust bisection IV solver.

        - Rejects rows where price < intrinsic or <= 0.
        - Uses wide sigma bounds to allow very high IVs (~1000%+).
        - Returns np.nan if cannot find a solution.
        """
        option_type = option_type.upper()
        intrinsic = max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)

        # Basic sanity checks
        if market_price <= 0 or market_price < intrinsic:
            return np.nan

        # Price at low/high
        p_low = bs_price(S, K, r, q, T, sigma_low, option_type)
        p_high = bs_price(S, K, r, q, T, sigma_high, option_type)

        # If even huge sigma can't reach market_price -> unrealistic
        if np.isnan(p_low) or np.isnan(p_high) or p_high < market_price:
            return np.nan

        low, high = sigma_low, sigma_high
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            p_mid = bs_price(S, K, r, q, T, mid, option_type)

            if np.isnan(p_mid):
                return np.nan

            diff = p_mid - market_price
            if abs(diff) < tol:
                return mid

            if diff > 0:
                high = mid
            else:
                low = mid

        return mid  # best effort

    # ==============================
    # Beginner summary helper
    # ==============================

    def summarize_chain_iv_basic(df_valid: pd.DataFrame) -> list[str]:
        """
        Returns 3–4 short bullets explaining IV behaviour for the selected expiry.
        df_valid = rows with a valid IV column ('iv') for that expiry.
        """
        lines: list[str] = []
        if df_valid is None or df_valid.empty:
            return lines

        # Range of strikes
        k_min = df_valid["strike"].min()
        k_max = df_valid["strike"].max()
        lines.append(
            f"Strikes range from **{k_min:.2f} to {k_max:.2f}** for this expiry."
        )

        # IV levels (convert to %)
        iv_vals = df_valid["iv"] * 100.0
        iv_min, iv_max, iv_med = iv_vals.min(), iv_vals.max(), iv_vals.median()
        lines.append(
            f"Implied volatilities span **{iv_min:.1f}% to {iv_max:.1f}%**, with a median around {iv_med:.1f}%."
        )

        # Skew (low-strike vs high-strike IV)
        df_sorted = df_valid.sort_values("strike")
        q = max(1, len(df_sorted) // 4)
        low_iv = (df_sorted.head(q)["iv"] * 100.0).mean()
        high_iv = (df_sorted.tail(q)["iv"] * 100.0).mean()

        if low_iv > high_iv + 1.0:
            lines.append(
                f"Lower strikes have **higher IV** ({low_iv:.1f}% vs {high_iv:.1f}%) → downside is priced richer (put skew)."
            )
        elif high_iv > low_iv + 1.0:
            lines.append(
                f"Higher strikes have **higher IV** ({high_iv:.1f}% vs {low_iv:.1f}%) → upside calls trade richer."
            )
        else:
            lines.append(
                "IV curve is **fairly flat across strikes** → little skew for this expiry."
            )

        return lines[:4]

    # ==============================
    # UI + logic
    # ==============================

    st.subheader("Upload Option Chain CSV")

    st.markdown(
        """
        **Expected CSV columns (exact names):**
        - `strike`  
        - `type` (C/P or c/p)  
        - `price` (market option price)  
        - `spot`  
        - `rate` (risk-free, decimal)  
        - `dividend` (q, decimal)  
        - `ttm` (time to maturity in years)  
        - `expiry` (YYYY-MM-DD)
        """
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="chain_csv")

    if uploaded is None:
        st.info("Upload your CSV to see IV smiles and surfaces.")
    else:
        # ---- Read + normalise columns ----
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # standardise column names to lower
        df.columns = [c.strip().lower() for c in df.columns]

        required_cols = {"strike", "type", "price", "spot", "rate", "dividend", "ttm", "expiry"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"CSV missing required columns: {', '.join(sorted(missing))}")
            st.stop()

        # parse expiry to datetime
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")

        # clean type column
        df["type"] = df["type"].astype(str).str.strip().str.upper()
        df = df[df["type"].isin(["C", "P"])]

        # numeric types
        for col in ["strike", "price", "spot", "rate", "dividend", "ttm"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["strike", "price", "spot", "rate", "dividend", "ttm", "expiry"])

        if df.empty:
            st.error("No valid rows after cleaning. Check your CSV.")
            st.stop()

        st.write("### Cleaned option chain (all expiries)")
        st.dataframe(df.head(50), use_container_width=True)

        # ---- Expiry selection for IV smile ----
        expiries = sorted(df["expiry"].dropna().unique())
        if len(expiries) == 0:
            st.error("No valid expiry dates parsed from 'expiry' column.")
            st.stop()

        exp_choice = st.selectbox(
            "Select expiry for IV smile",
            options=expiries,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
        )

        df_exp = df[df["expiry"] == exp_choice].copy()
        st.write(f"### Chain for expiry {exp_choice.strftime('%Y-%m-%d')}")
        st.dataframe(df_exp, use_container_width=True)

        # ---- Compute IV for this expiry ----
        st.write("### Computing IV for selected expiry…")
        ivs = []
        for _, row in df_exp.iterrows():
            iv = implied_vol_bs(
                market_price=row["price"],
                S=row["spot"],
                K=row["strike"],
                r=row["rate"],
                q=row["dividend"],
                T=row["ttm"],
                option_type=row["type"],
            )
            ivs.append(iv)

        df_exp["iv"] = ivs
        df_valid = df_exp.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv"])

        st.write(f"Valid IV rows for this expiry: **{len(df_valid)} / {len(df_exp)}**")
        st.dataframe(df_valid, use_container_width=True)

        # ---- Plot IV smile for this expiry ----
        if df_valid.empty:
            st.warning("No valid IVs solved for this expiry. Check price vs intrinsic, ttm, etc.")
        else:
            smile_chart = (
                alt.Chart(df_valid)
                .mark_line(point=True)
                .encode(
                    x=alt.X("strike:Q", title="Strike"),
                    y=alt.Y("iv:Q", title="Implied Vol (decimal)"),
                    color=alt.Color("type:N", title="Type (C/P)"),
                    tooltip=["strike", "type", "price", "iv"],
                )
                .properties(width=700, height=400, title="IV Smile")
            )
            st.altair_chart(smile_chart, use_container_width=True)

            # 🔰 Beginner-friendly IV smile interpretation
            summary_lines = summarize_chain_iv_basic(df_valid)
            with st.expander("notes & interpretation (IV Smile)", expanded=True):
                for ln in summary_lines:
                    st.markdown(f"- {ln}")

        # ---- IV surface across ALL expiries ----
        st.write("### IV Surface Data (all expiries)")

        iv_all = []
        for _, row in df.iterrows():
            iv = implied_vol_bs(
                market_price=row["price"],
                S=row["spot"],
                K=row["strike"],
                r=row["rate"],
                q=row["dividend"],
                T=row["ttm"],
                option_type=row["type"],
            )
            iv_all.append(iv)

        df["iv"] = iv_all
        df_surface = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv"])

        st.write(f"Total rows with valid IV: **{len(df_surface)} / {len(df)}**")
        st.dataframe(df_surface.head(50), use_container_width=True)

        # Example 2D slice: IV vs TTM by moneyness bucket
        st.write("#### Example: IV vs TTM at ATM-ish strikes (for quick sanity check)")
        atm_mask = (df_surface["strike"] >= df_surface["spot"] * 0.98) & (
            df_surface["strike"] <= df_surface["spot"] * 1.02
        )
        df_atm = df_surface[atm_mask].copy()
        if not df_atm.empty:
            atm_chart = (
                alt.Chart(df_atm)
                .mark_point()
                .encode(
                    x=alt.X("ttm:Q", title="TTM (years)"),
                    y=alt.Y("iv:Q", title="ATM Implied Vol"),
                    color="type:N",
                    tooltip=["expiry", "strike", "iv"],
                )
                .properties(width=700, height=350)
            )
            st.altair_chart(atm_chart, use_container_width=True)
        else:
            st.info("No near-ATM points found for the surface preview.")

# -------- Helper: Beginner-friendly summary for Bonds (Tab 4) --------
def summarize_bond_output(face, coupon_pct, ytm, price, mac, mod, conv, mode_label):
    """
    Returns 3–4 short, high-signal bullets explaining what the bond output means.
    """

    lines = []

    # 1) Premium/Par/Discount classification
    if abs(price - face) < 1e-6 or abs((price - face) / face) < 0.005:
        lines.append(
            f"The bond is trading at **par** (Price ≈ Face). Coupon rate ≈ market yield."
        )
    elif price > face:
        lines.append(
            f"The bond trades at a **premium** (Price {price:.2f} > Face {face:.2f}) → coupon is above market yields."
        )
    else:
        lines.append(
            f"The bond trades at a **discount** (Price {price:.2f} < Face {face:.2f}) → coupon is below market yields."
        )

    # 2) Yield meaning
    if ytm is not None:
        lines.append(
            f"YTM ≈ **{ytm * 100:.3f}%** → approximate annual return if held to maturity assuming coupon reinvestment at YTM."
        )

    # 3) Duration interpretation
    if mod is not None:
        if mod < 3:
            risk_label = "low interest-rate sensitivity"
        elif mod < 7:
            risk_label = "moderate interest-rate sensitivity"
        else:
            risk_label = "high interest-rate sensitivity"

        lines.append(
            f"Modified duration ≈ **{mod:.3f}** → a 1% move in yields → price changes by ~{mod:.2f}%. ({risk_label})"
        )

    # 4) Convexity interpretation
    if conv is not None:
        lines.append(
            f"Convexity ≈ **{conv:.2f}** → improves accuracy of duration-based estimates for larger rate moves."
        )

    return lines[:4]

# ===== TAB 4: Bonds (numeric & dates) =====
with tab4:
    st.subheader("Bond Pricer — Price, YTM, Duration & Convexity")
    mode_bond_input = st.radio("Input mode", ["Numeric T", "Dates"], horizontal=True)

    bc1, bc2, bc3 = st.columns(3)
    face = bc1.number_input("Face/Redemption", min_value=1.0, value=100.0, step=1.0)
    coupon_pct = bc2.number_input("Coupon rate (% p.a.)", value=5.00, step=0.25, format="%.2f") / 100.0
    freq = int(bc3.selectbox("Payments per year", options=[1, 2, 4], index=1))

    # ======================================
    # ===== Numeric T branch =====
    # ======================================
    if mode_bond_input == "Numeric T":
        bc4, bc5 = st.columns(2)
        T_years = bc4.number_input("Time to maturity (years)", min_value=0.25, value=5.0, step=0.25)
        ytm_pct = bc5.number_input("Yield to maturity (% p.a.)", value=6.00, step=0.10, format="%.2f") / 100.0

        mode_bond = st.radio("Mode", ["Inputs → Price", "Price → YTM"], horizontal=True)

        # ----- Inputs → Price -----
        if mode_bond == "Inputs → Price":
            P = price_bond(face, coupon_pct, ytm_pct, T_years, freq)
            mac = macaulay_duration(face, coupon_pct, ytm_pct, T_years, freq)
            mod = modified_duration(face, coupon_pct, ytm_pct, T_years, freq)
            conv = convexity_numeric(face, coupon_pct, ytm_pct, T_years, freq)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{P:,.4f}")
            c2.metric("Macaulay Duration (yrs)", f"{mac:,.4f}")
            c3.metric("Modified Duration (yrs)", f"{mod:,.4f}")
            c4.metric("Convexity", f"{conv:,.4f}")

            # ---- Beginner analysis ----
            summary_lines = summarize_bond_output(
                face=face,
                coupon_pct=coupon_pct,
                ytm=ytm_pct,
                price=P,
                mac=mac,
                mod=mod,
                conv=conv,
                mode_label="Inputs → Price (Numeric T)",
            )
            with st.expander("Beginner notes & interpretation", expanded=True):
                for ln in summary_lines:
                    st.markdown(f"- {ln}")

        # ----- Price → YTM -----
        else:
            target_price = st.number_input("Target clean price", min_value=0.01, value=100.00, step=0.25)
            ytm_solved = ytm_from_price(target_price, face, coupon_pct, T_years, freq)

            if ytm_solved is None:
                st.error("Could not solve YTM in bounds [-99%, 100%]. Check inputs.")
            else:
                mac = macaulay_duration(face, coupon_pct, ytm_solved, T_years, freq)
                mod = modified_duration(face, coupon_pct, ytm_solved, T_years, freq)
                conv = convexity_numeric(face, coupon_pct, ytm_solved, T_years, freq)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Solved YTM", f"{ytm_solved * 100:.4f}%")
                c2.metric("Macaulay Duration (yrs)", f"{mac:,.4f}")
                c3.metric("Modified Duration (yrs)", f"{mod:,.4f}")
                c4.metric("Convexity", f"{conv:,.4f}")

                # ---- Beginner interpretation ----
                summary_lines = summarize_bond_output(
                    face=face,
                    coupon_pct=coupon_pct,
                    ytm=ytm_solved,
                    price=target_price,
                    mac=mac,
                    mod=mod,
                    conv=conv,
                    mode_label="Price → YTM (Numeric T)",
                )
                with st.expander("Beginner notes & interpretation", expanded=True):
                    for ln in summary_lines:
                        st.markdown(f"- {ln}")

    # ======================================
    # ===== Dates branch =====
    # ======================================
    else:
        colD1, colD2, colD3 = st.columns(3)
        val_d_str = colD1.text_input("Settlement / Valuation date (YYYY-MM-DD)", value=str(date.today()))
        mat_d_str = colD2.text_input(
            "Maturity date (YYYY-MM-DD)",
            value=str(date.today().replace(year=date.today().year + 5)),
        )
        dc_bond: DayCount = colD3.selectbox(
            "Day-count",
            ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"],
            index=0,
        )
        biz: BizConv = st.selectbox(
            "Business-day convention",
            ["Following", "Modified Following", "Preceding"],
            index=0,
        )

        try:
            settle_d = parse_date(val_d_str)
            mat_d = parse_date(mat_d_str)
        except Exception:
            st.error("Invalid date format. Use YYYY-MM-DD.")
            settle_d, mat_d = date.today(), date.today()

        ytm_pct_d = st.number_input(
            "Yield to maturity (% p.a.)", value=6.00, step=0.10, format="%.2f"
        ) / 100.0
        mode_bond_d = st.radio("Mode (dates)", ["Inputs → Price", "Price → YTM"], horizontal=True)

        # ----- Inputs → Price (Dates) -----
        if mode_bond_d == "Inputs → Price":
            P = price_bond_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)
            mac = macaulay_duration_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)
            mod = modified_duration_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)

            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"{P:,.4f}")
            c2.metric("Macaulay Duration (yrs)", f"{mac:,.4f}")
            c3.metric("Modified Duration (yrs)", f"{mod:,.4f}")

            # ---- Beginner interpretation ----
            summary_lines = summarize_bond_output(
                face=face,
                coupon_pct=coupon_pct,
                ytm=ytm_pct_d,
                price=P,
                mac=mac,
                mod=mod,
                conv=None,
                mode_label="Inputs → Price (Dates)",
            )
            with st.expander("Beginner notes & interpretation", expanded=True):
                for ln in summary_lines:
                    st.markdown(f"- {ln}")

        # ----- Price → YTM (Dates) -----
        else:
            target_price = st.number_input("Target clean price (dates)", min_value=0.01, value=100.00, step=0.25)
            ytm_solved = ytm_from_price_dates(
                target_price, face, coupon_pct, settle_d, mat_d, freq, dc_bond, biz
            )

            if ytm_solved is None:
                st.error("Could not solve YTM in bounds [-99%, 100%]. Check inputs.")
            else:
                mac = macaulay_duration_dates(face, coupon_pct, ytm_solved, settle_d, mat_d, freq, dc_bond, biz)
                mod = modified_duration_dates(face, coupon_pct, ytm_solved, settle_d, mat_d, freq, dc_bond, biz)

                c1, c2, c3 = st.columns(3)
                c1.metric("Solved YTM", f"{ytm_solved * 100:.4f}%")
                c2.metric("Macaulay Duration (yrs)", f"{mac:,.4f}")
                c3.metric("Modified Duration (yrs)", f"{mod:,.4f}")

                # ---- Beginner interpretation ----
                summary_lines = summarize_bond_output(
                    face=face,
                    coupon_pct=coupon_pct,
                    ytm=ytm_solved,
                    price=target_price,
                    mac=mac,
                    mod=mod,
                    conv=None,
                    mode_label="Price → YTM (Dates)",
                )
                with st.expander("Beginner notes & interpretation", expanded=True):
                    for ln in summary_lines:
                        st.markdown(f"- {ln}")

# -------- Helper: Beginner-friendly summary for Yield Curve (Tab 5) --------
def summarize_yield_curve(curve) -> list[str]:
    """
    Build 3–4 short bullets explaining the shape and key points
    of the bootstrapped zero curve.
    Expects a YieldCurve with attributes: mats, zeros, dfs.
    """
    lines: list[str] = []
    if curve is None or len(curve.mats) == 0:
        return lines

    mats = list(curve.mats)
    zeros = list(curve.zeros)

    # 1) Short vs long end (slope)
    short_T, long_T = mats[0], mats[-1]
    short_r, long_r = zeros[0] * 100, zeros[-1] * 100

    if long_r > short_r + 0.10:
        slope = "upward-sloping"
        msg = "longer maturities yield more than short ones."
    elif long_r + 0.10 < short_r:
        slope = "inverted"
        msg = "long-term yields are **below** short-term yields (often seen as a stress/recession signal)."
    else:
        slope = "fairly flat"
        msg = "little difference between short and long rates."

    lines.append(
        f"Curve is **{slope}**: short end ≈ {short_r:.2f}% at T={short_T:.2f} yrs, long end ≈ {long_r:.2f}% at T={long_T:.2f} yrs — {msg}"
    )

    # 2) Mid-point rate
    if len(mats) >= 3:
        mid_i = len(mats) // 2
        lines.append(
            f"Mid-horizon rate around T ≈ {mats[mid_i]:.2f} yrs is ≈ {zeros[mid_i] * 100:.2f}%, between the short and long ends."
        )

    # 3) Overall level
    avg_r = sum(zeros) / len(zeros) * 100
    lines.append(
        f"Average level of rates across the curve is ≈ {avg_r:.2f}%."
    )

    # 4) Hint about forward rates
    lines.append(
        "Use the forward-rate box below to see the implied short-term rate between any two maturities T₁ and T₂."
    )

    return lines[:4]

# ===== TAB 5: Yield Curve =====
with tab5:
    st.subheader("Yield Curve — Bootstrapping")

    curve_mode = st.radio("Curve input mode", ["Numeric", "Dates"], horizontal=True)
    deposits, bonds = [], []

    if curve_mode == "Numeric":
        st.markdown("**Deposits (T, Rate%)**")
        dep_df = st.data_editor(
            pd.DataFrame(
                {
                    "T": [0.5, 1.0, 2.0],
                    "Rate (%)": [3.0, 3.2, 3.5],
                }
            ),
            num_rows="dynamic",
            use_container_width=True,
            key="dep_num",
        )
        for _, r0 in dep_df.iterrows():
            try:
                deposits.append((float(r0["T"]), float(r0["Rate (%)"]) / 100.0))
            except Exception:
                pass

        st.markdown("**Coupon Bonds**")
        bond_df = st.data_editor(
            pd.DataFrame(
                {
                    "T": [3.0],
                    "Price": [99.50],
                    "Coupon (%)": [4.0],
                    "Face": [100.0],
                    "Freq": [2],
                }
            ),
            num_rows="dynamic",
            use_container_width=True,
            key="bond_num",
        )
        for _, r0 in bond_df.iterrows():
            try:
                bonds.append(
                    {
                        "T": float(r0["T"]),
                        "price": float(r0["Price"]),
                        "coupon": float(r0["Coupon (%)"]) / 100.0,
                        "face": float(r0["Face"]),
                        "freq": int(r0["Freq"]),
                    }
                )
            except Exception:
                pass

        if st.button("Bootstrap Curve (Numeric)", type="primary"):
            try:
                curve = cached_bootstrap_curve(deposits, bonds)
                st.session_state["bootstrapped_curve"] = curve
                st.success("Curve bootstrapped.")
            except Exception as e:
                st.error(f"Bootstrapping error: {e}")
    else:
        dc_curve: DayCount = st.selectbox(
            "Day-count for T",
            ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"],
            index=0,
        )
        val_date_str2 = st.text_input(
            "Valuation date (YYYY-MM-DD)",
            value=str(date.today()),
            key="curve_valdate",
        )
        try:
            curve_val_date = parse_date(val_date_str2)
        except Exception:
            st.error("Invalid valuation date; using today.")
            curve_val_date = date.today()

        st.markdown("**Deposits (Maturity Date, Rate%)**")
        dep_df2 = st.data_editor(
            pd.DataFrame(
                {
                    "Maturity": [str(curve_val_date)],
                    "Rate (%)": [3.0],
                }
            ),
            num_rows="dynamic",
            use_container_width=True,
            key="dep_dt",
        )
        for _, r0 in dep_df2.iterrows():
            try:
                deposits.append(
                    {
                        "date": parse_date(str(r0["Maturity"])),
                        "rate": float(r0["Rate (%)"]) / 100.0,
                    }
                )
            except Exception:
                pass

        st.markdown("**Coupon Bonds**")
        bond_df2 = st.data_editor(
            pd.DataFrame(
                {
                    "Maturity": [str(curve_val_date)],
                    "Price": [99.50],
                    "Coupon (%)": [4.0],
                    "Face": [100.0],
                    "Freq": [2],
                }
            ),
            num_rows="dynamic",
            use_container_width=True,
            key="bond_dt",
        )
        for _, r0 in bond_df2.iterrows():
            try:
                bonds.append(
                    {
                        "maturity_date": parse_date(str(r0["Maturity"])),
                        "price": float(r0["Price"]),
                        "coupon": float(r0["Coupon (%)"]) / 100.0,
                        "face": float(r0["Face"]),
                        "freq": int(r0["Freq"]),
                        "day_count": dc_curve,
                    }
                )
            except Exception:
                pass

        if st.button("Bootstrap Curve (Dates)", type="primary"):
            try:
                curve = cached_bootstrap_curve(deposits, bonds, valuation_date=curve_val_date, day_count=dc_curve)
                st.session_state["bootstrapped_curve"] = curve
                st.success("Curve bootstrapped.")
            except Exception as e:
                st.error(f"Bootstrapping error: {e}")

    curve: YieldCurve | None = st.session_state.get("bootstrapped_curve")
    if curve is not None:
        st.write("**Zero Curve**")
        st.dataframe(
            curve.as_dataframe().style.format({"Zero(%)": "{:.3f}", "DF": "{:.6f}"}),
            use_container_width=True,
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.plot(curve.mats, [z * 100 for z in curve.zeros], marker="o")
        ax1.set_title("Spot (Zero) Curve")
        ax1.set_xlabel("T (years)")
        ax1.set_ylabel("Zero (%)")
        ax2.plot(curve.mats, curve.dfs, marker="o")
        ax2.set_title("Discount Factors")
        ax2.set_xlabel("T (years)")
        ax2.set_ylabel("DF")
        st.pyplot(fig, use_container_width=True)

        colf1, colf2 = st.columns(2)
        t1 = colf1.number_input("Forward start T₁ (years)", value=1.0, step=0.25, min_value=0.0)
        t2 = colf2.number_input("Forward end T₂ (years)", value=2.0, step=0.25, min_value=0.01)
        st.metric("Annual forward rate T₁ → T₂", f"{curve.get_fwd(t1, t2) * 100:.3f}%")

        # 🔰 Beginner-friendly interpretation of the curve
        summary_lines_curve = summarize_yield_curve(curve)
        with st.expander("notes & interpretation (Yield Curve)", expanded=True):
            for ln in summary_lines_curve:
                st.markdown(f"- {ln}")



# ===== TAB 6: Swaps =====
with tab6:
    st.subheader("Plain-Vanilla Interest Rate Swap")

    curve: YieldCurve | None = st.session_state.get("bootstrapped_curve")
    if curve is None:
        st.info("Bootstrap a curve first in the Yield Curve tab to use swaps.")
    else:
        colS1, colS2, colS3 = st.columns(3)
        notional = colS1.number_input("Notional", min_value=1_000.0, value=1_000_000.0, step=50_000.0)
        T_swap = colS2.number_input("Swap maturity T (years)", min_value=0.5, value=5.0, step=0.5)
        freq = int(colS3.selectbox("Fixed leg payments per year", options=[1, 2, 4], index=1))

        s_par = par_swap_rate(curve, T_swap, freq)
        st.metric("Par swap fixed rate", f"{s_par * 100:.4f}%")
        fixed_rate_user = (
            st.number_input(
                "Fixed rate to value (% p.a.)",
                value=round(s_par * 100, 4),
                step=0.01,
                format="%.4f",
            )
            / 100.0
        )
        st.metric(
            "Swap PV (Fixed - Float)",
            f"{swap_pv(curve, T_swap, fixed_rate_user, notional, freq):,.2f}",
        )
        st.metric(
            "DV01 (per +1bp parallel shift)",
            f"{dv01(curve, T_swap, fixed_rate_user, notional, freq, bp=1.0):,.2f}",
        )


# ===== TAB 7: Futures & Black-76 (curve-aware) =====
with tab7:
    st.subheader("Futures/Forwards & Black-76 Options")

    opts = render_options_inputs("opt_inputs_tab8")

    curve: YieldCurve | None = st.session_state.get("bootstrapped_curve")
    disc_src = st.radio(
        "Discounting source",
        ["Flat r (continuous)", "From curve (DF(T))"],
        horizontal=True,
    )

    colF0, colF1, colF2 = st.columns(3)
    mode_fut = colF0.radio(
        "Forward input",
        ["Compute F₀ from S₀, r, q", "Enter F₀ directly"],
        horizontal=False,
    )
    K_fut = colF1.number_input(
        "Strike (K)",
        min_value=0.0001,
        value=float(opts["K"]),
        step=1.0,
        key="b76_k",
    )
    T_fut = colF2.number_input(
        "Time to expiry T (years, futures option)",
        min_value=0.00274,
        value=float(opts["T"]),
        step=0.01,
        help="0.00274 ≈ 1 day.",
        key="b76_T",
    )

    if mode_fut.startswith("Compute"):
        if disc_src == "From curve (DF(T))" and curve is not None:
            q_flat = (
                st.number_input(
                    "Dividend / convenience yield q (% per year, continuous)",
                    min_value=0.0,
                    value=float(opts["q"] * 100),
                    step=0.25,
                    format="%.2f",
                    key="q_curve",
                )
                / 100.0
            )
            F0 = forward_price_from_curve(opts["S0"], curve, T_fut, q_cont=q_flat)
        else:
            F0 = forward_price(opts["S0"], opts["r"], opts["q"], T_fut)
    else:
        F0 = colF0.number_input(
            "F₀ (futures/forward)",
            min_value=0.0001,
            value=float(forward_price(opts["S0"], opts["r"], opts["q"], T_fut)),
            step=0.1,
            key="F0_direct",
        )

    st.metric("F₀ (futures/forward)", f"{F0:,.4f}")

    sigma_b76 = (
        st.number_input(
            "Volatility σ (% per year)",
            min_value=0.01,
            value=float(opts["sigma"] * 100),
            step=0.5,
            format="%.2f",
            key="b76_sigma",
        )
        / 100.0
    )
    otype_b76 = st.radio(
        "Option type (on futures)",
        ["Call", "Put"],
        horizontal=True,
        key="b76_otype",
    ).lower()

    DF_T = None
    if disc_src == "From curve (DF(T))":
        if curve is None:
            st.info("No curve in session. Bootstrap one in the Yield Curve tab, or switch to Flat r.")
        else:
            DF_T = curve.get_df(T_fut)
            st.metric("DF(T) from curve", f"{DF_T:.6f}")

    st.markdown("**Implied Volatility (from market price)**")
    mkt_b76 = st.number_input(
        "Market option price (any currency)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="b76_mkt",
    )
    if mkt_b76 > 0:
        iv76 = implied_vol_b76(
            mkt_b76,
            F0,
            K_fut,
            r=opts["r"],
            T=T_fut,
            otype=otype_b76,
            DF_T=DF_T,
        )
        if iv76 is None:
            st.error("No valid IV found in [1e-6, 5.0] for these inputs.")
        else:
            st.metric("Black-76 IV", f"{iv76 * 100:.3f}%")

    if DF_T is None:
        price_b76 = black76_price(F0, K_fut, opts["r"], T_fut, sigma_b76, otype_b76)
        G76 = black76_greeks(F0, K_fut, opts["r"], T_fut, sigma_b76)
        delta_disp = G76["delta_fut"]["call"] if otype_b76 == "call" else G76["delta_fut"]["put"]
        gamma_disp = G76["gamma_fut"]
        vega_disp = G76["vega"]
        theta_day = G76["theta_per_day"]
        rho_disp = G76["rho"]
    else:
        price_b76 = black76_price_df(F0, K_fut, DF_T, T_fut, sigma_b76, otype_b76)
        G76 = black76_greeks_df(F0, K_fut, DF_T, T_fut, sigma_b76)
        delta_disp = G76["delta_fut"]["call"] if otype_b76 == "call" else G76["delta_fut"]["put"]
        gamma_disp = G76["gamma_fut"]
        vega_disp = G76["vega"]
        theta_day = G76["theta_per_day"]
        rho_disp = G76["rho_df"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Option price", f"{price_b76:,.4f}")
    c2.metric("Δ (futures)", f"{delta_disp:.6f}")
    c3.metric("Γ (futures)", f"{gamma_disp:.6f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Vega (per 1.0 σ)", f"{vega_disp:.6f}")
    c5.metric("Theta/day", f"{theta_day:.6f}")
    c6.metric(
        ("Rho (per 1.0 r)" if DF_T is None else "Sensitivity to DF(T)"),
        f"{rho_disp:.6f}",
    )

    st.subheader("Charts")
    F_grid = np.linspace(max(0.01, F0 * 0.4), F0 * 1.6, 200)
    payoff_call = np.maximum(F_grid - K_fut, 0.0)
    payoff_put = np.maximum(K_fut - F_grid, 0.0)
    figp, axp = plt.subplots()
    axp.plot(F_grid, payoff_call if otype_b76 == "call" else payoff_put,
             label=f"{otype_b76.capitalize()} payoff at expiry")
    axp.axhline(0, linewidth=1)
    axp.set_xlabel("F_T")
    axp.set_ylabel("Payoff")
    axp.legend()
    st.pyplot(figp, use_container_width=True)

    F0_grid = np.linspace(max(0.01, F0 * 0.6), F0 * 1.4, 120)
    if DF_T is None:
        values = [black76_price(fv, K_fut, opts["r"], T_fut, sigma_b76, otype_b76) for fv in F0_grid]
    else:
        values = [black76_price_df(fv, K_fut, DF_T, T_fut, sigma_b76, otype_b76) for fv in F0_grid]
    figv, axv = plt.subplots()
    axv.plot(F0_grid, values, label="Option value today")
    axv.scatter([F0], [price_b76], marker="o")
    axv.set_xlabel("F₀")
    axv.set_ylabel("Option value")
    axv.set_title("Option value vs F₀")
    axv.legend()
    st.pyplot(figv, use_container_width=True)


# ===== TAB 8: Reports & Export =====
with tab8:
    st.subheader("Reports & Export")

    base_opts = render_options_inputs("opt_inputs_reports")

    oi = OptionInput(
        S0=base_opts["S_for_options"],
        K=base_opts["K"],
        r=base_opts["r"],
        sigma=base_opts["sigma"],
        T=base_opts["T"],
        q=base_opts["q"],
    )
    call_px, put_px, d1, d2 = bs_prices(oi)
    G = bs_greeks(oi)

    curve: YieldCurve | None = st.session_state.get("bootstrapped_curve")
    curve_df = curve.as_dataframe() if curve is not None else None

    metrics = {
        "valuation_date": str(base_opts["val_date"]),
        "option_type": base_opts["opt_type"].lower(),
        "S0": base_opts["S0"],
        "S0_eff": base_opts["S_for_options"],
        "K": base_opts["K"],
        "T_years": base_opts["T"],
        "r_cont": base_opts["r"],
        "q_cont": base_opts["q"],
        "sigma": base_opts["sigma"],
        "call_price": call_px,
        "put_price": put_px,
        "delta_call": G["delta"]["call"],
        "delta_put": G["delta"]["put"],
        "gamma": G["gamma"],
        "vega_per_1pct": G["vega_per_1pct"],
        "theta_day_call": G["theta_per_day"]["call"],
        "theta_day_put": G["theta_per_day"]["put"],
        "rho_call": G["rho"]["call"],
        "rho_put": G["rho"]["put"],
        "d1": d1,
        "d2": d2,
    }

    colR1, colR2 = st.columns(2)
    if colR1.button("Generate Options Report (PDF)", type="primary"):
        pdf_bytes = build_options_report_pdf(
            S0=base_opts["S0"],
            S0_eff=base_opts["S_for_options"],
            K=base_opts["K"],
            r=base_opts["r"],
            q=base_opts["q"],
            T=base_opts["T"],
            sigma=base_opts["sigma"],
            opt_type=base_opts["opt_type"],
            valuation_date_str=str(base_opts["val_date"]),
            curve_df=curve_df,
        )
        st.download_button(
            "Download options_report.pdf",
            data=pdf_bytes,
            file_name="options_report.pdf",
            mime="application/pdf",
        )

    if colR2.button("Export Key Metrics (CSV)"):
        csv_bytes = build_metrics_csv(metrics)
        st.download_button(
            "Download metrics.csv",
            data=csv_bytes,
            file_name="metrics.csv",
            mime="text/csv",
        )
