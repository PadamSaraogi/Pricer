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

# forwards & Black-76
from forwards import forward_price, forward_price_from_curve
from black76 import (
    black76_price,
    black76_greeks,
    implied_vol_b76,
    black76_price_df,
    black76_greeks_df,
)

# discrete dividends
from dividends import CashDividend, spot_adjusted_for_dividends

# reports
from reports import build_options_report_pdf, build_metrics_csv


# ---------------------------
# Micro-caching wrappers
# ---------------------------
@st.cache_data(show_spinner=False, ttl=300)
def cached_bootstrap_curve(
    deposits, bonds, valuation_date: date | None = None, day_count: str = "ACT/365F"
):
    from yield_curve import bootstrap_curve as _bootstrap_curve

    return _bootstrap_curve(
        deposits, bonds, valuation_date=valuation_date, day_count=day_count
    )


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
        raw_df,
        S0=S0,
        r=r,
        q=q,
        default_T=default_T,
        default_type=default_type,
        sigma_seed=sigma_seed,
    )


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Options & Bonds Pricer (Beginner + Pro)", page_icon="📊", layout="wide"
)

st.title("📊 Options & Bonds Pricer (Beginner + Pro)")
st.caption(
    "Price options and bonds with plain-English summaries, plus advanced tools for IV surfaces, yield curves, swaps, and futures."
)


# ========== Shared UI helper for Options inputs (inside tabs) ==========
def render_options_inputs(defaults_key: str = "opt_defaults"):
    """
    Renders options inputs INSIDE the current tab (no sidebar).

    Returns dict:
      S0, K, r, sigma, T, q, opt_type, dc, val_date, S_for_options
    """
    from datetime import date as _date
    import pandas as _pd
    from daycount import DayCount as _DC, year_fraction as _yf
    from date_utils import parse_date as _pdate
    from dividends import CashDividend as _CashDiv, spot_adjusted_for_dividends as _spot_adj

    # Initialise defaults in session_state
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

    # ---- BASIC INPUTS (expander 1) ----
    with st.expander("⚙️ Options Inputs", expanded=True):
        st.markdown("**Basic inputs** – start by changing only these.")

        # Spot & strike
        colA, colB = st.columns(2)
        S0 = colA.number_input(
            "Spot price (S₀)",
            min_value=0.01,
            value=float(s["S0"]),
            step=1.0,
            key=key("S0"),
            help="Current underlying price.",
        )
        K = colB.number_input(
            "Strike price (K)",
            min_value=0.01,
            value=float(s["K"]),
            step=1.0,
            key=key("K"),
            help="Price at which you can buy (call) or sell (put) at expiry.",
        )
        s["S0"], s["K"] = S0, K

        # Rate & volatility
        col1, col2 = st.columns(2)
        r = (
            col1.number_input(
                "Risk-free interest rate (% per year)",
                value=float(s["r"] * 100),
                step=0.25,
                format="%.2f",
                key=key("r_pct"),
                help="Approximate risk-free annual rate (continuous compounding under the hood).",
            )
            / 100.0
        )
        sigma = (
            col2.number_input(
                "Volatility (% per year)",
                min_value=0.01,
                value=float(s["sigma"] * 100),
                step=0.5,
                format="%.2f",
                key=key("sigma_pct"),
                help="How much the price typically moves in a year. Example: 20 = 20% volatility.",
            )
            / 100.0
        )
        s["r"], s["sigma"] = r, sigma

        # Time & dividend yield
        col3, col4 = st.columns(2)
        T_basic = col3.number_input(
            "Time to expiry (years)",
            min_value=0.00274,
            value=float(s["T"]),
            step=0.01,
            help="Example: 0.5 ≈ 6 months, 0.00274 ≈ 1 day.",
            key=key("T_years"),
        )
        q = (
            col4.number_input(
                "Dividend yield (% per year)",
                value=float(s["q"] * 100),
                step=0.25,
                format="%.2f",
                key=key("q_pct_simple"),
                help="Flat annual dividend or foreign interest rate (continuous).",
            )
            / 100.0
        )
        s["T"], s["q"] = T_basic, q

        opt_type = st.radio(
            "Option type",
            ["Call", "Put"],
            horizontal=True,
            index=0 if s["opt_type"] == "Call" else 1,
            key=key("opt_type"),
            help="Call benefits if price goes up; Put benefits if price goes down.",
        )
        s["opt_type"] = opt_type

    # Defaults for advanced if user doesn't touch it
    dc: _DC = "ACT/365F"
    val_date = _date.today()
    T_eff = T_basic
    S_eff = S0

    # ---- ADVANCED INPUTS (expander 2, not nested) ----
    with st.expander("Advanced settings (dates, day-counts, dividends)", expanded=False):
        # Use dates vs T
        use_dates = st.checkbox(
            "Use dates instead of 'Time to expiry (years)'",
            value=s["use_dates"],
            key=key("use_dates_checkbox"),
        )
        s["use_dates"] = use_dates

        if use_dates:
            dc = st.selectbox(
                "Day-count convention (for T)",
                ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"],
                index=["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"].index(
                    s["dc"]
                ),
                key=key("dc_select"),
            )

            val_date_str = st.text_input(
                "Valuation date (YYYY-MM-DD)",
                value=s["val_date"],
                key=key("val_date"),
            )
            expiry_str = st.text_input(
                "Option expiry date (YYYY-MM-DD)",
                value=s["expiry"],
                key=key("expiry"),
            )

            try:
                val_date = _pdate(val_date_str)
                exp_date = _pdate(expiry_str)
                T_eff = _yf(val_date, exp_date, dc)
            except Exception:
                val_date = _date.today()
                T_eff = 0.0

            s["dc"], s["val_date"], s["expiry"] = dc, val_date_str, expiry_str

        # Discrete cash dividends
        st.markdown("---")
        use_disc_div = st.checkbox(
            "Enable discrete **cash** dividends",
            value=s["use_disc_div"],
            key=key("use_disc_div"),
        )
        s["use_disc_div"] = use_disc_div

        div_dc: _DC = st.selectbox(
            "Day-count for dividends",
            ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"],
            index=["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"].index(
                s["div_dc"]
            ),
            disabled=not use_disc_div,
            key=key("div_dc"),
        )
        s["div_dc"] = div_dc

        if use_disc_div:
            st.caption(
                "Add future cash dividends (per share) that occur **after valuation date and before expiry**."
            )
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
                    dpay = _pdate(str(row["Pay Date (YYYY-MM-DD)"]))
                    amt = float(row["Amount"])
                    if amt > 0 and dpay > val_date:
                        dividends_list.append(
                            _CashDiv(pay_date=dpay, amount=amt)
                        )
                except Exception:
                    pass

            if len(dividends_list) > 0:
                S_eff_calc = _spot_adj(
                    S0, dividends_list, r_cont=r, valuation_date=val_date, dc=div_dc
                )
                pv_div = S0 - S_eff_calc
                S_eff = S_eff_calc
                st.info(
                    f"S₀,eff = S₀ − PV(dividends) = {S0:.4f} − {pv_div:.4f} = **{S_eff:.4f}**"
                )

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "🎯 Option Pricer",
        "🧮 American (CRR)",
        "📊 Chain & Smiles",
        "🌈 Vol Surface",
        "💰 Bonds",
        "📈 Yield Curve",
        "🔁 Swaps",
        "📃 Futures & Black-76",
        "📄 Reports & Export",
    ]
)

# ===== TAB 1: Option Pricer (BSM) =====
with tab1:
    opts = render_options_inputs("opt_inputs_tab1")

    st.subheader("European Options — Black–Scholes")
    st.markdown(
        """
**How to use this tab (beginner):**
1. Change the **Basic inputs** in the box above (spot, strike, time, rate, volatility).
2. See **Call/Put prices** and the **moneyness tag**.
3. Scroll down to view **Greeks** and charts to understand sensitivity and payoff.
"""
    )

    colm1, colm2, _ = st.columns(3)
    bid = colm1.number_input(
        "Bid (optional, for IV)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="bid1",
    )
    ask = colm2.number_input(
        "Ask (optional, for IV)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="ask1",
    )
    mid = (
        0.5 * (bid + ask)
        if (bid > 0 and ask > 0 and ask >= bid)
        else None
    )
    mkt_price = st.number_input(
        "Single market option price (for IV)",
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

    # Basic metrics row: prices + moneyness
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Call price", f"{call:,.4f}")
    m2.metric("Put price", f"{put:,.4f}")
    m3.metric("S/K", tag_info["S_over_K"])
    m4.metric("Moneyness tag", tag_info["tag"])

    # Moneyness details & interpretation
    with st.expander("Moneyness details & diagnostics", expanded=False):
        mtc1, mtc2, mtc3 = st.columns(3)
        mtc1.metric("log-moneyness", tag_info["log_moneyness"])
        mtc2.metric("d₁", tag_info["d1"])
        mtc3.metric("d₂", f"{d2:.4f}")

    st.markdown("### 📌 Interpretation (plain English)")
    delta_call = G["delta"]["call"]
    vega_per_1pct = G["vega_per_1pct"]

    lines = []
    if tag_info["tag"].startswith("ITM"):
        lines.append("• Your option is **in the money** – the strike is favourable at current price.")
    elif tag_info["tag"].startswith("OTM"):
        lines.append(
            "• Your option is **out of the money** – it currently only has time value."
        )
    else:
        lines.append(
            "• Your option is **at the money** – the strike is close to the current price."
        )

    lines.append(
        f"• **Delta (Call)** ≈ {delta_call:.3f}: if the underlying moves by ₹1, the call changes by about {delta_call:.3f} × ₹1."
    )
    lines.append(
        f"• **Vega** ≈ {vega_per_1pct:.4f}: option price change for a 1% change in volatility."
    )
    st.info("\n".join(lines))

    # Greeks: basic vs advanced
    st.markdown("### Sensitivities (Greeks)")

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

    with st.expander("Show all Greeks (advanced)", expanded=False):
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

    # Implied vol from market price
    st.markdown("### Implied Volatility (from market price)")
    coliv1, coliv2 = st.columns(2)
    otype = opts["opt_type"].lower()
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

    # Charts
    st.markdown("### Visuals")

    S_min = max(0.01, opts["S_for_options"] * 0.4)
    S_max = opts["S_for_options"] * 1.6
    S_grid = np.linspace(S_min, S_max, 200)

    # Payoff at expiry
    payoff = (
        np.maximum(S_grid - opts["K"], 0.0)
        if opts["opt_type"] == "Call"
        else np.maximum(opts["K"] - S_grid, 0.0)
    )
    fig1, ax1 = plt.subplots()
    ax1.plot(S_grid, payoff, label=f"{opts['opt_type']} payoff at expiry")
    ax1.axvline(
        opts["S_for_options"], linestyle="--", linewidth=1, label="Current spot"
    )
    ax1.axhline(0, linewidth=1)
    ax1.set_xlabel("Underlying price at expiry (S_T)")
    ax1.set_ylabel("Payoff")
    ax1.set_title(f"{opts['opt_type']} payoff at expiry (not discounted)")
    ax1.legend()
    st.pyplot(fig1, use_container_width=True)

    # Value today vs spot
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
    if opts["opt_type"] == "Call":
        ax2.plot(S_grid, call_vals, label="Call value (today)")
        ax2.scatter([opts["S_for_options"]], [call], marker="o")
    else:
        ax2.plot(S_grid, put_vals, label="Put value (today)")
        ax2.scatter([opts["S_for_options"]], [put], marker="o")
    ax2.axvline(
        opts["S_for_options"], linestyle="--", linewidth=1, label="Current spot"
    )
    ax2.set_xlabel("Spot (S₀,eff)")
    ax2.set_ylabel("Option value today")
    ax2.set_title(
        f"{opts['opt_type']} value vs effective spot (Black–Scholes, today’s price)"
    )
    ax2.legend()
    st.pyplot(fig2, use_container_width=True)


# ===== TAB 2: American (CRR) =====
with tab2:
    st.subheader("American vs European (CRR Binomial)")
    st.caption(
        "Advanced: compares CRR tree pricing for American options with European (tree + Black–Scholes)."
    )

    opts = render_options_inputs("opt_inputs_tab2")

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
    euro_tree = crr_price_european(inp, otype, steps)
    amer_tree = crr_price_american(inp, otype, steps)
    bs_euro = bs_prices(inp)[0] if otype == "call" else bs_prices(inp)[1]

    c1, c2, c3 = st.columns(3)
    c1.metric("CRR European", f"{euro_tree:,.6f}")
    c2.metric(
        "BS European",
        f"{bs_euro:,.6f}",
        delta=f"{(euro_tree - bs_euro):+.6f}",
    )
    c3.metric("CRR American", f"{amer_tree:,.6f}")
    st.caption(f"Early-exercise premium vs BS: {amer_tree - bs_euro:+.6f}.")


# ===== TAB 3: Chain & Smiles =====
with tab3:
    st.subheader("Options Chain (CSV) → Bulk IV & Smiles")
    st.caption("Advanced: upload an options chain, compute IV row-wise, see smiles.")

    opts = render_options_inputs("opt_inputs_tab3")

    st.markdown(
        """
Upload an options chain CSV with columns like:

- `K` or `strike` – strike prices  
- `Mid` **or** (`Bid` + `Ask`) **or** `Price` / `Last` / `Market_Price` – option prices  
- Optional: `T` (years), `type` (`call`/`put` or `c`/`p`)

If you don't have real data, download the sample and play with it.
"""
    )

    sample_chain_csv = """K,Mid,T,type
100,5.2,0.5,call
100,4.8,0.5,put
110,2.4,0.5,call
90,1.8,0.5,put
"""
    st.download_button(
        "Download sample chain CSV",
        data=sample_chain_csv,
        file_name="sample_chain.csv",
        mime="text/csv",
    )

    chain_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="chain_upl",
    )
    if chain_file is not None:
        try:
            chain_df = pd.read_csv(chain_file)
            cols = {c.lower().strip(): c for c in chain_df.columns}

            def getcol(*names):
                for n in names:
                    if n in cols:
                        return cols[n]
                return None

            col_K = getcol("k", "strike")
            col_bid = getcol("bid")
            col_ask = getcol("ask")
            col_mid = getcol("mid", "mkt_mid")
            col_px = getcol("price", "last", "market_price")
            col_T = getcol("t", "ttm", "maturity", "time_to_maturity_years")
            col_ty = getcol("type", "option_type")

            if col_K is None:
                st.error("CSV must include a strike column: one of [K, strike].")
            else:
                df = chain_df.copy()
                df["K"] = df[col_K].astype(float)

                if col_mid is not None:
                    price_series = df[col_mid].astype(float)
                elif col_bid is not None and col_ask is not None:
                    price_series = (
                        df[col_bid].astype(float) + df[col_ask].astype(float)
                    ) / 2.0
                elif col_px is not None:
                    price_series = df[col_px].astype(float)
                else:
                    st.error("Provide Mid, or Bid+Ask, or Price column.")
                    price_series = None

                df["T_row"] = (
                    df[col_T].astype(float) if col_T is not None else opts["T"]
                )

                if col_ty is not None:
                    df["otype"] = (
                        df[col_ty]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                        .replace({"c": "call", "p": "put"})
                    )
                    df.loc[~df["otype"].isin(["call", "put"]), "otype"] = (
                        opts["opt_type"].lower()
                    )
                else:
                    df["otype"] = opts["opt_type"].lower()

                if price_series is not None:
                    df_iv = cached_compute_chain_iv(
                        df.assign(MarketPrice=price_series),
                        S0=opts["S_for_options"],
                        r=opts["r"],
                        q=opts["q"],
                        default_T=opts["T"],
                        default_type=opts["opt_type"].lower(),
                        sigma_seed=max(opts["sigma"], 1e-6),
                    )

                    st.dataframe(
                        df_iv[
                            ["K", "T", "otype", "MarketPrice", "IV", "IV_%", "Tag"]
                        ]
                        .sort_values(["T", "K"])
                        .style.format(
                            {
                                "MarketPrice": "{:.4f}",
                                "IV": "{:.6f}",
                                "IV_%": "{:.3f}",
                            }
                        ),
                        use_container_width=True,
                    )

                    st.markdown("**IV Smile (IV vs Strike)**")
                    valid = df_iv.dropna(subset=["IV"])
                    if not valid.empty:
                        for tval in sorted(valid["T"].unique()):
                            sub = valid[valid["T"] == tval].sort_values("K")
                            fig, ax = plt.subplots()
                            ax.plot(
                                sub["K"],
                                sub["IV"] * 100.0,
                                marker="o",
                            )
                            ax.set_xlabel("Strike (K)")
                            ax.set_ylabel("IV (%)")
                            ax.set_title(f"T = {tval:.4f} years")
                            st.pyplot(fig, use_container_width=True)

                    out = (
                        df_iv.sort_values(["T", "K"])
                        .to_csv(index=False)
                        .encode("utf-8")
                    )
                    st.download_button(
                        "Download processed chain with IV (CSV)",
                        data=out,
                        file_name="options_chain_with_iv.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Could not process CSV: {e}")


# ===== TAB 4: Vol Surface =====
with tab4:
    st.subheader("Volatility Surface (Strike × Expiry)")
    st.caption("Advanced: multi-expiry IV surface from an options chain.")

    opts = render_options_inputs("opt_inputs_tab4")

    st.markdown(
        """
Upload a CSV similar to the Chain tab, but with **multiple expiries**.

We'll compute IV per row and show:
- IV smiles per expiry
- 3D IV surface & contour
"""
    )

    surf_file = st.file_uploader(
        "Upload multi-expiry CSV (K, price fields; optional T, type)",
        type=["csv"],
        key="surf_upl",
    )
    if surf_file is not None:
        try:
            raw = pd.read_csv(surf_file)
            df_iv = cached_compute_chain_iv(
                raw,
                S0=opts["S_for_options"],
                r=opts["r"],
                q=opts["q"],
                default_T=opts["T"],
                default_type=opts["opt_type"].lower(),
                sigma_seed=max(opts["sigma"], 1e-6),
            )
            st.dataframe(
                df_iv[
                    ["K", "T", "otype", "MarketPrice", "IV_%", "Tag"]
                ]
                .sort_values(["T", "K"])
                .style.format(
                    {"MarketPrice": "{:.4f}", "IV_%": "{:.3f}"}
                ),
                use_container_width=True,
            )

            valid = df_iv.dropna(subset=["IV"])
            if not valid.empty:
                st.markdown("**IV Smiles by Expiry**")
                for t_val in sorted(valid["T"].unique()):
                    sub = valid[valid["T"] == t_val].sort_values("K")
                    fig, ax = plt.subplots()
                    ax.plot(
                        sub["K"],
                        sub["IV_%"],
                        marker="o",
                        linestyle="-",
                    )
                    ax.set_xlabel("Strike (K)")
                    ax.set_ylabel("Implied Volatility (%)")
                    ax.set_title(f"Smile — T = {t_val:.4f} years")
                    st.pyplot(fig, use_container_width=True)

                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                K_arr, T_arr, IVp_arr = prepare_surface_arrays(valid)
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111, projection="3d")
                ax3.scatter(K_arr, T_arr, IVp_arr)
                ax3.set_xlabel("Strike (K)")
                ax3.set_ylabel("Expiry (T, years)")
                ax3.set_zlabel("IV (%)")
                ax3.set_title("IV Surface — 3D scatter")
                st.pyplot(fig3, use_container_width=True)

                fig4, ax4 = plt.subplots()
                try:
                    tric = ax4.tricontourf(K_arr, T_arr, IVp_arr, levels=12)
                    fig4.colorbar(tric, ax=ax4, label="IV (%)")
                    ax4.set_xlabel("Strike (K)")
                    ax4.set_ylabel("Expiry (T, years)")
                    ax4.set_title("IV Surface — contour")
                    st.pyplot(fig4, use_container_width=True)
                except Exception:
                    st.info(
                        "Not enough distinct points for a filled contour; 3D scatter still shows the shape."
                    )

            out2 = (
                df_iv.sort_values(["T", "K"])
                .to_csv(index=False)
                .encode("utf-8")
            )
            st.download_button(
                "Download IV dataset (CSV)",
                data=out2,
                file_name="iv_surface_dataset.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error computing IV: {e}")


# ===== TAB 5: Bonds (numeric & dates) =====
with tab5:
    st.subheader("Bond Pricer — Price, YTM, Duration & Convexity")

    st.markdown(
        """
**Quick guide:**
- To get **bond price**: enter coupon, time to maturity, yield → we give you price + durations.
- To get **YTM**: enter coupon, time to maturity, and **target price** → we solve the yield.
"""
    )

    mode_bond_input = st.radio(
        "Input mode",
        ["Numeric T", "Dates"],
        horizontal=True,
    )

    bc1, bc2, bc3 = st.columns(3)
    face = bc1.number_input(
        "Face / Redemption value", min_value=1.0, value=100.0, step=1.0
    )
    coupon_pct = (
        bc2.number_input(
            "Coupon rate (% per year)", value=5.00, step=0.25, format="%.2f"
        )
        / 100.0
    )
    freq = int(
        bc3.selectbox(
            "Payments per year", options=[1, 2, 4], index=1
        )
    )

    if mode_bond_input == "Numeric T":
        bc4, bc5 = st.columns(2)
        T_years = bc4.number_input(
            "Time to maturity (years)",
            min_value=0.25,
            value=5.0,
            step=0.25,
        )
        ytm_pct = (
            bc5.number_input(
                "Yield to maturity (% per year)",
                value=6.00,
                step=0.10,
                format="%.2f",
            )
            / 100.0
        )

        mode_bond = st.radio(
            "Mode", ["Inputs → Price", "Price → YTM"], horizontal=True
        )
        if mode_bond == "Inputs → Price":
            P = price_bond(face, coupon_pct, ytm_pct, T_years, freq)
            mac = macaulay_duration(face, coupon_pct, ytm_pct, T_years, freq)
            mod = modified_duration(face, coupon_pct, ytm_pct, T_years, freq)
            conv = convexity_numeric(
                face, coupon_pct, ytm_pct, T_years, freq
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"{P:,.4f}")
            c2.metric("Macaulay Duration (yrs)", f"{mac:,.4f}")
            c3.metric("Modified Duration (yrs)", f"{mod:,.4f}")
            c4.metric("Convexity", f"{conv:,.4f}")
            st.info(
                "Duration ≈ interest rate sensitivity; convexity ≈ how that sensitivity bends for larger rate moves."
            )
        else:
            target_price = st.number_input(
                "Target clean price",
                min_value=0.01,
                value=100.00,
                step=0.25,
            )
            ytm_solved = ytm_from_price(
                target_price, face, coupon_pct, T_years, freq
            )
            if ytm_solved is None:
                st.error(
                    "Could not solve YTM in bounds [-99%, 100%]. Check inputs."
                )
            else:
                mac = macaulay_duration(
                    face, coupon_pct, ytm_solved, T_years, freq
                )
                mod = modified_duration(
                    face, coupon_pct, ytm_solved, T_years, freq
                )
                conv = convexity_numeric(
                    face, coupon_pct, ytm_solved, T_years, freq
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Solved YTM", f"{ytm_solved * 100:.4f}%")
                c2.metric("Macaulay Duration (yrs)", f"{mac:,.4f}")
                c3.metric("Modified Duration (yrs)", f"{mod:,.4f}")
                c4.metric("Convexity", f"{conv:,.4f}")
                st.info(
                    "Solved YTM is the internal rate of return of all bond cash flows at the target price."
                )

    else:
        colD1, colD2, colD3 = st.columns(3)
        val_d_str = colD1.text_input(
            "Settlement / Valuation date (YYYY-MM-DD)",
            value=str(date.today()),
        )
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

        ytm_pct_d = (
            st.number_input(
                "Yield to maturity (% per year)",
                value=6.00,
                step=0.10,
                format="%.2f",
            )
            / 100.0
        )
        mode_bond_d = st.radio(
            "Mode (dates)", ["Inputs → Price", "Price → YTM"], horizontal=True
        )

        if mode_bond_d == "Inputs → Price":
            P = price_bond_dates(
                face,
                coupon_pct,
                ytm_pct_d,
                settle_d,
                mat_d,
                freq,
                dc_bond,
                biz,
            )
            mac = macaulay_duration_dates(
                face,
                coupon_pct,
                ytm_pct_d,
                settle_d,
                mat_d,
                freq,
                dc_bond,
                biz,
            )
            mod = modified_duration_dates(
                face,
                coupon_pct,
                ytm_pct_d,
                settle_d,
                mat_d,
                freq,
                dc_bond,
                biz,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"{P:,.4f}")
            c2.metric("Macaulay Duration (yrs)", f"{mac:,.4f}")
            c3.metric("Modified Duration (yrs)", f"{mod:,.4f}")
        else:
            target_price = st.number_input(
                "Target clean price (dates)",
                min_value=0.01,
                value=100.00,
                step=0.25,
            )
            ytm_solved = ytm_from_price_dates(
                target_price,
                face,
                coupon_pct,
                settle_d,
                mat_d,
                freq,
                dc_bond,
                biz,
            )
            if ytm_solved is None:
                st.error(
                    "Could not solve YTM in bounds [-99%, 100%]. Check inputs."
                )
            else:
                mac = macaulay_duration_dates(
                    face,
                    coupon_pct,
                    ytm_solved,
                    settle_d,
                    mat_d,
                    freq,
                    dc_bond,
                    biz,
                )
                mod = modified_duration_dates(
                    face,
                    coupon_pct,
                    ytm_solved,
                    settle_d,
                    mat_d,
                    freq,
                    dc_bond,
                    biz,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Solved YTM", f"{ytm_solved * 100:.4f}%")
                c2.metric("Macaulay Duration (yrs)", f"{mac:,.4f}")
                c3.metric("Modified Duration (yrs)", f"{mod:,.4f}")


# ===== TAB 6: Yield Curve =====
with tab6:
    st.subheader("Yield Curve — Bootstrapping")
    st.caption(
        "Advanced: build a zero curve from deposits and coupon bonds, then derive discount factors and forward rates."
    )

    curve_mode = st.radio(
        "Curve input mode", ["Numeric", "Dates"], horizontal=True
    )
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
                deposits.append(
                    (float(r0["T"]), float(r0["Rate (%)"]) / 100.0)
                )
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
                from date_utils import parse_date as pd_

                deposits.append(
                    {
                        "date": pd_(str(r0["Maturity"])),
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
                from date_utils import parse_date as pd_

                bonds.append(
                    {
                        "maturity_date": pd_(str(r0["Maturity"])),
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
                curve = cached_bootstrap_curve(
                    deposits,
                    bonds,
                    valuation_date=curve_val_date,
                    day_count=dc_curve,
                )
                st.session_state["bootstrapped_curve"] = curve
                st.success("Curve bootstrapped.")
            except Exception as e:
                st.error(f"Bootstrapping error: {e}")

    curve: YieldCurve | None = st.session_state.get(
        "bootstrapped_curve"
    )
    if curve is not None:
        st.write("**Zero Curve**")
        st.dataframe(
            curve.as_dataframe().style.format(
                {"Zero(%)": "{:.3f}", "DF": "{:.6f}"}
            ),
            use_container_width=True,
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.plot(
            curve.mats, [z * 100 for z in curve.zeros], marker="o"
        )
        ax1.set_title("Spot (Zero) Curve")
        ax1.set_xlabel("T (years)")
        ax1.set_ylabel("Zero (%)")
        ax2.plot(curve.mats, curve.dfs, marker="o")
        ax2.set_title("Discount Factors")
        ax2.set_xlabel("T (years)")
        ax2.set_ylabel("DF")
        st.pyplot(fig, use_container_width=True)

        colf1, colf2 = st.columns(2)
        t1 = colf1.number_input(
            "Forward start T₁ (years)",
            value=1.0,
            step=0.25,
            min_value=0.0,
        )
        t2 = colf2.number_input(
            "Forward end T₂ (years)",
            value=2.0,
            step=0.25,
            min_value=0.01,
        )
        st.metric(
            "Annual forward rate T₁ → T₂",
            f"{curve.get_fwd(t1, t2) * 100:.3f}%",
        )


# ===== TAB 7: Swaps =====
with tab7:
    st.subheader("Plain-Vanilla Interest Rate Swap")
    st.caption(
        "Advanced: uses the bootstrapped curve to compute par fixed rate, PV, and DV01 of a fixed-for-floating swap."
    )

    curve: YieldCurve | None = st.session_state.get(
        "bootstrapped_curve"
    )
    if curve is None:
        st.info(
            "Bootstrap a curve first in the **Yield Curve** tab to use swaps."
        )
    else:
        colS1, colS2, colS3 = st.columns(3)
        notional = colS1.number_input(
            "Notional",
            min_value=1_000.0,
            value=1_000_000.0,
            step=50_000.0,
        )
        T_swap = colS2.number_input(
            "Swap maturity T (years)",
            min_value=0.5,
            value=5.0,
            step=0.5,
        )
        freq = int(
            colS3.selectbox(
                "Fixed leg payments per year", options=[1, 2, 4], index=1
            )
        )
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


# ===== TAB 8: Futures & Black-76 (curve-aware) =====
with tab8:
    st.subheader("Futures/Forwards & Black-76 Options")
    st.caption(
        "Advanced: price options on futures/forwards using Black–76, with optional curve-based discounting."
    )

    opts = render_options_inputs("opt_inputs_tab8")

    curve: YieldCurve | None = st.session_state.get(
        "bootstrapped_curve"
    )
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

    # Compute/enter forward (use original S0 for cost-of-carry)
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
            F0 = forward_price_from_curve(
                opts["S0"], curve, T_fut, q_cont=q_flat
            )
        else:
            F0 = forward_price(
                opts["S0"], opts["r"], opts["q"], T_fut
            )
    else:
        F0 = colF0.number_input(
            "F₀ (futures/forward)",
            min_value=0.0001,
            value=float(
                forward_price(
                    opts["S0"], opts["r"], opts["q"], T_fut
                )
            ),
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
            st.info(
                "No curve in session. Bootstrap one in the **Yield Curve** tab, or switch to Flat r."
            )
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
            st.error(
                "No valid IV found in [1e-6, 5.0] for these inputs."
            )
        else:
            st.metric("Black-76 IV", f"{iv76 * 100:.3f}%")

    # Price & Greeks
    if DF_T is None:
        price_b76 = black76_price(
            F0, K_fut, opts["r"], T_fut, sigma_b76, otype_b76
        )
        G76 = black76_greeks(
            F0, K_fut, opts["r"], T_fut, sigma_b76
        )
        delta_disp = (
            G76["delta_fut"]["call"]
            if otype_b76 == "call"
            else G76["delta_fut"]["put"]
        )
        gamma_disp = G76["gamma_fut"]
        vega_disp = G76["vega"]
        theta_day = G76["theta_per_day"]
        rho_disp = G76["rho"]
    else:
        price_b76 = black76_price_df(
            F0, K_fut, DF_T, T_fut, sigma_b76, otype_b76
        )
        G76 = black76_greeks_df(
            F0, K_fut, DF_T, T_fut, sigma_b76
        )
        delta_disp = (
            G76["delta_fut"]["call"]
            if otype_b76 == "call"
            else G76["delta_fut"]["put"]
        )
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
    axp.plot(
        F_grid,
        payoff_call if otype_b76 == "call" else payoff_put,
        label=f"{otype_b76.capitalize()} payoff at expiry",
    )
    axp.axhline(0, linewidth=1)
    axp.set_xlabel("F_T")
    axp.set_ylabel("Payoff")
    axp.legend()
    st.pyplot(figp, use_container_width=True)

    F0_grid = np.linspace(max(0.01, F0 * 0.6), F0 * 1.4, 120)
    if DF_T is None:
        values = [
            black76_price(
                fv, K_fut, opts["r"], T_fut, sigma_b76, otype_b76
            )
            for fv in F0_grid
        ]
    else:
        values = [
            black76_price_df(
                fv, K_fut, DF_T, T_fut, sigma_b76, otype_b76
            )
            for fv in F0_grid
        ]
    figv, axv = plt.subplots()
    axv.plot(F0_grid, values, label="Option value today")
    axv.scatter([F0], [price_b76], marker="o")
    axv.set_xlabel("F₀")
    axv.set_ylabel("Option value")
    axv.set_title("Option value vs F₀")
    axv.legend()
    st.pyplot(figv, use_container_width=True)


# ===== TAB 9: Reports & Export =====
with tab9:
    st.subheader("Reports & Export")
    st.caption(
        "Use this after you’ve set up your option in the Option Pricer tab to export reports."
    )

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

    curve: YieldCurve | None = st.session_state.get(
        "bootstrapped_curve"
    )
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

st.caption(
    "All tabs are available: use Option Pricer & Bonds to start, then experiment with the advanced tabs as you learn."
)
