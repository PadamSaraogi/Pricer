# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from bs_core import (
    OptionInput,
    bs_prices,
    bs_greeks,
    implied_vol,
    moneyness_tags,
)
from american_binomial import (
    crr_price_european,
    crr_price_american,
)
from vol_surface import compute_chain_iv, prepare_surface_arrays
from bonds import (
    price_bond,
    ytm_from_price,
    macaulay_duration,
    modified_duration,
    convexity_numeric,
)

st.set_page_config(page_title="Option Analytics Suite", page_icon="📊", layout="wide")
st.title("📊 Option & Bond Analytics Dashboard")

# -----------------------------
# Sidebar (global option inputs)
# -----------------------------
st.sidebar.header("Global Inputs — Options")
colA, colB = st.sidebar.columns(2)
S0 = colA.number_input("Spot price (S₀)", min_value=0.01, value=100.0, step=1.0)
K = colB.number_input("Strike (K)", min_value=0.01, value=105.0, step=1.0)

col1, col2 = st.sidebar.columns(2)
r = col1.number_input("Risk-free r (annual, %)", value=5.0, step=0.25, format="%.2f") / 100.0
sigma = col2.number_input("Volatility σ (annual, %)", min_value=0.01, value=20.0, step=0.5, format="%.2f") / 100.0

col3, col4 = st.sidebar.columns(2)
T = col3.number_input("Time to expiry T (years)", min_value=0.00274, value=0.5, step=0.01, help="0.00274 ≈ 1 day")
q = col4.number_input("Dividend yield q (annual, %)", value=0.0, step=0.25, format="%.2f") / 100.0

opt_type = st.sidebar.radio("Option type", ["Call", "Put"], horizontal=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Option Pricer",
        "American (CRR)",
        "Chain & Smiles",
        "Vol Surface",
        "Bonds",
    ]
)

# =============================
# TAB 1: Option Pricer (BSM)
# =============================
with tab1:
    st.subheader("European Options — Black–Scholes")

    # Optional market data for IV
    st.caption("Optional: market quotes (any currency)")
    colm1, colm2, _ = st.columns(3)
    bid = colm1.number_input("Bid", min_value=0.0, value=0.0, step=0.1, key="bid1")
    ask = colm2.number_input("Ask", min_value=0.0, value=0.0, step=0.1, key="ask1")
    mid = None
    if bid > 0 and ask > 0 and ask >= bid:
        mid = 0.5 * (bid + ask)

    mkt_price = st.number_input("Single market price (for IV)", min_value=0.0, value=0.0, step=0.1, key="mkt1")

    inp = OptionInput(S0=S0, K=K, r=r, sigma=sigma, T=T, q=q)
    try:
        call, put, d1, d2 = bs_prices(inp)
        G = bs_greeks(inp)
    except Exception as e:
        st.error(f"Input error: {e}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Call price", f"{call:,.4f}")
    m2.metric("Put price", f"{put:,.4f}")
    m3.metric("d₁", f"{d1:,.4f}")
    m4.metric("d₂", f"{d2:,.4f}")

    st.subheader("Moneyness & Tags")
    tag_info = moneyness_tags(S0, K, d1)
    mtc1, mtc2, mtc3, mtc4 = st.columns(4)
    mtc1.metric("S/K", tag_info["S_over_K"])
    mtc2.metric("log-moneyness", tag_info["log_moneyness"])
    mtc3.metric("d₁", tag_info["d1"])
    mtc4.metric("Tag", tag_info["tag"])

    st.subheader("Greeks")
    greek_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega (per 1% σ)", "Theta/day", "Theta/year", "Rho"],
            "Call": [
                G["delta"]["call"],
                G["gamma"],
                G["vega_per_1pct"],
                G["theta_per_day"]["call"],
                G["theta_per_year"]["call"],
                G["rho"]["call"],
            ],
            "Put": [
                G["delta"]["put"],
                G["gamma"],
                G["vega_per_1pct"],
                G["theta_per_day"]["put"],
                G["theta_per_year"]["put"],
                G["rho"]["put"],
            ],
        }
    )
    st.dataframe(greek_df.style.format({"Call": "{:.6f}", "Put": "{:.6f}"}), use_container_width=True)

    st.subheader("Implied Volatility (from market price)")
    coliv1, coliv2 = st.columns(2)
    otype = opt_type.lower()
    if mkt_price > 0:
        iv = implied_vol(mkt_price, inp, otype)
        if iv is None:
            coliv1.error("No valid IV found in [1e-6, 5.0]. Check inputs/price.")
        else:
            coliv1.metric("IV (single price)", f"{iv*100:.3f}%")
    if mid is not None:
        iv_mid = implied_vol(mid, inp, otype)
        if iv_mid is None:
            coliv2.error("No valid IV for Bid/Ask mid.")
        else:
            coliv2.metric("IV (Bid/Ask mid)", f"{iv_mid*100:.3f}%")

    st.subheader("Charts")
    S_min = max(0.01, S0 * 0.4)
    S_max = S0 * 1.6
    S_grid = np.linspace(S_min, S_max, 200)
    payoff_call = np.maximum(S_grid - K, 0.0)
    payoff_put = np.maximum(K - S_grid, 0.0)

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

    st.subheader("Scenario table (spot ladder)")
    spots = np.linspace(S0 * 0.7, S0 * 1.3, 11)
    rows = []
    for s in spots:
        _in = OptionInput(S0=float(s), K=K, r=r, sigma=sigma, T=T, q=q)
        c, p, *_ = bs_prices(_in)
        rows.append({"Spot": s, "Call": c, "Put": p})
    df_scn = pd.DataFrame(rows)
    st.dataframe(df_scn.style.format({"Spot": "{:.4f}", "Call": "{:.4f}", "Put": "{:.4f}"}), use_container_width=True)

# =============================
# TAB 2: American (CRR)
# =============================
with tab2:
    st.subheader("American vs European (CRR Binomial)")
    steps = st.slider(
        "CRR steps (accuracy vs speed)",
        min_value=25,
        max_value=1000,
        value=200,
        step=25,
        key="crr_steps",
    )
    inp = OptionInput(S0=S0, K=K, r=r, sigma=sigma, T=T, q=q)
    otype = opt_type.lower()
    euro_tree = crr_price_european(inp, otype, steps)
    amer_tree = crr_price_american(inp, otype, steps)
    bs_euro = (bs_prices(inp)[0] if otype == "call" else bs_prices(inp)[1])

    c1, c2, c3 = st.columns(3)
    c1.metric("CRR European", f"{euro_tree:,.6f}")
    c2.metric("BS European", f"{bs_euro:,.6f}", delta=f"{(euro_tree - bs_euro):+.6f}")
    c3.metric("CRR American", f"{amer_tree:,.6f}")

    premium = amer_tree - bs_euro
    st.caption(
        f"Early-exercise premium vs BS: {premium:+.6f}. "
        "(American call with q≈0 ≈ European; American put ≥ European.)"
    )

# =============================
# TAB 3: Chain & Smiles
# =============================
with tab3:
    st.subheader("Options Chain (CSV) → Bulk IV & Smiles")
    chain_file = st.file_uploader(
        "Upload CSV with: K, Mid or Bid+Ask or Price; optional T (years), type (call/put)",
        type=["csv"],
        key="chain_upl",
    )
    if chain_file is not None:
        try:
            chain_df = pd.read_csv(chain_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            chain_df = None

        if chain_df is not None:
            # Flexible columns
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
                st.error("CSV must include strike column [K/strike].")
            else:
                df = chain_df.copy()
                df["K"] = df[col_K].astype(float)

                # price series
                if col_mid is not None:
                    price_series = df[col_mid].astype(float)
                elif col_bid is not None and col_ask is not None:
                    price_series = (df[col_bid].astype(float) + df[col_ask].astype(float)) / 2.0
                elif col_px is not None:
                    price_series = df[col_px].astype(float)
                else:
                    st.error("Provide Mid or Bid+Ask or Price column.")
                    price_series = None

                # T series
                if col_T is not None:
                    df["T_row"] = df[col_T].astype(float)
                else:
                    df["T_row"] = T

                # type series
                if col_ty is not None:
                    df["otype"] = (
                        df[col_ty].astype(str).str.lower().str.strip().replace({"c": "call", "p": "put"})
                    )
                    df.loc[~df["otype"].isin(["call", "put"]), "otype"] = opt_type.lower()
                else:
                    df["otype"] = opt_type.lower()

                if price_series is not None:
                    iv_list, tag_list = [], []
                    for k, px, t, ty in zip(df["K"], price_series, df["T_row"], df["otype"]):
                        _inp = OptionInput(S0=S0, K=float(k), r=r, sigma=max(sigma, 1e-6), T=float(t), q=q)
                        iv = implied_vol(float(px), _inp, ty)
                        if iv is None:
                            iv_list.append(float("nan"))
                            tag_list.append("no-root")
                        else:
                            iv_list.append(iv)
                            G_row = bs_greeks(_inp)
                            tag_list.append(moneyness_tags(S0, float(k), G_row["d1"])["tag"])

                    df["MarketPrice"] = price_series
                    df["IV"] = iv_list
                    df["IV_%"] = df["IV"] * 100.0
                    df["Tag"] = tag_list

                    st.dataframe(
                        df[["K", "T_row", "otype", "MarketPrice", "IV_%", "Tag"]]
                        .rename(columns={"T_row": "T"})
                        .style.format({"MarketPrice": "{:.4f}", "IV_%": "{:.3f}"}),
                        use_container_width=True,
                    )

                    st.markdown("**IV Smile (IV vs Strike)**")
                    valid = df.dropna(subset=["IV"])
                    if not valid.empty:
                        for tval in sorted(valid["T_row"].unique()):
                            sub = valid[valid["T_row"] == tval].sort_values("K")
                            fig, ax = plt.subplots()
                            ax.plot(sub["K"], sub["IV_%"], marker="o")
                            ax.set_xlabel("Strike (K)")
                            ax.set_ylabel("IV (%)")
                            ax.set_title(f"T={tval:.4f} yrs")
                            st.pyplot(fig, use_container_width=True)

                    out = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download processed chain (CSV)",
                        data=out,
                        file_name="options_chain_with_iv.csv",
                        mime="text/csv",
                    )

# =============================
# TAB 4: Vol Surface
# =============================
with tab4:
    st.subheader("Volatility Surface (K × T)")
    surf_file = st.file_uploader(
        "Upload multi-expiry CSV (K, T, price fields; optional type)",
        type=["csv"],
        key="surf_upl",
    )

    if surf_file is not None:
        try:
            raw = pd.read_csv(surf_file)
            df_iv = compute_chain_iv(
                raw,
                S0=S0,
                r=r,
                q=q,
                default_T=T,
                default_type=opt_type.lower(),
                sigma_seed=max(sigma, 1e-6),
            )
        except Exception as e:
            st.error(f"Error computing IV: {e}")
            df_iv = None

        if df_iv is not None:
            st.dataframe(
                df_iv[["K", "T", "otype", "MarketPrice", "IV_%", "Tag"]]
                .sort_values(["T", "K"])
                .style.format({"MarketPrice": "{:.4f}", "IV_%": "{:.3f}"}),
                use_container_width=True,
            )

            valid = df_iv.dropna(subset=["IV"])
            if not valid.empty:
                st.markdown("**IV Smiles by Expiry**")
                for t_val in sorted(valid["T"].unique()):
                    sub = valid[valid["T"] == t_val].sort_values("K")
                    fig, ax = plt.subplots()
                    ax.plot(sub["K"], sub["IV_%"], marker="o", linestyle="-")
                    ax.set_xlabel("Strike (K)")
                    ax.set_ylabel("Implied Volatility (%)")
                    ax.set_title(f"Smile — T={t_val:.4f} yrs")
                    st.pyplot(fig, use_container_width=True)

                # 3D scatter
                st.markdown("**Volatility Surface (3D scatter)**")
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

                # 2D contour
                st.markdown("**Volatility Surface (2D contour)**")
                fig4, ax4 = plt.subplots()
                try:
                    tric = ax4.tricontourf(K_arr, T_arr, IVp_arr, levels=12)
                    fig4.colorbar(tric, ax=ax4, label="IV (%)")
                    ax4.set_xlabel("Strike (K)")
                    ax4.set_ylabel("Expiry (T, years)")
                    ax4.set_title("IV Surface — contour")
                    st.pyplot(fig4, use_container_width=True)
                except Exception:
                    st.info("Not enough distinct points for a filled contour; 3D scatter still shows shape.")

            out2 = df_iv.sort_values(["T", "K"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download IV dataset (CSV)",
                data=out2,
                file_name="iv_surface_dataset.csv",
                mime="text/csv",
            )

# =============================
# TAB 5: Bonds
# =============================
with tab5:
    st.subheader("Bond Pricer — Price, YTM, Duration & Convexity")
    bc1, bc2, bc3 = st.columns(3)
    face = bc1.number_input("Face/Redemption", min_value=1.0, value=100.0, step=1.0)
    coupon_pct = bc2.number_input("Coupon rate (% p.a.)", value=5.00, step=0.25, format="%.2f") / 100.0
    freq = int(bc3.selectbox("Payments per year", options=[1, 2, 4], index=1))

    bc4, bc5 = st.columns(2)
    T_years = bc4.number_input("Time to maturity (years)", min_value=0.25, value=5.0, step=0.25)
    ytm_pct = bc5.number_input("Yield to maturity (% p.a.)", value=6.00, step=0.10, format="%.2f") / 100.0

    mode_bond = st.radio("Mode", ["Inputs → Price", "Price → YTM"], horizontal=True)

    if mode_bond == "Inputs → Price":
        P = price_bond(face, coupon_pct, ytm_pct, T_years, freq)
        mac = macaulay_duration(face, coupon_pct, ytm_pct, T_years, freq)
        mod = modified_duration(face, coupon_pct, ytm_pct, T_years, freq)
        conv = convexity_numeric(face, coupon_pct, ytm_pct, T_years, freq)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{P:,.4f}")
        c2.metric("Macaulay Dur (yrs)", f"{mac:,.4f}")
        c3.metric("Modified Dur (yrs)", f"{mod:,.4f}")
        c4.metric("Convexity", f"{conv:,.4f}")
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
            c1.metric("Solved YTM", f"{ytm_solved*100:.4f}%")
            c2.metric("Macaulay Dur (yrs)", f"{mac:,.4f}")
            c3.metric("Modified Dur (yrs)", f"{mod:,.4f}")
            c4.metric("Convexity", f"{conv:,.4f}")

st.caption(
    "Options: European (BSM), American (CRR). IV smiles & surface. "
    "Bonds: price, YTM, duration, convexity. Rates are nominal p.a. unless stated."
)
