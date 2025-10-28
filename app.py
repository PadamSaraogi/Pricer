# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

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
from yield_curve import bootstrap_curve, YieldCurve
from swaps import par_swap_rate, swap_pv, dv01
from daycount import year_fraction, DayCount, BizConv
from date_utils import parse_date

st.set_page_config(page_title="Option & Fixed-Income Analytics", page_icon="📊", layout="wide")
st.title("📊 Option & Fixed-Income Analytics Dashboard")

# -----------------------------
# Sidebar (global option inputs)
# -----------------------------
st.sidebar.header("Global Inputs — Options")
use_dates = st.sidebar.toggle("Use Dates for maturities", value=False, help="Compute T from dates using day-counts.")

colA, colB = st.sidebar.columns(2)
S0 = colA.number_input("Spot price (S₀)", min_value=0.01, value=100.0, step=1.0)
K = colB.number_input("Strike (K)", min_value=0.01, value=105.0, step=1.0)

col1, col2 = st.sidebar.columns(2)
r = col1.number_input("Risk-free r (annual, %)", value=5.0, step=0.25, format="%.2f") / 100.0
sigma = col2.number_input("Volatility σ (annual, %)", min_value=0.01, value=20.0, step=0.5, format="%.2f") / 100.0

col3, col4 = st.sidebar.columns(2)
if use_dates:
    dc: DayCount = st.sidebar.selectbox("Day-count (for T)", ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"], index=0)
    val_date_str = st.sidebar.text_input("Valuation date (YYYY-MM-DD)", value=str(date.today()))
    expiry_str = st.sidebar.text_input("Option expiry (YYYY-MM-DD)", value=str(date.today()))
    try:
        val_date = parse_date(val_date_str)
        exp_date = parse_date(expiry_str)
        T = year_fraction(val_date, exp_date, dc)
    except Exception:
        val_date = date.today()
        exp_date = date.today()
        T = 0.0
    q = col4.number_input("Dividend yield q (annual, %)", value=0.0, step=0.25, format="%.2f") / 100.0
else:
    T = col3.number_input("Time to expiry T (years)", min_value=0.00274, value=0.5, step=0.01, help="0.00274 ≈ 1 day")
    q = col4.number_input("Dividend yield q (annual, %)", value=0.0, step=0.25, format="%.2f") / 100.0
    val_date = date.today()  # for other tabs default

opt_type = st.sidebar.radio("Option type", ["Call", "Put"], horizontal=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Option Pricer",
        "American (CRR)",
        "Chain & Smiles",
        "Vol Surface",
        "Bonds",
        "Yield Curve",
        "Swaps",
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
    mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0 and ask >= bid) else None

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
        coliv1.metric("IV (single price)" if iv is not None else "IV error", f"{iv*100:.3f}%" if iv else "No root")
    if mid is not None:
        iv_mid = implied_vol(mid, inp, otype)
        coliv2.metric("IV (Bid/Ask mid)" if iv_mid is not None else "IV error", f"{iv_mid*100:.3f}%" if iv_mid else "No root")

    st.subheader("Charts")
    S_min = max(0.01, S0 * 0.4)
    S_max = S0 * 1.6
    S_grid = np.linspace(S_min, S_max, 200)
    payoff_call = np.maximum(S_grid - K, 0.0)
    payoff_put = np.maximum(K - S_grid, 0.0)

    fig1, ax1 = plt.subplots()
    ax1.plot(S_grid, payoff_call if opt_type == "Call" else payoff_put, label=f"{opt_type} payoff at expiry")
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
    st.dataframe(pd.DataFrame(rows).style.format({"Spot": "{:.4f}", "Call": "{:.4f}", "Put": "{:.4f}"}), use_container_width=True)

# =============================
# TAB 2: American (CRR)
# =============================
with tab2:
    st.subheader("American vs European (CRR Binomial)")
    steps = st.slider("CRR steps (accuracy vs speed)", min_value=25, max_value=1000, value=200, step=25, key="crr_steps")
    inp = OptionInput(S0=S0, K=K, r=r, sigma=sigma, T=T, q=q)
    otype = opt_type.lower()
    euro_tree = crr_price_european(inp, otype, steps)
    amer_tree = crr_price_american(inp, otype, steps)
    bs_euro = (bs_prices(inp)[0] if otype == "call" else bs_prices(inp)[1])

    c1, c2, c3 = st.columns(3)
    c1.metric("CRR European", f"{euro_tree:,.6f}")
    c2.metric("BS European", f"{bs_euro:,.6f}", delta=f"{(euro_tree - bs_euro):+.6f}")
    c3.metric("CRR American", f"{amer_tree:,.6f}")
    st.caption(f"Early-exercise premium vs BS: {amer_tree - bs_euro:+.6f}.")

# =============================
# TAB 3: Chain & Smiles
# =============================
with tab3:
    st.subheader("Options Chain (CSV) → Bulk IV & Smiles")
    chain_file = st.file_uploader("Upload CSV with: K, Mid or Bid+Ask or Price; optional T (years), type (call/put)", type=["csv"], key="chain_upl")
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
                st.error("CSV must include strike column [K/strike].")
            else:
                df = chain_df.copy()
                df["K"] = df[col_K].astype(float)
                if col_mid is not None:
                    price_series = df[col_mid].astype(float)
                elif col_bid is not None and col_ask is not None:
                    price_series = (df[col_bid].astype(float) + df[col_ask].astype(float)) / 2.0
                elif col_px is not None:
                    price_series = df[col_px].astype(float)
                else:
                    st.error("Provide Mid or Bid+Ask or Price column.")
                    price_series = None

                if col_T is not None:
                    df["T_row"] = df[col_T].astype(float)
                else:
                    df["T_row"] = T
                if col_ty is not None:
                    df["otype"] = df[col_ty].astype(str).str.lower().str.strip().replace({"c": "call", "p": "put"})
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

                    st.dataframe(df[["K", "T_row", "otype", "MarketPrice", "IV_%", "Tag"]].rename(columns={"T_row": "T"}).style.format({"MarketPrice": "{:.4f}", "IV_%": "{:.3f}"}), use_container_width=True)

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
                    st.download_button("Download processed chain (CSV)", data=out, file_name="options_chain_with_iv.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process CSV: {e}")

# =============================
# TAB 4: Vol Surface
# =============================
with tab4:
    st.subheader("Volatility Surface (K × T)")
    surf_file = st.file_uploader("Upload multi-expiry CSV (K, T, price fields; optional type)", type=["csv"], key="surf_upl")
    if surf_file is not None:
        try:
            raw = pd.read_csv(surf_file)
            df_iv = compute_chain_iv(raw, S0=S0, r=r, q=q, default_T=T, default_type=opt_type.lower(), sigma_seed=max(sigma, 1e-6))
            st.dataframe(df_iv[["K", "T", "otype", "MarketPrice", "IV_%", "Tag"]].sort_values(["T", "K"]).style.format({"MarketPrice": "{:.4f}", "IV_%": "{:.3f}"}), use_container_width=True)
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

                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                K_arr, T_arr, IVp_arr = prepare_surface_arrays(valid)
                fig3 = plt.figure(); ax3 = fig3.add_subplot(111, projection="3d")
                ax3.scatter(K_arr, T_arr, IVp_arr)
                ax3.set_xlabel("Strike (K)"); ax3.set_ylabel("Expiry (T, years)"); ax3.set_zlabel("IV (%)")
                ax3.set_title("IV Surface — 3D scatter"); st.pyplot(fig3, use_container_width=True)

                fig4, ax4 = plt.subplots()
                try:
                    tric = ax4.tricontourf(K_arr, T_arr, IVp_arr, levels=12)
                    fig4.colorbar(tric, ax=ax4, label="IV (%)")
                    ax4.set_xlabel("Strike (K)"); ax4.set_ylabel("Expiry (T, years)")
                    ax4.set_title("IV Surface — contour"); st.pyplot(fig4, use_container_width=True)
                except Exception:
                    st.info("Not enough distinct points for a filled contour; 3D scatter still shows shape.")
            out2 = df_iv.sort_values(["T", "K"]).to_csv(index=False).encode("utf-8")
            st.download_button("Download IV dataset (CSV)", data=out2, file_name="iv_surface_dataset.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error computing IV: {e}")

# =============================
# TAB 5: Bonds (numeric & dates)
# =============================
with tab5:
    st.subheader("Bond Pricer — Price, YTM, Duration & Convexity")

    mode_bond_input = st.radio("Input mode", ["Numeric T", "Dates"], horizontal=True)

    bc1, bc2, bc3 = st.columns(3)
    face = bc1.number_input("Face/Redemption", min_value=1.0, value=100.0, step=1.0)
    coupon_pct = bc2.number_input("Coupon rate (% p.a.)", value=5.00, step=0.25, format="%.2f") / 100.0
    freq = int(bc3.selectbox("Payments per year", options=[1, 2, 4], index=1))

    if mode_bond_input == "Numeric T":
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
            c1.metric("Price", f"{P:,.4f}"); c2.metric("Macaulay Dur (yrs)", f"{mac:,.4f}")
            c3.metric("Modified Dur (yrs)", f"{mod:,.4f}"); c4.metric("Convexity", f"{conv:,.4f}")
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
                c1.metric("Solved YTM", f"{ytm_solved*100:.4f}%"); c2.metric("Macaulay Dur (yrs)", f"{mac:,.4f}")
                c3.metric("Modified Dur (yrs)", f"{mod:,.4f}"); c4.metric("Convexity", f"{conv:,.4f}")

    else:
        colD1, colD2, colD3 = st.columns(3)
        val_d_str = colD1.text_input("Settlement / Valuation date (YYYY-MM-DD)", value=str(val_date))
        mat_d_str = colD2.text_input("Maturity date (YYYY-MM-DD)", value=str(date(val_date.year + 5, val_date.month, min(val_date.day, 28))))
        dc_bond: DayCount = colD3.selectbox("Day-count", ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"], index=0)
        biz: BizConv = st.selectbox("Business-day convention", ["Following", "Modified Following", "Preceding"], index=0)

        try:
            settle_d = parse_date(val_d_str)
            mat_d = parse_date(mat_d_str)
        except Exception:
            st.error("Invalid date format. Use YYYY-MM-DD.")
            settle_d, mat_d = val_date, val_date

        ytm_pct_d = st.number_input("Yield to maturity (% p.a.)", value=6.00, step=0.10, format="%.2f") / 100.0
        mode_bond_d = st.radio("Mode (dates)", ["Inputs → Price", "Price → YTM"], horizontal=True)

        if mode_bond_d == "Inputs → Price":
            P = price_bond_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)
            mac = macaulay_duration_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)
            mod = modified_duration_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"{P:,.4f}"); c2.metric("Macaulay Dur (yrs)", f"{mac:,.4f}"); c3.metric("Modified Dur (yrs)", f"{mod:,.4f}")
        else:
            target_price = st.number_input("Target clean price (dates)", min_value=0.01, value=100.00, step=0.25)
            ytm_solved = ytm_from_price_dates(target_price, face, coupon_pct, settle_d, mat_d, freq, dc_bond, biz)
            if ytm_solved is None:
                st.error("Could not solve YTM in bounds [-99%, 100%]. Check inputs.")
            else:
                mac = macaulay_duration_dates(face, coupon_pct, ytm_solved, settle_d, mat_d, freq, dc_bond, biz)
                mod = modified_duration_dates(face, coupon_pct, ytm_solved, settle_d, mat_d, freq, dc_bond, biz)
                c1, c2, c3 = st.columns(3)
                c1.metric("Solved YTM", f"{ytm_solved*100:.4f}%"); c2.metric("Macaulay Dur (yrs)", f"{mac:,.4f}"); c3.metric("Modified Dur (yrs)", f"{mod:,.4f}")

# =============================
# TAB 6: Yield Curve (Bootstrapping)
# =============================
with tab6:
    st.subheader("Yield Curve — Bootstrapping")
    st.caption("Provide deposits (zero instruments) and optional coupon bonds. Use numeric T or dates.")

    curve_mode = st.radio("Curve input mode", ["Numeric", "Dates"], horizontal=True)

    deposits = []
    bonds = []

    if curve_mode == "Numeric":
        st.markdown("**Deposits (T, Rate%)**")
        dep_df = st.data_editor(pd.DataFrame({"T": [0.5, 1.0, 2.0], "Rate (%)": [3.0, 3.2, 3.5]}), num_rows="dynamic", use_container_width=True, key="dep_num")
        for _, r in dep_df.iterrows():
            try:
                deposits.append((float(r["T"]), float(r["Rate (%)"]) / 100.0))
            except Exception:
                pass

        st.markdown("**Coupon Bonds**")
        bond_df = st.data_editor(pd.DataFrame({"T": [3.0], "Price": [99.50], "Coupon (%)": [4.0], "Face": [100.0], "Freq": [2]}), num_rows="dynamic", use_container_width=True, key="bond_num")
        for _, r in bond_df.iterrows():
            try:
                bonds.append({"T": float(r["T"]), "price": float(r["Price"]), "coupon": float(r["Coupon (%)"]) / 100.0, "face": float(r["Face"]), "freq": int(r["Freq"])})
            except Exception:
                pass

        if st.button("Bootstrap Curve (Numeric)", type="primary"):
            try:
                curve = bootstrap_curve(deposits, bonds)
                st.session_state["bootstrapped_curve"] = curve
                st.success("Curve bootstrapped.")
            except Exception as e:
                st.error(f"Bootstrapping error: {e}")

    else:
        dc_curve: DayCount = st.selectbox("Day-count for T", ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"], index=0)
        val_date_str2 = st.text_input("Valuation date (YYYY-MM-DD)", value=str(val_date), key="curve_valdate")

        try:
            curve_val_date = parse_date(val_date_str2)
        except Exception:
            st.error("Invalid valuation date; using today.")
            curve_val_date = date.today()

        st.markdown("**Deposits (Maturity Date, Rate%)**")
        dep_df2 = st.data_editor(pd.DataFrame({"Maturity": [str(curve_val_date)], "Rate (%)": [3.0]}), num_rows="dynamic", use_container_width=True, key="dep_dt")
        for _, r in dep_df2.iterrows():
            try:
                d = parse_date(str(r["Maturity"]))
                deposits.append({"date": d, "rate": float(r["Rate (%)"]) / 100.0})
            except Exception:
                pass

        st.markdown("**Coupon Bonds**")
        bond_df2 = st.data_editor(pd.DataFrame({"Maturity": [str(curve_val_date)], "Price": [99.50], "Coupon (%)": [4.0], "Face": [100.0], "Freq": [2]}), num_rows="dynamic", use_container_width=True, key="bond_dt")
        for _, r in bond_df2.iterrows():
            try:
                bonds.append({"maturity_date": parse_date(str(r["Maturity"])), "price": float(r["Price"]), "coupon": float(r["Coupon (%)"]) / 100.0, "face": float(r["Face"]), "freq": int(r["Freq"]), "day_count": dc_curve})
            except Exception:
                pass

        if st.button("Bootstrap Curve (Dates)", type="primary"):
            try:
                curve = bootstrap_curve(deposits, bonds, valuation_date=curve_val_date, day_count=dc_curve)
                st.session_state["bootstrapped_curve"] = curve
                st.success("Curve bootstrapped.")
            except Exception as e:
                st.error(f"Bootstrapping error: {e}")

    curve: YieldCurve | None = st.session_state.get("bootstrapped_curve")
    if curve is not None:
        st.write("**Zero Curve**")
        st.dataframe(curve.as_dataframe().style.format({"Zero(%)": "{:.3f}", "DF": "{:.6f}"}), use_container_width=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.plot(curve.mats, [z * 100 for z in curve.zeros], marker="o"); ax1.set_title("Spot (Zero) Curve"); ax1.set_xlabel("T (years)"); ax1.set_ylabel("Zero (%)")
        ax2.plot(curve.mats, curve.dfs, marker="o"); ax2.set_title("Discount Factors"); ax2.set_xlabel("T (years)"); ax2.set_ylabel("DF")
        st.pyplot(fig, use_container_width=True)
        colf1, colf2 = st.columns(2)
        t1 = colf1.number_input("Forward start T1 (yrs)", value=1.0, step=0.25, min_value=0.0)
        t2 = colf2.number_input("Forward end T2 (yrs)", value=2.0, step=0.25, min_value=0.01)
        st.metric("Annual forward rate T1→T2", f"{curve.get_fwd(t1, t2) * 100:.3f}%")

# =============================
# TAB 7: Swaps
# =============================
with tab7:
    st.subheader("Plain-Vanilla Interest Rate Swap")
    st.caption("Uses the bootstrapped curve from the previous tab.")
    curve: YieldCurve | None = st.session_state.get("bootstrapped_curve")
    if curve is None:
        st.info("Bootstrap a curve first in the **Yield Curve** tab.")
    else:
        colS1, colS2, colS3 = st.columns(3)
        notional = colS1.number_input("Notional", min_value=1_000.0, value=1_000_000.0, step=50_000.0)
        T_swap = colS2.number_input("Maturity T (years)", min_value=0.5, value=5.0, step=0.5)
        freq = int(colS3.selectbox("Payments per year", options=[1, 2, 4], index=1))

        s_par = par_swap_rate(curve, T_swap, freq)
        st.metric("Par swap rate", f"{s_par * 100:.4f}%")

        fixed_rate_user = st.number_input("Fixed rate to value (% p.a.)", value=round(s_par * 100, 4), step=0.01, format="%.4f") / 100.0
        st.metric("Swap PV (Fixed - Float)", f"{swap_pv(curve, T_swap, fixed_rate_user, notional, freq):,.2f}")
        st.metric("DV01 (per +1bp shift)", f"{dv01(curve, T_swap, fixed_rate_user, notional, freq, bp=1.0):,.2f}")

st.caption("Date-aware maturities with day-counts; options (BSM/CRR), IV smiles & surfaces; bonds (price, YTM, duration); bootstrapped curve; swaps (par, PV, DV01).")
