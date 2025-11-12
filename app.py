import streamlit as st
import pandas as pd
from datetime import date
from bonds import price_bond, ytm_from_price, macaulay_duration, modified_duration, convexity_numeric, price_bond_dates, ytm_from_price_dates, macaulay_duration_dates, modified_duration_dates
from daycount import year_fraction
from date_utils import parse_date
from reports import build_bond_report_pdf

# Streamlit page configuration
st.set_page_config(page_title="Bond & Option Pricer", layout="wide")

# Title for the dashboard
st.title("📊 Bond & Option Pricing Dashboard")

# Sidebar navigation
tabs = ["Home", "Bond Pricing", "Option Pricing", "Reports"]
selected_tab = st.sidebar.radio("Choose a tab", tabs)

# ========== Bond Pricing Tab ==========

if selected_tab == "Bond Pricing":
    st.subheader("Bond Pricer — Price, YTM, Duration & Convexity")

    # Set default bond values for input
    st.markdown("### Enter bond details")

    # Bond parameters input
    col1, col2 = st.columns(2)
    with col1:
        face = st.number_input("Face/Redemption Value (₹)", min_value=1.0, value=100.0, step=1.0)
        coupon_pct = st.number_input("Coupon Rate (% p.a.)", value=5.00, step=0.25, format="%.2f") / 100.0
    with col2:
        freq = st.selectbox("Coupon Frequency", options=[1, 2, 4, 12], index=1)  # 12 = monthly, 4 = quarterly, 2 = semiannual, 1 = annual
        maturity_years = st.number_input("Years to Maturity (T)", min_value=0.25, value=5.0, step=0.25)

    # Mode for bond calculation
    mode_bond_input = st.radio("Input Mode", ["Numeric T", "Dates"], index=0)

    if mode_bond_input == "Numeric T":
        # Input for YTM and Price Calculation Mode
        col3, col4 = st.columns(2)
        with col3:
            ytm_pct = st.number_input("Yield to Maturity (YTM, % p.a.)", value=6.00, step=0.10, format="%.2f") / 100.0
        with col4:
            mode_bond = st.radio("Mode", ["Inputs → Price", "Price → YTM"], horizontal=True)

        if mode_bond == "Inputs → Price":
            P = price_bond(face, coupon_pct, ytm_pct, maturity_years, freq)
            mac = macaulay_duration(face, coupon_pct, ytm_pct, maturity_years, freq)
            mod = modified_duration(face, coupon_pct, ytm_pct, maturity_years, freq)
            conv = convexity_numeric(face, coupon_pct, ytm_pct, maturity_years, freq)

            st.metric("Price", f"₹{P:.4f}")
            st.metric("Macaulay Duration (years)", f"{mac:.4f}")
            st.metric("Modified Duration (years)", f"{mod:.4f}")
            st.metric("Convexity", f"{conv:.4f}")

        else:
            target_price = st.number_input("Target Clean Price (₹)", min_value=0.01, value=100.0, step=0.25)
            ytm_solved = ytm_from_price(target_price, face, coupon_pct, maturity_years, freq)
            if ytm_solved is None:
                st.error("Could not solve YTM within bounds [-99%, 100%]. Check inputs.")
            else:
                mac = macaulay_duration(face, coupon_pct, ytm_solved, maturity_years, freq)
                mod = modified_duration(face, coupon_pct, ytm_solved, maturity_years, freq)
                conv = convexity_numeric(face, coupon_pct, ytm_solved, maturity_years, freq)

                st.metric("Solved YTM", f"{ytm_solved*100:.4f}%")
                st.metric("Macaulay Duration (years)", f"{mac:.4f}")
                st.metric("Modified Duration (years)", f"{mod:.4f}")
                st.metric("Convexity", f"{conv:.4f}")

    else:
        # Dates input for bond calculation
        colD1, colD2, colD3 = st.columns(3)
        val_d_str = colD1.text_input("Settlement Date (YYYY-MM-DD)", value=str(date.today()))
        mat_d_str = colD2.text_input("Maturity Date (YYYY-MM-DD)", value=str(date.today().replace(year=date.today().year + 5)))
        dc_bond: DayCount = colD3.selectbox("Day-count Convention", ["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"], index=0)
        biz: BizConv = st.selectbox("Business-day Convention", ["Following", "Modified Following", "Preceding"], index=0)

        try:
            settle_d = parse_date(val_d_str)
            mat_d = parse_date(mat_d_str)
        except Exception:
            st.error("Invalid date format. Use YYYY-MM-DD.")
            settle_d, mat_d = date.today(), date.today()

        ytm_pct_d = st.number_input("Yield to Maturity (YTM, % p.a.)", value=6.00, step=0.10, format="%.2f") / 100.0
        mode_bond_d = st.radio("Mode (Dates)", ["Inputs → Price", "Price → YTM"], horizontal=True)

        if mode_bond_d == "Inputs → Price":
            P = price_bond_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)
            mac = macaulay_duration_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)
            mod = modified_duration_dates(face, coupon_pct, ytm_pct_d, settle_d, mat_d, freq, dc_bond, biz)

            st.metric("Price", f"₹{P:.4f}")
            st.metric("Macaulay Duration (years)", f"{mac:.4f}")
            st.metric("Modified Duration (years)", f"{mod:.4f}")

        else:
            target_price = st.number_input("Target Clean Price (₹)", min_value=0.01, value=100.0, step=0.25)
            ytm_solved = ytm_from_price_dates(target_price, face, coupon_pct, settle_d, mat_d, freq, dc_bond, biz)
            if ytm_solved is None:
                st.error("Could not solve YTM within bounds [-99%, 100%]. Check inputs.")
            else:
                mac = macaulay_duration_dates(face, coupon_pct, ytm_solved, settle_d, mat_d, freq, dc_bond, biz)
                mod = modified_duration_dates(face, coupon_pct, ytm_solved, settle_d, mat_d, freq, dc_bond, biz)

                st.metric("Solved YTM", f"{ytm_solved*100:.4f}%")
                st.metric("Macaulay Duration (years)", f"{mac:.4f}")
                st.metric("Modified Duration (years)", f"{mod:.4f}")

    # --------------------------- Generate Reports ---------------------------

    st.markdown("### Generate Bond Report PDF")
    if st.button("Download Bond Report (PDF)"):
        pdf_data = build_bond_report_pdf(
            face=face,
            coupon_rate=coupon_pct,
            ytm=ytm_pct,
            T=maturity_years,
            freq=freq,
            price=P,
            macaulay_dur=mac,
            modified_dur=mod,
            convexity=conv,
            valuation_date_str=str(date.today())
        )
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="bond_report.pdf",
            mime="application/pdf"
        )

# ========== Option Pricing Tab ==========

elif selected_tab == "Option Pricing":
    st.subheader("Option Pricer — Black-Scholes Model")

    # Add code for Option pricing here (if needed)

# ========== Reports Tab ==========

elif selected_tab == "Reports":
    st.subheader("Generate Reports")
    
    # Add code for reports here (e.g., export bond reports)
