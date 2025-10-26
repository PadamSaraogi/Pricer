import math
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
