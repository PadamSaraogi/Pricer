# tests/test_black76_curve.py
import math
from yield_curve import YieldCurve
from black76 import black76_price_df, implied_vol_b76, black76_price


def make_flat_curve(r_annual: float):
    mats = [0.5, 1.0, 2.0]
    dfs = [1.0 / ((1.0 + r_annual) ** t) for t in mats]
    zeros = [r_annual for _ in mats]
    return YieldCurve(mats, zeros, dfs)


def test_df_vs_flat_r_consistency_when_r_cont_matches_DF():
    F0, K, T, sigma = 100.0, 100.0, 1.0, 0.25
    curve = make_flat_curve(0.05)            # annual comp 5%
    DF_T = curve.get_df(T)                   # DF(T) = 1/(1+0.05)^1
    r_cont = -math.log(DF_T) / T             # convert DF to continuous r so flat-r price is apples-to-apples

    p_df = black76_price_df(F0, K, DF_T, T, sigma, "call")
    p_r  = black76_price(F0, K, r_cont, T, sigma, "call")
    assert abs(p_df - p_r) < 1e-10


def test_iv_solver_with_df():
    F0, K, T, sigma_true = 120.0, 100.0, 0.75, 0.35
    curve = make_flat_curve(0.03)
    DF_T = curve.get_df(T)
    price = black76_price_df(F0, K, DF_T, T, sigma_true, "put")
    iv = implied_vol_b76(price, F0, K, r=0.0, T=T, otype="put", DF_T=DF_T)
    assert iv is not None
    assert abs(iv - sigma_true) < 1e-4
