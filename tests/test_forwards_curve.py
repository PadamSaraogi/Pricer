# tests/test_forwards_curve.py
import numpy as np
from math import log, exp

from forwards import forward_price_from_curve, discount_factor_from_curve
from yield_curve import YieldCurve


def make_flat_curve(r_annual: float):
    # DF(T) = 1 / (1 + r)^T
    mats = [0.5, 1.0, 2.0, 5.0]
    dfs = [1.0 / ((1.0 + r_annual) ** t) for t in mats]
    zeros = [r_annual for _ in mats]
    return YieldCurve(mats, zeros, dfs)


def test_forward_from_curve_matches_formula():
    curve = make_flat_curve(0.05)  # 5% annual compounded
    S0, T, q = 100.0, 2.0, 0.01  # q is continuous
    dfT = discount_factor_from_curve(curve, T)
    F_expected = S0 * np.exp(-q * T) / dfT
    F_curve = forward_price_from_curve(S0, curve, T, q_cont=q)
    assert abs(F_curve - F_expected) < 1e-12
