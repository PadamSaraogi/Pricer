# tests/test_yield_curve.py
import numpy as np
from yield_curve import bootstrap_curve, df_from_zero

def test_bootstrap_with_deposits_only():
    # 3 zero points should interpolate/extrapolate smoothly
    deposits = [(0.5, 0.03), (1.0, 0.032), (2.0, 0.035)]
    curve = bootstrap_curve(deposits, bonds=[])
    assert len(curve.mats) == 3
    # Check df mapping consistency
    for T, r in deposits:
        assert abs(curve.get_df(T) - df_from_zero(r, T)) < 1e-12

def test_bootstrap_with_bond():
    # Start with a short zero, add a 2Y coupon bond to back out DF(2Y)
    deposits = [(1.0, 0.02)]
    # 2Y bond: 5% coupon, semiannual, price ~ 104.9 if rates ~2% flat
    bonds = [{"T": 2.0, "price": 104.9, "coupon": 0.05, "face": 100.0, "freq": 2}]
    curve = bootstrap_curve(deposits, bonds)
    assert any(abs(t - 2.0) < 1e-12 for t in curve.mats)
    df2 = curve.get_df(2.0)
    assert 0.90 < df2 < 1.05  # sanity
