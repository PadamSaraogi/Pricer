"""
Unit tests for bonds.py
Covers:
- Annual and monthly coupon bond pricing
- YTM inversion accuracy
- Robustness for par and discount bonds
"""

from bonds import price_bond, yield_to_maturity


def test_annual_par_bond():
    """At par: coupon rate = YTM."""
    price = price_bond(face=100, coupon_rate=0.06, ytm=0.06, T=1, freq=1)
    assert abs(price - 100.0) < 1e-6


def test_monthly_coupon_bond_par():
    """Monthly-paying par bond should price at 100 when coupon == YTM."""
    price = price_bond(face=100, coupon_rate=0.06, ytm=0.06, T=1, freq=12)
    assert abs(price - 100.0) < 1e-6


def test_discount_bond_price_lower_than_par():
    """Discount bond should price below 100."""
    price = price_bond(face=100, coupon_rate=0.05, ytm=0.06, T=5, freq=2)
    assert price < 100


def test_premium_bond_price_above_par():
    """Premium bond should price above 100."""
    price = price_bond(face=100, coupon_rate=0.08, ytm=0.06, T=5, freq=2)
    assert price > 100


def test_monthly_coupon_bond_inverse_ytm():
    """YTM inversion for monthly coupon bond."""
    face, coupon, T, freq = 100, 0.07, 5, 12
    ytm_true = 0.065
    price = price_bond(face, coupon, ytm_true, T, freq)
    ytm_back = yield_to_maturity(price, face, coupon, T, freq)
    assert abs(ytm_true - ytm_back) < 1e-6


def test_ytm_convergence_accuracy():
    """Ensure YTM solver converges to within tolerance."""
    price = 98.5
    ytm = yield_to_maturity(price=price, face=100, coupon_rate=0.06, T=5, freq=2)
    assert 0.05 < ytm < 0.07
