# tests/test_forwards.py
import numpy as np
from forwards import forward_price, ladder_forward_prices, fair_value_from_forward


def test_forward_basic_parity():
    S0, r, q, T = 100.0, 0.05, 0.02, 0.5
    F0 = forward_price(S0, r, q, T)
    # Analytical check
    assert abs(F0 - S0 * np.exp((r - q) * T)) < 1e-12
    # Discounted forward level equals PV via exp(-rT)
    pv = fair_value_from_forward(F0, r, T)
    assert abs(pv - F0 * np.exp(-r * T)) < 1e-12


def test_ladder_monotonic_in_spot():
    r, q, T = 0.03, 0.01, 1.25
    S_grid = np.linspace(50, 150, 11)
    F_grid = ladder_forward_prices(100.0, r, q, T, S_grid)
    assert F_grid[0] < F_grid[-1]
    # linear relation given constant carry
    ratio = F_grid[1] / S_grid[1]
    assert np.allclose(F_grid / S_grid, ratio, atol=1e-12)
