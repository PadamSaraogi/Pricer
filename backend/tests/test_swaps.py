# tests/test_swaps.py
from yield_curve import bootstrap_curve
from swaps import par_swap_rate, swap_pv, dv01

def test_par_rate_makes_pv_zero():
    deposits = [(1.0, 0.03), (2.0, 0.032), (3.0, 0.035)]
    curve = bootstrap_curve(deposits, bonds=[])
    T = 3.0
    freq = 2
    s_par = par_swap_rate(curve, T, freq)
    pv = swap_pv(curve, T, s_par, notional=1_000_000, freq=freq)
    assert abs(pv) < 1.0  # near zero in currency units

def test_dv01_positive_for_receiver():
    deposits = [(1.0, 0.02), (2.0, 0.025), (5.0, 0.03)]
    curve = bootstrap_curve(deposits, bonds=[])
    T = 5.0
    freq = 2
    s_par = par_swap_rate(curve, T, freq)
    # Receiver of fixed at par has DV01 ~ annuity * notional * 1bp (sign depends on definition)
    pv_shift = dv01(curve, T, s_par, notional=1_000_000, freq=freq, bp=1.0)
    # Parallel +1bp raises discount rates -> lowers fixed leg PV more than float, PV decreases -> dv01 negative
    assert pv_shift < 0
