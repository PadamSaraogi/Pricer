# tests/test_vol_surface.py
import math
import numpy as np
import pandas as pd

from bs_core import OptionInput, bs_prices
from vol_surface import compute_chain_iv, prepare_surface_arrays


def approx(a, b, tol=1e-3):
    return abs(a - b) < tol


def test_compute_chain_iv_recovers_sigma_on_synthetic_calls():
    """
    Build a synthetic multi-expiry chain using BS prices with a known sigma,
    then ensure the IV solver recovers ~that sigma across rows.
    """
    S0, r, q = 100.0, 0.05, 0.00
    true_sigma = 0.20

    rows = []
    for T in [0.25, 0.5, 1.0]:
        for K in [90.0, 100.0, 110.0]:
            inp = OptionInput(S0=S0, K=K, r=r, sigma=true_sigma, T=T, q=q)
            call_px, _, *_ = bs_prices(inp)
            rows.append({"K": K, "T": T, "Price": call_px, "type": "call"})
    df = pd.DataFrame(rows)

    out = compute_chain_iv(
        df,
        S0=S0,
        r=r,
        q=q,
        default_T=0.5,
        default_type="call",
        sigma_seed=0.15,
    )

    # All IVs should be close to true_sigma
    ivs = out["IV"].dropna().to_numpy()
    assert len(ivs) == len(df)
    assert np.all(np.isfinite(ivs))
    assert np.all(np.abs(ivs - true_sigma) < 5e-3)  # tight since prices are exact BS


def test_mid_price_is_used_when_bid_ask_present():
    """
    If Bid/Ask are provided, compute_chain_iv should use their mid as MarketPrice.
    """
    S0, r, q, T, sigma = 100.0, 0.03, 0.00, 0.5, 0.25
    K = 100.0
    inp = OptionInput(S0=S0, K=K, r=r, sigma=sigma, T=T, q=q)
    call_px, _, *_ = bs_prices(inp)

    df = pd.DataFrame(
        [
            {
                "K": K,
                "T": T,
                "Bid": call_px - 0.10,
                "Ask": call_px + 0.10,
                "type": "call",
            }
        ]
    )

    out = compute_chain_iv(
        df,
        S0=S0,
        r=r,
        q=q,
        default_T=T,
        default_type="call",
        sigma_seed=0.10,
    )

    expected_mid = (df.loc[0, "Bid"] + df.loc[0, "Ask"]) / 2.0
    assert approx(out.loc[0, "MarketPrice"], expected_mid, tol=1e-9)
    # IV should still be close to true sigma (minor difference due to rounding)
    assert abs(out.loc[0, "IV"] - sigma) < 5e-3


def test_prepare_surface_arrays_shapes_and_types():
    """
    prepare_surface_arrays should return finite arrays matching valid rows.
    """
    # Small synthetic dataset with two expiries, three strikes
    S0, r, q = 100.0, 0.05, 0.0
    sigma = 0.2
    rows = []
    for T in [0.25, 0.5]:
        for K in [90.0, 100.0, 110.0]:
            px, _, *_ = bs_prices(OptionInput(S0, K, r, sigma, T, q))
            rows.append({"K": K, "T": T, "Price": px, "type": "call"})
    df = pd.DataFrame(rows)

    out = compute_chain_iv(
        df,
        S0=S0,
        r=r,
        q=q,
        default_T=0.5,
        default_type="call",
        sigma_seed=0.20,
    )

    K_arr, T_arr, IVp_arr = prepare_surface_arrays(out)
    n = len(df)
    assert K_arr.shape == (n,)
    assert T_arr.shape == (n,)
    assert IVp_arr.shape == (n,)

    assert np.all(np.isfinite(K_arr))
    assert np.all(np.isfinite(T_arr))
    assert np.all(np.isfinite(IVp_arr))
