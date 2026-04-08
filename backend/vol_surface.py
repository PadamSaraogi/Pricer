"""
vol_surface.py
Utilities to compute Implied Volatility (IV) for multi-expiry option chains
and prepare data for Volatility Smile/Surface visualization.

Input expectation (flexible headers; case-insensitive):
- Required:  K  (or 'strike')
- One of:    Mid  OR (Bid + Ask) OR Price/Last/Market_Price
- Optional:  T (years), type (call/put or c/p)

Usage:
    from vol_surface import compute_chain_iv
    df_iv = compute_chain_iv(df, S0=100, r=0.05, q=0.0, default_T=0.5, default_type="call")
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from bs_core import OptionInput, implied_vol, bs_greeks, moneyness_tags


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a dict mapping lowercase/stripped names -> original column names, for flexible access.
    """
    return {c.lower().strip(): c for c in df.columns}


def _pick_price_series(df: pd.DataFrame, cols: Dict[str, str]) -> pd.Series:
    """
    Choose a market price series: prefer 'mid', else (bid+ask)/2, else 'price/last/market_price'.
    Raises ValueError if none found.
    """
    col_mid = next((cols[n] for n in ("mid", "mkt_mid") if n in cols), None)
    col_bid = cols.get("bid")
    col_ask = cols.get("ask")
    col_px = next((cols[n] for n in ("price", "last", "market_price") if n in cols), None)

    if col_mid is not None:
        return df[col_mid].astype(float)
    if col_bid is not None and col_ask is not None:
        return (df[col_bid].astype(float) + df[col_ask].astype(float)) / 2.0
    if col_px is not None:
        return df[col_px].astype(float)

    raise ValueError("Provide either Mid, or Bid+Ask, or Price column.")


def _resolve_type_series(df: pd.DataFrame, cols: Dict[str, str], default_type: str) -> pd.Series:
    """
    Resolve option type per row; default to 'default_type' if absent/invalid.
    """
    col_ty = next((cols[n] for n in ("type", "option_type") if n in cols), None)
    if col_ty is None:
        return pd.Series([default_type.lower()] * len(df), index=df.index)
    s = (
        df[col_ty]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"c": "call", "p": "put"})
    )
    s.loc[~s.isin(["call", "put"])] = default_type.lower()
    return s


def compute_chain_iv(
    df: pd.DataFrame,
    S0: float,
    r: float,
    q: float,
    default_T: float,
    default_type: str = "call",
    sigma_seed: float = 0.20,
) -> pd.DataFrame:
    """
    Compute IV for each row of a (possibly multi-expiry) options chain DataFrame.

    Returns a copy of the DataFrame with added columns:
      - K (float), T (float), otype (call/put), MarketPrice (float)
      - IV (as decimal), IV_% (percent), Tag (moneyness label)
    Rows where IV cannot be solved get NaN in IV fields and Tag='no-root'.
    """
    if S0 <= 0 or default_T <= 0:
        raise ValueError("S0 and default_T must be positive.")

    cols = _normalize_columns(df)
    col_K = cols.get("k") or cols.get("strike")
    if col_K is None:
        raise ValueError("CSV must include a strike column: one of [K, strike].")

    # Build base frame
    out = df.copy()
    out["K"] = out[col_K].astype(float)

    # Time to maturity
    col_T = cols.get("t") or cols.get("ttm") or cols.get("maturity") or cols.get("time_to_maturity_years")
    if col_T is not None:
        out["T"] = out[col_T].astype(float)
    else:
        out["T"] = float(default_T)

    # Option type per row
    out["otype"] = _resolve_type_series(out, cols, default_type)

    # Market price
    price_series = _pick_price_series(out, cols)
    out["MarketPrice"] = price_series.astype(float)

    # Compute IV and moneyness
    iv_list = []
    tag_list = []
    for k, px, t, ty in zip(out["K"], out["MarketPrice"], out["T"], out["otype"]):
        _inp = OptionInput(S0=S0, K=float(k), r=r, sigma=max(sigma_seed, 1e-6), T=float(t), q=q)
        iv = implied_vol(float(px), _inp, ty)
        if iv is None:
            iv_list.append(np.nan)
            tag_list.append("no-root")
        else:
            iv_list.append(iv)
            g = bs_greeks(_inp)  # to reuse d1 for tag
            tag_list.append(moneyness_tags(S0, float(k), g["d1"])["tag"])

    out["IV"] = iv_list
    out["IV_%"] = out["IV"] * 100.0
    out["Tag"] = tag_list
    return out


def prepare_surface_arrays(df_iv: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a DataFrame returned by compute_chain_iv, extract arrays (K, T, IV_%)
    dropping NaNs, suitable for 3D scatter and tricontourf plotting.
    """
    valid = df_iv.dropna(subset=["IV"])
    K = valid["K"].to_numpy(dtype=float)
    T = valid["T"].to_numpy(dtype=float)
    IVp = (valid["IV"] * 100.0).to_numpy(dtype=float)
    return K, T, IVp
