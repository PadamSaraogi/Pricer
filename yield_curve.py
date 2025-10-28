"""
yield_curve.py
Simple bootstrapped yield curve utility.

- Inputs:
  * Deposits (zero instruments): list of (maturity_years, annual_rate)
  * Coupon bonds: list of dicts with keys:
      {'T': years, 'price': clean_price, 'coupon': annual_coupon_rate, 'face': face_value, 'freq': payments_per_year}
- Conventions:
  * Annual compounding (nominal) throughout
  * Discount factor DF(t) = 1 / (1 + r(t))**t
  * For coupon bonds, we solve the last period DF from price, using earlier DFs.

- Outputs:
  * YieldCurve object with:
      maturities (sorted)
      zeros (spot annual rates per maturity)
      dfs (discount factors)
    Methods:
      get_df(t), get_zero(t), get_fwd(t1,t2), as_dataframe()
      bumped(bp): +bp to all zero rates, returns a new curve.

- Interpolation:
  * ln(DF) linear in t (more stable than DF linear).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import bisect
import pandas as pd


def df_from_zero(r: float, t: float) -> float:
    return 1.0 / ((1.0 + r) ** t)


def zero_from_df(df: float, t: float) -> float:
    if t <= 0:
        return 0.0
    return (df ** (-1.0 / t)) - 1.0


@dataclass
class YieldCurve:
    mats: List[float]   # maturities (years), ascending
    zeros: List[float]  # annual spot rates
    dfs: List[float]    # discount factors

    # ---- core accessors ----
    def get_df(self, t: float) -> float:
        """Interpolate ln DF linearly by maturity."""
        if t <= 0:
            return 1.0
        if t <= self.mats[0]:
            # linear lnDF extrapolation to 0..first
            ln_df0 = math.log(self.dfs[0])
            Rt = ln_df0 / self.mats[0]
            return math.exp(Rt * t)
        if t >= self.mats[-1]:
            # flat in lnDF slope beyond last two points
            i0 = len(self.mats) - 2
            i1 = len(self.mats) - 1
            t0, t1 = self.mats[i0], self.mats[i1]
            ln0, ln1 = math.log(self.dfs[i0]), math.log(self.dfs[i1])
            slope = (ln1 - ln0) / (t1 - t0)
            ln_t = ln1 + slope * (t - t1)
            return math.exp(ln_t)

        i = bisect.bisect_left(self.mats, t)
        t0, t1 = self.mats[i - 1], self.mats[i]
        ln0, ln1 = math.log(self.dfs[i - 1]), math.log(self.dfs[i])
        w = (t - t0) / (t1 - t0)
        return math.exp(ln0 * (1 - w) + ln1 * w)

    def get_zero(self, t: float) -> float:
        df = self.get_df(t)
        return zero_from_df(df, t)

    def get_fwd(self, t1: float, t2: float) -> float:
        """Simple annual forward rate between t1 and t2 (t2>t1)."""
        if t2 <= t1:
            return 0.0
        df1, df2 = self.get_df(t1), self.get_df(t2)
        return (df1 / df2) ** (1.0 / (t2 - t1)) - 1.0

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"T": self.mats, "Zero(%)": [z * 100 for z in self.zeros], "DF": self.dfs}
        )

    def bumped(self, bp: float) -> "YieldCurve":
        """Parallel shift of zeros by bp (1bp = 0.0001)."""
        bump = bp / 10000.0
        new_zeros = [z + bump for z in self.zeros]
        new_dfs = [df_from_zero(z, t) for z, t in zip(new_zeros, self.mats)]
        return YieldCurve(self.mats[:], new_zeros, new_dfs)


# ---------- Bootstrapping ----------

def bootstrap_curve(
    deposits: List[Tuple[float, float]],
    bonds: List[Dict],
) -> YieldCurve:
    """
    Build curve in ascending maturities:

    deposits: list of (T, r) — zeros directly observed (e.g., 0.25y, 1.5%)
    bonds: list of dicts:
        {"T": 2.0, "price": 99.2, "coupon": 0.02, "face": 100.0, "freq": 2}

    Returns a YieldCurve with mats, zeros, dfs.
    """
    # Collect points
    pts: Dict[float, float] = {}  # maturity -> DF

    # Step 1: deposits give direct zeros
    for T, r in deposits:
        if T <= 0:
            continue
        DF = df_from_zero(r, T)
        pts[T] = DF

    # Step 2: sort by maturity, then add bonds in ascending T
    bonds_sorted = sorted(bonds, key=lambda x: x["T"])
    mats_sorted = sorted(set(list(pts.keys()) + [b["T"] for b in bonds_sorted]))

    # Helper to get DF (interpolated if not on the grid yet)
    temp_curve = None

    for T in mats_sorted:
        if T in pts:
            continue  # already set by deposit

        # Find bond with this final maturity
        match = next((b for b in bonds_sorted if abs(b["T"] - T) < 1e-12), None)
        if match is None:
            # If no exact bond, skip — we'll interpolate later (not ideal for bootstrapping).
            continue

        price = float(match["price"])
        face = float(match.get("face", 100.0))
        coupon = float(match["coupon"])
        freq = int(match.get("freq", 2))
        n = int(round(T * freq))
        c = face * coupon / freq

        # Ensure we can value earlier cashflows: build a provisional curve with what we have so far
        known_mats = sorted(pts.keys())
        if not known_mats:
            raise ValueError("No initial zero points (deposits) to start bootstrapping.")

        # Build a temporary curve for interim DFs
        zeros_tmp = [zero_from_df(pts[t], t) for t in known_mats]
        temp_curve = YieldCurve(known_mats, zeros_tmp, [pts[t] for t in known_mats])

        # Present value of earlier coupons using known DFs
        pv_earlier = 0.0
        for k in range(1, n):
            t_k = k / freq
            df_k = temp_curve.get_df(t_k)  # interpolate ln DF if t_k not known
            pv_earlier += c * df_k

        # Solve DF at final maturity from bond price equation:
        # price = pv_earlier + (c + face) * DF_T  =>  DF_T = (price - pv_earlier)/(c+face)
        denom = (c + face)
        if denom <= 0:
            raise ValueError("Invalid bond cashflows.")
        DF_T = (price - pv_earlier) / denom
        if DF_T <= 0 or DF_T > 1.5:
            raise ValueError(f"Unreasonable DF at T={T:.4f}: {DF_T}")
        pts[T] = DF_T

    # Final curve arrays
    mats = sorted(pts.keys())
    dfs = [pts[t] for t in mats]
    zeros = [zero_from_df(df, t) for df, t in zip(dfs, mats)]
    return YieldCurve(mats, zeros, dfs)
