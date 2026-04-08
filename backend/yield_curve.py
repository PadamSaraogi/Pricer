"""
yield_curve.py
---------------
Bootstrapped zero curve with date-aware inputs (optional) and numeric fallbacks.

Inputs
------
- deposits:
    * numeric: list[(T_years, rate)]
    * date-aware: list[{"date": <maturity_date>, "rate": <annual rate>}]

- bonds:
    * numeric: list[{"T": years, "price": ..., "coupon": ..., "face": 100.0, "freq": 2}]
    * date-aware: list[{"maturity_date": <date>, "price": ..., "coupon": ..., "face": 100.0,
                        "freq": 2 or 12, "issue_date": <optional>}]

Conventions
-----------
- Annual compounding spot rates r(T) with DF(T) = 1 / (1 + r)^T
- If dates are provided, T is computed via selected day-count (default ACT/365F)
- Interpolation on ln(DF) across maturities
- Supports monthly (freq=12), quarterly (4), semiannual (2), annual (1)

Outputs
-------
- YieldCurve: object with methods
    get_df(t), get_zero(t), get_fwd(t1, t2), as_dataframe(), bumped(bp)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from math import log, exp
from datetime import date

from daycount import year_fraction, DayCount
from date_utils import generate_coupon_schedule
from bonds import price_bond_dates


# -------------------------------------------------------------------------
# --- Utility conversions -------------------------------------------------
# -------------------------------------------------------------------------
def df_from_zero(r: float, t: float) -> float:
    """Discount factor from zero rate."""
    return 1.0 / ((1.0 + r) ** t)


def zero_from_df(df: float, t: float) -> float:
    """Zero rate from discount factor."""
    if t <= 0:
        return 0.0
    return (df ** (-1.0 / t)) - 1.0


# -------------------------------------------------------------------------
# --- YieldCurve data class -----------------------------------------------
# -------------------------------------------------------------------------
@dataclass
class YieldCurve:
    mats: List[float]
    zeros: List[float]
    dfs: List[float]

    # ------------------------
    def get_df(self, t: float) -> float:
        """Return discount factor at time t via log-linear interpolation."""
        if t <= 0:
            return 1.0
        if t <= self.mats[0]:
            ln_df0 = log(self.dfs[0])
            return exp(ln_df0 / self.mats[0] * t)
        if t >= self.mats[-1]:
            i0, i1 = len(self.mats) - 2, len(self.mats) - 1
            t0, t1 = self.mats[i0], self.mats[i1]
            ln0, ln1 = log(self.dfs[i0]), log(self.dfs[i1])
            slope = (ln1 - ln0) / (t1 - t0)
            return exp(ln1 + slope * (t - t1))

        import bisect
        i = bisect.bisect_left(self.mats, t)
        t0, t1 = self.mats[i - 1], self.mats[i]
        ln0, ln1 = log(self.dfs[i - 1]), log(self.dfs[i])
        w = (t - t0) / (t1 - t0)
        return exp(ln0 * (1 - w) + ln1 * w)

    # ------------------------
    def get_zero(self, t: float) -> float:
        """Return zero rate at time t."""
        return zero_from_df(self.get_df(t), t)

    # ------------------------
    def get_fwd(self, t1: float, t2: float) -> float:
        """Return forward rate between t1 and t2."""
        if t2 <= t1:
            return 0.0
        df1, df2 = self.get_df(t1), self.get_df(t2)
        return (df1 / df2) ** (1.0 / (t2 - t1)) - 1.0

    # ------------------------
    def as_dataframe(self):
        """Return curve as pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(
            {"T": self.mats, "Zero(%)": [z * 100 for z in self.zeros], "DF": self.dfs}
        )

    # ------------------------
    def bumped(self, bp: float) -> YieldCurve:
        """Return bumped-up curve by bp (1bp = 0.0001)."""
        bump = bp / 10000.0
        zs = [z + bump for z in self.zeros]
        return YieldCurve(self.mats[:], zs, [df_from_zero(z, t) for z, t in zip(zs, self.mats)])


# -------------------------------------------------------------------------
# --- Helper: Convert deposit items --------------------------------------
# -------------------------------------------------------------------------
def _to_T_from_deposit_item(
    item: Union[Tuple[float, float], Dict],
    valuation_date: Optional[date],
    day_count: DayCount,
) -> Tuple[float, float]:
    """Return (T, rate) for a deposit input."""
    if isinstance(item, tuple) and len(item) == 2:
        return float(item[0]), float(item[1])
    if isinstance(item, dict):
        if "T" in item:
            return float(item["T"]), float(item["rate"])
        if "date" in item:
            if valuation_date is None:
                raise ValueError("valuation_date required when using date-based deposits")
            T = year_fraction(valuation_date, item["date"], day_count)
            return float(T), float(item["rate"])
    raise ValueError("Invalid deposit item")


# -------------------------------------------------------------------------
# --- Bootstrapping -------------------------------------------------------
# -------------------------------------------------------------------------
def bootstrap_curve(
    deposits: List[Union[Tuple[float, float], Dict]],
    bonds: List[Dict],
    valuation_date: Optional[date] = None,
    day_count: DayCount = "ACT/365F",
) -> YieldCurve:
    """
    Build a bootstrapped yield curve from deposits and bonds.
    Supports both numeric and date-based instruments.
    """
    pts: Dict[float, float] = {}  # maturity T -> DF

    # 1️⃣ Deposits → direct discount factors
    clean_deposits: List[Tuple[float, float]] = []
    for item in deposits:
        T, r = _to_T_from_deposit_item(item, valuation_date, day_count)
        if T > 0:
            clean_deposits.append((T, r))
            pts[T] = df_from_zero(r, T)

    # 2️⃣ Bonds → bootstrap by solving DF at each maturity
    if not bonds:
        mats = sorted(pts.keys())
        dfs = [pts[t] for t in mats]
        zeros = [zero_from_df(df, t) for df, t in zip(dfs, mats)]
        return YieldCurve(mats, zeros, dfs)

    # Sort bonds by maturity
    def _bond_T(b) -> float:
        if "T" in b:
            return float(b["T"])
        if "maturity_date" in b:
            if valuation_date is None:
                raise ValueError("valuation_date required for date-based bonds")
            return year_fraction(valuation_date, b["maturity_date"], day_count)
        raise ValueError("Bond missing 'T' or 'maturity_date'")

    bonds_sorted = sorted(bonds, key=_bond_T)

    # helper to get temporary curve for interim PVs
    def _temp_curve() -> Optional[YieldCurve]:
        if not pts:
            return None
        mats = sorted(pts.keys())
        dfs = [pts[t] for t in mats]
        zeros = [zero_from_df(df, t) for df, t in zip(dfs, mats)]
        return YieldCurve(mats, zeros, dfs)

    for b in bonds_sorted:
        price = float(b["price"])
        face = float(b.get("face", 100.0))
        coupon = float(b["coupon"])
        freq = int(b.get("freq", 2))

        # --- Numeric T bonds ---
        if "T" in b:
            T = float(b["T"])
            temp = _temp_curve()
            if temp is None:
                raise ValueError("Need at least one deposit before bonds with numeric T.")

            n = int(round(T * freq))
            pv_earlier = 0.0
            c = face * coupon / freq
            for k in range(1, n):
                t_k = k / freq
                pv_earlier += c * temp.get_df(t_k)
            DF_T = (price - pv_earlier) / (c + face)
            if DF_T <= 0 or DF_T > 1.5:
                raise ValueError(f"Unreasonable DF at T={T}")
            pts[T] = DF_T

        # --- Date-based bonds ---
        elif "maturity_date" in b:
            if valuation_date is None:
                raise ValueError("valuation_date required for date-based bonds")
            maturity_date = b["maturity_date"]
            issue_date = b.get("issue_date", valuation_date)
            dayc = b.get("day_count", day_count)

            temp = _temp_curve()
            if temp is None:
                raise ValueError("Need at least one deposit before date-based bonds.")

            # Build coupon schedule
            schedule = generate_coupon_schedule(issue_date, maturity_date, freq)
            if not schedule:
                continue

            final_date = schedule[-1]
            earlier = [d for d in schedule[:-1] if d > valuation_date]

            c = face * coupon / freq
            pv_earlier = 0.0
            for d in earlier:
                t = year_fraction(valuation_date, d, dayc)
                pv_earlier += c * temp.get_df(t)

            T_final = year_fraction(valuation_date, final_date, dayc)
            DF_T = (price - pv_earlier) / (c + face)
            if DF_T <= 0 or DF_T > 1.5:
                raise ValueError("Unreasonable DF from dated bond.")
            pts[T_final] = DF_T

        else:
            raise ValueError("Bond must have 'T' or 'maturity_date'.")

    # 3️⃣ Final curve assembly
    mats = sorted(pts.keys())
    dfs = [pts[t] for t in mats]
    zeros = [zero_from_df(df, t) for df, t in zip(dfs, mats)]
    return YieldCurve(mats, zeros, dfs)
