"""
yield_curve.py
Bootstrapped zero curve with date-aware inputs (optional) and numeric fallbacks.

Inputs:
- deposits:
    * numeric: list[(T_years, rate)]
    * date-aware: list[{"date": <maturity_date>, "rate": <annual rate>}]
- bonds:
    * numeric: list[{"T": years, "price": ..., "coupon": ..., "face": 100.0, "freq": 2}]
    * date-aware: list[{"maturity_date": <date>, "price": ..., "coupon": ..., "face": 100.0, "freq": 2, "issue_date": <optional>}]

Conventions:
- Annual compounding spot rates r(T) with DF(T) = 1/(1+r)^T
- If dates are provided, we compute T via selected day-count (default ACT/365F) from valuation_date.
- Interpolation on ln(DF) across maturities.

Outputs:
- YieldCurve with mats (years), zeros (spot), dfs, and helpers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from math import log, exp
from datetime import date

from daycount import year_fraction, DayCount
from date_utils import generate_coupon_schedule
from bonds import price_bond_dates

# --------- utilities ----------
def df_from_zero(r: float, t: float) -> float:
    return 1.0 / ((1.0 + r) ** t)


def zero_from_df(df: float, t: float) -> float:
    if t <= 0:
        return 0.0
    return (df ** (-1.0 / t)) - 1.0


@dataclass
class YieldCurve:
    mats: List[float]
    zeros: List[float]
    dfs: List[float]

    def get_df(self, t: float) -> float:
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
        # inside
        import bisect
        i = bisect.bisect_left(self.mats, t)
        t0, t1 = self.mats[i - 1], self.mats[i]
        ln0, ln1 = log(self.dfs[i - 1]), log(self.dfs[i])
        w = (t - t0) / (t1 - t0)
        return exp(ln0 * (1 - w) + ln1 * w)

    def get_zero(self, t: float) -> float:
        return zero_from_df(self.get_df(t), t)

    def get_fwd(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            return 0.0
        df1, df2 = self.get_df(t1), self.get_df(t2)
        return (df1 / df2) ** (1.0 / (t2 - t1)) - 1.0

    def as_dataframe(self):
        import pandas as pd
        return pd.DataFrame({"T": self.mats, "Zero(%)": [z * 100 for z in self.zeros], "DF": self.dfs})

    def bumped(self, bp: float) -> "YieldCurve":
        bump = bp / 10000.0
        zs = [z + bump for z in self.zeros]
        return YieldCurve(self.mats[:], zs, [df_from_zero(z, t) for z, t in zip(zs, self.mats)])


# --------- bootstrapping ----------
def _to_T_from_deposit_item(
    item: Union[Tuple[float, float], Dict],
    valuation_date: Optional[date],
    day_count: DayCount,
) -> Tuple[float, float]:
    """Return (T, rate)."""
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


def bootstrap_curve(
    deposits: List[Union[Tuple[float, float], Dict]],
    bonds: List[Dict],
    valuation_date: Optional[date] = None,
    day_count: DayCount = "ACT/365F",
) -> YieldCurve:
    """
    Build curve from deposits and bonds. Supports date-aware inputs when valuation_date is provided.
    """
    pts: Dict[float, float] = {}  # maturity T -> DF

    # 1) Deposits -> direct DFs
    clean_deposits: List[Tuple[float, float]] = []
    for item in deposits:
        T, r = _to_T_from_deposit_item(item, valuation_date, day_count)
        if T > 0:
            clean_deposits.append((T, r))
            pts[T] = df_from_zero(r, T)

    # 2) Bonds: solve final DF using prior curve for earlier coupons
    bonds_sorted = sorted(bonds, key=lambda x: x.get("T", 0.0) if "T" in x else 1e9)

    # if date-aware bonds exist, sort by their maturity T computed on the fly
    def _bond_T(b) -> float:
        if "T" in b:
            return float(b["T"])
        if "maturity_date" in b:
            if valuation_date is None:
                raise ValueError("valuation_date required for date-based bonds")
            return year_fraction(valuation_date, b["maturity_date"], day_count)
        raise ValueError("Bond missing 'T' or 'maturity_date'")

    bonds_sorted = sorted(bonds, key=_bond_T)

    # helper to get temp curve for interim DFs
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

        if "T" in b:
            T = float(b["T"])
            # PV earlier via temp curve by summing coupon payments before T
            temp = _temp_curve()
            if temp is None:
                raise ValueError("Need at least one deposit point before bootstrapping bonds with numeric T.")

            # earlier coupon times: 1/freq, 2/freq ... (n-1)/freq
            n = int(round(T * freq))
            pv_earlier = 0.0
            c = face * coupon / freq
            for k in range(1, n):
                t_k = k / freq
                pv_earlier += c * temp.get_df(t_k)
            DF_T = (price - pv_earlier) / (c + face)
            if DF_T <= 0 or DF_T > 1.5:
                raise ValueError(f"Unreasonable DF from bond T={T}")
            pts[T] = DF_T

        elif "maturity_date" in b:
            if valuation_date is None:
                raise ValueError("valuation_date required for date-based bonds")
            mat = b["maturity_date"]
            issue_date = b.get("issue_date", None)
            dayc = b.get("day_count", day_count)

            # Use bond pricer with dates to compute earlier PV using prior curve via DFs
            # We'll replicate PV earlier by summing coupons before final date with curves' DFs
            temp = _temp_curve()
            if temp is None:
                raise ValueError("Need at least one deposit point before bootstrapping bonds with dates.")

            # Build schedule strictly after valuation_date
            schedule = generate_coupon_schedule(issue_date or valuation_date, mat, freq)
            # split into earlier (all but final) and final
            if not schedule:
                continue
            final_date = schedule[-1]
            earlier = [d for d in schedule[:-1] if d > valuation_date]

            # PV earlier coupons using temp curve
            from daycount import year_fraction
            c = face * coupon / freq
            pv_earlier = 0.0
            for d in earlier:
                t = year_fraction(valuation_date, d, dayc)
                pv_earlier += c * temp.get_df(t)

            # Solve DF(T_final)
            T_final = year_fraction(valuation_date, final_date, dayc)
            DF_T = (price - pv_earlier) / (c + face)
            if DF_T <= 0 or DF_T > 1.5:
                raise ValueError("Unreasonable DF from dated bond.")
            pts[T_final] = DF_T

        else:
            raise ValueError("Bond must have 'T' or 'maturity_date'.")

    mats = sorted(pts.keys())
    dfs = [pts[t] for t in mats]
    zeros = [zero_from_df(df, t) for df, t in zip(dfs, mats)]
    return YieldCurve(mats, zeros, dfs)
