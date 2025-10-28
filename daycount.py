"""
daycount.py
Year fraction and business-day utilities with common market conventions.

Supported day-counts:
- "ACT/365F" (Actual/365 Fixed)
- "ACT/360"  (Actual/360)
- "30/360US" (30/360 U.S. NASD)
- "30/360EU" (30E/360 European)
- "ACT/ACT"  (ISDA-style split across years)

Business-day adjustment:
- Following
- Modified Following
- Preceding

Calendar: weekends-only (Sat/Sun). No holiday set.
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Literal

DayCount = Literal["ACT/365F", "ACT/360", "30/360US", "30/360EU", "ACT/ACT"]
BizConv = Literal["Following", "Modified Following", "Preceding"]


# ------------------------------
# Helpers
# ------------------------------
def _is_leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def _days_in_year(y: int) -> int:
    return 366 if _is_leap(y) else 365


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # 5=Sat, 6=Sun


def adjust_business_day(d: date, convention: BizConv = "Following") -> date:
    """Adjust a date to a business day using weekend-only calendar."""
    if convention not in ("Following", "Modified Following", "Preceding"):
        raise ValueError("Invalid business day convention")

    if not _is_weekend(d):
        return d

    if convention in ("Following", "Modified Following"):
        orig_month = d.month
        dd = d
        while _is_weekend(dd):
            dd += timedelta(days=1)
        if convention == "Modified Following" and dd.month != orig_month:
            # go backwards instead
            dd = d
            while _is_weekend(dd):
                dd -= timedelta(days=1)
        return dd

    # Preceding
    dd = d
    while _is_weekend(dd):
        dd -= timedelta(days=1)
    return dd


# ------------------------------
# Day-count fractions
# ------------------------------
def _thirty_360_us(start: date, end: date) -> float:
    d1 = min(start.day, 30)
    d2 = end.day
    if start.day == 31 and end.day == 31:
        d1 = 30
        d2 = 30
    elif start.day == 30 and end.day == 31:
        d2 = 30
    elif start.day == 31 and end.day < 31:
        d1 = 30
    return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0


def _thirty_360_eu(start: date, end: date) -> float:
    d1 = 30 if start.day == 31 else start.day
    d2 = 30 if end.day == 31 else end.day
    return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360.0


def _act_365f(start: date, end: date) -> float:
    return (end - start).days / 365.0


def _act_360(start: date, end: date) -> float:
    return (end - start).days / 360.0


def _act_act_isda(start: date, end: date) -> float:
    """ACT/ACT (ISDA-like): split across calendar years using actual days / days-in-year for each part."""
    if end <= start:
        return 0.0
    y1 = start.year
    y2 = end.year
    if y1 == y2:
        return (end - start).days / _days_in_year(y1)

    # split
    end_y1 = date(y1, 12, 31) + timedelta(days=1)  # Jan 1 next year exclusive
    part1 = (end_y1 - start).days / _days_in_year(y1)
    part = part1
    for y in range(y1 + 1, y2):
        part += _days_in_year(y) / _days_in_year(y)  # exactly 1.0
    start_y2 = date(y2, 1, 1)
    part_last = (end - start_y2).days / _days_in_year(y2)
    return part + part_last


def year_fraction(start: date, end: date, convention: DayCount = "ACT/365F") -> float:
    """Compute year fraction between two dates using the given convention."""
    if end <= start:
        return 0.0
    conv = convention.upper()
    if conv == "ACT/365F":
        return _act_365f(start, end)
    if conv == "ACT/360":
        return _act_360(start, end)
    if conv == "30/360US":
        return _thirty_360_us(start, end)
    if conv == "30/360EU":
        return _thirty_360_eu(start, end)
    if conv == "ACT/ACT":
        return _act_act_isda(start, end)
    raise ValueError(f"Unsupported day-count: {convention}")
