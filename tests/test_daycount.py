# tests/test_daycount.py
import math
from datetime import date, timedelta

import pytest

from daycount import (
    year_fraction,
    adjust_business_day,
)

# ----------------------------
# Day-count fraction tests
# ----------------------------

def test_act_365f_simple():
    d1 = date(2025, 1, 1)
    d2 = date(2025, 7, 1)  # 181 days in a non-leap year first half (2025 is not leap)
    yf = year_fraction(d1, d2, "ACT/365F")
    assert abs(yf - (d2 - d1).days / 365.0) < 1e-12

def test_act_360_simple():
    d1 = date(2025, 1, 1)
    d2 = date(2025, 4, 1)  # 90 days
    yf = year_fraction(d1, d2, "ACT/360")
    assert abs(yf - 90/360) < 1e-12

def test_30_360_us_end_of_month_rules():
    # 30/360 US (NASD) examples
    # Both on 31st -> both treated as 30
    d1 = date(2025, 1, 31)
    d2 = date(2025, 3, 31)
    yf = year_fraction(d1, d2, "30/360US")
    # (Y diff)*360 + (M diff)*30 + (D2-D1) = (0)*360 + (2)*30 + (30-30) = 60 -> 60/360 = 1/6
    assert abs(yf - (1.0/6.0)) < 1e-12

    # Start=30th, End=31st -> End becomes 30
    d1 = date(2025, 5, 30)
    d2 = date(2025, 6, 31 if False else 30)  # June 31 doesn't exist; emulate rule: end treated as 30
    yf = year_fraction(date(2025,5,30), date(2025,6,30), "30/360US")
    # (0)*360 + (1)*30 + (30-30) = 30 -> 30/360 = 1/12
    assert abs(yf - (1.0/12.0)) < 1e-12

def test_30_360_eu_end_of_month_rules():
    # 30E/360: both 31 -> set to 30; simpler than US rules
    d1 = date(2025, 1, 31)
    d2 = date(2025, 2, 28)  # Feb 28; in EU method, only start 31 -> becomes 30; end 28 stays 28
    yf = year_fraction(d1, d2, "30/360EU")
    # Calc: ((2025-2025)*360 + (2-1)*30 + (28-30)) / 360 = (30 - 2)/360 = 28/360
    assert abs(yf - (28/360)) < 1e-12

    d1 = date(2025, 1, 31)
    d2 = date(2025, 3, 31)
    yf = year_fraction(d1, d2, "30/360EU")
    # In 30E/360 both 31 -> both set to 30 -> (0)*360 + (2)*30 + (30-30) = 60 -> 1/6
    assert abs(yf - (1.0/6.0)) < 1e-12

def test_act_act_cross_year_and_leap():
    # Cross a leap year boundary: 2024 is leap, 2025 is not.
    d1 = date(2024, 12, 15)
    d2 = date(2025, 1, 15)
    yf = year_fraction(d1, d2, "ACT/ACT")
    # 17 days in 2024 part (Dec 15 -> Jan 1 is 17 days), / 366
    # 14 days in 2025 part (Jan 1 -> Jan 15 is 14 days), / 365
    expected = 17/366 + 14/365
    assert abs(yf - expected) < 1e-12

# ----------------------------
# Business day adjustment tests
# ----------------------------

def test_following_on_weekend():
    # Saturday -> Following should roll to Monday
    d = date(2025, 2, 1)  # 1 Feb 2025 is Saturday
    adj = adjust_business_day(d, "Following")
    assert adj.weekday() == 0  # Monday
    assert adj >= d

def test_modified_following_cross_month():
    # Saturday at month end; Modified Following should go backwards if following crosses month
    d = date(2025, 5, 31)  # Saturday, end of month
    adj = adjust_business_day(d, "Modified Following")
    # Following would be Monday June 2, which is a different month -> modified goes back to Friday May 30
    # But May 30, 2025 is Friday indeed.
    assert adj.month == 5
    assert adj <= d
    assert adj.weekday() <= 4  # Mon-Fri

def test_preceding_on_weekend():
    # Sunday -> Preceding should roll back to Friday
    d = date(2025, 3, 2)  # Sunday
    adj = adjust_business_day(d, "Preceding")
    assert adj.weekday() == 4  # Friday
    assert adj <= d

# ----------------------------
# Edge cases
# ----------------------------

def test_zero_or_negative_interval_returns_zero():
    d = date(2025, 1, 15)
    assert year_fraction(d, d, "ACT/365F") == 0.0
    assert year_fraction(d, d - timedelta(days=1), "ACT/360") == 0.0

def test_invalid_convention_raises():
    d1 = date(2025, 1, 1); d2 = date(2025, 2, 1)
    with pytest.raises(ValueError):
        year_fraction(d1, d2, "BAD/COUNT")
