def generate_coupon_schedule(
    issue_date: date,
    maturity_date: date,
    freq: int,
    biz_conv: BizConv = "Following",
) -> List[date]:
    """
    Return adjusted coupon dates strictly after issue_date up to and including maturity_date.

    Supports any frequency (1, 2, 4, 12).
    Example:
        freq=12 -> monthly coupons
        freq=4  -> quarterly
        freq=2  -> semiannual
        freq=1  -> annual
    """
    if freq <= 0:
        raise ValueError("freq must be positive")

    # ✅ Flexible step size in months
    months_per_period = 12 / freq

    dates: List[date] = []
    d = issue_date

    while True:
        # Add months (rounded for fractional step sizes)
        d = add_months(d, int(round(months_per_period)))
        if d >= maturity_date:
            dates.append(adjust_business_day(maturity_date, biz_conv))
            break
        dates.append(adjust_business_day(d, biz_conv))

    return dates
