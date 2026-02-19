"""
Bond pricing utilities.

This module provides functions to calculate the theoretical price of a bond
given its characteristics: maturity, coupon rate, face value, and compounding convention.
"""

import math
from typing import Literal, Optional


def bond_price(
    maturity_years: float,
    coupon_rate: float,
    yield_rate: float,
    face_value: float = 100.0,
    compounding: Literal["continuous", "discrete"] = "continuous",
    compounding_periods_per_year: Optional[int] = None,
    coupon_frequency: int = 2,
) -> float:
    """
    Calculate the theoretical price of a bond.

    Parameters
    ----------
    maturity_years : float
        Time to maturity in years (e.g., 5.0 for a 5-year bond).
    coupon_rate : float
        Annual coupon rate in decimal form (e.g., 0.05 for 5%).
    yield_rate : float
        Annual yield to maturity (YTM) in decimal form (e.g., 0.04 for 4%).
    face_value : float, optional
        Face value (par value) of the bond, default 100.0.
    compounding : {"continuous", "discrete"}, optional
        Compounding convention, default "continuous".
    compounding_periods_per_year : int, optional
        Number of compounding periods per year for discrete compounding
        (e.g., 2 for semi-annual). Required if compounding="discrete".
    coupon_frequency : int, optional
        Number of coupon payments per year, default 2 (semi-annual).

    Returns
    -------
    float
        Theoretical bond price (present value of all cash flows).

    Notes
    -----
    For continuous compounding:
        P = sum(CF_i * exp(-y * t_i)) + FV * exp(-y * T)

    For discrete compounding:
        P = sum(CF_i / (1 + y/n)^(n*t_i)) + FV / (1 + y/n)^(n*T)

    where:
        - CF_i are coupon payments
        - y is the yield rate
        - t_i are payment times
        - T is maturity
        - n is compounding_periods_per_year
    """
    if compounding == "discrete" and compounding_periods_per_year is None:
        raise ValueError(
            "compounding_periods_per_year must be provided when compounding='discrete'"
        )

    # Calculate coupon payment per period
    coupon_payment = face_value * (coupon_rate / coupon_frequency)

    # Generate coupon payment times
    num_coupons = int(maturity_years * coupon_frequency)
    coupon_times = [(i + 1) / coupon_frequency for i in range(num_coupons)]

    # Adjust last coupon time to exactly match maturity
    if coupon_times and coupon_times[-1] > maturity_years:
        coupon_times[-1] = maturity_years

    # Calculate present value of coupon payments
    pv_coupons = 0.0
    for t in coupon_times:
        if compounding == "continuous":
            discount_factor = math.exp(-yield_rate * t)
        else:  # discrete
            discount_factor = 1.0 / (
                (1.0 + yield_rate / compounding_periods_per_year)
                ** (compounding_periods_per_year * t)
            )
        pv_coupons += coupon_payment * discount_factor

    # Calculate present value of face value (principal repayment at maturity)
    if compounding == "continuous":
        discount_factor_principal = math.exp(-yield_rate * maturity_years)
    else:  # discrete
        discount_factor_principal = 1.0 / (
            (1.0 + yield_rate / compounding_periods_per_year)
            ** (compounding_periods_per_year * maturity_years)
        )

    pv_principal = face_value * discount_factor_principal

    # Total bond price
    bond_price_total = pv_coupons + pv_principal

    return bond_price_total


def test_bond_pricing() -> None:
    """
    Test the bond_price function with a simple example bond.

    Example bond:
        - Maturity: 5 years
        - Coupon rate: 4.5% annual (0.045)
        - Yield to maturity: 4.0% annual (0.04)
        - Face value: $100
        - Semi-annual coupons (2 payments per year)
        - Continuous compounding
    """
    print("=" * 60)
    print("Bond Pricing Test")
    print("=" * 60)

    # Test bond parameters
    maturity = 5.0  # 5 years
    coupon_rate = 0.045  # 4.5% annual
    ytm = 0.04  # 4.0% yield to maturity
    face = 100.0  # $100 face value
    coupon_freq = 2  # Semi-annual coupons

    print(f"\nBond Characteristics:")
    print(f"  Maturity: {maturity} years")
    print(f"  Coupon rate: {coupon_rate:.2%} annual")
    print(f"  Yield to maturity: {ytm:.2%} annual")
    print(f"  Face value: ${face:.2f}")
    print(f"  Coupon frequency: {coupon_freq} payments per year (semi-annual)")

    # Test 1: Continuous compounding
    print(f"\n{'─' * 60}")
    print("Test 1: Continuous Compounding")
    print(f"{'─' * 60}")
    price_continuous = bond_price(
        maturity_years=maturity,
        coupon_rate=coupon_rate,
        yield_rate=ytm,
        face_value=face,
        compounding="continuous",
        coupon_frequency=coupon_freq,
    )
    print(f"Bond price (continuous compounding): ${price_continuous:.4f}")

    # Test 2: Discrete compounding (semi-annual, matching coupon frequency)
    print(f"\n{'─' * 60}")
    print("Test 2: Discrete Compounding (semi-annual)")
    print(f"{'─' * 60}")
    price_discrete = bond_price(
        maturity_years=maturity,
        coupon_rate=coupon_rate,
        yield_rate=ytm,
        face_value=face,
        compounding="discrete",
        compounding_periods_per_year=2,
        coupon_frequency=coupon_freq,
    )
    print(f"Bond price (discrete compounding, n=2): ${price_discrete:.4f}")

    # Test 3: Discrete compounding (annual)
    print(f"\n{'─' * 60}")
    print("Test 3: Discrete Compounding (annual)")
    print(f"{'─' * 60}")
    price_discrete_annual = bond_price(
        maturity_years=maturity,
        coupon_rate=coupon_rate,
        yield_rate=ytm,
        face_value=face,
        compounding="discrete",
        compounding_periods_per_year=1,
        coupon_frequency=1,  # Annual coupons for this test
    )
    print(f"Bond price (discrete compounding, n=1): ${price_discrete_annual:.4f}")

    # Summary
    print(f"\n{'─' * 60}")
    print("Summary")
    print(f"{'─' * 60}")
    print(f"Continuous:  ${price_continuous:.4f}")
    print(f"Discrete (n=2): ${price_discrete:.4f}")
    print(f"Discrete (n=1): ${price_discrete_annual:.4f}")
    print(f"\nNote: Since YTM < coupon rate, bond should trade above par (${face:.2f})")
    print(f"All prices are above par, as expected.")


if __name__ == "__main__":
    test_bond_pricing()

