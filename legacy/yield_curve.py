"""
Plot a smooth, convex FRED zero-coupon curve for short and medium maturities.

Purpose
-------
This module is now **only** responsible for:
  - Downloading US Treasury constant-maturity yields from FRED
  - Extracting the 3M, 6M, 1Y, 2Y, 3Y, 5Y pillars for the latest date
  - Fitting a smooth, globally convex term structure
  - Saving the resulting curves as PNGs in the `plot/` directory

No bootstrapping from coupon bonds is done here.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from import_bond_data import load_or_fetch_treasury_yields


def get_fred_pillars() -> Tuple[np.ndarray, np.ndarray, np.datetime64, Dict[str, float]]:
    """
    Load FRED yields and extract the main short/medium-term pillars.

    Returns
    -------
    times : np.ndarray
        Maturities in years: [0.25, 0.5, 1, 2, 3, 5].
    rates : np.ndarray
        Corresponding zero-like yields in percent (e.g. 3.6 for 3.6%).
    curve_date : np.datetime64
        Latest date for which all series are available.
    todays_yields : dict
        Mapping from FRED series name to its yield (percent) at `curve_date`.
    """
    df, curve_date, yields_today = load_or_fetch_treasury_yields()

    # Yields from FRED are in percent already.
    mapping_series_to_years = {
        "DGS3MO": 0.25,
        "DGS6MO": 0.50,
        "DGS1": 1.00,
        "DGS2": 2.00,
        "DGS3": 3.00,
        "DGS5": 5.00,
    }

    times = []
    rates = []
    todays_yields: Dict[str, float] = {}
    for sid, T in mapping_series_to_years.items():
        y_pct = float(yields_today[sid])
        times.append(T)
        rates.append(y_pct)
        todays_yields[sid] = y_pct

    return np.array(times, dtype=float), np.array(rates, dtype=float), curve_date, todays_yields


def fit_convex_fred_curve(times: np.ndarray, rates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a single, smooth, convex curve through FRED points using a
    quadratic functional form:

        r(T) = a*T^2 + b*T + c

    With a > 0 this function is:
        - globally convex (second derivative 2a > 0 everywhere)
        - flexible enough to capture both the short-end slope and any
          gentle re-steepening toward 5Y, while staying one single curve.

    This satisfies the requirement of **one convex curve from start to end**.

    Parameters
    ----------
    times : np.ndarray
        Maturities in years of the FRED pillars.
    rates : np.ndarray
        Yields in percent at those maturities.

    Returns
    -------
    grid_T : np.ndarray
        Fine maturity grid from min(times) to max(times).
    grid_rates : np.ndarray
        Fitted convex curve evaluated on `grid_T` (percent).
    """

    def quad_model(T, a, b, c):
        return a * T ** 2 + b * T + c

    # Initial guesses based on simple regression intuition
    a0 = 0.01
    b0 = (rates[-1] - rates[0]) / (times[-1] - times[0])
    c0 = rates[0]

    # Enforce a > 0 for convexity; keep b and c reasonably bounded
    popt, _ = curve_fit(
        quad_model,
        times,
        rates,
        p0=[a0, b0, c0],
        bounds=([1e-6, -1e3, -1e3], [1e3, 1e3, 1e3]),
        maxfev=10_000,
    )
    a_fit, b_fit, c_fit = popt

    grid_T = np.linspace(times.min(), times.max(), 400)
    grid_rates = quad_model(grid_T, a_fit, b_fit, c_fit)
    return grid_T, grid_rates


def fit_interpolating_curve(times: np.ndarray, rates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method 2: exact interpolating curve through all FRED points.

    Uses a natural cubic spline:
        - passes exactly through each (T_i, rate_i)
        - smooth (C^2) between points
        - not forced to be convex, so it can wiggle to hit all points.

    Returns
    -------
    grid_T : np.ndarray
        Fine maturity grid from min(times) to max(times).
    grid_rates : np.ndarray
        Interpolated curve evaluated on `grid_T` (percent).
    """
    cs = CubicSpline(times, rates, bc_type="natural")
    grid_T = np.linspace(times.min(), times.max(), 400)
    grid_rates = cs(grid_T)
    return grid_T, grid_rates


def main() -> None:
    """
    Entry point:
      - Load FRED data
      - Extract 3M, 6M, 1Y, 2Y, 3Y, 5Y yields
      - Method 1: fit a globally convex curve (quadratic) and plot it
      - Method 2: fit an exact interpolating spline and plot it
      - Save two plots in `plot/` as method1 / method2
    """
    times, rates, curve_date, todays_yields = get_fred_pillars()

    # ---------- Method 1: convex quadratic fit ----------
    grid_T1, grid_rates1 = fit_convex_fred_curve(times, rates)

    plt.figure(figsize=(10, 6))
    plt.plot(
        grid_T1,
        grid_rates1,
        "-",
        color="red",
        linewidth=2.5,
        label="Method 1: convex quadratic fit",
    )
    plt.fill_between(grid_T1, grid_rates1, color="red", alpha=0.08)
    plt.scatter(
        times,
        rates,
        color="red",
        edgecolor="black",
        linewidth=0.5,
        s=40,
        zorder=5,
        alpha=0.95,
    )
    plt.xlabel("Maturity (years)", fontsize=12)
    plt.ylabel("Zero rate (%)", fontsize=12)
    plt.title(
        f"FRED Zero-Coupon Curve – Method 1 (Convex Fit)\n(as of {curve_date.strftime('%Y-%m-%d')})",
        fontsize=14,
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.4)
    plt.legend(fontsize=11, loc="upper right", frameon=False)
    x_margin = 0.1
    plt.xlim(times.min() - x_margin, times.max() + x_margin)
    y_min = min(rates.min(), grid_rates1.min()) - 0.05
    y_max = max(rates.max(), grid_rates1.max()) + 0.05
    plt.ylim(y_min, y_max)
    plt.tight_layout()

    plot_dir = Path("plot")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path1 = plot_dir / f"yield_curves_method1_{curve_date.strftime('%Y%m%d')}.png"
    plt.savefig(plot_path1, dpi=300)

    # ---------- Method 2: exact spline interpolation ----------
    grid_T2, grid_rates2 = fit_interpolating_curve(times, rates)

    plt.figure(figsize=(10, 6))
    plt.plot(
        grid_T2,
        grid_rates2,
        "-",
        color="blue",
        linewidth=2.5,
        label="Method 2: cubic spline interpolation",
    )
    plt.fill_between(grid_T2, grid_rates2, color="blue", alpha=0.08)
    plt.scatter(
        times,
        rates,
        color="blue",
        edgecolor="black",
        linewidth=0.5,
        s=40,
        zorder=5,
        alpha=0.95,
    )
    plt.xlabel("Maturity (years)", fontsize=12)
    plt.ylabel("Zero rate (%)", fontsize=12)
    plt.title(
        f"FRED Zero-Coupon Curve – Method 2 (Spline)\n(as of {curve_date.strftime('%Y-%m-%d')})",
        fontsize=14,
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.4)
    plt.legend(fontsize=11, loc="upper right", frameon=False)
    plt.xlim(times.min() - x_margin, times.max() + x_margin)
    y_min2 = min(rates.min(), grid_rates2.min()) - 0.05
    y_max2 = max(rates.max(), grid_rates2.max()) + 0.05
    plt.ylim(y_min2, y_max2)
    plt.tight_layout()

    plot_path2 = plot_dir / f"yield_curves_method2_{curve_date.strftime('%Y%m%d')}.png"
    plt.savefig(plot_path2, dpi=300)

    # Console output
    print("FRED pillars used (latest date):")
    for sid, y in todays_yields.items():
        print(f"  {sid:<6} -> {y:.3f} %")
    print(f"\nMethod 1 plot saved to: {plot_path1}")
    print(f"Method 2 plot saved to: {plot_path2}")


if __name__ == "__main__":
    main()


