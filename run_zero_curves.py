"""
Compute and compare US Treasury zero-coupon curves.

This script is pure computation -- no Refinitiv session required.
It reads the prepared datasets produced by `import_refinitiv_data.py`:

  data/refinitiv/us_treasury_input_YYYYMMDD.csv   (coupon bonds)
  data/refinitiv/us_strips_input_YYYYMMDD.csv     (STRIPS)

Then it:
  1. Bootstraps a zero curve from coupon bonds
  2. Computes zero rates directly from STRIPS prices
  3. Compares both curves with plots and statistics

Outputs (unique per run never overwritten):
  data/bootstrapped/us_zero_curve_from_coupons_YYYYMMDD_YYYYMMDD_HHMMSS.csv
  data/bootstrapped/us_zero_curve_from_strips_YYYYMMDD_YYYYMMDD_HHMMSS.csv
  plot/zero_curves/zero_curve_comparison_YYYYMMDD_YYYYMMDD_HHMMSS.png
  plot/zero_curves/zero_curve_comparison_table_YYYYMMDD_YYYYMMDD_HHMMSS.csv

  Format: <data_date>_<run_timestamp>

Financial theory from my course
----------------
For a coupon bond with cash flows CF_k at times T_k:

    Price = Σ_k CF_k · P(0, T_k)

where P(0,T) = exp(-r(T)·T) is the discount factor, r(T) the continuous
zero rate. Bootstrapping solves for each P(0,T) sequentially.

For a zero-coupon STRIP with maturity T:

    P(0,T) = Price / 100,  r(T) = -ln(P(0,T)) / T

The bootstrapping proc


# Default: auto-select the most recent import
python run_zero_curves.py

# Specific import run
python run_zero_curves.py --import 20240219_194144



"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REFINITIV_DIR = DATA_DIR / "refinitiv"
BOOTSTRAP_DIR = DATA_DIR / "bootstrapped"
ZERO_CURVE_DIR = Path("plot") / "zero_curves"


# ============================================================================
# 1. Bootstrap zero curve from coupon bonds
# ============================================================================

@dataclass
class CashFlow:
    T: float
    amount: float


def _generate_cash_flows(
    T_years: float, coupon_rate: float, freq: int, face: float = 100.0
) -> List[CashFlow]:
    """
    Simplified level-coupon schedule: equal spacing, last payment at maturity.
    coupon_rate is the annual rate as a decimal (e.g. 0.04 for 4%).
    """
    if T_years <= 0:
        return []
    n = max(int(round(T_years * freq)), 1)
    cpn = face * coupon_rate / freq
    times = [(k + 1) / freq for k in range(n)]
    times[-1] = T_years
    cfs = [CashFlow(T=t, amount=cpn) for t in times]
    cfs[-1] = CashFlow(T=T_years, amount=cpn + face)
    return cfs


def _interpolate_P(T: float, known_T: List[float], known_P: List[float]) -> float:
    """Log-linear interpolation of discount factor (piecewise-constant forward rate)."""
    if not known_T:
        return 1.0
    arr_T = np.array(known_T)
    arr_P = np.array(known_P)
    if T <= arr_T.min():
        return float(arr_P[arr_T.argmin()])
    if T >= arr_T.max():
        return float(arr_P[arr_T.argmax()])
    idx = np.argsort(arr_T)
    Ts, Ps = arr_T[idx], arr_P[idx]
    i = int(np.searchsorted(Ts, T))
    T0, T1 = Ts[i - 1], Ts[i]
    P0, P1 = Ps[i - 1], Ps[i]
    w = (T - T0) / (T1 - T0)
    return float(math.exp((1 - w) * math.log(P0) + w * math.log(P1)))


def bootstrap_from_coupons(prepared_path: Path) -> pd.DataFrame:
    """Bootstrap P(0,T) and r(T) from coupon bonds."""
    df = pd.read_csv(prepared_path, parse_dates=["maturity_date"])
    required = ["ric", "T_years", "coupon_rate", "coupon_frequency", "clean_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prepared coupon dataset missing columns: {missing}")

    df = df.sort_values("T_years").reset_index(drop=True)

    known_T: List[float] = []
    known_P: List[float] = []
    rows: List[Tuple[float, float, float, str]] = []

    for _, row in df.iterrows():
        ric = row["ric"]
        T = float(row["T_years"])
        rate = float(row["coupon_rate"]) / 100.0
        freq = int(row["coupon_frequency"])
        price = float(row["clean_price"])

        cfs = _generate_cash_flows(T, rate, freq)
        if not cfs:
            continue

        if len(cfs) == 1 and math.isclose(cfs[0].T, T, rel_tol=1e-8):
            P_T = price / 100.0
        else:
            pv_earlier = sum(
                cf.amount * (known_P[known_T.index(cf.T)] if cf.T in known_T else _interpolate_P(cf.T, known_T, known_P))
                for cf in cfs[:-1]
            )
            P_T = (price - pv_earlier) / cfs[-1].amount

        P_T = max(min(P_T, 1.0), 1e-8)
        r_T = -math.log(P_T) / T if T > 0 else 0.0
        known_T.append(T)
        known_P.append(P_T)
        rows.append((T, P_T, r_T, ric))

    return pd.DataFrame(rows, columns=["T_years", "discount_factor", "zero_rate_cont", "ric"])


# ============================================================================
# 2. Zero rates from STRIPS
# ============================================================================

def zero_rates_from_strips(strips_path: Path) -> pd.DataFrame:
    """P(0,T) = price/100, r(T) = -ln(P)/T."""
    df = pd.read_csv(strips_path)
    required = ["ric", "T_years", "mid_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Prepared STRIPS dataset missing columns: {missing}")

    df = df[df["T_years"] > 0].copy()
    df["discount_factor"] = df["mid_price"] / 100.0
    df["zero_rate_cont"] = df.apply(
        lambda r: -math.log(r["discount_factor"]) / r["T_years"]
        if r["discount_factor"] > 0 else None,
        axis=1,
    )
    df = df.dropna(subset=["zero_rate_cont"]).sort_values("T_years")

    return pd.DataFrame({
        "T_years": df["T_years"].values,
        "discount_factor": df["discount_factor"].values,
        "zero_rate_cont": df["zero_rate_cont"].values,
        "ric": df["ric"].values,
    })


# ============================================================================
# 3. Compare and plot
# ============================================================================

def _interp_rate(T: float, curve: pd.DataFrame) -> Optional[float]:
    """Linear interpolation of zero rate at T."""
    if curve.empty:
        return None
    Ts = curve["T_years"].values
    Rs = curve["zero_rate_cont"].values
    if T <= Ts[0]:
        return float(Rs[0])
    if T >= Ts[-1]:
        return float(Rs[-1])
    i = int(np.searchsorted(Ts, T))
    T0, T1 = Ts[i - 1], Ts[i]
    r0, r1 = Rs[i - 1], Rs[i]
    if T1 == T0:
        return float(r0)
    return float(r0 + (r1 - r0) * (T - T0) / (T1 - T0))


def compare_and_plot(
    df_coupons: pd.DataFrame, df_strips: pd.DataFrame, date_tag: str
) -> None:
    """Overlay curves, compute statistics, save dated plots."""
    T_min = max(df_coupons["T_years"].min(), df_strips["T_years"].min())
    T_max = min(df_coupons["T_years"].max(), df_strips["T_years"].max())
    print(f"\nCommon maturity range: {T_min:.2f} – {T_max:.2f} years")

    T_grid = np.linspace(T_min, T_max, 200)
    r_coup = np.array([(_interp_rate(t, df_coupons) or np.nan) * 100 for t in T_grid])
    r_strp = np.array([(_interp_rate(t, df_strips) or np.nan) * 100 for t in T_grid])
    diff = r_strp - r_coup
    ok = ~(np.isnan(r_coup) | np.isnan(r_strp))

    print(f"\nCurve difference (STRIPS – Coupons, in percentage points):")
    print(f"  Mean:  {np.nanmean(diff[ok]):+.4f} pp")
    print(f"  Std:   {np.nanstd(diff[ok]):.4f} pp")
    print(f"  Min:   {np.nanmin(diff[ok]):+.4f} pp")
    print(f"  Max:   {np.nanmax(diff[ok]):+.4f} pp")
    print(f"  RMSE:  {np.sqrt(np.nanmean(diff[ok]**2)):.4f} pp")

    ZERO_CURVE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(
        df_coupons["T_years"], df_coupons["zero_rate_cont"] * 100,
        "o-", label="Bootstrapped from Coupons", markersize=6, linewidth=1.5,
    )
    ax1.plot(
        df_strips["T_years"], df_strips["zero_rate_cont"] * 100,
        ".", label="Direct from STRIPS", markersize=3, alpha=0.5,
    )
    ax1.set_ylabel("Zero Rate (%)")
    ax1.set_title(f"US Treasury Zero-Coupon Curves ({date_tag})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(T_grid[ok], diff[ok], "-", linewidth=1, color="red", alpha=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("Δ (pp)")
    ax2.set_title("STRIPS – Coupons")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = ZERO_CURVE_DIR / f"zero_curve_comparison_{date_tag}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {plot_path}")

    key_T = [0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    table_rows = []
    for t in key_T:
        if T_min <= t <= T_max:
            rc = _interp_rate(t, df_coupons)
            rs = _interp_rate(t, df_strips)
            if rc is not None and rs is not None:
                table_rows.append({
                    "Maturity": t,
                    "Coupon (%)": f"{rc * 100:.4f}",
                    "STRIPS (%)": f"{rs * 100:.4f}",
                    "Diff (bp)": f"{(rs - rc) * 10000:.1f}",
                })
    if table_rows:
        table = pd.DataFrame(table_rows)
        table_path = ZERO_CURVE_DIR / f"zero_curve_comparison_table_{date_tag}.csv"
        table.to_csv(table_path, index=False)
        print(f"Table saved -> {table_path}")
        print(f"\n{table.to_string(index=False)}")


# ============================================================================
# Main
# ============================================================================

def _list_import_folders(directory: Path = REFINITIV_DIR) -> List[Path]:
    """
    Return all import subfolders in data/refinitiv/, sorted oldest -> newest.
    A valid import folder contains at least one us_treasury_input_*.csv file.
    """
    folders = sorted(
        p for p in directory.iterdir()
        if p.is_dir() and list(p.glob("us_treasury_input_*.csv"))
    )
    return folders


def _resolve_import_folder(import_tag: Optional[str] = None) -> Path:
    """
    If import_tag is given (e.g. '20240219_194144'), look for that specific
    subfolder in data/refinitiv/. Otherwise return the most recent one.
    """
    if import_tag:
        folder = REFINITIV_DIR / import_tag
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Import folder not found: {folder}\n"
                f"Available folders: {[f.name for f in _list_import_folders()]}"
            )
        return folder

    folders = _list_import_folders()
    if not folders:
        raise FileNotFoundError(
            f"No import folders found in {REFINITIV_DIR}. "
            "Run import_refinitiv_data.py first."
        )
    return folders[-1]   # most recent


def _find_in_folder(prefix: str, folder: Path) -> Path:
    """Find the single CSV matching prefix inside the given import folder."""
    candidates = sorted(folder.glob(f"{prefix}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.csv found in {folder}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Compute US Treasury zero curves from a Refinitiv import."
    )
    parser.add_argument(
        "--import",
        dest="import_tag",
        metavar="YYYYMMDD_HHMMSS",
        default=None,
        help=(
            "Import folder to use (e.g. 20240219_194144). "
            "Defaults to the most recent import in data/refinitiv/."
        ),
    )
    args = parser.parse_args()

    # --- Resolve import folder ---
    import_folder = _resolve_import_folder(args.import_tag)
    print(f"Import folder: {import_folder}")

    coupon_path = _find_in_folder("us_treasury_input_", import_folder)
    strips_path = _find_in_folder("us_strips_input_", import_folder)
    print(f"Coupon bonds:  {coupon_path}")
    print(f"STRIPS:        {strips_path}")

    data_tag   = import_folder.name          # e.g. 20240219_194144
    run_ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag    = f"{data_tag}_{run_ts}"
    print(f"Run tag:       {run_tag}")

    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    ZERO_CURVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Bootstrap from coupons
    print("\n--- Bootstrapping zero curve from coupon bonds ---")
    df_coupons = bootstrap_from_coupons(coupon_path)
    out_coup = BOOTSTRAP_DIR / f"us_zero_curve_from_coupons_{run_tag}.csv"
    df_coupons.to_csv(out_coup, index=False)
    print(f"  {len(df_coupons)} points -> {out_coup}")

    # 2. Zero rates from STRIPS
    print("\n--- Computing zero rates from STRIPS ---")
    df_strips = zero_rates_from_strips(strips_path)
    out_strips = BOOTSTRAP_DIR / f"us_zero_curve_from_strips_{run_tag}.csv"
    df_strips.to_csv(out_strips, index=False)
    print(f"  {len(df_strips)} points -> {out_strips}")

    # 3. Compare and plot
    print("\n--- Comparing curves ---")
    compare_and_plot(df_coupons, df_strips, run_tag)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
