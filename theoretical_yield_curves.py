"""
Plot the four classic yield curve shapes using the Nelson-Siegel model.

Saves the combined plot and individual plots to plot/theoretical/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Nelson-Siegel Model
# ---------------------------------------------------------------------------

def nelson_siegel(tau, beta0, beta1, beta2, lambda_):
    """
    Calculates the yield for a given maturity using the Nelson-Siegel model.

    Args:
        tau (array-like): Maturities in years.
        beta0 (float): Long-term level component.
        beta1 (float): Short-term component (slope).
        beta2 (float): Medium-term component (curvature).
        lambda_ (float): Decay factor determining the hump's location.

    Returns:
        numpy.ndarray: Yields corresponding to the input maturities.
    """
    tau = np.maximum(tau, 1e-6)  # avoid division by zero

    term1 = (1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)
    term2 = term1 - np.exp(-lambda_ * tau)

    return beta0 + beta1 * term1 + beta2 * term2


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------

def setup_plot_style():
    """Apply a clean, modern aesthetic to matplotlib."""
    plt.rcParams.update({
        "axes.facecolor": "#f8f9fa",
        "figure.facecolor": "#ffffff",
        "axes.grid": True,
        "grid.color": "#e9ecef",
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "axes.edgecolor": "#ced4da",
        "axes.linewidth": 1.2,
        "axes.labelcolor": "#495057",
        "xtick.color": "#495057",
        "ytick.color": "#495057",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
    })

def plot_single_curve(maturities, yields, title, filename, color):
    plt.figure(figsize=(10, 6))
    
    # Fill under curve for aesthetics
    plt.fill_between(maturities, yields, 0, color=color, alpha=0.1)
    
    plt.plot(maturities, yields, color=color, linewidth=3)
    
    plt.title(title, pad=15)
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Yield (%)")
    plt.xlim(0, 30)
    plt.ylim(0, 8)
    
    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    out_dir = Path("plot") / "theoretical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_plot_style()
    
    # 1. Maturities grid
    maturities = np.linspace(0.01, 30, 200)

    # 2. Parameters for each curve shape
    curves = {
        "Normal (Upward Sloping)": {
            "params": {"beta0": 5.0, "beta1": -3.0, "beta2": 0.0, "lambda_": 0.5},
            "color": "#1f77b4",  # Blue
            "filename": "yield_curve_normal.png"
        },
        "Inverted (Downward Sloping)": {
            "params": {"beta0": 3.0, "beta1":  3.0, "beta2": 0.0, "lambda_": 0.5},
            "color": "#d62728",  # Red
            "filename": "yield_curve_inverted.png"
        },
        "Flat": {
            "params": {"beta0": 4.0, "beta1":  0.0, "beta2": 0.0, "lambda_": 0.5},
            "color": "#7f7f7f",  # Gray
            "filename": "yield_curve_flat.png"
        },
        "Humped": {
            "params": {"beta0": 4.5, "beta1": -1.5, "beta2": 3.0, "lambda_": 0.8},
            "color": "#ff7f0e",  # Orange
            "filename": "yield_curve_humped.png"
        }
    }

    # 3. Generate yields and plot individuals
    for name, config in curves.items():
        config["yields"] = nelson_siegel(maturities, **config["params"])
        plot_single_curve(
            maturities, 
            config["yields"], 
            f"Theoretical Yield Curve: {name}", 
            config["filename"], 
            config["color"]
        )

    # 4. Plot Combined
    plt.figure(figsize=(12, 8))

    for name, config in curves.items():
        style = "--" if name == "Flat" else "-"
        plt.plot(maturities, config["yields"], label=name, color=config["color"], 
                 linewidth=3, linestyle=style)

    plt.title("Four Classic Types of Yield Curves", pad=20)
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Yield (%)")
    plt.legend(frameon=True, facecolor="white", edgecolor="#ced4da", 
               fontsize=12, loc="lower right")
    
    plt.xlim(0, 30)
    plt.ylim(0, 8)
    
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    
    plt.tight_layout()

    # 5. Save combined
    out_dir = Path("plot") / "theoretical"
    out_path = out_dir / "yield_curves_combined.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Combined plot saved -> {out_path}")


if __name__ == "__main__":
    main()
