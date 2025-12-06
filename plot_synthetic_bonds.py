"""
Visualize synthetic Treasury bonds on a graph.

This script reads the synthetic bonds from CSV and creates informative plots
showing their characteristics: maturity, coupon rate, and price.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("data")
SYNTHETIC_BONDS_PATH = DATA_DIR / "synthetic_treasury_notes.csv"


def load_synthetic_bonds() -> pd.DataFrame:
    """Load synthetic bonds from CSV."""
    if not SYNTHETIC_BONDS_PATH.exists():
        raise FileNotFoundError(
            f"Synthetic bonds file not found at {SYNTHETIC_BONDS_PATH}. "
            "Run `design_synth_bonds.py` first."
        )
    return pd.read_csv(SYNTHETIC_BONDS_PATH)


def plot_synthetic_bonds(bonds_df: pd.DataFrame) -> None:
    """
    Create a comprehensive visualization of synthetic bonds.

    Creates a figure with multiple subplots showing:
    1. Maturity vs Coupon Rate
    2. Maturity vs Price
    3. Coupon Rate vs Price
    4. All bonds on a single maturity-price plot with coupon annotations
    """
    fig = plt.figure(figsize=(14, 10))

    # Subplot 1: Maturity vs Coupon Rate
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(
        bonds_df["maturity_years"],
        bonds_df["coupon_rate"] * 100,
        s=200,
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
        color="steelblue",
    )
    for idx, row in bonds_df.iterrows():
        ax1.annotate(
            f"{row['maturity_years']:.0f}Y",
            (row["maturity_years"], row["coupon_rate"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax1.set_xlabel("Maturity (years)", fontsize=11)
    ax1.set_ylabel("Coupon Rate (%)", fontsize=11)
    ax1.set_title("Maturity vs Coupon Rate", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(bonds_df["maturity_years"].min() - 0.5, bonds_df["maturity_years"].max() + 0.5)

    # Subplot 2: Maturity vs Price
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(
        bonds_df["maturity_years"],
        bonds_df["price"],
        s=200,
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
        color="coral",
    )
    for idx, row in bonds_df.iterrows():
        ax2.annotate(
            f"${row['price']:.2f}",
            (row["maturity_years"], row["price"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax2.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Par ($100)")
    ax2.set_xlabel("Maturity (years)", fontsize=11)
    ax2.set_ylabel("Price ($)", fontsize=11)
    ax2.set_title("Maturity vs Price", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_xlim(bonds_df["maturity_years"].min() - 0.5, bonds_df["maturity_years"].max() + 0.5)

    # Subplot 3: Coupon Rate vs Price
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(
        bonds_df["coupon_rate"] * 100,
        bonds_df["price"],
        s=200,
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
        color="mediumseagreen",
    )
    for idx, row in bonds_df.iterrows():
        ax3.annotate(
            f"{row['maturity_years']:.0f}Y",
            (row["coupon_rate"] * 100, row["price"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax3.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Par ($100)")
    ax3.set_xlabel("Coupon Rate (%)", fontsize=11)
    ax3.set_ylabel("Price ($)", fontsize=11)
    ax3.set_title("Coupon Rate vs Price", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # Subplot 4: Comprehensive view - Maturity vs Price with coupon info
    ax4 = plt.subplot(2, 2, 4)
    scatter = ax4.scatter(
        bonds_df["maturity_years"],
        bonds_df["price"],
        c=bonds_df["coupon_rate"] * 100,
        s=300,
        alpha=0.7,
        edgecolors="black",
        linewidth=1.5,
        cmap="viridis",
    )
    for idx, row in bonds_df.iterrows():
        ax4.annotate(
            f"{row['maturity_years']:.0f}Y\n{row['coupon_rate']*100:.1f}%",
            (row["maturity_years"], row["price"]),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
    ax4.axhline(y=100, color="red", linestyle="--", linewidth=1.5, alpha=0.6, label="Par ($100)")
    ax4.set_xlabel("Maturity (years)", fontsize=11)
    ax4.set_ylabel("Price ($)", fontsize=11)
    ax4.set_title("Synthetic Bonds Overview\n(Color = Coupon Rate)", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label("Coupon Rate (%)", fontsize=10)
    ax4.set_xlim(bonds_df["maturity_years"].min() - 0.5, bonds_df["maturity_years"].max() + 0.5)

    plt.suptitle("Synthetic Treasury Bonds Visualization", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    # Save plot
    plot_dir = Path("plot")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "synthetic_bonds_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    # Also print summary table
    print("\n" + "=" * 60)
    print("Synthetic Bonds Summary")
    print("=" * 60)
    print(bonds_df.to_string(index=False))
    print("=" * 60)


def main() -> None:
    """Load synthetic bonds and create visualization."""
    bonds_df = load_synthetic_bonds()
    plot_synthetic_bonds(bonds_df)
    plt.show()


if __name__ == "__main__":
    main()

