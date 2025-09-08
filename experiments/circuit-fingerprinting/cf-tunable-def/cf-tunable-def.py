#!/usr/bin/env python3
"""
plot_cost_accuracy_ephcf.py – Draw accuracy-vs-cost curves for RF/DF
with 10-fold CV error bars, using a colour-blind-friendly palette.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- publication-quality defaults -------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Three high-contrast, colour-blind-friendly hues (Okabe-Ito palette)
CB_BLUE   = "#0072B2"   # strong blue
CB_ORANGE = "#E69F00"   # strong orange
CB_GREEN = "#009E73"    # green
# ---------------------------------------------------------------------------


def plot_cost_accuracy(csv_path: str, suffix: str = "eph-cf") -> None:
    """Parse *csv_path* and plot accuracy vs. cost for RF/DF (with error bars)
    using only rows whose *dataset* column contains *suffix*.
    """
    df = pd.read_csv(csv_path)

    # Keep the rows of interest
    df = df[df["dataset"].str.contains(suffix)]

    # Composite cost = bandwidth load + delay
    df["cost"] = df["load"] + df["delay"]

    # Aggregate the 10 folds: mean & std-dev per dataset
    grouped = (
        df.groupby("dataset")
          .agg(cost=("cost", "mean"),
               rf_mean=("rf", "mean"),
               rf_std=("rf", "std"),
               df_mean=("df", "mean"),
               df_std=("df", "std"))
          .reset_index()
          .sort_values("cost")
    )

    # ---- PLOT ----------------------------------------------------------------
    plt.figure(figsize=(6.0, 2.5), dpi=300)

    # RF – blue circles
    plt.errorbar(grouped["cost"], grouped["rf_mean"],
                 yerr=grouped["rf_std"],
                 linestyle="-", marker="o", capsize=3,
                 color=CB_ORANGE, label="RF")

    # DF – orange squares
    plt.errorbar(grouped["cost"], grouped["df_mean"],
                 yerr=grouped["df_std"],
                 linestyle="-", marker="s", capsize=3,
                 color=CB_BLUE, label="DF")

    plt.xlabel("Overhead (bandwidth + delay)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend(frameon=False)
    plt.xlim(0.0, 26.0)
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig("cf-tunable-def.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_cost_accuracy_ephcf.py  path/to/data.csv")
    plot_cost_accuracy(sys.argv[1])

