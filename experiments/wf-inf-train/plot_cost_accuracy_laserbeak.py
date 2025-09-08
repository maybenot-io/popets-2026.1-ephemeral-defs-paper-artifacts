#!/usr/bin/env python3
"""plot_cost_accuracy_laserbeak.py – Draw accuracy-vs-cost curve for Laserbeak
using the cost and accuracy columns from a CSV file.

Usage
-----
    python plot_cost_accuracy_laserbeak.py  path/to/your/data.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---- publication‑quality defaults (tweak to taste) -------------------------
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
# ---------------------------------------------------------------------------

# Three high-contrast, colour-blind-friendly hues (Okabe-Ito palette)
CB_BLUE   = "#0072B2"   # strong blue
CB_ORANGE = "#E69F00"   # strong orange
CB_GREEN = "#009E73"    # green

def plot_cost_accuracy(csv_path: str) -> None:
    """Parse *csv_path* and plot accuracy vs. cost for Laserbeak."""
    df = pd.read_csv(csv_path)

    # Adjust column names if they contain spaces / special chars.
    #
    # Expected:
    #   - 'Cost (bandwidth+delay)-mean'  → cost
    #   - 'Accuracy-mean'                → accuracy
    #
    df = df.rename(columns={
        "Cost (bandwidth+delay)-mean": "cost",
        "Accuracy-mean": "accuracy"
    })

    df = df.sort_values("cost")

    # ---- PLOT ----------------------------------------------------------------
    plt.figure(figsize=(6.0, 2.5), dpi=300)

    plt.plot(df["cost"], df["accuracy"],
             linestyle="-", marker="o", color=CB_GREEN, label="Laserbeak$^-$")

    plt.xlabel("Overhead (bandwidth + delay)")
    plt.ylabel("Accuracy")
    plt.ylim(top=1.0)   # force the upper y-axis limit to 1.0
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("laserbeak_cost_accuracy.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_cost_accuracy_laserbeak.py  path/to/data.csv")
    plot_cost_accuracy(sys.argv[1])
