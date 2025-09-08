#!/usr/bin/env python3
"""
plot_cost_accuracy.py – Draw accuracy-vs-cost curves for RF/DF (start & end).

Usage
-----
    python plot_cost_accuracy.py  path/to/your/data.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- publication-quality defaults (tweak to taste) --------------------------
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


def plot_cost_accuracy(csv_path: str) -> None:
    """Parse *csv_path* and plot accuracy vs. cost for RF/DF (start & end)."""
    df = pd.read_csv(csv_path)

    # Cost = bandwidth load + delay
    df["cost"] = df["load"] + df["delay"]

    # Split rows into the “start/…” and “end/…” subsets
    df_start = df[df["dataset"].str.startswith("start")].copy().sort_values("cost")
    df_end   = df[df["dataset"].str.startswith("end")].copy().sort_values("cost")

    plt.figure(figsize=(6.0, 2.5), dpi=300)

    # RF curves
    plt.plot(df_start["cost"], df_start["rf"],
             linestyle="dotted", marker="o", color=CB_ORANGE, label="RF – start")
    plt.plot(df_end["cost"], df_end["rf"],
             linestyle="-",  marker="o", color=CB_ORANGE, label="RF – end")

    # DF curves
    plt.plot(df_start["cost"], df_start["df"],
             linestyle="dotted", marker="s", color=CB_BLUE, label="DF – start")
    plt.plot(df_end["cost"], df_end["df"],
             linestyle="-",  marker="s", color=CB_BLUE, label="DF – end")

    plt.xlabel("Overhead (bandwidth+delay)")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend(frameon=False)
    plt.xlim(0.0, 26.0)
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(f"cf-tuning-phases.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_cost_accuracy.py  path/to/data.csv")
    plot_cost_accuracy(sys.argv[1])

