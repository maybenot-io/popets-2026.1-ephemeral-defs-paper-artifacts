#!/usr/bin/env python3
"""
plot.py – Plot accuracy‑vs‑cost curves for RF and DF (per defence)

This version now **exactly matches the colour/line conventions** used in
`cf‑tuning‑phases.py`:

* **RF** curves are **red**
* **DF** curves are **green**
* **eph‑cf** → solid line ("-")
* **eph‑wf** → dashed line ("--")
* Any other defence falls back to dash‑dot ("-.")

All other stylistic choices remain as in the previous update: serif fonts,
6 × 2.5 inch figure, light dotted grid, legend inside axes, tight layout, and
PDF/PNG export.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Publication-quality defaults (copied from cf-tuning-phases.py)
# ---------------------------------------------------------------------------
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
# Line‑style map (per defence)
# ---------------------------------------------------------------------------
DEFENCE_LINESTYLE = {
    "eph-cf": "-",   # solid
    "eph-wf": "dotted",  # dashed
}
DEFAULT_LINESTYLE = "-."  # Fallback for any other defence


def _ls(defence: str) -> str:
    """Matplotlib line‑style for *defence*."""
    return DEFENCE_LINESTYLE.get(defence, DEFAULT_LINESTYLE)

def labelname(defense: str) -> str:
    if defense == "eph-cf":
        return "CF"
    else:
        return "WF"


# ---------------------------------------------------------------------------
# Helper: apply common styling to an *Axes* instance
# ---------------------------------------------------------------------------

def _style_axis(ax) -> None:
    #ax.set_facecolor("#fbfbfb")
    ax.set_xlabel("Overhead (bandwidth + delay)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle=":", linewidth=0.5)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="Path to the input CSV file")
    parser.add_argument("-o", default="eph-wf-vs-cf",
                        help="Basename for the output plot (without extension)")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load & validate data
    # ---------------------------------------------------------------------
    df = pd.read_csv(args.i)
    required = {"dataset", "rf", "df", "load", "delay"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"CSV file must contain: {missing}")

    # Defence name = part before first slash in *dataset*
    df["defense"] = df["dataset"].str.split("/").str[0]

    # Cost (bandwidth + delay)
    df["cost"] = df["load"] + df["delay"]

    # ---------------------------------------------------------------------
    # Plot – figure & axes
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 2.5), dpi=300)

    for defense, group in df.groupby("defense"):
        summary = (
            group.groupby("cost")
                 .agg(rf_mean=("rf", "mean"), rf_std=("rf", "std"),
                      df_mean=("df", "mean"), df_std=("df", "std"))
                 .reset_index()
                 .sort_values("cost")
        )

        # --- RF curve (red) ------------------------------------------------
        ax.errorbar(summary["cost"], summary["rf_mean"], yerr=summary["rf_std"],
                    label=f"RF - {labelname(defense)}", linestyle=_ls(defense), marker="o",
                    color=CB_ORANGE, capsize=3)

        # --- DF curve (green) ---------------------------------------------
        ax.errorbar(summary["cost"], summary["df_mean"], yerr=summary["df_std"],
                    label=f"DF - {labelname(defense)}", linestyle=_ls(defense), marker="s",
                    color=CB_BLUE, capsize=3)

    _style_axis(ax)

    # Legend inside the axes; `frameon=False` matches cf-tuning-phases.py
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(f"{args.o}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.o}.png", bbox_inches="tight")
    print(f"Plot saved as {args.o}.pdf and {args.o}.png")
    plt.show()


if __name__ == "__main__":
    main()

