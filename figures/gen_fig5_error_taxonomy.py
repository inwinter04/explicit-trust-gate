from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "R023_QUALITATIVE_DIAGNOSIS_CATEGORIES.csv"


def compact_label(value: str) -> str:
    return (
        value.replace("FN_", "")
        .replace("FP_", "")
        .replace("_", " ")
        .replace("lex benign", "lex-benign")
        .replace("both experts", "both experts")
    )


def main() -> None:
    df = pd.read_csv(DATA).sort_values(["error_type", "rows"], ascending=[True, True])
    colors = [COLORS["fn"] if kind == "FN" else COLORS["fp"] for kind in df["error_type"]]
    labels = [compact_label(value) for value in df["primary_category"]]

    fig_height = max(3.2, 0.3 * len(df))
    fig, ax = plt.subplots(figsize=(6.8, fig_height))
    ax.barh(labels, df["rows"], color=colors)
    ax.set_xlabel("Number of gate errors")
    ax.set_ylabel("Error category")
    for y, value in enumerate(df["rows"]):
        ax.text(float(value) + 2.0, y, str(int(value)), va="center", fontsize=8)
    ax.set_xlim(0, max(df["rows"]) * 1.18)
    handles = [
        plt.Line2D([0], [0], color=COLORS["fn"], lw=6, label="False negative"),
        plt.Line2D([0], [0], color=COLORS["fp"], lw=6, label="False positive"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    save_fig(fig, "fig5_error_taxonomy")


if __name__ == "__main__":
    main()
