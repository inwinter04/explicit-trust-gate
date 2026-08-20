from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


FIG_DIR = Path(__file__).resolve().parent
DPI = 300
FORMAT = "pdf"

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "gate": "#0072B2",
    "residual": "#E69F00",
    "direct": "#009E73",
    "domurls": "#CC79A7",
    "char": "#56B4E9",
    "fn": "#D55E00",
    "fp": "#0072B2",
}


def save_fig(fig: plt.Figure, name: str, fmt: str = FORMAT) -> Path:
    out = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(out)
    print(f"Saved: {out}")
    return out
