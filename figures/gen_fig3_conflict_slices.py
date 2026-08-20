from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "R025_INDEPENDENT_CONFLICT_AGGREGATE.csv"

SUBSETS = [
    "lex_benign_shape",
    "lex_benign_shape_multi_ip",
    "multi_ip_dns",
    "ttl_low",
]
SYSTEMS = [
    ("cross_attentive_gate", "Trust gate", COLORS["gate"]),
    ("residual_correction_gate", "Residual", COLORS["residual"]),
    ("standalone_conflict_classifier", "Direct head", COLORS["direct"]),
]
LABELS = ["Lex-benign\nshape", "Lex-benign\n+ multi-IP", "Multi-IP\nDNS", "Low TTL"]


def main() -> None:
    df = pd.read_csv(DATA)
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    x = np.arange(len(SUBSETS))
    width = 0.24
    for idx, (system, label, color) in enumerate(SYSTEMS):
        values = [
            float(df[(df["subset"].eq(subset)) & (df["system"].eq(system))]["AUPRC_mean"].iloc[0])
            for subset in SUBSETS
        ]
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=color)
    ax.set_ylabel("AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0.5, 0.95)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_fig(fig, "fig3_conflict_slices")


if __name__ == "__main__":
    main()
