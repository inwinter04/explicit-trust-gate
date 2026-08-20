#!/usr/bin/env python
"""Fig. 6 (R027): g-value trust routing.

Panel (a): routing accuracy by policy on expert-disagreement rows
(|p_lex - p_dns| >= 0.10), mean +/- std over 3 seeds.
Panel (b): mean gate lexical-trust g in the two correctness regimes
(lex-correct/beh-wrong vs beh-correct/lex-wrong).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]
DELTA = 0.10


def main() -> None:
    metrics = pd.read_csv(ROOT / "refine-logs" / "R027_GATE_TRUST_ROUTING_METRICS.csv")
    m = metrics[(metrics["delta"] == DELTA) & metrics["gate_routing_acc"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # --- panel (a): routing accuracy by policy ---
    policies = {
        "g-route": m["gate_routing_acc"],
        "fixed-lex": m["fixed_lex_acc"],
        "fixed-beh": m["fixed_beh_acc"],
    }
    names = list(policies)
    means = [float(policies[n].mean()) for n in names]
    stds = [float(policies[n].std(ddof=0)) for n in names]
    ax = axes[0]
    colors = [COLORS["gate"], COLORS["char"], COLORS["fn"]]
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.4, width=0.6)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Routing accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(a)", fontsize=9)

    # --- panel (b): mean g by correctness regime ---
    regimes = {
        "lex-correct only": m["g_mean_lex_correct_only"],
        "beh-correct only": m["g_mean_beh_correct_only"],
    }
    names_b = list(regimes)
    means_b = [float(regimes[n].mean()) for n in names_b]
    stds_b = [float(regimes[n].std(ddof=0)) for n in names_b]
    ax = axes[1]
    bars_b = ax.bar(
        names_b,
        means_b,
        yerr=stds_b,
        capsize=4,
        color=[COLORS["gate"], COLORS["residual"]],
        edgecolor="black",
        linewidth=0.4,
        width=0.5,
    )
    for bar, val in zip(bars_b, means_b):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean lexical trust g")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("(b)", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "fig6_gate_trust_routing")


if __name__ == "__main__":
    main()
