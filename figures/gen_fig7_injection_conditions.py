#!/usr/bin/env python
"""Fig. 7 (R028b): verified-timing controlled injection.

Grouped bar: AUPRC by behavior-view condition (original degenerate, injected
TTL, injected IP diversity) and system (lexical-only, DNS-only, gate), mean
over 3 seeds with +/- std where available.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metrics = pd.read_csv(ROOT / "refine-logs" / "R028B_PROSPECTIVE_INJECTION_METRICS.csv")
    metrics = metrics[metrics["system"].isin(["lexical_only", "dns_behavior_only", "cross_attentive_gate"])]
    cond_order = ["original_degenerate", "injected_ttl_informative", "injected_ip_diversity_informative"]
    cond_labels = ["original\ndegenerate", "injected TTL\n(uninformative)", "injected IP\ndiversity"]
    system_labels = ["lexical-only", "DNS-only", "gate"]
    system_keys = ["lexical_only", "dns_behavior_only", "cross_attentive_gate"]

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    x = range(len(cond_order))
    width = 0.26
    for i, (skey, slab) in enumerate(zip(system_keys, system_labels)):
        vals = []
        for cond in cond_order:
            part = metrics[(metrics["condition"] == cond) & (metrics["system"] == skey)]["AUPRC"]
            vals.append(float(part.mean()))
        color = COLORS["char"] if skey == "lexical_only" else (COLORS["residual"] if skey == "dns_behavior_only" else COLORS["gate"])
        bars = ax.bar([xi + (i - 1) * width for xi in x], vals, width=width, label=slab, color=color, edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save_fig(fig, "fig7_injection_conditions")


if __name__ == "__main__":
    main()
