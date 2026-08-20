from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "R012_R014_GATE_ABLATIONS_METRICS.csv"

SYSTEMS = [
    ("fixed_lexical", "Lexical", "#666666"),
    ("fixed_average", "Fixed avg.", "#999999"),
    ("validation_tuned_constant_gate", "Const. gate", "#CC79A7"),
    ("pooled_mlp_same_tokens", "Pooled MLP", "#E69F00"),
    ("query_only_gate", "Query-only", "#009E73"),
    ("cross_attention_with_gate_loss", "Trust gate", COLORS["gate"]),
]
SPLITS = [
    ("test", "Full test"),
    ("independent_lex_benign_multi_ip", "Lex-benign + multi-IP"),
]


def mean_metric(df: pd.DataFrame, split: str, ablation: str, metric: str) -> float:
    rows = df[df["split"].eq(split) & df["ablation"].eq(ablation)]
    if rows.empty:
        raise ValueError(f"Missing rows for split={split!r}, ablation={ablation!r}")
    return float(rows[metric].mean())


def main() -> None:
    df = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)

    for ax, (split, title) in zip(axes, SPLITS):
        values = [mean_metric(df, split, ablation, "AUPRC") for ablation, _, _ in SYSTEMS]
        x = np.arange(len(SYSTEMS))
        ax.bar(x, values, color=[color for _, _, color in SYSTEMS])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label, _ in SYSTEMS], rotation=0, ha="center", fontsize=8.5)
        ax.set_ylabel("AUPRC")
        ax.set_title(title, fontsize=10)
        lower = 0.55 if split == "test" else 0.25
        ax.set_ylim(lower, max(values) + 0.04)
        for xpos, value in zip(x, values):
            ax.text(xpos, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(w_pad=1.2)

    save_fig(fig, "fig2_gate_controls")


if __name__ == "__main__":
    main()
