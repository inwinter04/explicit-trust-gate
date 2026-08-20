from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from paper_plot_style import save_fig


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "figures" / "specs" / "fig1_architecture.json"


def node_bounds(node: dict) -> tuple[float, float, float, float]:
    width = float(node.get("width", 120))
    height = float(node.get("height", 50))
    x = float(node["x"]) - width / 2
    y = float(node["y"]) - height / 2
    return x, y, width, height


def draw_group(ax: plt.Axes, group: dict, nodes: dict[str, dict]) -> None:
    padding = float(group.get("padding", 20))
    boxes = [node_bounds(nodes[node_id]) for node_id in group["node_ids"]]
    left = min(x for x, _, _, _ in boxes) - padding
    bottom = min(y for _, y, _, _ in boxes) - padding
    right = max(x + width for x, _, width, _ in boxes) + padding
    top = max(y + height for _, y, _, height in boxes) + padding
    patch = FancyBboxPatch(
        (left, bottom),
        right - left,
        top - bottom,
        boxstyle="round,pad=0.012,rounding_size=10",
        facecolor=group.get("fill", "#FAFAFA"),
        edgecolor=group.get("stroke", "#D1D5DB"),
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        left + 10,
        top - 12,
        group["label"],
        ha="left",
        va="top",
        fontsize=9,
        color="#374151",
        zorder=1,
    )


def draw_node(ax: plt.Axes, node: dict) -> None:
    x, y, width, height = node_bounds(node)
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=8",
        facecolor=node.get("fill", "#FFFFFF"),
        edgecolor=node.get("stroke", "#555555"),
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(patch)

    label = node["label"]
    sublabel = node.get("sublabel")
    if sublabel:
        ax.text(
            node["x"],
            node["y"] + 6,
            label,
            ha="center",
            va="center",
            fontsize=node.get("font_size", 10),
            color=node.get("text_color", "#333333"),
            zorder=4,
        )
        ax.text(
            node["x"],
            node["y"] - 19,
            sublabel,
            ha="center",
            va="center",
            fontsize=8,
            color="#4B5563",
            zorder=4,
        )
    else:
        ax.text(
            node["x"],
            node["y"],
            label,
            ha="center",
            va="center",
            fontsize=node.get("font_size", 10),
            color=node.get("text_color", "#333333"),
            zorder=4,
        )


def draw_edge(ax: plt.Axes, edge: dict, nodes: dict[str, dict]) -> None:
    src = nodes[edge["from"]]
    dst = nodes[edge["to"]]
    style = edge.get("style", "solid")
    linestyle = {"solid": "-", "dashed": "--", "dotted": ":"}.get(style, "-")
    color = edge.get("color", "#555555")
    rad = 0.18 if edge.get("curve") else 0.0
    arrow = FancyArrowPatch(
        (src["x"], src["y"]),
        (dst["x"], dst["y"]),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=edge.get("thickness", 1.4),
        linestyle=linestyle,
        color=color,
        shrinkA=42,
        shrinkB=42,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)
    label = edge.get("label")
    if label:
        mx = (float(src["x"]) + float(dst["x"])) / 2
        my = (float(src["y"]) + float(dst["y"])) / 2 + (20 if rad else 8)
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.5, color=color, zorder=5)


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    width = spec["canvas"]["width"]
    height = spec["canvas"]["height"]
    nodes = {node["id"]: node for node in spec["nodes"]}

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), width, height, facecolor="white", edgecolor="none", zorder=-1))

    for group in spec.get("groups", []):
        draw_group(ax, group, nodes)
    for edge in spec.get("edges", []):
        draw_edge(ax, edge, nodes)
    for node in spec["nodes"]:
        draw_node(ax, node)
    for label in spec.get("labels", []):
        anchor = {"middle": "center", "start": "left", "end": "right"}.get(
            label.get("anchor", "middle"),
            "center",
        )
        ax.text(
            label["x"],
            label["y"],
            label["text"],
            ha=anchor,
            va="center",
            fontsize=label.get("font_size", 10),
            color=label.get("color", "#333333"),
            zorder=5,
        )

    save_fig(fig, "fig1_architecture")


if __name__ == "__main__":
    main()
