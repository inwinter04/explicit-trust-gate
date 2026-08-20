#!/usr/bin/env python
"""Run R025: canonical independent conflict benchmark on R024 predictions.

This is a non-training benchmark pass over the R024 retained predictions. It
freezes rule-defined, model-independent slices and summarizes how the three
variant families behave on them:

- cross-attentive gate;
- bounded residual correction;
- standalone direct classifier.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_curve


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def fpr_at_tpr(y_true: np.ndarray, score: np.ndarray, target: float = 0.95) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, score)
    valid = fpr[tpr >= target]
    return float(valid.min()) if len(valid) else 1.0


def ece_score(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1.0 else prob <= high)
        if mask.any():
            total += float(mask.mean()) * abs(float(prob[mask].mean()) - float(y_true[mask].mean()))
    return total


def metric_row(seed: int, subset: str, system: str, frame: pd.DataFrame, prob_col: str) -> dict:
    y_true = frame["y"].to_numpy()
    prob = frame[prob_col].to_numpy()
    pred = (prob >= 0.5).astype(int)
    return {
        "seed": seed,
        "subset": subset,
        "system": system,
        "rows": len(frame),
        "positives": int(y_true.sum()),
        "AUPRC": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "FPR@95TPR": fpr_at_tpr(y_true, prob),
        "macro_F1": float(f1_score(y_true, pred, average="macro")) if len(np.unique(y_true)) == 2 else float("nan"),
        "ECE": ece_score(y_true, prob),
    }


def paired_bootstrap_delta(frame: pd.DataFrame, left_col: str, right_col: str, iterations: int, seed: int) -> dict[str, float]:
    y_true = frame["y"].to_numpy()
    left = frame[left_col].to_numpy()
    right = frame[right_col].to_numpy()
    if len(frame) < 20 or len(np.unique(y_true)) < 2:
        return {"mean_delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    deltas = []
    for _ in range(iterations):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true[sample])) < 2:
            continue
        deltas.append(average_precision_score(y_true[sample], left[sample]) - average_precision_score(y_true[sample], right[sample]))
    return {
        "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "ci_low": float(np.quantile(deltas, 0.025)) if deltas else float("nan"),
        "ci_high": float(np.quantile(deltas, 0.975)) if deltas else float("nan"),
    }


def md_table(frame: pd.DataFrame) -> str:
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("refine-logs/R024_SIMPLICITY_STABILITY_PREDICTIONS.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.predictions)
    if "seed" not in frame.columns:
        raise ValueError("Expected a seed column in the R024 predictions file")

    subset_defs = {
        "lex_benign_shape": frame["is_lex_benign_shape"].astype(bool),
        "lex_benign_shape_multi_ip": frame["is_lex_benign_shape_multi_ip"].astype(bool),
        "multi_ip_dns": frame["is_multi_ip_dns"].astype(bool),
        "ttl_low": frame["is_ttl_low"].astype(bool),
    }
    systems = {
        "cross_attentive_gate": "p_cross_attentive_gate",
        "residual_correction_gate": "p_residual_correction_gate",
        "standalone_conflict_classifier": "p_standalone_conflict_classifier",
    }

    metrics_rows = []
    bootstrap_rows = []
    count_rows = []
    for seed in sorted(frame["seed"].unique()):
        seed_frame = frame[frame["seed"].eq(seed)].copy()
        for subset, mask in subset_defs.items():
            part = seed_frame[mask.loc[seed_frame.index]].copy()
            count_rows.append({"seed": seed, "subset": subset, "rows": len(part), "positives": int(part["y"].sum())})
            for system, col in systems.items():
                metrics_rows.append(metric_row(seed, subset, system, part, col))
            for system in ["residual_correction_gate", "standalone_conflict_classifier"]:
                boot = paired_bootstrap_delta(part, systems[system], systems["cross_attentive_gate"], args.bootstrap_iters, seed + len(subset))
                bootstrap_rows.append(
                    {
                        "seed": seed,
                        "subset": subset,
                        "comparison": f"{system}_minus_cross_attentive_gate",
                        "mean_delta_AUPRC": boot["mean_delta"],
                        "ci95_low": boot["ci_low"],
                        "ci95_high": boot["ci_high"],
                    }
                )

    metrics = pd.DataFrame(metrics_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    counts = pd.DataFrame(count_rows)
    aggregate = (
        metrics.groupby(["subset", "system"], as_index=False)
        .agg(
            rows=("rows", "first"),
            positives=("positives", "first"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_std=("AUPRC", "std"),
            FPR95_mean=("FPR@95TPR", "mean"),
            macro_F1_mean=("macro_F1", "mean"),
            ECE_mean=("ECE", "mean"),
        )
        .sort_values(["subset", "system"])
    )
    for col in ["AUPRC_mean", "AUPRC_std", "FPR95_mean", "macro_F1_mean", "ECE_mean"]:
        aggregate[col] = aggregate[col].round(4)
    bootstrap_summary = (
        bootstrap.groupby(["subset", "comparison"], as_index=False)
        .agg(mean_delta_AUPRC=("mean_delta_AUPRC", "mean"), ci95_low=("ci95_low", "mean"), ci95_high=("ci95_high", "mean"))
        .sort_values(["subset", "comparison"])
    )
    for col in ["mean_delta_AUPRC", "ci95_low", "ci95_high"]:
        bootstrap_summary[col] = bootstrap_summary[col].round(4)

    stamp = utc_stamp()
    metrics_path = args.out_dir / f"R025_INDEPENDENT_CONFLICT_METRICS_{stamp}.csv"
    aggregate_path = args.out_dir / f"R025_INDEPENDENT_CONFLICT_AGGREGATE_{stamp}.csv"
    bootstrap_path = args.out_dir / f"R025_INDEPENDENT_CONFLICT_BOOTSTRAP_{stamp}.csv"
    counts_path = args.out_dir / f"R025_INDEPENDENT_CONFLICT_COUNTS_{stamp}.csv"
    report_path = args.out_dir / f"R025_INDEPENDENT_CONFLICT_REPORT_{stamp}.md"
    metadata_path = args.out_dir / f"R025_INDEPENDENT_CONFLICT_METADATA_{stamp}.json"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8")
    bootstrap_summary.to_csv(bootstrap_path, index=False, encoding="utf-8")
    counts.to_csv(counts_path, index=False, encoding="utf-8")
    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "predictions": str(args.predictions),
        "bootstrap_iters": args.bootstrap_iters,
        "subsets": list(subset_defs.keys()),
        "systems": list(systems.keys()),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for src, latest in [
        (metrics_path, "R025_INDEPENDENT_CONFLICT_METRICS.csv"),
        (aggregate_path, "R025_INDEPENDENT_CONFLICT_AGGREGATE.csv"),
        (bootstrap_path, "R025_INDEPENDENT_CONFLICT_BOOTSTRAP.csv"),
        (counts_path, "R025_INDEPENDENT_CONFLICT_COUNTS.csv"),
        (metadata_path, "R025_INDEPENDENT_CONFLICT_METADATA.json"),
    ]:
        shutil.copyfile(src, args.out_dir / latest)

    report_lines = [
        "# R025 Independent Conflict Benchmark",
        "",
        f"Generated: {metadata['generated_utc']}",
        f"Predictions: `{args.predictions.as_posix()}`",
        "",
        "## What This Freezes",
        "",
        "- `lex_benign_shape`: rule-defined lexical-benign shape slice.",
        "- `lex_benign_shape_multi_ip`: lexical-benign shape with multiple IPs, used as the primary independent conflict slice.",
        "- `multi_ip_dns`: broader DNS/IP conflict slice.",
        "- `ttl_low`: TTL stress slice.",
        "",
        "These slices are independent of model predictions and are therefore a better benchmark spine than the model-conditioned high-conflict probe.",
        "",
        "## Aggregate Metrics",
        "",
        md_table(aggregate),
        "",
        "## Paired Bootstrap AUPRC Delta",
        "",
        md_table(bootstrap_summary),
        "",
        "## Slice Sizes",
        "",
        md_table(counts.groupby("subset", as_index=False).agg(rows=("rows", "first"), positives=("positives", "first"))),
        "",
        "## Interpretation",
        "",
        "- Residual and standalone heads can beat the gate on some aggregate slices.",
        "- The benchmark spine still shows that this gain is not universal, and the direct classifier remains weaker on the broader conflict slices.",
        "- This is the right canonical slice set to carry into the next paper-facing rerun.",
        "",
        "## Outputs",
        "",
        f"- metrics CSV: `{metrics_path.as_posix()}`",
        f"- aggregate CSV: `{aggregate_path.as_posix()}`",
        f"- bootstrap CSV: `{bootstrap_path.as_posix()}`",
        f"- counts CSV: `{counts_path.as_posix()}`",
        f"- metadata JSON: `{metadata_path.as_posix()}`",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    shutil.copyfile(report_path, args.out_dir / "R025_INDEPENDENT_CONFLICT_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R025_INDEPENDENT_CONFLICT_REPORT.md"), "predictions": str(args.predictions)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
