#!/usr/bin/env python
"""R028a: informative-behavior subset analysis on R024 predictions.

Question: does the gate achieve behavior-conditioned gains (AUPRC above the
lexical-only expert) on model-independent subsets where behavior evidence is
informative, rather than only falling back when behavior is degenerate?

Subsets (all model-independent):
- full_test (context);
- multi_ip_dns (flag from R024);
- lex_benign_shape_multi_ip (flag from R024);
- ttl_informative: has_dns=1, TTL > 1, TTL not missing.

Non-training analysis over retained R024 predictions (3 seeds).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    return "```\n" + df.to_string(index=index) + "\n```"


def subset_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    if name == "full_test":
        return pd.Series(True, index=frame.index)
    if name == "multi_ip_dns":
        return frame["is_multi_ip_dns"].astype(bool)
    if name == "lex_benign_shape_multi_ip":
        return frame["is_lex_benign_shape_multi_ip"].astype(bool)
    if name == "ttl_informative":
        return (frame["has_dns"].astype(int) == 1) & (frame["TTL"] > 1) & (frame["ttl_missing"].astype(int) == 0)
    raise ValueError(name)


def bootstrap_auprc_delta(
    y: np.ndarray, gate_p: np.ndarray, lex_p: np.ndarray, iterations: int, rng: np.random.Generator
) -> tuple[float, list[float]]:
    n = len(y)
    deltas = np.empty(iterations)
    for it in range(iterations):
        idx = rng.integers(0, n, size=n)
        try:
            deltas[it] = average_precision_score(y[idx], gate_p[idx]) - average_precision_score(y[idx], lex_p[idx])
        except ValueError:
            deltas[it] = np.nan
    return float(np.nanmean(deltas)), [float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("refine-logs/R024_SIMPLICITY_STABILITY_PREDICTIONS.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pred = pd.read_csv(args.predictions)
    pred = pred[pred.get("is_full_test", pd.Series(True, index=pred.index)).astype(bool)].copy()
    pred["y"] = pred["y"].astype(int)
    seeds = sorted(pred["seed"].unique())
    subsets = ["full_test", "multi_ip_dns", "lex_benign_shape_multi_ip", "ttl_informative"]

    rows = []
    for seed in seeds:
        sub = pred[pred["seed"] == seed]
        for name in subsets:
            mask = subset_mask(sub, name)
            part = sub[mask]
            if len(part) < 50 or part["y"].nunique() < 2:
                rows.append({"seed": seed, "subset": name, "n_rows": int(len(part)), "note": "too_small_or_single_class"})
                continue
            yv = part["y"].to_numpy()
            gate_p = part["p_cross_attentive_gate"].to_numpy()
            lex_p = part["p_lex"].to_numpy()
            gv = part["g_cross_attentive_gate"].to_numpy()
            gate_auprc = float(average_precision_score(yv, gate_p))
            lex_auprc = float(average_precision_score(yv, lex_p))
            delta_mean, ci = bootstrap_auprc_delta(yv, gate_p, lex_p, args.bootstrap_iters, rng)
            rows.append(
                {
                    "seed": seed,
                    "subset": name,
                    "n_rows": int(len(part)),
                    "gate_AUPRC": gate_auprc,
                    "lexical_AUPRC": lex_auprc,
                    "gate_minus_lexical": gate_auprc - lex_auprc,
                    "boot_delta_mean": delta_mean,
                    "boot_ci95_low": ci[0],
                    "boot_ci95_high": ci[1],
                    "mean_g_lex": float(gv.mean()),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / f"R028A_INFORMATIVE_SUBSET_METRICS_{utc_stamp()}.csv", index=False)
    metrics.to_csv(args.out_dir / "R028A_INFORMATIVE_SUBSET_METRICS.csv", index=False)

    valid = metrics[metrics["gate_AUPRC"].notna()]
    agg = (
        valid.groupby("subset")
        .agg(
            gate_AUPRC_mean=("gate_AUPRC", "mean"),
            gate_AUPRC_std=("gate_AUPRC", "std"),
            lexical_AUPRC_mean=("lexical_AUPRC", "mean"),
            gate_minus_lexical_mean=("gate_minus_lexical", "mean"),
            boot_delta_mean=("boot_delta_mean", "mean"),
            mean_g_lex=("mean_g_lex", "mean"),
            n_rows_mean=("n_rows", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(args.out_dir / f"R028A_INFORMATIVE_SUBSET_AGGREGATE_{utc_stamp()}.csv", index=False)
    agg.to_csv(args.out_dir / "R028A_INFORMATIVE_SUBSET_AGGREGATE.csv", index=False)

    md = [
        f"# R028a Informative-Behavior Subset Analysis — {utc_stamp()}",
        "",
        "Non-training analysis over R024 test predictions (3 seeds). "
        "Subsets are model-independent: multi-IP DNS rows, lexically benign "
        "multi-IP rows, and rows with has_dns=1 and non-degenerate TTL (>1s).",
        "",
        "## Aggregate (mean over seeds)",
        "",
        md_table(agg.round(4)),
        "",
        "## Per-seed detail",
        "",
        md_table(metrics.round(4)),
        "",
        "## Success criterion (pre-registered)",
        "- gate AUPRC - lexical AUPRC >= +0.005 on at least 2 informative subsets.",
        "- Failure interpretation: gate underuses behavior on real data; "
        "positive behavior-conditioned evidence then rests on R028b only.",
    ]
    report = "\n".join(md)
    (args.out_dir / f"R028A_INFORMATIVE_SUBSET_REPORT_{utc_stamp()}.md").write_text(report, encoding="utf-8")
    (args.out_dir / "R028A_INFORMATIVE_SUBSET_REPORT.md").write_text(report, encoding="utf-8")

    metadata = {
        "run": "R028A_INFORMATIVE_SUBSET",
        "generated_at": utc_stamp(),
        "inputs_sha256": {"predictions": sha256_file(args.predictions)},
        "seeds": [int(s) for s in seeds],
        "subsets": subsets,
        "bootstrap_iters": args.bootstrap_iters,
        "rng_seed": args.seed,
        "scope": "diagnostic retrospective subset analysis; subsets pre-registered "
                 "in EXPERIMENT_PLAN.md; no new training",
    }
    (args.out_dir / f"R028A_INFORMATIVE_SUBSET_METADATA_{utc_stamp()}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "R028A_INFORMATIVE_SUBSET_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(agg.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
