#!/usr/bin/env python
"""R027: g-value trust-routing diagnostic on R024 predictions.

Question: is the explicit gate's scalar trust weight g informative about
expert quality, i.e. usable for routing, or is it decoration?

On test rows where the two frozen experts disagree (|p_lex - p_dns| >= delta),
we compare three routing policies:

- g-route: trust lexical iff g >= 0.5, else trust behavior;
- fixed-lex: always trust lexical;
- fixed-beh: always trust behavior.

"Routing accuracy" is the fraction of rows where the trusted expert's
probability agrees with the label direction. We also measure mean g in the
two correctness regimes (lex-correct/beh-wrong vs beh-correct/lex-wrong) and
paired bootstrap deltas of routing accuracy. This is a non-training analysis
over retained R024 predictions (3 seeds).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def routing_accuracy(route_probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((route_probs > 0.5) == (y == 1)))


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    return "```\n" + df.to_string(index=index) + "\n```"


def bootstrap_deltas(
    p_lex: np.ndarray,
    p_dns: np.ndarray,
    g: np.ndarray,
    y: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> dict:
    n = len(y)
    gate_minus_lex = np.empty(iterations)
    gate_minus_beh = np.empty(iterations)
    g_regime_diff = np.empty(iterations)
    for it in range(iterations):
        idx = rng.integers(0, n, size=n)
        lex = p_lex[idx]
        beh = p_dns[idx]
        gv = g[idx]
        yv = y[idx]
        route = np.where(gv >= 0.5, lex, beh)
        gate_acc = routing_accuracy(route, yv)
        lex_acc = routing_accuracy(lex, yv)
        beh_acc = routing_accuracy(beh, yv)
        gate_minus_lex[it] = gate_acc - lex_acc
        gate_minus_beh[it] = gate_acc - beh_acc
        lex_correct = (lex > 0.5) == (yv == 1)
        beh_correct = (beh > 0.5) == (yv == 1)
        reg_a = gv[lex_correct & ~beh_correct]
        reg_b = gv[beh_correct & ~lex_correct]
        g_regime_diff[it] = (
            float(reg_a.mean()) - float(reg_b.mean()) if len(reg_a) and len(reg_b) else np.nan
        )
    return {
        "gate_minus_lex_mean": float(np.nanmean(gate_minus_lex)),
        "gate_minus_lex_ci95": [float(np.nanpercentile(gate_minus_lex, 2.5)), float(np.nanpercentile(gate_minus_lex, 97.5))],
        "gate_minus_beh_mean": float(np.nanmean(gate_minus_beh)),
        "gate_minus_beh_ci95": [float(np.nanpercentile(gate_minus_beh, 2.5)), float(np.nanpercentile(gate_minus_beh, 97.5))],
        "g_regime_diff_mean": float(np.nanmean(g_regime_diff)),
        "g_regime_diff_ci95": [float(np.nanpercentile(g_regime_diff, 2.5)), float(np.nanpercentile(g_regime_diff, 97.5))],
    }


def build_case_table(cases: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    cols = [
        "error_type",
        "primary_category",
        "domain",
        "label",
        "g_cross_attentive_gate",
        "p_lex",
        "p_dns",
        "p_cross_attentive_gate",
        "expert_gap",
    ]
    if not all(col in cases.columns for col in cols):
        return pd.DataFrame()
    selected = []
    for err in ("FP", "FN"):
        sub = cases[cases["error_type"] == err].sort_values("expert_gap", ascending=False).head(top_k)
        selected.append(sub)
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected)[cols].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("refine-logs/R024_SIMPLICITY_STABILITY_PREDICTIONS.csv"))
    parser.add_argument("--cases", type=Path, default=Path("refine-logs/R023_QUALITATIVE_DIAGNOSIS_CASES.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--deltas", default="0.05,0.1,0.2")
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--case-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deltas = [float(item) for item in args.deltas.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pred = pd.read_csv(args.predictions)
    pred = pred[pred.get("is_full_test", pd.Series(True, index=pred.index)).astype(bool)].copy()
    pred["y"] = pred["y"].astype(int)
    seeds = sorted(pred["seed"].unique())

    rows = []
    bootstrap_rows = []
    for seed in seeds:
        sub_seed = pred[pred["seed"] == seed]
        p_lex = sub_seed["p_lex"].to_numpy()
        p_dns = sub_seed["p_dns"].to_numpy()
        g = sub_seed["g_cross_attentive_gate"].to_numpy()
        yv = sub_seed["y"].to_numpy()
        for delta in deltas:
            mask = np.abs(p_lex - p_dns) >= delta
            if mask.sum() < 50:
                rows.append(
                    {
                        "seed": seed,
                        "delta": delta,
                        "n_rows": int(mask.sum()),
                        "note": "too_few_rows",
                    }
                )
                continue
            lex = p_lex[mask]
            beh = p_dns[mask]
            gv = g[mask]
            ysub = yv[mask]
            route = np.where(gv >= 0.5, lex, beh)
            gate_acc = routing_accuracy(route, ysub)
            lex_acc = routing_accuracy(lex, ysub)
            beh_acc = routing_accuracy(beh, ysub)
            lex_correct = (lex > 0.5) == (ysub == 1)
            beh_correct = (beh > 0.5) == (ysub == 1)
            reg_a = gv[lex_correct & ~beh_correct]
            reg_b = gv[beh_correct & ~lex_correct]
            boot = bootstrap_deltas(lex, beh, gv, ysub, args.bootstrap_iters, rng)
            rows.append(
                {
                    "seed": seed,
                    "delta": delta,
                    "n_rows": int(mask.sum()),
                    "gate_routing_acc": gate_acc,
                    "fixed_lex_acc": lex_acc,
                    "fixed_beh_acc": beh_acc,
                    "delta_gate_minus_lex": gate_acc - lex_acc,
                    "delta_gate_minus_beh": gate_acc - beh_acc,
                    "g_mean_lex_correct_only": float(reg_a.mean()) if len(reg_a) else np.nan,
                    "g_mean_beh_correct_only": float(reg_b.mean()) if len(reg_b) else np.nan,
                    "n_lex_correct_only": int(len(reg_a)),
                    "n_beh_correct_only": int(len(reg_b)),
                    **{f"boot_{k}": v for k, v in boot.items()},
                }
            )
            bootstrap_rows.append(
                {
                    "seed": seed,
                    "delta": delta,
                    "gate_minus_lex_mean": boot["gate_minus_lex_mean"],
                    "gate_minus_lex_ci95": boot["gate_minus_lex_ci95"],
                    "gate_minus_beh_mean": boot["gate_minus_beh_mean"],
                    "gate_minus_beh_ci95": boot["gate_minus_beh_ci95"],
                    "g_regime_diff_mean": boot["g_regime_diff_mean"],
                    "g_regime_diff_ci95": boot["g_regime_diff_ci95"],
                }
            )

    metrics = pd.DataFrame(rows)
    metrics_csv = args.out_dir / f"R027_GATE_TRUST_ROUTING_METRICS_{utc_stamp()}.csv"
    metrics.to_csv(metrics_csv, index=False)
    metrics.to_csv(args.out_dir / "R027_GATE_TRUST_ROUTING_METRICS.csv", index=False)

    cases = pd.read_csv(args.cases) if args.cases.exists() else pd.DataFrame()
    case_table = build_case_table(cases, top_k=args.case_top_k)
    if not case_table.empty:
        case_csv = args.out_dir / f"R027_GATE_TRUST_CASE_TABLE_{utc_stamp()}.csv"
        case_table.to_csv(case_csv, index=False)
        case_table.to_csv(args.out_dir / "R027_GATE_TRUST_CASE_TABLE.csv", index=False)

    agg = metrics[metrics["delta"].notna()].groupby("delta").agg(
        gate_routing_acc_mean=("gate_routing_acc", "mean"),
        fixed_lex_acc_mean=("fixed_lex_acc", "mean"),
        fixed_beh_acc_mean=("fixed_beh_acc", "mean"),
        delta_gate_minus_lex_mean=("delta_gate_minus_lex", "mean"),
        delta_gate_minus_beh_mean=("delta_gate_minus_beh", "mean"),
        g_mean_lex_correct_only=("g_mean_lex_correct_only", "mean"),
        g_mean_beh_correct_only=("g_mean_beh_correct_only", "mean"),
        n_rows=("n_rows", "mean"),
    )

    md = [f"# R027 Gate Trust-Routing Diagnostic — {utc_stamp()}", ""]
    md.append("Non-training analysis over R024 test predictions (seeds "
              f"{', '.join(str(s) for s in seeds)}), test rows 3,372/seed.")
    md.append("Routing accuracy = fraction of rows where the trusted expert's "
              "probability direction matches the label, restricted to expert-"
              "disagreement rows (|p_lex - p_dns| >= delta).")
    md.append("")
    md.append("## Aggregate by delta (mean over seeds)")
    md.append("")
    md.append(md_table(agg.round(4)))
    md.append("")
    md.append("## Per-seed detail")
    md.append("")
    md.append(md_table(metrics.round(4)))
    md.append("")
    md.append("## Case table (top-k by expert gap, from R023)")
    md.append("")
    if not case_table.empty:
        md.append(md_table(case_table.round(4)))
    else:
        md.append("(no case table generated)")
    md.append("")
    md.append("## Interpretation rule (pre-registered)")
    md.append("- g-route > fixed-lex and g-route > fixed-beh on disagreement rows, "
              "with bootstrap CI excluding 0 -> g is informative for routing.")
    md.append("- g mean separates lex-correct-only from beh-correct-only regimes "
              "(high vs low) with non-overlapping CIs -> trust tracks expert quality.")
    report = "\n".join(md)
    report_path = args.out_dir / f"R027_GATE_TRUST_ROUTING_REPORT_{utc_stamp()}.md"
    report_path.write_text(report, encoding="utf-8")
    (args.out_dir / "R027_GATE_TRUST_ROUTING_REPORT.md").write_text(report, encoding="utf-8")

    metadata = {
        "run": "R027_GATE_TRUST_ROUTING",
        "generated_at": utc_stamp(),
        "inputs_sha256": {
            "predictions": sha256_file(args.predictions),
            "cases": sha256_file(args.cases) if args.cases.exists() else None,
        },
        "seeds": [int(s) for s in seeds],
        "deltas": deltas,
        "bootstrap_iters": args.bootstrap_iters,
        "rng_seed": args.seed,
        "scope": "diagnostic analysis of retained R024 predictions; no new training; "
                 "routing accuracy is a mechanism-use metric, not a detection claim",
    }
    meta_path = args.out_dir / f"R027_GATE_TRUST_ROUTING_METADATA_{utc_stamp()}.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "R027_GATE_TRUST_ROUTING_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics[["seed", "delta", "n_rows", "gate_routing_acc", "fixed_lex_acc", "fixed_beh_acc"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
