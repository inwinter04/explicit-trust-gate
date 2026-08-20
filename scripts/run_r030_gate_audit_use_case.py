#!/usr/bin/env python
"""R030: g-conditioned alarm audit -- a concrete downstream use of the gate's
scalar trust weight.

Security triage setting: an analyst receives candidate malicious domains
flagged by a high-confidence lexical alarm (p_lex >= 0.9). The gate's g says
how much to trust that lexical alarm. We measure alarm precision and coverage
in three g tiers:

- high-trust (g >= 0.9): gate strongly trusts the lexical alarm;
- mid (0.5 <= g < 0.9);
- low-trust (g < 0.5): gate leans toward the behavior view.

If g is informative, low-trust lexical alarms should be markedly less precise
than high-trust alarms (the gate downgrades likely-false lexical alarms),
giving the analyst a concrete triage signal. The same audit is applied to
high-confidence behavior alarms (p_dns >= 0.9) with tiers g <= 0.1
(gate trusts behavior), mid, and g >= 0.9.

Non-training analysis over retained R024 predictions (3 seeds).
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


def tier_audit(
    p_alarm: np.ndarray, g: np.ndarray, y: np.ndarray, alarm_threshold: float, low: float, high: float
) -> list[dict]:
    flag = p_alarm >= alarm_threshold
    rows = []
    for name, mask in [
        ("all_alarms", flag),
        ("high_trust", flag & (g >= high)),
        ("mid_trust", flag & (g >= low) & (g < high)),
        ("low_trust", flag & (g < low)),
    ]:
        n = int(mask.sum())
        rows.append(
            {
                "tier": name,
                "n_alarms": n,
                "precision": float(y[mask].mean()) if n else float("nan"),
                "coverage_of_all_positives": float(y[mask].sum() / max(y.sum(), 1)),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("refine-logs/R024_SIMPLICITY_STABILITY_PREDICTIONS.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions)
    pred = pred[pred.get("is_full_test", pd.Series(True, index=pred.index)).astype(bool)].copy()
    pred["y"] = pred["y"].astype(int)
    seeds = sorted(pred["seed"].unique())

    rows = []
    for seed in seeds:
        sub = pred[pred["seed"] == seed]
        yv = sub["y"].to_numpy()
        g = sub["g_cross_attentive_gate"].to_numpy()
        for use_case, p_alarm, alarm_threshold, low, high in [
            ("lexical_alarms", sub["p_lex"].to_numpy(), 0.9, 0.5, 0.9),
            ("behavior_alarms", sub["p_dns"].to_numpy(), 0.9, 0.1, 0.9),
        ]:
            for r in tier_audit(p_alarm, g, yv, alarm_threshold, low, high):
                rows.append({"seed": seed, "use_case": use_case, **r})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / f"R030_GATE_AUDIT_USE_CASE_METRICS_{utc_stamp()}.csv", index=False)
    metrics.to_csv(args.out_dir / "R030_GATE_AUDIT_USE_CASE_METRICS.csv", index=False)

    valid = metrics[metrics["precision"].notna()]
    agg = (
        valid.groupby(["use_case", "tier"])
        .agg(n_alarms_mean=("n_alarms", "mean"), precision_mean=("precision", "mean"), coverage_mean=("coverage_of_all_positives", "mean"))
        .reset_index()
    )
    agg.to_csv(args.out_dir / f"R030_GATE_AUDIT_USE_CASE_AGGREGATE_{utc_stamp()}.csv", index=False)
    agg.to_csv(args.out_dir / "R030_GATE_AUDIT_USE_CASE_AGGREGATE.csv", index=False)

    md = [
        f"# R030 g-Conditioned Alarm Audit — {utc_stamp()}",
        "",
        "Concrete triage use of the gate's scalar trust weight. Alarm precision "
        "and positive coverage by g tier, per use case (lexical high-confidence "
        "alarms p_lex>=0.9; behavior high-confidence alarms p_dns>=0.9), mean "
        "over three seeds.",
        "",
        "```",
        agg.round(4).to_string(index=False),
        "```",
        "",
        "Interpretation: if g is informative, low-trust alarms are less precise "
        "than high-trust alarms, so g gives an analyst a triage signal "
        "(prioritize high-trust alarms).",
    ]
    report = "\n".join(md)
    (args.out_dir / f"R030_GATE_AUDIT_USE_CASE_REPORT_{utc_stamp()}.md").write_text(report, encoding="utf-8")
    (args.out_dir / "R030_GATE_AUDIT_USE_CASE_REPORT.md").write_text(report, encoding="utf-8")

    metadata = {
        "run": "R030_GATE_AUDIT_USE_CASE",
        "generated_at": utc_stamp(),
        "inputs_sha256": {"predictions": sha256_file(args.predictions)},
        "seeds": [int(s) for s in seeds],
        "scope": "non-training triage-use analysis over R024 predictions; "
                 "precisions are alarm-tier statistics, not detection claims",
    }
    (args.out_dir / f"R030_GATE_AUDIT_USE_CASE_METADATA_{utc_stamp()}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "R030_GATE_AUDIT_USE_CASE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(agg.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
