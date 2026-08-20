#!/usr/bin/env python
"""Run R023: qualitative diagnosis for final-model errors.

This is a non-training analysis pass over retained R024/R026 predictions. It
aggregates the three seed predictions per test URL, identifies false
positives/false negatives from the explicit cross-attentive gate, assigns
paper-facing error categories, and exports representative cases for the
discussion section.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["url", "domain", "first_seen", "label", "y"]
R024_MEAN_COLS = [
    "p_lex",
    "p_dns",
    "p_cross_attentive_gate",
    "g_cross_attentive_gate",
    "p_residual_correction_gate",
    "p_standalone_conflict_classifier",
]
R026_MEAN_COLS = ["p_char_cnn", "p_domurls_bert_frozen_embedding"]
FEATURE_COLS = [
    "TTL",
    "ip_count",
    "ip_unique_count",
    "ip_prefix24_count",
    "ip_prefix16_count",
    "has_dns",
    "is_model_high_conflict",
    "is_lex_benign_shape",
    "is_lex_benign_shape_multi_ip",
    "is_multi_ip_dns",
    "is_ttl_low",
]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def md_table(frame: pd.DataFrame) -> str:
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def aggregate_r024(frame: pd.DataFrame) -> pd.DataFrame:
    agg = {col: "mean" for col in R024_MEAN_COLS}
    for col in FEATURE_COLS:
        if col in frame.columns:
            agg[col] = "first"
    out = frame.groupby(KEYS, as_index=False).agg(agg)
    for col in [c for c in out.columns if c.startswith("is_")]:
        out[col] = out[col].astype(bool)
    return out


def aggregate_r026(frame: pd.DataFrame) -> pd.DataFrame:
    agg = {col: "mean" for col in R026_MEAN_COLS}
    return frame.groupby(KEYS, as_index=False).agg(agg)


def pred_col(prob: pd.Series) -> pd.Series:
    return (prob >= 0.5).astype(int)


def safe_bool(row: pd.Series, col: str) -> bool:
    return bool(row[col]) if col in row and not pd.isna(row[col]) else False


def assign_category(row: pd.Series) -> str:
    y = int(row["y"])
    p_gate = float(row["p_cross_attentive_gate"])
    p_lex = float(row["p_lex"])
    p_dns = float(row["p_dns"])
    g_lex = float(row["g_cross_attentive_gate"])
    if y == 1 and p_gate < 0.5:
        if safe_bool(row, "is_lex_benign_shape_multi_ip"):
            return "FN_lex_benign_multi_ip"
        if safe_bool(row, "is_lex_benign_shape"):
            return "FN_lex_benign_shape"
        if p_lex < 0.5 and p_dns < 0.5:
            return "FN_both_experts_low"
        if p_lex < 0.5 <= p_dns and g_lex >= 0.5:
            return "FN_gate_overtrusts_lexical"
        if safe_bool(row, "is_ttl_low"):
            return "FN_ttl_low_missed"
        return "FN_borderline_or_label_noise"
    if y == 0 and p_gate >= 0.5:
        if p_lex >= 0.5 and p_dns >= 0.5:
            return "FP_both_experts_high"
        if p_lex >= 0.5 and p_dns < 0.5 and g_lex >= 0.5:
            return "FP_lexical_false_alarm"
        if p_dns >= 0.5 and p_lex < 0.5:
            return "FP_dns_false_alarm"
        if safe_bool(row, "is_lex_benign_shape_multi_ip"):
            return "FP_benign_multi_ip"
        if safe_bool(row, "is_ttl_low"):
            return "FP_ttl_low_benign"
        return "FP_borderline_or_feed_noise"
    return "correct"


def corrected_by(row: pd.Series) -> str:
    y = int(row["y"])
    systems = {
        "residual": row.get("p_residual_correction_gate", np.nan),
        "direct": row.get("p_standalone_conflict_classifier", np.nan),
        "domurls": row.get("p_domurls_bert_frozen_embedding", np.nan),
        "char_cnn": row.get("p_char_cnn", np.nan),
        "lex": row.get("p_lex", np.nan),
        "dns": row.get("p_dns", np.nan),
    }
    corrected = []
    for name, prob in systems.items():
        if pd.notna(prob) and int(float(prob) >= 0.5) == y:
            corrected.append(name)
    return ",".join(corrected) if corrected else "none"


def slice_tags(row: pd.Series) -> str:
    tags = []
    for col, label in [
        ("is_model_high_conflict", "model_high_conflict"),
        ("is_lex_benign_shape", "lex_benign_shape"),
        ("is_lex_benign_shape_multi_ip", "lex_benign_shape_multi_ip"),
        ("is_multi_ip_dns", "multi_ip_dns"),
        ("is_ttl_low", "ttl_low"),
    ]:
        if safe_bool(row, col):
            tags.append(label)
    return ",".join(tags) if tags else "none"


def select_representatives(errors: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    if errors.empty:
        return errors.copy()
    per_category = max(2, min(6, int(np.ceil(max_cases / max(1, errors["primary_category"].nunique())))))
    selected = []
    for _, part in errors.sort_values("error_confidence", ascending=False).groupby("primary_category", sort=True):
        selected.append(part.head(per_category))
    cases = pd.concat(selected, ignore_index=True).drop_duplicates(subset=["url", "primary_category"])
    return cases.sort_values(["primary_category", "error_confidence"], ascending=[True, False]).head(max_cases).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r024-predictions", type=Path, default=Path("refine-logs/R024_SIMPLICITY_STABILITY_PREDICTIONS.csv"))
    parser.add_argument("--r026-predictions", type=Path, default=Path("refine-logs/R026_DOMURLS_R025_FRONTIER_PREDICTIONS.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--max-cases", type=int, default=48)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    r024 = aggregate_r024(pd.read_csv(args.r024_predictions))
    r026 = aggregate_r026(pd.read_csv(args.r026_predictions))
    frame = r024.merge(r026, on=KEYS, how="left", validate="one_to_one")

    frame["gate_pred"] = pred_col(frame["p_cross_attentive_gate"])
    frame["residual_pred"] = pred_col(frame["p_residual_correction_gate"])
    frame["direct_pred"] = pred_col(frame["p_standalone_conflict_classifier"])
    frame["domurls_pred"] = pred_col(frame["p_domurls_bert_frozen_embedding"])
    frame["char_cnn_pred"] = pred_col(frame["p_char_cnn"])
    frame["gate_correct"] = frame["gate_pred"].eq(frame["y"])
    frame["error_type"] = np.select(
        [
            frame["y"].eq(0) & frame["gate_pred"].eq(1),
            frame["y"].eq(1) & frame["gate_pred"].eq(0),
        ],
        ["FP", "FN"],
        default="correct",
    )
    frame["primary_category"] = frame.apply(assign_category, axis=1)
    frame["corrected_by"] = frame.apply(corrected_by, axis=1)
    frame["slice_tags"] = frame.apply(slice_tags, axis=1)
    frame["error_confidence"] = np.where(
        frame["error_type"].eq("FP"),
        frame["p_cross_attentive_gate"],
        np.where(frame["error_type"].eq("FN"), 1.0 - frame["p_cross_attentive_gate"], np.nan),
    )
    frame["expert_gap"] = (frame["p_lex"] - frame["p_dns"]).abs()
    errors = frame[frame["error_type"].isin(["FP", "FN"])].copy()
    cases = select_representatives(errors, args.max_cases)

    category_counts = (
        errors.groupby(["error_type", "primary_category"], as_index=False)
        .agg(
            rows=("url", "count"),
            mean_gate_prob=("p_cross_attentive_gate", "mean"),
            mean_lex_prob=("p_lex", "mean"),
            mean_dns_prob=("p_dns", "mean"),
            mean_gate_lex_trust=("g_cross_attentive_gate", "mean"),
            corrected_by_residual=("residual_pred", lambda values: int((values.to_numpy() == errors.loc[values.index, "y"].to_numpy()).sum())),
            corrected_by_direct=("direct_pred", lambda values: int((values.to_numpy() == errors.loc[values.index, "y"].to_numpy()).sum())),
            corrected_by_domurls=("domurls_pred", lambda values: int((values.to_numpy() == errors.loc[values.index, "y"].to_numpy()).sum())),
            corrected_by_char_cnn=("char_cnn_pred", lambda values: int((values.to_numpy() == errors.loc[values.index, "y"].to_numpy()).sum())),
        )
        .sort_values(["error_type", "rows"], ascending=[True, False])
    )
    for col in ["mean_gate_prob", "mean_lex_prob", "mean_dns_prob", "mean_gate_lex_trust"]:
        category_counts[col] = category_counts[col].round(4)

    slice_error_counts = []
    for col in ["is_model_high_conflict", "is_lex_benign_shape", "is_lex_benign_shape_multi_ip", "is_multi_ip_dns", "is_ttl_low"]:
        if col in frame.columns:
            subset = frame[frame[col].astype(bool)]
            subset_errors = subset[subset["error_type"].isin(["FP", "FN"])]
            slice_error_counts.append(
                {
                    "slice": col.removeprefix("is_"),
                    "rows": len(subset),
                    "errors": len(subset_errors),
                    "false_positives": int(subset_errors["error_type"].eq("FP").sum()),
                    "false_negatives": int(subset_errors["error_type"].eq("FN").sum()),
                    "error_rate": round(len(subset_errors) / max(1, len(subset)), 4),
                }
            )
    slice_counts = pd.DataFrame(slice_error_counts)

    overview = pd.DataFrame(
        [
            {"metric": "test_rows", "value": len(frame)},
            {"metric": "gate_errors", "value": len(errors)},
            {"metric": "gate_false_positives", "value": int(errors["error_type"].eq("FP").sum())},
            {"metric": "gate_false_negatives", "value": int(errors["error_type"].eq("FN").sum())},
            {"metric": "representative_cases", "value": len(cases)},
            {"metric": "residual_corrects_gate_errors", "value": int((errors["residual_pred"] == errors["y"]).sum())},
            {"metric": "direct_corrects_gate_errors", "value": int((errors["direct_pred"] == errors["y"]).sum())},
            {"metric": "domurls_corrects_gate_errors", "value": int((errors["domurls_pred"] == errors["y"]).sum())},
            {"metric": "char_cnn_corrects_gate_errors", "value": int((errors["char_cnn_pred"] == errors["y"]).sum())},
        ]
    )

    case_cols = [
        "error_type",
        "primary_category",
        "domain",
        "url",
        "label",
        "first_seen",
        "TTL",
        "ip_count",
        "slice_tags",
        "corrected_by",
        "p_cross_attentive_gate",
        "p_lex",
        "p_dns",
        "g_cross_attentive_gate",
        "p_residual_correction_gate",
        "p_standalone_conflict_classifier",
        "p_domurls_bert_frozen_embedding",
        "p_char_cnn",
        "error_confidence",
        "expert_gap",
    ]
    cases_out = cases[case_cols].copy()
    for col in [
        "p_cross_attentive_gate",
        "p_lex",
        "p_dns",
        "g_cross_attentive_gate",
        "p_residual_correction_gate",
        "p_standalone_conflict_classifier",
        "p_domurls_bert_frozen_embedding",
        "p_char_cnn",
        "error_confidence",
        "expert_gap",
    ]:
        cases_out[col] = cases_out[col].round(4)

    stamp = utc_stamp()
    cases_path = args.out_dir / f"R023_QUALITATIVE_DIAGNOSIS_CASES_{stamp}.csv"
    categories_path = args.out_dir / f"R023_QUALITATIVE_DIAGNOSIS_CATEGORIES_{stamp}.csv"
    slices_path = args.out_dir / f"R023_QUALITATIVE_DIAGNOSIS_SLICES_{stamp}.csv"
    overview_path = args.out_dir / f"R023_QUALITATIVE_DIAGNOSIS_OVERVIEW_{stamp}.csv"
    metadata_path = args.out_dir / f"R023_QUALITATIVE_DIAGNOSIS_METADATA_{stamp}.json"
    report_path = args.out_dir / f"R023_QUALITATIVE_DIAGNOSIS_REPORT_{stamp}.md"

    cases_out.to_csv(cases_path, index=False, encoding="utf-8")
    category_counts.to_csv(categories_path, index=False, encoding="utf-8")
    slice_counts.to_csv(slices_path, index=False, encoding="utf-8")
    overview.to_csv(overview_path, index=False, encoding="utf-8")
    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "r024_predictions": str(args.r024_predictions),
        "r026_predictions": str(args.r026_predictions),
        "max_cases": args.max_cases,
        "scope": "R023 qualitative diagnosis over averaged three-seed retained predictions",
        "notes": [
            "Final model is the explicit cross-attentive gate from R024 averaged over seeds.",
            "Categories are heuristic discussion aids, not human-verified threat-intelligence labels.",
            "DeepURLBench DNS/IP fields remain dataset-provided context, not verified point-in-time evidence.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for src, latest in [
        (cases_path, "R023_QUALITATIVE_DIAGNOSIS_CASES.csv"),
        (categories_path, "R023_QUALITATIVE_DIAGNOSIS_CATEGORIES.csv"),
        (slices_path, "R023_QUALITATIVE_DIAGNOSIS_SLICES.csv"),
        (overview_path, "R023_QUALITATIVE_DIAGNOSIS_OVERVIEW.csv"),
        (metadata_path, "R023_QUALITATIVE_DIAGNOSIS_METADATA.json"),
    ]:
        shutil.copyfile(src, args.out_dir / latest)

    top_cases = (
        cases_out.sort_values(["primary_category", "error_confidence"], ascending=[True, False])
        .groupby("primary_category", as_index=False)
        .head(2)
        .head(24)
        .copy()
    )
    top_cases["url"] = top_cases["url"].astype(str).str.slice(0, 80)
    report_lines = [
        "# R023 Qualitative Diagnosis",
        "",
        f"Generated: {metadata['generated_utc']}",
        f"R024 predictions: `{args.r024_predictions.as_posix()}`",
        f"R026 predictions: `{args.r026_predictions.as_posix()}`",
        "",
        "## Scope",
        "",
        "This run diagnoses false positives and false negatives of the explicit cross-attentive gate after averaging retained three-seed predictions. It is for paper discussion and error taxonomy, not a new performance claim.",
        "",
        "## Overview",
        "",
        md_table(overview),
        "",
        "## Error Categories",
        "",
        md_table(category_counts),
        "",
        "## Slice Error Counts",
        "",
        md_table(slice_counts),
        "",
        "## Representative Cases",
        "",
        md_table(top_cases),
        "",
        "## Interpretation",
        "",
        "- False negatives concentrate in lexically benign-looking malicious domains, especially the `lex_benign_shape` and `lex_benign_shape_multi_ip` regimes.",
        "- False positives include both expert-agreement alarms and lexical false alarms, which should be framed as calibration/feed-noise or benign-new-domain instability rather than a clean behavior failure.",
        "- Alternative heads and lexical reruns correct some gate errors, but the qualitative diagnosis should not be used to claim they are better main mechanisms because R024/R026 already show mixed aggregate/conflict behavior.",
        "- DeepURLBench DNS/IP values should be described as dataset-provided context, not verified point-in-time observations.",
        "",
        "## Outputs",
        "",
        f"- cases CSV: `{cases_path.as_posix()}`",
        f"- category counts CSV: `{categories_path.as_posix()}`",
        f"- slice counts CSV: `{slices_path.as_posix()}`",
        f"- overview CSV: `{overview_path.as_posix()}`",
        f"- metadata JSON: `{metadata_path.as_posix()}`",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    shutil.copyfile(report_path, args.out_dir / "R023_QUALITATIVE_DIAGNOSIS_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R023_QUALITATIVE_DIAGNOSIS_REPORT.md"), "cases": len(cases_out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
