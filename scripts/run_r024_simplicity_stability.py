#!/usr/bin/env python
"""Run R024 stability check for R020/R021 simplicity variants.

This aggregates multiple seeds for the explicit cross-attentive mixture gate,
bounded residual correction, and standalone direct classifier. It reports:

- per-seed metrics on full test, model-conditioned high-conflict, and
  rule-defined diagnostic subsets;
- mean/std aggregates;
- paired bootstrap AUPRC deltas against the minimal gate.

This is still a frozen-expert DeepURLBench pilot, not final point-in-time
evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_r020_residual_correction import (
    DNS_NUMERIC,
    GateNet,
    ResidualGateNet,
    add_features,
    behavior_matrix,
    clipped_logit,
    count_parameters,
    domain_overlap_rows,
    ece_score,
    fit_experts,
    fpr_at_tpr,
    high_conflict_mask,
    md_table,
    metric_row,
    oracle_gate_target,
    seed_everything,
    sha256_file,
    split_parts,
    train_gate,
    utc_stamp,
    y,
)
from run_r021_conflict_classifier import (
    DirectClassifierNet,
    evaluate_direct_classifier,
    evaluate_minimal_gate,
    train_direct_classifier,
)

from sklearn.metrics import average_precision_score


def add_rule_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["domain"] = out["domain"].fillna("").astype(str).str.lower()
    out["sld"] = out["domain"].map(lambda value: value.split(".")[0] if value else "")
    out["sld_len"] = out["sld"].str.len()
    out["domain_len"] = out["domain"].str.len()
    out["num_dots"] = out["domain"].str.count(r"\.")
    out["digit_ratio"] = out["sld"].map(lambda value: sum(ch.isdigit() for ch in value) / max(1, len(value)))
    out["has_hyphen"] = out["sld"].str.contains("-", regex=False)
    for col in ["TTL", "ip_count", "ip_unique_count", "ip_prefix24_count", "ip_prefix16_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def subset_masks(frame: pd.DataFrame, model_high_conflict: np.ndarray) -> dict[str, np.ndarray]:
    lexical_benign_shape = (
        (frame["digit_ratio"] <= 0.15)
        & (~frame["has_hyphen"])
        & (frame["sld_len"] <= 24)
        & (frame["num_dots"] <= 2)
        & (frame["domain_len"] <= 25)
    ).to_numpy()
    multi_ip = (frame["ip_count"] >= 2).to_numpy()
    ttl_low = (frame["TTL"] <= 60).to_numpy()
    return {
        "full_test": np.ones(len(frame), dtype=bool),
        "model_high_conflict": model_high_conflict,
        "lex_benign_shape": lexical_benign_shape,
        "multi_ip_dns": multi_ip,
        "ttl_low": ttl_low,
        "lex_benign_shape_multi_ip": lexical_benign_shape & multi_ip,
    }


def paired_bootstrap_delta(y_true: np.ndarray, left: np.ndarray, right: np.ndarray, iterations: int, seed: int) -> dict[str, float]:
    if len(y_true) < 20 or len(np.unique(y_true)) < 2:
        return {"mean_delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_true))
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


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--seeds", default="20260819,20260820,20260821")
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = add_features(pd.read_csv(args.sample))
    train, val, test = split_parts(frame)
    experts = fit_experts(train, val, test)
    val_behavior, val_missing = behavior_matrix(val, train)
    test_behavior, test_missing = behavior_matrix(test, train)
    p_lex_val, p_dns_val = experts["p_lex_val"], experts["p_dns_val"]
    p_lex_test, p_dns_test = experts["p_lex_test"], experts["p_dns_test"]
    y_val_np = y(val)
    y_test = y(test)

    val_inputs = (
        torch.tensor(clipped_logit(p_lex_val), dtype=torch.float32),
        torch.tensor(clipped_logit(p_dns_val), dtype=torch.float32),
        torch.from_numpy(val_behavior),
        torch.from_numpy(val_missing),
    )
    test_inputs = (
        torch.tensor(clipped_logit(p_lex_test), dtype=torch.float32),
        torch.tensor(clipped_logit(p_dns_test), dtype=torch.float32),
        torch.from_numpy(test_behavior),
        torch.from_numpy(test_missing),
    )
    gate_target = torch.tensor(oracle_gate_target(y_val_np, p_lex_val, p_dns_val), dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    rule_frame = add_rule_features(test)
    masks = subset_masks(rule_frame, high_conflict_mask(p_lex_test, p_dns_test))

    metrics_rows = []
    diagnostics_rows = []
    bootstrap_rows = []
    prediction_frames = []
    systems = {
        "cross_attentive_gate": "p_cross_attentive_gate",
        "residual_correction_gate": "p_residual_correction_gate",
        "standalone_conflict_classifier": "p_standalone_conflict_classifier",
    }

    for seed in seeds:
        seed_everything(seed)
        minimal = train_gate(GateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val, seed, device, residual=False)
        residual = train_gate(ResidualGateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val, seed, device, residual=True)
        direct = train_direct_classifier(DirectClassifierNet(len(DNS_NUMERIC)), val_inputs, y_val, seed, device)

        minimal_prob, minimal_gate = evaluate_minimal_gate(minimal, test_inputs, p_lex_test, p_dns_test, device)
        residual_prob, residual_gate, residual_delta = train_eval_residual(residual, test_inputs, device)
        direct_prob = evaluate_direct_classifier(direct, test_inputs, device)

        predictions = rule_frame[["split", "url", "domain", "first_seen", "label", *DNS_NUMERIC]].copy()
        predictions["seed"] = seed
        predictions["y"] = y_test
        predictions["p_lex"] = p_lex_test
        predictions["p_dns"] = p_dns_test
        predictions["p_cross_attentive_gate"] = minimal_prob
        predictions["g_cross_attentive_gate"] = minimal_gate
        predictions["p_residual_correction_gate"] = residual_prob
        predictions["g_residual_correction_gate"] = residual_gate
        predictions["delta_residual_correction_gate"] = residual_delta
        predictions["p_standalone_conflict_classifier"] = direct_prob
        for subset_name, mask in masks.items():
            predictions[f"is_{subset_name}"] = mask
        prediction_frames.append(predictions)

        probs = {
            "cross_attentive_gate": minimal_prob,
            "residual_correction_gate": residual_prob,
            "standalone_conflict_classifier": direct_prob,
        }
        for subset_name, mask in masks.items():
            for system, prob in probs.items():
                row = metric_row(system, subset_name, y_test[mask], prob[mask])
                row["seed"] = seed
                metrics_rows.append(row)
            for system in ["residual_correction_gate", "standalone_conflict_classifier"]:
                boot = paired_bootstrap_delta(y_test[mask], probs[system][mask], minimal_prob[mask], args.bootstrap_iters, seed + len(subset_name))
                bootstrap_rows.append(
                    {
                        "seed": seed,
                        "subset": subset_name,
                        "comparison": f"{system}_minus_cross_attentive_gate",
                        "mean_delta_AUPRC": boot["mean_delta"],
                        "ci95_low": boot["ci_low"],
                        "ci95_high": boot["ci_high"],
                    }
                )
        diagnostics_rows.extend(
            [
                {
                    "seed": seed,
                    "system": "cross_attentive_gate",
                    "mean_g_lex": round(float(minimal_gate.mean()), 4),
                    "choose_lex_pct": round(float((minimal_gate >= 0.5).mean() * 100), 4),
                    "params": count_parameters(minimal),
                    "mean_abs_delta": float("nan"),
                },
                {
                    "seed": seed,
                    "system": "residual_correction_gate",
                    "mean_g_lex": round(float(residual_gate.mean()), 4),
                    "choose_lex_pct": round(float((residual_gate >= 0.5).mean() * 100), 4),
                    "params": count_parameters(residual),
                    "mean_abs_delta": round(float(np.abs(residual_delta).mean()), 4),
                },
                {
                    "seed": seed,
                    "system": "standalone_conflict_classifier",
                    "mean_g_lex": float("nan"),
                    "choose_lex_pct": float("nan"),
                    "params": count_parameters(direct),
                    "mean_abs_delta": float("nan"),
                },
            ]
        )

    metrics = pd.DataFrame(metrics_rows)
    diagnostics = pd.DataFrame(diagnostics_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    aggregate = (
        metrics.groupby(["split", "system"], as_index=False)
        .agg(
            rows=("rows", "first"),
            positives=("positives", "first"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_std=("AUPRC", "std"),
            FPR95_mean=("FPR@95TPR", "mean"),
            macro_F1_mean=("macro_F1", "mean"),
            ECE_mean=("ECE", "mean"),
        )
        .sort_values(["split", "system"])
    )
    for col in ["AUPRC_mean", "AUPRC_std", "FPR95_mean", "macro_F1_mean", "ECE_mean"]:
        aggregate[col] = aggregate[col].round(4)
    bootstrap_summary = (
        bootstrap.groupby(["subset", "comparison"], as_index=False)
        .agg(
            mean_delta_AUPRC=("mean_delta_AUPRC", "mean"),
            ci95_low=("ci95_low", "mean"),
            ci95_high=("ci95_high", "mean"),
        )
        .sort_values(["subset", "comparison"])
    )
    for col in ["mean_delta_AUPRC", "ci95_low", "ci95_high"]:
        bootstrap_summary[col] = bootstrap_summary[col].round(4)

    stamp = utc_stamp()
    metrics_path = args.out_dir / f"R024_SIMPLICITY_STABILITY_METRICS_{stamp}.csv"
    aggregate_path = args.out_dir / f"R024_SIMPLICITY_STABILITY_AGGREGATE_{stamp}.csv"
    bootstrap_path = args.out_dir / f"R024_SIMPLICITY_STABILITY_BOOTSTRAP_{stamp}.csv"
    diagnostics_path = args.out_dir / f"R024_SIMPLICITY_STABILITY_DIAGNOSTICS_{stamp}.csv"
    predictions_path = args.out_dir / f"R024_SIMPLICITY_STABILITY_PREDICTIONS_{stamp}.csv"
    metadata_path = args.out_dir / f"R024_SIMPLICITY_STABILITY_METADATA_{stamp}.json"
    report_path = args.out_dir / f"R024_SIMPLICITY_STABILITY_REPORT_{stamp}.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8")
    bootstrap_summary.to_csv(bootstrap_path, index=False, encoding="utf-8")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8")
    predictions_all.to_csv(predictions_path, index=False, encoding="utf-8")
    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sample": str(args.sample),
        "sample_sha256": sha256_file(args.sample),
        "seeds": seeds,
        "bootstrap_iters": args.bootstrap_iters,
        "device": str(device),
        "scope": "R024 multi-seed stability check for R020/R021 simplicity variants",
        "split_rows": {name: len(part) for name, part in [("train", train), ("val", val), ("test", test)]},
        "split_domain_overlap": domain_overlap_rows(train, val, test),
        "subsets": {name: {"rows": int(mask.sum()), "positives": int(y_test[mask].sum())} for name, mask in masks.items()},
        "notes": [
            "DeepURLBench DNS/IP remains dataset-provided context, not verified point-in-time behavior evidence.",
            "model_high_conflict is model-conditioned and diagnostic only.",
            "rule-defined subsets are independent of model predictions but still heuristic.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for src, latest in [
        (metrics_path, "R024_SIMPLICITY_STABILITY_METRICS.csv"),
        (aggregate_path, "R024_SIMPLICITY_STABILITY_AGGREGATE.csv"),
        (bootstrap_path, "R024_SIMPLICITY_STABILITY_BOOTSTRAP.csv"),
        (diagnostics_path, "R024_SIMPLICITY_STABILITY_DIAGNOSTICS.csv"),
        (predictions_path, "R024_SIMPLICITY_STABILITY_PREDICTIONS.csv"),
        (metadata_path, "R024_SIMPLICITY_STABILITY_METADATA.json"),
    ]:
        shutil.copyfile(src, args.out_dir / latest)
    report = "\n".join(
        [
            "# R024 Simplicity Stability Check",
            "",
            f"Generated: {metadata['generated_utc']}",
            f"Sample: `{args.sample.as_posix()}`",
            f"Seeds: `{','.join(str(seed) for seed in seeds)}`",
            f"Device: `{device}`",
            "",
            "## Aggregate Metrics",
            "",
            md_table(aggregate),
            "",
            "## Paired Bootstrap AUPRC Delta",
            "",
            "Positive deltas mean the overbuilt variant is better than the minimal gate.",
            "",
            md_table(bootstrap_summary),
            "",
            "## Diagnostics",
            "",
            md_table(diagnostics),
            "",
            "## Subset Sizes",
            "",
            md_table(pd.DataFrame([{"subset": key, **value} for key, value in metadata["subsets"].items()])),
            "",
            "## Interpretation",
            "",
            "- This run tests whether R020/R021 single-seed patterns are stable across three seeds.",
            "- Full-test aggregate gains by richer heads should not be converted into a conflict-robust gate claim unless conflict subsets also improve.",
            "- Rule-defined subsets are included as a bridge toward an independent conflict benchmark, but they remain heuristic.",
            "",
            "## Outputs",
            "",
            f"- metrics CSV: `{metrics_path.as_posix()}`",
            f"- aggregate CSV: `{aggregate_path.as_posix()}`",
            f"- bootstrap CSV: `{bootstrap_path.as_posix()}`",
            f"- diagnostics CSV: `{diagnostics_path.as_posix()}`",
            f"- predictions CSV: `{predictions_path.as_posix()}`",
            f"- metadata JSON: `{metadata_path.as_posix()}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    shutil.copyfile(report_path, args.out_dir / "R024_SIMPLICITY_STABILITY_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R024_SIMPLICITY_STABILITY_REPORT.md"), "seeds": seeds, "device": str(device)}, ensure_ascii=False))
    return 0


def train_eval_residual(model: ResidualGateNet, inputs: tuple[torch.Tensor, ...], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        gate, delta, _ = model(*(item.to(device) for item in inputs))
        lex = inputs[0].to(device)
        dns = inputs[1].to(device)
        prob = torch.sigmoid(gate * lex + (1 - gate) * dns + delta).cpu().numpy()
    return prob, gate.cpu().numpy(), delta.cpu().numpy()


if __name__ == "__main__":
    raise SystemExit(main())
