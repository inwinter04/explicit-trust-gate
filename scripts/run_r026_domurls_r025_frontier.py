#!/usr/bin/env python
"""Run R026: DomURLs_BERT rerun on the frozen R025 conflict spine.

This is the final lexical-frontier rerun for the current paper branch. It
evaluates DomURLs_BERT frozen embeddings plus a lightweight logistic head and a
from-scratch char-CNN on the same DeepURLBench with-DNS split used by R024/R025,
then compares them against the frozen R025 gate/residual/direct benchmark spine.

The slice definitions are model-independent:

- lex_benign_shape
- lex_benign_shape_multi_ip
- multi_ip_dns
- ttl_low
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from run_r019_frontier_check import (
    add_features,
    fit_embedding_logreg,
    load_domurls_embeddings,
    predict_char_cnn,
    seed_everything,
    split_parts,
    subset_metrics,
    train_char_cnn,
)


SEEDS = (20260819, 20260820, 20260821)
DOMURLS_BERT = "amahdaouy/DomURLs_BERT"


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def md_table(frame: pd.DataFrame) -> str:
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def paired_bootstrap_delta(
    y_true: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
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


def subset_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    domain = frame["domain"].fillna("").astype(str).str.lower()
    sld = domain.map(lambda value: value.split(".")[0] if value else "")
    sld_len = sld.str.len()
    domain_len = domain.str.len()
    num_dots = domain.str.count(r"\.")
    digit_ratio = sld.map(lambda value: sum(ch.isdigit() for ch in value) / max(1, len(value)))
    has_hyphen = sld.str.contains("-", regex=False)
    multi_ip = pd.to_numeric(frame["ip_count"], errors="coerce").fillna(0) >= 2
    ttl_low = pd.to_numeric(frame["TTL"], errors="coerce").fillna(0) <= 60
    lexical_benign_shape = (
        (digit_ratio <= 0.15)
        & (~has_hyphen)
        & (sld_len <= 24)
        & (num_dots <= 2)
        & (domain_len <= 25)
    ).to_numpy()
    return {
        "full_test": np.ones(len(frame), dtype=bool),
        "lex_benign_shape": lexical_benign_shape,
        "lex_benign_shape_multi_ip": lexical_benign_shape & multi_ip.to_numpy(),
        "multi_ip_dns": multi_ip.to_numpy(),
        "ttl_low": ttl_low.to_numpy(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--hf-cache", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seeds", default="20260819,20260820,20260821")
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument(
        "--r025-aggregate",
        type=Path,
        default=Path("refine-logs/R025_INDEPENDENT_CONFLICT_AGGREGATE.csv"),
    )
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = add_features(pd.read_csv(args.data))
    train, val, test = split_parts(frame)
    masks = subset_masks(test)

    train_texts = train["domain"].tolist()
    val_texts = val["domain"].tolist()
    test_texts = test["domain"].tolist()

    domurls_embeddings, domurls_status = load_domurls_embeddings(train_texts + val_texts + test_texts, args.hf_cache)
    if domurls_embeddings is None:
        raise RuntimeError(f"DomURLs_BERT embeddings unavailable: {domurls_status}")

    train_n = len(train_texts)
    val_n = len(val_texts)
    train_emb = domurls_embeddings[:train_n]
    val_emb = domurls_embeddings[train_n : train_n + val_n]
    test_emb = domurls_embeddings[train_n + val_n :]

    y_train = train["label_int"].to_numpy()
    y_val = val["label_int"].to_numpy()
    y_test = test["label_int"].to_numpy()

    metrics_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in seeds:
        seed_everything(seed)

        _, _, char_history, char_model, vocab = train_char_cnn(
            train_texts,
            y_train,
            val_texts,
            y_val,
            seed,
        )
        char_probs = {
            subset: predict_char_cnn(char_model, vocab, test[mask]["domain"].tolist())
            for subset, mask in masks.items()
        }

        domurls_model = fit_embedding_logreg(train_emb, y_train, seed)
        domurls_probs = {
            subset: domurls_model.predict_proba(test_emb[mask])[:, 1]
            for subset, mask in masks.items()
        }

        prediction_frame = test[["split", "url", "domain", "first_seen", "label", "TTL", "ip_count"]].copy()
        prediction_frame["seed"] = seed
        prediction_frame["y"] = y_test
        prediction_frame["p_char_cnn"] = domurls_probs["full_test"] * 0.0
        prediction_frame["p_domurls_bert_frozen_embedding"] = domurls_probs["full_test"]
        prediction_frame["p_char_cnn"] = predict_char_cnn(char_model, vocab, test_texts)
        for subset, mask in masks.items():
            prediction_frame[f"is_{subset}"] = mask
        prediction_frames.append(prediction_frame)

        for subset, mask in masks.items():
            part = test[mask].copy()
            metrics_rows.append({"seed": seed, "system": "char_cnn", **subset_metrics(part, char_probs[subset], subset)})
            metrics_rows.append(
                {
                    "seed": seed,
                    "system": "domurls_bert_frozen_embedding",
                    **subset_metrics(part, domurls_probs[subset], subset),
                }
            )
            boot = paired_bootstrap_delta(
                part["label_int"].to_numpy(),
                domurls_probs[subset],
                char_probs[subset],
                args.bootstrap_iters,
                seed + len(subset),
            )
            bootstrap_rows.append(
                {
                    "seed": seed,
                    "subset": subset,
                    "comparison": "domurls_bert_minus_char_cnn",
                    "mean_delta_AUPRC": boot["mean_delta"],
                    "ci95_low": boot["ci_low"],
                    "ci95_high": boot["ci_high"],
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

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

    combined = aggregate.copy()
    gate_summary = None
    if args.r025_aggregate.exists():
        gate_summary = pd.read_csv(args.r025_aggregate)
        gate_summary = gate_summary[gate_summary["subset"].isin(["lex_benign_shape", "lex_benign_shape_multi_ip", "multi_ip_dns", "ttl_low"])]
        gate_summary = gate_summary.rename(
            columns={
                "AUPRC_mean": "AUPRC_mean",
                "AUPRC_std": "AUPRC_std",
                "FPR95_mean": "FPR95_mean",
                "macro_F1_mean": "macro_F1_mean",
                "ECE_mean": "ECE_mean",
            }
        )
        gate_summary["source"] = "R025_gate_spine"
        combined["source"] = "R026_domurls_rerun"
        combined = pd.concat([gate_summary, combined], ignore_index=True, sort=False)
    else:
        combined["source"] = "R026_domurls_rerun"

    gate_baseline = None
    if gate_summary is not None and not gate_summary.empty:
        gate_baseline = (
            gate_summary[gate_summary["system"].eq("cross_attentive_gate")]
            .loc[:, ["subset", "AUPRC_mean"]]
            .rename(columns={"AUPRC_mean": "gate_AUPRC_mean"})
        )
        combined = combined.merge(gate_baseline, on="subset", how="left")
        combined["AUPRC_vs_gate"] = (combined["AUPRC_mean"] - combined["gate_AUPRC_mean"]).round(4)

    stamp = utc_stamp()
    metrics_path = args.out_dir / f"R026_DOMURLS_R025_FRONTIER_METRICS_{stamp}.csv"
    aggregate_path = args.out_dir / f"R026_DOMURLS_R025_FRONTIER_AGGREGATE_{stamp}.csv"
    bootstrap_path = args.out_dir / f"R026_DOMURLS_R025_FRONTIER_BOOTSTRAP_{stamp}.csv"
    combined_path = args.out_dir / f"R026_DOMURLS_R025_FRONTIER_COMBINED_{stamp}.csv"
    predictions_path = args.out_dir / f"R026_DOMURLS_R025_FRONTIER_PREDICTIONS_{stamp}.csv"
    metadata_path = args.out_dir / f"R026_DOMURLS_R025_FRONTIER_METADATA_{stamp}.json"
    report_path = args.out_dir / f"R026_DOMURLS_R025_FRONTIER_REPORT_{stamp}.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8")
    bootstrap_summary.to_csv(bootstrap_path, index=False, encoding="utf-8")
    combined.to_csv(combined_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")

    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "data": str(args.data),
        "seeds": seeds,
        "bootstrap_iters": args.bootstrap_iters,
        "domurls_status": domurls_status,
        "domurls_bert": DOMURLS_BERT,
        "split_rows": {name: len(part) for name, part in [("train", train), ("val", val), ("test", test)]},
        "subset_counts": {name: {"rows": int(mask.sum()), "positives": int(y_test[mask].sum())} for name, mask in masks.items()},
        "r025_aggregate": str(args.r025_aggregate),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for src, latest in [
        (metrics_path, "R026_DOMURLS_R025_FRONTIER_METRICS.csv"),
        (aggregate_path, "R026_DOMURLS_R025_FRONTIER_AGGREGATE.csv"),
        (bootstrap_path, "R026_DOMURLS_R025_FRONTIER_BOOTSTRAP.csv"),
        (combined_path, "R026_DOMURLS_R025_FRONTIER_COMBINED.csv"),
        (predictions_path, "R026_DOMURLS_R025_FRONTIER_PREDICTIONS.csv"),
        (metadata_path, "R026_DOMURLS_R025_FRONTIER_METADATA.json"),
    ]:
        shutil.copyfile(src, args.out_dir / latest)

    report_lines = [
        "# R026 DomURLs_BERT Rerun on the Frozen R025 Spine",
        "",
        f"Generated: {metadata['generated_utc']}",
        f"Data: `{args.data.as_posix()}`",
        f"Seeds: `{','.join(str(seed) for seed in seeds)}`",
        f"DomURLs_BERT status: `{domurls_status}`",
        "",
        "## What This Tests",
        "",
        "- `lex_benign_shape`",
        "- `lex_benign_shape_multi_ip`",
        "- `multi_ip_dns`",
        "- `ttl_low`",
        "",
        "These are the frozen, model-independent slices from R025.",
        "",
        "## R025 Gate Spine",
        "",
        md_table(
            combined[combined["source"].eq("R025_gate_spine")][
                ["subset", "system", "rows", "positives", "AUPRC_mean", "FPR95_mean", "macro_F1_mean", "ECE_mean"]
            ]
        ),
        "",
        "## DomURLs_BERT / char-CNN Rerun",
        "",
        md_table(
            combined[combined["source"].eq("R026_domurls_rerun")][
                [
                    "subset",
                    "system",
                    "rows",
                    "positives",
                    "AUPRC_mean",
                    "AUPRC_std",
                    "FPR95_mean",
                    "macro_F1_mean",
                    "ECE_mean",
                    "AUPRC_vs_gate",
                ]
            ]
        ),
        "",
        "## Paired Bootstrap AUPRC Delta",
        "",
        md_table(bootstrap_summary),
        "",
        "## Slice Sizes",
        "",
        md_table(pd.DataFrame([{"subset": key, "rows": int(mask.sum()), "positives": int(y_test[mask].sum())} for key, mask in masks.items()])),
        "",
        "## Interpretation",
        "",
        "- Compare the rerun table against the frozen R025 gate spine, not against the old model-conditioned conflict probe.",
        "- The primary question is whether the lexical frontier rerun changes the mechanism story on the independent slices.",
        "- If DomURLs_BERT does not dominate the explicit gate on the primary slice, the paper should keep the explicit gate as the main mechanism and treat DomURLs_BERT as a backbone choice rather than the claim driver.",
        "",
        "## Outputs",
        "",
        f"- metrics CSV: `{metrics_path.as_posix()}`",
        f"- aggregate CSV: `{aggregate_path.as_posix()}`",
        f"- bootstrap CSV: `{bootstrap_path.as_posix()}`",
        f"- combined CSV: `{combined_path.as_posix()}`",
        f"- predictions CSV: `{predictions_path.as_posix()}`",
        f"- metadata JSON: `{metadata_path.as_posix()}`",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    shutil.copyfile(report_path, args.out_dir / "R026_DOMURLS_R025_FRONTIER_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R026_DOMURLS_R025_FRONTIER_REPORT.md"), "domurls_status": domurls_status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
