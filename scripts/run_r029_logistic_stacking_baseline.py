#!/usr/bin/env python
"""R029: classic logistic-stacking fusion baseline over the two frozen experts.

The paper's main table compares the explicit gate only with its own neural
variants (residual correction, direct head). A hostile reviewer will ask how
the gate compares with a standard, published-style fusion baseline. Logistic
stacking -- a logistic regression over the frozen expert probabilities -- is
the canonical simple fusion baseline (cf. stacking / late-fusion literature).

Pre-registered main variant: LR(C=1.0, balanced) over [p_lex, p_dns] fitted
on validation rows and evaluated on test rows. Supporting variants: logit
inputs and an interaction feature. The experts are identical to R024/R028.
Deterministic (frozen experts + fixed LR random state), so a single run is
reported alongside the 3-seed neural rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from run_r020_residual_correction import (
    add_features,
    ece_score,
    fit_experts,
    fpr_at_tpr,
    high_conflict_mask,
    split_parts,
    y,
)


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, (prob > threshold).astype(int), average="macro", zero_division=0))


def feature_variant(p_lex: np.ndarray, p_dns: np.ndarray, name: str) -> np.ndarray:
    if name == "prob":
        return np.column_stack([p_lex, p_dns])
    if name == "logit":
        p = np.clip(np.column_stack([p_lex, p_dns]), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    if name == "prob_interaction":
        return np.column_stack([p_lex, p_dns, p_lex * p_dns])
    raise ValueError(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--variants", default="prob,logit,prob_interaction")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = add_features(pd.read_csv(args.sample))
    train, val, test = split_parts(frame)
    probs = fit_experts(train, val, test)
    y_val = y(val)
    y_test = y(test)
    hc_mask_test = high_conflict_mask(probs["p_lex_test"], probs["p_dns_test"])

    rows = []
    for name in variants:
        X_val = feature_variant(probs["p_lex_val"], probs["p_dns_val"], name)
        X_test = feature_variant(probs["p_lex_test"], probs["p_dns_test"], name)
        stack = LogisticRegression(C=args.c, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=args.seed)
        stack.fit(X_val, y_val)
        p_test = stack.predict_proba(X_test)[:, 1]
        for split_name, mask in [("full_test", np.ones(len(y_test), dtype=bool)), ("model_high_conflict", hc_mask_test)]:
            y_sub = y_test[mask]
            p_sub = p_test[mask]
            rows.append(
                {
                    "variant": name,
                    "split": split_name,
                    "rows": int(mask.sum()),
                    "positives": int(y_sub.sum()),
                    "AUPRC": float(average_precision_score(y_sub, p_sub)),
                    "FPR@95TPR": float(fpr_at_tpr(y_sub, p_sub)),
                    "macro_F1": macro_f1(y_sub, p_sub),
                    "ECE": float(ece_score(y_sub, p_sub)),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / f"R029_LOGISTIC_STACKING_METRICS_{utc_stamp()}.csv", index=False)
    metrics.to_csv(args.out_dir / "R029_LOGISTIC_STACKING_METRICS.csv", index=False)

    md = [
        f"# R029 Logistic-Stacking Fusion Baseline — {utc_stamp()}",
        "",
        "Classic late-fusion baseline: LR over frozen expert probabilities, "
        "fitted on validation rows, evaluated on test. Main pre-registered "
        f"variant: `prob` with C={args.c}. Deterministic single run.",
        "",
        "```",
        metrics.round(4).to_string(index=False),
        "```",
        "",
        "Success criterion: provide a standard fusion reference point for the "
        "main table; the gate's diagnostic contribution is not about beating "
        "stacking on AUPRC, and a higher stacking AUPRC is expected and "
        "disclosed.",
    ]
    report = "\n".join(md)
    (args.out_dir / f"R029_LOGISTIC_STACKING_REPORT_{utc_stamp()}.md").write_text(report, encoding="utf-8")
    (args.out_dir / "R029_LOGISTIC_STACKING_REPORT.md").write_text(report, encoding="utf-8")

    metadata = {
        "run": "R029_LOGISTIC_STACKING",
        "generated_at": utc_stamp(),
        "inputs_sha256": {"sample": sha256_file(args.sample)},
        "variants": variants,
        "c": args.c,
        "random_state": args.seed,
        "scope": "classic fusion baseline over the same frozen experts as R024; "
                 "deterministic; not a new model",
    }
    (args.out_dir / f"R029_LOGISTIC_STACKING_METADATA_{utc_stamp()}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "R029_LOGISTIC_STACKING_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
