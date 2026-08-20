#!/usr/bin/env python
"""Run CPU baselines on local DeepURLBench time samples.

This script is an M1 checkpoint, not the final neural model run. It tests:

- lexical-only URL/domain char TF-IDF baselines;
- DNS behavior-only signal on with-DNS rows;
- simple URL+DNS fusion;
- lexical robustness on without-DNS rows.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260708
DNS_NUMERIC = [
    "TTL",
    "ttl_log1p",
    "ttl_missing",
    "ip_count",
    "ip_unique_count",
    "ip_prefix24_count",
    "ip_prefix16_count",
    "ip_first_octet_min",
    "ip_first_octet_max",
    "ip_first_octet_mean",
    "ip_private_special_count",
    "has_dns",
]


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr[tpr >= target_tpr]
    return float(valid.min()) if len(valid) else 1.0


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= low) & (y_prob < high if high < 1.0 else y_prob <= high)
        if not mask.any():
            continue
        accuracy = y_true[mask].mean()
        confidence = y_prob[mask].mean()
        ece += mask.mean() * abs(float(confidence) - float(accuracy))
    return float(ece)


def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    pred = (y_prob >= 0.5).astype(int)
    return {
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "FPR@95TPR": fpr_at_tpr(y_true, y_prob),
        "macro_F1": float(f1_score(y_true, pred, average="macro")),
        "ECE": ece_score(y_true, y_prob),
    }


def y(df: pd.DataFrame) -> np.ndarray:
    return df["label"].eq("malicious").astype(int).to_numpy()


def onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def text_pipeline(c: float = 4.0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=SEED)),
        ]
    )


def tabular_pipeline(numeric_cols: list[str], categorical_cols: list[str], c: float = 1.0) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="[MISS]")), ("onehot", onehot_encoder())]), categorical_cols))
    pre = ColumnTransformer(transformers)
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=SEED)),
        ]
    )


def fusion_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("url", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, sublinear_tf=True, min_df=2), "url"),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="[MISS]")), ("onehot", onehot_encoder())]), categorical_cols),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=SEED)),
        ]
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["first_seen"] = pd.to_datetime(out["first_seen"], errors="coerce")
    out["TTL"] = pd.to_numeric(out.get("TTL", np.nan), errors="coerce")
    out["has_dns"] = out.get("has_dns", False).astype(str).str.lower().isin(["true", "1"]).astype(int)
    for col in [
        "ip_count",
        "ip_unique_count",
        "ip_prefix24_count",
        "ip_prefix16_count",
        "ip_first_octet_min",
        "ip_first_octet_max",
        "ip_first_octet_mean",
        "ip_private_special_count",
    ]:
        out[col] = pd.to_numeric(out.get(col, 0), errors="coerce").fillna(0)
    out["url"] = out["url"].fillna("").astype(str)
    out["domain"] = out["domain"].fillna("").astype(str)
    out["url_len"] = pd.to_numeric(out.get("url_len", out["url"].map(len)), errors="coerce")
    out["domain_len"] = pd.to_numeric(out.get("domain_len", out["domain"].map(len)), errors="coerce")
    out["num_dots"] = pd.to_numeric(out.get("num_dots", out["domain"].str.count(r"\.")), errors="coerce")
    out["ttl_missing"] = out["TTL"].isna().astype(int)
    out["ttl_log1p"] = np.log1p(out["TTL"].fillna(0).clip(lower=0))
    out["tld"] = out.get("tld", out["domain"].str.rsplit(".", n=1).str[-1]).fillna("[MISS]").astype(str).str.slice(0, 32)
    out = out[out["label"].isin(["benign", "malicious"]) & out["first_seen"].notna()].copy()
    return out


def split_parts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df[df["split"].eq("train")].copy(),
        df[df["split"].eq("val")].copy(),
        df[df["split"].eq("test")].copy(),
    )


def evaluate(name: str, model: Pipeline, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, x_cols: str | list[str]) -> list[dict]:
    if isinstance(x_cols, str):
        x_train = train[x_cols].astype(str)
        x_val = val[x_cols].astype(str)
        x_test = test[x_cols].astype(str)
    else:
        x_train = train[x_cols]
        x_val = val[x_cols]
        x_test = test[x_cols]
    model.fit(x_train, y(train))
    rows: list[dict] = []
    for split_name, split_df, x_split in [("train", train, x_train), ("val", val, x_val), ("test", test, x_test)]:
        prob = model.predict_proba(x_split)[:, 1]
        rows.append({"system": name, "split": split_name, **metrics(y(split_df), prob)})
    return rows


def split_diagnostics(name: str, df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for split in ["train", "val", "test"]:
        part = df[df["split"].eq(split)]
        rows.append(
            {
                "sample": name,
                "split": split,
                "rows": len(part),
                "benign": int(part["label"].eq("benign").sum()),
                "malicious": int(part["label"].eq("malicious").sum()),
                "first_seen_min": str(part["first_seen"].min()),
                "first_seen_max": str(part["first_seen"].max()),
                "has_dns_pct": round(float(part["has_dns"].mean() * 100), 2) if len(part) else 0.0,
            }
        )
    return rows


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def run_with_dns(df: pd.DataFrame) -> list[dict]:
    train, val, test = split_parts(df)
    dns_numeric = DNS_NUMERIC
    meta_numeric = dns_numeric + ["url_len", "domain_len", "num_dots"]
    dns_cat: list[str] = []
    meta_cat = ["tld"]
    rows: list[dict] = []
    rows.extend(evaluate("with_dns_url_char_tfidf", text_pipeline(), train, val, test, "url"))
    rows.extend(evaluate("with_dns_domain_char_tfidf", text_pipeline(), train, val, test, "domain"))
    rows.extend(evaluate("with_dns_dns_behavior", tabular_pipeline(dns_numeric, dns_cat), train, val, test, dns_numeric + dns_cat))
    rows.extend(evaluate("with_dns_metadata_plus_dns", tabular_pipeline(meta_numeric, meta_cat), train, val, test, meta_numeric + meta_cat))
    rows.extend(evaluate("with_dns_url_plus_dns_fusion", fusion_pipeline(meta_numeric, meta_cat), train, val, test, ["url"] + meta_numeric + meta_cat))
    return rows


def run_without_dns(df: pd.DataFrame) -> list[dict]:
    train, val, test = split_parts(df)
    rows: list[dict] = []
    rows.extend(evaluate("without_dns_url_char_tfidf", text_pipeline(), train, val, test, "url"))
    rows.extend(evaluate("without_dns_domain_char_tfidf", text_pipeline(), train, val, test, "domain"))
    return rows


def interpretation(metrics_df: pd.DataFrame) -> list[str]:
    test = metrics_df[metrics_df["split"].eq("test")].set_index("system")
    notes: list[str] = []
    if "with_dns_dns_behavior" in test.index:
        dns_auprc = float(test.loc["with_dns_dns_behavior", "AUPRC"])
        dns_fpr = float(test.loc["with_dns_dns_behavior", "FPR@95TPR"])
        if dns_auprc < 0.6:
            notes.append(f"DNS behavior-only is weak on this checkpoint (AUPRC={dns_auprc:.4f}); do not claim trust-gating yet.")
        else:
            notes.append(f"DNS behavior-only is above a weak-signal threshold (AUPRC={dns_auprc:.4f}), but FPR@95TPR={dns_fpr:.4f} still matters.")
    if {"with_dns_url_char_tfidf", "with_dns_url_plus_dns_fusion"}.issubset(test.index):
        lexical = float(test.loc["with_dns_url_char_tfidf", "AUPRC"])
        fusion = float(test.loc["with_dns_url_plus_dns_fusion", "AUPRC"])
        delta = fusion - lexical
        notes.append(f"Simple URL+DNS fusion delta over URL lexical baseline: {delta:+.4f} AUPRC.")
    notes.append("These are CPU baselines on a sampled benchmark; they decide the next experiment, not the final paper claim.")
    return notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-dns", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--without-dns", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_without_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_stamp = utc_stamp()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with_dns = add_features(pd.read_csv(args.with_dns))
    without_dns = add_features(pd.read_csv(args.without_dns))
    rows = run_with_dns(with_dns)
    rows.extend(run_without_dns(without_dns))

    metrics_df = pd.DataFrame(rows)
    metrics_path = args.out_dir / f"DEEPURLBENCH_LOCAL_BASELINE_METRICS_{run_stamp}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    latest_metrics = args.out_dir / "DEEPURLBENCH_LOCAL_BASELINE_METRICS.csv"
    shutil.copyfile(metrics_path, latest_metrics)

    split_df = pd.DataFrame(split_diagnostics("with_dns", with_dns) + split_diagnostics("without_dns", without_dns))
    test_df = metrics_df[metrics_df["split"].eq("test")].copy()
    for col in ["AUPRC", "FPR@95TPR", "macro_F1", "ECE"]:
        test_df[col] = test_df[col].round(4)

    report = "\n".join(
        [
            "# DeepURLBench Local Baseline Report",
            "",
            f"Generated: {datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
            f"With-DNS sample: `{args.with_dns.as_posix()}`",
            f"Without-DNS sample: `{args.without_dns.as_posix()}`",
            "",
            "## Split Diagnostics",
            "",
            df_to_markdown(split_df),
            "",
            "## Test Metrics",
            "",
            df_to_markdown(test_df[["system", "AUPRC", "FPR@95TPR", "macro_F1", "ECE"]]),
            "",
            "## Interpretation",
            "",
            "\n".join(f"- {note}" for note in interpretation(metrics_df)),
            "",
            "## Outputs",
            "",
            f"- metrics CSV: `{metrics_path.as_posix()}`",
            f"- latest metrics CSV: `{latest_metrics.as_posix()}`",
        ]
    )
    report_path = args.out_dir / f"DEEPURLBENCH_LOCAL_BASELINE_REPORT_{run_stamp}.md"
    latest_report = args.out_dir / "DEEPURLBENCH_LOCAL_BASELINE_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    shutil.copyfile(report_path, latest_report)
    print(f"[done] Wrote {latest_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
