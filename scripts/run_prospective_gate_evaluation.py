#!/usr/bin/env python
"""Evaluate the frozen DeepURLBench gate on the verified point-in-time sample.

The gate, lexical expert, and DNS/IP expert are trained only on DeepURLBench
(train/val). They are then applied to the local 2026-07-07 prospective sample
(200 domains, 100 malicious / 100 benign) whose DNS features were collected
within the amended 25h point-in-time window (R015B pass).

This is a small, noisy cross-sample transfer check with an approximate
feature mapping, not a deployment benchmark. All claims must stay diagnostic.
"""

from __future__ import annotations

import argparse
import ast
import ipaddress
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_r020_residual_correction import (
    DNS_NUMERIC,
    GateNet,
    add_features,
    behavior_matrix,
    clipped_logit,
    ece_score,
    fit_experts,
    fpr_at_tpr,
    oracle_gate_target,
    seed_everything,
    sha256_file,
    split_parts,
    train_gate,
    y,
)
from run_r021_conflict_classifier import evaluate_minimal_gate


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def parse_ip_list(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str)]
    return []


def reserved_network_summary(ips: list[str]) -> dict:
    """Fractions of unique IPv4 addresses inside reserved ranges."""
    v4 = [ip for ip in ips if ip.count(".") == 3]
    if not v4:
        return {
            "unique_ipv4": 0,
            "benchmark_198_18_0_0_15_fraction": float("nan"),
            "special_union_fraction": float("nan"),
        }
    benchmark = ipaddress.ip_network("198.18.0.0/15")
    special = [
        ipaddress.ip_network("0.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("100.64.0.0/10"),
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.0.0.0/24"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("198.18.0.0/15"),
        ipaddress.ip_network("198.51.100.0/24"),
        ipaddress.ip_network("203.0.113.0/24"),
    ]
    n_bench = 0
    n_special = 0
    for ip in v4:
        try:
            addr = ipaddress.IPv4Address(ip)
        except ipaddress.AddressValueError:
            continue
        if addr in benchmark:
            n_bench += 1
        if any(addr in net for net in special):
            n_special += 1
    return {
        "unique_ipv4": len(v4),
        "benchmark_198_18_0_0_15_fraction": round(n_bench / len(v4), 6),
        "special_union_fraction": round(n_special / len(v4), 6),
    }


def ip_v4(ip: str) -> bool:
    return ip.count(".") == 3


def is_private_special(ip: str) -> bool:
    if not ip_v4(ip):
        return False
    try:
        addr = ipaddress.IPv4Address(ip)
    except ipaddress.AddressValueError:
        return False
    return any(
        addr in net
        for net in [
            ipaddress.ip_network("0.0.0.0/8"),
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("100.64.0.0/10"),
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("169.254.0.0/16"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.0.0.0/24"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("198.18.0.0/15"),
            ipaddress.ip_network("198.51.100.0/24"),
            ipaddress.ip_network("203.0.113.0/24"),
        ]
    )


def build_prospective_features(cohort: pd.DataFrame, dns: pd.DataFrame) -> pd.DataFrame:
    merged = cohort.merge(dns, on="domain", how="left", suffixes=("", "_dns"))
    rows = []
    for _, row in merged.iterrows():
        ips = parse_ip_list(row.get("a_records")) + parse_ip_list(row.get("aaaa_records"))
        v4 = [ip for ip in ips if ip_v4(ip)]
        unique = set(ips)
        p24 = {ip for ip in v4 for ip in [".".join(ip.split(".")[:3])]}
        p16 = {ip for ip in v4 for ip in [".".join(ip.split(".")[:2])]}
        first_octets = [int(ip.split(".")[0]) for ip in v4]
        ttl = row.get("ttl_mean")
        if pd.isna(ttl):
            ttl = row.get("ttl_min")
        if pd.isna(ttl):
            ttl = row.get("ttl_max")
        ttl = float(ttl) if not pd.isna(ttl) else np.nan
        rows.append(
            {
                "url": row["raw_url"] if isinstance(row.get("raw_url"), str) and row["raw_url"] else f"https://{row['domain']}/",
                "domain": row["domain"],
                "first_seen": row.get("t0_utc") or "2026-07-07T00:00:00Z",
                "label": row["label"],
                "split": "test",
                "TTL": ttl,
                "ip_count": float(row.get("ip_count")) if pd.notna(row.get("ip_count")) else float(len(unique)),
                "ip_unique_count": float(len(unique)),
                "ip_prefix24_count": float(len(p24)),
                "ip_prefix16_count": float(len(p16)),
                "ip_first_octet_min": float(min(first_octets)) if first_octets else np.nan,
                "ip_first_octet_max": float(max(first_octets)) if first_octets else np.nan,
                "ip_first_octet_mean": float(np.mean(first_octets)) if first_octets else np.nan,
                "ip_private_special_count": float(sum(1 for ip in unique if is_private_special(ip))),
                "has_dns": str(bool(unique) or str(row.get("a_status")) == "OK" or str(row.get("aaaa_status")) == "OK").lower(),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--cohort", type=Path, default=Path("data/raw/2026-07-07/domain_cohort.csv"))
    parser.add_argument("--dns", type=Path, default=Path("data/raw/2026-07-07/dns_observations.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--seeds", default="20260819,20260820,20260821")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs_sha = {
        "deepurlbench_sample": sha256_file(args.sample),
        "prospective_cohort": sha256_file(args.cohort),
        "prospective_dns": sha256_file(args.dns),
        "script": sha256_file(Path(__file__).resolve()),
    }

    # --- train experts and gate on DeepURLBench only ---
    frame = add_features(pd.read_csv(args.sample))
    train, val, test = split_parts(frame)
    lexical = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=20260708)),
        ]
    ).fit(train["url"], y(train))
    behavior = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=20260708)),
        ]
    ).fit(train[DNS_NUMERIC], y(train))

    p_lex_val = lexical.predict_proba(val["url"])[:, 1]
    p_dns_val = behavior.predict_proba(val[DNS_NUMERIC])[:, 1]
    val_behavior, val_missing = behavior_matrix(val, train)
    val_inputs = (
        torch.tensor(clipped_logit(p_lex_val), dtype=torch.float32),
        torch.tensor(clipped_logit(p_dns_val), dtype=torch.float32),
        torch.from_numpy(val_behavior),
        torch.from_numpy(val_missing),
    )
    gate_target = torch.tensor(oracle_gate_target(y(val), p_lex_val, p_dns_val), dtype=torch.float32)
    y_val = torch.tensor(y(val), dtype=torch.float32)

    # --- build prospective frame ---
    cohort = pd.read_csv(args.cohort)
    dns = pd.read_csv(args.dns)
    pros = add_features(build_prospective_features(cohort, dns))
    pros_behavior, pros_missing = behavior_matrix(pros, train)
    pros_texts = pros["url"].tolist()
    p_lex_pros = lexical.predict_proba(pros_texts)[:, 1]
    p_dns_pros = behavior.predict_proba(pros[DNS_NUMERIC].fillna(0))[:, 1]
    y_pros = y(pros)

    # --- behavior-view diagnostics (computed, not asserted) ---
    all_ips: list[str] = []
    for col in ("a_records", "aaaa_records"):
        for value in dns[col]:
            all_ips.extend(parse_ip_list(value))
    ttl_vals = pros["TTL"].dropna().unique()
    ip_counts = pros["ip_unique_count"]
    behavior_diag = {
        "ttl_unique_values": sorted(float(v) for v in ttl_vals),
        "ttl_constant": int(len(ttl_vals)) <= 1,
        "ttl_min_seconds": float(pros["TTL"].min()),
        "ttl_max_seconds": float(pros["TTL"].max()),
        "max_unique_ips_per_domain": int(ip_counts.max()),
        "mean_unique_ips_per_domain": round(float(ip_counts.mean()), 6),
        "domains_with_multiple_unique_ips": int((ip_counts > 1).sum()),
        **reserved_network_summary(all_ips),
    }

    pros_inputs = (
        torch.tensor(clipped_logit(p_lex_pros), dtype=torch.float32),
        torch.tensor(clipped_logit(p_dns_pros), dtype=torch.float32),
        torch.from_numpy(pros_behavior),
        torch.from_numpy(pros_missing),
    )

    # --- per-seed gate evaluation ---
    rows = []
    gate_probs = []
    gate_weights = []
    for seed in seeds:
        seed_everything(seed)
        gate = train_gate(GateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val, seed, device, residual=False)
        prob, gate_w = evaluate_minimal_gate(gate, pros_inputs, p_lex_pros, p_dns_pros, device)
        gate_probs.append(prob)
        gate_weights.append(gate_w)
        for system, score in [
            ("lexical_only", p_lex_pros),
            ("dns_behavior_only", p_dns_pros),
            ("fixed_average", (p_lex_pros + p_dns_pros) / 2.0),
            ("cross_attentive_gate", prob),
        ]:
            rows.append(
                {
                    "seed": seed,
                    "system": system,
                    "AUPRC": float(average_precision_score(y_pros, score)),
                    "FPR@95TPR": float(fpr_at_tpr(y_pros, score)),
                    "macro_F1": float(_macro_f1(y_pros, score)),
                    "ECE": float(ece_score(y_pros, score)),
                }
            )

    gate_diag = [
        {
            "seed": int(seed),
            "mean_g_lex": round(float(gw.mean()), 6),
            "choose_lex_pct": round(float((gw > 0.5).mean()) * 100.0, 4),
        }
        for seed, gw in zip(seeds, gate_weights)
    ]
    mean_g_lex = float(np.mean([d["mean_g_lex"] for d in gate_diag]))

    gate_mean = np.mean(np.stack(gate_probs), axis=0)
    aggregate = []
    for system, score in [
        ("lexical_only", p_lex_pros),
        ("dns_behavior_only", p_dns_pros),
        ("fixed_average", (p_lex_pros + p_dns_pros) / 2.0),
        ("cross_attentive_gate_mean", gate_mean),
    ]:
        aggregate.append(
            {
                "system": system,
                "AUPRC": float(average_precision_score(y_pros, score)),
                "FPR@95TPR": float(fpr_at_tpr(y_pros, score)),
                "macro_F1": float(_macro_f1(y_pros, score)),
                "ECE": float(ece_score(y_pros, score)),
            }
        )

    # --- paired bootstrap: gate vs lexical on AUPRC ---
    rng = np.random.default_rng(20260819)
    idx = np.arange(len(y_pros))
    deltas = []
    for _ in range(args.bootstrap_iters):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_pros[s])) < 2:
            continue
        deltas.append(average_precision_score(y_pros[s], gate_mean[s]) - average_precision_score(y_pros[s], p_lex_pros[s]))
    bootstrap = {
        "gate_minus_lexical_mean_delta_AUPRC": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "iterations": len(deltas),
    }

    # --- emit artifacts ---
    stamp = utc_stamp()
    df = pd.DataFrame(rows)
    agg = pd.DataFrame(aggregate)
    metrics_path = args.out_dir / f"PROSPECTIVE_GATE_METRICS_{stamp}.csv"
    report_path = args.out_dir / f"PROSPECTIVE_GATE_EVALUATION_REPORT_{stamp}.md"
    metadata_path = args.out_dir / f"PROSPECTIVE_GATE_METADATA_{stamp}.json"
    latest = {
        "PROSPECTIVE_GATE_METRICS": args.out_dir / "PROSPECTIVE_GATE_METRICS.csv",
        "PROSPECTIVE_GATE_REPORT": args.out_dir / "PROSPECTIVE_GATE_EVALUATION_REPORT.md",
        "PROSPECTIVE_GATE_METADATA": args.out_dir / "PROSPECTIVE_GATE_METADATA.json",
    }
    df.to_csv(metrics_path, index=False)
    shutil.copyfile(metrics_path, latest["PROSPECTIVE_GATE_METRICS"])

    md = [
        "# Prospective Point-in-Time Gate Evaluation",
        "",
        f"**Generated**: {stamp}",
        f"**Prospective sample**: 200 domains (100 malicious / 100 benign), DNS features within the amended 25h window (R015B pass).",
        f"**Experts and gate**: trained only on DeepURLBench; frozen at evaluation time.",
        f"**Seeds**: {', '.join(str(s) for s in seeds)}",
        "",
        "## Aggregate Metrics (gate = mean over seeds)",
        "",
        _md_table(agg),
        "",
        "## Per-Seed Metrics",
        "",
        _md_table(df),
        "",
        "## Paired Bootstrap (gate mean vs lexical, AUPRC)",
        "",
        _md_table(pd.DataFrame([bootstrap])),
        "",
        "## Behavior-View Diagnostics",
        "",
        _md_table(pd.DataFrame([behavior_diag])),
        "",
        "## Gate Trust Diagnostics (prospective rows)",
        "",
        _md_table(pd.DataFrame(gate_diag)),
        "",
        "## Interpretation",
        "",
        "- This is a **small, noisy** 200-domain transfer check with an approximate feature mapping from local DNS records to DeepURLBench-style fields.",
        "- The DNS features used are dataset-provided local observations collected within the amended 25h point-in-time window; they are **not** DeepURLBench fields.",
        f"- The collected behavior view is near-degenerate: TTL has {len(behavior_diag['ttl_unique_values'])} unique value(s) (constant at {behavior_diag['ttl_min_seconds']:.0f} s), {behavior_diag['benchmark_198_18_0_0_15_fraction']:.0%} of resolved IPv4 addresses fall in the reserved 198.18.0.0/15 benchmark range, and every domain has at most {behavior_diag['max_unique_ips_per_domain']} unique IP(s). The frozen DNS expert is therefore at chance (AUPRC 0.50).",
        f"- Under this evidence the gate falls back to the lexical expert (mean lexical trust g = {mean_g_lex:.4f} > 0.999; gate AUPRC equals lexical-only), which is consistent with the missing-modality fallback behavior observed in R016/R017.",
        "- This confirms graceful fallback on verified-timing data but does **not** demonstrate behavior-conditioned gains on point-in-time evidence; richer prospective DNS/IP collection is required for that claim.",
        "- All numbers here are diagnostic and must not be presented as deployment performance.",
        "",
    ]
    (report_path).write_text("\n".join(md), encoding="utf-8")
    shutil.copyfile(report_path, latest["PROSPECTIVE_GATE_REPORT"])

    metadata = {
        "run": "PROSPECTIVE_GATE",
        "generated_at": stamp,
        "inputs_sha256": inputs_sha,
        "seeds": seeds,
        "bootstrap_iters": args.bootstrap_iters,
        "prospective_rows": int(len(pros)),
        "prospective_positives": int(y_pros.sum()),
        "behavior_view_diagnostics": behavior_diag,
        "gate_diagnostics": gate_diag,
        "mean_g_lex_overall": round(mean_g_lex, 6),
        "aggregate": aggregate,
        "per_seed": rows,
        "bootstrap": bootstrap,
        "scope": "diagnostic cross-sample transfer check; small sample, approximate feature mapping; not deployment evidence",
    }
    (metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(metadata_path, latest["PROSPECTIVE_GATE_METADATA"])

    print(agg.to_string(index=False))
    print()
    print(pd.DataFrame([bootstrap]).to_string(index=False))
    print("wrote", metrics_path, report_path, metadata_path)
    return 0


def _macro_f1(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tn = np.sum((pred == 0) & (y_true == 0))
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if precision_neg + recall_neg else 0.0
    return float((f1 + f1_neg) / 2.0)


def _md_table(frame: pd.DataFrame) -> str:
    def fmt(value) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
