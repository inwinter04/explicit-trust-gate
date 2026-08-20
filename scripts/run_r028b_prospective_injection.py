#!/usr/bin/env python
"""R028b: verified-timing prospective sample with controlled informative
behavior injection.

The 2026-07-07 prospective sample (200 domains, 100 malicious / 100 benign)
passes the amended 25h point-in-time window audit (R015B) but its collected
behavior view is degenerate (TTL constant at 1s, all IPs in 198.18.0.0/15,
single unique IP per domain), so the DNS expert is at chance and the gate
falls back to lexical. This experiment evaluates a controlled counterfactual
on the SAME verified-timing sample: we inject an explicitly disclosed,
label-consistent informative signal into (a) TTL or (b) IP-diversity features,
keep experts and gate frozen (trained only on DeepURLBench), and check whether
the gate routes toward behavior (mean g < 0.5) and improves AUPRC over the
lexical-only expert.

Scope: controlled counterfactual demonstrating mechanism capacity on
verified-timing data. It is NOT a deployment result and does not claim that
the injected distributions occur in the wild.
"""

from __future__ import annotations

import argparse
import json
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

from run_prospective_gate_evaluation import build_prospective_features, parse_ip_list, reserved_network_summary
from run_r020_residual_correction import (
    DNS_NUMERIC,
    GateNet,
    add_features,
    behavior_matrix,
    clipped_logit,
    ece_score,
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


def macro_f1(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, (prob > threshold).astype(int), average="macro", zero_division=0))


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    return "```\n" + df.to_string(index=index) + "\n```"


def inject_ttl(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """TTL rule aligned with the DeepURLBench label-TTL relationship the frozen
    expert learned (malicious median TTL ~3600s, benign median ~600s): injected
    malicious rows get high TTLs, benign rows get low TTLs. Disclosed in
    metadata; the direction is set so the condition tests the GATE's routing
    response to informative behavior, not the expert's mapping."""
    out = frame.copy()
    ttl = []
    for label in out["label"]:
        if label == "malicious":
            ttl.append(float(np.exp(rng.uniform(np.log(3600.0), np.log(21601.0)))))
        else:
            ttl.append(float(np.exp(rng.uniform(np.log(30.0), np.log(601.0)))))
    out["TTL"] = ttl
    out["has_dns"] = "true"
    return out


def inject_ip_diversity(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    octets_pool = [8, 34, 52, 91, 104, 125, 146, 193]
    rows = []
    for _, row in out.iterrows():
        row = row.copy()
        if row["label"] == "benign":
            n = int(rng.integers(2, 6))
            octets = sorted(rng.choice(octets_pool, size=n, replace=False).tolist())
            row["ip_count"] = n
            row["ip_unique_count"] = n
            row["ip_prefix24_count"] = n
            row["ip_prefix16_count"] = max(1, n - 1)
            row["ip_first_octet_min"] = float(min(octets))
            row["ip_first_octet_max"] = float(max(octets))
            row["ip_first_octet_mean"] = float(np.mean(octets))
            row["ip_private_special_count"] = 0
        else:
            row["ip_count"] = 1
            row["ip_unique_count"] = 1
            row["ip_prefix24_count"] = 1
            row["ip_prefix16_count"] = 1
            row["ip_first_octet_min"] = 91.0
            row["ip_first_octet_max"] = 91.0
            row["ip_first_octet_mean"] = 91.0
            row["ip_private_special_count"] = 0
        row["has_dns"] = "true"
        rows.append(row)
    return pd.DataFrame(rows)


def condition_diagnostics(frame: pd.DataFrame) -> dict:
    ttl_vals = frame["TTL"].dropna().unique()
    return {
        "rows": int(len(frame)),
        "positives": int((frame["label"] == "malicious").sum()),
        "ttl_unique_values": sorted(float(v) for v in ttl_vals)[:8],
        "ttl_constant": int(len(ttl_vals)) <= 1,
        "ttl_min_seconds": float(frame["TTL"].min()),
        "ttl_max_seconds": float(frame["TTL"].max()),
        "max_unique_ips_per_domain": int(frame["ip_unique_count"].max()),
        "mean_unique_ips_per_domain": round(float(frame["ip_unique_count"].mean()), 6),
        "domains_with_multiple_unique_ips": int((frame["ip_unique_count"] > 1).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--cohort", type=Path, default=Path("data/raw/2026-07-07/domain_cohort.csv"))
    parser.add_argument("--dns", type=Path, default=Path("data/raw/2026-07-07/dns_observations.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--seeds", default="20260819,20260820,20260821")
    parser.add_argument("--injection-seed", type=int, default=20260820)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.injection_seed)
    inputs_sha = {
        "deepurlbench_sample": sha256_file(args.sample),
        "prospective_cohort": sha256_file(args.cohort),
        "prospective_dns": sha256_file(args.dns),
        "script": sha256_file(Path(__file__).resolve()),
    }

    # --- frozen experts + gate, trained on DeepURLBench only (same as R011/prospective) ---
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

    # --- prospective raw frame + injected conditions ---
    cohort = pd.read_csv(args.cohort)
    dns = pd.read_csv(args.dns)
    pros_raw = build_prospective_features(cohort, dns)
    conditions = {
        "original_degenerate": pros_raw,
        "injected_ttl_informative": inject_ttl(pros_raw, rng),
        "injected_ip_diversity_informative": inject_ip_diversity(pros_raw, rng),
    }
    condition_frames = {name: add_features(raw) for name, raw in conditions.items()}

    prepared = {}
    for name, cf in condition_frames.items():
        beh, miss = behavior_matrix(cf, train)
        prepared[name] = {
            "frame": cf,
            "p_lex": lexical.predict_proba(cf["url"])[:, 1],
            "p_dns": behavior.predict_proba(cf[DNS_NUMERIC].fillna(0))[:, 1],
            "y": y(cf),
            "behavior": beh,
            "missing": miss,
        }

    # --- per-seed gate evaluation per condition ---
    rows = []
    for seed in seeds:
        seed_everything(seed)
        gate = train_gate(GateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val, seed, device, residual=False)
        for name, prep in prepared.items():
            inputs = (
                torch.tensor(clipped_logit(prep["p_lex"]), dtype=torch.float32),
                torch.tensor(clipped_logit(prep["p_dns"]), dtype=torch.float32),
                torch.from_numpy(prep["behavior"]),
                torch.from_numpy(prep["missing"]),
            )
            prob, gate_w = evaluate_minimal_gate(gate, inputs, prep["p_lex"], prep["p_dns"], device)
            for system, score in [
                ("lexical_only", prep["p_lex"]),
                ("dns_behavior_only", prep["p_dns"]),
                ("fixed_average", (prep["p_lex"] + prep["p_dns"]) / 2.0),
                ("cross_attentive_gate", prob),
            ]:
                rows.append(
                    {
                        "seed": seed,
                        "condition": name,
                        "system": system,
                        "AUPRC": float(average_precision_score(prep["y"], score)),
                        "FPR@95TPR": float(fpr_at_tpr(prep["y"], score)),
                        "macro_F1": macro_f1(prep["y"], score),
                        "ECE": float(ece_score(prep["y"], score)),
                    }
                )
            rows.append(
                {
                    "seed": seed,
                    "condition": name,
                    "system": "gate_diagnostic",
                    "AUPRC": float("nan"),
                    "FPR@95TPR": float("nan"),
                    "macro_F1": float("nan"),
                    "ECE": float("nan"),
                    "mean_g_lex": float(gate_w.mean()),
                    "choose_lex_pct": float((gate_w > 0.5).mean()) * 100.0,
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / f"R028B_PROSPECTIVE_INJECTION_METRICS_{utc_stamp()}.csv", index=False)
    metrics.to_csv(args.out_dir / "R028B_PROSPECTIVE_INJECTION_METRICS.csv", index=False)

    diag = {name: condition_diagnostics(cf) for name, cf in condition_frames.items()}
    agg = (
        metrics.dropna(subset=["AUPRC"])
        .groupby(["condition", "system"])["AUPRC"]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    md = [
        f"# R028b Verified-Timing Controlled Injection — {utc_stamp()}",
        "",
        "Frozen DeepURLBench experts and gate (seeds "
        f"{', '.join(str(s) for s in seeds)}) evaluated on the 2026-07-07 "
        "prospective sample (200 domains, 100 malicious / 100 benign; amended "
        "25h window, R015B pass) under three behavior-view conditions:",
        "",
        "1. original_degenerate: collected features (TTL=1s, single IP, "
        "198.18.0.0/15) — known fallback condition.",
        "2. injected_ttl_informative: TTL replaced by a disclosed rule aligned "
        "with the DeepURLBench label-TTL relationship the frozen expert learned "
        "(malicious: 3600-21600s; benign: 30-600s, log-uniform noise); IP "
        "features unchanged.",
        "3. injected_ip_diversity_informative: benign domains get 2-5 unique IPs "
        "with spread first octets; malicious domains get a single IP (91.x); TTL "
        "unchanged.",
        "",
        "Injection touches only feature values, never expert/gate parameters. "
        "Scope: controlled counterfactual on verified-timing data, not a "
        "deployment result.",
        "",
        "## AUPRC by condition and system (mean over seeds)",
        "",
        md_table(agg),
        "",
        "## Gate diagnostics (mean g, choose-lex %)",
        "",
        md_table(metrics[metrics["system"] == "gate_diagnostic"][["seed", "condition", "mean_g_lex", "choose_lex_pct"]].round(4)),
        "",
        "## Condition feature diagnostics",
        "",
        md_table(pd.DataFrame(diag).T.round(4)),
        "",
        "## Success criteria (pre-registered)",
        "- injected conditions: dns_behavior_only AUPRC > 0.7 (signal informative),",
        "- cross_attentive_gate AUPRC > lexical_only AUPRC,",
        "- mean g < 0.5 (gate routes to behavior);",
        "- original_degenerate: gate AUPRC == lexical_only and mean g > 0.999 (fallback).",
    ]
    report = "\n".join(md)
    (args.out_dir / f"R028B_PROSPECTIVE_INJECTION_REPORT_{utc_stamp()}.md").write_text(report, encoding="utf-8")
    (args.out_dir / "R028B_PROSPECTIVE_INJECTION_REPORT.md").write_text(report, encoding="utf-8")

    metadata = {
        "run": "R028B_PROSPECTIVE_INJECTION",
        "generated_at": utc_stamp(),
        "inputs_sha256": inputs_sha,
        "seeds": seeds,
        "injection_seed": args.injection_seed,
        "injection_rules": {
            "injected_ttl_informative": "aligned with DeepURLBench label-TTL relationship (malicious median ~3600s vs benign ~600s): malicious TTL ~ exp(U(ln3600, ln21601)); benign TTL ~ exp(U(ln30, ln601)); ip features unchanged",
            "injected_ip_diversity_informative": "benign: 2-5 unique IPs, first octets from {8,34,52,91,104,125,146,193}; malicious: single IP 91.x; TTL unchanged",
        },
        "conditions_diagnostics": diag,
        "device": str(device),
        "scope": "controlled counterfactual on verified-timing sample; diagnostic "
                 "mechanism-capacity evidence, not a deployment or in-the-wild claim",
    }
    (args.out_dir / f"R028B_PROSPECTIVE_INJECTION_METADATA_{utc_stamp()}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "R028B_PROSPECTIVE_INJECTION_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
