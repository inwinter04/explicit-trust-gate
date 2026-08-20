#!/usr/bin/env python
"""R011 latency budget for the explicit gate and its baselines/variants.

Measures end-to-end inference latency (per-sample and throughput), trainable
parameter counts, and peak GPU memory for:

- lexical-only (TF-IDF char n-gram + logistic)
- DNS/IP behavior-only (logistic)
- fixed average of expert probabilities (non-trainable control)
- cross-attentive trust gate (minimal)
- residual correction gate
- standalone direct conflict classifier

Latency is architecture-driven, so the timing suite runs on seed 20260819
models, while all three R024 seeds are trained for a metric sanity check.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_r020_residual_correction import (
    DNS_NUMERIC,
    GateNet,
    ResidualGateNet,
    add_features,
    behavior_matrix,
    clipped_logit,
    count_parameters,
    fit_experts,
    oracle_gate_target,
    seed_everything,
    sha256_file,
    split_parts,
    train_gate,
    y,
)
from run_r021_conflict_classifier import (
    DirectClassifierNet,
    evaluate_direct_classifier,
    evaluate_minimal_gate,
    train_direct_classifier,
)


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def fit_lexical_expert(train: pd.DataFrame):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=20260708)),
        ]
    ).fit(train["url"], y(train))


def fit_behavior_expert(train: pd.DataFrame):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=20260708)),
        ]
    ).fit(train[DNS_NUMERIC], y(train))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--seeds", default="20260819,20260820,20260821")
    parser.add_argument("--timing-seed", type=int, default=20260819)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


class BatchCycler:
    """Yields consecutive slices of [0, n), cycling after the last row."""

    def __init__(self, n: int, batch: int) -> None:
        self.n = n
        self.batch = batch
        self.pos = 0

    def next(self) -> slice:
        start = self.pos
        end = min(start + self.batch, self.n)
        self.pos = end if end < self.n else 0
        return slice(start, end)


def timed_median(fn, cycler: BatchCycler, n_runs: int, warmup: int, device: torch.device) -> float:
    """Median seconds per call after warmup; CUDA-synchronized when applicable."""
    for _ in range(warmup):
        sl = cycler.next()
        fn(sl)
    timings = []
    for _ in range(n_runs):
        sl = cycler.next()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(sl)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
    return float(np.median(timings))


def sklearn_params(clf) -> dict[str, int]:
    coef = getattr(clf, "coef_", None)
    inter = getattr(clf, "intercept_", None)
    trainable = int(coef.size + (inter.size if inter is not None else 0))
    return {"trainable_parameters": trainable}


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
    sample_sha = sha256_file(args.sample)
    script_sha = sha256_file(Path(__file__).resolve())

    frame = add_features(pd.read_csv(args.sample))
    train, val, test = split_parts(frame)
    lex_pipeline = fit_lexical_expert(train)
    dns_pipeline = fit_behavior_expert(train)
    tfidf = lex_pipeline.named_steps["tfidf"]
    p_lex_val = lex_pipeline.predict_proba(val["url"])[:, 1]
    p_dns_val = dns_pipeline.predict_proba(val[DNS_NUMERIC])[:, 1]
    p_lex_test = lex_pipeline.predict_proba(test["url"])[:, 1]
    p_dns_test = dns_pipeline.predict_proba(test[DNS_NUMERIC])[:, 1]
    y_val_np = y(val)
    y_test = y(test)

    val_behavior, val_missing = behavior_matrix(val, train)
    test_behavior, test_missing = behavior_matrix(test, train)
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
    y_val_t = torch.tensor(y_val_np, dtype=torch.float32)

    # --- train all seeds for metric sanity ---
    sanity_rows = []
    trained: dict[str, torch.nn.Module] = {}
    for seed in seeds:
        seed_everything(seed)
        minimal = train_gate(GateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val_t, seed, device, residual=False)
        residual = train_gate(ResidualGateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val_t, seed, device, residual=True)
        direct = train_direct_classifier(DirectClassifierNet(len(DNS_NUMERIC)), val_inputs, y_val_t, seed, device)

        minimal_prob, _ = evaluate_minimal_gate(minimal, test_inputs, p_lex_test, p_dns_test, device)
        residual_prob, _, _ = _evaluate_residual(residual, test_inputs, device)
        direct_prob = evaluate_direct_classifier(direct, test_inputs, device)
        sanity_rows.append(
            {
                "seed": int(seed),
                "cross_attentive_gate_AUPRC": float(average_precision_score(y_test, minimal_prob)),
                "residual_gate_AUPRC": float(average_precision_score(y_test, residual_prob)),
                "direct_classifier_AUPRC": float(average_precision_score(y_test, direct_prob)),
            }
        )
        if seed == args.timing_seed:
            trained = {"cross_attentive_gate": minimal, "residual_gate": residual, "direct_classifier": direct}
    sanity = pd.DataFrame(sanity_rows)

    # --- latency benchmark (seed = args.timing_seed) ---
    n_test = len(test)
    texts = test["url"].astype(str).tolist()
    dns_frame = test[DNS_NUMERIC].copy()
    beh_full = test_behavior
    miss_full = test_missing

    def make_fn(system: str, dev: torch.device):
        source = trained.get(system)
        if source is None and system == "gate_forward_only":
            source = trained.get("cross_attentive_gate")
        model = copy.deepcopy(source).to(dev) if source is not None else None
        if model is not None:
            model.eval()

        def lexical(sl: slice) -> None:
            lex_pipeline.predict_proba(texts[sl.start : sl.stop])[:, 1]

        def dns_only(sl: slice) -> None:
            dns_pipeline.predict_proba(dns_frame.iloc[sl])[:, 1]

        def fixed_average(sl: slice) -> None:
            p_lex = lex_pipeline.predict_proba(texts[sl.start : sl.stop])[:, 1]
            p_dns = dns_pipeline.predict_proba(dns_frame.iloc[sl])[:, 1]
            (p_lex + p_dns) / 2.0

        def neural(sl: slice) -> None:
            p_lex = lex_pipeline.predict_proba(texts[sl.start : sl.stop])[:, 1]
            p_dns = dns_pipeline.predict_proba(dns_frame.iloc[sl])[:, 1]
            lex_logit = torch.tensor(clipped_logit(p_lex), dtype=torch.float32)
            dns_logit = torch.tensor(clipped_logit(p_dns), dtype=torch.float32)
            beh = torch.from_numpy(beh_full[sl])
            miss = torch.from_numpy(miss_full[sl])
            inputs = tuple(item.to(dev) for item in (lex_logit, dns_logit, beh, miss))
            with torch.no_grad():
                if system == "cross_attentive_gate":
                    gate, _ = model(*inputs)
                    torch.sigmoid(gate * inputs[0] + (1 - gate) * inputs[1])
                elif system == "residual_gate":
                    gate, delta, _ = model(*inputs)
                    torch.sigmoid(gate * inputs[0] + (1 - gate) * inputs[1] + delta)
                else:  # direct_classifier
                    model(*inputs)

        def gate_forward_only(sl: slice) -> None:
            inputs = tuple(item.to(dev) for item in test_inputs_slices(sl))
            with torch.no_grad():
                gate, _ = model(*inputs)
                torch.sigmoid(gate * inputs[0] + (1 - gate) * inputs[1])

        if system == "lexical_only":
            return lexical
        if system == "dns_behavior_only":
            return dns_only
        if system == "fixed_average":
            return fixed_average
        if system == "gate_forward_only":
            return gate_forward_only
        return neural

    def test_inputs_slices(sl: slice):
        return (
            torch.tensor(clipped_logit(p_lex_test[sl]), dtype=torch.float32),
            torch.tensor(clipped_logit(p_dns_test[sl]), dtype=torch.float32),
            torch.from_numpy(test_behavior[sl]),
            torch.from_numpy(test_missing[sl]),
        )

    batch_sizes = [1, 16, 64, 256, n_test]
    runs = {1: 300, 16: 200, 64: 100, 256: 60, n_test: 20}
    systems = [
        "lexical_only",
        "dns_behavior_only",
        "fixed_average",
        "cross_attentive_gate",
        "residual_gate",
        "direct_classifier",
        "gate_forward_only",
    ]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    latency_rows = []
    param_rows = {}
    for system in systems:
        if system in {"lexical_only", "dns_behavior_only"}:
            pipeline = lex_pipeline if system == "lexical_only" else dns_pipeline
            params = sklearn_params(pipeline.named_steps["clf"])
            if system == "lexical_only":
                params["tfidf_vocab_size"] = len(tfidf.vocabulary_)
        elif system == "fixed_average":
            params = {"trainable_parameters": 0}
        else:
            params = {
                "trainable_parameters": count_parameters(
                    trained.get(system) or trained.get("cross_attentive_gate")
                )
            }
        param_rows[system] = params

        for dev in devices:
            fn = make_fn(system, dev)
            for batch in batch_sizes:
                cycler = BatchCycler(n_test, batch)
                seconds = timed_median(fn, cycler, runs[batch], warmup=10, device=dev)
                latency_rows.append(
                    {
                        "system": system,
                        "device": dev.type,
                        "batch_size": batch,
                        "median_seconds_per_batch": seconds,
                        "ms_per_sample": round(seconds / batch * 1000.0, 4),
                        "rows_per_second": round(batch / seconds, 1),
                        "trainable_parameters": params["trainable_parameters"],
                    }
                )

    # --- GPU peak memory (CUDA only) ---
    gpu_rows = []
    if torch.cuda.is_available():
        for system in ["cross_attentive_gate", "residual_gate", "direct_classifier"]:
            torch.cuda.reset_peak_memory_stats()
            fn = make_fn(system, torch.device("cuda"))
            fn(slice(0, n_test))
            gpu_rows.append(
                {
                    "system": system,
                    "device": "cuda",
                    "peak_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3),
                }
            )

    latency_df = pd.DataFrame(latency_rows)
    gpu_df = pd.DataFrame(gpu_rows)

    # --- emit artifacts ---
    stamp = utc_stamp()
    metrics_path = args.out_dir / f"R011_LATENCY_METRICS_{stamp}.csv"
    report_path = args.out_dir / f"R011_LATENCY_REPORT_{stamp}.md"
    metadata_path = args.out_dir / f"R011_LATENCY_METADATA_{stamp}.json"
    latest = {
        "R011_LATENCY_METRICS": args.out_dir / "R011_LATENCY_METRICS.csv",
        "R011_LATENCY_REPORT": args.out_dir / "R011_LATENCY_REPORT.md",
        "R011_LATENCY_METADATA": args.out_dir / "R011_LATENCY_METADATA.json",
    }
    latency_df.to_csv(metrics_path, index=False)
    shutil.copyfile(metrics_path, latest["R011_LATENCY_METRICS"])

    per_sample = latency_df[latency_df["batch_size"] == 1].pivot_table(
        index="system", columns="device", values="ms_per_sample"
    )
    throughput_full = latency_df[latency_df["batch_size"] == n_test].pivot_table(
        index="system", columns="device", values="rows_per_second"
    )
    md = [
        "# R011 Latency Budget Report",
        "",
        f"**Generated**: {stamp}",
        f"**Sample**: `{args.sample}` ({n_test} test rows)",
        f"**Timing seed**: {args.timing_seed} (latency is architecture-driven; all seeds trained for metric sanity)",
        f"**Device(s)**: {', '.join(d.type for d in devices)}",
        "",
        "## Metric Sanity vs R024 (full-test AUPRC)",
        "",
        _md_table(sanity),
        "",
        "R024 aggregate reference: gate 0.8797, residual 0.8884, direct 0.8979.",
        "",
        "## Trainable Parameters",
        "",
        _md_table(pd.DataFrame(param_rows).T.reset_index().rename(columns={"index": "system"})),
        "",
        "## Per-Sample Latency (batch size 1, median ms/sample)",
        "",
        _md_table(per_sample.reset_index().rename(columns={"index": "system"})),
        "",
        "## Throughput at Full Test Batch (rows/second)",
        "",
        _md_table(throughput_full.reset_index().rename(columns={"index": "system"})),
        "",
    ]
    if not gpu_df.empty:
        md += ["## Peak GPU Memory (CUDA, full-test forward, MB)", "", _md_table(gpu_df), ""]
    md += [
        "## Protocol",
        "",
        "- Timing uses median wall-clock over repeated batches after 10 warmup calls; CUDA runs are synchronized.",
        "- Batch cycler walks consecutive test rows to avoid single-row cache bias.",
        "- End-to-end systems include TF-IDF + logistic expert scoring, behavior standardization, and the neural forward; `gate_forward_only` isolates the neural gate on precomputed inputs.",
        "- Parameter counts are trainable parameters; TF-IDF vocabulary size is reported separately as storage.",
        "",
    ]
    (report_path).write_text("\n".join(md), encoding="utf-8")
    shutil.copyfile(report_path, latest["R011_LATENCY_REPORT"])

    metadata = {
        "run": "R011",
        "generated_at": stamp,
        "sample": str(args.sample),
        "sample_sha256": sample_sha,
        "script_sha256": script_sha,
        "seeds": seeds,
        "timing_seed": args.timing_seed,
        "devices": [d.type for d in devices],
        "batch_sizes": batch_sizes,
        "metric_sanity": sanity_rows,
        "latency_rows": latency_rows,
        "gpu_rows": gpu_rows,
        "parameters": param_rows,
        "timing_protocol": "median wall-clock after warmup, CUDA-synchronized, consecutive-batch cycler",
        "scope": "frozen-expert DeepURLBench pilot; latency is architecture-driven and does not change claim framing",
    }
    (metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(metadata_path, latest["R011_LATENCY_METADATA"])

    print(latency_df.to_string(index=False))
    print()
    print(sanity.to_string(index=False))
    print("wrote", metrics_path, report_path, metadata_path)
    return 0


def _evaluate_residual(model, inputs, device):
    model.eval()
    with torch.no_grad():
        gate, delta, _ = model(*(item.to(device) for item in inputs))
        prob = torch.sigmoid(gate * inputs[0].to(device) + (1 - gate) * inputs[1].to(device) + delta).cpu().numpy()
    return prob, gate.cpu().numpy(), delta.cpu().numpy()


def _md_table(frame: pd.DataFrame) -> str:
    def fmt(value) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
