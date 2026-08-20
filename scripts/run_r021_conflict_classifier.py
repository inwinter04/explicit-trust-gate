#!/usr/bin/env python
"""Run R021 simplicity check: explicit trust gate vs standalone classifier head.

This compares the minimal cross-attentive mixture gate against an overbuilt
direct classifier that consumes the same cross-attended lexical/behavior
representation but does not preserve the explicit expert-mixture decision path.
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
from torch import nn

from run_r020_residual_correction import (
    DNS_NUMERIC,
    BaseGate,
    GateNet,
    add_features,
    behavior_matrix,
    clipped_logit,
    count_parameters,
    domain_overlap_rows,
    fit_experts,
    high_conflict_mask,
    md_table,
    metric_row,
    oracle_gate_target,
    seed_everything,
    sha256_file,
    split_parts,
    utc_stamp,
    y,
)


class DirectClassifierNet(BaseGate):
    def __init__(self, feature_count: int, d_model: int = 32, heads: int = 4) -> None:
        super().__init__(feature_count, d_model=d_model, heads=heads)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, lex_logit: torch.Tensor, dns_logit: torch.Tensor, behavior: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        features, _, _ = self.encode(lex_logit, dns_logit, behavior, missing)
        return torch.sigmoid(self.classifier_head(features).squeeze(1))


def train_minimal_gate(
    model: GateNet,
    train_inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    y_true: torch.Tensor,
    seed: int,
    device: torch.device,
) -> GateNet:
    seed_everything(seed)
    model.to(device)
    inputs = tuple(item.to(device) for item in train_inputs)
    target = target.to(device)
    y_true = y_true.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best_state = None
    best_loss = float("inf")
    patience = 0
    for _ in range(350):
        model.train()
        gate, _ = model(*inputs)
        mixture_logit = gate * inputs[0] + (1 - gate) * inputs[1]
        loss = nn.functional.binary_cross_entropy(torch.sigmoid(mixture_logit).clamp(1e-6, 1 - 1e-6), y_true)
        loss = loss + 0.25 * nn.functional.mse_loss(gate, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss - 1e-5:
            best_loss = current
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 35:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_direct_classifier(
    model: DirectClassifierNet,
    train_inputs: tuple[torch.Tensor, ...],
    y_true: torch.Tensor,
    seed: int,
    device: torch.device,
) -> DirectClassifierNet:
    seed_everything(seed)
    model.to(device)
    inputs = tuple(item.to(device) for item in train_inputs)
    y_true = y_true.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best_state = None
    best_loss = float("inf")
    patience = 0
    for _ in range(350):
        model.train()
        prob = model(*inputs)
        loss = nn.functional.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss - 1e-5:
            best_loss = current
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 35:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_minimal_gate(
    model: GateNet,
    inputs: tuple[torch.Tensor, ...],
    p_lex: np.ndarray,
    p_dns: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        gate, _ = model(*(item.to(device) for item in inputs))
    gate_np = gate.cpu().numpy()
    prob = torch.sigmoid(gate * inputs[0].to(device) + (1 - gate) * inputs[1].to(device)).cpu().numpy()
    return prob, gate_np


def evaluate_direct_classifier(model: DirectClassifierNet, inputs: tuple[torch.Tensor, ...], device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(*(item.to(device) for item in inputs)).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = add_features(pd.read_csv(args.sample))
    train, val, test = split_parts(frame)
    experts = fit_experts(train, val, test)
    val_behavior, val_missing = behavior_matrix(val, train)
    test_behavior, test_missing = behavior_matrix(test, train)

    p_lex_val, p_dns_val = experts["p_lex_val"], experts["p_dns_val"]
    p_lex_test, p_dns_test = experts["p_lex_test"], experts["p_dns_test"]
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
    y_val_np = y(val)
    y_test = y(test)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    gate_target = torch.tensor(oracle_gate_target(y_val_np, p_lex_val, p_dns_val), dtype=torch.float32)

    minimal = train_minimal_gate(GateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val, args.seed, device)
    direct = train_direct_classifier(DirectClassifierNet(len(DNS_NUMERIC)), val_inputs, y_val, args.seed, device)

    minimal_prob, minimal_gate = evaluate_minimal_gate(minimal, test_inputs, p_lex_test, p_dns_test, device)
    direct_prob = evaluate_direct_classifier(direct, test_inputs, device)
    hc = high_conflict_mask(p_lex_test, p_dns_test)

    rows = [
        metric_row("cross_attentive_gate", "test", y_test, minimal_prob),
        metric_row("standalone_conflict_classifier", "test", y_test, direct_prob),
    ]
    if hc.any():
        rows.extend(
            [
                metric_row("cross_attentive_gate", "test_high_conflict", y_test[hc], minimal_prob[hc]),
                metric_row("standalone_conflict_classifier", "test_high_conflict", y_test[hc], direct_prob[hc]),
            ]
        )
    metrics = pd.DataFrame(rows)
    predictions = test[["split", "url", "domain", "first_seen", "label", *DNS_NUMERIC]].copy()
    predictions["y"] = y_test
    predictions["p_lex"] = p_lex_test
    predictions["p_dns"] = p_dns_test
    predictions["p_cross_attentive_gate"] = minimal_prob
    predictions["g_cross_attentive_gate"] = minimal_gate
    predictions["p_standalone_conflict_classifier"] = direct_prob
    predictions["is_model_conditioned_high_conflict"] = hc
    diagnostics = pd.DataFrame(
        [
            {
                "system": "cross_attentive_gate",
                "split": "test",
                "mean_g_lex": round(float(minimal_gate.mean()), 4),
                "choose_lex_pct": round(float((minimal_gate >= 0.5).mean() * 100), 4),
                "params": count_parameters(minimal),
            },
            {
                "system": "standalone_conflict_classifier",
                "split": "test",
                "mean_g_lex": float("nan"),
                "choose_lex_pct": float("nan"),
                "params": count_parameters(direct),
            },
        ]
    )
    stamp = utc_stamp()
    metrics_path = args.out_dir / f"R021_CONFLICT_CLASSIFIER_METRICS_{stamp}.csv"
    diagnostics_path = args.out_dir / f"R021_CONFLICT_CLASSIFIER_DIAGNOSTICS_{stamp}.csv"
    predictions_path = args.out_dir / f"R021_CONFLICT_CLASSIFIER_PREDICTIONS_{stamp}.csv"
    metadata_path = args.out_dir / f"R021_CONFLICT_CLASSIFIER_METADATA_{stamp}.json"
    report_path = args.out_dir / f"R021_CONFLICT_CLASSIFIER_REPORT_{stamp}.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sample": str(args.sample),
        "sample_sha256": sha256_file(args.sample),
        "seed": args.seed,
        "device": str(device),
        "evaluation_type": "real_gt",
        "scope": "R021 simplicity check for standalone direct classifier vs explicit mixture gate",
        "split_rows": {name: len(part) for name, part in [("train", train), ("val", val), ("test", test)]},
        "split_domain_overlap": domain_overlap_rows(train, val, test),
        "high_conflict_definition": "model-conditioned: (p_lex >= 0.7 and p_dns <= 0.3) or reverse, using frozen experts",
        "outputs": {
            "metrics_csv": str(metrics_path),
            "diagnostics_csv": str(diagnostics_path),
            "predictions_csv": str(predictions_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(metrics_path, args.out_dir / "R021_CONFLICT_CLASSIFIER_METRICS.csv")
    shutil.copyfile(diagnostics_path, args.out_dir / "R021_CONFLICT_CLASSIFIER_DIAGNOSTICS.csv")
    shutil.copyfile(predictions_path, args.out_dir / "R021_CONFLICT_CLASSIFIER_PREDICTIONS.csv")
    shutil.copyfile(metadata_path, args.out_dir / "R021_CONFLICT_CLASSIFIER_METADATA.json")
    report = "\n".join(
        [
            "# R021 Conflict-Classifier Simplicity Check",
            "",
            f"Generated: {datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
            f"Sample: `{args.sample.as_posix()}`",
            f"Seed: `{args.seed}`",
            f"Device: `{device}`",
            "This run compares the explicit cross-attentive mixture gate against a direct classifier head using the same cross-attended representation.",
            "",
            "## Metrics",
            "",
            md_table(metrics),
            "",
            "## Diagnostics",
            "",
            md_table(diagnostics),
            "",
            "## Domain Overlap Audit",
            "",
            md_table(pd.DataFrame(metadata["split_domain_overlap"])),
            "",
            "## Reproducibility Metadata",
            "",
            f"- script SHA256: `{metadata['script_sha256']}`",
            f"- sample SHA256: `{metadata['sample_sha256']}`",
            "",
            "## Interpretation",
            "",
            "- A direct classifier can improve raw metrics, but it discards the explicit trust-allocation path.",
            "- This run is a deletion/simplicity check, not a new contribution claim.",
            "",
            "## Outputs",
            "",
            f"- metrics CSV: `{metrics_path.as_posix()}`",
            f"- diagnostics CSV: `{diagnostics_path.as_posix()}`",
            f"- predictions CSV: `{predictions_path.as_posix()}`",
            f"- metadata JSON: `{metadata_path.as_posix()}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    shutil.copyfile(report_path, args.out_dir / "R021_CONFLICT_CLASSIFIER_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R021_CONFLICT_CLASSIFIER_REPORT.md"), "device": str(device), "rows": len(frame)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
