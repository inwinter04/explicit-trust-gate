#!/usr/bin/env python
"""Run R017 missing-signal and modality-dropout diagnostic.

R016 showed smooth fallback under synthetic behavior-view removal. R017 asks
whether that behavior is tied to the missing-mask input and/or training-time
modality dropout. This remains diagnostic only because R015 has not passed.
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
from sklearn.metrics import average_precision_score

import run_r016_missing_modality_diagnostic as r016


def train_gate_variant(
    p_lex: np.ndarray,
    p_dns: np.ndarray,
    p_dns_missing: np.ndarray,
    behavior: np.ndarray,
    missing: np.ndarray,
    y_true_np: np.ndarray,
    seed: int,
    device: torch.device,
    *,
    modality_dropout: float,
    use_missing_mask: bool,
) -> r016.GateNet:
    r016.seed_everything(seed)
    model = r016.GateNet(len(r016.DNS_NUMERIC)).to(device)
    y_true = torch.tensor(y_true_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    best_state = None
    best_loss = float("inf")
    patience = 0

    for _ in range(350):
        if modality_dropout > 0:
            mask = rng.random(len(y_true_np)) < modality_dropout
        else:
            mask = np.zeros(len(y_true_np), dtype=bool)
        p_dns_aug = p_dns.copy()
        p_dns_aug[mask] = p_dns_missing[mask]
        behavior_aug = behavior.copy()
        missing_aug = missing.copy()
        behavior_aug[mask, :] = 0.0
        missing_aug[mask, :] = 1.0
        if not use_missing_mask:
            missing_aug[:, :] = 0.0
        target_np = r016.oracle_gate_target(y_true_np, p_lex, p_dns_aug)
        inputs = tuple(item.to(device) for item in r016.make_inputs(p_lex, p_dns_aug, behavior_aug, missing_aug))
        target = torch.tensor(target_np, dtype=torch.float32, device=device)

        model.train()
        gate = model(*inputs)
        mixture = gate * torch.sigmoid(inputs[0]) + (1 - gate) * torch.sigmoid(inputs[1])
        loss = torch.nn.functional.binary_cross_entropy(mixture.clamp(1e-6, 1 - 1e-6), y_true)
        loss = loss + 0.25 * torch.nn.functional.mse_loss(gate, target)
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


def md_table(frame: pd.DataFrame) -> str:
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-dns", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--seeds", default="20260708,20260709,20260710")
    parser.add_argument("--missing-rates", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--dropout-prob", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    missing_rates = [float(rate.strip()) for rate in args.missing_rates.split(",") if rate.strip()]

    frame = r016.add_features(pd.read_csv(args.with_dns))
    train, val, test = r016.split_parts(frame)
    experts = r016.fit_experts(train, val, test)
    y_val = r016.y(val)
    y_test = r016.y(test)
    p_lex_val = experts["p_lex_val"]  # type: ignore[assignment]
    p_dns_val = experts["p_dns_val"]  # type: ignore[assignment]
    p_dns_val_missing = experts["p_dns_val_missing"]  # type: ignore[assignment]
    p_lex_test = experts["p_lex_test"]  # type: ignore[assignment]
    p_dns_test = experts["p_dns_test"]  # type: ignore[assignment]
    p_dns_test_missing = experts["p_dns_test_missing"]  # type: ignore[assignment]
    val_behavior, val_missing = r016.behavior_matrix(val, train)

    variants = [
        ("no_dropout_missing_mask", 0.0, True),
        ("dropout_missing_mask", args.dropout_prob, True),
        ("no_dropout_no_missing_mask", 0.0, False),
        ("dropout_no_missing_mask", args.dropout_prob, False),
    ]
    rows = []
    diag_rows = []
    for seed in seeds:
        models = {
            name: train_gate_variant(
                p_lex_val,
                p_dns_val,
                p_dns_val_missing,
                val_behavior,
                val_missing,
                y_val,
                seed,
                device,
                modality_dropout=dropout,
                use_missing_mask=use_mask,
            )
            for name, dropout, use_mask in variants
        }
        rng = np.random.default_rng(seed)
        for rate in missing_rates:
            mask = rng.random(len(test)) < rate
            masked_test = r016.apply_whole_view_missing(test, mask)
            test_behavior, test_missing = r016.behavior_matrix(masked_test, train)
            p_dns_mixed = p_dns_test.copy()
            p_dns_mixed[mask] = p_dns_test_missing[mask]
            for name, _, use_mask in variants:
                eval_missing = test_missing.copy()
                if not use_mask:
                    eval_missing[:, :] = 0.0
                inputs = r016.make_inputs(p_lex_test, p_dns_mixed, test_behavior, eval_missing)
                prob, gate = r016.eval_gate(models[name], inputs, p_lex_test, p_dns_mixed, device)
                rows.append(r016.metric_row(seed, name, "with_dns_test_synthetic_missing", rate, y_test, prob))
                diag_rows.append(
                    {
                        "seed": seed,
                        "variant": name,
                        "missing_rate": rate,
                        "actual_missing_pct": float(mask.mean() * 100),
                        "mean_g_lex": float(gate.mean()),
                        "choose_lex_pct": float((gate >= 0.5).mean() * 100),
                        "masked_mean_g_lex": float(gate[mask].mean()) if mask.any() else float("nan"),
                        "unmasked_mean_g_lex": float(gate[~mask].mean()) if (~mask).any() else float("nan"),
                    }
                )

    metrics = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diag_rows)
    summary = (
        metrics.groupby(["system", "missing_rate"], as_index=False)
        .agg(
            rows=("rows", "first"),
            positives=("positives", "first"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_std=("AUPRC", "std"),
            FPR95_mean=("FPR@95TPR", "mean"),
            ECE_mean=("ECE", "mean"),
        )
        .sort_values(["missing_rate", "system"])
    )
    diag_summary = (
        diagnostics.groupby(["variant", "missing_rate"], as_index=False)
        .agg(
            actual_missing_pct=("actual_missing_pct", "mean"),
            mean_g_lex=("mean_g_lex", "mean"),
            choose_lex_pct=("choose_lex_pct", "mean"),
            masked_mean_g_lex=("masked_mean_g_lex", "mean"),
            unmasked_mean_g_lex=("unmasked_mean_g_lex", "mean"),
        )
        .sort_values(["missing_rate", "variant"])
    )
    slope_rows = []
    for system, part in summary.groupby("system"):
        if len(part) >= 2:
            slope = float(np.polyfit(part["missing_rate"], part["AUPRC_mean"], 1)[0])
            retention = float(part.loc[part["missing_rate"].eq(1.0), "AUPRC_mean"].iloc[0] - part.loc[part["missing_rate"].eq(0.0), "AUPRC_mean"].iloc[0])
            full_missing = float(part.loc[part["missing_rate"].eq(1.0), "AUPRC_mean"].iloc[0])
        else:
            slope = float("nan")
            retention = float("nan")
            full_missing = float("nan")
        slope_rows.append({"system": system, "AUPRC_slope": slope, "AUPRC_delta_100_minus_0": retention, "AUPRC_at_100_missing": full_missing})
    slope_summary = pd.DataFrame(slope_rows).sort_values("system")

    for df in [summary, diag_summary, slope_summary]:
        for col in df.select_dtypes(include=[float]).columns:
            df[col] = df[col].round(4)

    stamp = r016.utc_stamp()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / f"R017_MISSING_ABLATION_METRICS_{stamp}.csv"
    diagnostics_path = args.out_dir / f"R017_MISSING_ABLATION_DIAGNOSTICS_{stamp}.csv"
    report_path = args.out_dir / f"R017_MISSING_ABLATION_REPORT_{stamp}.md"
    metadata_path = args.out_dir / f"R017_MISSING_ABLATION_METADATA_{stamp}.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8")
    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": r016.sha256_file(Path(__file__).resolve()),
        "dependency_r016_script": str(Path(r016.__file__).resolve()),
        "dependency_r016_sha256": r016.sha256_file(Path(r016.__file__).resolve()),
        "with_dns": str(args.with_dns),
        "with_dns_sha256": r016.sha256_file(args.with_dns),
        "seeds": seeds,
        "missing_rates": missing_rates,
        "dropout_prob": args.dropout_prob,
        "device": str(device),
        "evaluation_type": "real_gt diagnostic; validation-only gate training; synthetic whole-view masking; not point-in-time C2 evidence because R015 failed",
        "outputs": {"metrics_csv": str(metrics_path), "diagnostics_csv": str(diagnostics_path)},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(metrics_path, args.out_dir / "R017_MISSING_ABLATION_METRICS.csv")
    shutil.copyfile(diagnostics_path, args.out_dir / "R017_MISSING_ABLATION_DIAGNOSTICS.csv")
    shutil.copyfile(metadata_path, args.out_dir / "R017_MISSING_ABLATION_METADATA.json")

    report = "\n".join(
        [
            "# R017 Missing-Signal and Modality-Dropout Diagnostic",
            "",
            f"Generated: {datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
            f"With-DNS sample: `{args.with_dns.as_posix()}`",
            f"Seeds: `{', '.join(map(str, seeds))}`",
            f"Training modality dropout probability: `{args.dropout_prob}`",
            f"Device: `{device}`",
            "",
            "## Scope",
            "",
            "This is a diagnostic only. R015 currently fails the strict point-in-time provenance gate, and the final paper's `[MISS]` embedding is not implemented in this frozen-expert pilot.",
            "",
            "## Variants",
            "",
            "- `no_dropout_missing_mask`: R016-style gate training; missing-mask channel visible.",
            "- `dropout_missing_mask`: validation gate training uses random whole-view behavior dropout; missing-mask channel visible.",
            "- `no_dropout_no_missing_mask`: no modality dropout; missing-mask channel hidden from the gate.",
            "- `dropout_no_missing_mask`: modality dropout training, but missing-mask channel hidden from the gate.",
            "",
            "## Metric Summary",
            "",
            md_table(summary),
            "",
            "## Sensitivity Summary",
            "",
            md_table(slope_summary),
            "",
            "## Gate Diagnostics",
            "",
            md_table(diag_summary),
            "",
            "## Integrity Notes",
            "",
            "- Test metrics use dataset-provided labels.",
            "- Gate supervision uses validation labels only and is label-dependent training supervision.",
            "- Missingness is synthetically injected at train/eval time; it is not a natural deployment trace.",
            "- These results can guide R017 design, but they do not support C2 until R015 passes.",
            "",
            "## Outputs",
            "",
            f"- metrics CSV: `{metrics_path.as_posix()}`",
            f"- diagnostics CSV: `{diagnostics_path.as_posix()}`",
            f"- metadata JSON: `{metadata_path.as_posix()}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    shutil.copyfile(report_path, args.out_dir / "R017_MISSING_ABLATION_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R017_MISSING_ABLATION_REPORT.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
