#!/usr/bin/env python
"""Run R020 simplicity check: cross-attentive gate vs bounded residual correction.

This pre-screen reuses the frozen lexical and behavior experts from the
DeepURLBench local with-DNS temporal sample and compares:

- minimal cross-attentive trust gate;
- overbuilt residual-correction variant.

The question is whether the extra residual term buys enough to justify the
additional component. If not, the paper stays with the smaller gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def y(frame: pd.DataFrame) -> np.ndarray:
    return frame["label"].eq("malicious").astype(int).to_numpy()


def fpr_at_tpr(y_true: np.ndarray, score: np.ndarray, target: float = 0.95) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, score)
    valid = fpr[tpr >= target]
    return float(valid.min()) if len(valid) else 1.0


def ece_score(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1.0 else prob <= high)
        if mask.any():
            total += float(mask.mean()) * abs(float(prob[mask].mean()) - float(y_true[mask].mean()))
    return total


def metric_row(system: str, split: str, y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "system": system,
        "split": split,
        "rows": len(y_true),
        "positives": int(y_true.sum()),
        "AUPRC": round(float(average_precision_score(y_true, prob)), 4) if len(np.unique(y_true)) == 2 else float("nan"),
        "FPR@95TPR": round(fpr_at_tpr(y_true, prob), 4),
        "macro_F1": round(float(f1_score(y_true, pred, average="macro")), 4) if len(np.unique(y_true)) == 2 else float("nan"),
        "ECE": round(ece_score(y_true, prob), 4),
    }


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["first_seen"] = pd.to_datetime(out["first_seen"], errors="coerce")
    out["TTL"] = pd.to_numeric(out["TTL"], errors="coerce")
    out["has_dns"] = out["has_dns"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    for col in DNS_NUMERIC:
        if col not in {"TTL", "ttl_log1p", "ttl_missing", "has_dns"}:
            out[col] = pd.to_numeric(out.get(col, 0), errors="coerce").fillna(0)
    out["url"] = out["url"].fillna("").astype(str)
    out["ttl_missing"] = out["TTL"].isna().astype(int)
    out["ttl_log1p"] = np.log1p(out["TTL"].fillna(0).clip(lower=0))
    return out[out["label"].isin(["benign", "malicious"]) & out["first_seen"].notna()].copy()


def split_parts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(frame[frame["split"].eq(name)].copy() for name in ("train", "val", "test"))  # type: ignore[return-value]


def domain_overlap_rows(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    split_domains = {
        "train": set(train["domain"].astype(str)),
        "val": set(val["domain"].astype(str)),
        "test": set(test["domain"].astype(str)),
    }
    rows = []
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_domains[left] & split_domains[right]
        rows.append(
            {
                "left_split": left,
                "right_split": right,
                "left_domains": len(split_domains[left]),
                "right_domains": len(split_domains[right]),
                "overlap_domains": len(overlap),
                "overlap_pct_of_right": round(100 * len(overlap) / max(1, len(split_domains[right])), 4),
            }
        )
    return rows


def fit_experts(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    lexical = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=20260708)),
        ]
    )
    behavior = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=20260708)),
        ]
    )
    lexical.fit(train["url"], y(train))
    behavior.fit(train[DNS_NUMERIC], y(train))
    return {
        "p_lex_val": lexical.predict_proba(val["url"])[:, 1],
        "p_dns_val": behavior.predict_proba(val[DNS_NUMERIC])[:, 1],
        "p_lex_test": lexical.predict_proba(test["url"])[:, 1],
        "p_dns_test": behavior.predict_proba(test[DNS_NUMERIC])[:, 1],
    }


def clipped_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def oracle_gate_target(y_true: np.ndarray, p_lex: np.ndarray, p_dns: np.ndarray) -> np.ndarray:
    lex_loss = -(y_true * np.log(np.clip(p_lex, 1e-7, 1.0)) + (1 - y_true) * np.log(np.clip(1 - p_lex, 1e-7, 1.0)))
    dns_loss = -(y_true * np.log(np.clip(p_dns, 1e-7, 1.0)) + (1 - y_true) * np.log(np.clip(1 - p_dns, 1e-7, 1.0)))
    return (lex_loss <= dns_loss).astype(np.float32)


def high_conflict_mask(p_lex: np.ndarray, p_dns: np.ndarray) -> np.ndarray:
    return ((p_lex >= 0.7) & (p_dns <= 0.3)) | ((p_dns >= 0.7) & (p_lex <= 0.3))


def behavior_matrix(frame: pd.DataFrame, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[DNS_NUMERIC].apply(pd.to_numeric, errors="coerce")
    values = frame[DNS_NUMERIC].apply(pd.to_numeric, errors="coerce")
    medians = train_values.median().fillna(0.0)
    means = train_values.mean().fillna(0.0)
    stds = train_values.std().replace(0, 1.0).fillna(1.0)
    missing = values.isna().to_numpy(dtype=np.float32)
    scaled = ((values.fillna(medians) - means) / stds).clip(-8, 8).to_numpy(dtype=np.float32)
    return scaled, missing


class BaseGate(nn.Module):
    def __init__(self, feature_count: int, d_model: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.token_projection = nn.Linear(2, d_model)
        self.type_embedding = nn.Embedding(feature_count, d_model)
        self.query_projection = nn.Linear(2, d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate_head = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def encode(self, lex_logit: torch.Tensor, dns_logit: torch.Tensor, behavior: torch.Tensor, missing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_input = torch.stack([behavior, missing], dim=-1)
        token = self.token_projection(token_input) + self.type_embedding.weight.unsqueeze(0)
        query = self.query_projection(torch.stack([lex_logit, dns_logit], dim=-1)).unsqueeze(1)
        attended, _ = self.attention(query, token, token, need_weights=False)
        attended = self.norm(attended[:, 0, :])
        pooled = token.mean(dim=1)
        gate_input = torch.cat([lex_logit.unsqueeze(1), dns_logit.unsqueeze(1), attended, pooled], dim=1)
        return gate_input, attended, pooled


class GateNet(BaseGate):
    def forward(self, lex_logit: torch.Tensor, dns_logit: torch.Tensor, behavior: torch.Tensor, missing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_input, attended, _ = self.encode(lex_logit, dns_logit, behavior, missing)
        gate = torch.sigmoid(self.gate_head(gate_input).squeeze(1))
        return gate, attended


class ResidualGateNet(BaseGate):
    def __init__(self, feature_count: int, d_model: int = 32, heads: int = 4, residual_scale: float = 0.2) -> None:
        super().__init__(feature_count, d_model=d_model, heads=heads)
        self.residual_scale = residual_scale
        self.residual_head = nn.Sequential(
            nn.Linear(d_model * 2 + 2, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, lex_logit: torch.Tensor, dns_logit: torch.Tensor, behavior: torch.Tensor, missing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_input, attended, pooled = self.encode(lex_logit, dns_logit, behavior, missing)
        gate = torch.sigmoid(self.gate_head(gate_input).squeeze(1))
        delta = self.residual_scale * torch.tanh(self.residual_head(gate_input).squeeze(1))
        return gate, delta, attended


def train_gate(
    model: nn.Module,
    train_inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    y_true: torch.Tensor,
    seed: int,
    device: torch.device,
    residual: bool = False,
) -> nn.Module:
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
        if residual:
            gate, delta, _ = model(*inputs)
            mixture_logit = gate * inputs[0] + (1 - gate) * inputs[1] + delta
            loss = nn.functional.binary_cross_entropy(torch.sigmoid(mixture_logit).clamp(1e-6, 1 - 1e-6), y_true)
            loss = loss + 0.25 * nn.functional.mse_loss(gate, target) + 0.05 * delta.abs().mean()
        else:
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


def evaluate_gate(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    p_lex: np.ndarray,
    p_dns: np.ndarray,
    device: torch.device,
    residual: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        outputs = model(*(item.to(device) for item in inputs))
        if residual:
            gate, delta, _ = outputs
            lex = inputs[0].to(device)
            dns = inputs[1].to(device)
            prob = torch.sigmoid(gate * lex + (1 - gate) * dns + delta).cpu().numpy()
            return prob, gate.cpu().numpy(), delta.cpu().numpy()
        gate, _ = outputs
        lex = inputs[0].to(device)
        dns = inputs[1].to(device)
        prob = torch.sigmoid(gate * lex + (1 - gate) * dns).cpu().numpy()
        gate_np = gate.cpu().numpy()
        return prob, gate_np, np.zeros_like(prob)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def md_table(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


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
    gate_target = torch.tensor(oracle_gate_target(y(val), p_lex_val, p_dns_val), dtype=torch.float32)
    y_val = torch.tensor(y(val), dtype=torch.float32)

    minimal = train_gate(GateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val, args.seed, device, residual=False)
    residual = train_gate(ResidualGateNet(len(DNS_NUMERIC)), val_inputs, gate_target, y_val, args.seed, device, residual=True)

    minimal_prob, minimal_gate, _ = evaluate_gate(minimal, test_inputs, p_lex_test, p_dns_test, device, residual=False)
    residual_prob, residual_gate, residual_delta = evaluate_gate(residual, test_inputs, p_lex_test, p_dns_test, device, residual=True)

    y_test = y(test)
    hc = high_conflict_mask(p_lex_test, p_dns_test)
    rows = [
        metric_row("cross_attentive_gate", "test", y_test, minimal_prob),
        metric_row("residual_correction_gate", "test", y_test, residual_prob),
    ]
    if hc.any():
        rows.extend(
            [
                metric_row("cross_attentive_gate", "test_high_conflict", y_test[hc], minimal_prob[hc]),
                metric_row("residual_correction_gate", "test_high_conflict", y_test[hc], residual_prob[hc]),
            ]
        )
    metrics = pd.DataFrame(rows)
    predictions = test[["split", "url", "domain", "first_seen", "label", *DNS_NUMERIC]].copy()
    predictions["y"] = y_test
    predictions["p_lex"] = p_lex_test
    predictions["p_dns"] = p_dns_test
    predictions["p_cross_attentive_gate"] = minimal_prob
    predictions["g_cross_attentive_gate"] = minimal_gate
    predictions["p_residual_correction_gate"] = residual_prob
    predictions["g_residual_correction_gate"] = residual_gate
    predictions["delta_residual_correction_gate"] = residual_delta
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
                "system": "residual_correction_gate",
                "split": "test",
                "mean_g_lex": round(float(residual_gate.mean()), 4),
                "choose_lex_pct": round(float((residual_gate >= 0.5).mean() * 100), 4),
                "params": count_parameters(residual),
                "mean_abs_delta": round(float(np.abs(residual_delta).mean()), 4),
                "mean_delta": round(float(residual_delta.mean()), 4),
            },
        ]
    )
    stamp = utc_stamp()
    metrics_path = args.out_dir / f"R020_SIMPLICITY_CHECK_METRICS_{stamp}.csv"
    diagnostics_path = args.out_dir / f"R020_SIMPLICITY_CHECK_DIAGNOSTICS_{stamp}.csv"
    predictions_path = args.out_dir / f"R020_SIMPLICITY_CHECK_PREDICTIONS_{stamp}.csv"
    metadata_path = args.out_dir / f"R020_SIMPLICITY_CHECK_METADATA_{stamp}.json"
    report_path = args.out_dir / f"R020_SIMPLICITY_CHECK_REPORT_{stamp}.md"
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
        "scope": "R020 simplicity check for residual correction vs minimal gate",
        "split_rows": {name: len(part) for name, part in [("train", train), ("val", val), ("test", test)]},
        "split_domain_overlap": domain_overlap_rows(train, val, test),
        "high_conflict_definition": "model-conditioned: (p_lex >= 0.7 and p_dns <= 0.3) or reverse, using frozen experts",
        "outputs": {
            "metrics_csv": str(metrics_path),
            "diagnostics_csv": str(diagnostics_path),
            "predictions_csv": str(predictions_path),
        },
        "params": {
            "cross_attentive_gate": count_parameters(minimal),
            "residual_correction_gate": count_parameters(residual),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(metrics_path, args.out_dir / "R020_SIMPLICITY_CHECK_METRICS.csv")
    shutil.copyfile(diagnostics_path, args.out_dir / "R020_SIMPLICITY_CHECK_DIAGNOSTICS.csv")
    shutil.copyfile(predictions_path, args.out_dir / "R020_SIMPLICITY_CHECK_PREDICTIONS.csv")
    shutil.copyfile(metadata_path, args.out_dir / "R020_SIMPLICITY_CHECK_METADATA.json")
    report = "\n".join(
        [
            "# R020 Simplicity Check",
            "",
            f"Generated: {datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
            f"Sample: `{args.sample.as_posix()}`",
            f"Seed: `{args.seed}`",
            f"Device: `{device}`",
            "This run compares the minimal cross-attentive gate against a bounded residual-correction variant.",
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
            "- residual is bounded by `0.2 * tanh(.)` to prevent it from silently dominating the mixture.",
            "",
            "## Interpretation",
            "",
            "- If the residual variant does not beat the minimal gate, the paper should stay with the smaller mechanism.",
            "- The goal here is deletion pressure, not another positive claim.",
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
    shutil.copyfile(report_path, args.out_dir / "R020_SIMPLICITY_CHECK_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R020_SIMPLICITY_CHECK_REPORT.md"), "device": str(device), "rows": len(frame)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
