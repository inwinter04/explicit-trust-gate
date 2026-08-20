#!/usr/bin/env python
"""Run R012-R014 gate ablations for the DeepURLBench pilot.

This script keeps the frozen char-TF-IDF lexical expert and DNS/IP logistic
expert fixed, then changes only the gate path. It is a pilot ablation runner,
not the final DomURLs_BERT experiment.
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


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


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


def metric_row(seed: int, ablation: str, split: str, y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "seed": seed,
        "ablation": ablation,
        "split": split,
        "rows": len(y_true),
        "positives": int(y_true.sum()),
        "AUPRC": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "FPR@95TPR": fpr_at_tpr(y_true, prob),
        "macro_F1": float(f1_score(y_true, pred, average="macro")) if len(np.unique(y_true)) == 2 else float("nan"),
        "ECE": ece_score(y_true, prob),
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
    out["domain"] = out["domain"].fillna("").astype(str)
    out["ttl_missing"] = out["TTL"].isna().astype(int)
    out["ttl_log1p"] = np.log1p(out["TTL"].fillna(0).clip(lower=0))
    return out[out["label"].isin(["benign", "malicious"]) & out["first_seen"].notna()].copy()


def split_parts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(frame[frame["split"].eq(name)].copy() for name in ("train", "val", "test"))  # type: ignore[return-value]


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


def tune_constant_gate(y_true: np.ndarray, p_lex: np.ndarray, p_dns: np.ndarray) -> tuple[float, float]:
    best_g = 0.0
    best_score = -float("inf")
    for g in np.linspace(0.0, 1.0, 101):
        prob = g * p_lex + (1 - g) * p_dns
        score = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan")
        if score > best_score:
            best_g = float(g)
            best_score = score
    return best_g, best_score


def gate_entropy(gate: np.ndarray) -> float:
    p = np.clip(gate.astype(float), 1e-7, 1 - 1e-7)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def paired_bootstrap_auprc_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed: int,
    n_boot: int = 2000,
) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    observed = float(average_precision_score(y_true, score_a) - average_precision_score(y_true, score_b))
    deltas = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        deltas.append(float(average_precision_score(sample_y, score_a[idx]) - average_precision_score(sample_y, score_b[idx])))
    if not deltas:
        return {"delta": observed, "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"delta": observed, "ci_low": float(low), "ci_high": float(high), "n_boot": len(deltas)}


def high_conflict_mask(p_lex: np.ndarray, p_dns: np.ndarray) -> np.ndarray:
    return ((p_lex >= 0.7) & (p_dns <= 0.3)) | ((p_dns >= 0.7) & (p_lex <= 0.3))


def independent_rule_mask(frame: pd.DataFrame) -> np.ndarray:
    domain = frame["domain"].fillna("").astype(str)
    sld = domain.map(lambda value: value.split(".")[0])
    digit_ratio = sld.map(lambda value: sum(ch.isdigit() for ch in value) / max(1, len(value)))
    lex_benign = (digit_ratio <= 0.15) & (~sld.str.contains("-", regex=False)) & (sld.str.len() <= 24) & (domain.str.count(r"\.") <= 2) & (domain.str.len() <= 25)
    multi_ip = pd.to_numeric(frame["ip_count"], errors="coerce").fillna(0) >= 2
    return (lex_benign & multi_ip).to_numpy()


def behavior_matrix(frame: pd.DataFrame, train: pd.DataFrame, shuffle_tokens: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[DNS_NUMERIC].apply(pd.to_numeric, errors="coerce")
    values = frame[DNS_NUMERIC].apply(pd.to_numeric, errors="coerce")
    medians = train_values.median().fillna(0.0)
    means = train_values.mean().fillna(0.0)
    stds = train_values.std().replace(0, 1.0).fillna(1.0)
    missing = values.isna().to_numpy(dtype=np.float32)
    scaled = ((values.fillna(medians) - means) / stds).clip(-8, 8).to_numpy(dtype=np.float32)
    if shuffle_tokens:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(DNS_NUMERIC))
        scaled = scaled[:, order]
        missing = missing[:, order]
    return scaled, missing


class GateNet(nn.Module):
    def __init__(self, feature_count: int, d_model: int = 32, heads: int = 4, mode: str = "cross_attention") -> None:
        super().__init__()
        self.mode = mode
        self.token_projection = nn.Linear(2, d_model)
        self.type_embedding = nn.Embedding(feature_count, d_model)
        self.query_projection = nn.Linear(2, d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        if mode == "cross_attention":
            input_dim = 2 + 3 * d_model
        elif mode == "pooled_mlp":
            input_dim = 2 + d_model
        elif mode == "query_only":
            input_dim = 2 + d_model
        else:
            raise ValueError(f"unknown mode: {mode}")
        self.gate_head = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, lex_logit: torch.Tensor, dns_logit: torch.Tensor, behavior: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        token = self.token_projection(torch.stack([behavior, missing], dim=-1)) + self.type_embedding.weight.unsqueeze(0)
        query = self.query_projection(torch.stack([lex_logit, dns_logit], dim=-1)).unsqueeze(1)
        if self.mode == "cross_attention":
            attended, _ = self.attention(query, token, token, need_weights=False)
            gate_input = torch.cat([lex_logit.unsqueeze(1), dns_logit.unsqueeze(1), query[:, 0, :], self.norm(attended[:, 0, :]), token.mean(dim=1)], dim=1)
        elif self.mode == "pooled_mlp":
            gate_input = torch.cat([lex_logit.unsqueeze(1), dns_logit.unsqueeze(1), token.mean(dim=1)], dim=1)
        else:
            gate_input = torch.cat([lex_logit.unsqueeze(1), dns_logit.unsqueeze(1), query[:, 0, :]], dim=1)
        return torch.sigmoid(self.gate_head(gate_input).squeeze(1))


def train_gate(
    model: GateNet,
    inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    y_true: torch.Tensor,
    seed: int,
    device: torch.device,
    gate_loss_weight: float,
) -> GateNet:
    seed_everything(seed)
    model.to(device)
    inputs = tuple(item.to(device) for item in inputs)
    target = target.to(device)
    y_true = y_true.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best_state = None
    best_loss = float("inf")
    patience = 0
    for _ in range(350):
        model.train()
        gate = model(*inputs)
        mixture = gate * torch.sigmoid(inputs[0]) + (1 - gate) * torch.sigmoid(inputs[1])
        loss = nn.functional.binary_cross_entropy(mixture.clamp(1e-6, 1 - 1e-6), y_true)
        if gate_loss_weight:
            loss = loss + gate_loss_weight * nn.functional.mse_loss(gate, target)
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


def eval_gate(model: GateNet, inputs: tuple[torch.Tensor, ...], p_lex: np.ndarray, p_dns: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        gate = model(*(item.to(device) for item in inputs)).cpu().numpy()
    return gate * p_lex + (1 - gate) * p_dns, gate


def md_table(frame: pd.DataFrame) -> str:
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--seeds", default="20260708,20260709,20260710")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    frame = add_features(pd.read_csv(args.sample))
    train, val, test = split_parts(frame)
    experts = fit_experts(train, val, test)
    y_val = torch.tensor(y(val), dtype=torch.float32)
    y_test = y(test)
    p_lex_val, p_dns_val = experts["p_lex_val"], experts["p_dns_val"]
    p_lex_test, p_dns_test = experts["p_lex_test"], experts["p_dns_test"]
    y_val_np = y(val)
    target_val_np = oracle_gate_target(y_val_np, p_lex_val, p_dns_val)
    target_test_np = oracle_gate_target(y_test, p_lex_test, p_dns_test)
    target = torch.tensor(target_val_np, dtype=torch.float32)
    tuned_g, tuned_val_auprc = tune_constant_gate(y_val_np, p_lex_val, p_dns_val)
    hc = high_conflict_mask(p_lex_test, p_dns_test)
    rule = independent_rule_mask(test)

    rows = []
    diag_rows = []
    prediction_store: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    ablations = [
        ("fixed_lexical", None, 1.0, False),
        ("fixed_behavior", None, 0.0, False),
        ("fixed_average", None, 0.5, False),
        ("validation_tuned_constant_gate", None, tuned_g, False),
        ("cross_attention_with_gate_loss", "cross_attention", None, False),
        ("cross_attention_no_gate_loss", "cross_attention", None, False),
        ("cross_attention_token_shuffle", "cross_attention", None, True),
        ("pooled_mlp_same_tokens", "pooled_mlp", None, False),
        ("query_only_gate", "query_only", None, False),
    ]

    for seed in seeds:
        seed_everything(seed)
        val_behavior, val_missing = behavior_matrix(val, train, False, seed)
        test_behavior, test_missing = behavior_matrix(test, train, False, seed)
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
        for name, mode, fixed_g, shuffle in ablations:
            if fixed_g is not None:
                val_prob = fixed_g * p_lex_val + (1 - fixed_g) * p_dns_val
                val_gate = np.full(len(val), fixed_g)
                prob = fixed_g * p_lex_test + (1 - fixed_g) * p_dns_test
                gate = np.full(len(test), fixed_g)
                params = 0
            else:
                if shuffle:
                    shuffled_val_behavior, shuffled_val_missing = behavior_matrix(val, train, True, seed)
                    shuffled_test_behavior, shuffled_test_missing = behavior_matrix(test, train, True, seed)
                    run_val_inputs = (val_inputs[0], val_inputs[1], torch.from_numpy(shuffled_val_behavior), torch.from_numpy(shuffled_val_missing))
                    run_test_inputs = (test_inputs[0], test_inputs[1], torch.from_numpy(shuffled_test_behavior), torch.from_numpy(shuffled_test_missing))
                else:
                    run_val_inputs = val_inputs
                    run_test_inputs = test_inputs
                gate_weight = 0.25 if name != "cross_attention_no_gate_loss" else 0.0
                model = train_gate(GateNet(len(DNS_NUMERIC), mode=mode or "cross_attention"), run_val_inputs, target, y_val, seed, device, gate_weight)
                params = parameter_count(model)
                val_prob, val_gate = eval_gate(model, run_val_inputs, p_lex_val, p_dns_val, device)
                prob, gate = eval_gate(model, run_test_inputs, p_lex_test, p_dns_test, device)
            prediction_store.setdefault(seed, {})[name] = {"test_prob": prob, "val_prob": val_prob}
            for split_name, mask in [("test", np.ones(len(test), dtype=bool)), ("model_conditioned_high_conflict", hc), ("independent_lex_benign_multi_ip", rule)]:
                rows.append(metric_row(seed, name, split_name, y_test[mask], prob[mask]))
            lex_correct = (p_lex_test >= 0.5).astype(int) == y_test
            dns_correct = (p_dns_test >= 0.5).astype(int) == y_test
            diag_rows.append(
                {
                    "seed": seed,
                    "ablation": name,
                    "parameters": params,
                    "mean_g_lex": float(gate.mean()),
                    "gate_entropy_bits": gate_entropy(gate),
                    "val_gate_target_mse": float(np.mean((val_gate - target_val_np) ** 2)),
                    "test_gate_target_mse_diagnostic": float(np.mean((gate - target_test_np) ** 2)),
                    "choose_lex_pct": float((gate >= 0.5).mean() * 100),
                    "dns_fixes_lex_errors": int((~lex_correct & dns_correct).sum()),
                    "gate_chooses_dns_on_dns_fix": int(((~lex_correct & dns_correct) & (gate < 0.5)).sum()),
                    "lex_fixes_dns_errors": int((lex_correct & ~dns_correct).sum()),
                    "gate_chooses_lex_on_lex_fix": int(((lex_correct & ~dns_correct) & (gate >= 0.5)).sum()),
                }
            )

    bootstrap_rows = []
    references = ["fixed_lexical", "validation_tuned_constant_gate", "pooled_mlp_same_tokens"]
    split_masks = {
        "test": np.ones(len(test), dtype=bool),
        "model_conditioned_high_conflict": hc,
        "independent_lex_benign_multi_ip": rule,
    }
    for seed in seeds:
        predictions = prediction_store[seed]
        for split_name, mask in split_masks.items():
            split_y = y_test[mask]
            for ablation, payload in predictions.items():
                if ablation in references:
                    continue
                for reference in references:
                    if reference not in predictions:
                        continue
                    stats = paired_bootstrap_auprc_delta(
                        split_y,
                        payload["test_prob"][mask],
                        predictions[reference]["test_prob"][mask],
                        seed=stable_seed(seed, split_name, ablation, reference),
                    )
                    bootstrap_rows.append(
                        {
                            "seed": seed,
                            "split": split_name,
                            "ablation": ablation,
                            "reference": reference,
                            "delta_AUPRC": stats["delta"],
                            "ci_low": stats["ci_low"],
                            "ci_high": stats["ci_high"],
                            "n_boot": stats["n_boot"],
                        }
                    )

    metrics = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diag_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary = (
        metrics.groupby(["split", "ablation"], as_index=False)
        .agg(rows=("rows", "first"), positives=("positives", "first"), AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"), FPR95_mean=("FPR@95TPR", "mean"), ECE_mean=("ECE", "mean"))
        .sort_values(["split", "ablation"])
    )
    diag_summary = (
        diagnostics.groupby("ablation", as_index=False)
        .agg(
            parameters=("parameters", "first"),
            mean_g_lex=("mean_g_lex", "mean"),
            gate_entropy_bits=("gate_entropy_bits", "mean"),
            val_gate_target_mse=("val_gate_target_mse", "mean"),
            test_gate_target_mse_diagnostic=("test_gate_target_mse_diagnostic", "mean"),
            choose_lex_pct=("choose_lex_pct", "mean"),
            gate_chooses_dns_on_dns_fix=("gate_chooses_dns_on_dns_fix", "mean"),
            gate_chooses_lex_on_lex_fix=("gate_chooses_lex_on_lex_fix", "mean"),
        )
        .sort_values("ablation")
    )
    bootstrap_summary = (
        bootstrap.groupby(["split", "ablation", "reference"], as_index=False)
        .agg(delta_AUPRC_mean=("delta_AUPRC", "mean"), ci_low_mean=("ci_low", "mean"), ci_high_mean=("ci_high", "mean"), seeds=("seed", "nunique"))
        .sort_values(["split", "ablation", "reference"])
    )
    for df in [summary, diag_summary]:
        for col in df.select_dtypes(include=[float]).columns:
            df[col] = df[col].round(4)
    for df in [bootstrap, bootstrap_summary]:
        for col in df.select_dtypes(include=[float]).columns:
            df[col] = df[col].round(4)

    stamp = utc_stamp()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / f"R012_R014_GATE_ABLATIONS_METRICS_{stamp}.csv"
    diagnostics_path = args.out_dir / f"R012_R014_GATE_ABLATIONS_DIAGNOSTICS_{stamp}.csv"
    bootstrap_path = args.out_dir / f"R012_R014_GATE_ABLATIONS_BOOTSTRAP_{stamp}.csv"
    report_path = args.out_dir / f"R012_R014_GATE_ABLATIONS_REPORT_{stamp}.md"
    metadata_path = args.out_dir / f"R012_R014_GATE_ABLATIONS_METADATA_{stamp}.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8")
    bootstrap.to_csv(bootstrap_path, index=False, encoding="utf-8")
    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sample": str(args.sample),
        "sample_sha256": sha256_file(args.sample),
        "seeds": seeds,
        "device": str(device),
        "validation_tuned_constant_gate": {"g": tuned_g, "val_AUPRC": tuned_val_auprc},
        "evaluation_type": "real_gt metrics; validation-only label-dependent oracle target for gate-supervision ablations",
        "outputs": {"metrics_csv": str(metrics_path), "diagnostics_csv": str(diagnostics_path), "bootstrap_csv": str(bootstrap_path)},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(metrics_path, args.out_dir / "R012_R014_GATE_ABLATIONS_METRICS.csv")
    shutil.copyfile(diagnostics_path, args.out_dir / "R012_R014_GATE_ABLATIONS_DIAGNOSTICS.csv")
    shutil.copyfile(bootstrap_path, args.out_dir / "R012_R014_GATE_ABLATIONS_BOOTSTRAP.csv")
    shutil.copyfile(metadata_path, args.out_dir / "R012_R014_GATE_ABLATIONS_METADATA.json")

    report = "\n".join(
        [
            "# R012-R014 Gate Ablations",
            "",
            f"Generated: {datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
            f"Sample: `{args.sample.as_posix()}`",
            f"Seeds: `{', '.join(map(str, seeds))}`",
            "Frozen experts: char-TF-IDF lexical and DNS/IP logistic behavior.",
            "",
            "## Ablations",
            "",
            "- `fixed_lexical`: fixed gate `g=1`.",
            "- `fixed_behavior`: fixed gate `g=0`.",
            "- `fixed_average`: fixed gate `g=0.5`.",
            f"- `validation_tuned_constant_gate`: validation-tuned fixed gate `g={tuned_g:.2f}` (validation AUPRC {tuned_val_auprc:.4f}).",
            "- `cross_attention_with_gate_loss`: R008 pilot gate with validation-only oracle supervision.",
            "- `cross_attention_no_gate_loss`: same architecture without oracle gate loss.",
            "- `cross_attention_token_shuffle`: same architecture with behavior token order permuted, breaking feature-type alignment.",
            "- `pooled_mlp_same_tokens`: no attention; pooled behavior token gate.",
            "- `query_only_gate`: no behavior tokens in the gate, only frozen expert logits.",
            "",
            "## Metric Summary",
            "",
            md_table(summary),
            "",
            "## Gate Diagnostics",
            "",
            md_table(diag_summary),
            "",
            "## Paired Bootstrap Delta Summary",
            "",
            "Delta is AUPRC(ablation) - AUPRC(reference); intervals are averaged across the three seed-level paired bootstraps.",
            "",
            md_table(bootstrap_summary),
            "",
            "## Integrity Notes",
            "",
            "- Test metrics use dataset-provided labels.",
            "- Gate-supervision variants use a label-dependent validation-only oracle target, disclosed here as training supervision.",
            "- `validation_tuned_constant_gate` tunes one scalar on validation labels, then applies it unchanged to test.",
            "- `test_gate_target_mse_diagnostic` compares gates to a label-dependent test oracle only as post hoc analysis, not training.",
            "- The model-conditioned high-conflict split remains diagnostic; the independent rule split is heuristic.",
            "",
            "## Outputs",
            "",
            f"- metrics CSV: `{metrics_path.as_posix()}`",
            f"- diagnostics CSV: `{diagnostics_path.as_posix()}`",
            f"- bootstrap CSV: `{bootstrap_path.as_posix()}`",
            f"- metadata JSON: `{metadata_path.as_posix()}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    shutil.copyfile(report_path, args.out_dir / "R012_R014_GATE_ABLATIONS_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R012_R014_GATE_ABLATIONS_REPORT.md"), "seeds": seeds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
