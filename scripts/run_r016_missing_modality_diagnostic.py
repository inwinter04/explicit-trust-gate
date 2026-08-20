#!/usr/bin/env python
"""Run R016 missing-modality diagnostic for the DeepURLBench pilot.

This script probes how the frozen-expert cross-attentive gate behaves when the
DNS/IP behavior view is synthetically removed at test time. Because R015 fails
the strict point-in-time provenance gate, this is diagnostic only and must not
be used as deployable C2 evidence.
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


def metric_row(seed: int, system: str, split: str, missing_rate: float, y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "seed": seed,
        "system": system,
        "split": split,
        "missing_rate": missing_rate,
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
    out["TTL"] = pd.to_numeric(out.get("TTL", np.nan), errors="coerce")
    out["has_dns"] = out.get("has_dns", False).astype(str).str.lower().isin(["true", "1"]).astype(int)
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


def missing_code_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["TTL"] = 0.0
    out["ttl_log1p"] = 0.0
    out["ttl_missing"] = 1
    for col in DNS_NUMERIC:
        if col not in {"TTL", "ttl_log1p", "ttl_missing"}:
            out[col] = 0.0
    return out


def apply_whole_view_missing(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    missing = missing_code_frame(out)
    out.loc[mask, DNS_NUMERIC] = missing.loc[mask, DNS_NUMERIC]
    return out


def fit_experts(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
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
        "lexical": lexical,
        "behavior": behavior,
        "p_lex_val": lexical.predict_proba(val["url"])[:, 1],
        "p_dns_val": behavior.predict_proba(val[DNS_NUMERIC])[:, 1],
        "p_lex_test": lexical.predict_proba(test["url"])[:, 1],
        "p_dns_test": behavior.predict_proba(test[DNS_NUMERIC])[:, 1],
        "p_dns_val_missing": behavior.predict_proba(missing_code_frame(val)[DNS_NUMERIC])[:, 1],
        "p_dns_test_missing": behavior.predict_proba(missing_code_frame(test)[DNS_NUMERIC])[:, 1],
    }


def clipped_logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def oracle_gate_target(y_true: np.ndarray, p_lex: np.ndarray, p_dns: np.ndarray) -> np.ndarray:
    lex_loss = -(y_true * np.log(np.clip(p_lex, 1e-7, 1.0)) + (1 - y_true) * np.log(np.clip(1 - p_lex, 1e-7, 1.0)))
    dns_loss = -(y_true * np.log(np.clip(p_dns, 1e-7, 1.0)) + (1 - y_true) * np.log(np.clip(1 - p_dns, 1e-7, 1.0)))
    return (lex_loss <= dns_loss).astype(np.float32)


def tune_constant_gate(y_true: np.ndarray, p_lex: np.ndarray, p_dns: np.ndarray) -> float:
    best_g = 0.0
    best_score = -float("inf")
    for g in np.linspace(0.0, 1.0, 101):
        score = float(average_precision_score(y_true, g * p_lex + (1 - g) * p_dns))
        if score > best_score:
            best_g = float(g)
            best_score = score
    return best_g


def behavior_matrix(frame: pd.DataFrame, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[DNS_NUMERIC].apply(pd.to_numeric, errors="coerce")
    values = frame[DNS_NUMERIC].apply(pd.to_numeric, errors="coerce")
    medians = train_values.median().fillna(0.0)
    means = train_values.mean().fillna(0.0)
    stds = train_values.std().replace(0, 1.0).fillna(1.0)
    missing = values.isna().to_numpy(dtype=np.float32)
    whole_view_missing = ((values["has_dns"].fillna(0) <= 0) | (values["ttl_missing"].fillna(0) >= 1)).to_numpy()
    missing[whole_view_missing, :] = 1.0
    scaled = ((values.fillna(medians) - means) / stds).clip(-8, 8).to_numpy(dtype=np.float32)
    return scaled, missing


class GateNet(nn.Module):
    def __init__(self, feature_count: int, d_model: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.token_projection = nn.Linear(2, d_model)
        self.type_embedding = nn.Embedding(feature_count, d_model)
        self.query_projection = nn.Linear(2, d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate_head = nn.Sequential(nn.Linear(2 + 3 * d_model, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, lex_logit: torch.Tensor, dns_logit: torch.Tensor, behavior: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        token = self.token_projection(torch.stack([behavior, missing], dim=-1)) + self.type_embedding.weight.unsqueeze(0)
        query = self.query_projection(torch.stack([lex_logit, dns_logit], dim=-1)).unsqueeze(1)
        attended, _ = self.attention(query, token, token, need_weights=False)
        gate_input = torch.cat([lex_logit.unsqueeze(1), dns_logit.unsqueeze(1), query[:, 0, :], self.norm(attended[:, 0, :]), token.mean(dim=1)], dim=1)
        return torch.sigmoid(self.gate_head(gate_input).squeeze(1))


def train_gate(
    inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    y_true: torch.Tensor,
    seed: int,
    device: torch.device,
) -> GateNet:
    seed_everything(seed)
    model = GateNet(len(DNS_NUMERIC)).to(device)
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


def eval_gate(model: GateNet, inputs: tuple[torch.Tensor, ...], p_lex: np.ndarray, p_dns: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        gate = model(*(item.to(device) for item in inputs)).cpu().numpy()
    return gate * p_lex + (1 - gate) * p_dns, gate


def make_inputs(p_lex: np.ndarray, p_dns: np.ndarray, behavior: np.ndarray, missing: np.ndarray) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(clipped_logit(p_lex), dtype=torch.float32),
        torch.tensor(clipped_logit(p_dns), dtype=torch.float32),
        torch.from_numpy(behavior.astype(np.float32)),
        torch.from_numpy(missing.astype(np.float32)),
    )


def md_table(frame: pd.DataFrame) -> str:
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-dns", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--without-dns", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_without_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--seeds", default="20260708,20260709,20260710")
    parser.add_argument("--missing-rates", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    missing_rates = [float(rate.strip()) for rate in args.missing_rates.split(",") if rate.strip()]

    with_dns = add_features(pd.read_csv(args.with_dns))
    without_dns = add_features(pd.read_csv(args.without_dns))
    train, val, test = split_parts(with_dns)
    _, _, without_test = split_parts(without_dns)
    experts = fit_experts(train, val, test)

    lexical_model: Pipeline = experts["lexical"]  # type: ignore[assignment]
    without_prob = lexical_model.predict_proba(without_test["url"])[:, 1]
    without_y = y(without_test)

    y_val_np = y(val)
    y_test = y(test)
    p_lex_val = experts["p_lex_val"]  # type: ignore[assignment]
    p_dns_val = experts["p_dns_val"]  # type: ignore[assignment]
    p_lex_test = experts["p_lex_test"]  # type: ignore[assignment]
    p_dns_test = experts["p_dns_test"]  # type: ignore[assignment]
    p_dns_test_missing = experts["p_dns_test_missing"]  # type: ignore[assignment]
    tuned_g = tune_constant_gate(y_val_np, p_lex_val, p_dns_val)
    target = torch.tensor(oracle_gate_target(y_val_np, p_lex_val, p_dns_val), dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.float32)

    val_behavior, val_missing = behavior_matrix(val, train)
    val_inputs = make_inputs(p_lex_val, p_dns_val, val_behavior, val_missing)

    rows = []
    diag_rows = []
    rng_cache = {seed: np.random.default_rng(seed) for seed in seeds}
    for seed in seeds:
        seed_everything(seed)
        model = train_gate(val_inputs, target, y_val_t, seed, device)
        for rate in missing_rates:
            rng = rng_cache[seed]
            mask = rng.random(len(test)) < rate
            masked_test = apply_whole_view_missing(test, mask)
            test_behavior, test_missing = behavior_matrix(masked_test, train)
            p_dns_mixed = p_dns_test.copy()
            p_dns_mixed[mask] = p_dns_test_missing[mask]
            inputs = make_inputs(p_lex_test, p_dns_mixed, test_behavior, test_missing)
            gate_prob, gate = eval_gate(model, inputs, p_lex_test, p_dns_mixed, device)
            systems = {
                "lexical_only": p_lex_test,
                "behavior_masked": p_dns_mixed,
                "fixed_average_masked": 0.5 * p_lex_test + 0.5 * p_dns_mixed,
                "validation_tuned_constant_masked": tuned_g * p_lex_test + (1 - tuned_g) * p_dns_mixed,
                "cross_attention_masked": gate_prob,
            }
            for name, prob in systems.items():
                rows.append(metric_row(seed, name, "with_dns_test_synthetic_missing", rate, y_test, prob))
            diag_rows.append(
                {
                    "seed": seed,
                    "missing_rate": rate,
                    "actual_missing_pct": float(mask.mean() * 100),
                    "mean_g_lex": float(gate.mean()),
                    "choose_lex_pct": float((gate >= 0.5).mean() * 100),
                    "masked_mean_g_lex": float(gate[mask].mean()) if mask.any() else float("nan"),
                    "unmasked_mean_g_lex": float(gate[~mask].mean()) if (~mask).any() else float("nan"),
                    "masked_behavior_prob_mean": float(p_dns_mixed[mask].mean()) if mask.any() else float("nan"),
                }
            )
        rows.append(metric_row(seed, "without_dns_lexical_fallback", "without_dns_test", 1.0, without_y, without_prob))

    metrics = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diag_rows)
    summary = (
        metrics.groupby(["split", "system", "missing_rate"], as_index=False)
        .agg(
            rows=("rows", "first"),
            positives=("positives", "first"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_std=("AUPRC", "std"),
            FPR95_mean=("FPR@95TPR", "mean"),
            ECE_mean=("ECE", "mean"),
        )
        .sort_values(["split", "missing_rate", "system"])
    )
    diag_summary = (
        diagnostics.groupby("missing_rate", as_index=False)
        .agg(
            actual_missing_pct=("actual_missing_pct", "mean"),
            mean_g_lex=("mean_g_lex", "mean"),
            choose_lex_pct=("choose_lex_pct", "mean"),
            masked_mean_g_lex=("masked_mean_g_lex", "mean"),
            unmasked_mean_g_lex=("unmasked_mean_g_lex", "mean"),
            masked_behavior_prob_mean=("masked_behavior_prob_mean", "mean"),
        )
        .sort_values("missing_rate")
    )
    slope_frame = summary[(summary["split"].eq("with_dns_test_synthetic_missing")) & (summary["system"].eq("cross_attention_masked"))]
    if len(slope_frame) >= 2:
        slope = float(np.polyfit(slope_frame["missing_rate"], slope_frame["AUPRC_mean"], 1)[0])
    else:
        slope = float("nan")
    for df in [summary, diag_summary]:
        for col in df.select_dtypes(include=[float]).columns:
            df[col] = df[col].round(4)

    stamp = utc_stamp()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / f"R016_MISSING_MODALITY_METRICS_{stamp}.csv"
    diagnostics_path = args.out_dir / f"R016_MISSING_MODALITY_DIAGNOSTICS_{stamp}.csv"
    report_path = args.out_dir / f"R016_MISSING_MODALITY_REPORT_{stamp}.md"
    metadata_path = args.out_dir / f"R016_MISSING_MODALITY_METADATA_{stamp}.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8")
    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "with_dns": str(args.with_dns),
        "with_dns_sha256": sha256_file(args.with_dns),
        "without_dns": str(args.without_dns),
        "without_dns_sha256": sha256_file(args.without_dns),
        "seeds": seeds,
        "missing_rates": missing_rates,
        "device": str(device),
        "evaluation_type": "real_gt diagnostic; synthetic whole-view behavior masking; not point-in-time C2 evidence because R015 failed",
        "cross_attention_auprc_slope_per_missing_rate": slope,
        "outputs": {"metrics_csv": str(metrics_path), "diagnostics_csv": str(diagnostics_path)},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(metrics_path, args.out_dir / "R016_MISSING_MODALITY_METRICS.csv")
    shutil.copyfile(diagnostics_path, args.out_dir / "R016_MISSING_MODALITY_DIAGNOSTICS.csv")
    shutil.copyfile(metadata_path, args.out_dir / "R016_MISSING_MODALITY_METADATA.json")

    report = "\n".join(
        [
            "# R016 Missing-Modality Diagnostic",
            "",
            f"Generated: {datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
            f"With-DNS sample: `{args.with_dns.as_posix()}`",
            f"Without-DNS sample: `{args.without_dns.as_posix()}`",
            f"Seeds: `{', '.join(map(str, seeds))}`",
            f"Device: `{device}`",
            "",
            "## Scope",
            "",
            "This is a diagnostic only. R015 currently fails the strict point-in-time provenance gate, so these results must not be used as deployable C2 evidence.",
            "",
            "Synthetic missingness replaces the entire DNS/IP behavior view with missing-coded features on a random fraction of with-DNS test rows. The gate was not trained with the final paper's `[MISS]` token or modality-dropout regime.",
            "",
            "## Metric Summary",
            "",
            md_table(summary),
            "",
            "## Gate Diagnostics",
            "",
            md_table(diag_summary),
            "",
            "## Sensitivity",
            "",
            f"- cross-attention AUPRC slope per unit missing rate: `{slope:.4f}`",
            "- More negative slope means stronger sensitivity to behavior-view removal.",
            "",
            "## Integrity Notes",
            "",
            "- Test metrics use dataset-provided labels.",
            "- Missingness is synthetically injected at test time and is not a natural deployment trace.",
            "- DeepURLBench DNS/IP timing remains unverified; this is dataset-provided DNS-context evidence only.",
            "- `without_dns_lexical_fallback` is evaluated on the separate without-DNS temporal sample and is not row-matched to the with-DNS test split.",
            "",
            "## Outputs",
            "",
            f"- metrics CSV: `{metrics_path.as_posix()}`",
            f"- diagnostics CSV: `{diagnostics_path.as_posix()}`",
            f"- metadata JSON: `{metadata_path.as_posix()}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    shutil.copyfile(report_path, args.out_dir / "R016_MISSING_MODALITY_REPORT.md")
    print(json.dumps({"report": str(args.out_dir / "R016_MISSING_MODALITY_REPORT.md"), "slope": slope}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
