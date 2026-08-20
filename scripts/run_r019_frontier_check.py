#!/usr/bin/env python
"""Run R019 frontier check: DomURLs_BERT vs from-scratch char-CNN.

This is a lexical frontier comparison on the local DeepURLBench time sample.
It uses a rule-defined hard lexical subset and compares:

- frozen DomURLs_BERT embeddings + logistic regression;
- from-scratch char-CNN on domain strings.

The hard subset is independent of model predictions: low digit ratio, no
hyphen, short SLD, few dots, and short domain length.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DOMURLS_BERT = "amahdaouy/DomURLs_BERT"


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None


def fpr_at_tpr(y_true: np.ndarray, prob: np.ndarray, target_tpr: float = 0.95) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, prob)
    valid = fpr[tpr >= target_tpr]
    return float(valid.min()) if len(valid) else 1.0


def ece_score(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1.0 else prob <= high)
        if mask.any():
            total += float(mask.mean()) * abs(float(prob[mask].mean()) - float(y_true[mask].mean()))
    return total


def metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "AUPRC": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "FPR@95TPR": fpr_at_tpr(y_true, prob),
        "macro_F1": float(f1_score(y_true, pred, average="macro")) if len(np.unique(y_true)) == 2 else float("nan"),
        "ECE": ece_score(y_true, prob),
        "recall@0.5": float(((pred == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())),
    }
    return out


def hard_subset_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["digit_ratio"] <= 0.15)
        & (~frame["has_hyphen"])
        & (frame["sld_len"] <= 24)
        & (frame["num_dots"] <= 2)
        & (frame["domain_len"] <= 25)
    )


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["first_seen"] = pd.to_datetime(out["first_seen"], errors="coerce")
    out["domain"] = out["domain"].fillna("").astype(str).str.lower()
    out["url"] = out["url"].fillna("").astype(str).str.lower()
    out["split"] = out["split"].astype(str)
    out["label_int"] = out["label"].eq("malicious").astype(int)
    out["sld"] = out["domain"].map(lambda value: value.split(".")[0] if value else "")
    out["sld_len"] = out["sld"].str.len()
    out["domain_len"] = out["domain"].str.len()
    out["num_dots"] = out["domain"].str.count(r"\.")
    out["digit_ratio"] = out["sld"].map(lambda value: sum(ch.isdigit() for ch in value) / max(1, len(value)))
    out["has_hyphen"] = out["sld"].str.contains("-", regex=False)
    return out[out["label"].isin(["benign", "malicious"]) & out["first_seen"].notna()].copy()


def split_parts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        frame[frame["split"].eq("train")].copy(),
        frame[frame["split"].eq("val")].copy(),
        frame[frame["split"].eq("test")].copy(),
    )


def make_char_vocab(texts: list[str]) -> dict[str, int]:
    chars = Counter()
    for text in texts:
        chars.update(text)
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for ch, _ in chars.most_common():
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    encoded = np.zeros((len(texts), max_len), dtype=np.int64)
    unk = vocab["[UNK]"]
    for row, text in enumerate(texts):
        ids = [vocab.get(ch, unk) for ch in text[:max_len]]
        encoded[row, : len(ids)] = ids
    return encoded


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 48, channels: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, channels, kernel_size=k) for k in (3, 4, 5)]
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 3, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            feat = torch.relu(conv(emb))
            pooled.append(torch.max(feat, dim=2).values)
        out = torch.cat(pooled, dim=1)
        return self.head(out).squeeze(1)


def train_char_cnn(
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
    val_labels: np.ndarray,
    seed: int,
    max_len: int = 128,
    epochs: int = 8,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    seed_everything(seed)
    vocab = make_char_vocab(train_texts)
    x_train = torch.from_numpy(encode_texts(train_texts, vocab, max_len))
    y_train = torch.from_numpy(train_labels.astype(np.float32))
    x_val = torch.from_numpy(encode_texts(val_texts, vocab, max_len))
    y_val = torch.from_numpy(val_labels.astype(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharCNN(len(vocab)).to(device)
    pos = float(train_labels.sum())
    neg = float(len(train_labels) - pos)
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best_state = None
    best_val = -math.inf
    patience = 0
    history: dict[str, float] = {}

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x_val.to(device))).cpu().numpy()
        val_ap = float(average_precision_score(val_labels, val_prob))
        history[f"epoch_{epoch+1}_val_auprc"] = val_ap
        if val_ap > best_val + 1e-5:
            best_val = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(x_train.to(device))).cpu().numpy()
        val_prob = torch.sigmoid(model(x_val.to(device))).cpu().numpy()
    history["best_val_auprc"] = best_val
    return train_prob, val_prob, history, model, vocab


def predict_char_cnn(model: CharCNN, vocab: dict[str, int], texts: list[str], max_len: int = 128) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.from_numpy(encode_texts(texts, vocab, max_len)).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).cpu().numpy()
    return prob


def load_domurls_embeddings(texts: list[str], cache_dir: Path | None = None) -> tuple[np.ndarray | None, str]:
    try:
        import torch as _torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        return None, f"transformers/torch import failed: {exc}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(DOMURLS_BERT, cache_dir=str(cache_dir) if cache_dir else None)
        model = AutoModel.from_pretrained(DOMURLS_BERT, cache_dir=str(cache_dir) if cache_dir else None)
        model.eval()
        model.to("cpu")
        vectors: list[np.ndarray] = []
        with _torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = texts[start : start + 32]
                tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                output = model(**tokens)
                cls = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
                vectors.append(cls)
        return np.vstack(vectors), "ok"
    except Exception as exc:
        return None, f"DomURLs_BERT embedding failed: {type(exc).__name__}: {exc}"


def fit_embedding_logreg(embeddings: np.ndarray, y_train: np.ndarray, seed: int) -> LogisticRegression:
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed)
    clf.fit(embeddings, y_train)
    return clf


def subset_metrics(frame: pd.DataFrame, prob: np.ndarray, label: str) -> dict[str, float | int | str]:
    y_true = frame["label_int"].to_numpy()
    return {
        "subset": label,
        "rows": len(frame),
        "positives": int(y_true.sum()),
        **metrics(y_true, prob),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_without_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--hf-cache", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    frame = add_features(pd.read_csv(args.data))
    train, val, test = split_parts(frame)
    hard_mask = hard_subset_mask(test)
    hard = test[hard_mask].copy()

    train_texts = train["domain"].tolist()
    val_texts = val["domain"].tolist()
    test_texts = test["domain"].tolist()
    hard_texts = hard["domain"].tolist()

    # From-scratch char-CNN
    _, _, history, char_model, vocab = train_char_cnn(
        train_texts,
        train["label_int"].to_numpy(),
        val_texts,
        val["label_int"].to_numpy(),
        args.seed,
    )
    char_test_prob = predict_char_cnn(char_model, vocab, test_texts)
    char_hard_prob = predict_char_cnn(char_model, vocab, hard_texts) if len(hard_texts) else np.array([])

    # DomURLs_BERT frozen embeddings
    domurls_embeddings, domurls_status = load_domurls_embeddings(train_texts + val_texts + test_texts, args.hf_cache)
    domurls_model = None
    domurls_probs: dict[str, np.ndarray] = {}
    if domurls_embeddings is not None:
        train_n = len(train_texts)
        val_n = len(val_texts)
        train_emb = domurls_embeddings[:train_n]
        val_emb = domurls_embeddings[train_n : train_n + val_n]
        test_emb = domurls_embeddings[train_n + val_n :]
        domurls_model = fit_embedding_logreg(train_emb, train["label_int"].to_numpy(), args.seed)
        domurls_probs["test"] = domurls_model.predict_proba(test_emb)[:, 1]
        domurls_probs["hard"] = (
            domurls_model.predict_proba(test_emb[hard_mask.to_numpy()])[:, 1]
            if len(hard_texts)
            else np.array([])
        )
        # val embeddings used only to keep the split aligned in case future tuning is added.
        _ = val_emb

    results = []
    results.append(
        {
            "system": "char_cnn",
            "split": "test",
            **subset_metrics(test, char_test_prob, "full_test"),
        }
    )
    if len(hard):
        results.append(
            {
                "system": "char_cnn",
                "split": "test",
                **subset_metrics(hard, char_hard_prob, "hard_lex_benign_shape"),
            }
        )
    if domurls_embeddings is not None and domurls_model is not None:
        results.append(
            {
                "system": "domurls_bert_frozen_embedding",
                "split": "test",
                **subset_metrics(test, domurls_probs["test"], "full_test"),
            }
        )
        if len(hard):
            results.append(
                {
                    "system": "domurls_bert_frozen_embedding",
                    "split": "test",
                    **subset_metrics(hard, domurls_probs["hard"], "hard_lex_benign_shape"),
                }
            )

    metrics_df = pd.DataFrame(results)
    stamp = utc_stamp()
    metrics_path = args.out_dir / f"R019_FRONTIER_CHECK_METRICS_{stamp}.csv"
    report_path = args.out_dir / f"R019_FRONTIER_CHECK_REPORT_{stamp}.md"
    metadata_path = args.out_dir / f"R019_FRONTIER_CHECK_METADATA_{stamp}.json"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    metadata = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "script": str(Path(__file__).resolve()),
        "data": str(args.data),
        "seed": args.seed,
        "domurls_status": domurls_status,
        "history": history,
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "hard_subset_rows": int(len(hard)),
        "hard_subset_positives": int(hard["label_int"].sum()) if len(hard) else 0,
        "notes": [
            "Hard subset is rule-defined and independent of model predictions.",
            "DomURLs_BERT result is frozen embeddings + logistic regression, not full fine-tuning.",
            "Char-CNN is from-scratch on domain strings only.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# R019 Frontier Check",
        "",
        f"Generated: {metadata['generated_utc']}",
        f"Data: `{args.data.as_posix()}`",
        f"Hard subset: `lex_benign_shape` on test split, rows={len(hard)}",
        f"DomURLs_BERT status: `{domurls_status}`",
        "",
        "## Metrics",
        "",
        "| system | subset | rows | positives | AUPRC | FPR@95TPR | macro_F1 | ECE | recall@0.5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        report_lines.append(
            f"| {row['system']} | {row['subset']} | {row['rows']} | {row['positives']} | {row['AUPRC']:.4f} | {row['FPR@95TPR']:.4f} | {row['macro_F1']:.4f} | {row['ECE']:.4f} | {row['recall@0.5']:.4f} |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "- The hard subset is rule-defined from domain shape only; it does not depend on model predictions.",
        "- This comparison is a frontier check for the lexical backbone, not a gate or behavior experiment.",
        "- If DomURLs_BERT beats the from-scratch char-CNN on the hard subset, that supports the pretrained lexical expert necessity claim on this checkpoint.",
        "",
        "## Outputs",
        "",
        f"- metrics CSV: `{metrics_path.as_posix()}`",
        f"- metadata JSON: `{metadata_path.as_posix()}`",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    shutil.copyfile(metrics_path, args.out_dir / "R019_FRONTIER_CHECK_METRICS.csv")
    shutil.copyfile(report_path, args.out_dir / "R019_FRONTIER_CHECK_REPORT.md")
    shutil.copyfile(metadata_path, args.out_dir / "R019_FRONTIER_CHECK_METADATA.json")
    print(json.dumps({"report": str(args.out_dir / "R019_FRONTIER_CHECK_REPORT.md"), "domurls_status": domurls_status, "hard_rows": len(hard)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
