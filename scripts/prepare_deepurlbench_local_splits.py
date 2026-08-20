#!/usr/bin/env python
"""Create time-split local DeepURLBench samples from parquet directories.

The raw DeepURLBench parquet data is large. This script avoids per-row Python
streaming over all 40M rows. It reads one parquet part at a time, samples a
small candidate pool per year/label in that file, then performs a second-stage
balanced sample per year/label across all candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq


DATASETS = {
    "with_dns": "urls_with_dns",
    "without_dns": "urls_without_dns",
}


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def normalize_label(label: object) -> str:
    value = "" if label is None else str(label).strip().lower()
    if value == "benign":
        return "benign"
    if value in {"mal", "malware", "phishing"}:
        return "malicious"
    return "unknown"


def normalize_url(url: object) -> str:
    return "" if url is None else str(url).strip()


def normalize_domain(url: object) -> str:
    text = normalize_url(url)
    parsed = urlparse(text if "://" in text else "http://" + text)
    return (parsed.hostname or "").strip(".").lower()


def safe_tld(domain: str) -> str:
    if "." not in domain:
        return "[NONE]"
    return domain.rsplit(".", 1)[-1] or "[NONE]"


def ip_values(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        if converted is None:
            return []
        return [converted]
    return []


def ip_count(value: object) -> int:
    values = ip_values(value)
    if values:
        return len(values)
    if value is None:
        return 0
    try:
        return len(value)
    except Exception:
        return 0


def ip_octets(ip: object) -> tuple[int, int, int, int] | None:
    text = "" if ip is None else str(ip).strip()
    parts = text.split(".")
    if len(parts) != 4:
        return None
    try:
        octets = tuple(int(part) for part in parts)
    except ValueError:
        return None
    if any(part < 0 or part > 255 for part in octets):
        return None
    return octets


def ip_is_private_or_special(octets: tuple[int, int, int, int]) -> bool:
    first, second, *_ = octets
    return (
        first == 10
        or first == 127
        or (first == 172 and 16 <= second <= 31)
        or (first == 192 and second == 168)
        or (first == 169 and second == 254)
        or first == 0
    )


def ip_features(value: object) -> dict:
    ips = ip_values(value)
    parsed = [octets for octets in (ip_octets(ip) for ip in ips) if octets is not None]
    unique_ips = sorted(set(parsed))
    first_octets = [octets[0] for octets in unique_ips]
    return {
        "ip_unique_count": len(unique_ips),
        "ip_prefix24_count": len({octets[:3] for octets in unique_ips}),
        "ip_prefix16_count": len({octets[:2] for octets in unique_ips}),
        "ip_first_octet_min": min(first_octets) if first_octets else 0,
        "ip_first_octet_max": max(first_octets) if first_octets else 0,
        "ip_first_octet_mean": round(sum(first_octets) / len(first_octets), 4) if first_octets else 0.0,
        "ip_private_special_count": sum(1 for octets in unique_ips if ip_is_private_or_special(octets)),
        "ip_address_json": json.dumps([".".join(f"{part:03d}" for part in octets) for octets in unique_ips], ensure_ascii=False),
    }


def list_parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".parquet":
        return [path]
    return sorted(path.rglob("*.parquet"))


def normalize_label_series(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    return values.map(
        {
            "benign": "benign",
            "mal": "malicious",
            "malware": "malicious",
            "phishing": "malicious",
        }
    ).fillna("unknown")


def load_frame(file: Path, subset: str) -> pd.DataFrame:
    parquet = pq.ParquetFile(file)
    schema_names = set(parquet.schema_arrow.names)
    columns = ["url", "first_seen", "label"]
    if "TTL" in schema_names:
        columns.append("TTL")
    if "ip_address" in schema_names:
        columns.append("ip_address")

    frame = pq.read_table(file, columns=columns).to_pandas()
    if "TTL" not in frame.columns:
        frame["TTL"] = None
    if "ip_address" not in frame.columns:
        frame["ip_address"] = None
    frame["subset"] = subset
    frame["label_norm"] = normalize_label_series(frame["label"])
    frame["first_seen_text"] = frame["first_seen"].astype(str)
    frame["year"] = frame["first_seen_text"].str.slice(0, 4)
    frame = frame[
        frame["label_norm"].isin(["benign", "malicious"])
        & frame["year"].str.match(r"^\d{4}$", na=False)
        & frame["url"].notna()
    ].copy()
    return frame


def row_from_series(raw: pd.Series, subset: str) -> dict:
    url = normalize_url(raw.get("url", ""))
    domain = normalize_domain(url)
    ttl = raw.get("TTL", "")
    ips = raw.get("ip_address", [])
    label = str(raw.get("label_norm", normalize_label(raw.get("label", ""))))
    ip_stats = ip_features(ips)
    return {
        "subset": subset,
        "split": "",
        "url": url,
        "domain": domain,
        "first_seen": str(raw.get("first_seen_text", raw.get("first_seen", ""))),
        "year": str(raw.get("year", "")),
        "raw_label": "" if raw.get("label") is None else str(raw.get("label")),
        "label": label,
        "TTL": "" if ttl is None else ttl,
        "ip_count": ip_count(ips),
        **ip_stats,
        "has_dns": bool(ttl is not None and ip_count(ips) > 0),
        "url_len": len(url),
        "domain_len": len(domain),
        "num_dots": domain.count("."),
        "tld": safe_tld(domain),
    }


def sample_subset(data_dir: Path, subset: str, per_year_label: int, seed: int, file_bucket_cap: int) -> tuple[list[dict], dict]:
    path = data_dir / DATASETS[subset]
    files = list_parquet_files(path)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")

    subset_seed_offset = {"with_dns": 101, "without_dns": 202}[subset]
    seen: Counter[tuple[str, str]] = Counter()
    candidates: list[pd.DataFrame] = []
    skipped = Counter()

    for file_idx, file in enumerate(files):
        frame = load_frame(file, subset)
        for key, count in frame.groupby(["year", "label_norm"], observed=True).size().items():
            seen[(str(key[0]), str(key[1]))] += int(count)
        if frame.empty:
            skipped["empty_file_after_filter"] += 1
            continue
        sampled_parts: list[pd.DataFrame] = []
        for group_idx, ((year, label), group) in enumerate(frame.groupby(["year", "label_norm"], observed=True)):
            part = group.sample(
                n=min(len(group), file_bucket_cap),
                random_state=seed + subset_seed_offset + file_idx + group_idx,
            ).copy()
            part["year"] = str(year)
            part["label_norm"] = str(label)
            sampled_parts.append(part)
        if sampled_parts:
            candidates.append(pd.concat(sampled_parts, ignore_index=True))

    if not candidates:
        return [], {"subset": subset, "files": len(files), "seen": {}, "kept": 0, "skipped": dict(skipped)}

    candidate_frame = pd.concat(candidates, ignore_index=True)
    final_parts: list[pd.DataFrame] = []
    for idx, (_, group) in enumerate(candidate_frame.groupby(["year", "label_norm"], observed=True)):
        final_parts.append(
            group.sample(
                n=min(len(group), per_year_label),
                random_state=seed + subset_seed_offset + 10_000 + idx,
            )
        )
    final_frame = pd.concat(final_parts, ignore_index=True)
    rows = [row_from_series(row, subset) for _, row in final_frame.iterrows()]
    rows = [row for row in rows if row["domain"]]
    rows.sort(key=lambda item: (item["first_seen"], item["url"]))
    return rows, {
        "subset": subset,
        "files": len(files),
        "file_bucket_cap": file_bucket_cap,
        "candidate_rows": len(candidate_frame),
        "seen": {f"{year}:{label}": count for (year, label), count in sorted(seen.items())},
        "kept": len(rows),
        "skipped": dict(skipped),
    }


def assign_temporal_split(rows: list[dict]) -> None:
    rows.sort(key=lambda item: (item["first_seen"], item["url"]))
    n = len(rows)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    for idx, row in enumerate(rows):
        if idx < train_end:
            row["split"] = "train"
        elif idx < val_end:
            row["split"] = "val"
        else:
            row["split"] = "test"


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "subset",
        "split",
        "url",
        "domain",
        "first_seen",
        "year",
        "raw_label",
        "label",
        "TTL",
        "ip_count",
        "ip_unique_count",
        "ip_prefix24_count",
        "ip_prefix16_count",
        "ip_first_octet_min",
        "ip_first_octet_max",
        "ip_first_octet_mean",
        "ip_private_special_count",
        "ip_address_json",
        "has_dns",
        "url_len",
        "domain_len",
        "num_dots",
        "tld",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_summary(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for split in ["train", "val", "test"]:
        part = [row for row in rows if row["split"] == split]
        labels = Counter(row["label"] for row in part)
        has_dns = sum(1 for row in part if str(row["has_dns"]).lower() == "true")
        summary.append(
            {
                "split": split,
                "rows": len(part),
                "benign": labels.get("benign", 0),
                "malicious": labels.get("malicious", 0),
                "first_seen_min": part[0]["first_seen"] if part else "",
                "first_seen_max": part[-1]["first_seen"] if part else "",
                "has_dns_pct": round(has_dns * 100 / len(part), 2) if part else 0.0,
            }
        )
    return summary


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/deepurlbench"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim/deepurlbench"))
    parser.add_argument("--report-dir", type=Path, default=Path("refine-logs"))
    parser.add_argument("--per-year-label", type=int, default=1500)
    parser.add_argument("--file-bucket-cap", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = utc_stamp()

    all_rows: list[dict] = []
    subset_reports: list[dict] = []
    report_blocks: list[str] = []
    for subset in DATASETS:
        rows, report = sample_subset(
            args.data_dir,
            subset,
            args.per_year_label,
            args.seed,
            args.file_bucket_cap,
        )
        assign_temporal_split(rows)
        subset_path = args.out_dir / f"deepurlbench_{subset}_time_sample_{run_stamp}.csv"
        subset_latest = args.out_dir / f"deepurlbench_{subset}_time_sample.csv"
        write_csv(subset_path, rows)
        shutil.copyfile(subset_path, subset_latest)
        all_rows.extend(rows)
        subset_reports.append(
            {
                "subset": subset,
                "rows": len(rows),
                "csv": subset_latest.as_posix(),
                "timestamped_csv": subset_path.as_posix(),
            }
        )
        report_blocks.extend(
            [
                f"### {subset}",
                "",
                md_table(split_summary(rows), ["split", "rows", "benign", "malicious", "first_seen_min", "first_seen_max", "has_dns_pct"]),
                "",
                "<details><summary>bucket counts</summary>",
                "",
                "```json",
                json.dumps(report, ensure_ascii=False, indent=2),
                "```",
                "",
                "</details>",
                "",
            ]
        )

    all_rows.sort(key=lambda item: (item["subset"], item["first_seen"], item["url"]))
    combined_path = args.out_dir / f"deepurlbench_time_sample_{run_stamp}.csv"
    combined_latest = args.out_dir / "deepurlbench_time_sample.csv"
    write_csv(combined_path, all_rows)
    shutil.copyfile(combined_path, combined_latest)

    report = "\n".join(
        [
            "# DeepURLBench Local Time Sample",
            "",
            f"Generated: {datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
            f"Data dir: `{args.data_dir.as_posix()}`",
            f"Sampling: parquet-file candidate sampling, then cap `{args.per_year_label}` rows per `subset x year x label`.",
            "",
            "## Outputs",
            "",
            md_table(subset_reports, ["subset", "rows", "csv", "timestamped_csv"]),
            "",
            f"- combined latest CSV: `{combined_latest.as_posix()}`",
            f"- combined timestamped CSV: `{combined_path.as_posix()}`",
            "",
            "## Temporal Splits",
            "",
            *report_blocks,
            "## Notes",
            "",
            "- This sample is for CPU baselines and pipeline validation, not the final paper table.",
            "- The split is temporal inside each subset: earliest 60% train, next 20% validation, latest 20% test.",
            "- `with_dns` supports behavior and fusion baselines; `without_dns` supports missing-modality and lexical-only checks.",
        ]
    )
    report_path = args.report_dir / f"DEEPURLBENCH_LOCAL_TIME_SAMPLE_{run_stamp}.md"
    latest_report = args.report_dir / "DEEPURLBENCH_LOCAL_TIME_SAMPLE.md"
    report_path.write_text(report, encoding="utf-8")
    shutil.copyfile(report_path, latest_report)
    print(f"[done] Wrote {latest_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
