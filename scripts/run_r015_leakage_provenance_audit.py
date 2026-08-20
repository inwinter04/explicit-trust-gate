#!/usr/bin/env python
"""Run R015 leakage and provenance audit.

This is a C2 gate, not a performance benchmark. It checks whether the current
data artifacts can support point-in-time and missing-modality claims.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md_table(frame: pd.DataFrame) -> str:
    cols = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(lines)


def ymd_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_leakage(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(data_dir.glob("leakage_audit*.csv"))
    if not paths:
        raise FileNotFoundError(f"No leakage_audit*.csv files under {data_dir}")
    frames = []
    path_rows = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame["within_window_bool"] = frame["within_window"].astype(str).str.lower().eq("true")
        frame["hours_after_t0"] = pd.to_numeric(frame["hours_after_t0"], errors="coerce")
        frames.append(frame)
        path_rows.append(
            {
                "source_file": path.name,
                "rows": len(frame),
                "outside_24h": int((~frame["within_window_bool"]).sum()),
                "missing_query_time": int(frame["query_time_utc"].isna().sum()),
                "max_hours_after_t0": float(frame["hours_after_t0"].max()),
                "sha256": file_sha256(path)[:16],
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(path_rows)


def summarize_leakage(leakage: pd.DataFrame) -> pd.DataFrame:
    summary = (
        leakage.groupby(["source_file", "feature_group"], as_index=False)
        .agg(
            rows=("domain", "count"),
            outside_24h=("within_window_bool", lambda s: int((~s).sum())),
            max_hours_after_t0=("hours_after_t0", "max"),
        )
        .sort_values(["source_file", "feature_group"])
    )
    summary["outside_pct"] = (summary["outside_24h"] / summary["rows"] * 100).round(2)
    summary["max_hours_after_t0"] = summary["max_hours_after_t0"].round(4)
    return summary


def summarize_deepurlbench(path: Path, name: str) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    frame["first_seen"] = pd.to_datetime(frame["first_seen"], errors="coerce")
    frame["TTL"] = pd.to_numeric(frame.get("TTL", np.nan), errors="coerce")
    frame["has_dns_bool"] = frame.get("has_dns", False).astype(str).str.lower().isin(["true", "1"])
    split = (
        frame.groupby("split", as_index=False)
        .agg(
            rows=("url", "count"),
            malicious=("label", lambda s: int(s.eq("malicious").sum())),
            benign=("label", lambda s: int(s.eq("benign").sum())),
            first_seen_min=("first_seen", "min"),
            first_seen_max=("first_seen", "max"),
            ttl_missing=("TTL", lambda s: int(s.isna().sum())),
            has_dns_pct=("has_dns_bool", lambda s: float(s.mean() * 100)),
        )
        .sort_values("split")
    )
    split.insert(0, "sample", name)
    for col in ["first_seen_min", "first_seen_max"]:
        split[col] = split[col].astype(str)
    split["has_dns_pct"] = split["has_dns_pct"].round(2)
    facts = {
        "sample": name,
        "path": str(path),
        "rows": int(len(frame)),
        "url_unique": int(frame["url"].nunique()),
        "domain_unique": int(frame["domain"].nunique()),
        "ttl_missing": int(frame["TTL"].isna().sum()),
        "has_dns_true": int(frame["has_dns_bool"].sum()),
        "first_seen_min": str(frame["first_seen"].min()),
        "first_seen_max": str(frame["first_seen"].max()),
        "sha256": file_sha256(path),
    }
    return split, facts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-dir", type=Path, default=Path("data/raw/2026-07-07"))
    parser.add_argument("--with-dns", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_with_dns_time_sample.csv"))
    parser.add_argument("--without-dns", type=Path, default=Path("data/interim/deepurlbench/deepurlbench_without_dns_time_sample.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()

    leakage, leakage_files = load_leakage(args.prospective_dir)
    leakage_summary = summarize_leakage(leakage)
    total_rows = int(len(leakage))
    outside_rows = int((~leakage["within_window_bool"]).sum())
    missing_query_time = int(leakage["query_time_utc"].isna().sum())
    max_hours = float(leakage["hours_after_t0"].max())

    with_split, with_facts = summarize_deepurlbench(args.with_dns, "with_dns")
    without_split, without_facts = summarize_deepurlbench(args.without_dns, "without_dns")
    split_summary = pd.concat([with_split, without_split], ignore_index=True)

    with_urls = set(pd.read_csv(args.with_dns, usecols=["url"])["url"].astype(str))
    without_urls = set(pd.read_csv(args.without_dns, usecols=["url"])["url"].astype(str))
    overlap_urls = len(with_urls & without_urls)

    prospective_status = "pass" if outside_rows == 0 and missing_query_time == 0 else "fail"
    deepurlbench_status = "warn"
    overall = "fail" if prospective_status == "fail" else "warn"

    metadata = {
        "generated_utc": ymd_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "prospective": {
            "status": prospective_status,
            "rows": total_rows,
            "outside_24h_rows": outside_rows,
            "missing_query_time_rows": missing_query_time,
            "max_hours_after_t0": max_hours,
        },
        "deepurlbench": {
            "status": deepurlbench_status,
            "reason": "Dataset has first_seen and DNS fields but no per-feature collection timestamp; treat DNS/IP as dataset-provided context, not verified point-in-time evidence.",
            "with_dns": with_facts,
            "without_dns": without_facts,
            "with_without_url_overlap": overlap_urls,
        },
        "overall_verdict": overall,
        "claim_impact": {
            "C2": "unsupported until prospective leakage rows are inside the declared window and DeepURLBench DNS timing is either verified or kept diagnostic-only",
            "missing_modality": "with/without DNS samples are suitable for missing-modality diagnostics but not point-in-time deployment claims",
        },
    }

    leakage_files_path = args.out_dir / f"R015_LEAKAGE_FILES_{stamp}.csv"
    leakage_summary_path = args.out_dir / f"R015_LEAKAGE_SUMMARY_{stamp}.csv"
    split_summary_path = args.out_dir / f"R015_DEEPURLBENCH_SPLIT_SUMMARY_{stamp}.csv"
    metadata_path = args.out_dir / f"R015_LEAKAGE_PROVENANCE_METADATA_{stamp}.json"
    report_path = args.out_dir / f"R015_LEAKAGE_PROVENANCE_REPORT_{stamp}.md"

    leakage_files.to_csv(leakage_files_path, index=False, encoding="utf-8")
    leakage_summary.to_csv(leakage_summary_path, index=False, encoding="utf-8")
    split_summary.to_csv(split_summary_path, index=False, encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report = "\n".join(
        [
            "# R015 Leakage and Provenance Audit",
            "",
            f"Generated: {ymd_now()}",
            "",
            "## Verdict",
            "",
            f"- overall: `{overall}`",
            f"- prospective point-in-time window: `{prospective_status}`",
            f"- DeepURLBench DNS timing: `{deepurlbench_status}`",
            "",
            "## Prospective Point-in-Time Check",
            "",
            f"- leakage rows checked: {total_rows}",
            f"- rows outside declared 24h window: {outside_rows}",
            f"- rows missing query time: {missing_query_time}",
            f"- max hours after t0: {max_hours:.4f}",
            "",
            "### Leakage Files",
            "",
            md_table(leakage_files.drop(columns=["sha256"])),
            "",
            "### Leakage by Feature Group",
            "",
            md_table(leakage_summary),
            "",
            "## DeepURLBench Provenance Check",
            "",
            "DeepURLBench contains `first_seen` and DNS/IP fields, but the local artifacts do not include per-feature collection timestamps. These fields can support dataset-provided DNS-context experiments and missing-modality diagnostics; they cannot by themselves prove point-in-time behavior collection.",
            "",
            f"- with-DNS sample rows: {with_facts['rows']}",
            f"- without-DNS sample rows: {without_facts['rows']}",
            f"- URL overlap between current with/without DNS samples: {overlap_urls}",
            "",
            "### Split Summary",
            "",
            md_table(split_summary),
            "",
            "## Claim Impact",
            "",
            "- C2 is not supported yet. The prospective audit fails the strict 24h window, and DeepURLBench DNS timing remains unverified.",
            "- R016/R017 can proceed as missing-modality diagnostics using the with/without DNS split, but reports must avoid deployability or verified point-in-time wording.",
            "- Before paper-level C2, fix or re-declare the prospective collection window and rerun the leakage audit.",
            "",
            "## Outputs",
            "",
            f"- leakage files CSV: `{leakage_files_path.as_posix()}`",
            f"- leakage summary CSV: `{leakage_summary_path.as_posix()}`",
            f"- DeepURLBench split summary CSV: `{split_summary_path.as_posix()}`",
            f"- metadata JSON: `{metadata_path.as_posix()}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    shutil.copyfile(leakage_files_path, args.out_dir / "R015_LEAKAGE_FILES.csv")
    shutil.copyfile(leakage_summary_path, args.out_dir / "R015_LEAKAGE_SUMMARY.csv")
    shutil.copyfile(split_summary_path, args.out_dir / "R015_DEEPURLBENCH_SPLIT_SUMMARY.csv")
    shutil.copyfile(metadata_path, args.out_dir / "R015_LEAKAGE_PROVENANCE_METADATA.json")
    shutil.copyfile(report_path, args.out_dir / "R015_LEAKAGE_PROVENANCE_REPORT.md")

    print(json.dumps({"report": str(args.out_dir / "R015_LEAKAGE_PROVENANCE_REPORT.md"), "overall": overall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
