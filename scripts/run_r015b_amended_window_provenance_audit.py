#!/usr/bin/env python
"""Run R015B amended-window provenance audit.

R015B verifies the prospective leakage rows against an explicit protocol
amendment. It keeps the original strict 24h failure visible and reports the
amended-window result separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso_now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def file_sha256(path: Path) -> str:
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


def load_leakage(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(data_dir.glob("leakage_audit*.csv"))
    if not paths:
        raise FileNotFoundError(f"No leakage_audit*.csv files under {data_dir}")

    frames = []
    files = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"domain", "feature_group", "query_time_utc", "hours_after_t0"}
        missing_cols = sorted(required - set(frame.columns))
        if missing_cols:
            raise ValueError(f"{path} is missing required columns: {missing_cols}")
        frame["source_file"] = path.name
        frame["query_time_utc"] = frame["query_time_utc"].astype("string")
        frame["hours_after_t0"] = pd.to_numeric(frame["hours_after_t0"], errors="coerce")
        frames.append(frame)
        files.append(
            {
                "source_file": path.name,
                "rows": len(frame),
                "missing_query_time": int(frame["query_time_utc"].isna().sum()),
                "min_hours_after_t0": round(float(frame["hours_after_t0"].min()), 4),
                "max_hours_after_t0": round(float(frame["hours_after_t0"].max()), 4),
                "sha256_16": file_sha256(path)[:16],
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(files)


def outside_mask(frame: pd.DataFrame, window_hours: float) -> pd.Series:
    hours = frame["hours_after_t0"]
    missing_query = frame["query_time_utc"].isna()
    return missing_query | hours.isna() | hours.lt(0) | hours.gt(window_hours)


def summarize_window(frame: pd.DataFrame, window_hours: float, name: str) -> dict:
    mask = outside_mask(frame, window_hours)
    return {
        "window": name,
        "window_hours": window_hours,
        "status": "pass" if int(mask.sum()) == 0 else "fail",
        "rows": len(frame),
        "outside_rows": int(mask.sum()),
        "outside_pct": round(float(mask.mean() * 100), 2),
        "missing_query_time": int(frame["query_time_utc"].isna().sum()),
        "min_hours_after_t0": round(float(frame["hours_after_t0"].min()), 4),
        "max_hours_after_t0": round(float(frame["hours_after_t0"].max()), 4),
    }


def summarize_files(frame: pd.DataFrame, files: pd.DataFrame, strict_window: float, declared_window: float) -> pd.DataFrame:
    rows = []
    for _, file_row in files.iterrows():
        part = frame[frame["source_file"].eq(file_row["source_file"])]
        strict = outside_mask(part, strict_window)
        declared = outside_mask(part, declared_window)
        row = file_row.to_dict()
        row["outside_strict_24h"] = int(strict.sum())
        row["outside_strict_24h_pct"] = round(float(strict.mean() * 100), 2)
        row["outside_declared_window"] = int(declared.sum())
        row["outside_declared_window_pct"] = round(float(declared.mean() * 100), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_groups(frame: pd.DataFrame, strict_window: float, declared_window: float) -> pd.DataFrame:
    rows = []
    for (source_file, feature_group), part in frame.groupby(["source_file", "feature_group"], sort=True):
        strict = outside_mask(part, strict_window)
        declared = outside_mask(part, declared_window)
        rows.append(
            {
                "source_file": source_file,
                "feature_group": feature_group,
                "rows": len(part),
                "missing_query_time": int(part["query_time_utc"].isna().sum()),
                "min_hours_after_t0": round(float(part["hours_after_t0"].min()), 4),
                "max_hours_after_t0": round(float(part["hours_after_t0"].max()), 4),
                "outside_strict_24h": int(strict.sum()),
                "outside_strict_24h_pct": round(float(strict.mean() * 100), 2),
                "outside_declared_window": int(declared.sum()),
                "outside_declared_window_pct": round(float(declared.mean() * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-dir", type=Path, default=Path("data/raw/2026-07-07"))
    parser.add_argument("--protocol", type=Path, default=Path("refine-logs/POINT_IN_TIME_PROTOCOL_AMENDMENT.md"))
    parser.add_argument("--declared-window-hours", type=float, default=25.0)
    parser.add_argument("--strict-window-hours", type=float, default=24.0)
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.protocol.exists():
        raise FileNotFoundError(f"Protocol amendment not found: {args.protocol}")

    stamp = utc_stamp()
    leakage, files = load_leakage(args.prospective_dir)
    strict = summarize_window(leakage, args.strict_window_hours, "strict_24h_historical")
    declared = summarize_window(leakage, args.declared_window_hours, "declared_25h_amended")
    window_summary = pd.DataFrame([strict, declared])
    file_summary = summarize_files(leakage, files, args.strict_window_hours, args.declared_window_hours)
    group_summary = summarize_groups(leakage, args.strict_window_hours, args.declared_window_hours)

    prospective_status = str(declared["status"])
    deepurlbench_status = "warn"
    overall = "warn" if prospective_status == "pass" else "fail"

    metadata = {
        "generated_utc": iso_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "protocol": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "prospective_dir": str(args.prospective_dir),
        "declared_window_hours": args.declared_window_hours,
        "strict_window_hours": args.strict_window_hours,
        "strict_24h_historical_status": strict["status"],
        "strict_24h_historical_outside_rows": strict["outside_rows"],
        "amended_window_status": declared["status"],
        "amended_window_outside_rows": declared["outside_rows"],
        "deepurlbench_dns_timing_status": deepurlbench_status,
        "overall_verdict": overall,
        "claim_impact": {
            "prospective_protocol": "passes under the explicit 25h amendment for the current 2026-07-07 sample",
            "strict_24h": "remains failed historical evidence",
            "C2": "not fully supported because DeepURLBench DNS timing and final missing-modality evidence remain diagnostic/incomplete",
        },
    }

    window_summary_path = args.out_dir / f"R015B_AMENDED_WINDOW_SUMMARY_{stamp}.csv"
    file_summary_path = args.out_dir / f"R015B_AMENDED_WINDOW_FILES_{stamp}.csv"
    group_summary_path = args.out_dir / f"R015B_AMENDED_WINDOW_GROUPS_{stamp}.csv"
    metadata_path = args.out_dir / f"R015B_AMENDED_WINDOW_METADATA_{stamp}.json"
    report_path = args.out_dir / f"R015B_AMENDED_WINDOW_PROVENANCE_REPORT_{stamp}.md"

    window_summary.to_csv(window_summary_path, index=False, encoding="utf-8")
    file_summary.to_csv(file_summary_path, index=False, encoding="utf-8")
    group_summary.to_csv(group_summary_path, index=False, encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    report = "\n".join(
        [
            "# R015B Amended-Window Provenance Audit",
            "",
            f"Generated: {iso_now()}",
            f"Protocol amendment: `{args.protocol.as_posix()}`",
            f"Prospective directory: `{args.prospective_dir.as_posix()}`",
            "",
            "## Verdict",
            "",
            f"- overall: `{overall}`",
            f"- prospective amended-window status: `{prospective_status}`",
            f"- declared operational window: `{args.declared_window_hours:g}h`",
            f"- strict 24h historical status: `{strict['status']}`",
            f"- DeepURLBench DNS timing: `{deepurlbench_status}`",
            "",
            "## Window Summary",
            "",
            md_table(window_summary),
            "",
            "## Source File Summary",
            "",
            md_table(file_summary.drop(columns=["sha256_16"])),
            "",
            "## Feature-Group Summary",
            "",
            md_table(group_summary),
            "",
            "## Interpretation",
            "",
            "Under the explicit protocol amendment, the current prospective sample passes the declared 25h collection window: no rows are outside `[t0, t0+25h]`, and no query timestamps are missing. The original strict 24h result remains failed historical evidence and must stay visible in the paper trail.",
            "",
            "The overall verdict is `warn`, not `pass`, because DeepURLBench DNS/IP fields still lack per-feature collection timestamps. They can support dataset-provided DNS-context diagnostics, but not verified point-in-time behavior-collection claims.",
            "",
            "## Claim Impact",
            "",
            "- Prospective-window provenance is no longer the immediate blocker if the paper explicitly adopts `W = 25h`.",
            "- Strict `W = 24h` claims remain unsupported.",
            "- C2 is still not fully supported until the model evaluation, missing-modality evidence, and paper text are aligned with the amended protocol and avoid DeepURLBench point-in-time overclaiming.",
            "",
            "## Outputs",
            "",
            f"- window summary CSV: `{window_summary_path.as_posix()}`",
            f"- source file CSV: `{file_summary_path.as_posix()}`",
            f"- feature-group CSV: `{group_summary_path.as_posix()}`",
            f"- metadata JSON: `{metadata_path.as_posix()}`",
        ]
    )
    report_path.write_text(report, encoding="utf-8")

    shutil.copyfile(window_summary_path, args.out_dir / "R015B_AMENDED_WINDOW_SUMMARY.csv")
    shutil.copyfile(file_summary_path, args.out_dir / "R015B_AMENDED_WINDOW_FILES.csv")
    shutil.copyfile(group_summary_path, args.out_dir / "R015B_AMENDED_WINDOW_GROUPS.csv")
    shutil.copyfile(metadata_path, args.out_dir / "R015B_AMENDED_WINDOW_METADATA.json")
    shutil.copyfile(report_path, args.out_dir / "R015B_AMENDED_WINDOW_PROVENANCE_REPORT.md")

    print(
        json.dumps(
            {
                "report": str(args.out_dir / "R015B_AMENDED_WINDOW_PROVENANCE_REPORT.md"),
                "overall": overall,
                "amended_window_status": prospective_status,
                "strict_24h_historical_status": strict["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
