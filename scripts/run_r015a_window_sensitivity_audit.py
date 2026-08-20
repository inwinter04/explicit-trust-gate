#!/usr/bin/env python
"""Run R015A window-sensitivity audit for prospective leakage rows.

This is a protocol audit for the failed R015 strict 24h point-in-time gate.
It does not rewrite R015: strict 24h remains failed if any row is outside
[t0, t0 + 24h]. R015A asks whether the observed failure is compatible with a
clearly re-declared wider operational follow-up window.
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


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


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


def parse_windows(value: str) -> list[float]:
    windows = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        window = float(item)
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        windows.append(window)
    if 24.0 not in windows:
        windows.insert(0, 24.0)
    return sorted(set(windows))


def format_window(window: float) -> str:
    return str(int(window)) if window.is_integer() else str(window).rstrip("0").rstrip(".")


def window_column(window: float, suffix: str) -> str:
    return f"{suffix}_{format_window(window).replace('.', 'p')}h"


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
                "min_hours_after_t0": float(frame["hours_after_t0"].min()),
                "max_hours_after_t0": float(frame["hours_after_t0"].max()),
                "sha256_16": file_sha256(path)[:16],
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(files)


def outside_mask(frame: pd.DataFrame, window: float) -> pd.Series:
    hours = frame["hours_after_t0"]
    missing_query = frame["query_time_utc"].isna()
    return missing_query | hours.isna() | hours.lt(0) | hours.gt(window)


def summarize_files(leakage: pd.DataFrame, files: pd.DataFrame, windows: list[float]) -> pd.DataFrame:
    rows = []
    for _, file_row in files.iterrows():
        part = leakage[leakage["source_file"].eq(file_row["source_file"])]
        row = file_row.to_dict()
        for window in windows:
            mask = outside_mask(part, window)
            row[window_column(window, "outside")] = int(mask.sum())
            row[window_column(window, "outside_pct")] = round(float(mask.mean() * 100), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ["min_hours_after_t0", "max_hours_after_t0"]:
        out[col] = out[col].round(4)
    return out


def summarize_groups(leakage: pd.DataFrame, windows: list[float]) -> pd.DataFrame:
    rows = []
    grouped = leakage.groupby(["source_file", "feature_group"], sort=True)
    for (source_file, feature_group), part in grouped:
        row = {
            "source_file": source_file,
            "feature_group": feature_group,
            "rows": len(part),
            "missing_query_time": int(part["query_time_utc"].isna().sum()),
            "min_hours_after_t0": round(float(part["hours_after_t0"].min()), 4),
            "max_hours_after_t0": round(float(part["hours_after_t0"].max()), 4),
        }
        for window in windows:
            mask = outside_mask(part, window)
            row[window_column(window, "outside")] = int(mask.sum())
            row[window_column(window, "outside_pct")] = round(float(mask.mean() * 100), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_file", "feature_group"])


def summarize_windows(leakage: pd.DataFrame, windows: list[float]) -> pd.DataFrame:
    rows = []
    for window in windows:
        mask = outside_mask(leakage, window)
        rows.append(
            {
                "window_hours": format_window(window),
                "status": "pass" if int(mask.sum()) == 0 else "fail",
                "rows": len(leakage),
                "outside_rows": int(mask.sum()),
                "outside_pct": round(float(mask.mean() * 100), 2),
                "missing_query_time": int(leakage["query_time_utc"].isna().sum()),
                "min_hours_after_t0": round(float(leakage["hours_after_t0"].min()), 4),
                "max_hours_after_t0": round(float(leakage["hours_after_t0"].max()), 4),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-dir", type=Path, default=Path("data/raw/2026-07-07"))
    parser.add_argument("--windows", default="24,24.5,25,26", help="Comma-separated hour windows to audit.")
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = parse_windows(args.windows)
    stamp = utc_stamp()

    leakage, files = load_leakage(args.prospective_dir)
    window_summary = summarize_windows(leakage, windows)
    file_summary = summarize_files(leakage, files, windows)
    group_summary = summarize_groups(leakage, windows)

    strict_24 = window_summary[window_summary["window_hours"].eq("24")].iloc[0].to_dict()
    first_passing = window_summary[window_summary["status"].eq("pass")]
    first_passing_window = None if first_passing.empty else str(first_passing.iloc[0]["window_hours"])
    overall = "partial" if first_passing_window is not None and strict_24["status"] == "fail" else str(strict_24["status"])

    metadata = {
        "generated_utc": iso_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "prospective_dir": str(args.prospective_dir),
        "source_files": files.to_dict(orient="records"),
        "windows_hours": windows,
        "strict_24h_status": str(strict_24["status"]),
        "strict_24h_outside_rows": int(strict_24["outside_rows"]),
        "first_passing_window_hours": first_passing_window,
        "overall_verdict": overall,
        "claim_impact": {
            "strict_24h": "failed historical evidence; do not describe existing R015 as passing strict 24h",
            "amended_window": "a wider window can only support C2 after the paper protocol explicitly declares it and downstream analyses use it consistently",
            "C2": "still not paper-supported until the amended provenance protocol and model-evaluation protocol are aligned",
        },
    }

    window_summary_path = args.out_dir / f"R015A_WINDOW_SENSITIVITY_SUMMARY_{stamp}.csv"
    file_summary_path = args.out_dir / f"R015A_WINDOW_SENSITIVITY_FILES_{stamp}.csv"
    group_summary_path = args.out_dir / f"R015A_WINDOW_SENSITIVITY_GROUPS_{stamp}.csv"
    metadata_path = args.out_dir / f"R015A_WINDOW_SENSITIVITY_METADATA_{stamp}.json"
    report_path = args.out_dir / f"R015A_WINDOW_SENSITIVITY_REPORT_{stamp}.md"

    window_summary.to_csv(window_summary_path, index=False, encoding="utf-8")
    file_summary.to_csv(file_summary_path, index=False, encoding="utf-8")
    group_summary.to_csv(group_summary_path, index=False, encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    claim_lines = [
        "- Strict 24h remains failed historical evidence and must not be described as passed.",
        "- If the protocol is amended to the first passing window, that is a protocol amendment, not a retroactive repair of R015.",
        "- C2 remains unsupported at paper level until the amended collection window is declared and all downstream claims/reports use the same window consistently.",
    ]
    if first_passing_window is None:
        claim_lines.insert(1, "- None of the audited wider windows passed; a new prospective collection is required.")
    else:
        claim_lines.insert(1, f"- The first audited passing window is `{first_passing_window}h`; this can motivate a re-declared operational follow-up window.")

    report = "\n".join(
        [
            "# R015A Window-Sensitivity Audit",
            "",
            f"Generated: {iso_now()}",
            f"Prospective directory: `{args.prospective_dir.as_posix()}`",
            f"Audited windows: `{', '.join(format_window(window) + 'h' for window in windows)}`",
            "",
            "## Verdict",
            "",
            f"- overall: `{overall}`",
            f"- strict 24h status: `{strict_24['status']}`",
            f"- strict 24h outside rows: {int(strict_24['outside_rows'])}",
            f"- first passing audited window: `{first_passing_window + 'h' if first_passing_window else 'none'}`",
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
            "R015A distinguishes a strict-window failure from a possible protocol-boundary mismatch. The already declared `[t0, t0+24h]` claim remains false for the current prospective DNS follow-up artifact. A wider window can only be used if the protocol is explicitly re-declared before paper claims are written.",
            "",
            "## Claim Impact",
            "",
            *claim_lines,
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

    shutil.copyfile(window_summary_path, args.out_dir / "R015A_WINDOW_SENSITIVITY_SUMMARY.csv")
    shutil.copyfile(file_summary_path, args.out_dir / "R015A_WINDOW_SENSITIVITY_FILES.csv")
    shutil.copyfile(group_summary_path, args.out_dir / "R015A_WINDOW_SENSITIVITY_GROUPS.csv")
    shutil.copyfile(metadata_path, args.out_dir / "R015A_WINDOW_SENSITIVITY_METADATA.json")
    shutil.copyfile(report_path, args.out_dir / "R015A_WINDOW_SENSITIVITY_REPORT.md")

    print(
        json.dumps(
            {
                "report": str(args.out_dir / "R015A_WINDOW_SENSITIVITY_REPORT.md"),
                "overall": overall,
                "strict_24h_status": strict_24["status"],
                "first_passing_window_hours": first_passing_window,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
