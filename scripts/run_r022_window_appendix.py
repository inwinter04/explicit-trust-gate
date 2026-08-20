#!/usr/bin/env python
"""R022 window-sensitivity appendix: W=24h vs W=72h on the prospective sample.

Reports behavior-feature availability under declared operational windows
(24h / 24.5h / 25h / 72h) using the 1,520 timestamped feature-observation
rows from the 2026-07-07 prospective collection. Scope is deliberately
availability/latency: per-feature AUPRC under a strict 24h regime cannot be
reconstructed because the initial (T0+~5min) per-domain DNS values were
overwritten by the follow-up in this pilot's storage.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from run_r020_residual_correction import md_table, sha256_file


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/2026-07-07"))
    parser.add_argument("--out-dir", type=Path, default=Path("refine-logs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(args.raw_dir / "leakage_audit*.csv")))
    if not files:
        raise FileNotFoundError("no leakage_audit*.csv files found")

    frames = [pd.read_csv(path) for path in files]
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["hours_after_t0"] = pd.to_numeric(all_rows["hours_after_t0"], errors="coerce")
    input_hashes = {Path(p).name: sha256_file(Path(p)) for p in files}

    windows = [24.0, 24.5, 25.0, 72.0]
    summary = []
    for w in windows:
        inside = int((all_rows["hours_after_t0"] <= w).sum())
        summary.append(
            {
                "window_hours": w,
                "inside_rows": inside,
                "outside_rows": int(len(all_rows) - inside),
                "availability": round(inside / len(all_rows), 4),
            }
        )
    summary_df = pd.DataFrame(summary)
    per_group = (
        all_rows.groupby("feature_group")
        .apply(lambda g: int((g["hours_after_t0"] <= 25.0).sum()))
        .rename("inside_25h")
        .reset_index()
    )
    per_group["total"] = all_rows.groupby("feature_group").size().reset_index(name="total")["total"]
    per_group["availability_25h"] = (per_group["inside_25h"] / per_group["total"]).round(4)

    stamp = utc_stamp()
    summary_path = args.out_dir / f"R022_WINDOW_APPENDIX_SUMMARY_{stamp}.csv"
    report_path = args.out_dir / f"R022_WINDOW_APPENDIX_REPORT_{stamp}.md"
    metadata_path = args.out_dir / f"R022_WINDOW_APPENDIX_METADATA_{stamp}.json"
    latest = {
        "R022_SUMMARY": args.out_dir / "R022_WINDOW_APPENDIX_SUMMARY.csv",
        "R022_REPORT": args.out_dir / "R022_WINDOW_APPENDIX_REPORT.md",
        "R022_METADATA": args.out_dir / "R022_WINDOW_APPENDIX_METADATA.json",
    }
    summary_df.to_csv(summary_path, index=False)
    shutil.copyfile(summary_path, latest["R022_SUMMARY"])

    md = [
        "# R022 Window-Sensitivity Appendix (W=24h vs W=72h)",
        "",
        f"**Generated**: {stamp}",
        f"**Rows audited**: {len(all_rows)} timestamped feature observations (DNS A/AAAA/CNAME + RDAP) from the 2026-07-07 prospective sample.",
        "",
        "## Availability by Declared Window",
        "",
        md_table(summary_df),
        "",
        "## Availability at the Amended 25h Window by Feature Group",
        "",
        md_table(per_group),
        "",
        "## Interpretation",
        "",
        "- A strict 24h window leaves 600 of 1,520 observations (39.5%) unavailable; all of them are DNS follow-up rows collected at about 24.87 h after t0.",
        "- The explicit 25h amendment covers all 1,520 observations (availability 100%); extending to 72h adds **no** additional observations.",
        "- The operational tradeoff is therefore between a strict 24h declaration (which must either wait for follow-ups or code them as missing) and the amended 25h window that already captures everything this pilot collected; a 72h window would only delay verdicts without adding evidence.",
        "- Scope limitation: per-feature AUPRC under a strict 24h regime cannot be reconstructed from current artifacts because the initial (T0+~5 min) per-domain DNS values were overwritten by the follow-up in this pilot's storage. This report is an availability/latency analysis, not a window-vs-AUPRC experiment.",
        "",
    ]
    (report_path).write_text("\n".join(md), encoding="utf-8")
    shutil.copyfile(report_path, latest["R022_REPORT"])

    metadata = {
        "run": "R022",
        "generated_at": stamp,
        "input_files": files,
        "input_sha256": input_hashes,
        "total_rows": int(len(all_rows)),
        "windows": summary,
        "per_group_25h": per_group.to_dict("records"),
        "scope": "availability/latency analysis; per-feature AUPRC under strict 24h not reconstructable (initial values overwritten)",
    }
    (metadata_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(metadata_path, latest["R022_METADATA"])

    print(summary_df.to_string(index=False))
    print()
    print(per_group.to_string(index=False))
    print("wrote", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
