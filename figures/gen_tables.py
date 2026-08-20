from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
TABLE_DIR.mkdir(exist_ok=True)


def esc(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def write_main_results() -> Path:
    r024 = pd.read_csv(ROOT / "results" / "R024_SIMPLICITY_STABILITY_AGGREGATE.csv")
    keep = r024[r024["split"].isin(["full_test", "model_high_conflict"])].copy()
    keep = keep[keep["system"].isin(["cross_attentive_gate", "residual_correction_gate", "standalone_conflict_classifier"])]
    keep["system"] = keep["system"].map(
        {
            "cross_attentive_gate": "Trust gate",
            "residual_correction_gate": "Residual",
            "standalone_conflict_classifier": "Direct head",
        }
    )
    keep["split"] = keep["split"].map({"full_test": "Full test", "model_high_conflict": "Model-high-conflict"})

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main R024 aggregate results. The direct head is strongest on aggregate AUPRC, but the explicit gate remains the mechanism analyzed for conflict routing.}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Split & System & AUPRC & FPR@95TPR & Macro-F1 & ECE \\\\",
        "\\midrule",
    ]
    for _, row in keep.iterrows():
        lines.append(
            f"{esc(row['split'])} & {esc(row['system'])} & {row['AUPRC_mean']:.4f} & {row['FPR95_mean']:.4f} & {row['macro_F1_mean']:.4f} & {row['ECE_mean']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out = TABLE_DIR / "TABLE_main_results.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")
    return out


def write_integrity_table() -> Path:
    r015a = pd.read_csv(ROOT / "results" / "R015A_WINDOW_SENSITIVITY_SUMMARY.csv")
    r015b = pd.read_csv(ROOT / "results" / "R015B_AMENDED_WINDOW_SUMMARY.csv")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Point-in-time provenance audit summary. The original 24h declaration fails; an explicit 25h operational amendment passes for the local prospective sample, while DeepURLBench DNS/IP timing remains unverified.}",
        "\\label{tab:integrity_audit}",
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        "Audit & Window & Outside rows & Status \\\\",
        "\\midrule",
    ]
    for _, row in r015a.iterrows():
        status = "pass" if int(row["outside_rows"]) == 0 else "fail"
        lines.append(f"R015A sensitivity & {row['window_hours']}h & {int(row['outside_rows'])} & {status} \\\\")
    for _, row in r015b.iterrows():
        status = "pass" if int(row["outside_rows"]) == 0 else "fail"
        lines.append(f"R015B {esc(row['window'])} & {row['window_hours']}h & {int(row['outside_rows'])} & {status} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out = TABLE_DIR / "TABLE_integrity_audit.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")
    return out


def write_conflict_slice_table() -> Path:
    r025 = pd.read_csv(ROOT / "results" / "R025_INDEPENDENT_CONFLICT_AGGREGATE.csv")
    keep = r025[r025["system"].isin(["cross_attentive_gate", "residual_correction_gate", "standalone_conflict_classifier"])].copy()
    pivot = keep.pivot(index="subset", columns="system", values="AUPRC_mean").reset_index()
    pivot["subset"] = pivot["subset"].map(
        {
            "lex_benign_shape": "Lex-benign shape",
            "lex_benign_shape_multi_ip": "Lex-benign + multi-IP",
            "multi_ip_dns": "Multi-IP DNS",
            "ttl_low": "Low TTL",
        }
    )
    order = ["Lex-benign shape", "Lex-benign + multi-IP", "Multi-IP DNS", "Low TTL"]
    pivot["order"] = pivot["subset"].map({name: i for i, name in enumerate(order)})
    pivot = pivot.sort_values("order")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{R025 rule-defined conflict-slice AUPRC with sample size and positives. Residual fusion is best on three slices and the direct head is best on one, so the explicit gate is analyzed as an inspectable mechanism rather than the best slice classifier.}",
        "\\label{tab:conflict_slices}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Slice & Rows & Pos. & Trust gate & Residual & Direct head \\\\",
        "\\midrule",
    ]
    for _, row in pivot.iterrows():
        subset_key = {
            "Lex-benign shape": "lex_benign_shape",
            "Lex-benign + multi-IP": "lex_benign_shape_multi_ip",
            "Multi-IP DNS": "multi_ip_dns",
            "Low TTL": "ttl_low",
        }[row["subset"]]
        counts = keep[(keep["subset"].eq(subset_key)) & (keep["system"].eq("cross_attentive_gate"))].iloc[0]
        lines.append(
            f"{esc(row['subset'])} & {int(counts['rows'])} & {int(counts['positives'])} & {row['cross_attentive_gate']:.4f} & {row['residual_correction_gate']:.4f} & {row['standalone_conflict_classifier']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    out = TABLE_DIR / "TABLE_conflict_slices.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")
    return out


def main() -> None:
    write_main_results()
    write_conflict_slice_table()
    write_integrity_table()


if __name__ == "__main__":
    main()
