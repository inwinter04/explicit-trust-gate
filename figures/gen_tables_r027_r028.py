#!/usr/bin/env python
"""LaTeX tables for R027 / R028a / R028b."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)


def esc(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def write_trust_routing() -> Path:
    m = pd.read_csv(ROOT / "refine-logs" / "R027_GATE_TRUST_ROUTING_METRICS.csv")
    m = m[m["gate_routing_acc"].notna()].copy()
    agg = m.groupby("delta").agg(
        gate=("gate_routing_acc", "mean"),
        lex=("fixed_lex_acc", "mean"),
        beh=("fixed_beh_acc", "mean"),
    ).reset_index()
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            " & ".join(
                [
                    f"{r['delta']:.2f}",
                    f"{r['gate']:.4f}",
                    f"{r['lex']:.4f}",
                    f"{r['beh']:.4f}",
                    f"{r['gate'] - r['lex']:+.4f}",
                    f"{r['gate'] - r['beh']:+.4f}",
                ]
            )
            + r" \\"
        )
    table = (
        r"\begin{table}[t]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{R027 trust-routing accuracy on expert-disagreement rows "
        + r"(\(|p_{\mathrm{lex}}-p_{\mathrm{dns}}|\ge\delta\)), averaged over three seeds. "
        + r"The g-route trusts lexical when \(g\ge0.5\), otherwise behavior. "
        + r"g-routing dominates always-trusting behavior and matches or slightly exceeds "
        + r"always-trusting lexical, and the mean trust weight separates the two "
        + r"correctness regimes (lex-correct-only vs.\ beh-correct-only).}" + "\n"
        + r"\label{tab:trust_routing}" + "\n"
        + r"\resizebox{\textwidth}{!}{%" + "\n"
        + r"\begin{tabular}{lrrrrr}" + "\n"
        + r"\toprule" + "\n"
        + r"$\delta$ & g-route & fixed-lex & fixed-beh & $\Delta$ vs.\ lex & $\Delta$ vs.\ beh \\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"}" + "\n"
        + r"\end{table}" + "\n"
    )
    path = OUT / "TABLE_trust_routing.tex"
    path.write_text(table, encoding="utf-8")
    return path


def write_informative_subsets() -> Path:
    agg = pd.read_csv(ROOT / "refine-logs" / "R028A_INFORMATIVE_SUBSET_AGGREGATE.csv")
    label = {
        "full_test": "Full test",
        "multi_ip_dns": "Multi-IP DNS",
        "lex_benign_shape_multi_ip": "Lex-benign + multi-IP",
        "ttl_informative": "TTL-informative (DNS present, TTL $>$ 1s)",
    }
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            " & ".join(
                [
                    label.get(r["subset"], r["subset"]),
                    f"{r['n_rows_mean']:.0f}",
                    f"{r['gate_AUPRC_mean']:.4f}",
                    f"{r['lexical_AUPRC_mean']:.4f}",
                    f"{r['gate_minus_lexical_mean']:+.4f}",
                    f"{r['mean_g_lex']:.3f}",
                ]
            )
            + r" \\"
        )
    table = (
        r"\begin{table}[t]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{R028a AUPRC on model-independent behavior-informative subsets "
        + r"(mean over three seeds). The explicit gate improves over the lexical-only "
        + r"expert on every subset, showing behavior-conditioned gains where the "
        + r"behavior view carries signal.}" + "\n"
        + r"\label{tab:informative_subsets}" + "\n"
        + r"\resizebox{\textwidth}{!}{%" + "\n"
        + r"\begin{tabular}{lrrrrr}" + "\n"
        + r"\toprule" + "\n"
        + r"Subset & Rows & Gate & Lexical & $\Delta$ AUPRC & Mean $g$ \\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"}" + "\n"
        + r"\end{table}" + "\n"
    )
    path = OUT / "TABLE_informative_subsets.tex"
    path.write_text(table, encoding="utf-8")
    return path


def write_injection() -> Path:
    m_all = pd.read_csv(ROOT / "refine-logs" / "R028B_PROSPECTIVE_INJECTION_METRICS.csv")
    gate_diag = m_all[m_all["system"] == "gate_diagnostic"].groupby("condition")["mean_g_lex"].mean()
    m = m_all[m_all["system"].isin(["lexical_only", "dns_behavior_only", "cross_attentive_gate"])]
    cond_order = ["original_degenerate", "injected_ttl_informative", "injected_ip_diversity_informative"]
    cond_label = {
        "original_degenerate": "Original (degenerate behavior)",
        "injected_ttl_informative": "Injected TTL (uninformative)",
        "injected_ip_diversity_informative": "Injected IP diversity (informative)",
    }
    rows = []
    for cond in cond_order:
        part = m[m["condition"] == cond]
        get = lambda key: float(part[part["system"] == key]["AUPRC"].mean())  # noqa: E731
        gmean = gate_diag.get(cond, float("nan"))
        rows.append(
            " & ".join(
                [
                    cond_label[cond],
                    f"{get('lexical_only'):.4f}",
                    f"{get('dns_behavior_only'):.4f}",
                    f"{get('cross_attentive_gate'):.4f}",
                    f"{gmean:.3f}",
                ]
            )
            + r" \\"
        )
    table = (
        r"\begin{table}[t]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{R028b controlled counterfactual on the verified-timing "
        + r"2026-07-07 prospective sample (200 rows, 100 malicious), mean over three seeds. "
        + r"When a disclosed informative IP-diversity signal is injected into the behavior "
        + r"view, the DNS expert becomes informative and the gate improves AUPRC over "
        + r"lexical-only through partial reweighting (mean \(g\) drops below 1 but stays "
        + r"above 0.5). The injected-TTL condition is inconclusive because the frozen DNS "
        + r"expert does not respond to the signal (AUPRC 0.58); the original degenerate "
        + r"condition reproduces lexical fallback. Diagnostic counterfactual, not a "
        + r"deployment result.}" + "\n"
        + r"\label{tab:injection_conditions}" + "\n"
        + r"\resizebox{\textwidth}{!}{%" + "\n"
        + r"\begin{tabular}{lrrrr}" + "\n"
        + r"\toprule" + "\n"
        + r"Condition & Lexical AUPRC & DNS AUPRC & Gate AUPRC & Mean $g$ \\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"}" + "\n"
        + r"\end{table}" + "\n"
    )
    path = OUT / "TABLE_injection_conditions.tex"
    path.write_text(table, encoding="utf-8")
    return path


def main() -> None:
    paths = [write_trust_routing(), write_informative_subsets(), write_injection()]
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
