# Explicit Trust Gate for Conflict-Aware Malicious-Domain Detection

Public code and canonical results for the manuscript **"Explicit Trust Allocation
for Conflict-Aware Malicious-Domain Detection"** (under review at the *Journal of
Information Security and Applications*).

## Overview

Malicious-domain detectors routinely combine lexical evidence from the domain
string with DNS/IP context, yet the two views can disagree. This repository
releases the experiment pipeline and result tables behind a study of an
**explicit cross-attentive trust gate**: a scalar mixture weight that decides how
much to trust a frozen lexical expert versus a frozen DNS/IP behavior expert when
the views conflict.

Key results (three-seed study on a temporal DeepURLBench sample; test split
3,372 rows with 1,683 positives; seeds 20260819-20260821):

- **Main aggregate results (Table 1):** gate AUPRC **0.8797**, bounded residual
  correction **0.8884**, standalone direct classifier **0.8979**. The richer
  heads win aggregate AUPRC; the gate is studied because it exposes an explicit,
  inspectable scalar trust weight.
- **Conflict slices (Table 2 / Fig. 3)** and **lexical-frontier reruns (Fig. 4):**
  rule-defined, model-independent slices plus DomURLs_BERT / char-CNN reruns.
- **Mechanism controls (Fig. 2)** and **error taxonomy (Fig. 5).**
- **Deployment budget (Table 4):** the gate adds 7,041 parameters, ~1.6 ms
  per-query CPU latency, and is throughput-neutral in batch mode (~16.6k rows/s).
- **Prospective transfer check (Discussion):** on a 2026-07-07 sample where
  behavior evidence is uninformative, the gate correctly falls back to the
  lexical expert (gate lexical trust > 0.999).

The paper frames this as a **diagnostic mechanism study**, not a
state-of-the-art detector claim.

## Repository layout

- `scripts/` — the experiment pipeline (18 scripts; see table below)
- `figures/` — generators for the paper's figures and LaTeX tables
- `results/` — canonical CSV/JSON result tables, mapped to paper artifacts in
  [results/README.md](results/README.md)

## Installation

Python 3.10+:

```bash
pip install -r requirements.txt
```

Torch with CUDA is optional; training runs use CUDA when available, and CPU
inference is supported.

## Data

- **DeepURLBench** (CC BY-NC 4.0) is the public benchmark used to build the
  sampled splits. Download it from
  [GitHub](https://github.com/deepinstinct-algo/DeepURLBench) or
  [Hugging Face](https://huggingface.co/datasets/DeepInstinct/DeepURLBench) and
  place the parquet files where `prepare_deepurlbench_local_splits.py` expects
  them (paths are configured at the top of the script).
- The **DNS/IP follow-up observations** (2026-07-07 prospective sample) were
  collected by the authors and are **not redistributed** here; they are
  available from the corresponding author on reasonable request.
- No raw data is included in this repository.

## Reproduce

Run from the repository root (scripts import each other as sibling modules):

| Step | Command | Produces |
|---|---|---|
| 1. Splits | `python scripts/prepare_deepurlbench_local_splits.py` | train/val/test samples |
| 2. Frozen experts | `python scripts/run_deepurlbench_local_baselines.py` | lexical + DNS/IP baselines |
| 3. Mechanism controls | `python scripts/run_deepurlbench_gate_ablations.py` | Fig. 2 (R012-R014) |
| 4. Simplicity variants | `python scripts/run_r020_residual_correction.py`, `python scripts/run_r021_conflict_classifier.py` | residual/direct heads (code dependencies) |
| 5. Main stability | `python scripts/run_r024_simplicity_stability.py` | Table 1 (R024) |
| 6. Conflict slices | `python scripts/run_r025_independent_conflict_benchmark.py` | Table 2 / Fig. 3 (R025) |
| 7. Lexical frontier | `python scripts/run_r026_domurls_r025_frontier.py` | Fig. 4 (R026; downloads DomURLs_BERT) |
| 8. Latency | `python scripts/run_r011_latency_benchmark.py` | Table 4 (R011) |
| 9. Error taxonomy | `python scripts/run_r023_qualitative_diagnosis.py` | Fig. 5 (R023) |
| 10. Missing-modality | `python scripts/run_r016_missing_modality_diagnostic.py`, `python scripts/run_r017_missing_ablation_diagnostic.py` | Appendix A.5 (R016/R017) |
| 11. Window appendix | `python scripts/run_r022_window_appendix.py` | Appendix A.4 (R022) |
| 12. Prospective transfer | `python scripts/run_prospective_gate_evaluation.py` | Discussion (needs the prospective sample, on request) |
| 13. Provenance audits | `python scripts/run_r015_leakage_provenance_audit.py`, `python scripts/run_r015a_window_sensitivity_audit.py`, `python scripts/run_r015b_amended_window_provenance_audit.py` | R015/R015A/R015B audit process (needs raw observations, on request) |

Figures and LaTeX tables:

```bash
cd figures
python gen_fig2_gate_controls.py    # Fig. 2
python gen_fig3_conflict_slices.py  # Fig. 3
python gen_fig4_lexical_frontier.py # Fig. 4
python gen_fig5_error_taxonomy.py   # Fig. 5
python gen_tables.py                # LaTeX tables from results/
```

Generators read `../results` and write output next to themselves.

## License

- **Code** (`scripts/`, `figures/`): MIT — see [LICENSE](LICENSE).
- **Results** (`results/`): derived from the CC BY-NC 4.0 DeepURLBench dataset;
  distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
  for non-commercial research use — see [results/LICENSE.results](results/LICENSE.results).
- **Third-party datasets/models** are not redistributed here; see
  [NOTICE.md](NOTICE.md).

## Citation

```bibtex
@article{explicittrust2026,
  title   = {Explicit Trust Allocation for Conflict-Aware Malicious-Domain Detection},
  author  = {Tan, Jixiang and Cai, Shenfan and Lin, Xianning and Xu, Lijin},
  journal = {Journal of Information Security and Applications},
  year    = {2026},
  note    = {Under review; DOI to be added after publication}
}
```
