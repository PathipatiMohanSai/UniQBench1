# UniQBench

Revised reproducible benchmark manuscript and supporting artifacts for:

**UniQBench: A Reproducible Framework for Evaluating Quantum Machine Learning Across Model Families, Noise Levels, and Shot Budgets**

Repository: https://github.com/PathipatiMohanSai/UniQBench1

Release package: `release_2026_09/` (mirrored in the public repository).

## What is included

- `manuscript/` — final revised manuscript DOCX.
- `figures/` — supplementary high-resolution figures for external validation, encoding ablation, seed replication, and error analysis.
- `results/` — supplementary experiment tables and machine-readable analysis outputs.
- `supplementary/` — scripts used for the external-validation, encoding-ablation, seed-expansion, sensitivity, and figure-generation work.
- `src/` — available UniQBench source modules recovered from the project artifacts.

## Experimental additions in the revised manuscript

1. External validation on the Breast Cancer Wisconsin (Diagnostic) dataset.
2. Encoding-choice ablation using a second angle-based ZZ-interaction feature construction.
3. Ten-seed replication of the external-validation study.
4. Class-level error analysis with confusion matrices and P/R/F1 summaries.
5. Replication-size sensitivity analysis around a prespecified 0.05 absolute accuracy effect.
6. Computational-complexity and scaling analysis.

## Reproducibility scope

The original manuscript benchmark contains 432 source-result records and uses the primary MNIST protocol described in the paper. The supplied manuscript artifacts did not include the original instance-level MNIST prediction snapshot, so the revised error analysis is explicitly based on the supplementary WDBC validation rather than reconstructed MNIST predictions.

The WDBC extensions were implemented independently while preserving the documented preprocessing, representation, model-family, seed, and evaluation controls. They are therefore reported as external validation and replication evidence, not as a byte-level re-execution of the original 432-record MNIST benchmark.

## Data sources

- Primary benchmark dataset: OpenML `MNIST 784` (dataset 554), referenced in the manuscript.
- External validation dataset: UCI Machine Learning Repository, `Breast Cancer Wisconsin (Diagnostic)`, Dataset 17.

Raw datasets are not redistributed in this repository; the scripts load them through the documented dataset interfaces.

## Publication note

The repository is intended to be the public code/artifact location cited by the manuscript. Before submission, verify that the repository visibility, commit corresponding to the manuscript version, and dataset access instructions remain available to reviewers.
