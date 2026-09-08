"""Module 10 — reproducibility report and paper-ready output package."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .analysis import aggregate_results, nps_table
from .config import ExperimentConfig


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame without requiring pandas' optional tabulate dependency."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in frame.columns]

        lines = [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]

        for row in frame.itertuples(
            index=False,
            name=None,
        ):
            lines.append(
                "| "
                + " | ".join(str(value) for value in row)
                + " |"
            )

        return "\n".join(lines)


def generate_report(
    config: ExperimentConfig,
    raw_path: Path,
    output_path: Path | None = None,
    svm_summary_path: Path | None = None,
) -> Path:
    """
    Generate the reproducibility report for the quantum benchmark
    and the integrated classical SVM baseline.
    """

    output_path = (
        output_path
        or (
            config.root
            / "reports"
            / "reproducibility_report.md"
        )
    )

    df = pd.DataFrame(
        [
            json.loads(line)
            for line in raw_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
    )

    if df.empty:
        raise ValueError(
            "No quantum benchmark records were found."
        )

    agg = aggregate_results(df)
    nps = nps_table(df)

    svm_summary = None

    if (
        svm_summary_path is not None
        and svm_summary_path.exists()
    ):
        svm_summary = pd.read_csv(
            svm_summary_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# UniQBench Reproducibility Report",
        "",
        (
            "Generated at: "
            f"{datetime.now(timezone.utc).isoformat()}"
        ),
        (
            "Configuration hash: "
            f"`{config.config_hash()}`"
        ),
        "",
        "## Scope",
        "",
        (
            "This report was generated from the ten "
            "UniQBench modules. It includes the "
            "quantum-feature benchmark results and "
            "the integrated classical SVM baseline."
        ),
        "",
        "## Experimental configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]

    for key, value in config.to_dict().items():
        lines.append(
            f"| {key} | `{value}` |"
        )

    lines += [
        "",
        "## Dataset",
        "",
        (
            "The experiment uses the MNIST dataset obtained "
            "from OpenML as `mnist_784`, version 1. Only "
            "classes 0 and 1 are retained, with labels "
            "mapped as 0 → 0 and 1 → 1."
        ),
        "",
        (
            "The final experiment uses deterministic "
            "stratified train/validation/test partitions "
            "of 200, 80, and 80 samples for seeds 7, 19, "
            "and 31. Pixel values are scaled by 1/255. "
            "Standardization and PCA are fitted on the "
            "training data only and then applied to "
            "validation and test data."
        ),
        "",
        "## Raw-data summary",
        "",
        (
            f"Raw quantum benchmark records: "
            f"**{len(df)}**"
        ),
        "",
        (
            "| Backend | Runs | Mean accuracy | "
            "Accuracy standard deviation | Mean macro-F1 |"
        ),
        "|---|---:|---:|---:|---:|",
    ]

    for backend, group in df.groupby(
        "backend"
    ):
        accuracy_std = (
            group["accuracy"].std(ddof=1)
            if len(group) > 1
            else 0.0
        )

        lines.append(
            f"| {backend} | "
            f"{len(group)} | "
            f"{group['accuracy'].mean():.4f} | "
            f"{accuracy_std:.4f} | "
            f"{group['macro_f1'].mean():.4f} |"
        )

    lines += [
        "",
        "## Backend-by-configuration aggregates",
        "",
        _markdown_table(agg),
        "",
        "## Classical Baseline — SVM",
        "",
        (
            "The classical baseline is an RBF-kernel "
            "support vector machine evaluated using "
            "the same PCA-processed features and the "
            "same deterministic train/validation/test "
            "partitions as the quantum model."
        ),
        "",
        (
            "The SVM hyperparameter C is selected using "
            "the validation set only. The selected model "
            "is then fitted on the training set and "
            "evaluated on the untouched test set."
        ),
        "",
    ]

    if svm_summary is not None:
        lines.append(
            _markdown_table(svm_summary)
        )
    else:
        lines.append(
            "Classical SVM results were not supplied."
        )

    lines += [
        "",
        "## Exploratory Noise Portability Score",
        "",
        (
            "NPS is reported only as an exploratory "
            "statistic. It is not interpreted as a "
            "universal deployment threshold."
        ),
        "",
        _markdown_table(nps),
        "",
        "## Model description",
        "",
        (
            "The quantum model is a fixed variational "
            "quantum feature map with a trainable "
            "classical prediction head."
        ),
        "",
        (
            "The variational quantum parameters are "
            "initialized once and held fixed during "
            "the reported training procedure. Only "
            "the classical prediction-head weights "
            "and biases are optimized."
        ),
        "",
        (
            "Accordingly, the reported model should not "
            "be described as a fully trainable variational "
            "quantum classifier."
        ),
        "",
        "## Reproducibility notes",
        "",
        (
            "- Dataset, split, and PCA metadata are "
            "generated by Module 2 using the shared deterministic split helper."
        ),
        (
            "- Circuit structure and nominal gate "
            "statistics are generated by Module 3."
        ),
        (
            "- Backend execution records are generated "
            "by Module 5."
        ),
        (
            "- Training logs and checkpoints are "
            "generated by Module 6."
        ),
        (
            "- Raw quantum run-level observations are "
            "generated by Module 7."
        ),
        (
            "- Metrics and the classical SVM baseline "
            "are generated by Module 8."
        ),
        (
            "- Tables and figures are generated by "
            "Module 9."
        ),
        (
            "- This report is generated by Module 10."
        ),
        "",
        "## Scientific limitations",
        "",
        (
            "The final experiment uses the OpenML "
            "`mnist_784` dataset restricted to classes "
            "0 and 1 and evaluates only a small binary "
            "subset with 200 training, 80 validation, "
            "and 80 test samples per seed."
        ),
        "",
        (
            "The quantum model is a hybrid quantum-feature classifier "
"using a fixed variational quantum feature map and a "
"trainable classical prediction head. The variational quantum "
"parameters are initialized once and held fixed during the "
"reported training procedure; only the classical prediction "
"head weights and biases are optimized."
        ),
        "",
        (
            "The `noisy_shot` backend is a transparent "
            "shot-and-noise simulation approximation "
            "rather than execution on physical quantum "
            "hardware."
        ),
        "",
        (
            "The classical SVM is included as a reference "
            "baseline using the same processed features "
            "and data partitions. The current experiments "
            "do not establish quantum advantage."
        ),
    ]

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return output_path