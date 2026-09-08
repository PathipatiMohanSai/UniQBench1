"""Module 8 — metrics, uncertainty-aware summaries, and classical baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

import numpy as np
from sklearn.svm import SVC


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    f1_scores = []

    for label in sorted(np.unique(y_true).tolist()):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        f1_scores.append(
            2 * precision * recall
            / max(precision + recall, 1e-12)
        )

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "mean_confidence": float(
            np.mean(np.max(probabilities, axis=1))
        ),
        "n_test_samples": int(len(y_true)),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("Cannot summarize an empty array.")

    mean = float(np.mean(values))

    std = (
        float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0
    )

    sem = (
        std / np.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )

    return {
        "n": int(len(values)),
        "mean": mean,
        "std_sample": std,
        "ci95_low_normal": mean - 1.96 * sem,
        "ci95_high_normal": mean + 1.96 * sem,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def noise_portability_score(
    accuracies: np.ndarray,
) -> float:
    values = np.asarray(accuracies, dtype=float)

    if len(values) == 0:
        raise ValueError(
            "NPS requires at least one accuracy value."
        )

    mean = float(np.mean(values))

    if np.isclose(mean, 0.0):
        return float("nan")

    std = (
        float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0
    )

    return float(1.0 - (std / mean))


def _macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = sorted(
        np.unique(y_true).tolist()
    )

    f1_scores = []

    for label in labels:
        tp = np.sum(
            (y_true == label)
            & (y_pred == label)
        )
        fp = np.sum(
            (y_true != label)
            & (y_pred == label)
        )
        fn = np.sum(
            (y_true == label)
            & (y_pred != label)
        )

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        f1_scores.append(
            2 * precision * recall
            / max(
                precision + recall,
                1e-12,
            )
        )

    return float(np.mean(f1_scores))


def _select_svm_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Select SVM C using validation data only."""

    candidate_c_values = [0.1, 1.0, 10.0]

    best_c = None
    best_val_f1 = -1.0

    for c_value in candidate_c_values:
        model = SVC(
            C=c_value,
            kernel="rbf",
            gamma="scale",
        )

        model.fit(
            X_train,
            y_train,
        )

        val_predictions = model.predict(
            X_val
        )

        val_macro_f1 = _macro_f1(
            y_val,
            val_predictions,
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_c = c_value

    if best_c is None:
        raise RuntimeError(
            "SVM hyperparameter selection failed."
        )

    return float(best_c), float(best_val_f1)


def run_svm_baseline(
    config,
) -> list[dict[str, Any]]:
    """
    Run the classical SVM baseline using the same prepared
    MNIST features and deterministic seeds as the quantum
    benchmark.

    Validation data is used only to select C.
    Test data is used only for final evaluation.
    """

    from .data import prepare_data

    records: list[dict[str, Any]] = []

    for seed in config.seeds:
        prepared = prepare_data(
            config,
            seed=seed,
        )

        for n_components in config.n_components:
            X_train = prepared.X_train[:, :n_components]
            X_val = prepared.X_val[:, :n_components]
            X_test = prepared.X_test[:, :n_components]

            best_c, validation_macro_f1 = _select_svm_c(
                X_train,
                prepared.y_train,
                X_val,
                prepared.y_val,
            )

            final_model = SVC(
                C=best_c,
                kernel="rbf",
                gamma="scale",
            )

            final_model.fit(
                X_train,
                prepared.y_train,
            )

            test_predictions = final_model.predict(
                X_test
            )

            test_accuracy = float(
                np.mean(
                    test_predictions
                    == prepared.y_test
                )
            )

            test_macro_f1 = _macro_f1(
                prepared.y_test,
                test_predictions,
            )

            records.append(
                {
                    "seed": int(seed),
                    "n_components": int(
                        n_components
                    ),
                    "model": "SVM_RBF",
                    "kernel": "rbf",
                    "best_C": best_c,
                    "validation_macro_f1": (
                        validation_macro_f1
                    ),
                    "test_accuracy": test_accuracy,
                    "test_macro_f1": test_macro_f1,
                }
            )

    return records


def write_svm_results(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write run-level and aggregate SVM baseline results."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "classical_svm_runs.csv"
    )

    summary_path = (
        output_dir
        / "classical_svm_summary.csv"
    )

    fieldnames = [
        "seed",
        "n_components",
        "model",
        "kernel",
        "best_C",
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
    ]

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "n_components",
                "mean_test_accuracy",
                "std_test_accuracy",
                "mean_test_macro_f1",
                "std_test_macro_f1",
                "n_runs",
            ]
        )

        for n_components in config_n_components(records):
            subset = [
                row
                for row in records
                if row["n_components"] == n_components
            ]

            if not subset:
                continue

            accuracies = np.asarray(
                [
                    row["test_accuracy"]
                    for row in subset
                ],
                dtype=float,
            )

            macro_f1s = np.asarray(
                [
                    row["test_macro_f1"]
                    for row in subset
                ],
                dtype=float,
            )

            writer.writerow(
                [
                    n_components,
                    float(
                        np.mean(accuracies)
                    ),
                    float(
                        np.std(
                            accuracies,
                            ddof=1,
                        )
                    )
                    if len(accuracies) > 1
                    else 0.0,
                    float(
                        np.mean(macro_f1s)
                    ),
                    float(
                        np.std(
                            macro_f1s,
                            ddof=1,
                        )
                    )
                    if len(macro_f1s) > 1
                    else 0.0,
                    len(subset),
                ]
            )

    return {
        "svm_raw": raw_path,
        "svm_summary": summary_path,
    }


def config_n_components(
    records: list[dict[str, Any]],
) -> list[int]:
    """Return sorted PCA dimensions present in the SVM results."""

    return sorted(
        {
            int(row["n_components"])
            for row in records
        }
    )