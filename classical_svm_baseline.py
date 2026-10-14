from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from uniqbench.config import default_config
from uniqbench.data import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results_classical_svm_v1"


def evaluate_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Select C using only the validation set, then evaluate the selected
    model once on the untouched test set.
    """

    candidate_c_values = [0.1, 1.0, 10.0]

    best_c = None
    best_val_f1 = -1.0

    for c_value in candidate_c_values:
        model = SVC(
            C=c_value,
            kernel="rbf",
            gamma="scale",
        )

        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)
        val_f1 = f1_score(
            y_val,
            val_predictions,
            average="macro",
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_c = c_value

    final_model = SVC(
        C=best_c,
        kernel="rbf",
        gamma="scale",
    )

    final_model.fit(X_train, y_train)

    test_predictions = final_model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
    )

    return (
        float(best_c),
        float(best_val_f1),
        float(test_accuracy),
        float(test_macro_f1),
    )


def main() -> None:
    config = default_config()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for seed in config.seeds:
        prepared = prepare_data(
            config,
            seed=seed,
        )

        for n_components in config.n_components:
            X_train = prepared.X_train[:, :n_components]
            X_val = prepared.X_val[:, :n_components]
            X_test = prepared.X_test[:, :n_components]

            (
                best_c,
                validation_macro_f1,
                test_accuracy,
                test_macro_f1,
            ) = evaluate_svm(
                X_train,
                prepared.y_train,
                X_val,
                prepared.y_val,
                X_test,
                prepared.y_test,
            )

            records.append(
                {
                    "seed": seed,
                    "n_components": n_components,
                    "n_qubits_equivalent": n_components,
                    "model": "SVM_RBF",
                    "kernel": "rbf",
                    "best_C": best_c,
                    "validation_macro_f1": validation_macro_f1,
                    "test_accuracy": test_accuracy,
                    "test_macro_f1": test_macro_f1,
                }
            )

    # ---------------------------------------------------------
    # Save run-level results
    # ---------------------------------------------------------
    raw_path = OUTPUT_DIR / "svm_runs.csv"

    fieldnames = [
        "seed",
        "n_components",
        "n_qubits_equivalent",
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

    # ---------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------
    summary_path = OUTPUT_DIR / "svm_summary.csv"

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

        for n_components in config.n_components:
            subset = [
                row
                for row in records
                if row["n_components"] == n_components
            ]

            accuracies = np.array(
                [row["test_accuracy"] for row in subset],
                dtype=float,
            )

            macro_f1s = np.array(
                [row["test_macro_f1"] for row in subset],
                dtype=float,
            )

            writer.writerow(
                [
                    n_components,
                    accuracies.mean(),
                    accuracies.std(ddof=1),
                    macro_f1s.mean(),
                    macro_f1s.std(ddof=1),
                    len(subset),
                ]
            )

    print("Classical SVM baseline complete")
    print("Raw results:", raw_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()