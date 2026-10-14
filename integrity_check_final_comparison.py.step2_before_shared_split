from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "results_mnist_final_comparison_v1"

RAW_JSONL = FINAL_DIR / "raw" / "benchmark_runs.jsonl"
RAW_CSV = FINAL_DIR / "raw" / "benchmark_runs.csv"


EXPECTED_SEEDS = {7, 19, 31}
EXPECTED_PCA = {4, 8}
EXPECTED_QUBITS = {4, 8}
EXPECTED_DEPTHS = {1, 2, 4}
EXPECTED_BACKENDS = {
    "numpy_statevector",
    "noisy_shot",
}

EXPECTED_CLASS_COUNTS = {
    "train": [93, 107],
    "validation": [37, 43],
    "test": [37, 43],
}


def load_jsonl(path: Path) -> pd.DataFrame:
    records = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not records:
        raise AssertionError(
            "JSONL file contains no records."
        )

    return pd.DataFrame(records)


def check_run_count_and_combinations(
    df: pd.DataFrame,
) -> None:
    required_columns = {
        "seed",
        "n_components",
        "n_qubits",
        "depth",
        "backend",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise AssertionError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    if len(df) != 36:
        raise AssertionError(
            f"Expected exactly 36 raw runs, "
            f"found {len(df)}."
        )

    df["seed"] = df["seed"].astype(int)
    df["n_components"] = (
        df["n_components"].astype(int)
    )
    df["n_qubits"] = (
        df["n_qubits"].astype(int)
    )
    df["depth"] = df["depth"].astype(int)

    if set(df["seed"]) != EXPECTED_SEEDS:
        raise AssertionError(
            f"Unexpected seeds: "
            f"{sorted(df['seed'].unique())}"
        )

    if (
        set(df["n_components"])
        != EXPECTED_PCA
    ):
        raise AssertionError(
            "Unexpected PCA dimensions: "
            f"{sorted(df['n_components'].unique())}"
        )

    if (
        set(df["n_qubits"])
        != EXPECTED_QUBITS
    ):
        raise AssertionError(
            "Unexpected qubit widths: "
            f"{sorted(df['n_qubits'].unique())}"
        )

    if set(df["depth"]) != EXPECTED_DEPTHS:
        raise AssertionError(
            f"Unexpected depths: "
            f"{sorted(df['depth'].unique())}"
        )

    if (
        set(df["backend"])
        != EXPECTED_BACKENDS
    ):
        raise AssertionError(
            "Unexpected backends: "
            f"{sorted(df['backend'].unique())}"
        )

    combination_columns = [
        "seed",
        "n_components",
        "n_qubits",
        "depth",
        "backend",
    ]

    duplicates = df.duplicated(
        subset=combination_columns,
        keep=False,
    )

    if duplicates.any():
        duplicate_rows = df.loc[
            duplicates,
            combination_columns,
        ]

        raise AssertionError(
            "Duplicate run combinations detected:\n"
            + duplicate_rows.to_string(
                index=False
            )
        )

    # Effective unique run ID.
    run_ids = (
        "s"
        + df["seed"].astype(str)
        + "_p"
        + df["n_components"].astype(str)
        + "_q"
        + df["n_qubits"].astype(str)
        + "_d"
        + df["depth"].astype(str)
        + "_"
        + df["backend"].astype(str)
    )

    if not run_ids.is_unique:
        raise AssertionError(
            "Effective run IDs are not unique."
        )

    expected_count = (
        len(EXPECTED_SEEDS)
        * len(EXPECTED_PCA)
        * len(EXPECTED_DEPTHS)
        * len(EXPECTED_BACKENDS)
    )

    if expected_count != 36:
        raise AssertionError(
            f"Internal expected-count calculation "
            f"is {expected_count}, not 36."
        )

    print(
        "PASS: 36 unique run combinations."
    )

    print(
        "PASS: effective run IDs are unique "
        f"({run_ids.nunique()})."
    )


def check_metrics(
    df: pd.DataFrame,
) -> None:
    metric_columns = [
        "accuracy",
        "macro_f1",
        "mean_confidence",
        "training_wall_seconds",
    ]

    missing = [
        column
        for column in metric_columns
        if column not in df.columns
    ]

    if missing:
        raise AssertionError(
            f"Missing metric columns: {missing}"
        )

    if (
        df[metric_columns]
        .isnull()
        .any()
        .any()
    ):
        null_counts = (
            df[metric_columns]
            .isnull()
            .sum()
            .to_dict()
        )

        raise AssertionError(
            f"Null metrics found: {null_counts}"
        )

    numeric_values = (
        df[metric_columns]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise AssertionError(
            "Non-finite metric value detected."
        )

    if not df["accuracy"].between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "Accuracy outside [0, 1]."
        )

    if not df["macro_f1"].between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "Macro-F1 outside [0, 1]."
        )

    if not df["mean_confidence"].between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "Mean confidence outside [0, 1]."
        )

    if (
        df["training_wall_seconds"] < 0
    ).any():
        raise AssertionError(
            "Negative training time detected."
        )

    print(
        "PASS: all required metrics are "
        "non-null and finite."
    )


def load_filtered_mnist():
    cache_dir = ROOT / "data_cache"

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mnist = fetch_openml(
        name="mnist_784",
        version=1,
        as_frame=False,
        data_home=str(cache_dir),
    )

    X = np.asarray(
        mnist.data,
        dtype=np.float32,
    )

    y_raw = np.asarray(
        mnist.target,
        dtype=np.int64,
    )

    mask = np.isin(
        y_raw,
        [0, 1],
    )

    X = X[mask]
    y = y_raw[mask]

    return X, y


def reproduce_indices(
    seed: int,
    y: np.ndarray,
):
    all_indices = np.arange(
        len(y)
    )

    train_indices, remaining_indices = (
        train_test_split(
            all_indices,
            train_size=200,
            stratify=y,
            random_state=seed,
        )
    )

    validation_indices, test_indices = (
        train_test_split(
            remaining_indices,
            train_size=80,
            test_size=80,
            stratify=y[remaining_indices],
            random_state=seed + 1,
        )
    )

    return (
        train_indices,
        validation_indices,
        test_indices,
    )


def check_split_indices() -> None:
    _, y = load_filtered_mnist()

    for seed in sorted(
        EXPECTED_SEEDS
    ):
        (
            train_indices,
            validation_indices,
            test_indices,
        ) = reproduce_indices(
            seed,
            y,
        )

        train_set = set(
            train_indices.tolist()
        )

        validation_set = set(
            validation_indices.tolist()
        )

        test_set = set(
            test_indices.tolist()
        )

        if train_set & validation_set:
            raise AssertionError(
                f"Seed {seed}: "
                "train/validation overlap."
            )

        if train_set & test_set:
            raise AssertionError(
                f"Seed {seed}: "
                "train/test overlap."
            )

        if validation_set & test_set:
            raise AssertionError(
                f"Seed {seed}: "
                "validation/test overlap."
            )

        total_unique = (
            len(train_set)
            + len(validation_set)
            + len(test_set)
        )

        if total_unique != 360:
            raise AssertionError(
                f"Seed {seed}: split does not "
                "contain exactly 360 unique samples."
            )

        counts = {
            "train": np.bincount(
                y[train_indices],
                minlength=2,
            ).tolist(),
            "validation": np.bincount(
                y[validation_indices],
                minlength=2,
            ).tolist(),
            "test": np.bincount(
                y[test_indices],
                minlength=2,
            ).tolist(),
        }

        if counts != EXPECTED_CLASS_COUNTS:
            raise AssertionError(
                f"Seed {seed}: unexpected "
                f"class counts: {counts}"
            )

        print(
            f"PASS: seed {seed} split is disjoint "
            "with expected class counts."
        )


def check_jsonl_csv_agreement(
    jsonl_df: pd.DataFrame,
    csv_df: pd.DataFrame,
) -> None:
    """
    Check that JSONL and CSV contain the same run records.

    Identifier fields are compared exactly.
    Numeric metrics are compared with a very small
    floating-point tolerance because CSV serialization
    and parsing can change the final decimal representation.
    """

    if len(jsonl_df) != len(csv_df):
        raise AssertionError(
            "JSONL and CSV row counts differ: "
            f"{len(jsonl_df)} vs {len(csv_df)}"
        )

    key_columns = [
        "seed",
        "n_components",
        "n_qubits",
        "depth",
        "backend",
    ]

    numeric_columns = [
        "accuracy",
        "macro_f1",
        "mean_confidence",
        "training_wall_seconds",
    ]

    required_columns = (
        key_columns + numeric_columns
    )

    missing_jsonl = [
        column
        for column in required_columns
        if column not in jsonl_df.columns
    ]

    missing_csv = [
        column
        for column in required_columns
        if column not in csv_df.columns
    ]

    if missing_jsonl:
        raise AssertionError(
            "JSONL is missing columns: "
            f"{missing_jsonl}"
        )

    if missing_csv:
        raise AssertionError(
            "CSV is missing columns: "
            f"{missing_csv}"
        )

    left = jsonl_df[
        required_columns
    ].copy()

    right = csv_df[
        required_columns
    ].copy()

    # Normalize identifier columns.
    for column in key_columns:
        left[column] = (
            left[column]
            .astype(str)
            .str.strip()
        )

        right[column] = (
            right[column]
            .astype(str)
            .str.strip()
        )

    # Convert identifiers to numeric where appropriate.
    for column in [
        "seed",
        "n_components",
        "n_qubits",
        "depth",
    ]:
        left[column] = pd.to_numeric(
            left[column],
            errors="raise",
        ).astype(int)

        right[column] = pd.to_numeric(
            right[column],
            errors="raise",
        ).astype(int)

    left = left.sort_values(
        key_columns,
        kind="stable",
    ).reset_index(drop=True)

    right = right.sort_values(
        key_columns,
        kind="stable",
    ).reset_index(drop=True)

    # Exact identity comparison.
    for column in key_columns:
        if not left[column].equals(
            right[column]
        ):
            raise AssertionError(
                f"JSONL and CSV differ in "
                f"identifier column '{column}'."
            )

    # Floating-point comparison.
    for column in numeric_columns:
        left_values = pd.to_numeric(
            left[column],
            errors="raise",
        ).to_numpy(dtype=float)

        right_values = pd.to_numeric(
            right[column],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.allclose(
            left_values,
            right_values,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=False,
        ):
            differences = np.abs(
                left_values
                - right_values
            )

            max_difference = float(
                np.max(differences)
            )

            raise AssertionError(
                f"JSONL and CSV differ in "
                f"numeric column '{column}'. "
                f"Maximum absolute difference: "
                f"{max_difference}"
            )

    print(
        "PASS: raw JSONL and CSV records agree "
        "for all identifiers and metrics."
    )


def check_svm_results() -> None:
    """
    Validate the integrated classical SVM baseline.
    """

    svm_raw = (
        FINAL_DIR
        / "processed"
        / "classical_svm_runs.csv"
    )

    svm_summary = (
        FINAL_DIR
        / "processed"
        / "classical_svm_summary.csv"
    )

    if not svm_raw.exists():
        raise AssertionError(
            f"Missing SVM raw results: {svm_raw}"
        )

    if not svm_summary.exists():
        raise AssertionError(
            f"Missing SVM summary: {svm_summary}"
        )

    raw = pd.read_csv(
        svm_raw
    )

    summary = pd.read_csv(
        svm_summary
    )

    if len(raw) != 6:
        raise AssertionError(
            f"Expected 6 SVM runs, "
            f"found {len(raw)}."
        )

    expected_seeds = {7, 19, 31}
    expected_pca = {4, 8}

    if set(raw["seed"].astype(int)) != expected_seeds:
        raise AssertionError(
            "Unexpected SVM seeds."
        )

    if (
        set(raw["n_components"].astype(int))
        != expected_pca
    ):
        raise AssertionError(
            "Unexpected SVM PCA dimensions."
        )

    if raw[
        [
            "validation_macro_f1",
            "test_accuracy",
            "test_macro_f1",
        ]
    ].isnull().any().any():
        raise AssertionError(
            "Null SVM metrics detected."
        )

    if not raw["test_accuracy"].between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "SVM test accuracy outside [0, 1]."
        )

    if not raw["test_macro_f1"].between(
        0.0,
        1.0,
    ).all():
        raise AssertionError(
            "SVM test macro-F1 outside [0, 1]."
        )

    if len(summary) != 2:
        raise AssertionError(
            "Expected 2 SVM summary rows."
        )

    if set(
        summary["n_components"].astype(int)
    ) != {4, 8}:
        raise AssertionError(
            "SVM summary does not contain "
            "both PCA dimensions."
        )

    if not np.isclose(
        summary["n_runs"].to_numpy(dtype=float),
        3,
    ).all():
        raise AssertionError(
            "Each SVM PCA configuration must "
            "contain exactly 3 runs."
        )

    print(
        "PASS: SVM baseline contains 6 runs "
        "and complete PCA summaries."
    )


def main() -> None:
    if not FINAL_DIR.exists():
        raise FileNotFoundError(
            f"Final run directory not found: "
            f"{FINAL_DIR}"
        )

    if not RAW_JSONL.exists():
        raise FileNotFoundError(
            RAW_JSONL
        )

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            RAW_CSV
        )

    jsonl_df = load_jsonl(
        RAW_JSONL
    )

    csv_df = pd.read_csv(
        RAW_CSV
    )

    check_run_count_and_combinations(
        jsonl_df
    )

    check_metrics(
        jsonl_df
    )

    check_split_indices()

    check_jsonl_csv_agreement(
        jsonl_df,
        csv_df,
    )

    check_svm_results()

    print()
    print(
        "FINAL INTEGRITY CHECK: PASS"
    )
    print(
        "The frozen comparative result set "
        "passed all integrity checks."
    )


if __name__ == "__main__":
    main()