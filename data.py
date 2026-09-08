"""Module 2 — deterministic dataset preparation and PCA."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    explained_variance_ratio: list[float]
    metadata: dict[str, Any]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            directory / "prepared_data.npz",
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            X_test=self.X_test,
            y_test=self.y_test,
        )

        (directory / "data_metadata.json").write_text(
            json.dumps(self.metadata, indent=2),
            encoding="utf-8",
        )


def _dataset_hash(X: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()[:16]


def make_split_indices(
    y: np.ndarray,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the deterministic stratified train/validation/test split.

    This is the single split implementation used by the benchmark, the
    integrated SVM baseline, the reproducibility manifest, and the integrity
    checker. The second split uses seed + 1, exactly as in the final run.
    """
    requested = train_size + validation_size + test_size
    if requested > len(y):
        raise ValueError(
            f"Requested {requested} samples but only {len(y)} labels are available."
        )

    all_indices = np.arange(len(y))
    train_indices, remaining_indices = train_test_split(
        all_indices,
        train_size=train_size,
        stratify=y,
        random_state=seed,
    )

    validation_indices, test_indices = train_test_split(
        remaining_indices,
        train_size=validation_size,
        test_size=test_size,
        stratify=y[remaining_indices],
        random_state=seed + 1,
    )

    return train_indices, validation_indices, test_indices


def prepare_data(config: ExperimentConfig, seed: int) -> PreparedData:
    """Load MNIST, create deterministic stratified splits, and apply
    training-only standardization and PCA.
    """
    config.validate()

    # ---------------------------------------------------------
    # 1. Local cache directory for OpenML data
    # ---------------------------------------------------------
    cache_dir = config.root / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Load MNIST from OpenML
    # ---------------------------------------------------------
    mnist = fetch_openml(
        name="mnist_784",
        version=1,
        as_frame=False,
        data_home=str(cache_dir),
    )

    X = np.asarray(mnist.data, dtype=np.float32) / 255.0
    y_raw = np.asarray(mnist.target, dtype=np.int64)

    # ---------------------------------------------------------
    # 3. Keep only classes 0 and 1
    # ---------------------------------------------------------
    mask = np.isin(y_raw, config.classes)

    X = X[mask]
    y_raw = y_raw[mask]

    # Explicit label mapping:
    # 0 -> 0
    # 1 -> 1
    label_map = {
        config.classes[0]: 0,
        config.classes[1]: 1,
    }

    y = np.array(
        [label_map[int(value)] for value in y_raw],
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # 4. Check requested sample count
    # ---------------------------------------------------------
    requested = (
        config.train_size
        + config.validation_size
        + config.test_size
    )

    if requested > len(X):
        raise ValueError(
            f"Requested {requested} samples but only "
            f"{len(X)} MNIST samples are available."
        )

    # ---------------------------------------------------------
    # 5. Deterministic stratified train split
    #
    # Train = 200
    # Remaining = 160
    # ---------------------------------------------------------
    train_indices, val_indices, test_indices = make_split_indices(
        y=y,
        train_size=config.train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        seed=seed,
    )

    # ---------------------------------------------------------
    # 7. Build raw train/validation/test datasets
    # ---------------------------------------------------------
    X_train_raw = X[train_indices]
    y_train = y[train_indices]

    X_val_raw = X[val_indices]
    y_val = y[val_indices]

    X_test_raw = X[test_indices]
    y_test = y[test_indices]

    # ---------------------------------------------------------
    # 8. Training-only standardization
    #
    # IMPORTANT:
    # The scaler is fitted only on X_train_raw.
    # Validation and test data are transformed using the
    # training-fitted scaler.
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # ---------------------------------------------------------
    # 9. Training-only PCA
    #
    # PCA is fitted only on standardized training data.
    # ---------------------------------------------------------
    n_components = int(max(config.n_components))

    pca = PCA(
        n_components=n_components,
        random_state=seed,
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    ).astype(np.float32)

    X_val_pca = pca.transform(
        X_val_scaled
    ).astype(np.float32)

    X_test_pca = pca.transform(
        X_test_scaled
    ).astype(np.float32)

    # ---------------------------------------------------------
    # 10. Reproducibility metadata
    # ---------------------------------------------------------
    metadata = {
        "dataset_name": config.dataset_name,
        "source": "OpenML mnist_784",
        "openml_version": 1,
        "original_shape": list(mnist.data.shape),
        "filtered_shape": list(X.shape),
        "classes": list(config.classes),
        "label_map": {
            str(key): value
            for key, value in label_map.items()
        },
        "pixel_scaling": "divide_by_255",
        "split_method": "stratified_train_validation_test",
        "seed": seed,
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "class_counts": {
            "train": np.bincount(
                y_train,
                minlength=2,
            ).tolist(),
            "validation": np.bincount(
                y_val,
                minlength=2,
            ).tolist(),
            "test": np.bincount(
                y_test,
                minlength=2,
            ).tolist(),
        },
        "standardization": "training_only",
        "pca": "training_only",
        "pca_components": n_components,
        "explained_variance_ratio": (
            pca.explained_variance_ratio_.tolist()
        ),
        "explained_variance_cumulative": (
            np.cumsum(
                pca.explained_variance_ratio_
            ).tolist()
        ),
        "dataset_hash": _dataset_hash(X, y),
    }

    return PreparedData(
        X_train=X_train_pca,
        y_train=y_train,
        X_val=X_val_pca,
        y_val=y_val,
        X_test=X_test_pca,
        y_test=y_test,
        explained_variance_ratio=(
            pca.explained_variance_ratio_.tolist()
        ),
        metadata=metadata,
    )


def select_components(
    data: PreparedData,
    n_components: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return the requested leading PCA components."""
    if n_components > data.X_train.shape[1]:
        raise ValueError(
            "Requested components exceed prepared PCA dimensions."
        )

    return (
        data.X_train[:, :n_components],
        data.y_train,
        data.X_val[:, :n_components],
        data.y_val,
        data.X_test[:, :n_components],
        data.y_test,
    )


if __name__ == "__main__":
    from .config import default_config

    cfg = default_config()
    prepared = prepare_data(
        cfg,
        seed=cfg.seeds[0],
    )

    prepared.save(
        cfg.results_root / "processed"
    )

    print(
        json.dumps(
            prepared.metadata,
            indent=2,
        )
    )


