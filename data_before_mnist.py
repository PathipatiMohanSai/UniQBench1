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
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )


def _dataset_hash(X: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    return digest.hexdigest()[:16]


def prepare_data(config: ExperimentConfig, seed: int) -> PreparedData:
    config.validate()

    # ---------------------------------------------------------
    # 1. Load MNIST from OpenML: mnist_784, version 1
    # ---------------------------------------------------------
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
    )

    X_all = np.asarray(mnist.data, dtype=np.float32)
    y_all = np.asarray(mnist.target, dtype=np.int64)

    # ---------------------------------------------------------
    # 2. Keep only classes 0 and 1
    # ---------------------------------------------------------
    mask = np.isin(y_all, config.classes)

    X = X_all[mask]
    y_raw = y_all[mask]

    # Explicit label mapping: 0 -> 0, 1 -> 1
    label_map = {
        config.classes[0]: 0,
        config.classes[1]: 1,
    }

    y = np.array(
        [label_map[int(v)] for v in y_raw],
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # 3. Pixel scaling: 0-255 -> 0-1
    # ---------------------------------------------------------
    X = X / 255.0

    # ---------------------------------------------------------
    # 4. Stratified deterministic split
    #
    # Total = 200 train + 80 validation + 80 test = 360
    # ---------------------------------------------------------
    requested = (
        config.train_size
        + config.validation_size
        + config.test_size
    )

    if requested > len(X):
        raise ValueError(
            f"Requested {requested} samples but only {len(X)} are available."
        )

    # First create the required 360-sample subset using a
    # deterministic stratified split.
    X_subset, _, y_subset, _ = train_test_split(
        X,
        y,
        train_size=requested,
        stratify=y,
        random_state=seed,
    )

    # ---------------------------------------------------------
    # 5. Split 360 samples into:
    #    train = 200
    #    validation = 80
    #    test = 80
    # ---------------------------------------------------------
    temp_size = config.validation_size + config.test_size

    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X_subset,
        y_subset,
        train_size=config.train_size,
        stratify=y_subset,
        random_state=seed,
    )

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=config.test_size,
        stratify=y_temp,
        random_state=seed,
    )

    # ---------------------------------------------------------
    # 6. Training-only standardization
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # ---------------------------------------------------------
    # 7. Training-only PCA
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
    # 8. Reproducibility metadata
    # ---------------------------------------------------------
    metadata = {
        "dataset_name": config.dataset_name,
        "source": "OpenML mnist_784",
        "version": 1,
        "original_shape": list(X_all.shape),
        "filtered_shape": list(X.shape),
        "classes": list(config.classes),
        "label_map": {
            str(k): v for k, v in label_map.items()
        },
        "pixel_scaling": "divide_by_255",
        "split_method": "stratified_train_validation_test",
        "seed": seed,
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
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
        "dataset_hash": _dataset_hash(
            X_subset,
            y_subset,
        ),
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


def select_components(data: PreparedData, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_components > data.X_train.shape[1]:
        raise ValueError("Requested components exceed prepared PCA dimensions.")
    return (
        data.X_train[:, :n_components], data.y_train,
        data.X_val[:, :n_components], data.y_val,
        data.X_test[:, :n_components], data.y_test,
    )


if __name__ == "__main__":
    from .config import default_config
    cfg = default_config()
    prepared = prepare_data(cfg, seed=cfg.seeds[0])
    prepared.save(cfg.results_root / "processed")
    print(json.dumps(prepared.metadata, indent=2))
