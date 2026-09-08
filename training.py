"""Module 6 — deterministic hybrid training and validation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .backends import execute
from .circuit import CircuitSpec, normalize_angles
from .config import ExperimentConfig
from .model import HybridVQC


@dataclass
class TrainingResult:
    model: HybridVQC
    train_features: np.ndarray
    val_features: np.ndarray
    test_features: np.ndarray
    logs: list[dict[str, Any]]
    execution_records: list[dict[str, Any]]


def _cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    selected = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(selected, 1e-8, 1.0))))


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


def extract_features(
    X: np.ndarray,
    spec: CircuitSpec,
    model: HybridVQC,
    backend_name: str,
    seed: int,
    config: ExperimentConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    X_angles = normalize_angles(X)
    features = []
    records = []
    for row_id, x in enumerate(X_angles):
        result = execute(
            backend_name=backend_name,
            x=x,
            spec=spec,
            params=model.quantum_parameters,
            seed=seed + row_id,
            shots=config.shots,
            noise_probability=config.noise_probability,
        )
        features.append(result.expectations)
        records.append({
            "row_id": row_id,
            "backend": backend_name,
            "elapsed_seconds": result.elapsed_seconds,
            **result.metadata,
        })
    return np.asarray(features, dtype=np.float32), records


def train_head(
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    y_val: np.ndarray,
    model: HybridVQC,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    rng = np.random.default_rng(model.seed)
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), config.batch_size):
            batch_idx = order[start:start + config.batch_size]
            Xb, yb = train_features[batch_idx], y_train[batch_idx]
            probs = model.probabilities(Xb)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(yb)), yb] = 1.0
            grad_logits = (probs - one_hot) / len(yb)
            grad_w = Xb.T @ grad_logits
            grad_b = np.sum(grad_logits, axis=0)
            model.head_weights -= config.learning_rate * grad_w.astype(np.float32)
            model.head_bias -= config.learning_rate * grad_b.astype(np.float32)
        train_probs = model.probabilities(train_features)
        val_probs = model.probabilities(val_features)
        logs.append({
            "epoch": epoch,
            "train_loss": _cross_entropy(train_probs, y_train),
            "val_loss": _cross_entropy(val_probs, y_val),
            "train_accuracy": _accuracy(np.argmax(train_probs, axis=1), y_train),
            "val_accuracy": _accuracy(np.argmax(val_probs, axis=1), y_val),
        })
    return logs


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    spec: CircuitSpec,
    backend_name: str,
    seed: int,
    config: ExperimentConfig,
) -> TrainingResult:
    started = time.perf_counter()
    model = HybridVQC(spec=spec, seed=seed)
    train_features, train_records = extract_features(X_train, spec, model, backend_name, seed, config)
    val_features, val_records = extract_features(X_val, spec, model, backend_name, seed + 100000, config)
    test_features, test_records = extract_features(X_test, spec, model, backend_name, seed + 200000, config)
    logs = train_head(train_features, y_train, val_features, y_val, model, config)
    elapsed = time.perf_counter() - started
    execution_records = train_records + val_records + test_records
    for record in execution_records:
        record["total_training_wall_seconds"] = elapsed
    return TrainingResult(model, train_features, val_features, test_features, logs, execution_records)
