"""Module 8 — metrics and uncertainty-aware summaries."""
from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
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
        f1_scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "n_test_samples": int(len(y_true)),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot summarize an empty array.")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    sem = std / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean": mean,
        "std_sample": std,
        "ci95_low_normal": mean - 1.96 * sem,
        "ci95_high_normal": mean + 1.96 * sem,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def noise_portability_score(accuracies: np.ndarray) -> float:
    values = np.asarray(accuracies, dtype=float)
    if len(values) == 0:
        raise ValueError("NPS requires at least one accuracy value.")
    mean = float(np.mean(values))
    if np.isclose(mean, 0.0):
        return float("nan")
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(1.0 - (std / mean))
