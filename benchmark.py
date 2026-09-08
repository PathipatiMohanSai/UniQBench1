"""Module 7 — benchmark matrix orchestration and raw result recording."""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .circuit import CircuitSpec
from .config import ExperimentConfig
from .data import prepare_data, select_components
from .metrics import classification_metrics
from .training import run_training


def run_benchmark(config: ExperimentConfig, raw_path: Path | None = None) -> Path:
    config.ensure_directories()
    destination = raw_path or (config.results_root / "raw" / "benchmark_runs.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        prepared = prepare_data(config, seed)
        for n_components, n_qubits in zip(config.n_components, config.qubit_widths):
            X_train, y_train, X_val, y_val, X_test, y_test = select_components(prepared, n_components)
            for depth in config.depths:
                spec = CircuitSpec(n_qubits=n_qubits, depth=depth)
                for backend_name in config.backend_names:
                    started = time.perf_counter()
                    result = run_training(
                        X_train, y_train, X_val, y_val, X_test,
                        spec=spec,
                        backend_name=backend_name,
                        seed=seed,
                        config=config,
                    )
                    test_pred = result.model.predict(result.test_features)
                    test_probs = result.model.probabilities(result.test_features)
                    metrics = classification_metrics(y_test, test_pred, test_probs)
                    row = {
                        "run_id": f"s{seed}_q{n_qubits}_d{depth}_{backend_name}",
                        "seed": seed,
                        "dataset": config.dataset_name,
                        "n_components": n_components,
                        "n_qubits": n_qubits,
                        "depth": depth,
                        "backend": backend_name,
                        "shots": config.shots if backend_name == "noisy_shot" else None,
                        "noise_probability": config.noise_probability if backend_name == "noisy_shot" else 0.0,
                        "train_size": len(y_train),
                        "validation_size": len(y_val),
                        "test_size": len(y_test),
                        "parameter_count": spec.parameter_count,
                        "nominal_gate_count": spec.nominal_gate_count,
                        "nominal_two_qubit_gates": spec.nominal_two_qubit_gates,
                        "training_wall_seconds": time.perf_counter() - started,
                        "config_hash": config.config_hash(),
                        **metrics,
                    }
                    rows.append(row)
                    run_dir = config.results_root / "processed" / "checkpoints"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / f"{row['run_id']}.json").write_text(
                        json.dumps({"model": result.model.state_dict(), "logs": result.logs}, indent=2),
                        encoding="utf-8",
                    )
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return destination


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> Path:
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("No records available for CSV conversion.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return csv_path
