"""Module 3 — backend-neutral angle-encoded variational circuit."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CircuitSpec:
    n_qubits: int
    depth: int
    entanglement: str = "linear_cnot"
    encoding: str = "angle_ry"
    trainable_rotations: tuple[str, ...] = ("RY", "RZ")

    def validate(self) -> None:
        if self.n_qubits <= 0 or self.depth <= 0:
            raise ValueError("Circuit width and depth must be positive.")
        if self.entanglement not in {"linear_cnot", "ring_cnot"}:
            raise ValueError("Unsupported entanglement pattern.")

    @property
    def parameter_count(self) -> int:
        return self.n_qubits * self.depth * len(self.trainable_rotations)

    @property
    def nominal_gate_count(self) -> int:
        encoding_gates = self.n_qubits
        rotation_gates = self.n_qubits * self.depth * len(self.trainable_rotations)
        entangling_gates = max(0, self.n_qubits - 1) * self.depth
        return encoding_gates + rotation_gates + entangling_gates

    @property
    def nominal_two_qubit_gates(self) -> int:
        return max(0, self.n_qubits - 1) * self.depth

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "parameter_count": self.parameter_count,
            "nominal_gate_count": self.nominal_gate_count,
            "nominal_two_qubit_gates": self.nominal_two_qubit_gates,
        }


def normalize_angles(X: np.ndarray) -> np.ndarray:
    """Scale each feature column to [-pi, pi] using training-independent bounds."""
    X = np.asarray(X, dtype=np.float32)
    max_abs = np.maximum(np.max(np.abs(X), axis=0, keepdims=True), 1e-8)
    return np.clip((X / max_abs) * np.pi, -np.pi, np.pi).astype(np.float32)


def initial_parameters(spec: CircuitSpec, seed: int, scale: float = 0.01) -> np.ndarray:
    spec.validate()
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(spec.depth, spec.n_qubits, 2)).astype(np.float32)


def feature_map(x: np.ndarray) -> np.ndarray:
    """Return the angle-encoded input vector for a single sample."""
    x = np.asarray(x, dtype=np.float32)
    return np.clip(x, -np.pi, np.pi)


def circuit_metadata(spec: CircuitSpec) -> str:
    spec.validate()
    return json.dumps(spec.to_dict(), sort_keys=True)


if __name__ == "__main__":
    spec = CircuitSpec(n_qubits=4, depth=2)
    print(circuit_metadata(spec))
