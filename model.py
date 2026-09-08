"""Module 4 — hybrid quantum-feature classifier and classical head."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .circuit import CircuitSpec, initial_parameters


@dataclass
class HybridVQC:
    spec: CircuitSpec
    seed: int
    n_outputs: int = 2
    quantum_parameters_trainable: bool = False

    def __post_init__(self) -> None:
        self.spec.validate()
        rng = np.random.default_rng(self.seed)
        self.quantum_parameters = initial_parameters(self.spec, self.seed)
        self.head_weights = (rng.normal(0.0, 0.05, size=(self.spec.n_qubits, self.n_outputs))).astype(np.float32)
        self.head_bias = np.zeros(self.n_outputs, dtype=np.float32)

    def logits(self, quantum_features: np.ndarray) -> np.ndarray:
        features = np.asarray(quantum_features, dtype=np.float32)
        return features @ self.head_weights + self.head_bias

    def probabilities(self, quantum_features: np.ndarray) -> np.ndarray:
        logits = self.logits(quantum_features)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, quantum_features: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(quantum_features), axis=1).astype(np.int64)

    def state_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "seed": self.seed,
            "n_outputs": self.n_outputs,
            "quantum_parameters_trainable": self.quantum_parameters_trainable,
            "quantum_parameters": self.quantum_parameters.tolist(),
            "head_weights": self.head_weights.tolist(),
            "head_bias": self.head_bias.tolist(),
        }

    def save(self, path) -> None:
        path.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "HybridVQC":
        state = json.loads(path.read_text(encoding="utf-8"))
        spec_data = state["spec"]
        spec = CircuitSpec(
            n_qubits=int(spec_data["n_qubits"]),
            depth=int(spec_data["depth"]),
            entanglement=spec_data["entanglement"],
            encoding=spec_data["encoding"],
            trainable_rotations=tuple(spec_data["trainable_rotations"]),
        )
        model = cls(spec, int(state["seed"]), int(state["n_outputs"]), bool(state["quantum_parameters_trainable"]))
        model.quantum_parameters = np.asarray(state["quantum_parameters"], dtype=np.float32)
        model.head_weights = np.asarray(state["head_weights"], dtype=np.float32)
        model.head_bias = np.asarray(state["head_bias"], dtype=np.float32)
        return model
