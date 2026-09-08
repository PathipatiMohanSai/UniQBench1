"""Module 5 — local backend adapters for ideal and noisy execution."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .circuit import CircuitSpec, feature_map


@dataclass
class ExecutionResult:
    expectations: np.ndarray
    elapsed_seconds: float
    backend_name: str
    metadata: dict[str, Any]


def _ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(angle: float) -> np.ndarray:
    return np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]], dtype=np.complex128)


def _apply_single(state: np.ndarray, gate: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, wire, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, wire)
    return tensor.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    result = state.copy()
    for index in range(len(state)):
        control_bit = (index >> (n_qubits - 1 - control)) & 1
        target_bit = (index >> (n_qubits - 1 - target)) & 1
        if control_bit == 1 and target_bit == 0:
            flipped = index | (1 << (n_qubits - 1 - target))
            result[index], result[flipped] = state[flipped], state[index]
    return result


def _expectation_z(state: np.ndarray, wire: int, n_qubits: int) -> float:
    total = 0.0
    for index, amplitude in enumerate(state):
        bit = (index >> (n_qubits - 1 - wire)) & 1
        total += (1.0 if bit == 0 else -1.0) * float(np.abs(amplitude) ** 2)
    return float(np.real(total))


def run_statevector(x: np.ndarray, spec: CircuitSpec, params: np.ndarray, backend_name: str = "numpy_statevector") -> ExecutionResult:
    start = time.perf_counter()
    spec.validate()
    angles = feature_map(x)
    state = np.zeros(2 ** spec.n_qubits, dtype=np.complex128)
    state[0] = 1.0
    for wire in range(spec.n_qubits):
        state = _apply_single(state, _ry(float(angles[wire])), wire, spec.n_qubits)
    for layer in range(spec.depth):
        for wire in range(spec.n_qubits):
            state = _apply_single(state, _ry(float(params[layer, wire, 0])), wire, spec.n_qubits)
            state = _apply_single(state, _rz(float(params[layer, wire, 1])), wire, spec.n_qubits)
        for wire in range(spec.n_qubits - 1):
            state = _apply_cnot(state, wire, wire + 1, spec.n_qubits)
    expectations = np.array([_expectation_z(state, w, spec.n_qubits) for w in range(spec.n_qubits)], dtype=np.float32)
    return ExecutionResult(
        expectations=expectations,
        elapsed_seconds=time.perf_counter() - start,
        backend_name=backend_name,
        metadata={"shots": None, "noise_probability": 0.0, "execution_mode": "statevector"},
    )


def run_noisy_shot(
    x: np.ndarray,
    spec: CircuitSpec,
    params: np.ndarray,
    seed: int,
    shots: int,
    noise_probability: float,
) -> ExecutionResult:
    base = run_statevector(x, spec, params, backend_name="noisy_shot")
    rng = np.random.default_rng(seed)
    noisy = base.expectations.copy()
    # A transparent shot-and-noise approximation around the ideal expectation.
    noisy *= (1.0 - 2.0 * noise_probability) ** spec.depth
    probabilities_zero = np.clip((1.0 + noisy) / 2.0, 0.0, 1.0)
    counts_zero = rng.binomial(shots, probabilities_zero)
    sampled = (2.0 * counts_zero / shots) - 1.0
    return ExecutionResult(
        expectations=sampled.astype(np.float32),
        elapsed_seconds=base.elapsed_seconds,
        backend_name="noisy_shot",
        metadata={
            "shots": shots,
            "noise_probability": noise_probability,
            "execution_mode": "shot_approximation",
        },
    )


def execute(
    backend_name: str,
    x: np.ndarray,
    spec: CircuitSpec,
    params: np.ndarray,
    seed: int,
    shots: int,
    noise_probability: float,
) -> ExecutionResult:
    if backend_name == "numpy_statevector":
        return run_statevector(x, spec, params)
    if backend_name == "noisy_shot":
        return run_noisy_shot(x, spec, params, seed, shots, noise_probability)
    raise ValueError(f"Unknown backend: {backend_name}")
