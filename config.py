"""Module 1 — experiment configuration and reproducibility contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: str = str(Path(__file__).resolve().parents[1])
    dataset_name: str = "mnist_openml"
    classes: tuple[int, int] = (0, 1)
    n_components: tuple[int, ...] = (4, 8)
    qubit_widths: tuple[int, ...] = (4, 8)
    depths: tuple[int, ...] = (1, 2, 4)
    seeds: tuple[int, ...] = (7, 19, 31)
    train_size: int = 200
    validation_size: int = 80
    test_size: int = 80
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 0.02
    shots: int = 256
    backend_names: tuple[str, ...] = ("numpy_statevector", "noisy_shot")
    noise_probability: float = 0.01
    max_samples_per_run: int = 400
    output_dir: str = str(
        Path(__file__).resolve().parents[1]
        / "results_mnist_final_comparison_v1"
    )

    def validate(self) -> None:
        if len(self.classes) != 2 or self.classes[0] == self.classes[1]:
            raise ValueError("Exactly two distinct classes are required.")
        if any(n <= 0 for n in self.n_components):
            raise ValueError("PCA component counts must be positive.")
        if any(q <= 0 for q in self.qubit_widths):
            raise ValueError("Qubit widths must be positive.")
        if any(q != n for q, n in zip(self.qubit_widths, self.n_components)):
            raise ValueError("Each configured qubit width must match its feature count.")
        if any(d <= 0 for d in self.depths):
            raise ValueError("Circuit depths must be positive.")
        if any(s < 0 for s in self.seeds):
            raise ValueError("Seeds must be non-negative.")
        if not 0.0 <= self.noise_probability <= 1.0:
            raise ValueError("Noise probability must lie in [0, 1].")
        if self.train_size + self.validation_size + self.test_size > self.max_samples_per_run:
            raise ValueError("Configured split exceeds max_samples_per_run.")

    @property
    def root(self) -> Path:
        return Path(self.project_root)

    @property
    def results_root(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=list).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def ensure_directories(self) -> None:
        self.validate()
        for path in (
            self.results_root / "raw",
            self.results_root / "processed",
            self.results_root / "figures",
            self.root / "reports",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, path: Path | None = None) -> Path:
        self.ensure_directories()
        destination = path or self.results_root / "processed" / "run_manifest.json"
        manifest = {"config": self.to_dict(), "config_hash": self.config_hash()}
        destination.write_text(json.dumps(manifest, indent=2, default=list), encoding="utf-8")
        return destination


def default_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.validate()
    return config


if __name__ == "__main__":
    cfg = default_config()
    print(json.dumps({"config_hash": cfg.config_hash(), **cfg.to_dict()}, indent=2, default=list))
