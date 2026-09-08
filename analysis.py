"""Module 9 — aggregation, comparisons, and figure generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import noise_portability_score, summarize


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("No benchmark records found.")
    return pd.DataFrame(records)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["n_components", "n_qubits", "depth", "backend"]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        for metric in ["accuracy", "macro_f1", "mean_confidence", "training_wall_seconds"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        row["n_runs"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def nps_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["n_components", "n_qubits", "depth"], dropna=False):
        row = {
            "n_components": keys[0],
            "n_qubits": keys[1],
            "depth": keys[2],
            "nps_across_backend_run_means": noise_portability_score(group.groupby("backend")["accuracy"].mean().to_numpy()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_components", "n_qubits", "depth"])


def write_tables(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_results(df)
    nps = nps_table(df)
    raw_summary = pd.DataFrame({"metric": ["accuracy", "macro_f1", "training_wall_seconds"]})
    summary_rows = []
    for metric in raw_summary["metric"]:
        summary_rows.append({"metric": metric, **summarize(df[metric].to_numpy())})
    summary = pd.DataFrame(summary_rows)
    paths = {
        "aggregate": output_dir / "aggregate_results.csv",
        "nps": output_dir / "nps_results.csv",
        "summary": output_dir / "overall_summary.csv",
    }
    aggregate.to_csv(paths["aggregate"], index=False)
    nps.to_csv(paths["nps"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths


def generate_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    agg = aggregate_results(df)

    plt.figure(figsize=(8, 5))
    for backend, group in agg.groupby("backend"):
        key = group.groupby("n_qubits", as_index=False)["accuracy_mean"].mean().sort_values("n_qubits")
        plt.plot(key["n_qubits"], key["accuracy_mean"], marker="o", label=backend)
    plt.xlabel("Qubit width")
    plt.ylabel("Mean test accuracy")
    plt.title("UniQBench backend accuracy comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    path = output_dir / "accuracy_by_backend.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(); paths.append(path)

    plt.figure(figsize=(8, 5))
    pivot = agg.pivot_table(index="depth", columns="backend", values="accuracy_mean", aggfunc="mean")
    pivot.plot(kind="bar", ax=plt.gca())
    plt.xlabel("Circuit depth")
    plt.ylabel("Mean test accuracy")
    plt.title("Accuracy by circuit depth")
    plt.grid(axis="y", alpha=0.25)
    path = output_dir / "accuracy_by_depth.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(); paths.append(path)

    plt.figure(figsize=(8, 5))
    for backend, group in agg.groupby("backend"):
        key = group.groupby("n_qubits", as_index=False)["training_wall_seconds_mean"].mean()
        plt.plot(key["n_qubits"], key["training_wall_seconds_mean"], marker="o", label=backend)
    plt.xlabel("Qubit width")
    plt.ylabel("Mean wall-clock seconds")
    plt.title("Training time by backend and qubit width")
    plt.legend(); plt.grid(alpha=0.25)
    path = output_dir / "training_time_by_backend.png"
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close(); paths.append(path)
    return paths
