"""Thin orchestration entry point; scientific logic remains in the ten modules."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from uniqbench.config import default_config
from uniqbench.benchmark import run_benchmark, jsonl_to_csv
from uniqbench.analysis import (
    load_results,
    write_tables,
    generate_figures,
)
from uniqbench.metrics import (
    run_svm_baseline,
    write_svm_results,
)
from uniqbench.report import generate_report


def main() -> None:
    config = default_config()

    config.write_manifest()

    # ---------------------------------------------------------
    # Quantum benchmark
    # ---------------------------------------------------------
    raw_path = run_benchmark(config)

    csv_path = jsonl_to_csv(
        raw_path,
        config.results_root
        / "raw"
        / "benchmark_runs.csv",
    )

    df = load_results(raw_path)

    table_paths = write_tables(
        df,
        config.results_root / "processed",
    )

    figure_paths = generate_figures(
        df,
        config.results_root / "figures",
    )

    # ---------------------------------------------------------
    # Classical SVM baseline — Module 8
    # ---------------------------------------------------------
    svm_records = run_svm_baseline(config)

    svm_paths = write_svm_results(
        svm_records,
        config.results_root / "processed",
    )

    # ---------------------------------------------------------
    # Module 10 report
    # ---------------------------------------------------------
    report_path = generate_report(
        config,
        raw_path,
        svm_summary_path=svm_paths["svm_summary"],
    )

    print("UniQBench comparative run complete")
    print(f"Raw JSONL: {raw_path}")
    print(f"Raw CSV:   {csv_path}")

    for name, path in table_paths.items():
        print(f"{name}: {path}")

    for name, path in svm_paths.items():
        print(f"{name}: {path}")

    for path in figure_paths:
        print(f"figure: {path}")

    print(f"report: {report_path}")


if __name__ == "__main__":
    main()