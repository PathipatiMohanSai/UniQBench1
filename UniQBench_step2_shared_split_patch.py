from __future__ import annotations

from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "uniqbench" / "data.py"
INTEGRITY = ROOT / "integrity_check_final_comparison.py"
MANIFEST = ROOT / "create_reproducibility_manifest.py"
TESTS = ROOT / "tests" / "test_smoke.py"
README = ROOT / "README.md"
REPORT = ROOT / "uniqbench" / "report.py"

REQUIRED = [DATA, INTEGRITY, MANIFEST, TESTS, README, REPORT]
for path in REQUIRED:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# Create one-time source backups. Frozen result directories are never touched.
for path in REQUIRED:
    backup = path.with_name(path.name + ".step2_before_shared_split")
    if not backup.exists():
        shutil.copy2(path, backup)

changes: list[str] = []

# ---------------------------------------------------------------------------
# Module 2: define and use the single split helper.
# ---------------------------------------------------------------------------
data_text = DATA.read_text(encoding="utf-8")

helper = '''def make_split_indices(\n    y: np.ndarray,\n    train_size: int,\n    validation_size: int,\n    test_size: int,\n    seed: int,\n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n    """Create the deterministic stratified train/validation/test split.\n\n    This is the single split implementation used by the benchmark, the\n    integrated SVM baseline, the reproducibility manifest, and the integrity\n    checker. The second split uses seed + 1, exactly as in the final run.\n    """\n    requested = train_size + validation_size + test_size\n    if requested > len(y):\n        raise ValueError(\n            f"Requested {requested} samples but only {len(y)} labels are available."\n        )\n\n    all_indices = np.arange(len(y))\n    train_indices, remaining_indices = train_test_split(\n        all_indices,\n        train_size=train_size,\n        stratify=y,\n        random_state=seed,\n    )\n\n    validation_indices, test_indices = train_test_split(\n        remaining_indices,\n        train_size=validation_size,\n        test_size=test_size,\n        stratify=y[remaining_indices],\n        random_state=seed + 1,\n    )\n\n    return train_indices, validation_indices, test_indices\n\n\n'''

if "def make_split_indices(" not in data_text:
    marker = "def prepare_data(config: ExperimentConfig, seed: int) -> PreparedData:\n"
    if marker not in data_text:
        raise RuntimeError("Could not locate prepare_data() in data.py")
    data_text = data_text.replace(marker, helper + marker, 1)
    changes.append("data.py: added make_split_indices()")

split_pattern = re.compile(
    r"    all_indices = np\.arange\(len\(X\)\).*?"
    r"    (?:val_indices|validation_indices), test_indices = train_test_split\(.*?"
    r"    \)\n",
    re.DOTALL,
)
split_replacement = '''    train_indices, val_indices, test_indices = make_split_indices(\n        y=y,\n        train_size=config.train_size,\n        validation_size=config.validation_size,\n        test_size=config.test_size,\n        seed=seed,\n    )\n'''
if "train_indices, val_indices, test_indices = make_split_indices(" not in data_text:
    data_text, count = split_pattern.subn(split_replacement, data_text, count=1)
    if count != 1:
        raise RuntimeError("Could not replace the old split block in data.py")
    changes.append("data.py: prepare_data() now calls make_split_indices()")

# Ensure the import is available in the same module (train_test_split remains
# required by the shared helper).
DATA.write_text(data_text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Integrity checker: call the Module 2 helper instead of duplicating logic.
# ---------------------------------------------------------------------------
integrity_text = INTEGRITY.read_text(encoding="utf-8")
if "from uniqbench.data import make_split_indices" not in integrity_text:
    import_marker = "import pandas as pd\n"
    if import_marker not in integrity_text:
        raise RuntimeError("Could not locate the pandas import in integrity checker")
    integrity_text = integrity_text.replace(
        import_marker,
        import_marker + "from uniqbench.data import make_split_indices\n",
        1,
    )
    changes.append("integrity checker: imports make_split_indices()")

if "return make_split_indices(" not in integrity_text:
    reproduce_pattern = re.compile(
        r"def reproduce_indices\(.*?\n\n\ndef check_split_indices\(",
        re.DOTALL,
    )
    reproduce_replacement = '''def reproduce_indices(seed: int, y: np.ndarray):\n    """Use the exact split helper used by Module 2 and the benchmark."""\n    return make_split_indices(\n        y=y,\n        train_size=200,\n        validation_size=80,\n        test_size=80,\n        seed=seed,\n    )\n\n\ndef check_split_indices('''
    integrity_text, count = reproduce_pattern.subn(
        reproduce_replacement,
        integrity_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not locate reproduce_indices() in integrity checker")
    changes.append("integrity checker: uses Module 2 split helper")

INTEGRITY.write_text(integrity_text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Reproducibility manifest: use the same helper for saved split indices.
# ---------------------------------------------------------------------------
manifest_text = MANIFEST.read_text(encoding="utf-8")
if "from uniqbench.data import make_split_indices" not in manifest_text:
    import_marker = "from sklearn.model_selection import train_test_split\n"
    if import_marker not in manifest_text:
        raise RuntimeError("Could not locate train_test_split import in manifest script")
    manifest_text = manifest_text.replace(
        import_marker,
        import_marker + "from uniqbench.data import make_split_indices\n",
        1,
    )
    changes.append("manifest script: imports make_split_indices()")

if "make_split_indices(" not in manifest_text:
    manifest_split_pattern = re.compile(
        r"    all_indices = np\.arange\(len\(X\)\).*?"
        r"    (?:val_indices|validation_indices), test_indices = train_test_split\(.*?"
        r"    \)\n",
        re.DOTALL,
    )
    manifest_replacement = '''    train_indices, validation_indices, test_indices = make_split_indices(\n        y=y,\n        train_size=config.train_size,\n        validation_size=config.validation_size,\n        test_size=config.test_size,\n        seed=seed,\n    )\n'''
    manifest_text, count = manifest_split_pattern.subn(
        manifest_replacement,
        manifest_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace the old split block in manifest script")
    changes.append("manifest script: uses Module 2 split helper")

MANIFEST.write_text(manifest_text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Tests: add a deterministic shared-helper test.
# ---------------------------------------------------------------------------
test_text = TESTS.read_text(encoding="utf-8")
if "from uniqbench.data import make_split_indices" not in test_text:
    marker = "from uniqbench.data import prepare_data\n"
    if marker not in test_text:
        raise RuntimeError("Could not locate data imports in test_smoke.py")
    test_text = test_text.replace(
        marker,
        marker + "from uniqbench.data import make_split_indices\n",
        1,
    )
    test_text = test_text.replace(
        "import sys\n",
        "import sys\nimport numpy as np\n",
        1,
    )
    test_function = '''\n\ndef test_shared_split_helper_is_deterministic_and_disjoint():\n    y = np.array([0, 1] * 180, dtype=np.int64)\n    first = make_split_indices(y, 200, 80, 80, seed=7)\n    second = make_split_indices(y, 200, 80, 80, seed=7)\n\n    for left, right in zip(first, second):\n        assert np.array_equal(left, right)\n\n    train, validation, test = first\n    assert len(train) == 200\n    assert len(validation) == 80\n    assert len(test) == 80\n    assert len(set(train) & set(validation)) == 0\n    assert len(set(train) & set(test)) == 0\n    assert len(set(validation) & set(test)) == 0\n'''
    insertion_marker = "\n\ndef test_statevector_expectations_are_bounded():\n"
    if insertion_marker not in test_text:
        raise RuntimeError("Could not locate test insertion point")
    test_text = test_text.replace(insertion_marker, test_function + insertion_marker, 1)
    changes.append("tests: added deterministic shared-split test")

TESTS.write_text(test_text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Documentation: state the shared-helper contract without changing results.
# ---------------------------------------------------------------------------
readme_text = README.read_text(encoding="utf-8")
if "shared deterministic split helper" not in readme_text:
    target = "Standardization and PCA are fitted on the training\nset only and then applied to the validation and test sets."
    replacement = target + " The benchmark, SVM baseline, manifest, and integrity checker use the same shared deterministic split helper."
    if target in readme_text:
        readme_text = readme_text.replace(target, replacement, 1)
        changes.append("README: documented shared deterministic split helper")

README.write_text(readme_text, encoding="utf-8")

report_text = REPORT.read_text(encoding="utf-8")
if "shared deterministic split helper" not in report_text:
    if "generated by Module 2." in report_text:
        report_text = report_text.replace(
            "generated by Module 2.",
            "generated by Module 2 using the shared deterministic split helper.",
            1,
        )
        changes.append("report.py: documented shared deterministic split helper")

REPORT.write_text(report_text, encoding="utf-8")

# Final source-level checks. The script writes a small audit log in the code
# release root, not in any frozen-results directory.
checks = {
    "data_helper_present": "def make_split_indices(" in DATA.read_text(encoding="utf-8"),
    "data_uses_helper": "make_split_indices(" in DATA.read_text(encoding="utf-8"),
    "checker_uses_helper": "return make_split_indices(" in INTEGRITY.read_text(encoding="utf-8"),
    "manifest_uses_helper": "make_split_indices(" in MANIFEST.read_text(encoding="utf-8"),
    "test_uses_helper": "test_shared_split_helper_is_deterministic_and_disjoint" in TESTS.read_text(encoding="utf-8"),
    "readme_documents_helper": "shared deterministic split helper" in README.read_text(encoding="utf-8"),
    "report_documents_helper": "shared deterministic split helper" in REPORT.read_text(encoding="utf-8"),
}
if not all(checks.values()):
    raise RuntimeError(f"Step 2 patch checks failed: {checks}")

log = [
    "UniQBench Step 2 shared-split patch completed.",
    f"Release root: {ROOT}",
    "Frozen result directories were not modified.",
    "",
    "Changes:",
]
log.extend(f"- {item}" for item in changes)
log.extend(["", "Checks:"])
log.extend(f"- {key}: {value}" for key, value in checks.items())
(ROOT / "step2_shared_split_patch_audit.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("STEP 2 PATCH COMPLETED")
print(f"Changes applied: {len(changes)}")
for item in changes:
    print(f"- {item}")
print("Frozen result directories were not modified.")
print("Audit log:", ROOT / "step2_shared_split_patch_audit.txt")
