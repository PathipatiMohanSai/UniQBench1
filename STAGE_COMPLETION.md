# Experimental Stage Completion

| Stage | Status | Evidence boundary |
|---|---|---|
| Stage 1: reference audit, figure redesign, sensitivity analysis, error-analysis reporting, complexity/scaling section, repository release | **Completed** | Manuscript and supporting artifacts updated. Original MNIST instance-level predictions remain unavailable, so error analysis uses the external WDBC validation. |
| Stage 2: external-dataset generalization | **Completed** | WDBC external-validation study with 4/8 components, five model families, statevector evaluation, and a controlled noisy condition. |
| Stage 3: encoding-choice ablation | **Completed** | Frozen QFM and Quantum Kernel SVM at 8 components, three primary seeds, angle-RY vs ZZ-interaction feature construction. |
| Stage 4: seed expansion | **Completed** | External WDBC 8-component statevector study expanded from 3 to 10 seeds for all five model families. |

The supplemental experiments were implemented as an independent validation layer aligned with the manuscript's documented controls. They are not presented as a byte-level re-execution of every original 432-record MNIST implementation detail.
