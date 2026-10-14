# UniQBench

UniQBench is a reproducibility-oriented benchmark for evaluating
hybrid quantum-classical classification pipelines under controlled
experimental conditions.

## Final MNIST Comparative Experiment

The final comparative experiment uses:

- **Dataset:** MNIST, OpenML `mnist_784`
- **OpenML version:** 1
- **Classes:** digits `0` and `1`
- **Label mapping:** `0 → 0`, `1 → 1`
- **Training samples:** 200
- **Validation samples:** 80
- **Test samples:** 80
- **PCA dimensions:** 4 and 8
- **Qubit widths:** 4 and 8
- **Circuit depths:** 1, 2, and 4
- **Seeds:** 7, 19, and 31

The data split is deterministic and stratified. Pixel values are
scaled by `1/255`. Standardization and PCA are fitted on the training
set only and then applied to the validation and test sets.

## Quantum Model

The evaluated quantum model is a:

> **Fixed variational quantum feature map with a trainable classical
> prediction head.**

The variational quantum parameters are initialized once and held fixed
during the reported training procedure. Only the classical
prediction-head weights and biases are optimized.

Therefore, the implementation should **not** be described as a fully
trainable variational quantum classifier or as an end-to-end trainable
VQC.

## Quantum Backends

The final benchmark evaluates:

- `numpy_statevector`
- `noisy_shot`

The `noisy_shot` backend is a transparent shot-and-noise simulation
approximation. It is not execution on physical quantum hardware.

The benchmark uses:

```text
shots = 256
noise_probability = 0.01