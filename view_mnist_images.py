from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
OUTPUT_DIR = ROOT / "mnist_visual_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mnist = fetch_openml(
    name="mnist_784",
    version=1,
    as_frame=False,
    data_home=str(CACHE_DIR),
)

X = np.asarray(mnist.data, dtype=np.float32)
y = np.asarray(mnist.target, dtype=np.int64)

mask = np.isin(y, [0, 1])
X = X[mask]
y = y[mask]

fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))

for row, label in enumerate([0, 1]):
    label_indices = np.flatnonzero(y == label)[:5]
    for col, index in enumerate(label_indices):
        axes[row, col].imshow(X[index].reshape(28, 28), cmap="gray")
        axes[row, col].set_title(f"Digit {label}")
        axes[row, col].axis("off")

fig.suptitle("OpenML MNIST examples after filtering to digits 0 and 1")
fig.tight_layout()

output_path = OUTPUT_DIR / "mnist_examples_digits_0_1.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Dataset:", "mnist_784")
print("Version:", 1)
print("Filtered shape:", X.shape)
print("Classes:", sorted(np.unique(y).tolist()))
print("Saved image:", output_path)
