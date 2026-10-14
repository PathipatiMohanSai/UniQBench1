from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
OUTPUT_DIR = ROOT / "mnist_visual_check" / "individual_examples"
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

saved = 0
for label in [0, 1]:
    indices = np.flatnonzero(y == label)[:10]
    for number, index in enumerate(indices, start=1):
        output_path = OUTPUT_DIR / f"digit_{label}_{number:02d}.png"
        plt.imsave(
            output_path,
            X[index].reshape(28, 28),
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        saved += 1

print("Filtered images:", X.shape[0])
print("Individual preview images saved:", saved)
print("Folder:", OUTPUT_DIR)
