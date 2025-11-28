'''
For viewing EMNIST images
'''

import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt

# Helper Functions

def load_images(path):
    with gzip.open(path, "rb") as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)

def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load EMNIST

x_train = load_images("gzip/emnist-balanced-train-images-idx3-ubyte.gz")
y_train = load_labels("gzip/emnist-balanced-train-labels-idx1-ubyte.gz")

# Display Images

plt.figure(figsize=(8, 8))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
