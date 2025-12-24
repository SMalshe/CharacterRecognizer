import os
import numpy as np
import cv2
import tensorflow as tf
import gzip
import struct
import matplotlib.pyplot as plt

# Helper Function Definitions
def load_images(path):
    with gzip.open(path, "rb") as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)

def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# Data Loading
x_train = load_images("gzip/emnist-balanced-train-images-idx3-ubyte.gz")
y_train = load_labels("gzip/emnist-balanced-train-labels-idx1-ubyte.gz")

x_test  = load_images("gzip/emnist-balanced-test-images-idx3-ubyte.gz")
y_test  = load_labels("gzip/emnist-balanced-test-labels-idx1-ubyte.gz")

# Image Orientation
x_train = np.rot90(x_train, k=1, axes=(1, 2))
x_train = np.fliplr(x_train)
x_test  = np.rot90(x_test, k=1, axes=(1, 2))
x_test  = np.fliplr(x_test)

# Input Reduction
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test  = x_test[..., np.newaxis]

NUM_CLASSES = 47

# ================= CNN MODEL =================
cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.fit(x_train, y_train, epochs=3)
cnn_model.save('handwritten_cnn.keras')

# ================= DENSE MODEL =================
dense_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

dense_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

dense_model.fit(x_train, y_train, epochs=3)
dense_model.save('handwritten_dense.keras')

# ================= EVALUATION =================
print("\nCNN model evaluation:")
cnn_model.evaluate(x_test, y_test)

print("\nDense model evaluation:")
dense_model.evaluate(x_test, y_test)
