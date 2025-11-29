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
        data = np.frombuffer(f.read(), dtype = np.uint8)
        return data.reshape(num_images, rows, cols)

def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# For streamlit
def load_streamlit_corrections(path="data/streamlit_corrections.npy"):
    if not os.path.exists(path):
        print("No streamlit corrections found.")
        return None, None

    images, labels = np.load(path, allow_pickle=True)
    print(f"Loaded {len(images)} corrected samples from Streamlit.")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.uint8)


# Data Loading
x_train = load_images("gzip/emnist-balanced-train-images-idx3-ubyte.gz")
y_train = load_labels("gzip/emnist-balanced-train-labels-idx1-ubyte.gz")

x_test  = load_images("gzip/emnist-balanced-test-images-idx3-ubyte.gz")
y_test  = load_labels("gzip/emnist-balanced-test-labels-idx1-ubyte.gz")

# Image Orientation
x_train = np.rot90(x_train, k = 1, axes = (1, 2))
x_train = np.fliplr(x_train)
x_test = np.rot90(x_test, k = 1, axes = (1, 2))
x_test = np.fliplr(x_test)

# Input Reduction
x_train = x_train/255.0
x_test = x_test/255.0

# Model - CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(47, activation='softmax')
])

# Compile Model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# Load Streamlit corrections (persistent)
corr_images, corr_labels = load_streamlit_corrections()

if corr_images is not None:
    corr_images = corr_images.reshape((-1, 28, 28, 1))
    x_train = np.concatenate([x_train, corr_images], axis=0)
    y_train = np.concatenate([y_train, corr_labels], axis=0)
    print(f"Added {len(corr_images)} corrected samples.")
    print(f"New total training samples: {len(x_train)}")
else:
    print("No extra training samples found.")

model.fit(x_train, y_train, epochs = 5)

# Save Model
model.save('handwritten_characters.keras')
model = tf.keras.models.load_model('handwritten_characters.keras')

# Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test)

# Custom Image Testing
image_number = 0
while os.path.isfile(f"Character{image_number}.png"):
    try:
        img = cv2.imread(f"Character{image_number}.png")[:,:,0]
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.invert(img)
        img = img.reshape(1, 28, 28, 1)   # batch = 1 for one image, height, width, channels
        prediction = model.predict(img)
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
        print(f"The character is probably: {np.argmax(prediction)}")
    except FileNotFoundError:
        print("File wasn't found")
    except IndexError:
        print("Index out of range")
    except:
        print("Unexpected error:", sys.exc_info()[0])
    finally: image_number += 1

# Edge Case Testing
image_number = 1

emnist_map = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

while os.path.isfile(f"Character_{image_number}.png"):
    try:
        img = cv2.imread(f"Character_{image_number}.png")[:,:,0]
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.invert(img)
        img = img.reshape(1, 28, 28, 1)   # batch = 1 for one image, height, width, channels

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        predicted_char = emnist_map[class_index]

        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()

        print(f"The character is probably: {predicted_char}")

    except FileNotFoundError:
        print("File wasn't found")
    except IndexError:
        print("Index out of range")
    except:
        print("Unexpected error:", sys.exc_info()[0])
    finally: image_number += 1