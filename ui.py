import numpy as np
print("loaded numpy")
import streamlit as st
print("loaded streamlit")
from PIL import Image, ImageOps
print("loaded PIL")
from streamlit_drawable_canvas import st_canvas
print("loaded st_canvas")
import tensorflow as tf
print("loaded tensorflow")

# Label map
emnist_map = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

model = tf.keras.models.load_model("handwritten_cnn.keras")

# Center non-center images and preprocess to feed into model
def preprocess(img):
    img = img.convert("L") # Convert to grayscale
    img = ImageOps.invert(img) # Invert colors

    arr = np.array(img) # Convert to numpy array
    mask = arr > 20 # Find non-white pixels and creates a numpy array that has true in all the places with non-empty pixels

    coords = np.column_stack(np.where(mask)) # Combines row and column indices to make a list of coordinates where there are non-empty pixels, lets something like image1 edge case work
    if coords.size == 0: # Empty drawing
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    y0, x0 = coords.min(axis=0) # Get bottom left corner
    y1, x1 = coords.max(axis=0) # Get top right corner

    cropped = arr[y0:y1 + 1, x0:x1 + 1] # Crop to non-empty pixels
    cropped = Image.fromarray(cropped).resize((20, 20)) # 20x20 because emnist characters are typically within this size box

    canvas = Image.new("L", (28, 28), 0) # Make new grayscale 28x28 image with black background
    canvas.paste(cropped, (4, 4)) # Paste the 20x20 cropped image onto the center of the 28x28 black canvas

    final = np.array(canvas) / 255.0 # Normalize to [0, 1]
    return final.reshape(1, 28, 28, 1) # Reshape to feed into model

# UI
st.title("Handwritten Character Recognizer")

canvas = st_canvas( # Drawing Canvas
    fill_color="rgba(255,255,255,1)",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict Button
if st.button("Predict"):
    if canvas.image_data is not None: # Make sure something is drawn
        pil_img = Image.fromarray(canvas.image_data.astype("uint8"))
        processed = preprocess(pil_img)

        prediction = model.predict(processed)
        idx = np.argmax(prediction) # Get highest probability index

        st.success(f"Prediction: **{emnist_map[idx]}**") # Translate
    else:
        st.error("Draw something first!")