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

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("handwritten_cnn.keras")

model = load_model()

# Center non-center images and preprocess to feed into model
def preprocess(img):
    img = img.convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)
    mask = arr > 20

    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = arr[y0:y1 + 1, x0:x1 + 1]
    cropped = Image.fromarray(cropped).resize((20, 20))

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(cropped, (4, 4))

    final = np.array(canvas) / 255.0
    return final.reshape(1, 28, 28, 1)

# UI
st.title("✏️ Handwritten Character Recognizer")

canvas = st_canvas(
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
    if canvas.image_data is not None:
        pil_img = Image.fromarray(canvas.image_data.astype("uint8"))
        processed = preprocess(pil_img)

        prediction = model.predict(processed)
        idx = np.argmax(prediction)

        st.success(f"Prediction: **{emnist_map[idx]}**")
    else:
        st.error("Draw something first!")