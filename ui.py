import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# EMNIST map
emnist_map = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]


#Load model
@st.cache_resource
def load_model():
    if RUNNING_IN_STREAMLIT_CLOUD:
        interpreter = tflite.Interpreter(model_path="handwritten_characters.tflite")
        interpreter.allocate_tensors()
        return interpreter, interpreter.get_input_details(), interpreter.get_output_details()
    else:
        model = tf.keras.models.load_model("handwritten_characters.keras")
        return model, None, None

model, input_details, output_details = load_model()


# Send image to model
def predict(img_array):
    if RUNNING_IN_STREAMLIT_CLOUD:
        model.set_tensor(input_details[0]['index'], img_array)
        model.invoke()
        return model.get_tensor(output_details[0]['index'])
    else:
        return model.predict(img_array)


Find image anywhere in the box
def preprocess(img):
    img = img.convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)
    mask = arr > 20  # threshold to detect drawing

    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        blank = np.zeros((28, 28), dtype=np.float32)
        return blank.reshape(1, 28, 28, 1)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = arr[y0:y1+1, x0:x1+1]
    cropped = Image.fromarray(cropped).resize((20, 20))

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(cropped, (4, 4))

    final = np.array(canvas) / 255.0
    return final.reshape(1, 28, 28, 1)


# ============================================================================
# STREAMLIT STATE
# ============================================================================
st.session_state.setdefault("train_images", stored_images.copy())
st.session_state.setdefault("train_labels", stored_labels.copy())
st.session_state.setdefault("last_image", None)
st.session_state.setdefault("last_prediction", None)
st.session_state.setdefault("awaiting_correction", False)


# ============================================================================
# UI
# ============================================================================
st.title("✏️ Handwritten Character Recognizer")
st.write("If prediction is wrong, correct it — your corrections are saved permanently.")

st.subheader("Draw Here:")

canvas = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas_main"
)


# Predict Burron
if st.button("Predict"):
    if canvas.image_data is not None:
        pil_img = Image.fromarray(canvas.image_data.astype("uint8"))
        processed = preprocess(pil_img)

        # store clean image (28,28,1)
        st.session_state.last_image = processed.reshape(28, 28, 1)

        preds = predict(processed)
        idx = np.argmax(preds)
        pred_char = emnist_map[idx]

        st.session_state.last_prediction = pred_char
        st.session_state.awaiting_correction = False

        st.success(f"Prediction: **{pred_char}**")
    else:
        st.error("Please draw something first.")
