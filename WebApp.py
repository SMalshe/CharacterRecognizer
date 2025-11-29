import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# =============================================================================
# ENVIRONMENT CHECK
# =============================================================================
RUNNING_IN_STREAMLIT_CLOUD = "STREAMLIT_SERVER_URL" in os.environ

if RUNNING_IN_STREAMLIT_CLOUD:
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow as tf


# =============================================================================
# EMNIST CLASS MAP
# =============================================================================
emnist_map = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# =============================================================================
# MODEL LOADING
# =============================================================================
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


# =============================================================================
# PREDICT FUNCTION
# =============================================================================
def predict(img):
    if RUNNING_IN_STREAMLIT_CLOUD:
        model.set_tensor(input_details[0]["index"], img)
        model.invoke()
        return model.get_tensor(output_details[0]["index"])
    else:
        return model.predict(img)


# =============================================================================
# BOUNDING BOX PREPROCESSING
# =============================================================================
def preprocess(img):
    img = img.convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)
    mask = arr > 20
    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        blank = np.zeros((28, 28, 1), dtype=np.float32)
        return blank.reshape(1, 28, 28, 1)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    cropped = arr[y0:y1+1, x0:x1+1]
    cropped = Image.fromarray(cropped).resize((20, 20))

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(cropped, (4, 4))

    processed = np.array(canvas).astype(np.float32) / 255.0
    return processed.reshape(1, 28, 28, 1)


# =============================================================================
# STREAMLIT STATE
# =============================================================================
st.session_state.setdefault("last_image", None)
st.session_state.setdefault("last_prediction", None)
st.session_state.setdefault("awaiting_correction", False)


# =============================================================================
# UI
# =============================================================================
st.title("✏️ Handwritten Character Recognizer")
st.write("Draw a character and let the model guess it!")

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

# =============================================================================
# PREDICT
# =============================================================================
if st.button("Predict"):
    if canvas.image_data is not None:
        pil = Image.fromarray(canvas.image_data.astype("uint8"))
        processed = preprocess(pil)

        st.session_state.last_image = processed
        probs = predict(processed)
        pred_idx = np.argmax(probs)
        pred_char = emnist_map[pred_idx]

        st.session_state.last_prediction = pred_char
        st.session_state.awaiting_correction = False

        st.success(f"Prediction: **{pred_char}**")
    else:
        st.error("Please draw something first!")

# =============================================================================
# CORRECTION (NOT SAVED)
# =============================================================================
if st.session_state.last_prediction:

    st.subheader("Was the prediction correct?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Yes"):
            st.success("Awesome!")
            st.session_state.awaiting_correction = False

    with col2:
        if st.button("👎 No"):
            st.session_state.awaiting_correction = True

    if st.session_state.awaiting_correction:
        correct = st.text_input("What was the correct character?")

        if st.button("Submit Correction"):
            if correct in emnist_map:
                st.success("Correction noted (not saved).")
                st.session_state.awaiting_correction = False
            else:
                st.error("Invalid label.")


# =============================================================================
# FOOTER
# =============================================================================
st.caption("Model using TensorFlow locally or TFLite in Streamlit Cloud.")
