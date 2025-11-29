import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load model with cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("handwritten_characters.keras")

model = load_model()

# EMNIST label map
emnist_map = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]

# Temporary dataset
if "train_images" not in st.session_state:
    st.session_state.train_images = []
    st.session_state.train_labels = []

# Store the most recent drawn image for correction
if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

st.title("✏️ Handwritten Character Recognition")
st.write("Draw a character and let the model guess. Correct it if it's wrong!")

# -----------------------
# Canvas for drawing
# -----------------------
st.subheader("Draw a character")

canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess(img):
    img = img.convert("L")
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# -----------------------
# Predict button
# -----------------------
if st.button("Predict"):
    if canvas.image_data is not None:
        pil_img = Image.fromarray(canvas.image_data.astype("uint8"))
        processed = preprocess(pil_img)
        st.session_state.last_image = processed  # store for correction

        pred = model.predict(processed)
        class_idx = np.argmax(pred)
        predicted_char = emnist_map[class_idx]

        st.session_state.last_prediction = predicted_char

        st.success(f"Model prediction: **{predicted_char}**")
    else:
        st.error("Draw something first.")

# -----------------------
# Correction UI
# -----------------------
# -----------------------
# Correction UI
# -----------------------
if st.session_state.last_prediction is not None:
    st.write("### Was the prediction correct?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, correct"):
            st.success("Great! No correction needed.")

    with col2:
        if st.button("❌ No, incorrect"):
            st.warning("Please enter the correct label below.")

            correct_label = st.text_input("Correct Label:", key="correct_label_box")

            if st.button("Submit Correction"):
                if correct_label in emnist_map:

                    # Convert Numpy array → Pure list so streamlit can store it
                    img_list = st.session_state.last_image.tolist()

                    st.session_state.train_images.append(img_list)
                    st.session_state.train_labels.append(emnist_map.index(correct_label))

                    st.success("Correction saved! Added to training dataset.")
                else:
                    st.error("Label must be a valid EMNIST class.")

# -----------------------
# Training Section
# -----------------------
st.subheader("Train model on collected corrections")

if st.button("Train Model"):
    if len(st.session_state.train_images) == 0:
        st.warning("No corrected examples collected.")
    else:
        X = np.vstack(st.session_state.train_images)
        y = np.array(st.session_state.train_labels)

        model.fit(X, y, epochs=3)
        model.save("handwritten_characters.keras")

        st.success("Model retrained with corrected examples!")

# Show dataset size
st.write(f"📦 Stored corrected samples: **{len(st.session_state.train_images)}**")
