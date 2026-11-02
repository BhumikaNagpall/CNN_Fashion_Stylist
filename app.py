
import streamlit as st
import mediapipe as mp
import tensorflow as tf
import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="StyleSense AI",
    page_icon="💇",
    layout="wide"
)

st.title("💇 StyleSense AI")
st.subheader("Your AI-powered Face Shape & Hairstyle Assistant")

st.write(
    "Upload up to 3 photographs of the same person. "
    "Our AI will analyze them and calculate a combined prediction."
)


# ==========================================
# CONFIGURATION
# ==========================================

FACE_MODEL_PATH = "face_landmarker.task"
CLASSIFIER_PATH = "best_mobilenet_finetuned.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Heart",
    "Oblong",
    "Oval",
    "Round",
    "Square"
]


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_face_model():

    base_options = python.BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    return vision.FaceLandmarker.create_from_options(options)


@st.cache_resource
def load_classifier():

    return tf.keras.models.load_model(CLASSIFIER_PATH)


# ==========================================
# FACE DETECTION
# ==========================================

def detect_and_crop_face(image):

    rgb = np.array(image.convert("RGB"))

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    detector = load_face_model()
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    landmarks = result.face_landmarks[0]

    height, width, _ = rgb.shape

    x_values = [p.x * width for p in landmarks]
    y_values = [p.y * height for p in landmarks]

    x_min = int(min(x_values))
    x_max = int(max(x_values))
    y_min = int(min(y_values))
    y_max = int(max(y_values))

    padding_x = int((x_max - x_min) * 0.20)
    padding_y = int((y_max - y_min) * 0.20)

    x_min = max(0, x_min - padding_x)
    x_max = min(width, x_max + padding_x)

    y_min = max(0, y_min - padding_y)
    y_max = min(height, y_max + padding_y)

    if x_max <= x_min or y_max <= y_min:
        return None, None

    face_crop = image.crop(
        (x_min, y_min, x_max, y_max)
    )

    annotated = rgb.copy()

    for point in landmarks:

        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(
            annotated,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    return face_crop, Image.fromarray(annotated)


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_face(face_crop):

    face = ImageOps.pad(
        face_crop.convert("RGB"),
        IMG_SIZE,
        method=Image.Resampling.LANCZOS,
        color=(127, 127, 127),
        centering=(0.5, 0.5)
    )

    image_array = np.array(
        face,
        dtype=np.float32
    )

    return np.expand_dims(image_array, axis=0)


# ==========================================
# PREDICTION
# ==========================================

def predict_face_shape(face_crop):

    model = load_classifier()

    processed = preprocess_face(face_crop)

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    return probabilities


# ==========================================
# HAIRSTYLE RECOMMENDATIONS
# ==========================================

HAIRSTYLES = {

    "Heart": [
        "Side-swept bangs",
        "Shoulder-length layers",
        "Chin-length bob"
    ],

    "Oblong": [
        "Curtain bangs",
        "Layered shoulder-length hair",
        "Wavy bob"
    ],

    "Oval": [
        "Long layered hair",
        "Soft waves",
        "Curtain bangs"
    ],

    "Round": [
        "Long layers",
        "Side-parted hairstyle",
        "Long curtain bangs"
    ],

    "Square": [
        "Soft layered waves",
        "Side-swept bangs",
        "Textured long bob"
    ]
}
# Reference images for AI hairstyle try-on
HAIRSTYLE_REFERENCES = {
    "Round": {
        "Long layers": "hairstyle_references/round_long_layers.jpg",
        "Side-parted hairstyle": "hairstyle_references/round_side_parted.jpg",
        "Long curtain bangs": "hairstyle_references/round_curtain_bangs.jpg"
    }
}


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About StyleSense AI")

    st.write(
        "MediaPipe detects facial landmarks, while "
        "fine-tuned MobileNetV2 predicts face shape."
    )

    st.write("Supported categories:")

    for shape in CLASS_NAMES:
        st.write(f"• {shape}")


# ==========================================
# MULTIPLE PHOTO UPLOAD
# ==========================================

st.header("📸 Upload Your Photos")

uploaded_files = st.file_uploader(
    "Select 1 to 3 photos of the same person",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="multiple_photo_upload",
    help="Use Ctrl + Click to select multiple photos on Windows."
)

if uploaded_files:

    if len(uploaded_files) > 3:

        st.error("Maximum 3 photos are allowed.")

    else:

        st.success(
            f"{len(uploaded_files)} photo(s) uploaded successfully!"
        )

        st.subheader("Uploaded Photos")

        columns = st.columns(len(uploaded_files))

        for index, uploaded_file in enumerate(uploaded_files):

            uploaded_file.seek(0)

            image = Image.open(uploaded_file).convert("RGB")

            with columns[index]:

                st.image(
                    image,
                    caption=f"Photo {index + 1}",
                    use_container_width=True
                )

        analyze = st.button(
            "🔍 Analyze All Photos",
            type="primary",
            use_container_width=True
        )

        if analyze:

            all_predictions = []
            individual_predictions = []

            st.header("Face Detection Results")

            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):

                uploaded_file.seek(0)

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                with st.spinner(
                    f"Analyzing photo {index + 1}..."
                ):

                    face_crop, annotated_image = detect_and_crop_face(
                        image
                    )

                    if face_crop is None:

                        st.warning(
                            f"No face detected in Photo {index + 1}. "
                            "This photo was skipped."
                        )

                    else:

                        probabilities = predict_face_shape(
                            face_crop
                        )

                        all_predictions.append(probabilities)

                        predicted_index = np.argmax(probabilities)

                        individual_predictions.append(
                            CLASS_NAMES[predicted_index]
                        )

                        col1, col2 = st.columns(2)

                        with col1:

                            st.image(
                                annotated_image,
                                caption=f"Photo {index + 1}: Face detection",
                                use_container_width=True
                            )

                        with col2:

                            st.image(
                                face_crop,
                                caption=f"Photo {index + 1}: Face crop",
                                use_container_width=True
                            )

                        st.write(
                            f"**Individual prediction:** "
                            f"{CLASS_NAMES[predicted_index]}"
                        )

                        st.write(
                            f"**Individual confidence:** "
                            f"{probabilities[predicted_index] * 100:.2f}%"
                        )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            # ======================================
            # COMBINED PREDICTION
            # ======================================

            if len(all_predictions) == 0:

                st.error(
                    "No usable faces were detected. "
                    "Please upload clearer photographs."
                )

            else:

                average_probabilities = np.mean(
                    np.array(all_predictions),
                    axis=0
                )

                final_index = np.argmax(
                    average_probabilities
                )

                final_shape = CLASS_NAMES[final_index]

                confidence = (
                    average_probabilities[final_index] * 100
                )

                st.divider()

                st.header("✨ Your Combined Result")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Predicted Face Shape",
                        final_shape
                    )

                with col2:

                    st.metric(
                        "Average Model Confidence",
                        f"{confidence:.2f}%"
                    )

                st.write(
                    f"Successfully analyzed {len(all_predictions)} "
                    f"out of {len(uploaded_files)} photos."
                )

                # Individual predictions
                st.subheader("Individual Photo Predictions")

                for index, prediction in enumerate(
                    individual_predictions
                ):

                    st.write(
                        f"Photo {index + 1}: **{prediction}**"
                    )

                # Combined probabilities
                st.subheader("Combined Probability Distribution")

                for index, shape in enumerate(CLASS_NAMES):

                    probability = float(
                        average_probabilities[index]
                    )

                    st.write(
                        f"**{shape}: {probability * 100:.2f}%**"
                    )

                    st.progress(probability)

                # Agreement check
                if len(set(individual_predictions)) == 1:

                    st.success(
                        "All photos agree on the predicted face shape."
                    )

                else:

                    st.warning(
                        "Your photos produced different predictions. "
                        "The combined result is an estimate."
                    )

                if confidence < 50:

                    st.warning(
                        "The model is uncertain about this result. "
                        "Try clearer, front-facing photographs."
                    )

                # Hairstyle suggestions
                st.divider()

                st.header(
                    f"💇 Hairstyles for {final_shape} Face Shape"
                )

                hairstyle_columns = st.columns(3)

                for index, hairstyle in enumerate(
                    HAIRSTYLES[final_shape]
                ):

                    with hairstyle_columns[index]:

                        st.markdown(
                            f"### {hairstyle}"
                        )

                st.info(
                    "These are general recommendations. "
                    "Hair texture, density and personal preference "
                    "also matter when choosing a hairstyle."
                )

                st.caption(
                    "AI predictions are experimental and may vary "
                    "with lighting, face angle and image quality."
                )


# ---------------- AI HAIRSTYLE TRY-ON ----------------

st.divider()
st.header("✨ AI Virtual Hairstyle Try-On")

st.write(
    "Upload reference photos to see how a different hairstyle "
    "and hair colour could look on you."
)

if not uploaded_files:
    st.info("Upload your face photo in the section above first.")

else:
    col1, col2 = st.columns(2)

    with col1:
        hairstyle_reference = st.file_uploader(
            "Upload hairstyle reference",
            type=["jpg", "jpeg", "png"],
            key="hairstyle_reference"
        )

    with col2:
        color_reference = st.file_uploader(
            "Upload hair colour reference",
            type=["jpg", "jpeg", "png"],
            key="color_reference"
        )

    st.caption(
        "Your first uploaded photo will be used as your face photo. "
        "The hairstyle reference provides the hairstyle, while the "
        "colour reference provides the hair colour."
    )

    if st.button("Generate AI Try-On", type="primary"):
        if hairstyle_reference is None or color_reference is None:
            st.warning("Please upload both reference photos.")
        else:
            try:
                api_url = st.secrets["HAIRFAST_API_URL"].rstrip("/")
                api_token = st.secrets["HAIRFAST_API_TOKEN"]

                face_file = uploaded_files[0]

                files = {
                    "face": (
                        face_file.name,
                        face_file.getvalue(),
                        face_file.type
                    ),
                    "hairstyle": (
                        hairstyle_reference.name,
                        hairstyle_reference.getvalue(),
                        hairstyle_reference.type
                    ),
                    "color": (
                        color_reference.name,
                        color_reference.getvalue(),
                        color_reference.type
                    )
                }

                with st.spinner(
                    "AI is generating your new hairstyle. Please wait..."
                ):
                    response = requests.post(
                        f"{api_url}/tryon",
                        headers={
                            "Authorization": f"Bearer {api_token}"
                        },
                        files=files,
                        timeout=600
                    )

                if response.status_code == 200 and response.headers.get(
                    "content-type", ""
                ).startswith("image/"):

                    st.success("Your AI hairstyle is ready!")

                    st.image(
                        response.content,
                        caption="Your AI Hairstyle Try-On",
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Your New Look",
                        data=response.content,
                        file_name="stylesense_tryon.png",
                        mime="image/png"
                    )

                else:
                    st.error(
                        f"Try-on failed ({response.status_code}): "
                        f"{response.text[:1000]}"
                    )

            except requests.exceptions.Timeout:
                st.error("The generation took too long. Please try again.")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the AI server: {e}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------------- END AI HAIRSTYLE TRY-ON ----------------

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "StyleSense AI | Fine-Tuned MobileNetV2 | Face Shape Classification"
)
