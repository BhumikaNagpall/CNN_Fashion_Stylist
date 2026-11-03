
import streamlit as st
import mediapipe as mp
import tensorflow as tf
import cv2
import numpy as np
import os
from pathlib import Path
from stylesense_catalogue import CLASS_NAMES, HAIRSTYLES, get_references
from stylesense_backend import generate_three, generate_tryon, TryOnError
from stylesense_state import (decode_upload, image_png, upload_fingerprint,
                              sync_upload_state, analyze_records, ImageValidationError)
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

PROJECT_ROOT = Path(__file__).resolve().parent
FACE_MODEL_PATH = str(PROJECT_ROOT / "face_landmarker.task")
CLASSIFIER_PATH = str(PROJECT_ROOT / "best_mobilenet_finetuned.keras")

IMG_SIZE = (224, 224)

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
# STATEFUL PHOTO ANALYSIS AND TRY-ON
# ==========================================

def api_configuration():
    # Server-side settings only; never include values in errors or UI output.
    try:
        url = st.secrets.get("HAIRFAST_API_URL", "")
        token = st.secrets.get("HAIRFAST_API_TOKEN", "")
    except Exception:
        url, token = "", ""
    return (url or os.environ.get("HAIRFAST_API_URL", ""),
            token or os.environ.get("HAIRFAST_API_TOKEN", ""))


st.header("📸 Upload Your Photos")
st.write("Choose clear, front-facing photos of the same person, with one face per photo.")
st.caption("When you analyse photos, the first successfully detected portrait and the selected hairstyle references are sent to the configured AI service to generate your looks. The portrait is also used as the colour reference. Generation requires an available AI service.")
uploaded_files = st.file_uploader(
    "Select 1 to 3 photos of the same person", type=["jpg", "jpeg", "png"],
    accept_multiple_files=True, key="multiple_photo_upload")
items = [(file.name, file.getvalue()) for file in (uploaded_files or [])]
sync_upload_state(st.session_state, upload_fingerprint(items))

if len(items) > 3:
    st.error("Maximum 3 photos are allowed. Remove extra photos to continue.")
elif items:
    if "input_records" not in st.session_state:
        records = []
        for index, (name, data) in enumerate(items, 1):
            record = {"index": index, "name": name}
            try:
                record["image"] = decode_upload(data)
            except ImageValidationError as error:
                record["error"] = str(error)
            records.append(record)
        st.session_state.input_records = records
    records = st.session_state.input_records
    for column, record in zip(st.columns(len(records)), records):
        with column:
            if record.get("error"):
                st.warning(f"Photo {record['index']}: {record['error']}")
            else:
                st.image(record["image"], caption=f"Photo {record['index']}", use_container_width=True)

    analyse = st.button("🔍 Analyze All Photos", type="primary", use_container_width=True,
                        disabled="analysis" in st.session_state)
    if analyse:
        with st.spinner("Analysing your photos..."):
            st.session_state.analysis = analyze_records(records, detect_and_crop_face, predict_face_shape)

    analysis = st.session_state.get("analysis")
    if analysis is not None:
        st.header("Face Detection Results")
        for photo in analysis["photos"]:
            if photo.get("error"):
                st.warning(f"Photo {photo['index']}: {photo['error']}")
                continue
            left, right = st.columns(2)
            left.image(photo["annotated"], caption=f"Photo {photo['index']}: Face detection", use_container_width=True)
            right.image(photo["crop"], caption=f"Photo {photo['index']}: Face crop", use_container_width=True)
            winner = int(np.argmax(photo["probabilities"]))
            st.write(f"Photo {photo['index']}: **{CLASS_NAMES[winner]}** — {photo['probabilities'][winner] * 100:.2f}%")

        combined = analysis["combined"]
        if combined is None:
            st.error("No usable faces were detected. Please upload clearer photographs.")
            if st.button("Retry analysis"):
                st.session_state.pop("analysis", None)
                st.rerun()
        else:
            final_index = int(np.argmax(combined))
            final_shape = CLASS_NAMES[final_index]
            confidence = float(combined[final_index] * 100)
            st.header("✨ Your Combined Result")
            left, right = st.columns(2)
            left.metric("Predicted Face Shape", final_shape)
            right.metric("Average Model Confidence", f"{confidence:.2f}%")
            st.write(f"Successfully analysed {len(analysis['valid'])} out of {len(records)} photos.")
            st.subheader("Combined Probability Distribution")
            for shape, probability in zip(CLASS_NAMES, combined):
                st.write(f"**{shape}: {probability * 100:.2f}%**")
                st.progress(float(probability))
            predictions = [int(np.argmax(photo["probabilities"])) for photo in analysis["valid"]]
            if len(predictions) > 1:
                if len(set(predictions)) == 1:
                    st.success("All successfully analysed photos agree on the predicted face shape.")
                else:
                    st.warning("Your photos produced different predictions. The combined result is an estimate.")
            if confidence < 50:
                st.warning("The model is uncertain. Try clearer, front-facing photographs.")

            st.header(f"💇 Hairstyles for {final_shape} Face Shape")
            for column, hairstyle in zip(st.columns(3), HAIRSTYLES[final_shape]):
                column.markdown(f"### {hairstyle}")
            st.info("These are general recommendations. Hair texture, density and personal preference also matter.")
            st.caption("AI predictions are experimental and may vary with lighting, face angle and image quality.")

            portrait = analysis["valid"][0]["portrait"]
            st.caption(f"Photo {analysis['valid'][0]['index']} is used for try-on and natural hair colour reference.")
            api_url, api_token = api_configuration()
            references = dict(get_references(final_shape))
            missing = [name for name, path in references.items() if path is None or not path.is_file()]
            # Scope retained results to both the portrait and exact catalogue assets.
            reference_identity = []
            for name, path in references.items():
                try:
                    reference_identity.append((name, path.read_bytes() if path else b"missing"))
                except OSError:
                    reference_identity.append((name, b"unreadable"))
            generation_fingerprint = upload_fingerprint([("portrait", portrait)] + reference_identity)
            if st.session_state.get("generation_fingerprint") != generation_fingerprint:
                st.session_state.pop("generated", None)
                st.session_state.pop("generation_attempted", None)
                st.session_state.generation_fingerprint = generation_fingerprint
            st.header("✨ Your Three AI Hairstyle Try-Ons")
            if missing:
                st.warning("Automatic try-on references are not yet available for: " + ", ".join(missing))
            elif not api_url or not api_token:
                st.info("AI generation is not configured yet. Your analysis is saved; configure the AI service to generate your three looks.")
            else:
                generated = st.session_state.get("generated", {})
                incomplete = any(not generated.get(name, {}).get("image") for name in references)
                retry = False
                if st.session_state.get("generation_attempted") and incomplete:
                    retry = st.button("Retry unfinished hairstyles")
                if not st.session_state.get("generation_attempted") or retry:
                    st.session_state.generation_attempted = True
                    with st.spinner("Generating your three hairstyles. Completed looks will be kept if one fails..."):
                        try:
                            st.session_state.generated = generate_three(
                                portrait, references, api_url, api_token,
                                previous_results=generated)
                        except Exception:
                            st.session_state.generated = generated
                            st.error("Generation could not be completed. Please retry the unfinished styles.")
                for index, (column, name) in enumerate(zip(st.columns(3), references)):
                    with column:
                        st.subheader(name)
                        result = st.session_state.get("generated", {}).get(name, {})
                        if result.get("image"):
                            st.image(result["image"], use_container_width=True)
                            st.download_button("Download this look", data=result["image"],
                                               file_name=f"stylesense_{final_shape.lower()}_{index+1}.png",
                                               mime="image/png", key=f"download_{index}")
                        elif result.get("error"):
                            st.warning(result["error"])

            with st.expander("Advanced: try your own hairstyle and colour references"):
                hairstyle_reference = st.file_uploader("Upload hairstyle reference", type=["jpg", "jpeg", "png"], key="hairstyle_reference")
                color_reference = st.file_uploader("Upload hair colour reference", type=["jpg", "jpeg", "png"], key="color_reference")
                manual_items = [(file.name, file.getvalue()) for file in (hairstyle_reference, color_reference) if file is not None]
                manual_fingerprint = upload_fingerprint(manual_items)
                if st.session_state.get("manual_fingerprint") != manual_fingerprint:
                    st.session_state.pop("manual_result", None)
                    st.session_state.manual_fingerprint = manual_fingerprint
                if st.button("Generate AI Try-On", type="primary"):
                    if hairstyle_reference is None or color_reference is None:
                        st.warning("Please upload both reference photos.")
                    elif not api_url or not api_token:
                        st.info("Configure the AI service before generating a try-on.")
                    else:
                        try:
                            hair = image_png(decode_upload(hairstyle_reference.getvalue()))
                            colour = image_png(decode_upload(color_reference.getvalue()))
                            with st.spinner("Generating your custom hairstyle..."):
                                st.session_state.manual_result = generate_tryon(portrait, hair, colour, api_url, api_token)
                        except (ImageValidationError, TryOnError) as error:
                            st.warning(str(error))
                        except Exception:
                            st.error("Your custom try-on could not be completed. Please try again.")
                if st.session_state.get("manual_result"):
                    st.image(st.session_state.manual_result, caption="Your AI Hairstyle Try-On", use_container_width=True)
                    st.download_button("Download Your New Look", st.session_state.manual_result,
                                       file_name="stylesense_tryon.png", mime="image/png")
else:
    st.info("Upload your face photos to begin.")

st.divider()
st.caption("StyleSense AI | Fine-Tuned MobileNetV2 | Face Shape Classification")
