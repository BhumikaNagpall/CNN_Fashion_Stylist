
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np

from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(
    page_title="StyleSense AI",
    page_icon="💇",
    layout="wide"
)

st.title("StyleSense AI")
st.subheader("Face Shape Detection & Hairstyle Recommendations")
st.write(
    "Upload a clear, front-facing photo to estimate your face shape "
    "and explore hairstyle ideas."
)

MODEL_PATH = "face_landmarker.task"


@st.cache_resource
def load_face_model():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


def distance(points, a, b):
    return float(np.linalg.norm(points[a] - points[b]))


def estimate_face_shape(points):
    # MediaPipe Face Mesh landmark indices
    forehead_width = distance(points, 103, 332)
    cheek_width = distance(points, 234, 454)
    jaw_width = distance(points, 172, 397)
    face_length = distance(points, 10, 152)

    if min(forehead_width, cheek_width, jaw_width, face_length) <= 0:
        return "Uncertain", {}

    ratios = {
        "forehead_to_cheeks": forehead_width / cheek_width,
        "jaw_to_cheeks": jaw_width / cheek_width,
        "length_to_width": face_length / cheek_width,
        "cheeks_to_jaw": cheek_width / jaw_width
    }

    f = ratios["forehead_to_cheeks"]
    j = ratios["jaw_to_cheeks"]
    r = ratios["length_to_width"]

    # Initial heuristics only; thresholds need validation on real photos.
    if r >= 1.50 and j < 0.85:
        shape = "Oblong"
    elif r >= 1.35 and f >= 0.90 and j >= 0.85:
        shape = "Oval"
    elif r <= 1.20 and f >= 0.85 and j >= 0.85:
        shape = "Round"
    elif j <= 0.78 and f >= 0.90:
        shape = "Heart"
    elif j >= 0.88 and f >= 0.85 and r < 1.50:
        shape = "Square"
    else:
        shape = "Oval-like / Uncertain"

    return shape, ratios


HAIRSTYLES = {
    "Oval": [
        ("Soft waves", "Adds movement while keeping facial proportions balanced."),
        ("Long layers", "Creates shape and movement around the face."),
        ("Low ponytail", "A versatile option for everyday styling.")
    ],
    "Oval-like / Uncertain": [
        ("Soft waves", "A versatile option with gentle movement."),
        ("Long layers", "Adds movement and framing."),
        ("Side-parted hairstyle", "Offers an alternative to a centre part.")
    ],
    "Round": [
        ("Long layers", "Can create a more elongated visual effect."),
        ("Side-swept fringe", "Adds an angled framing effect."),
        ("High ponytail", "Creates height at the crown.")
    ],
    "Square": [
        ("Soft waves", "Can soften the appearance of angular lines."),
        ("Curtain bangs", "Frames the forehead and sides of the face."),
        ("Layered lob", "Adds movement around the jaw area.")
    ],
    "Heart": [
        ("Side-swept bangs", "Can balance the visual emphasis of the forehead."),
        ("Chin-length bob", "Adds visual width around the lower face."),
        ("Soft shoulder-length waves", "Adds movement around the jaw.")
    ],
    "Oblong": [
        ("Curtain bangs", "Can visually reduce the apparent length of the face."),
        ("Shoulder-length waves", "Adds width and movement."),
        ("Layered bob", "Creates shape around the sides of the face.")
    ],
    "Uncertain": [
        ("Soft waves", "A versatile hairstyle to explore."),
        ("Long layers", "Adds movement and shape."),
        ("Low ponytail", "A simple everyday option.")
    ]
}


def draw_landmarks(image_array, landmarks):
    output = image_array.copy()
    height, width = output.shape[:2]

    for landmark in landmarks:
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)

        cv2.circle(output, (x, y), 1, (0, 255, 0), -1)

    return output


photo = st.file_uploader(
    "Upload your photo",
    type=["jpg", "jpeg", "png"]
)

if photo:
    image = Image.open(photo).convert("RGB")
    image_array = np.array(image)

    st.image(image, caption="Your uploaded photo", width=350)

    if st.button("Analyze My Face", type="primary"):
        if image.width < 100 or image.height < 100:
            st.warning("Please upload a larger, clearer photo.")
        else:
            try:
                with st.spinner("Analyzing facial landmarks..."):
                    face_model = load_face_model()

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=image_array
                    )

                    result = face_model.detect(mp_image)

                if not result.face_landmarks:
                    st.error(
                        "No face detected. Try a front-facing photo "
                        "with good lighting and your full face visible."
                    )
                else:
                    landmarks = result.face_landmarks[0]

                    # Face-shape estimation expects standard face-mesh indices.
                    if len(landmarks) < 455:
                        st.error("The detected face mesh is incomplete.")
                    else:
                        points = np.array([
                            [p.x * image.width, p.y * image.height]
                            for p in landmarks
                        ])

                        face_shape, ratios = estimate_face_shape(points)

                        st.success("Facial landmarks detected!")

                        annotated = draw_landmarks(
                            image_array, landmarks
                        )

                        left, right = st.columns(2)

                        with left:
                            st.image(
                                annotated,
                                caption="Detected facial landmarks",
                                use_container_width=True
                            )

                        with right:
                            st.markdown("### Estimated face shape")
                            st.subheader(face_shape)

                            st.caption(
                                "This is a preliminary estimate based on "
                                "facial proportions, not a validated "
                                "face-shape classification model."
                            )

                            st.markdown("### Measurements")
                            st.write(
                                f"Forehead-to-cheek ratio: "
                                f"{ratios['forehead_to_cheeks']:.2f}"
                            )
                            st.write(
                                f"Jaw-to-cheek ratio: "
                                f"{ratios['jaw_to_cheeks']:.2f}"
                            )
                            st.write(
                                f"Face-length-to-cheek-width ratio: "
                                f"{ratios['length_to_width']:.2f}"
                            )

                        st.markdown("---")
                        st.markdown("## Hairstyle suggestions for you")

                        recommendations = HAIRSTYLES.get(
                            face_shape, HAIRSTYLES["Uncertain"]
                        )

                        for i, (name, reason) in enumerate(
                            recommendations, start=1
                        ):
                            with st.container(border=True):
                                st.markdown(f"**{i}. {name}**")
                                st.write(reason)

                        st.info(
                            "These are styling ideas, not strict rules. "
                            "Personal preference, hair texture, hair "
                            "density, maintenance, and lifestyle matter too."
                        )

            except Exception as error:
                st.error(f"Face analysis failed: {error}")
                st.caption(
                    "Check that face_landmarker.task is beside app.py "
                    "and that the required libraries are installed."
                )
