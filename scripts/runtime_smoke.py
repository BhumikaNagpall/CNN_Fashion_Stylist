"""Exercise the shipped CPU detector and classifier with a bundled portrait."""

from streamlit.testing.v1 import AppTest


def main():
    # Import the application inside Streamlit's runner so its existing cached
    # model loaders and UI have a real session. No uploads or buttons are set.
    smoke = AppTest.from_string(
        """
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import app

portrait = Path(app.PROJECT_ROOT) / "hairstyle_references" / "round_long_layers.jpg"
with Image.open(portrait) as image:
    crop, annotated = app.detect_and_crop_face(image.convert("RGB"))
if crop is None or annotated is None:
    raise RuntimeError("No face detected in the bundled Round smoke-test portrait.")
scores = np.asarray(app.predict_face_shape(crop))
if scores.shape != (5,) or not np.isfinite(scores).all():
    raise RuntimeError("Expected five finite classifier scores.")
if (scores < 0).any() or (scores > 1).any() or not np.isclose(scores.sum(), 1, atol=1e-4):
    raise RuntimeError("Classifier output is not a valid probability distribution.")
st.success("runtime-smoke-passed")
"""
    ).run(timeout=180)
    if smoke.exception:
        raise RuntimeError(smoke.exception[0].message)
    if not any(item.value == "runtime-smoke-passed" for item in smoke.success):
        raise RuntimeError("Smoke test did not reach its final assertion.")
    print("MediaPipe detected a face and the active classifier returned five valid probabilities.")


if __name__ == "__main__":
    main()
