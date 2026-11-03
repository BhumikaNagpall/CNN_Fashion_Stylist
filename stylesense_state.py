"""Pure validation and session helpers; no model loading or network calls."""
from hashlib import sha256
from io import BytesIO
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
MIN_SIDE = 100

class ImageValidationError(ValueError):
    """Safe, user-facing image validation error."""


def decode_upload(data):
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("Please use an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                if probe.format not in {"JPEG", "PNG"}:
                    raise ImageValidationError("Only JPEG and PNG images are supported.")
                width, height = probe.size
                if min(width, height) < MIN_SIDE or width * height > MAX_PIXELS:
                    raise ImageValidationError("Use an image at least 100 pixels on each side and no larger than 16 megapixels.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise ImageValidationError("Please use a still image, not an animation.")
                probe.verify()
            with Image.open(BytesIO(data)) as image:
                return ImageOps.exif_transpose(image).convert("RGB").copy()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError("This image could not be read. Please upload a valid JPEG or PNG.") from None


def image_png(image):
    output = BytesIO()
    rgb = image.convert("RGB")
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    clean.save(output, format="PNG")
    return output.getvalue()


def upload_fingerprint(items):
    """Order and names are part of identity, preserving original photo numbering."""
    digest = sha256()
    for name, data in items:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(data).digest())
    return digest.hexdigest()


def sync_upload_state(state, fingerprint):
    """Reset derived state only when the upload selection changes."""
    if state.get("upload_fingerprint") == fingerprint:
        return False
    for key in ("input_records", "analysis", "generated", "generation_attempted",
                "manual_result", "manual_fingerprint", "generation_fingerprint"):
        state.pop(key, None)
    state["upload_fingerprint"] = fingerprint
    return True


def aggregate_predictions(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 5:
        raise ValueError("Expected one or more five-class probability vectors.")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Invalid prediction probabilities.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Prediction probabilities must sum to one.")
    return np.mean(values, axis=0)


def analyze_records(records, detect, predict):
    """Keep upload indices, skip failures, and retain only detected portraits."""
    outcomes = []
    for record in records:
        item = {"index": record["index"], "name": record["name"]}
        if record.get("error"):
            item["error"] = record["error"]
        else:
            try:
                image = record["image"]
                crop, annotated = detect(image)
                if crop is None:
                    item["error"] = "No face detected. This photo was skipped."
                else:
                    probabilities = np.asarray(predict(crop), dtype=np.float32)
                    aggregate_predictions([probabilities])
                    item.update(portrait=image_png(image), crop=crop,
                                annotated=annotated, probabilities=probabilities)
            except Exception:
                item["error"] = "This photo could not be analysed. Check the model setup or try another photo."
        outcomes.append(item)
    valid = [item for item in outcomes if "probabilities" in item]
    return {"photos": outcomes, "valid": valid,
            "combined": aggregate_predictions([item["probabilities"] for item in valid]) if valid else None}
