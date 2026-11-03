"""Small, synchronous client for the existing HairFastGAN /tryon contract.

This module does not run HairFastGAN. Live service compatibility must be checked
separately; contract tests cannot establish generation quality or availability.
"""

from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import requests


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_SIDE = 10_000
MIN_IMAGE_SIDE = 64


class TryOnError(Exception):
    """An intentionally safe message that may be shown to the user."""


def normalize_image(data: bytes) -> bytes:
    """Decode bounded JPEG/PNG bytes and strip metadata by re-encoding to PNG."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_IMAGE_BYTES:
        raise TryOnError("Image is empty or exceeds the 20 MiB limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise TryOnError("Use a JPEG or PNG image.")
                width, height = image.size
                if (min(width, height) < MIN_IMAGE_SIDE
                        or max(width, height) > MAX_IMAGE_SIDE
                        or width * height > MAX_IMAGE_PIXELS):
                    raise TryOnError("Image dimensions are outside the supported limits.")
                if getattr(image, "n_frames", 1) != 1:
                    raise TryOnError("Use a single-frame image.")
                image.load()
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                # A fresh image avoids forwarding EXIF, comments and ICC data.
                clean = Image.new("RGB", normalized.size)
                clean.paste(normalized)
                output = BytesIO()
                clean.save(output, format="PNG")
                result = output.getvalue()
                if len(result) > MAX_IMAGE_BYTES:
                    raise TryOnError("Decoded image exceeds the supported transfer size.")
                return result
    except TryOnError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise TryOnError("Image could not be decoded safely. Use a valid JPEG or PNG.") from None


def _endpoint(api_url: str, api_token: str) -> str:
    if not isinstance(api_url, str) or not isinstance(api_token, str):
        raise TryOnError("The hairstyle service is not configured.")
    try:
        parsed = urlsplit(api_url.strip())
        if (parsed.scheme != "https" or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError
        # Accessing port checks malformed port strings before requests does.
        _ = parsed.port
    except ValueError:
        raise TryOnError("Configure a valid HTTPS hairstyle service URL.") from None
    if not api_token.strip() or any(ord(c) < 32 or ord(c) == 127 for c in api_token):
        raise TryOnError("The hairstyle service token is missing or invalid.")
    return api_url.strip().rstrip("/") + "/tryon"


def generate_tryon(face_bytes: bytes, hairstyle_bytes: bytes, color_bytes: bytes,
                   api_url: str, api_token: str) -> bytes:
    """Send one validated request and return a decoded, normalized PNG.

    Portrait face detection belongs to the caller. HTTPS is mandatory; redirects
    are rejected rather than risking forwarding a portrait or authorization.
    """
    endpoint = _endpoint(api_url, api_token)
    images = {
        "face": normalize_image(face_bytes),
        "hairstyle": normalize_image(hairstyle_bytes),
        "color": normalize_image(color_bytes),
    }
    files = {name: (f"{name}.png", data, "image/png") for name, data in images.items()}
    response = None
    try:
        response = requests.post(
            endpoint, headers={"Authorization": f"Bearer {api_token}"},
            files=files, timeout=(15, 600), allow_redirects=False, stream=True,
        )
        if response.status_code in (401, 403):
            raise TryOnError("The hairstyle service rejected authentication. Check server configuration.")
        if response.status_code == 429:
            raise TryOnError("The hairstyle service is busy. Please retry later.")
        if response.status_code != 200:
            raise TryOnError("The hairstyle service could not complete this request. Please retry later.")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in {"image/png", "image/jpeg"}:
            raise TryOnError("The hairstyle service returned an unsupported image response.")
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                raise TryOnError("The hairstyle service returned an invalid response.") from None
            if length < 0 or length > MAX_IMAGE_BYTES:
                raise TryOnError("The generated image exceeds the supported transfer size.")
        payload = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                if len(payload) + len(chunk) > MAX_IMAGE_BYTES:
                    raise TryOnError("The generated image exceeds the supported transfer size.")
                payload.extend(chunk)
        return normalize_image(bytes(payload))
    except TryOnError:
        raise
    except requests.exceptions.Timeout:
        raise TryOnError("Hairstyle generation timed out. Please retry this style.") from None
    except requests.exceptions.RequestException:
        raise TryOnError("Could not connect to the hairstyle service. Please retry later.") from None
    except (ValueError, TypeError, OSError):
        raise TryOnError("The hairstyle service returned an invalid response.") from None
    finally:
        if response is not None:
            response.close()


def generate_three(face_bytes: bytes, references: dict, api_url: str, api_token: str,
                   previous_results: dict | None = None) -> dict:
    """Generate three styles independently, retaining already successful results.

    references maps exactly three style names to image bytes, local Paths/strings,
    or None for unavailable assets. The caller MUST scope previous_results to the
    same portrait and catalogue; clear them whenever inputs change. The original
    portrait is used as colour reference, pending live backend compatibility QA.
    """
    if len(references) != 3:
        raise TryOnError("Exactly three hairstyle references are required.")
    previous_results = previous_results or {}
    results = {}
    for name, reference in references.items():
        previous = previous_results.get(name, {})
        if previous.get("image") and not previous.get("error"):
            results[name] = dict(previous)
            continue
        try:
            if reference is None:
                raise TryOnError("This hairstyle reference is not available yet.")
            if isinstance(reference, (str, Path)):
                with Path(reference).open("rb") as file:
                    data = file.read(MAX_IMAGE_BYTES + 1)
            else:
                data = reference
            result = generate_tryon(face_bytes, data, face_bytes, api_url, api_token)
            results[name] = {"image": result, "error": None}
        except TryOnError as error:
            results[name] = {"image": None, "error": str(error)}
        except (OSError, TypeError, ValueError):
            results[name] = {"image": None, "error": "This hairstyle reference could not be read."}
    return results
