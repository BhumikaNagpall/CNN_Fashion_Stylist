"""Lightweight fixtures: no GPU, model training or live external service."""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def image_bytes():
    def make(size=(256, 256), format="PNG", color=(120, 80, 40)):
        buffer = io.BytesIO()
        Image.new("RGB", size, color).save(buffer, format=format)
        return buffer.getvalue()
    return make
