import pytest

import stylesense_backend as backend


@pytest.mark.parametrize("oversized", [0, 1, 2])
def test_reencoded_part_over_notebook_limit_never_posts(monkeypatch, oversized):
    # The server measures transferred bytes, even when the uploaded JPEG was small.
    normalized = [b"png"] * 3
    normalized[oversized] = b"x" * (10 * 1024 * 1024 + 1)
    converted = iter(normalized)
    monkeypatch.setattr(backend, "normalize_image", lambda _: next(converted))
    monkeypatch.setattr(backend.requests, "post", lambda *a, **kw: pytest.fail("network call"))
    with pytest.raises(backend.TryOnError, match="10 MiB"):
        backend.generate_tryon(b"a", b"b", b"c", "https://example.invalid", "test")
