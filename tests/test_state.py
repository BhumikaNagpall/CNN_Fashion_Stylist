from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from stylesense_state import (ImageValidationError, aggregate_predictions, analyze_records,
    decode_upload, image_png, sync_upload_state, upload_fingerprint)


def test_probability_averaging_not_majority_vote():
    vectors = [[.51, .49, 0, 0, 0], [.51, .49, 0, 0, 0], [0, 1, 0, 0, 0]]
    result = aggregate_predictions(vectors)
    np.testing.assert_allclose(result, np.mean(vectors, axis=0), atol=1e-7)
    assert result.argmax() == 1  # Two individual winners are class 0; mean winner is 1.


@pytest.mark.parametrize('vectors', [[], [[1, 0]], [[float('nan'), 0, 0, 0, 1]], [[-1, 1, 1, 0, 0]], [[1, 1, 0, 0, 0]]])
def test_invalid_model_outputs_rejected(vectors):
    with pytest.raises(ValueError):
        aggregate_predictions(vectors)


def test_failed_detection_keeps_original_number_and_valid_portrait(image_bytes):
    image = decode_upload(image_bytes())
    detector = Mock(side_effect=[(None, None), (image, image)])
    predictor = Mock(return_value=[0, 0, 0, 1, 0])
    result = analyze_records([{'index': 1, 'name': 'badface.png', 'image': image}, {'index': 2, 'name': 'good.png', 'image': image}], detector, predictor)
    assert result['photos'][0]['error']
    assert len(result['valid']) == 1
    assert result['valid'][0]['index'] == 2
    assert result['valid'][0]['name'] == 'good.png'
    assert Image.open(BytesIO(result['valid'][0]['portrait'])).format == 'PNG'
    np.testing.assert_array_equal(result['combined'], [0, 0, 0, 1, 0])
    assert predictor.call_count == 1


def test_corrupt_record_and_detector_failure_are_skipped_without_leaking_errors(image_bytes):
    image = decode_upload(image_bytes())
    detector = Mock(side_effect=RuntimeError('secret path'))
    predictor = Mock()
    result = analyze_records([{'index': 1, 'name': 'bad.png', 'error': 'Invalid image'}, {'index': 2, 'name': 'photo.png', 'image': image}], detector, predictor)
    assert result['valid'] == [] and result['combined'] is None
    assert 'secret path' not in str(result)
    assert detector.call_count == 1
    predictor.assert_not_called()


@pytest.mark.parametrize('payload', [b'', b'fake image'])
def test_corrupt_uploads_fail_safely(payload):
    with pytest.raises(ImageValidationError):
        decode_upload(payload)


@pytest.mark.parametrize('size,format', [((99, 256), 'PNG'), ((256, 256), 'GIF'), ((4100, 4100), 'PNG')])
def test_upload_format_and_dimension_limits(image_bytes, size, format):
    with pytest.raises(ImageValidationError):
        decode_upload(image_bytes(size=size, format=format))


def test_exif_orientation_corrected():
    image = Image.new('RGB', (100, 200))
    exif = image.getexif()
    exif[274] = 6
    exif[315] = 'private artist metadata'
    output = BytesIO()
    image.save(output, format='JPEG', exif=exif)
    decoded = decode_upload(output.getvalue())
    assert decoded.size == (200, 100)
    portrait = image_png(decoded)
    with Image.open(BytesIO(portrait)) as clean:
        assert not clean.getexif()
    assert b'private artist metadata' not in portrait


def test_same_upload_preserves_state_changed_upload_invalidates_all_derived_results():
    key = upload_fingerprint([('photo.png', b'one')])
    state = {'upload_fingerprint': key, 'analysis': object(), 'generated': {'A': b'image'},
             'manual_result': b'image', 'generation_attempted': True, 'input_records': [],
             'manual_fingerprint': 'old', 'unrelated': 'keep'}
    before = state.copy()
    assert not sync_upload_state(state, key)
    assert state == before
    assert sync_upload_state(state, upload_fingerprint([('photo.png', b'two')]))
    assert state == {'upload_fingerprint': upload_fingerprint([('photo.png', b'two')]), 'unrelated': 'keep'}


def test_order_and_names_are_in_upload_identity():
    a, b = ('first.png', b'a'), ('second.png', b'b')
    assert upload_fingerprint([a, b]) != upload_fingerprint([b, a])
    assert upload_fingerprint([a]) != upload_fingerprint([('renamed.png', b'a')])
