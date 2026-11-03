"""Real Streamlit reruns, with upload controls/inference/network stubbed.

These tests exercise app state and rendering; they do not establish ML quality or
HairFastGAN availability. AppTest does not provide a native uploader setter.
"""
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

import stylesense_backend
import stylesense_catalogue
from stylesense_backend import generate_three as real_generate_three
import stylesense_state


class Upload(BytesIO):
    def __init__(self, data, name='photo.png'):
        super().__init__(data)
        self.name = name
        self.size = len(data)
        self.type = 'image/png'


class UnreadableUpload:
    name = 'rejected.png'
    type = 'image/png'

    def __init__(self, size):
        self.size = size

    def getvalue(self):
        raise AssertionError('Rejected upload contents should never be materialised')


@pytest.fixture
def app_environment(monkeypatch, image_bytes):
    # Only import-time placeholders; models are never called in these UI tests.
    for name in ['tensorflow', 'cv2', 'mediapipe', 'mediapipe.tasks', 'mediapipe.tasks.python', 'mediapipe.tasks.python.vision']:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['mediapipe.tasks'].python = sys.modules['mediapipe.tasks.python']
    sys.modules['mediapipe.tasks.python'].vision = sys.modules['mediapipe.tasks.python.vision']
    data = image_bytes()
    uploads = [Upload(data)]
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: uploads if kwargs.get('key') == 'multiple_photo_upload' else None)
    monkeypatch.setenv('HAIRFAST_API_URL', 'https://example.test')
    monkeypatch.setenv('HAIRFAST_API_TOKEN', 'test-token')
    image = Image.open(BytesIO(data)).convert('RGB')
    probabilities = np.array([0, 0, 0, 1, 0], dtype=np.float32)
    photo = {'index': 1, 'name': 'photo.png', 'portrait': data, 'crop': image, 'annotated': image, 'probabilities': probabilities}
    analysis = Mock(return_value={'photos': [photo], 'valid': [photo], 'combined': probabilities})
    generation = Mock(return_value={name: {'image': data, 'error': None} for name in ['Long layers', 'Side-parted hairstyle', 'Long curtain bangs']})
    monkeypatch.setattr(stylesense_state, 'analyze_records', analysis)
    monkeypatch.setattr(stylesense_backend, 'generate_three', generation)
    return uploads, analysis, generation


def test_reruns_keep_results_without_repeated_inference_or_generation(app_environment):
    uploads, analysis, generation = app_environment
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert analysis.call_count == 1 and generation.call_count == 1
    assert len(app.get('download_button')) == 3
    app.run()
    assert not app.exception
    assert analysis.call_count == 1 and generation.call_count == 1
    assert app.button[0].disabled
    assert len(app.get('download_button')) == 3
    assert app.metric[0].value == 'Round'


def test_changed_upload_clears_old_results_until_reanalysed(app_environment, image_bytes):
    uploads, analysis, generation = app_environment
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    app.button[0].click().run()
    uploads[:] = [Upload(image_bytes(color=(50, 60, 70)), 'new.png')]
    app.run()
    assert not app.exception
    assert not app.button[0].disabled
    assert len(app.metric) == 0
    assert len(app.get('download_button')) == 0
    assert analysis.call_count == 1 and generation.call_count == 1
    app.button[0].click().run()
    assert not app.exception
    assert analysis.call_count == 2 and generation.call_count == 2


def test_over_limit_uploads_do_not_offer_analysis_or_generation(app_environment):
    uploads, analysis, generation = app_environment
    uploads[:] = [UnreadableUpload(256)] * 4
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    assert not app.exception
    assert any('Maximum 3' in error.value for error in app.error)
    assert not app.button
    analysis.assert_not_called()
    generation.assert_not_called()


def test_partial_generation_retries_only_on_explicit_action(app_environment, image_bytes):
    uploads, analysis, generation = app_environment
    data = image_bytes()
    partial = {'Long layers': {'image': data, 'error': None},
               'Side-parted hairstyle': {'image': None, 'error': 'Timed out'},
               'Long curtain bangs': {'image': data, 'error': None}}
    complete = {name: {'image': data, 'error': None} for name in partial}
    generation.side_effect = [partial, complete]
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    app.button[0].click().run()
    assert len(app.get('download_button')) == 2
    app.run()
    assert generation.call_count == 1
    next(button for button in app.button if button.label == 'Retry unfinished hairstyles').click().run()
    assert not app.exception
    assert generation.call_count == 2
    assert generation.call_args.kwargs['previous_results'] == partial
    assert analysis.call_count == 1
    assert len(app.get('download_button')) == 3


def test_no_valid_faces_never_sent_to_generation(app_environment):
    uploads, analysis, generation = app_environment
    analysis.return_value = {'photos': [{'index': 1, 'name': 'photo.png', 'error': 'No face detected'}], 'valid': [], 'combined': None}
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    app.button[0].click().run()
    assert not app.exception
    assert any('No usable faces' in error.value for error in app.error)
    generation.assert_not_called()
    assert not app.get('download_button')


def test_missing_catalogue_preserves_analysis_without_fake_generation(app_environment, monkeypatch, tmp_path):
    monkeypatch.setattr(stylesense_catalogue, 'PROJECT_ROOT', tmp_path)
    uploads, analysis, generation = app_environment
    analysis.return_value['combined'] = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == 'Heart'
    assert any('not yet available' in warning.value for warning in app.warning)
    generation.assert_not_called()


def test_manual_flow_survives_reruns_and_reference_changes_clear_old_image(app_environment, monkeypatch, image_bytes):
    uploads, analysis, generation = app_environment
    references = {'hairstyle_reference': Upload(image_bytes(), 'hair.png'),
                  'color_reference': Upload(image_bytes(color=(30, 30, 30)), 'colour.png')}
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: uploads if kwargs.get('key') == 'multiple_photo_upload' else references.get(kwargs.get('key')))
    manual = Mock(return_value=image_bytes())
    monkeypatch.setattr(stylesense_backend, 'generate_tryon', manual)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    app.button[0].click().run()
    next(button for button in app.button if button.label == 'Generate AI Try-On').click().run()
    assert not app.exception
    assert manual.call_count == 1
    assert len(app.get('download_button')) == 4
    app.run()
    assert manual.call_count == 1 and generation.call_count == 1 and analysis.call_count == 1
    assert len(app.get('download_button')) == 4
    references['hairstyle_reference'] = Upload(image_bytes(color=(100, 30, 30)), 'changed.png')
    app.run()
    assert not app.exception
    assert len(app.get('download_button')) == 3
    assert manual.call_count == 1


@pytest.mark.parametrize('shape', ['Heart', 'Oblong', 'Oval', 'Round', 'Square'])
def test_three_downloads_for_every_class_with_fixture_references(
        app_environment, shape, tmp_path, monkeypatch, image_bytes):
    """Exercise UI, orchestration and HTTP client with synthetic reference fixtures."""
    _, analysis, _ = app_environment
    scores = np.zeros(5, dtype=np.float32)
    scores[stylesense_catalogue.CLASS_NAMES.index(shape)] = 1
    analysis.return_value['combined'] = scores
    analysis.return_value['valid'][0]['probabilities'] = scores
    references = []
    for index, name in enumerate(stylesense_catalogue.HAIRSTYLES[shape]):
        path = tmp_path / f'reference{index}.png'
        path.write_bytes(image_bytes(color=(50 * index, 100, 100)))
        references.append((name, path))

    def fixture_references(predicted_shape):
        assert predicted_shape == shape
        return references

    calls = []
    def mock_post(url, **kwargs):
        files = kwargs['files']
        assert files['face'][1] == files['color'][1]
        calls.append(files['hairstyle'][1])
        response = Mock(status_code=200, headers={'content-type': 'image/png'})
        response.iter_content.return_value = [image_bytes()]
        return response

    monkeypatch.setattr(stylesense_catalogue, 'get_references', fixture_references)
    monkeypatch.setattr(stylesense_backend, 'generate_three', real_generate_three)
    monkeypatch.setattr(stylesense_backend.requests, 'post', mock_post)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == shape
    assert len(calls) == len(set(calls)) == 3
    assert len(app.get('download_button')) == 3
    app.run()
    assert not app.exception
    assert len(calls) == 3
    assert len(app.get('download_button')) == 3


def test_oversized_portrait_safely_rejected_without_read(app_environment):
    uploads, _, generation = app_environment
    uploads[:] = [UnreadableUpload(stylesense_state.MAX_UPLOAD_BYTES + 1)]
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    assert not app.exception
    assert any('10 MB' in w.value for w in app.warning)
    generation.assert_not_called()


def test_oversized_manual_reference_rejected_without_read(app_environment, monkeypatch, image_bytes):
    uploads, _, _ = app_environment
    references = {'hairstyle_reference': UnreadableUpload(stylesense_state.MAX_UPLOAD_BYTES + 1),
                  'color_reference': Upload(image_bytes())}
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: uploads if kwargs.get('key') == 'multiple_photo_upload' else references.get(kwargs.get('key')))
    manual = Mock()
    monkeypatch.setattr(stylesense_backend, 'generate_tryon', manual)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    app.button[0].click().run()
    next(button for button in app.button if button.label == 'Generate AI Try-On').click().run()
    assert not app.exception
    assert any('10 MB' in w.value for w in app.warning)
    manual.assert_not_called()
