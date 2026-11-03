"""API contract tests use simulated HTTP responses, never live HairFastGAN."""
from io import BytesIO
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

import stylesense_backend as backend


class Response:
    def __init__(self, payload=b'', status=200, content_type='image/png', headers=None):
        self.payload, self.status_code = payload, status
        self.headers = {'content-type': content_type, **(headers or {})}
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.payload

    def close(self):
        self.closed = True


def test_success_contract_and_jpeg_normalization(monkeypatch, image_bytes):
    response = Response(image_bytes(format='JPEG'), content_type='image/jpeg; charset=binary')
    post = Mock(return_value=response)
    monkeypatch.setattr(backend.requests, 'post', post)
    portrait = image_bytes()
    result = backend.generate_tryon(portrait, portrait, portrait, 'https://example.test/', 'test-token')
    assert Image.open(BytesIO(result)).format == 'PNG'
    args, kwargs = post.call_args
    assert args == ('https://example.test/tryon',)
    assert kwargs['headers'] == {'Authorization': 'Bearer test-token'}
    assert set(kwargs['files']) == {'face', 'hairstyle', 'color'}
    assert kwargs['files']['face'][1] == kwargs['files']['color'][1]
    assert kwargs['allow_redirects'] is False
    assert kwargs['stream'] is True
    assert kwargs['timeout'] == (15, 600)
    assert response.closed


@pytest.mark.parametrize('status', [302, 401, 403, 429, 500])
def test_error_status_never_exposes_server_body(monkeypatch, image_bytes, status):
    response = Response(b'password=SUPER_SECRET /private/server/path', status=status)
    monkeypatch.setattr(backend.requests, 'post', Mock(return_value=response))
    data = image_bytes()
    with pytest.raises(backend.TryOnError) as error:
        backend.generate_tryon(data, data, data, 'https://example.test', 'test-token')
    assert 'SUPER_SECRET' not in str(error.value)
    assert 'private/server' not in str(error.value)
    assert response.closed


@pytest.mark.parametrize('exception', [requests.exceptions.Timeout('secret-token'), requests.exceptions.ConnectionError('secret-token')])
def test_transport_failures_are_safe(monkeypatch, image_bytes, exception):
    monkeypatch.setattr(backend.requests, 'post', Mock(side_effect=exception))
    data = image_bytes()
    with pytest.raises(backend.TryOnError) as error:
        backend.generate_tryon(data, data, data, 'https://example.test', 'test-token')
    assert 'secret-token' not in str(error.value)


@pytest.mark.parametrize('payload,ctype,headers', [
    (b'not an image', 'image/png', {}),
    (b'<html>secret</html>', 'text/html', {}),
    (b'', 'image/png', {}),
    (b'', 'image/png', {'content-length': '999999999'}),
    (b'', 'image/png', {'content-length': '-1'}),
    (b'', 'image/png', {'content-length': 'bad'}),
])
def test_invalid_generated_images_rejected(monkeypatch, image_bytes, payload, ctype, headers):
    response = Response(payload, content_type=ctype, headers=headers)
    monkeypatch.setattr(backend.requests, 'post', Mock(return_value=response))
    data = image_bytes()
    with pytest.raises(backend.TryOnError):
        backend.generate_tryon(data, data, data, 'https://example.test', 'test-token')
    assert response.closed


def test_streamed_response_size_bound(monkeypatch, image_bytes):
    data = image_bytes()
    response = Response(b'x' * (len(data) + 1))
    monkeypatch.setattr(backend, 'MAX_IMAGE_BYTES', len(data))
    monkeypatch.setattr(backend.requests, 'post', Mock(return_value=response))
    with pytest.raises(backend.TryOnError, match='transfer size'):
        backend.generate_tryon(data, data, data, 'https://example.test', 'test-token')
    assert response.closed


@pytest.mark.parametrize('url,token', [
    ('http://example.test', 'x'), ('https://user:pass@example.test', 'x'),
    ('https://example.test?token=secret', 'x'), ('https://example.test#fragment', 'x'),
    ('https://example.test:bad', 'x'), ('https://example.test', ''),
    ('https://example.test', 'bad\nheader'),
])
def test_invalid_configuration_rejected_before_network(monkeypatch, image_bytes, url, token):
    post = Mock()
    monkeypatch.setattr(backend.requests, 'post', post)
    data = image_bytes()
    with pytest.raises(backend.TryOnError):
        backend.generate_tryon(data, data, data, url, token)
    post.assert_not_called()


@pytest.mark.parametrize('size,format', [((10, 10), 'PNG'), ((256, 256), 'GIF'), ((10001, 64), 'PNG')])
def test_invalid_input_dimensions_and_actual_format(image_bytes, size, format):
    with pytest.raises(backend.TryOnError):
        backend.normalize_image(image_bytes(size=size, format=format))


def test_three_generation_calls_and_colour_reference(monkeypatch, image_bytes):
    data = image_bytes()
    generator = Mock(return_value=data)
    monkeypatch.setattr(backend, 'generate_tryon', generator)
    references = {'Layers': data, 'Side part': data, 'Bangs': data}
    results = backend.generate_three(data, references, 'https://example.test', 'test-token')
    assert list(results) == list(references)
    assert generator.call_count == 3
    assert all(r == {'image': data, 'error': None} for r in results.values())
    assert all(call.args[0] == call.args[2] == data for call in generator.call_args_list)


def test_partial_failure_retry_keeps_successes(monkeypatch, image_bytes):
    data = image_bytes()
    generator = Mock(side_effect=[data, backend.TryOnError('Timed out'), data])
    monkeypatch.setattr(backend, 'generate_tryon', generator)
    refs = {'A': data, 'B': data, 'C': data}
    first = backend.generate_three(data, refs, 'https://example.test', 'test-token')
    assert first['A']['image'] == first['C']['image'] == data
    assert first['B'] == {'image': None, 'error': 'Timed out'}
    generator.reset_mock(side_effect=True)
    generator.return_value = data
    second = backend.generate_three(data, refs, 'https://example.test', 'test-token', first)
    assert generator.call_count == 1
    assert second['A'] == first['A']
    assert second['C'] == first['C']
    assert all(r['image'] for r in second.values())


def test_missing_references_are_partial_results(monkeypatch, image_bytes, tmp_path):
    data = image_bytes()
    generator = Mock(return_value=data)
    monkeypatch.setattr(backend, 'generate_tryon', generator)
    results = backend.generate_three(data, {'A': data, 'B': None, 'C': tmp_path/'absent.png'}, 'https://example.test', 'test-token')
    assert results['A']['image'] == data
    assert results['B']['error'] and results['C']['error']
    assert generator.call_count == 1


def test_non_three_catalogue_is_rejected():
    with pytest.raises(backend.TryOnError, match='Exactly three'):
        backend.generate_three(b'', {}, '', '')
