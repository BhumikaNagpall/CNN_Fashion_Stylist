from pathlib import Path

import pytest
from PIL import Image

import stylesense_catalogue as catalogue

EXPECTED = {
    'Heart': ['Side-swept bangs', 'Shoulder-length layers', 'Chin-length bob'],
    'Oblong': ['Curtain bangs', 'Layered shoulder-length hair', 'Wavy bob'],
    'Oval': ['Long layered hair', 'Soft waves', 'Curtain bangs'],
    'Round': ['Long layers', 'Side-parted hairstyle', 'Long curtain bangs'],
    'Square': ['Soft layered waves', 'Side-swept bangs', 'Textured long bob'],
}


def test_existing_class_order_and_recommendations_preserved():
    assert list(catalogue.CLASS_NAMES) == list(EXPECTED)
    assert catalogue.HAIRSTYLES == EXPECTED


@pytest.mark.parametrize('shape', list(EXPECTED))
def test_all_classes_have_three_ordered_reference_slots(shape):
    references = catalogue.get_references(shape)
    assert [name for name, _ in references] == EXPECTED[shape]
    assert len(references) == 3


def test_round_references_exist_decode_and_are_location_independent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    paths = [path for _, path in catalogue.get_references('Round')]
    assert [path.name for path in paths] == ['round_long_layers.jpg', 'round_side_parted.jpg', 'round_curtain_bangs.jpg']
    for path in paths:
        assert isinstance(path, Path) and path.is_absolute()
        with Image.open(path) as image:
            image.load()
            assert min(image.size) >= 64


@pytest.mark.parametrize('shape', ['Heart', 'Oblong', 'Oval', 'Square'])
def test_missing_assets_remain_explicit_not_mislabeled_round_images(shape, monkeypatch, tmp_path):
    monkeypatch.setattr(catalogue, 'PROJECT_ROOT', tmp_path)
    assert all(path is None for _, path in catalogue.get_references(shape))


def test_unknown_shape_rejected():
    with pytest.raises(ValueError):
        catalogue.get_references('Triangle')


@pytest.mark.parametrize('shape', list(EXPECTED))
def test_three_shipped_references_are_distinct_and_ready_for_transfer(shape):
    from stylesense_backend import normalize_image

    paths = [path for _, path in catalogue.get_references(shape)]
    assert all(path is not None for path in paths), shape
    assert len(set(paths)) == 3, shape
    for path in paths:
        # Exercise the same decode, dimension and size checks as the HTTP client.
        assert normalize_image(path.read_bytes()).startswith(b'\x89PNG\r\n\x1a\n')
