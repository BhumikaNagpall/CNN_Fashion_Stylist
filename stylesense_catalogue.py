"""Existing face-shape recommendations and explicit reference availability.

Missing references stay None: an unrelated image must never silently stand in
for a requested style. Paths resolve relative to this module, not the launch cwd.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

HAIRSTYLES = {'Heart': ['Side-swept bangs', 'Shoulder-length layers', 'Chin-length bob'],
 'Oblong': ['Curtain bangs', 'Layered shoulder-length hair', 'Wavy bob'],
 'Oval': ['Long layered hair', 'Soft waves', 'Curtain bangs'],
 'Round': ['Long layers', 'Side-parted hairstyle', 'Long curtain bangs'],
 'Square': ['Soft layered waves', 'Side-swept bangs', 'Textured long bob']}

HAIRSTYLE_REFERENCES = {'Heart': {'Side-swept bangs': None,
           'Shoulder-length layers': None,
           'Chin-length bob': None},
 'Oblong': {'Curtain bangs': None,
            'Layered shoulder-length hair': None,
            'Wavy bob': None},
 'Oval': {'Long layered hair': None, 'Soft waves': None, 'Curtain bangs': None},
 'Round': {'Long layers': 'hairstyle_references/round_long_layers.jpg',
           'Side-parted hairstyle': 'hairstyle_references/round_side_parted.jpg',
           'Long curtain bangs': 'hairstyle_references/round_curtain_bangs.jpg'},
 'Square': {'Soft layered waves': None,
            'Side-swept bangs': None,
            'Textured long bob': None}}

def get_references(shape: str) -> list[tuple[str, Path | None]]:
    """Return the three ordered recommendations and available reference paths.

    Unknown shapes raise ValueError. A missing mapping or absent file returns
    None, allowing the UI to report incomplete coverage without a false fallback.
    Image decoding and resource validation remain the API client's responsibility.
    """
    if shape not in HAIRSTYLES:
        raise ValueError("Unsupported face shape.")
    references = []
    for name in HAIRSTYLES[shape]:
        relative = HAIRSTYLE_REFERENCES[shape].get(name)
        path = PROJECT_ROOT / relative if relative else None
        references.append((name, path if path is not None and path.is_file() else None))
    return references
