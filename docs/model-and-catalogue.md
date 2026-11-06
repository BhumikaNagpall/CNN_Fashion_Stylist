# Active model and hairstyle catalogue

## Preserved inference contract

The application retains `best_mobilenet_finetuned.keras` and `face_landmarker.task`.
No weights were edited, replaced or retrained during this work.

Static inspection of the actual Keras ZIP on 2026-11-02 found:

- Functional model saved with Keras 3.15.1 on 2026-11-01 at 16:41:33 (timezone unspecified by metadata).
- Input shape `[None, 224, 224, 3]`.
- Embedded augmentation, followed by Rescaling `scale=1/127.5`, `offset=-1`.
- Nested `mobilenetv2_1.00_224` backbone, global average pooling, dropout, and five-unit softmax output.
- Input pixels must remain RGB float32 in the 0–255 range; do not normalize them again in the application.
- Current application class order is Heart, Oblong, Oval, Round, Square. This agrees with the expected alphabetical directory order in training code, but the checkpoint itself does not contain a class-name manifest. Training dataset provenance still needs confirmation.

Training preparation and inference both use MediaPipe landmark bounds with 20% padding and `ImageOps.pad` to 224×224 with grey (127,127,127) padding. Existing boundary-clamping differences have not been changed. Model loading, predictions, accuracy, calibration and live GPU transfer are not verified merely by ZIP inspection.

| Active asset | SHA-256 |
|---|---|
| `best_mobilenet_finetuned.keras` | `a1bc4b22abbe22df24edc2e4f75751621a94937d90260d77da7f1a792473aaef` |
| `face_landmarker.task` | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

The excluded `face_shape_model.keras` and `best_face_shape_model.keras` are historical CNN artifacts and are not required for this application.

## Catalogue availability

`stylesense_catalogue.py` preserves all recommendation strings and their ordering. `get_references(shape)` returns three `(name, absolute_path_or_None)` pairs. Unknown categories raise `ValueError`; missing mappings/files return `None`. This is deliberate: incomplete coverage must not silently substitute an unrelated hairstyle.

All 15 assignments resolve to files. Nine synthetic studio portraits were created for this project using image generation. They show one adult, front-facing subject on a neutral background. Each generated image was inspected for the requested cut and decoded as JPEG. The existing Round references remain separate. Actual hairstyle transfer fidelity still requires evaluation on the running backend.

| Shape | First recommendation | Second recommendation | Third recommendation |
|---|---|---|---|
| Heart | Side-swept bangs | Shoulder-length layers | Chin-length bob |
| Oblong | Curtain bangs | Layered shoulder-length hair | Wavy bob |
| Oval | Long layered hair | Soft waves | Curtain bangs |
| Round | Long layers | Side-parted hairstyle | Long curtain bangs |
| Square | Soft layered waves | Side-swept bangs | Textured long bob |

The shoulder-length layered portrait serves Heart and Oblong. The curtain bangs portrait serves Oblong and Oval. The side-swept bangs portrait serves Heart and Square. The two wave references are visually distinct: Oval uses long flowing waves, while Square uses shoulder-length layered waves. The bob portraits distinguish chin-length blunt hair, short waves and a longer textured cut. The exact style-to-filename mapping lives in `stylesense_catalogue.py`.

## Round reference validation

All existing files remain byte-for-byte unchanged. Pillow fully decoded each as RGB PNG, despite its `.jpg` suffix. API clients should use decoded content or normalize to PNG rather than infer MIME type from the filename.

| Existing file | Actual format | Dimensions | Visual inspection |
|---|---|---|---|
| `round_long_layers.jpg` | PNG | 486×630 | Long layered hair is visible; also includes bangs and hands touching the hair. Transfer may reproduce these additional attributes or be affected by occlusion. |
| `round_side_parted.jpg` | PNG | 438×701 | Side part and long layers are visible; slight head tilt and face framing require transfer validation. |
| `round_curtain_bangs.jpg` | PNG | 387×516 | Long face-framing fringe/layers are visible; angled pose and busy mirrored salon background may affect segmentation. |

These are moderate-resolution candidate inputs. Visual inspection supports the general style mapping, but does not prove HairFastGAN compatibility or output quality. Validate all three with a real service and a consented test portrait before claiming a working end-to-end result. Reference licensing/source attribution is not present in the repository and remains unverified.

| Round asset | SHA-256 |
|---|---|
| `round_long_layers.jpg` | `298e5ceb3c598d6c48b587a01cdc84d09d5a3b880db8c3c3773a4fb2af813f8a` |
| `round_side_parted.jpg` | `58da100450e57082dc4f483f4d912a4f4c76062db3cdd893914a20d1b6d7eab8` |
| `round_curtain_bangs.jpg` | `01fa12c955ea97b1c9e25b70c2965519c5511ab9e112b065446f30792cc0f51b` |
