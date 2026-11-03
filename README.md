# StyleSense AI

StyleSense AI uses MediaPipe facial landmarks and the existing fine-tuned MobileNetV2 classifier to analyse one to three photographs. It averages the valid photographs' five-class probability vectors and recommends three hairstyles. A separately hosted HairFastGAN service performs realistic hairstyle transfer.

The classifier, class order, averaging rule and recommendation names are preserved. No retraining is needed to run the application.

## Current scope

- Face categories: Heart, Oblong, Oval, Round and Square.
- Analysis and generated images are retained across Streamlit reruns; changing uploads invalidates stale results.
- Automatic generation uses built-in references and the validated original portrait as the colour reference. Three results are handled independently, including partial failures and individual downloads.
- The three existing **Round** reference images are included. The other twelve catalogue entries explicitly have missing assets; they must be supplied and visually validated before all-five-class generation is complete.
- Manual hairstyle/colour reference uploads remain available as an advanced option.
- HairFastGAN server source, GPU hosting, URL and credentials are **not included**. Automated contract tests use simulated responses and do not prove live generation quality or availability.

## Local setup

Use Python 3.12 and run commands from the repository directory:

```sh
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The two active assets must remain present:

| Asset | Purpose |
| --- | --- |
| `best_mobilenet_finetuned.keras` | Five-class face-shape inference |
| `face_landmarker.task` | MediaPipe face landmarks |

The excluded `face_shape_model.keras` and `best_face_shape_model.keras` are historical CNN artifacts, not runtime dependencies. Do not replace the active classifier with them. The training/dataset scripts are retained for research and are not startup steps.

## Configure HairFastGAN

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, then populate it locally with the existing server's HTTPS base URL and token. The actual secrets file is ignored by Git. Alternatively, set `HAIRFAST_API_URL` and `HAIRFAST_API_TOKEN` in the deployment environment. Streamlit secrets take precedence.

Never commit credentials or include them in screenshots, issues or logs. The base URL must not contain `/tryon` at the end: the client appends that path. A backend hosted behind a temporary tunnel must have its configured base URL updated when the tunnel changes.

The service is expected to accept `POST /tryon`, Bearer authentication and multipart fields `face`, `hairstyle`, and `color`, and return a JPEG or PNG image. See [backend contract](docs/backend-contract.md) for limits and live verification requirements.

## Use the application

1. Upload one to three clear photographs of the same person.
2. Run analysis. Images with no detected face are skipped without changing the original photo numbering.
3. Review the combined face-shape prediction and three recommendations.
4. With service configuration and references available, the app generates the three recommended looks without asking you to upload hairstyle references.
5. Download each successful result; retry failed styles without discarding successful ones.

Only portraits that passed face analysis are eligible for generation. Sending the original portrait as the colour reference is the client's intended natural-colour policy; actual colour preservation must be assessed on the live backend.

## Tests

The test environment deliberately avoids downloading TensorFlow and MediaPipe. Network and model calls are mocked for unit/UI regression tests:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Run full model-loading and real portrait checks separately using `requirements.txt`; run real HairFastGAN verification only against the configured GPU service. Passing mocked tests is not an end-to-end AI validation.

## Project layout

| File | Responsibility |
| --- | --- |
| `app.py` | Existing face inference, Streamlit presentation and workflow |
| `stylesense_state.py` | Upload validation, aggregation and session-state helpers |
| `stylesense_catalogue.py` | Existing recommendations and reference coverage |
| `stylesense_backend.py` | Bounded, authenticated HairFastGAN requests and partial-failure handling |
| `tests/` | Unit and regression tests |
| `hairstyle_references/` | Existing Round reference images |

See [deployment instructions](docs/deployment.md) and [model/catalogue notes](docs/model-and-catalogue.md). Development is tracked in [GitHub Issues](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues).
