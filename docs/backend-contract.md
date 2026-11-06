# HairFastGAN integration contract and verification status

## Verified from source

The original `app.py` makes `POST {HAIRFAST_API_URL}/tryon` requests with
`Authorization: Bearer <HAIRFAST_API_TOKEN>` and three multipart file fields:

| Field | Meaning |
| --- | --- |
| `face` | Validated source portrait |
| `hairstyle` | Selected hairstyle reference image |
| `color` | Hair-colour reference image |

The client expects HTTP 200 and a raw image body. The hardened client accepts
JPEG/PNG, fully decodes the image and normalizes it to PNG for correct downloads.
Its multipart filenames are fixed, stripped of user metadata, and use image/png.
It uses a 15-second connect timeout and 600-second read timeout, streaming at most
20 MiB. A read timeout is an inactivity timeout, not a durable job deadline.
Requests are synchronous and sequential, with no automatic HTTP retries that
could duplicate GPU work. Retrying a failed style preserves successful siblings.

`stylesense_backend.generate_tryon(...)` handles one transfer. `generate_three`
handles exactly three named references independently and returns a mapping from
style name to `{image: PNG bytes or None, error: safe text or None}`. Callers must
discard previous results whenever the portrait or reference catalogue changes.

The automatic flow submits the original validated portrait for both `face` and
`color`. This is a proposed colour-preservation strategy, **not a verified claim
about live service output quality**. The optional manual flow can
continue supplying a different colour reference.

## Verified exported notebook implementation

The project owner supplied a notebook export for local source review. Its final
API cell writes a FastAPI application with an unauthenticated GET `/health`
and an authenticated POST `/tryon`. The POST route compares the Authorization
header with the Bearer token from `HAIRFAST_API_TOKEN`. Its three required
UploadFile fields exactly match the table above. Each must be nonempty and at
most 10 MiB. The client enforces this outbound limit after PNG conversion.

The server creates a per-request temporary directory, initializes one HairFast
instance at startup, and serializes calls with a process-local lock under
PyTorch inference mode. It calls `hair_fast.swap(face_path, hairstyle_path,
color_path, align=True)`, saves the first result tensor to memory and returns
a PNG StreamingResponse. Wrong authorization returns 401, empty parts 400,
oversized parts 413 and inference exceptions 500. The client suppresses raw
backend error details.

The notebook provisions the HairFastGAN checkout, Python dependencies and
model checkpoints within a CUDA-capable Colab runtime. The final launcher
supplies a generated token to Uvicorn (`api_server:app`, port 8000).
A cloudflared tunnel exposes the local service. Start the model process
first, verify its local health endpoint, then configure the current HTTPS
tunnel and matching token privately on the Streamlit host. Multiple exploratory
installation and repair cells exist. An unattended clean GPU deployment and a
specific minimum VRAM have not been validated.

The export contains historical image display output after a local inference
cell and a historical HTTP 200 health probe. These concern local inference and
health, respectively. Neither verifies a current authenticated POST from the
Streamlit app. No live request or assessment of colour preservation was performed.

The server reads each uploaded part into memory before checking its size and
returns exception detail on inference failures. Before exposing a persistent
service, bound request sizes upstream, validate decoded images on the server,
limit concurrency and replace raw error text with safe diagnostics. The
notebook launcher can print connection credentials. Rotate the runtime token,
clear notebook outputs and keep the notebook out of version control.

## Live acceptance procedure

1. Review backend source for `/tryon`, Bearer auth and field compatibility.
2. Confirm HTTPS, model/checkpoint initialization and healthy GPU availability.
3. Use a consented portrait with a valid detected face and each unchanged Round
   reference. Confirm three distinct, decodable outputs and identity preservation.
4. Verify whether using the source portrait as `color` preserves natural colour.
5. Exercise invalid authentication, a missing/invalid upload, a timeout and one
   failed generation. Record sanitized status/results, never credentials or bodies.
6. Check portrait retention/deletion, isolated temporary files, upload limits,
   server-side validation, authentication and rate limits on the actual server.

Mocked tests validate client behaviour only and cannot satisfy steps 2–4.

## Deployment boundary

Configure `HAIRFAST_API_URL` as an HTTPS base URL (without `/tryon`, query, fragment,
or embedded credentials) and `HAIRFAST_API_TOKEN` as a private nonempty token.
TLS verification remains enabled. Redirects are rejected. Run the Streamlit client
and GPU service separately if their ML runtimes conflict. Expose only an
authenticated HTTPS service, not a development/debug server. Server-side queueing
and job polling are future enhancements requiring an agreed backend contract.
