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
about the unavailable service's output quality**. The optional manual flow can
continue supplying a different colour reference.

## Missing external evidence (blocking live acceptance)

The inspected repository contains no HairFastGAN server source, GPU environment,
model weights, public service URL, or credentials. The client contract above does
not prove that a deployed server implements these fields or produces images.
No live generation has been verified as part of implementing this client.

To complete issue #1, supply the existing backend repository/notebook or server
source and the HTTPS service base URL. Configure its token privately in deployment
secrets or environment variables; do not put it in an issue, commit, or report.
The service must be running with its required GPU/checkpoints. Backend versions,
VRAM requirements, upstream licensing and startup command must be established
from that actual backend, not guessed from the client.

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
