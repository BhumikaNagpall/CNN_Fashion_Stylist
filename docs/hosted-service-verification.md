# Official hosted HairFastGAN verification

Checked 2026-11-06 from the development environment. No user portraits, API secrets, or paid resources were used. This is operational evidence, not a guarantee about future service availability.

## Public API contract observed

Host: https://airi-institute-hairfastgan.hf.space/

The public `/config` and `/info` responses were readable. Server advertised Gradio 4.31.5 and SSE v3.

| Named endpoint | Inputs in order | Outputs |
| --- | --- | --- |
| `/resize_inner` | face FileData, alignment selection array | aligned face image |
| `/resize_inner_1` | hairstyle FileData, alignment selection array | aligned shape image |
| `/resize_inner_2` | color FileData, alignment selection array | aligned color image |
| `/swap_hair` | face FileData, shape FileData, color FileData, blending string, poisson iterations, poisson erosion | image and error textbox |

Observed defaults: alignment selection `["Face", "Shape", "Color"]`, blending `"Article"`, poisson iterations `0`, poisson erosion `15`.

The public Space source at https://huggingface.co/spaces/AIRI-Institute/HairFastGAN/blob/main/app.py connects upload events to the alignment functions. The swap function does not perform these alignment calls itself. It forwards images to a separate service. A running frontend and readable schema do not establish that generation works.

## Actual probes

Inputs: repository synthetic `hairstyle_references/soft_waves.jpg` as face and color; `hairstyle_references/round_long_layers.jpg` as shape. Images were decoded and saved as PNG before upload. The source image was visually checked and contains a clear frontal face.

1. `gradio_client==2.7.2`: client connected and `/resize_inner` was submitted with all three alignment choices. It failed before swap with `gradio_client.exceptions.AppError`: upstream Gradio app raised an exception and verbose error reporting was disabled. No image was generated. The underlying server exception was unavailable.
2. `gradio_client==0.16.4`: initial client construction failed with `httpx.ReadTimeout` during configuration retrieval. This was a network/configuration fetch failure, not evidence of an inference failure.
3. A final bounded diagnostic uses the older client with an increased configuration-fetch timeout and directly submits the already square 1024-pixel images to `/swap_hair`. This deliberately skips alignment to distinguish the alignment path from the generation path. It cannot establish acceptable alignment or visual quality. Client initialization succeeded, but `httpx.ReadTimeout` occurred during `_upload_file` (`httpx.post`) before a swap inference result was obtained. This does not establish that the generation backend itself failed. No output image was returned.

The scratch client environment required `socksio` for the execution environment's configured SOCKS proxy. That is an environment-specific dependency and has not been added to the application.

## Decision

No successful hosted generation has yet been verified. Do not label this service a working replacement, route user portraits to it automatically, or claim the project has passed real end-to-end generation. Failures observed here do not prove a global service outage. A provider adapter should be added only after successful output and explicit public-hosting/privacy behavior are established. Existing configurable FastAPI integration remains unchanged.
