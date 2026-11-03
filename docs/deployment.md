# Deploy the Streamlit application

Use a Linux x86-64 host with Python 3.12 and a separate HTTPS HairFastGAN service. Keep the application directory, model files and hairstyle reference directory together. The two inference files ship in this repository. Historical CNN checkpoints and training data are optional for serving.

## Host prerequisites

Install the graphics runtime used by the MediaPipe Tasks native library. On Ubuntu 24.04 the missing library `libGLESv2.so.2` is supplied by `libgles2`. A minimal image also needs OpenCV's native dependencies. For example, with administrator access:

```sh
apt-get update
apt-get install -y libgles2 libegl1 libgl1 libglib2.0-0t64
```

Use the package names for your distribution if deploying elsewhere. Install the Python environment using the commands in the [README](../README.md). Start the service with:

```sh
python -m pip check
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Terminate HTTPS at a reverse proxy or the hosting platform. Preserve WebSocket connections and use sticky sessions if running multiple replicas. Each process caches its own classifier and detector. Session data is held in process memory and will be lost on restart. Size your workers for decoded photos, the model and up to three output images per active session.

## Configure the generator

Set `HAIRFAST_API_URL` and `HAIRFAST_API_TOKEN` through the platform secret manager or ignored `.streamlit/secrets.toml`. Supply the HTTPS base URL without a trailing `/tryon`. Restart the application after changing secrets. Use a token limited to the intended backend. The Streamlit host requires outbound HTTPS access to that service.

The external server owner must provide the HairFastGAN implementation, compatible GPU/software environment, weights, a stable HTTPS endpoint and secure token. The client sends a validated portrait and style/colour references as multipart uploads. Confirm backend retention and cleanup policies before sending real portraits. A notebook tunnel must be online when generating looks. Document its startup sequence and update the service URL if the tunnel changes. See [the client contract](backend-contract.md).

## Verification and rollout

1. Install `requirements.txt` in a clean environment. Check package consistency with `python -m pip check`.
2. Launch Streamlit. Upload a consented portrait and verify detection, cropping, five scores and the combined prediction. Test a blank image alongside the portrait to check skipped-photo numbering.
3. Using a test portrait and the three built-in Round style references, exercise the real service. Record the status and visual result for each style, confirm identity/colour preservation, retry behaviour and downloads. Store only approved, non-sensitive evidence.
4. Verify missing configuration, service unavailability, malformed responses and partial results. Successful styles should remain available after retrying failed styles.
5. Verify each remaining catalogue entry using its approved reference. Coverage and release status are tracked in issues #1, #4 and #5.

During development the Linux Python 3.12 runtime packages resolved and installed together. The committed MobileNet model loaded and returned a finite five-class vector summing to 1 on a synthetic input. This confirms model loading and prediction shape. The original slim development host lacked `libGLESv2.so.2` at MediaPipe initialization. Confirm the native runtime on the deployment host. Model accuracy and live HairFastGAN image quality require separate validation.

The GitHub Actions job installs `requirements-dev.txt` and runs `python -m pytest -q` without live credentials. Use the application runtime for real model inference and the separately provisioned service for live GPU verification.
