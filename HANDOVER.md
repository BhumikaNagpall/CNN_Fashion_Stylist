# StyleSense AI development handover

Updated 3 November 2026. Repository: https://github.com/BhumikaNagpall/CNN_Fashion_Stylist. The current development branch is `develop/automatic-hairstyle-workflow`, based on main commit `d7e8582`. Review this branch and its open PR before starting any new work.

## Decisions already made

Preserve the current MediaPipe face detector, fine-tuned MobileNetV2, five labels, probability averaging, exact recommendation names and advanced manual try-on. No retraining. Active assets are `face_landmarker.task` and `best_mobilenet_finetuned.keras`. The two omitted 127.88 MiB original CNN models are historical and unnecessary for inference.

## Work tracker

| GitHub issue | Session owner | Implementation and remaining work |
|---|---|---|
| [1](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues/1) | Backend | Client contract documented in docs/backend-contract.md. Live authentication, GPU transfer quality and natural colour preservation still need verification. |
| [2](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues/2) | Frontend | Session state retains analysis, uploads and generated results; new uploads invalidate derived results. AppTest covers reruns. |
| [3](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues/3) | Frontend, QA | JPEG/PNG decoding, bounds, EXIF correction, original indices and first validated portrait implemented. |
| [4](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues/4) | ML, assets | All 15 recommendations preserved; three original Round assets decoded/inspected. Nine generated reference portraits cover the remaining twelve entries. All 15 were resolved and decoded, and the nine portraits were visually reviewed. Verify transfer quality on the live GPU. |
| [5](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues/5) | Backend, Frontend | Analysis triggers three independent transfers when references and service config exist. Per-style downloads and partial retry implemented. Live acceptance outstanding. |
| [6](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues/6) | Backend, QA | HTTPS, no redirects, bounded responses, decoded PNG output and sanitized errors implemented. |
| [7](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues/7) | QA | Latest isolated local run: 79 passed in 5.97s, including the 10 MiB multipart guard and all 15 real catalogue assignments. Model and HTTP doubles in UI tests cannot validate actual GPU output. Re-run after latest changes. |
| [8](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/issues/8) | DevOps | README, deployment guide, requirements, secret placeholders, regression workflow and CPU inference smoke workflow implemented. Check clean install before closing. |

The lead coordinates independent session agents, not GitHub bot accounts. P0: issues 1 and 5. Issue 5 depends on issues 1–4 and 6. Tests and deployment track the integrated flow.

## Current checkpoint

[Draft PR #9](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/pull/9) holds the development work. Main remains at the baseline. Use the PR branch as the source of truth across sessions. GitHub issue owners refer to session roles. No bot accounts were assigned.

- Model and landmark assets are present. A previous CPU run loaded the existing classifier and returned five finite probabilities summing to one. The slim local host lacked libGLESv2.so.2 for MediaPipe. The new CPU smoke workflow installs the native dependencies and exercises the production crop and classifier functions on a Round reference.
- The owner provided the exported Colab notebook. Source inspection confirmed FastAPI GET /health, authenticated POST /tryon, three multipart fields (face, hairstyle, color), and a PNG result from HairFast.swap with alignment. Each uploaded part has a 10 MiB limit. Source portrait reuse for colour matches the interface. Natural-colour preservation and visual identity still require a real transfer.
- The notebook contains historical model image output and a successful health check. No live authenticated POST was observed in this development session. A generated token appeared in saved notebook output. Rotate it and clear outputs before reuse. No notebook, credentials or tunnel URLs are committed.
- The original three Round references plus nine generated portraits cover all five categories. Some cuts are shared across categories. The catalogue documentation records which assets correspond to each recommendation.
- Latest local tests: 79 passed in 5.97 seconds. Simulated HTTP covers three results, partial failures, retries and downloads. Streamlit rerun tests cover retention and invalidation. Real file checks cover all 15 catalogue assignments.
- The 71-test predecessor passed [GitHub Actions](https://github.com/BhumikaNagpall/CNN_Fashion_Stylist/actions/runs/41619060581). Inspect the PR checks for the current revision.

## Resume from here

1. Open PR #9 and fetch its latest head. Inspect updated issues, CI and any new commits before editing. The audited model and completed features do not require another audit.
2. For local regression testing, use Python 3.12, install requirements-dev.txt and run python -m pytest -q. For CPU inference, use requirements.txt and the native packages in docs/deployment.md, then run python -m scripts.runtime_smoke.
3. The remaining external requirement is a running GPU service. Start the verified Colab backend, rotate the printed token and configure HAIRFAST_API_URL and HAIRFAST_API_TOKEN through private environment or Streamlit secrets. Never share tokens in chat, issues or commits.
4. Check GET /health. Run the application on an authorized test portrait and verify a real authenticated three-style Round transfer first. Check three distinct decoded images, visual style, face identity, natural hair colour, three downloads and isolated retry after one failure. Repeat for the other four categories. Validate backend-side resource limits and safe error reporting before public deployment.
5. Record sanitized results on #1 and #5, review the current PR checks and diff, and merge only when acceptance criteria pass. Update this handover before any further session break.
