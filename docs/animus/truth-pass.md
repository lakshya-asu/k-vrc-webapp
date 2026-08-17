# K-VRC truth pass for the Animus branch

Date: 2026-08-17. Base commit: `c8e5bdc`.

This is a repository evidence check. It does not test a deployed Modal service
or the live Vercel site.

## Verified in the repository

- `api/chat.js` asks Claude for a reply, emotion, gesture, and expression.
- `src/animationController.js` maps semantic gesture names to animation clips
  loaded from the GLB.
- `src/faceScreen.js` implements the canvas face system.
- `modal_app/` contains an encoder, four PyTorch heads, training code, a FastAPI
  endpoint, and extraction scripts.
- The Python tests exercise shapes and endpoint response structure.

## Claims not supported by tracked artifacts

The main README says body movement was distilled from video through SAM2,
ViTPose, MiniLM, and four trained heads, then served through a live Modal T4
endpoint.

The checked-in tree does not contain training data, trained `.pt` weights, a
model manifest, an evaluation report, or a deployment receipt. `data/` is
ignored. No tracked `.pt` or `.pth` file exists.

`modal_app/serve.py` creates each head with random initialization when the
corresponding weight file is absent. It also falls back to generic names such
as `clip_0` when `clip_names.json` is absent.

`src/animationController.js` names a method `blendClips`, but the current
implementation sorts weights and plays only the top clip. It does not perform
weighted multi-clip blending.

The repository therefore proves a learned-animation scaffold. It does not by
itself prove trained or deployed learned animation.

## Animus correction

This branch treats the existing baked-clip controller as the verified body
baseline. Learned heads remain experimental until a reproducible receipt
names:

- source commit
- dataset manifest and rights
- training command and environment
- checkpoint hashes
- held-out metrics
- qualitative review set
- deployment identifier
- end-to-end request and output artifact

The Animus proof does not depend on those heads. Its first learned component is
the provider-neutral actor planner, which is bounded by a strict contract and a
deterministic fallback.

## Verification scope on this branch

The JavaScript Animus tests, deterministic CLI check, and production web build
run locally without adding dependencies.

The inherited Python test suite was not run in this host interpreter because
NumPy and PyTorch are absent. This branch does not change the Python animation
scaffold. A later reproducible Python environment must run that suite before
any learned-head claim is restored.

## README policy for this branch

Do not cite the historic `Learned Body Animations` section as current proof.
Use this truth pass and the Animus proof README until training and deployment
receipts exist.
