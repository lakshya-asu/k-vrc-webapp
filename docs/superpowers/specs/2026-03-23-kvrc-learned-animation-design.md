# K-VRC Learned Animation System — Design Spec
**Date:** 2026-03-23
**Status:** Approved for implementation planning

---

## Overview

Replace K-VRC's rule-based animation system (fixed emotion → clip mapping) with a learned system that generates contextually appropriate body motion, face expressions, and screen content from conversational context. The system uses a shared text encoder with four task-specific heads, trained on three data streams including ground-truth motion extracted from ~20 minutes of K-VRC animation episodes.

---

## Goals

- Body motion driven by conversational context, not just a discrete gesture label
- Face expression parameters continuous and blended (32 dims), not 6 fixed states
- Screen content reactive to mood, energy, and topic
- Inference latency under 50ms (imperceptible on top of existing Claude + TTS delay)
- Each component independently trainable and deployable

---

## Architecture

### Data Flow

```
Conversation turn
      │
      ▼
Claude API (/api/chat) — extended response schema:
{ reply, emotion, gesture,
  expression_weights,   ← semantic labels for training only
  motion_energy,        ← 0–1 float, training only
  screen_mood }         ← categorical token, training only
      │
      ├── (production) → Modal Inference Server
      │                   POST /infer { text, history }
      │                   response: clip_weights, motion_deltas, face_params, screen_tokens
      │
      └── (training)  → Label Collector
                         logs (context, reply, semantic_labels) to data/raw/
```

```
Modal /infer
      │
      ├─ Shared Encoder (all-MiniLM-L6-v2, 22MB)
      │   input: last 3 turns of conversation context (concatenated)
      │   output: 384-dim embedding
      │
      ├─ Head 1: Clip Retrieval
      │   output: blend weights [N_clips] — N derived from GLB at training time
      │
      ├─ Head 2: MDM Refinement
      │   output: delta rotations [60 × 6 × 4] — axis-angle as quaternion xyzw
      │
      ├─ Head 3: Face Expression
      │   output: 32 continuous expression parameters [0–1 floats]
      │
      └─ Head 4: Screen Content
          output: 8 continuous params + 8-way color softmax
```

---

## Data Pipeline — 3 Streams

### Stream A — Synthetic Labels (Claude)
- Claude API prompt extended with `expression_weights`, `motion_energy`, `screen_mood` output fields
- Every production conversation turn logged to `data/raw/` as JSONL
- Schema: `{ timestamp, context_window, reply, emotion, gesture, expression_weights, motion_energy, screen_mood }`
- Passive accumulation — no manual work required
- Primary training signal for Head 3 (face) and Head 4 (screen)

### Stream B — Manual Annotation Tool
- Local Vite page (`annotate.html`) served by a small Express server (`pipeline/annotate.py` via Modal or local node)
- On startup: reads all records from `data/raw/*.jsonl`, presents them in a queue
- UI: K-VRC face canvas preview (same renderer as production) + 32 sliders for expression params
- Saves corrected records to `data/annotated/` with same JSONL schema as Stream A
- Annotation server discovers raw records by scanning `data/raw/` directory on load
- Higher quality than Stream A; used to fine-tune Head 3 and Head 4

### Stream C — Video Extraction (primary signal for body motion)
- Source: ~20 minutes of K-VRC animation episodes, single consolidated MP4
- Resolution: 1920×1080 @ 25fps (~30,000 frames)
- Camera angles inconsistent — strategy validated on test clip before full run

**Extraction pipeline (runs on Modal A100):**

```
MP4
 │
 ├─ SAM2 (segment-anything-2)
 │   input: one manual click on K-VRC in frame 0 (bounding box prompt)
 │   output: per-frame binary mask + tight bounding box
 │
 ├─ ViTPose (vitpose-h, HuggingFace: ViTPose/vitpose-h-simple)
 │   input: cropped + masked frames (person crop from SAM2 bbox)
 │   output: COCO-17 2D keypoints + confidence per frame
 │   selected joints mapped to K-VRC bones:
 │     nose        → Head proxy
 │     left/right shoulder → LeftArm / RightArm
 │     left/right elbow   → LeftForeArm / RightForeArm
 │     mid-shoulder avg   → Spine2 proxy
 │   note: full 3D lifting skipped — inconsistent camera angles make
 │         monocular depth unreliable; Head 2 learns 2D→rotation mapping
 │
 ├─ Screen region extractor
 │   input: SAM2 mask → isolate face screen bounding box (fixed relative
 │          position within K-VRC's head bbox)
 │   output: per-frame [color_histogram_32bins, mean_brightness, edge_density]
 │
 └─ Whisper (openai/whisper-large-v3)
     output: word-level timestamps
     aligned with frames → sliding 3-turn text windows matched to frame ranges
     final records: (text_window, keypoint_sequence_60frames, screen_features)
```

**Head 1 training target from Stream C:**
Observed 60-frame keypoint sequences are matched to the nearest Mixamo clip using cosine similarity against pre-baked clip keypoint sequences (bake by running each clip through ViTPose on a rendered front-facing reference). This provides pseudo-labels `(text_window → clip_id)` for Head 1 contrastive training. Confidence-filtered: only sequences where top-1 clip similarity > 0.7 are used.

**Stream C trains:** Head 1 (clip retrieval) and Head 2 (MDM refinement)
**Streams A+B train:** Head 3 (face) and Head 4 (screen)

---

## Model Architectures

### Shared Encoder
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Size: 22MB
- Input: concatenated last 3 conversation turns (user + K-VRC alternating), max 256 tokens
- Output: 384-dim context embedding
- Fine-tuned jointly in Phase 5 after all heads are pre-trained independently

### Head 1 — Clip Retrieval
- Architecture: single linear projection layer + softmax
- Input: 384-dim context embedding
- Output: blend weights `[N_clips]` — N baked from GLB at training time, stored as a constant in the model artifact alongside clip names. If GLB changes, model must be retrained.
- At inference: top-3 clips + weights returned; `AnimationController.blendClips(weights)` applies them
- Training: contrastive loss on (text_window, clip_id) pairs from Stream C + Stream A gesture labels
- Parameter count: ~68K

### Head 2 — MDM Refinement
- Architecture: 4-layer transformer encoder-decoder
  - Encoder: context embedding [1 × 384] repeated to [60 × 384], positional encoding added
  - Decoder: base clip joint rotations [60 × 6 × 4] as queries (quaternion xyzw, normalized)
  - Output: additive delta quaternions [60 × 6 × 4]
- **Rotation representation:** quaternion (xyzw). Deltas applied as: `q_final = normalize(q_base + q_delta)`. This is an approximation valid for small deltas; Head 2 is regularized to produce small deltas (L2 penalty on delta magnitude).
- Serialization in `/infer` response: flattened float array of length 1440 (60×6×4), reshaped client-side
- Joints (index order fixed): [Head, Spine2, LeftArm, RightArm, LeftForeArm, RightForeArm]
- Applied in `robot.js` `updateRobot()` on top of blended clip output each frame
- Training: Stream C upper body keypoint sequences → pseudo-rotation targets via inverse kinematics approximation
- Parameter count: ~2M
- Inference: ~30ms CPU, ~5ms T4

### Head 3 — Face Expression
- Architecture: 3-layer MLP (384 → 256 → 128 → 32), all sigmoid outputs
- Input: 384-dim context embedding
- Output: 32 continuous parameters [0–1]:
  - Brow: raise, furrow, asymmetry [3]
  - Eyes: openness, squint, glitch_flicker [3]
  - Mouth: curl_left, curl_right, open, smile_width [4]
  - Overlay: scan_line_speed, scan_line_opacity, glitch_intensity, glitch_color_shift [4]
  - Color: temperature, saturation, brightness [3]
  - Pattern: noise_scale, noise_speed, vignette [3]
  - Misc: blink_rate, pupil_dilation, iris_glow_intensity [3]
  - Reserved: [9] — available for future expression dimensions
- **faceScreen.js rewrite required:** current renderer uses 6 discrete `EXPRESSIONS` objects. Must be rewritten to accept and interpolate the 32 continuous parameters. `setEmotion(emotion)` replaced by `setParams(face_params)`. The old 6-state path is kept as fallback until Phase 6 ships. faceScreen.js rewrite is an explicit build step (Build Order step 6a).
- Training: Stream A (primary, MSE loss against Claude expression_weights mapped to param space) + Stream B fine-tune
- Parameter count: ~200K

### Head 4 — Screen Content
- Architecture: 3-layer MLP (384 → 128 → 64 → 16), shared training loop with Head 3
- Input: 384-dim context embedding
- Output: 16 values split into two groups:
  - Positions 0–7: continuous params via sigmoid [energy, pattern_scale, pattern_speed, vignette, noise_intensity, glow_radius, chromatic_aberration, flicker_rate]
  - Positions 8–15: mood classification via softmax over 8 buckets [cold, warm, glitch, static, data, boot, angry, dream]
- `color_accent` removed from Head 4 — mapped directly from mood bucket (deterministic lookup table, not learned)
- `mood` vocabulary is exactly 8 discrete values matching the softmax buckets — no compound values
- Drives canvas background layer and overlay effects in `main.js`
- Parameter count: ~150K

---

## Modal Inference Server

### Infrastructure
- Framework: Modal + FastAPI
- Compute: CPU container for inference (warm instance kept alive with Modal's `keep_warm=1`)
- `keep_warm=1` eliminates the cold start problem for production — first request after deploy may be slow, subsequent requests are ~40ms
- Vercel `/api/chat` timeout risk mitigated by `keep_warm` — cold start only on fresh deploy, not on idle
- T4 GPU available for training runs, not needed for inference

### Endpoints

**`POST /infer`**
```json
// Request
{
  "text": "current reply text",
  "history": [{ "role": "user"|"assistant", "text": "..." }]
}

// Response
{
  "clip_weights": { "breathing_idle": 0.6, "talking": 0.3, "explaining": 0.1 },
  "motion_deltas": [0.01, -0.02, ...],
  "motion_deltas_shape": [60, 6, 4],
  "face_params": { "brow_raise": 0.2, "glitch_intensity": 0.05, ... },
  "screen_tokens": {
    "mood": "cold",
    "continuous": { "energy": 0.4, "pattern_scale": 0.6, ... }
  }
}
```

**`POST /train`** (authenticated, internal)
Triggers a training run from `data/processed/` on Modal's GPU fleet.

**`POST /extract`** (authenticated, internal)
Triggers Stream C video extraction pipeline on a given MP4 path.

### File Structure
```
modal_app/
  app.py          ← Modal app definition, volume mounts, secrets, keep_warm
  encoder.py      ← MiniLM loading + embed(text_window) → 384-dim
  heads.py        ← all 4 head definitions, forward pass, weight loading
  serve.py        ← FastAPI /infer endpoint
  train.py        ← unified training loop, phase-aware (which heads to train)
  pipeline/
    extract.py    ← SAM2 + ViTPose + Whisper extraction
    label.py      ← Stream A: extends Claude response with semantic labels
    annotate.py   ← Stream B: annotation server (reads data/raw/, writes data/annotated/)
    dataset.py    ← merges all 3 streams → normalized tensors in data/processed/
```

---

## K-VRC Frontend Integration

### `api/chat.js` changes
- After receiving Claude response, fires `POST /infer` to Modal with conversation context
- Merges `clip_weights`, `motion_deltas`, `face_params`, `screen_tokens` into response to browser
- If Modal call fails or times out (>4s): falls back to returning Claude's `emotion` + `gesture` fields only; browser uses existing rule-based system. **Old `setEmotion` / `playGesture` path kept until Phase 6 is stable in production.**

### `animationController.js`
- New method: `blendClips(weights)` — takes `{ clip_name: weight }` dict
- Internally: sets `action.setEffectiveWeight(w)` for each active action, ensures weights sum to 1.0
- Replaces `_play()` as primary playback path; `_play()` kept for single-clip fallback
- `this.current` refactored to `this.activeActions = Map<clip_name, AnimationAction>`
- Idle rotation logic (`_playRandomIdle`, `_isIdling`) preserved but operates on `activeActions`

### `robot.js`
- `updateRobot(delta)` applies `motion_deltas` on top of blended clip output each frame
- Delta application: `q_final = normalize(q_base + q_delta)` per joint per frame
- `motion_deltas` stored as `Float32Array[60][6][4]` deserialized from flat array + shape
- Frame index tracked modulo 60; resets when new deltas arrive

### `faceScreen.js` (Phase 6a rewrite)
- New method: `setParams(face_params)` — accepts the 32-param dict
- Internal state: `_targetParams` dict; canvas renderer lerps toward target each frame
- `setEmotion(emotion)` kept as thin wrapper that maps emotion → approximate param dict (fallback path)
- Old `EXPRESSIONS` objects converted to param dicts and retained as named presets

### `main.js` canvas renderer
- Reads `screen_tokens.mood` (maps to 8-bucket lookup) and `screen_tokens.continuous`
- Updates canvas background layer with `mood` color palette and `pattern_scale`/`pattern_speed`

---

## Training Phases

| Phase | Data | Heads trained | Goal |
|-------|------|--------------|------|
| 1 | Single test clip (Stream C) | — | Validate SAM2 + ViTPose extraction; tune confidence threshold |
| 2 | Stream C full 20min | Head 1, Head 2 | Body motion working |
| 3 | Stream A (500+ turns) | Head 3, Head 4 | Face + screen working |
| 4 | Stream B annotations | Head 3, Head 4 | Expression quality |
| 5 | All streams | Encoder (fine-tune) + all heads frozen except output layers | End-to-end coherence; loss = weighted sum of Head 1–4 losses with equal initial weights, tuned empirically |
| 6 | Stream B top-quality subset | Head 3 only | Final expression fine-tune |

---

## Build Order

1. Modal project scaffold — `app.py`, `serve.py` stub, `encoder.py`, Modal secrets + volume
2. Stream C extraction pipeline — `extract.py` (SAM2 + ViTPose + Whisper), test on single clip
3. `dataset.py` — merge streams, bake Mixamo clip reference keypoints for Head 1 pseudo-labels
4. Head 1 + Head 2 training — `train.py` Phase 2
5. Claude API schema extension — add `expression_weights`, `motion_energy`, `screen_mood` to `api/chat.js` + `api/sidenote.js` prompt; start Stream A logging
6. Head 3 + Head 4 training — `train.py` Phase 3
6a. `faceScreen.js` rewrite — 32-param renderer, `setParams()`, keep `setEmotion()` as fallback
7. Frontend integration — `animationController.js` `blendClips()`, `robot.js` delta application, `main.js` screen tokens, `api/chat.js` Modal call
8. Stream B annotation tool — `annotate.py` + `annotate.html`
9. Phase 4 fine-tune (Head 3 + Head 4 on annotated data)
10. Phase 5 joint fine-tune (encoder + all heads)

---

## Out of Scope

- Full 3D pose lifting (inconsistent camera angles — 2D keypoints + learned mapping used)
- Leg/locomotion animation (K-VRC floats; legs are secondary)
- Real-time training from production conversations (batch only)
- Voice/audio features as encoder input (text-only for v1)
- Replacing SAM2 with automatic detection (one manual click per video is acceptable)
