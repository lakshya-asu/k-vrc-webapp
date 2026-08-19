# Viseme mapping: Rhubarb mouth shapes to K-VRC face keys

This is the P2 speech-and-lip-sync mapping. It converts Rhubarb Lip Sync
mouth cues into a timed shape-key track the K-VRC embodiment profile can
apply. The table lives in `visemes.py`; this file explains the choices.

## Why these two tool sets

- Kokoro-82M (Apache-2.0) renders the final dialogue text to a 24kHz WAV
  on CPU. Small enough to run while the GPU is busy.
- Rhubarb Lip Sync (MIT) reads that exact WAV plus the exact transcript
  and emits timed mouth cues.

Order matters: render the audio first, then lip-sync that same audio with
that same text. Regenerating the audio invalidates the old cue timing, so
the pipeline always rewrites audio, cues, and the viseme track together.

## Rhubarb's mouth shapes

Rhubarb uses an extended Preston Blair set. The basic shapes are A to F
plus the idle shape X; G and H are optional extended shapes enabled with
`--extendedShapes GHX`.

| Shape | Typical sound            | Mouth                          |
|-------|--------------------------|--------------------------------|
| A     | P, B, M                  | closed                         |
| B     | many consonants, EE      | slightly open, clenched teeth  |
| C     | EH, AE                   | open                           |
| D     | AA                       | wide open                      |
| E     | AO, ER                   | slightly rounded, open         |
| F     | UW, OW, W                | puckered                       |
| G     | F, V (extended)          | upper teeth on lower lip       |
| H     | L (extended)             | tongue up                      |
| X     | silence                  | idle, rest                     |

## K-VRC mouth channels

K-VRC has no physical mouth. Its face is a screen with named float
parameters (see `modal_app/heads.py` `FACE_PARAM_NAMES` and the
expression-library design). The mouth is driven by four of them, each a
float in 0..1:

- `mouth_open` how far the mouth opens
- `smile_width` how wide the mouth stretches
- `mouth_curl_left` left corner lift
- `mouth_curl_right` right corner lift

A single shape-key weight is one float, the exact analog of one FCurve
channel value in the P1 Blender bridge, where a bone channel carries one
float per keyframe.

## The mapping

Weights are authored defaults, not measured captures. They keep the two
mouth corners symmetric and land silence and the closed shapes on a shut
mouth so a pause reads as a resting face.

| Shape | mouth_open | smile_width | curl_left | curl_right |
|-------|-----------:|------------:|----------:|-----------:|
| X     | 0.00       | 0.10        | 0.05      | 0.05       |
| A     | 0.00       | 0.15        | 0.05      | 0.05       |
| B     | 0.20       | 0.55        | 0.10      | 0.10       |
| C     | 0.45       | 0.40        | 0.05      | 0.05       |
| D     | 0.85       | 0.30        | 0.00      | 0.00       |
| E     | 0.50       | 0.15        | 0.00      | 0.00       |
| F     | 0.30       | 0.00        | 0.00      | 0.00       |
| G     | 0.15       | 0.35        | 0.05      | 0.05       |
| H     | 0.35       | 0.30        | 0.05      | 0.05       |

## Artifact shape

The converter emits an `animus_viseme_take` whose take fields mirror the
P1 bridge `perform_take` params so the same bridge code path could later
apply it:

| bridge take            | viseme take           | meaning                    |
|------------------------|-----------------------|----------------------------|
| `object`               | `object`              | target datablock           |
| `name_hint`            | `name_hint`           | base name for the take     |
| `frame_start/end`      | `frame_start/end`     | frame span                 |
| `samples[].bone`       | `samples[].shape_key` | one animated channel       |
| `samples[].frame`      | `samples[].frame`     | integer keyframe           |
| `samples[].location`   | `samples[].weight`    | one scalar value           |

Each Rhubarb cue start becomes one keyframe. Every keyframe writes the
full four-key mouth set, so the mouth is always fully specified and never
inherits a stale weight from the previous cue. `frame = frame_start +
round(cue.start * fps)`. The artifact also carries a `cues` list (one
entry per cue with its full weight set) for readability and a `source`
provenance block (text, voice, TTS settings, audio sha256, cue file).

## Applying the take through the bridge

The current P1 bridge (`blender/animus_bridge`) writes bone channels, not
shape keys. To apply this track the bridge needs one added typed
operation, `apply_shape_keys`, that writes `key_blocks["<name>"].value`
FCurves the same way `apply_pose_keys` writes bone channels. The artifact
is already shaped for it: each sample is a `(frame, shape_key, weight)`
triple. That operation is future work under P2; this proof produces and
validates the track that would feed it.
