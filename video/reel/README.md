# Project Animus concept reel (Remotion)

Standing video pipeline for Animus: assembles rendered K-VRC scene
takes into the concept reel as a Remotion composition, 1920x1080 at
30 fps. Structure: cold open (the visor face booting, drawn live by
the same face renderer the takes use), a "PROJECT ANIMUS" LED title
card, the ten scenes with eased video crossfades whose audio runs an
equal-power crossfade over the same frames (no hard audio edges at
any join) and lower-third captions,
and an end card ("performed live by a local model").

The title, boot, and end cards reuse the actual face-canvas code from
`src/animus/face/` (expression library, faceScreenDraw, faceTimeline);
nothing is redrawn by hand. Captions come from the scene batch
manifest.

## Prerequisites

- Node 18+ (`npm install` inside this directory)
- ffmpeg and ffprobe on PATH (loudness normalization + probing)
- A rendered scene batch: `scene-01/scene.mp4` .. `scene-10/scene.mp4`
  plus the batch `manifest.json` (what the actor loop's scene shoots
  produce; see `docs/animus/actor-loop-runbook.md`)

## Render the reel

    npm install
    node tools/build-data.mjs --scenes <batch dir> --manifest <manifest.json>
    npx remotion render ProjectAnimusReel <out.mp4>

`build-data.mjs` loudness-normalizes every scene's audio to -16 LUFS
(video stream copied untouched) into `public/scenes/`, probes exact
durations, and writes `src/reel-data.json` (scene order, durations,
captions). The composition reads that file, so re-running the tool on
a new batch re-times the whole reel automatically.

`build-data.mjs` measures loudness first and applies a LINEAR
(constant-gain) second pass; dynamic single-pass loudnorm rides the
gain and pumps the noise floor up inside the silent gaps around each
line, which defeats the dialogue spacing below.

## Dialogue spacing (the v3 rule: fit video to audio)

Scene takes must be built so no line ever collides with a crossfade
or with a neighboring scene's line. With `CROSS = 16` frames at 30 fps
(533 ms per fade), each scene's speech must obey:

- head: speech starts at least 400 ms after the incoming crossfade
  ends, so at least 933 ms into the scene (use 1000 ms).
- tail: speech ends with at least 800 ms of clean silence before the
  outgoing crossfade begins, so the scene runs at least 1333 ms past
  the end of the wav (use 1400 ms).

Extend the scene's hold beats (gaze, face) to reach the required scene
end; the actor's strips hold their pose, so a longer scene ends on a
held idle, never a frozen mid-gesture frame. Never tempo-compress the
voice to fit a slot: compression is audibly robotic. The scene gets
longer; the audio does not get faster.

`npx remotion studio` opens the interactive preview.

Remotion downloads its own headless Chrome shell on first render; that
is expected. Rendering does not need the GPU.
