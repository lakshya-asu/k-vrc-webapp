# Project Animus concept reel (Remotion)

Standing video pipeline for Animus: assembles rendered K-VRC scene
takes into the concept reel as a Remotion composition, 1920x1080 at
30 fps. Structure: cold open (the visor face booting, drawn live by
the same face renderer the takes use), a "PROJECT ANIMUS" LED title
card, the ten scenes with short crossfades and lower-third captions,
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

`npx remotion studio` opens the interactive preview.

Remotion downloads its own headless Chrome shell on first render; that
is expected. Rendering does not need the GPU.
