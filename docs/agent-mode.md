# Agent mode: the reactive self-animated K-VRC

Agent mode replaces the old scripted idle randomizers with a
deterministic behavior brain that runs in the browser. The character
self-animates (idle life) and reacts to the visitor (cursor, clicks,
typing, chat, tab visibility). No model call is involved; the brain is
pure decision logic on a seeded PRNG.

## Flags

- On by default. `?agent=off` restores the old scripted behavior.
- `?agentseed=123` pins the seed so a session's behavior replays
  exactly (useful for debugging).
- Debug handle in the console: `window.KVRC_AGENT`
  (`.brain.getState()`, `.state.actionLog`, `.setEnabled(false)`).

## Architecture

- `src/agent/behaviorBrain.js`: the brain. Pure, no DOM, no three.js.
  `tick(nowMs)` and `notify(event, nowMs)` return action lists.
  All randomness comes from `src/agent/rng.js` (mulberry32), so the
  same seed plus the same call sequence gives the same action log.
- `src/agent/agentLoop.js`: browser glue. Translates real input into
  brain events, applies brain actions to the robot and face screen.
- `src/agent/glyphComposer.js`: parametric LED face vocabulary
  (8 eyes, 5 brows, 9 mouths, short LED text), ported from the animus
  repo (`feat/animus-poc`, `src/animus/face/glyphComposer.js`).
  Strictly validated; drawn by `faceScreen.js` with the same geometry
  as the expression faces, so blink and speech overlays keep working.

## Modes

- `idle`: weighted behaviors on a 6-11 s cadence: glances (head turns
  that drift back to center), micro expressions from the cold pool,
  one-shot gestures (shrug, sigh, nod, dismiss), weight shifts (new
  base idle clip), rare glyph moments (bored half-lidded face,
  "HMM" text, curious face).
- `attentive`: cursor near the robot or user typing. Tighter cadence
  (4-7 s), curious and focused expressions. Entering it fires a
  "noticed" reaction; getting very close triggers a personal-space
  wide-eyed glyph.
- `engaged`: a chat exchange is in flight or just landed. The brain
  stays silent so the LLM-driven emotion and expression own the face
  (12 s hold after a reply, like the old override timer).
- `drowsy`: no interaction for 75 s. Slow cadence (11-16 s), dream-mood
  half-lidded glyph, sighs. Any input wakes it with a startle.

## Reactions

- Cursor proximity: computed against the head bone projected to
  screen space, throttled to 5 Hz.
- Clicks on the robot: poke escalation. First poke wry deflection,
  second a head shake plus contempt, third and later an angry glyph
  (angry_in brows, gritted mouth, angry palette) plus a head jerk.
  Heat decays one step per 8 s.
- Typing in the chat box: listen gesture plus scanning expression,
  once per 8 s.
- Chat lifecycle: `chat.js` dispatches `kvrc:user-message`,
  `kvrc:reply`, `kvrc:reply-error` window events. Errors show an
  "ERR" glyph.
- Tab visibility: returning after 30+ s away gets a "BACK?" glyph.

## Tests

`npm test` runs `tests/agent/` under the Node test runner: brain
determinism, idle cadence bounds, vocabulary validity (every emitted
gesture and expression exists in `animationController.js` and
`expressionLibrary.js`, every glyph validates), poke escalation and
cooldown, engaged suppression, drowsy and wake, return greeting,
glance return, and glyph cleanup.

## Stretch goal: model-driven brain

Not built yet, deliberately. The loop accepts any object implementing
`tick(nowMs)` and `notify(event, nowMs)` returning the same action
vocabulary, so a WebLLM-backed brain can be swapped in behind a
`?brain=model` flag later without touching the embodiment side. The
deterministic brain stays the default and the fallback.
