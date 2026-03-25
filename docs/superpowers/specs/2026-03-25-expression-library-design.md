# K-VRC Expression Library — Design Spec

**Date:** 2026-03-25
**Status:** Approved

---

## Overview

Add a 100-entry named expression library to K-VRC's face screen system. Claude picks an expression by name per response. A local labeling tool accumulates (reply → expression) pairs for future classifier training or few-shot prompting.

---

## Goals

- Give K-VRC 100 distinct face screen states covering all 8 screen moods
- Let Claude select expressions by name without sending full weight vectors
- Collect human-labeled training data via a fast 4-option labeling UI
- Keep body motion type driven by the existing discrete emotion system; expression library modulates float amplitude only

---

## Architecture

### Hybrid Layering

Two parallel output channels from Claude per response:

| Channel | Drives | Values |
|---|---|---|
| `emotion` | Body motion type, background glow, emotion dot | `neutral / happy / excited / sad / angry / thinking` |
| `expression` | Face screen weight floats + float amplitude scalar | one of 100 named slugs |

These can diverge. Example: `emotion: thinking` (purple glow, head tilt body motion) + `expression: wry_deflection` (squinted eyes, slight smirk face weights).

### Replacing Raw Weights with Named Slugs

The current system has Claude output raw float fields (`expression_weights`, `motion_energy`, `screen_mood`). This design **replaces** all three with a single `expression` slug. After this change:

- `expression_weights`, `motion_energy`, and `screen_mood` are removed from Claude's system prompt and from the API response
- The expression library entry provides all three values looked up by slug
- `api/chat.js` system prompt is updated to output `"expression": "<slug>"`

---

## Components

### 1. Expression Library (`src/expressionLibrary.js`)

Plain exported ES module const — no server dependency.

Each entry:
```js
{
  name: "wry_deflection",       // unique slug, lowercase_underscore
  mood: "cold",                 // one of: cold, warm, glitch, static, data, boot, angry, dream
  weights: {
    brow_raise:       0.1,      // 0–1 float
    brow_furrow:      0.6,
    eye_squint:       0.5,
    mouth_open:       0.0,
    smile_width:      0.2,
    glitch_intensity: 0.0,
  },
  motion_energy: 0.3,           // 0–1; scales robot float amplitude (see §5)
  description: "Dry, dismissive. K-VRC isn't impressed.",
}
```

100 entries total, ~10–15 per mood, covering K-VRC's full personality range.

**The 8 valid mood keys:** `cold, warm, glitch, static, data, boot, angry, dream`

**Reserved mandatory entries** (must always be present):
- `"neutral_idle"` — mood: `cold`, all weights 0 except brow_raise: 0.1, motion_energy: 0.2. Universal fallback.
- `"thinking_default"` — mood: `cold`, brow_furrow: 0.4, eye_squint: 0.3, all others 0, motion_energy: 0.3. Applied during thinking state.

### 2. API Expression Menu (`api/expressionMenu.js`)

ES module. Exports an array of `{ slug, description }` objects, plus a pre-rendered menu text string:

```js
// api/expressionMenu.js
export const EXPRESSIONS = [
  { slug: 'neutral_idle',      description: 'Resting face. Default when nothing else applies.' },
  { slug: 'thinking_default',  description: 'Processing. Focused inward.' },
  { slug: 'wry_deflection',    description: "Dry, dismissive. K-VRC isn't impressed." },
  // ... 97 more
];

export const EXPRESSION_SLUGS = EXPRESSIONS.map(e => e.slug);

export const EXPRESSION_MENU_TEXT =
  'Available expressions (use exact slug):\n' +
  EXPRESSIONS.map(e => `${e.slug} — ${e.description}`).join('\n');
```

This file is the server-side counterpart to `src/expressionLibrary.js`. It contains slugs and descriptions only — no weight data. It must be kept in sync with the library when entries are added or renamed. **Drift risk:** if the two files diverge, valid slugs from `expressionLibrary.js` will be rejected by `api/chat.js` validation and silently fall back to `neutral_idle`. No automated check enforces sync — the accepted mitigation is manual inspection: both files are always edited together in the same commit when the expression set changes.

### 3. faceScreen.js — `setExpression(name)`

New exported function. Stores 6 weight floats as module-level state (`_expressionWeights`). The existing draw functions are extended to use these weights when non-null, overriding the discrete `EXPRESSIONS[currentEmotion]` lookup. When `_expressionWeights` is null, rendering falls back to the emotion-based path unchanged.

`setEmotion()` is modified to clear `_expressionWeights` unconditionally. The early-exit guard is **retained** but moved after the clear. The complete resulting function body:

```js
export function setEmotion(emotion) {
  _expressionWeights = null;                           // always clear first
  if (emotion === currentEmotion && !booting) return;  // guard retained, moved to line 2
  currentEmotion = emotion;
  morphP = 0;
}
```

This ensures weights are cleared even when the emotion doesn't change (e.g., `thinking` → `thinking`), preventing `thinking_default` weights from persisting into the reply phase.

```js
let _expressionWeights = null; // null = use emotion draw path

export function setExpression(name) {
  const entry = EXPRESSION_LIBRARY.find(e => e.name === name)
    ?? EXPRESSION_LIBRARY.find(e => e.name === 'neutral_idle');
  if (!entry) {
    _expressionWeights = null;
    return 0.5; // neutral motion_energy on hard fallback
  }
  _expressionWeights = { ...entry.weights };
  return entry.motion_energy;
}
```

Return contract: always returns a number. Returns `0.5` on the hard fallback path (absent `neutral_idle`).

### 4. api/chat.js — System Prompt Change

**System prompt:** removes `expression_weights`, `motion_energy`, `screen_mood` fields; adds `expression` field; appends `EXPRESSION_MENU_TEXT` from `api/expressionMenu.js`.

**Injected block format** (appended to system prompt):
```
Available expressions (use exact slug):
neutral_idle — Resting face. Default when nothing else applies.
thinking_default — Processing. Focused inward.
wry_deflection — Dry, dismissive. K-VRC isn't impressed.
[... 97 more lines ...]
```

**Validation:** `expression` is checked against `EXPRESSION_SLUGS`. If missing or invalid, defaults to `"neutral_idle"`.

**FALLBACK constant:** Updated to include `expression: 'neutral_idle'` so the error path also produces a valid expression field:
```js
const FALLBACK = { reply: "...", emotion: 'neutral', expression: 'neutral_idle' };
```

**Stream A logger:** Updated to log `expression` slug instead of the removed `expression_weights`, `motion_energy`, and `screen_mood` fields. The logged shape changes from `{ expression_weights: {...}, motion_energy: N, screen_mood: '...' }` to `{ expression: '...' }`.

Claude's output schema:
```json
{
  "reply": "...",
  "emotion": "neutral",
  "expression": "wry_deflection",
  "gesture": "dismiss",
  "sidenote_topic": null
}
```

### 5. src/robot.js

Imports `setExpression as faceSetExpression` from `faceScreen.js` (aliased to avoid name collision with the robot API method). Adds a new `setEmotionName(name)` method to the robot API that stores `_currentEmotionName`:

Both `_currentEmotionName` and `currentEmotionCfg` are **module-level `let` variables** in `robot.js`, shared by closure with all methods on the returned API object. They are not local to `initRobot`. `setExpression` is an arrow function property to match the style of existing API methods:

```js
// Module-level (outer scope of robot.js module):
let _currentEmotionName = 'neutral';
// currentEmotionCfg already exists at module level

// Added to the returned API object inside initRobot():
setEmotionName: name => { _currentEmotionName = name; },

setExpression: name => {
  const motionEnergy = faceSetExpression(name);
  if (currentEmotionCfg) {
    const baseAmp = EMOTION_MAP[_currentEmotionName]?.floatAmp ?? 0.03;
    // Shallow copy — never mutate EMOTION_MAP or the original cfg reference
    currentEmotionCfg = { ...currentEmotionCfg, floatAmp: baseAmp * motionEnergy };
  }
},
```

Each `setExpression` call is a full reset-then-scale: reads canonical `EMOTION_MAP` base value, writes a new copy of `currentEmotionCfg`. Never accumulates. `EMOTION_MAP` is never mutated.

### 6. src/chat.js

**`applyEmotionFull` is updated to:**
1. Call `robotRef?.setEmotionName(emotion)` **outside** the `if (sceneRefs)` block (not inside it alongside `setEmotionCfg`) — `setEmotionName` only writes a module-level variable and has no dependency on `sceneRefs`; placing it inside the guard would leave `_currentEmotionName` stale during boot before `sceneRefs` is set. Place it similarly to `triggerHeadJerk` (already outside the guard). `setEmotionCfg` is **retained and unchanged** inside the `if (sceneRefs)` block.
2. When `emotion === 'thinking'`, call `robotRef?.setExpression('thinking_default')` after the body motion calls

**Asymmetry is intentional:** the thinking expression is set inside `applyEmotionFull` because it fires before any reply data exists. The reply expression is set in `sendMessage` (outside `applyEmotionFull`) after `data.expression` arrives from the API. These are two different call sites driven by different lifecycle moments.

**On reply in `sendMessage`:**
```js
applyEmotionFull(emotion);
robotRef?.setExpression(data.expression ?? 'neutral_idle');
```

`motion_energy` is no longer read from `data`.

### 7. Labeling Tool (`tools/labeler.html`)

Local browser page. No server needed.

**`tools/samples.json` schema** — array of objects:
```json
[
  { "reply": "That's not how any of this works.", "hint_mood": "cold" },
  ...
]
```
- `reply`: string — the K-VRC response text
- `hint_mood`: string — one of the 8 expression moods (`cold, warm, glitch, static, data, boot, angry, dream`), not the 6 discrete emotions

**Labeling flow:**
1. Load `tools/samples.json`
2. Display one reply at a time
3. Show 4 face options (animated canvas + name + description), deduplicated — no slug appears twice:
   - 1 from same mood as `hint_mood`
   - 1 from any other mood
   - 2 fully random (re-sampled if they duplicate already-picked slugs)
4. User clicks one → label recorded, auto-advance
5. Accumulated `{ reply, expression_name }` pairs stored in localStorage, exportable as `labels.json`

**Session target:** ~20–30 replies per sitting (~10–15 min).

---

## Data Flow

```
Claude response
  └─ expression: "wry_deflection"
       └─ sendMessage() → applyEmotionFull(emotion) → robotRef.setEmotionName(emotion)
       └─ sendMessage() → robotRef.setExpression("wry_deflection")
            └─ src/robot.js → faceSetExpression("wry_deflection")  [aliased import]
                 └─ faceScreen.js → library lookup → _expressionWeights → renders
                 └─ returns motion_energy → robot.js computes floatAmp copy
```

```
Labeling tool
  └─ tools/samples.json (reply + hint_mood)
       └─ 4-option UI → human picks
            └─ labels.json (reply → expression_name)
                 └─ downstream: few-shot prompting OR classifier training
```

---

## Downstream Path

**Option A (immediate):** Labels used as few-shot examples in Claude's system prompt.

**Option B (future):** When ~500+ labels accumulated, train lightweight text classifier. LLM selection becomes fallback when classifier confidence < threshold.

Both options remain open.

---

## Out of Scope

- Real-time blending between expressions (snap only, no interpolation in v1)
- Per-word expression changes during TTS playback
- Multi-expression sequences
- Head 3/4 training (separate phase)

---

## Files Changed

| File | Change |
|---|---|
| `src/expressionLibrary.js` | New — 100-entry const with mandatory `neutral_idle` and `thinking_default` |
| `src/faceScreen.js` | Add `setExpression(name)` export; `_expressionWeights` module state; extend draw path; `setEmotion()` clears weights |
| `src/chat.js` | Add `robotRef.setEmotionName(emotion)` in `applyEmotionFull`; call `setExpression('thinking_default')` when thinking; call `setExpression(data.expression)` on reply; remove `motion_energy` read |
| `src/robot.js` | Import `setExpression as faceSetExpression`; add `setEmotionName` and `setExpression` to robot API; track `_currentEmotionName`; apply `motion_energy` via cfg shallow copy |
| `api/chat.js` | Update system prompt; inject `EXPRESSION_MENU_TEXT`; validate via `EXPRESSION_SLUGS`; update `FALLBACK`; update Stream A logger |
| `api/expressionMenu.js` | New — ES module; `EXPRESSIONS` array of `{slug, description}`; exports `EXPRESSION_SLUGS` and `EXPRESSION_MENU_TEXT` |
| `tools/labeler.html` | New — local labeling UI |
| `tools/samples.json` | New — ~200 entries with `reply` and `hint_mood` (8 expression moods) |
