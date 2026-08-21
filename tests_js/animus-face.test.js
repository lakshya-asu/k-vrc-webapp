// Units for the visor face timeline (src/animus/face/faceTimeline.js)
// and the copied expression library. The timeline is the deterministic
// half of the face renderer: same job and seed, same states.

import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire } from 'node:module';

import { EXPRESSION_LIBRARY } from '../src/animus/face/expressionLibrary.js';
import {
  MOOD_COLORS,
  VISEME_MOUTH_CLASSES,
  W,
  H,
  drawFaceFrame,
} from '../src/animus/face/faceScreenDraw.js';
import {
  FACE_WEIGHT_KEYS,
  buildFaceTimeline,
  durationToFrames,
  msToFrame,
  mulberry32,
  resolveExpression,
  visemeClassTrack,
} from '../src/animus/face/faceTimeline.js';

const require = createRequire(import.meta.url);

function makeJob(overrides = {}) {
  return {
    fps: 24,
    frame_end: 49,
    face_beats: [
      { expression: 'warm_amused', intensity: 0.5, at_ms: 0, duration_ms: 1800 },
    ],
    viseme_samples: [
      { frame: 1, shape_key: 'mouth_open', weight: 0.0 },
      { frame: 1, shape_key: 'smile_width', weight: 0.1 },
      { frame: 11, shape_key: 'mouth_open', weight: 0.5 },
      { frame: 21, shape_key: 'mouth_open', weight: 0.0 },
      { frame: 40, shape_key: 'mouth_open', weight: 0.0 },
    ],
    seed: 0,
    ...overrides,
  };
}

test('the copied expression library keeps the reserved entries and moods', () => {
  for (const name of ['neutral_idle', 'thinking_default', 'warm_amused']) {
    const entry = EXPRESSION_LIBRARY.find((e) => e.name === name);
    assert.ok(entry, `library must define ${name}`);
    assert.ok(MOOD_COLORS[entry.mood], `mood ${entry.mood} must have a palette`);
    for (const key of FACE_WEIGHT_KEYS) {
      const w = entry.weights[key];
      assert.ok(typeof w === 'number' && w >= 0 && w <= 1, `${name}.${key}`);
    }
  }
  // Every library entry maps onto a defined mood palette.
  for (const entry of EXPRESSION_LIBRARY) {
    assert.ok(MOOD_COLORS[entry.mood], `${entry.name} names mood ${entry.mood}`);
  }
});

test('unknown expressions resolve to neutral_idle, like the webapp face', () => {
  const entry = resolveExpression('definitely_not_a_face', EXPRESSION_LIBRARY);
  assert.equal(entry.name, 'neutral_idle');
});

test('frame math matches the embodiment mapper', () => {
  assert.equal(msToFrame(0, 24), 1);
  assert.equal(msToFrame(1000, 24), 25);
  assert.equal(durationToFrames(1800, 24), 43);
  assert.equal(durationToFrames(1, 24), 2);
});

test('the timeline is deterministic for a seed and differs across seeds', () => {
  const job = makeJob({ frame_end: 240 }); // 10 s: guarantees a blink
  const a = buildFaceTimeline(job, EXPRESSION_LIBRARY);
  const b = buildFaceTimeline(makeJob({ frame_end: 240 }), EXPRESSION_LIBRARY);
  assert.deepEqual(a, b);

  const c = buildFaceTimeline(
    makeJob({ frame_end: 240, seed: 99 }), EXPRESSION_LIBRARY,
  );
  const blinkFrames = (frames) =>
    frames.filter((f) => f.blinkProgress > 0).map((f) => f.frame);
  assert.ok(blinkFrames(a).length > 0, 'a 10 s take must blink');
  assert.notDeepEqual(blinkFrames(a), blinkFrames(c));
});

test('blinks follow the library timer and fully reopen', () => {
  const frames = buildFaceTimeline(
    makeJob({ frame_end: 240, viseme_samples: [] }), EXPRESSION_LIBRARY,
  );
  const first = frames.find((f) => f.blinkProgress > 0);
  assert.ok(first, 'a blink must happen');
  // First blink lands on the 3..7 s window (frame 72..169 at 24 fps).
  assert.ok(first.frame >= 72 && first.frame <= 169, String(first.frame));
  // The lid closes and fully reopens within a fraction of a second.
  const after = frames.filter(
    (f) => f.frame > first.frame && f.frame <= first.frame + 8,
  );
  assert.ok(Math.max(...after.map((f) => f.blinkProgress)) > 0.4);
  assert.equal(after[after.length - 1].blinkProgress, 0);
});

test('a warm_amused beat eases in over the mapper ramp and then holds', () => {
  const frames = buildFaceTimeline(makeJob({ viseme_samples: [] }), EXPRESSION_LIBRARY);
  const neutral = resolveExpression('neutral_idle', EXPRESSION_LIBRARY);
  const warm = resolveExpression('warm_amused', EXPRESSION_LIBRARY);
  // duration 1800 ms -> 43 frames, ramp = min(4, 43 // 3) = 4.
  const f1 = frames[0];
  assert.equal(f1.mood, 'warm');
  assert.equal(f1.weights.smile_width, neutral.weights.smile_width);
  const f5 = frames[4]; // base + ramp: fully eased in
  assert.ok(Math.abs(f5.weights.smile_width - warm.weights.smile_width * 0.5) < 1e-9);
  // Held to the end of the take, webapp setExpression semantics.
  const last = frames[frames.length - 1];
  assert.equal(last.mood, 'warm');
  assert.ok(Math.abs(last.weights.smile_width - warm.weights.smile_width * 0.5) < 1e-9);
});

test('visemes drive mouth_open and the speech amplitude between cues', () => {
  const frames = buildFaceTimeline(makeJob(), EXPRESSION_LIBRARY);
  const f6 = frames[5]; // halfway from frame 1 (0.0) to frame 11 (0.5)
  assert.ok(Math.abs(f6.weights.mouth_open - 0.25) < 1e-9, String(f6.weights.mouth_open));
  // The loudest track includes the frame-1 smile_width 0.1 sample:
  // halfway from 0.1 to 0.5 is 0.3.
  assert.ok(Math.abs(f6.amplitude - 0.3) < 1e-9, String(f6.amplitude));
  const f11 = frames[10];
  assert.equal(f11.weights.mouth_open, 0.5);
  // Past the last viseme sample the voice is silent.
  const f45 = frames[44];
  assert.equal(f45.amplitude, 0);
});

test('a second face beat replaces the held expression', () => {
  const frames = buildFaceTimeline(
    makeJob({
      frame_end: 60,
      viseme_samples: [],
      face_beats: [
        { expression: 'warm_amused', intensity: 1, at_ms: 0, duration_ms: 1000 },
        { expression: 'angry_snap', intensity: 1, at_ms: 1500, duration_ms: 1000 },
      ],
    }),
    EXPRESSION_LIBRARY,
  );
  assert.equal(frames[10].mood, 'warm');
  const base2 = msToFrame(1500, 24); // 37
  assert.equal(frames[base2 - 1].mood, 'angry');
  const angry = resolveExpression('angry_snap', EXPRESSION_LIBRARY);
  const settled = frames[base2 + 5];
  assert.ok(Math.abs(settled.weights.brow_furrow - angry.weights.brow_furrow) < 1e-9);
});

test('force_glitches turns the glitch overlay on for exactly its windows', () => {
  // A 2 s take can never reach the 8-20 s seeded glitch timer, so
  // without the hook no frame glitches.
  const quiet = buildFaceTimeline(makeJob({ frame_end: 48 }), EXPRESSION_LIBRARY);
  assert.ok(quiet.every((f) => !f.glitchActive), 'short takes never glitch on their own');

  // Forced window: 1000 ms to 1250 ms. Frame f covers t=(f-1)/fps, so
  // frames 25..30 sit inside [1.0 s, 1.25 s) at 24 fps.
  const forced = buildFaceTimeline(
    makeJob({ frame_end: 48, force_glitches: [{ at_ms: 1000, duration_ms: 250 }] }),
    EXPRESSION_LIBRARY,
  );
  const active = forced.filter((f) => f.glitchActive).map((f) => f.frame);
  assert.deepEqual(active, [25, 26, 27, 28, 29, 30]);

  // Everything else about the timeline is untouched by the hook.
  const strip = (frames) => frames.map(({ glitchActive, ...rest }) => rest);
  assert.deepEqual(strip(forced), strip(quiet));

  // Malformed windows are refused loudly, never ignored.
  assert.throws(() => buildFaceTimeline(
    makeJob({ force_glitches: [{ at_ms: -1, duration_ms: 100 }] }),
    EXPRESSION_LIBRARY,
  ));
  assert.throws(() => buildFaceTimeline(
    makeJob({ force_glitches: [{ at_ms: 0 }] }),
    EXPRESSION_LIBRARY,
  ));
});

test('mulberry32 streams are reproducible', () => {
  const a = mulberry32(1234);
  const b = mulberry32(1234);
  for (let i = 0; i < 16; i++) assert.equal(a(), b());
  const c = mulberry32(1235);
  assert.notEqual(mulberry32(1234)(), c());
});

// --- viseme-class mouths (Rhubarb A-H) ------------------------------------

const CLASSED_SAMPLES = [
  { frame: 1, shape_key: 'mouth_open', weight: 0.0, viseme: 'X' },
  { frame: 11, shape_key: 'mouth_open', weight: 0.85, viseme: 'D' },
  { frame: 21, shape_key: 'mouth_open', weight: 0.2, viseme: 'B' },
  { frame: 40, shape_key: 'mouth_open', weight: 0.0, viseme: 'X' },
];

test('viseme classes hold from cue start until the next cue, like Rhubarb', () => {
  const at = visemeClassTrack(CLASSED_SAMPLES);
  assert.equal(at(1), 'X');
  assert.equal(at(10), 'X');
  assert.equal(at(11), 'D');
  assert.equal(at(15), 'D');
  assert.equal(at(21), 'B');
  assert.equal(at(39), 'B');
  assert.equal(at(40), 'X');
  assert.equal(at(48), 'X'); // held past the last cue
  // No classes recorded: no track (legacy artifacts keep working).
  assert.equal(visemeClassTrack([{ frame: 1, shape_key: 'mouth_open', weight: 0.5 }])(1), null);
  assert.equal(visemeClassTrack([])(1), null);
});

test('timeline states carry the held viseme class only when one exists', () => {
  const classed = buildFaceTimeline(
    makeJob({ viseme_samples: CLASSED_SAMPLES }), EXPRESSION_LIBRARY,
  );
  assert.equal(classed[0].viseme, 'X');
  assert.equal(classed[14].viseme, 'D');
  assert.equal(classed[24].viseme, 'B');
  // Legacy jobs without classes never grow the field.
  const legacy = buildFaceTimeline(makeJob(), EXPRESSION_LIBRARY);
  assert.ok(legacy.every((state) => !('viseme' in state)));
});

test('each viseme class draws a distinct, deterministic LED mouth', () => {
  const { createCanvas } = require('@napi-rs/canvas');
  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  const neutral = resolveExpression('neutral_idle', EXPRESSION_LIBRARY);
  const baseState = {
    frame: 12,
    t: 11 / 24,
    mood: 'warm',
    weights: Object.fromEntries(FACE_WEIGHT_KEYS.map((key) => [
      key, neutral.weights[key] ?? 0,
    ])),
    amplitude: 0.6,
    blinkProgress: 0,
    glitchActive: false,
  };
  baseState.weights.mouth_open = 0.6;

  const render = (viseme) => {
    const state = viseme == null ? { ...baseState } : { ...baseState, viseme };
    drawFaceFrame(ctx, state, mulberry32(7));
    return canvas.toBuffer('image/png');
  };

  // Every classed mouth differs from the weight-driven fallback, and
  // the named classes differ from each other.
  const fallback = render(null);
  const rendered = new Map();
  for (const viseme of VISEME_MOUTH_CLASSES) {
    const png = render(viseme);
    assert.notDeepEqual(png, fallback, `viseme ${viseme} must differ from the fallback mouth`);
    for (const [other, otherPng] of rendered.entries()) {
      assert.notDeepEqual(png, otherPng, `viseme ${viseme} must differ from ${other}`);
    }
    rendered.set(viseme, png);
  }
  // X and unknown classes keep the fallback mouth exactly.
  assert.deepEqual(render('X'), fallback);
  assert.deepEqual(render('Q'), fallback);
  // Determinism: the same class renders byte-identically.
  assert.deepEqual(render('D'), rendered.get('D'));
});
