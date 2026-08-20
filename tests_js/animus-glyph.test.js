// Units for the generative glyph composer (face v2): the strict
// validator, the timeline's glyph beats, the contract's face_glyph
// channel, and byte-determinism of the drawn v2 frames.

import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire } from 'node:module';

import { validateActorPlan } from '../src/animus/contract.js';
import { EXPRESSION_LIBRARY } from '../src/animus/face/expressionLibrary.js';
import { drawFaceFrame, W, H } from '../src/animus/face/faceScreenDraw.js';
import { buildFaceTimeline, mulberry32 } from '../src/animus/face/faceTimeline.js';
import {
  GLYPH_BROWS,
  GLYPH_EYES,
  GLYPH_MOUTHS,
  GLYPH_TEXT_MAX,
  validateFaceGlyph,
} from '../src/animus/face/glyphComposer.js';

const require = createRequire(import.meta.url);

test('every vocabulary entry validates as a composed face', () => {
  for (const eyes of GLYPH_EYES) {
    for (const brows of GLYPH_BROWS) {
      for (const mouth of GLYPH_MOUTHS) {
        const checked = validateFaceGlyph({ eyes, brows, mouth });
        assert.ok(checked.ok, checked.errors.join('; '));
        assert.equal(checked.value.intensity, 1);
      }
    }
  }
});

test('text mode normalizes, caps length, and rejects a dirty charset', () => {
  const ok = validateFaceGlyph({ text: ' wtf ', mood: 'angry' });
  assert.ok(ok.ok);
  assert.equal(ok.value.text, 'WTF');
  assert.equal(ok.value.mood, 'angry');

  assert.ok(!validateFaceGlyph({ text: 'TOOLONGX' }).ok, 'over the cap');
  assert.equal(GLYPH_TEXT_MAX, 6);
  assert.ok(!validateFaceGlyph({ text: '<scré>' }).ok, 'non-LED charset');
  assert.ok(!validateFaceGlyph({ text: '' }).ok, 'empty');
  assert.ok(!validateFaceGlyph({ text: 'OK', eyes: 'round' }).ok, 'text excludes glyphs');
});

test('the validator refuses unknown keys, moods, and bad intensity', () => {
  assert.ok(!validateFaceGlyph({ eyes: 'round', python: 'import os' }).ok);
  assert.ok(!validateFaceGlyph({ eyes: 'laser' }).ok);
  assert.ok(!validateFaceGlyph({ eyes: 'round', mood: 'salsa' }).ok);
  assert.ok(!validateFaceGlyph({ eyes: 'round', intensity: 1.5 }).ok);
  assert.ok(!validateFaceGlyph(null).ok);
});

test('glyph beats pop into the timeline and inherit the held mood', () => {
  const frames = buildFaceTimeline(
    {
      fps: 24,
      frame_end: 60,
      face_beats: [
        { expression: 'warm_smile', intensity: 1, at_ms: 0, duration_ms: 800 },
        { glyph: { eyes: 'half_lidded', mouth: 'o_small' }, at_ms: 1000, duration_ms: 1000 },
      ],
      viseme_samples: [],
      seed: 3,
    },
    EXPRESSION_LIBRARY,
  );
  assert.equal(frames[10].glyph, undefined);
  const glyphFrame = frames[25]; // 1000 ms -> frame 25: pops, no ease
  assert.equal(glyphFrame.glyph.eyes, 'half_lidded');
  assert.equal(glyphFrame.mood, 'warm', 'no explicit mood: inherits warm');

  const explicit = buildFaceTimeline(
    {
      fps: 24,
      frame_end: 30,
      face_beats: [
        { glyph: { text: 'SCAN', mood: 'data' }, at_ms: 0, duration_ms: 1000 },
      ],
      viseme_samples: [],
      seed: 3,
    },
    EXPRESSION_LIBRARY,
  );
  assert.equal(explicit[5].mood, 'data');
  assert.equal(explicit[5].glyph.text, 'SCAN');
});

test('an invalid glyph beat fails the timeline loudly', () => {
  assert.throws(() => buildFaceTimeline(
    {
      fps: 24,
      frame_end: 30,
      face_beats: [{ glyph: { eyes: 'nope' }, at_ms: 0, duration_ms: 500 }],
      viseme_samples: [],
      seed: 0,
    },
    EXPRESSION_LIBRARY,
  ));
});

test('the actor contract accepts face_glyph as a beat channel', () => {
  const plan = {
    schema_version: '0.1',
    summary: 'glyph face check',
    beats: [
      {
        id: 'scan-1',
        at_ms: 0,
        duration_ms: 1200,
        face_glyph: { text: 'SCAN', mood: 'data' },
      },
      {
        id: 'grit-1',
        at_ms: 1200,
        duration_ms: 1000,
        face_glyph: { eyes: 'bar', brows: 'angry_in', mouth: 'gritted', mood: 'angry' },
      },
    ],
  };
  const checked = validateActorPlan(plan, { actorId: 'kvrc', controlLevel: 'suggest' });
  assert.ok(checked.ok, checked.errors.join('; '));

  const both = validateActorPlan(
    {
      ...plan,
      beats: [
        {
          id: 'x',
          at_ms: 0,
          duration_ms: 500,
          face: { expression: 'neutral_idle' },
          face_glyph: { eyes: 'round' },
        },
      ],
    },
    { actorId: 'kvrc', controlLevel: 'suggest' },
  );
  assert.ok(!both.ok, 'face and face_glyph together must be refused');

  const bad = validateActorPlan(
    {
      ...plan,
      beats: [
        { id: 'x', at_ms: 0, duration_ms: 500, face_glyph: { text: 'WAYTOOLONG' } },
      ],
    },
    { actorId: 'kvrc', controlLevel: 'suggest' },
  );
  assert.ok(!bad.ok);
});

test('v2 frames draw deterministically, and fx changes the pixels', () => {
  const { createCanvas } = require('@napi-rs/canvas');
  const job = {
    fps: 24,
    frame_end: 4,
    face_beats: [
      { glyph: { eyes: 'bar', brows: 'angry_in', mouth: 'gritted', mood: 'angry' }, at_ms: 0, duration_ms: 900 },
    ],
    viseme_samples: [{ frame: 2, shape_key: 'mouth_open', weight: 0.5 }],
    seed: 11,
  };
  const timeline = buildFaceTimeline(job, EXPRESSION_LIBRARY);
  const render = (opts) => {
    const canvas = createCanvas(W, H);
    const ctx = canvas.getContext('2d');
    drawFaceFrame(ctx, timeline[1], mulberry32(77), opts);
    return canvas.toBuffer('image/png');
  };
  const a = render();
  const b = render();
  assert.ok(a.equals(b), 'same state and seed must produce identical bytes');
  const flat = render({ fx: false });
  assert.ok(!a.equals(flat), 'the v2 fx pass must actually change the frame');
});
