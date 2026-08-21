// Tests for the glyph composer validation (src/agent/glyphComposer.js).
// Run: npm test (node --test tests/agent)

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  validateFaceGlyph, GLYPH_EYES, GLYPH_BROWS, GLYPH_MOUTHS, GLYPH_MOODS,
} from '../../src/agent/glyphComposer.js';

test('accepts a composed face and fills defaults', () => {
  const res = validateFaceGlyph({ eyes: 'round' });
  assert.equal(res.ok, true);
  assert.deepEqual(res.value, { eyes: 'round', brows: 'none', mouth: 'none', intensity: 1 });
});

test('accepts every vocabulary combination shape', () => {
  for (const eyes of GLYPH_EYES) {
    for (const brows of GLYPH_BROWS) {
      const res = validateFaceGlyph({ eyes, brows, mouth: GLYPH_MOUTHS[0] });
      assert.equal(res.ok, true, `eyes=${eyes} brows=${brows}`);
    }
  }
});

test('accepts text mode, uppercases and trims', () => {
  const res = validateFaceGlyph({ text: ' back? ' });
  assert.equal(res.ok, true);
  assert.equal(res.value.text, 'BACK?');
});

test('rejects text longer than 6 characters', () => {
  assert.equal(validateFaceGlyph({ text: 'TOOLONGX' }).ok, false);
});

test('rejects text outside the LED charset', () => {
  assert.equal(validateFaceGlyph({ text: 'a~b' }).ok, false);
});

test('rejects eyes together with text', () => {
  assert.equal(validateFaceGlyph({ text: 'HI', eyes: 'round' }).ok, false);
});

test('rejects unknown keys and unknown vocabulary', () => {
  assert.equal(validateFaceGlyph({ eyes: 'round', sneaky: 1 }).ok, false);
  assert.equal(validateFaceGlyph({ eyes: 'laser' }).ok, false);
  assert.equal(validateFaceGlyph({ eyes: 'round', mood: 'plaid' }).ok, false);
});

test('rejects non-object specs and bad intensity', () => {
  assert.equal(validateFaceGlyph(null).ok, false);
  assert.equal(validateFaceGlyph([]).ok, false);
  assert.equal(validateFaceGlyph({ eyes: 'round', intensity: 2 }).ok, false);
  assert.equal(validateFaceGlyph({ eyes: 'round', intensity: NaN }).ok, false);
});

test('mood vocabulary matches the face screen palettes', () => {
  assert.deepEqual(
    [...GLYPH_MOODS].sort(),
    ['angry', 'boot', 'cold', 'data', 'dream', 'glitch', 'static', 'warm'].sort(),
  );
});
