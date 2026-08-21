// Tests for the deterministic behavior brain (src/agent/behaviorBrain.js).
// The brain is pure: explicit nowMs in, actions out, seeded RNG. These
// tests replay scripted sessions and assert on the action log.
// Run: npm test (node --test tests/agent)

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createBrain,
  DROWSY_AFTER_MS,
  ENGAGED_REPLY_HOLD_MS,
  HIDDEN_GREET_MIN_MS,
  CALM_EXPRESSIONS, ATTENTIVE_EXPRESSIONS, DROWSY_EXPRESSIONS, IDLE_GESTURES,
} from '../../src/agent/behaviorBrain.js';
import { validateFaceGlyph } from '../../src/agent/glyphComposer.js';
import { EXPRESSION_LIBRARY } from '../../src/expressionLibrary.js';
import { GESTURE_NAMES } from '../../src/animationController.js';

const TICK_MS = 100;

// Replay: ticks every TICK_MS from 0 to durationMs; events fire at
// their timestamps. Returns [{t, action}] in order.
function run(brain, durationMs, events = []) {
  const log = [];
  const sorted = [...events].sort((a, b) => a.at - b.at);
  let ei = 0;
  for (let t = 0; t <= durationMs; t += TICK_MS) {
    while (ei < sorted.length && sorted[ei].at <= t) {
      for (const action of brain.notify(sorted[ei].event, sorted[ei].at)) {
        log.push({ t: sorted[ei].at, action });
      }
      ei += 1;
    }
    for (const action of brain.tick(t)) log.push({ t, action });
  }
  return log;
}

const SCRIPT_EVENTS = [
  { at: 5000, event: { type: 'cursor', proximity: 0.7 } },
  { at: 9000, event: { type: 'poke' } },
  { at: 20000, event: { type: 'typing' } },
  { at: 24000, event: { type: 'user_message' } },
  { at: 27000, event: { type: 'reply', emotion: 'happy' } },
  { at: 70000, event: { type: 'cursor', proximity: 0.95 } },
];

test('same seed and same script produce an identical action log', () => {
  const a = run(createBrain({ seed: 42 }), 120000, SCRIPT_EVENTS);
  const b = run(createBrain({ seed: 42 }), 120000, SCRIPT_EVENTS);
  assert.deepEqual(a, b);
  assert.ok(a.length > 0, 'expected some actions');
});

test('different seeds diverge', () => {
  const a = run(createBrain({ seed: 1 }), 60000);
  const b = run(createBrain({ seed: 2 }), 60000);
  assert.notDeepEqual(a, b);
});

test('idle life keeps acting with bounded gaps and no spam', () => {
  const log = run(createBrain({ seed: 7 }), 120000);
  const behaviorTimes = log
    .filter(({ action }) => ['gesture', 'expression', 'glyph', 'gaze'].includes(action.type))
    .map(({ t }) => t);
  assert.ok(behaviorTimes.length >= 8, `expected a lively idle loop, got ${behaviorTimes.length} actions`);
  // Bounded gaps: idle interval tops out at 11s, drowsy at 16s.
  for (let i = 1; i < behaviorTimes.length; i++) {
    const gap = behaviorTimes[i] - behaviorTimes[i - 1];
    assert.ok(gap <= 16100, `gap of ${gap}ms between behaviors`);
  }
});

test('every emitted action uses only known vocabulary', () => {
  const expressionNames = new Set(EXPRESSION_LIBRARY.map(e => e.name));
  const gestureNames = new Set([...GESTURE_NAMES, 'idle']);
  const seeds = [1, 2, 3, 99, 1234];
  for (const seed of seeds) {
    const log = run(createBrain({ seed }), 300000, SCRIPT_EVENTS);
    for (const { action } of log) {
      if (action.type === 'expression') {
        assert.ok(expressionNames.has(action.name), `unknown expression ${action.name}`);
      } else if (action.type === 'gesture') {
        assert.ok(gestureNames.has(action.name), `unknown gesture ${action.name}`);
      } else if (action.type === 'glyph') {
        assert.equal(validateFaceGlyph(action.spec).ok, true,
          `invalid glyph ${JSON.stringify(action.spec)}`);
      } else if (action.type === 'gaze') {
        assert.ok(Math.abs(action.yawDeg) <= 30 && Math.abs(action.pitchDeg) <= 15,
          `gaze out of range ${action.yawDeg}/${action.pitchDeg}`);
      } else {
        assert.ok(['clearGlyph', 'headJerk'].includes(action.type),
          `unknown action type ${action.type}`);
      }
    }
  }
});

test('expression pools reference real library entries', () => {
  const names = new Set(EXPRESSION_LIBRARY.map(e => e.name));
  for (const name of [...CALM_EXPRESSIONS, ...ATTENTIVE_EXPRESSIONS, ...DROWSY_EXPRESSIONS,
    'mild_contempt', 'polite_disbelief', 'precise_scan', 'wry_deflection',
    'cold_curiosity', 'skeptical_narrow', 'cold_assessment', 'neutral_idle']) {
    assert.ok(names.has(name), `expression ${name} missing from library`);
  }
  for (const g of [...IDLE_GESTURES, 'listen', 'shake_no', 'sigh']) {
    assert.ok(GESTURE_NAMES.includes(g), `gesture ${g} missing from controller`);
  }
});

test('poke escalation: deflect, refuse, then angry glyph with head jerk', () => {
  const brain = createBrain({ seed: 5 });
  brain.tick(0);
  const first = brain.notify({ type: 'poke' }, 1000);
  assert.ok(first.some(a => a.type === 'expression' && a.name === 'wry_deflection'));

  const second = brain.notify({ type: 'poke' }, 1600);
  assert.ok(second.some(a => a.type === 'gesture' && a.name === 'shake_no'));
  assert.ok(second.some(a => a.type === 'expression' && a.name === 'mild_contempt'));

  const third = brain.notify({ type: 'poke' }, 2200);
  const glyph = third.find(a => a.type === 'glyph');
  assert.ok(glyph, 'third poke should produce a glyph');
  assert.equal(glyph.spec.mood, 'angry');
  assert.equal(glyph.spec.brows, 'angry_in');
  assert.ok(third.some(a => a.type === 'headJerk'));
});

test('poke heat cools down again', () => {
  const brain = createBrain({ seed: 5 });
  brain.tick(0);
  brain.notify({ type: 'poke' }, 1000);
  brain.notify({ type: 'poke' }, 1500);
  // 30s later the heat has decayed; a poke reads as fresh again.
  const later = brain.notify({ type: 'poke' }, 31500);
  assert.ok(later.some(a => a.type === 'expression' && a.name === 'wry_deflection'));
});

test('typing reacts once per cooldown window', () => {
  const brain = createBrain({ seed: 9 });
  brain.tick(0);
  const first = brain.notify({ type: 'typing' }, 1000);
  assert.ok(first.some(a => a.type === 'gesture' && a.name === 'listen'));
  const second = brain.notify({ type: 'typing' }, 2000);
  assert.equal(second.filter(a => a.type === 'gesture').length, 0,
    'no second listen inside the cooldown');
});

test('chat exchange suppresses idle behaviors until the hold expires', () => {
  const brain = createBrain({ seed: 11 });
  brain.tick(0);
  brain.notify({ type: 'user_message' }, 2000);
  brain.notify({ type: 'reply', emotion: 'happy' }, 4000);

  // While engaged: nothing but (possibly) scheduled clears.
  const engagedEnd = 4000 + ENGAGED_REPLY_HOLD_MS;
  for (let t = 4100; t < engagedEnd; t += TICK_MS) {
    for (const action of brain.tick(t)) {
      assert.equal(action.type, 'clearGlyph',
        `unexpected ${action.type} while engaged at ${t}`);
    }
  }
  assert.equal(brain.getState().mode, 'engaged');

  // After the hold, idle life resumes within one idle interval.
  let resumed = false;
  for (let t = engagedEnd; t < engagedEnd + 12000; t += TICK_MS) {
    if (brain.tick(t).length > 0) { resumed = true; break; }
  }
  assert.ok(resumed, 'idle life should resume after the engaged hold');
});

test('goes drowsy after inactivity and wakes with a startle', () => {
  const brain = createBrain({ seed: 13 });
  const log = run(brain, DROWSY_AFTER_MS + 20000);
  assert.equal(brain.getState().mode, 'drowsy');
  assert.ok(log.length > 0);

  const wake = brain.notify({ type: 'poke' }, DROWSY_AFTER_MS + 21000);
  const glyph = wake.find(a => a.type === 'glyph');
  assert.ok(glyph, 'waking should startle');
  assert.equal(glyph.spec.eyes, 'wide');
  brain.tick(DROWSY_AFTER_MS + 21100);
  assert.notEqual(brain.getState().mode, 'drowsy');
});

test('greets a returning visitor after a long absence', () => {
  const brain = createBrain({ seed: 17 });
  brain.tick(0);
  brain.notify({ type: 'hidden' }, 5000);
  const back = brain.notify({ type: 'visible' }, 5000 + HIDDEN_GREET_MIN_MS + 1000);
  const glyph = back.find(a => a.type === 'glyph');
  assert.ok(glyph, 'expected a return greeting glyph');
  assert.equal(glyph.spec.text, 'BACK?');

  // A short tab-away does not trigger it.
  const brain2 = createBrain({ seed: 17 });
  brain2.tick(0);
  brain2.notify({ type: 'hidden' }, 5000);
  const quick = brain2.notify({ type: 'visible' }, 8000);
  assert.equal(quick.filter(a => a.type === 'glyph').length, 0);
});

test('glances schedule a drift back to center', () => {
  const log = run(createBrain({ seed: 21 }), 180000);
  const gazes = log.filter(({ action }) => action.type === 'gaze');
  const offCenter = gazes.filter(({ action }) => action.yawDeg !== 0 || action.pitchDeg !== 0);
  assert.ok(offCenter.length >= 1, 'expected at least one glance in 3 minutes');
  for (const { t, action } of offCenter) {
    const returned = gazes.some(g =>
      g.t > t && g.t <= t + action.holdMs + 2 * TICK_MS
      && g.action.yawDeg === 0 && g.action.pitchDeg === 0);
    assert.ok(returned, `glance at ${t} never returned to center`);
  }
});

test('every glyph shown is eventually cleared', () => {
  const log = run(createBrain({ seed: 23 }), 300000, SCRIPT_EVENTS);
  let open = 0;
  for (const { action } of log) {
    if (action.type === 'glyph') open += 1;
    if (action.type === 'clearGlyph') open = Math.max(0, open - 1);
  }
  assert.ok(open <= 1, `expected glyphs to clear, ${open} left open`);
});
