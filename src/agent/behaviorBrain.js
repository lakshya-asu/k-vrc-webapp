// src/agent/behaviorBrain.js
// The deterministic in-browser behavior brain for agent mode.
//
// Pure decision logic: no DOM, no three.js, no timers of its own.
// The caller (agentLoop.js in the browser, or a test) advances it
// with tick(nowMs) and feeds it events with notify(event, nowMs);
// both return a list of actions for the embodiment to execute.
// All randomness comes from a seeded PRNG, so the same seed plus the
// same call sequence always yields the same action log.
//
// Action vocabulary (what the loop knows how to apply):
//   { type: 'gesture',    name }                      -> robot.playGesture
//   { type: 'expression', name }                      -> robot.setExpression
//   { type: 'glyph',      spec }                      -> faceScreen.setFaceGlyph
//   { type: 'clearGlyph' }                            -> faceScreen.clearFaceGlyph
//   { type: 'gaze',       yawDeg, pitchDeg, holdMs }  -> robot.setGaze
//   { type: 'headJerk' }                              -> robot.triggerHeadJerk
//
// Modes:
//   idle      - nobody around; weighted idle-life behaviors on a
//               seeded cadence (glances, micro expressions, gestures,
//               sighs, rare glyph moments).
//   attentive - cursor near the robot or user typing; tighter cadence,
//               curious or focused expressions.
//   engaged   - a chat exchange is in flight or just finished; the
//               brain stays out of the way so the LLM-driven emotion
//               and expression own the face.
//   drowsy    - no interaction for a while; slow, low-energy behaviors
//               until something wakes it with a startle.

import { mulberry32, rangeFrom, weightedPick } from './rng.js';
import { validateFaceGlyph } from './glyphComposer.js';

// Timing constants (ms). Exported for tests.
export const DROWSY_AFTER_MS = 75000;
export const ENGAGED_REPLY_HOLD_MS = 12000;
export const ENGAGED_MESSAGE_TIMEOUT_MS = 30000;
export const ATTENTIVE_HOLD_MS = 6000;
export const POKE_DECAY_MS = 8000;
export const NOTICE_COOLDOWN_MS = 10000;
export const PERSONAL_SPACE_COOLDOWN_MS = 20000;
export const TYPING_REACTION_COOLDOWN_MS = 8000;
export const HIDDEN_GREET_MIN_MS = 30000;

export const IDLE_INTERVAL_MS = { idle: [6000, 11000], attentive: [4000, 7000], drowsy: [11000, 16000] };

// Expression pools. Names must exist in src/expressionLibrary.js.
export const CALM_EXPRESSIONS = [
  'neutral_idle', 'deadpan', 'cold_assessment', 'calculating',
  'cold_curiosity', 'wry_deflection', 'skeptical_narrow', 'precise_scan',
];
export const ATTENTIVE_EXPRESSIONS = [
  'cold_curiosity', 'precise_scan', 'focused_lock', 'calculating',
];
export const DROWSY_EXPRESSIONS = [
  'resigned_acceptance', 'logical_void', 'deadpan',
];

// One-shot gestures only (looping ones would never hand control back).
// Names must be keys of GESTURE_CLIPS in src/animationController.js.
export const IDLE_GESTURES = ['shrug', 'sigh', 'nod', 'dismiss'];

// Idle glyph moments: [spec, holdMs].
const IDLE_GLYPHS = [
  [{ eyes: 'half_lidded', mouth: 'flat' }, 1800],
  [{ text: 'HMM' }, 1400],
  [{ eyes: 'round', brows: 'raised', mouth: 'o_small' }, 1500],
];
const DROWSY_GLYPH = [{ eyes: 'half_lidded', mouth: 'none', mood: 'dream', intensity: 0.7 }, 2500];

const STARTLE_GLYPH = [{ eyes: 'wide', brows: 'raised', mouth: 'o_small' }, 500];
const PERSONAL_SPACE_GLYPH = [{ eyes: 'wide', brows: 'raised', mouth: 'o_small' }, 900];
const ANGRY_GLYPH = [{ eyes: 'bar', brows: 'angry_in', mouth: 'gritted', mood: 'angry' }, 1800];
const ERROR_GLYPH = [{ text: 'ERR' }, 1200];
const RETURN_GLYPH = [{ text: 'BACK?' }, 1600];

function assertGlyph(spec) {
  const res = validateFaceGlyph(spec);
  if (!res.ok) throw new Error(`brain produced invalid glyph: ${res.errors.join('; ')}`);
  return res.value;
}

export function createBrain({ seed = 1 } = {}) {
  const rng = mulberry32(seed);

  const state = {
    mode: 'idle',
    lastInteractionMs: 0,
    engagedUntilMs: -1,
    attentiveUntilMs: -1,
    nextIdleActMs: -1,        // scheduled on first tick
    cursorNear: false,
    lastNoticeMs: -Infinity,
    lastPersonalSpaceMs: -Infinity,
    lastTypingReactionMs: -Infinity,
    pokeHeat: 0,
    lastPokeMs: -Infinity,
    hiddenAtMs: -1,
    glyphActive: false,
    initialized: false,
  };

  // Scheduled follow-up actions: { dueMs, action }.
  let pending = [];

  function schedule(dueMs, action) {
    pending.push({ dueMs, action });
    pending.sort((a, b) => a.dueMs - b.dueMs);
  }

  function flushPending(nowMs, out) {
    while (pending.length && pending[0].dueMs <= nowMs) {
      const { action } = pending.shift();
      if (action.type === 'clearGlyph') state.glyphActive = false;
      out.push(action);
    }
  }

  function emitGlyph(out, nowMs, [spec, holdMs], followUp = null) {
    out.push({ type: 'glyph', spec: assertGlyph(spec) });
    state.glyphActive = true;
    schedule(nowMs + holdMs, { type: 'clearGlyph' });
    if (followUp) schedule(nowMs + holdMs, followUp);
  }

  function isEngaged(nowMs) {
    return nowMs < state.engagedUntilMs;
  }

  function currentMode(nowMs) {
    if (isEngaged(nowMs)) return 'engaged';
    if (nowMs < state.attentiveUntilMs) return 'attentive';
    if (nowMs - state.lastInteractionMs >= DROWSY_AFTER_MS) return 'drowsy';
    return 'idle';
  }

  function scheduleNextIdleAct(nowMs, mode) {
    const [lo, hi] = IDLE_INTERVAL_MS[mode] ?? IDLE_INTERVAL_MS.idle;
    state.nextIdleActMs = nowMs + rangeFrom(rng, lo, hi);
  }

  function pickExpression(pool) {
    return pool[Math.floor(rng() * pool.length)];
  }

  function idleBehavior(nowMs, mode, out) {
    if (mode === 'drowsy') {
      const kind = weightedPick(rng, [
        { value: 'drowsy_glyph', weight: 0.35 },
        { value: 'expression', weight: 0.35 },
        { value: 'sigh', weight: 0.2 },
        { value: 'glance', weight: 0.1 },
      ]);
      if (kind === 'drowsy_glyph' && !state.glyphActive) {
        emitGlyph(out, nowMs, DROWSY_GLYPH,
          { type: 'expression', name: pickExpression(DROWSY_EXPRESSIONS) });
      } else if (kind === 'expression') {
        out.push({ type: 'expression', name: pickExpression(DROWSY_EXPRESSIONS) });
      } else if (kind === 'sigh') {
        out.push({ type: 'gesture', name: 'sigh' });
      } else {
        emitGlance(nowMs, out, 0.5);
      }
      return;
    }

    const pools = mode === 'attentive'
      ? [
          { value: 'glance', weight: 0.2 },
          { value: 'expression', weight: 0.5 },
          { value: 'gesture', weight: 0.2 },
          { value: 'shift', weight: 0.1 },
        ]
      : [
          { value: 'glance', weight: 0.3 },
          { value: 'expression', weight: 0.28 },
          { value: 'gesture', weight: 0.18 },
          { value: 'glyph', weight: 0.09 },
          { value: 'shift', weight: 0.15 },
        ];
    const kind = weightedPick(rng, pools);

    if (kind === 'glance') {
      emitGlance(nowMs, out, 1);
    } else if (kind === 'expression') {
      const pool = mode === 'attentive' ? ATTENTIVE_EXPRESSIONS : CALM_EXPRESSIONS;
      out.push({ type: 'expression', name: pickExpression(pool) });
    } else if (kind === 'gesture') {
      const name = IDLE_GESTURES[Math.floor(rng() * IDLE_GESTURES.length)];
      out.push({ type: 'gesture', name });
    } else if (kind === 'glyph' && !state.glyphActive) {
      const pick = IDLE_GLYPHS[Math.floor(rng() * IDLE_GLYPHS.length)];
      emitGlyph(out, nowMs, pick,
        { type: 'expression', name: pickExpression(CALM_EXPRESSIONS) });
    } else {
      // Weight shift: re-roll the base idle clip.
      out.push({ type: 'gesture', name: 'idle' });
    }
  }

  function emitGlance(nowMs, out, scale) {
    const yawDeg = rangeFrom(rng, -24, 24) * scale;
    const pitchDeg = rangeFrom(rng, -10, 6) * scale;
    const holdMs = Math.round(rangeFrom(rng, 900, 2200));
    out.push({ type: 'gaze', yawDeg, pitchDeg, holdMs });
    // Drift back to center after the hold.
    schedule(nowMs + holdMs, { type: 'gaze', yawDeg: 0, pitchDeg: 0, holdMs: 0 });
  }

  function interaction(nowMs) {
    state.lastInteractionMs = nowMs;
  }

  function wakeIfDrowsy(nowMs, out) {
    if (currentMode(nowMs) !== 'drowsy') return false;
    // Startle: snap awake.
    emitGlyph(out, nowMs, STARTLE_GLYPH, { type: 'expression', name: 'precise_scan' });
    scheduleNextIdleAct(nowMs, 'idle');
    return true;
  }

  function decayPokeHeat(nowMs) {
    if (state.pokeHeat === 0) return;
    const steps = Math.floor((nowMs - state.lastPokeMs) / POKE_DECAY_MS);
    if (steps > 0) {
      state.pokeHeat = Math.max(0, state.pokeHeat - steps);
      state.lastPokeMs += steps * POKE_DECAY_MS;
    }
  }

  return {
    tick(nowMs) {
      const out = [];
      if (!state.initialized) {
        state.initialized = true;
        state.lastInteractionMs = nowMs;
        scheduleNextIdleAct(nowMs, 'idle');
      }
      flushPending(nowMs, out);
      decayPokeHeat(nowMs);

      const mode = currentMode(nowMs);
      state.mode = mode;

      if (mode !== 'engaged' && state.nextIdleActMs >= 0 && nowMs >= state.nextIdleActMs) {
        idleBehavior(nowMs, mode, out);
        scheduleNextIdleAct(nowMs, mode);
      } else if (mode === 'engaged' && nowMs >= state.nextIdleActMs) {
        // Keep the schedule moving while engaged so the brain does not
        // fire a burst the moment engagement ends.
        scheduleNextIdleAct(nowMs, 'idle');
      }
      return out;
    },

    notify(event, nowMs) {
      const out = [];
      if (!state.initialized) this.tick(nowMs);
      flushPending(nowMs, out);
      decayPokeHeat(nowMs);

      switch (event.type) {
        case 'cursor': {
          const near = event.proximity >= 0.6;
          const wasNear = state.cursorNear;
          state.cursorNear = near;
          if (near) {
            const woke = wakeIfDrowsy(nowMs, out);
            interaction(nowMs);
            state.attentiveUntilMs = nowMs + ATTENTIVE_HOLD_MS;
            if (!wasNear && !woke && !isEngaged(nowMs)
                && nowMs - state.lastNoticeMs >= NOTICE_COOLDOWN_MS) {
              state.lastNoticeMs = nowMs;
              out.push({ type: 'expression', name: 'cold_curiosity' });
            }
            if (event.proximity >= 0.9 && !isEngaged(nowMs) && !state.glyphActive
                && nowMs - state.lastPersonalSpaceMs >= PERSONAL_SPACE_COOLDOWN_MS) {
              state.lastPersonalSpaceMs = nowMs;
              emitGlyph(out, nowMs, PERSONAL_SPACE_GLYPH,
                { type: 'expression', name: 'skeptical_narrow' });
            }
          }
          break;
        }

        case 'poke': {
          const woke = wakeIfDrowsy(nowMs, out);
          interaction(nowMs);
          state.attentiveUntilMs = nowMs + ATTENTIVE_HOLD_MS;
          if (isEngaged(nowMs)) break;
          state.pokeHeat += 1;
          state.lastPokeMs = nowMs;
          if (woke) break; // the startle IS the reaction
          if (state.pokeHeat <= 1) {
            out.push({ type: 'expression', name: 'wry_deflection' });
          } else if (state.pokeHeat === 2) {
            out.push({ type: 'gesture', name: 'shake_no' });
            out.push({ type: 'expression', name: 'mild_contempt' });
          } else {
            if (!state.glyphActive) {
              emitGlyph(out, nowMs, ANGRY_GLYPH,
                { type: 'expression', name: 'cold_assessment' });
            }
            out.push({ type: 'headJerk' });
          }
          break;
        }

        case 'typing': {
          wakeIfDrowsy(nowMs, out);
          interaction(nowMs);
          state.attentiveUntilMs = nowMs + ATTENTIVE_HOLD_MS;
          if (isEngaged(nowMs)) break;
          if (nowMs - state.lastTypingReactionMs >= TYPING_REACTION_COOLDOWN_MS) {
            state.lastTypingReactionMs = nowMs;
            out.push({ type: 'gesture', name: 'listen' });
            out.push({ type: 'expression', name: 'precise_scan' });
          }
          break;
        }

        case 'user_message': {
          interaction(nowMs);
          state.engagedUntilMs = nowMs + ENGAGED_MESSAGE_TIMEOUT_MS;
          if (state.glyphActive) {
            state.glyphActive = false;
            pending = pending.filter(p => p.action.type !== 'clearGlyph');
            out.push({ type: 'clearGlyph' });
          }
          break;
        }

        case 'reply': {
          interaction(nowMs);
          state.engagedUntilMs = nowMs + ENGAGED_REPLY_HOLD_MS;
          break;
        }

        case 'reply_error': {
          interaction(nowMs);
          state.engagedUntilMs = nowMs + 3000;
          if (!state.glyphActive) {
            emitGlyph(out, nowMs, ERROR_GLYPH,
              { type: 'expression', name: 'polite_disbelief' });
          }
          break;
        }

        case 'hidden': {
          state.hiddenAtMs = nowMs;
          break;
        }

        case 'visible': {
          const away = state.hiddenAtMs >= 0 ? nowMs - state.hiddenAtMs : 0;
          state.hiddenAtMs = -1;
          interaction(nowMs);
          if (away >= HIDDEN_GREET_MIN_MS && !state.glyphActive) {
            emitGlyph(out, nowMs, RETURN_GLYPH,
              { type: 'expression', name: 'neutral_idle' });
          }
          scheduleNextIdleAct(nowMs, 'idle');
          break;
        }

        default:
          break;
      }
      return out;
    },

    getState() {
      return { ...state, pendingCount: pending.length };
    },
  };
}
