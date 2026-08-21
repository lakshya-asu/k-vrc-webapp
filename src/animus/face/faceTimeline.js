// src/animus/face/faceTimeline.js
// Deterministic per-frame state for the rendered visor face.
//
// Turns one take's face beats (expression names from the webapp's
// expression library), the voice pipeline's viseme samples, and a seed
// into an explicit frame-by-frame state list the drawer consumes. All
// randomness (blink schedule, glitch schedule) comes from a seeded
// PRNG, so the same job and seed always produce the same frames.
//
// Semantics follow the webapp face:
//   - an expression, once set by a beat, eases in over the beat's ramp
//     frames (the embodiment mapper's ramp rule) and then HOLDS until
//     the next face beat replaces it, exactly like setExpression in
//     the live webapp face; it does not decay back to neutral.
//   - visemes drive mouth_open (and smile_width) on top of the held
//     expression; the loudest viseme weight doubles as the speech
//     amplitude that powers the waveform and scale bob.
//   - blinks run on the library's timer: first blink between 3 and 7
//     seconds, later blinks between 3.5 and 7 seconds, eyelid moving
//     at 14 units/second closed then open.
//   - the rare full-screen glitch uses the library's 8 to 20 second
//     timer with 80 to 200 ms bursts. Because that timer can never
//     fire inside a short take, a job may also carry force_glitches:
//     [{at_ms, duration_ms}] windows that turn the glitch overlay on
//     deterministically, on top of (never instead of) the seeded timer.

import { validateFaceGlyph } from './glyphComposer.js';

export const FACE_WEIGHT_KEYS = [
  'brow_raise',
  'brow_furrow',
  'eye_squint',
  'mouth_open',
  'smile_width',
  'glitch_intensity',
];

// Small fast seeded PRNG (mulberry32), replaces Math.random.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// JS Math.round semantics, the same rule the embodiment mappers use.
export function msToFrame(atMs, fps) {
  return 1 + Math.round((atMs * fps) / 1000);
}

export function durationToFrames(durationMs, fps) {
  return Math.max(2, Math.round((durationMs * fps) / 1000));
}

export function resolveExpression(name, library) {
  const entry = library.find((e) => e.name === name)
    ?? library.find((e) => e.name === 'neutral_idle');
  if (!entry) {
    throw new Error("expression library is missing 'neutral_idle'");
  }
  return entry;
}

function scaledWeights(entry, intensity) {
  const out = {};
  for (const key of FACE_WEIGHT_KEYS) {
    const w = (entry.weights[key] ?? 0) * intensity;
    out[key] = Math.min(1, Math.max(0, w));
  }
  return out;
}

function lerpWeights(from, to, t) {
  const out = {};
  for (const key of FACE_WEIGHT_KEYS) {
    out[key] = from[key] + (to[key] - from[key]) * t;
  }
  return out;
}

// Piecewise-linear sampler over {frame: value} points.
function makeTrack(points) {
  const frames = Object.keys(points).map(Number).sort((a, b) => a - b);
  return function at(frame) {
    if (frames.length === 0) return 0;
    if (frame <= frames[0] || frame >= frames[frames.length - 1]) {
      // Outside the sampled span the voice is silent.
      const edge = frame <= frames[0] ? frames[0] : frames[frames.length - 1];
      return frame === edge ? points[edge] : 0;
    }
    let lo = 0;
    while (frames[lo + 1] < frame) lo += 1;
    const f0 = frames[lo], f1 = frames[lo + 1];
    if (frame === f0) return points[f0];
    const t = (frame - f0) / (f1 - f0);
    return points[f0] + (points[f1] - points[f0]) * t;
  };
}

function visemeTracks(samples) {
  const mouth = {};
  const smile = {};
  const loudest = {};
  for (const sample of samples || []) {
    const { frame, shape_key: key, weight } = sample;
    loudest[frame] = Math.max(loudest[frame] ?? 0, weight);
    if (key === 'mouth_open') mouth[frame] = Math.max(mouth[frame] ?? 0, weight);
    if (key === 'smile_width') smile[frame] = Math.max(smile[frame] ?? 0, weight);
  }
  return {
    mouth: makeTrack(mouth),
    smile: makeTrack(smile),
    loudest: makeTrack(loudest),
  };
}

// Rhubarb's mouth-shape classes ride on the viseme samples (the voice
// converter stamps every sample with its cue's class letter, A-H or
// X). A cue's class holds from its start frame until the next cue
// starts, exactly like Rhubarb's own cue semantics. Frames before the
// first cue have no class; the drawer's weight-driven mouth is the
// fallback there and everywhere a job carries no classes at all.
export function visemeClassTrack(samples) {
  const byFrame = new Map();
  for (const sample of samples || []) {
    if (typeof sample.viseme === 'string' && sample.viseme.length > 0) {
      byFrame.set(sample.frame, sample.viseme);
    }
  }
  const frames = [...byFrame.keys()].sort((a, b) => a - b);
  return function at(frame) {
    if (frames.length === 0 || frame < frames[0]) return null;
    let lo = 0;
    while (lo + 1 < frames.length && frames[lo + 1] <= frame) lo += 1;
    return byFrame.get(frames[lo]);
  };
}

// The webapp's blink and glitch timers, advanced one frame at a time.
function makeTwitchState(rng) {
  return {
    blinkT: 3 + rng() * 4,
    blinking: false,
    blinkProgress: 0,
    blinkDir: 1,
    glitchTimer: 8 + rng() * 12,
    glitchLeft: 0,
  };
}

function advanceTwitch(state, dt, rng) {
  state.blinkT -= dt;
  if (state.blinkT <= 0 && !state.blinking) {
    state.blinking = true;
    state.blinkDir = 1;
    state.blinkT = 3.5 + rng() * 3.5;
  }
  if (state.blinking) {
    state.blinkProgress += state.blinkDir * dt * 14;
    if (state.blinkProgress >= 1) {
      state.blinkProgress = 2 - state.blinkProgress; // reverse
      state.blinkDir = -1;
    }
    if (state.blinkProgress <= 0) {
      state.blinkProgress = 0;
      state.blinkDir = 1;
      state.blinking = false;
    }
  }
  state.glitchTimer -= dt;
  if (state.glitchTimer <= 0 && state.glitchLeft <= 0) {
    state.glitchLeft = 0.08 + rng() * 0.12;
    state.glitchTimer = 8 + rng() * 12;
  } else if (state.glitchLeft > 0) {
    state.glitchLeft -= dt;
  }
}

// Forced glitch windows: [{at_ms, duration_ms}] -> a frame-time
// predicate. Invalid entries are refused loudly; an absent or empty
// list means the seeded timer alone owns the glitch, as before.
function makeForcedGlitch(windows) {
  const spans = (windows || []).map((entry, index) => {
    const at = entry?.at_ms;
    const dur = entry?.duration_ms;
    if (!Number.isInteger(at) || at < 0 || !Number.isInteger(dur) || dur < 1) {
      throw new Error(
        `force_glitches[${index}] needs integer at_ms >= 0 and duration_ms >= 1`,
      );
    }
    return { from: at / 1000, to: (at + dur) / 1000 };
  });
  return (t) => spans.some((span) => t >= span.from && t < span.to);
}

// job: {
//   fps, frame_end,
//   face_beats: [{expression, intensity, at_ms, duration_ms}] where an
//     entry may carry {glyph: <validated composed-glyph spec>} instead
//     of an expression name (the face_glyph beat channel),
//   viseme_samples: [{frame, shape_key, weight, viseme?}] where viseme
//     is the sample's Rhubarb mouth-shape class (A-H or X) when the
//     voice pipeline recorded one,
//   seed,
//   force_glitches: [{at_ms, duration_ms}]  (optional),
// }
// library: EXPRESSION_LIBRARY
// Returns [{frame, t, mood, weights, amplitude, blinkProgress,
//           glitchActive}] for frames 1..frame_end inclusive, plus
//           viseme (the held Rhubarb class) on frames at or after the
//           first classed cue.
export function buildFaceTimeline(job, library) {
  const fps = job.fps;
  const frameEnd = job.frame_end;
  if (!Number.isInteger(fps) || fps < 1) throw new Error('job.fps must be a positive integer');
  if (!Number.isInteger(frameEnd) || frameEnd < 1) throw new Error('job.frame_end must be a positive integer');

  const neutral = resolveExpression('neutral_idle', library);
  const segments = (job.face_beats || [])
    .map((beat, index) => {
      const base = msToFrame(beat.at_ms, fps);
      const durFrames = durationToFrames(beat.duration_ms, fps);
      const ramp = Math.max(1, Math.min(4, Math.floor(durFrames / 3)));
      if (beat.glyph != null) {
        // A composed glyph face (reel-polish brief, directive 3). The
        // spec is strictly validated; invalid specs are refused loudly.
        const checked = validateFaceGlyph(
          beat.glyph, `face_beats[${index}].glyph`,
        );
        if (!checked.ok) {
          throw new Error(checked.errors.join('; '));
        }
        return {
          base,
          ramp,
          mood: checked.value.mood ?? null, // null: inherit, fixed below
          weights: null,
          glyph: checked.value,
        };
      }
      const entry = resolveExpression(beat.expression, library);
      const intensity = beat.intensity ?? 1;
      return {
        base,
        ramp,
        mood: entry.mood,
        weights: scaledWeights(entry, intensity),
        glyph: null,
      };
    })
    .sort((a, b) => a.base - b.base);
  // A glyph without an explicit mood keeps the palette of whatever
  // face came before it.
  let inheritMood = neutral.mood;
  for (const segment of segments) {
    if (segment.mood == null) segment.mood = inheritMood;
    inheritMood = segment.mood;
  }

  const tracks = visemeTracks(job.viseme_samples);
  const visemeAt = visemeClassTrack(job.viseme_samples);
  const forcedGlitch = makeForcedGlitch(job.force_glitches);
  const rng = mulberry32(job.seed >>> 0);
  const twitch = makeTwitchState(rng);
  const dt = 1 / fps;

  const frames = [];
  let held = { mood: neutral.mood, weights: scaledWeights(neutral, 1) };
  let segIndex = -1;
  for (let frame = 1; frame <= frameEnd; frame++) {
    advanceTwitch(twitch, dt, rng);

    // Latest face beat whose base frame we have reached owns the face.
    while (segIndex + 1 < segments.length && segments[segIndex + 1].base <= frame) {
      if (segIndex >= 0) {
        // The previous expression is fully held before it is replaced.
        // A glyph segment holds no weights; the last weighted face
        // stays the ease-from state.
        held = {
          mood: segments[segIndex].mood,
          weights: segments[segIndex].weights ?? held.weights,
        };
      }
      segIndex += 1;
    }

    let mood = held.mood;
    let weights = held.weights;
    let glyph = null;
    if (segIndex >= 0) {
      const seg = segments[segIndex];
      mood = seg.mood;
      if (seg.glyph != null) {
        // Glyph faces pop like an LED redraw; no weight easing.
        glyph = seg.glyph;
      } else {
        const ease = Math.min(1, (frame - seg.base) / seg.ramp);
        weights = lerpWeights(held.weights, seg.weights, ease);
      }
    }

    const amplitude = Math.min(1, Math.max(0, tracks.loudest(frame)));
    const finalWeights = { ...weights };
    finalWeights.mouth_open = Math.max(finalWeights.mouth_open, tracks.mouth(frame));
    finalWeights.smile_width = Math.max(finalWeights.smile_width, tracks.smile(frame));

    const state = {
      frame,
      t: (frame - 1) / fps,
      mood,
      weights: finalWeights,
      amplitude,
      blinkProgress: Math.min(1, Math.max(0, twitch.blinkProgress)),
      glitchActive: twitch.glitchLeft > 0 || forcedGlitch((frame - 1) / fps),
    };
    if (glyph != null) state.glyph = glyph;
    const viseme = visemeAt(frame);
    if (viseme != null) state.viseme = viseme;
    frames.push(state);
  }
  return frames;
}
