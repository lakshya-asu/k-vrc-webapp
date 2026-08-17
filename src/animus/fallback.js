import { SCHEMA_VERSION } from './contract.js';

function includesAny(text, words) {
  return words.some((word) => text.includes(word));
}

export function deterministicActorPlan(request) {
  const instruction = String(request.instruction ?? '').trim();
  const lower = instruction.toLowerCase();
  const target = String(request.target ?? 'camera').trim() || 'camera';

  let body = { action: 'gesture', gesture: 'talk', style: 'neutral', intensity: 0.45 };
  if (includesAny(lower, ['wave', 'hello', 'greet'])) {
    body = { action: 'gesture', gesture: 'wave', style: 'warm', intensity: 0.65 };
  } else if (includesAny(lower, ['walk', 'move', 'go to'])) {
    body = { action: 'walk_to', target, style: 'neutral', intensity: 0.5 };
  } else if (includesAny(lower, ['look', 'face', 'turn'])) {
    body = { action: 'turn_to', target, style: 'careful', intensity: 0.4 };
  } else if (includesAny(lower, ['wait', 'stop', 'hold'])) {
    body = { action: 'wait', style: 'neutral', intensity: 0.2 };
  }

  const beat = {
    id: 'fallback-1',
    at_ms: 0,
    duration_ms: 1800,
    body,
    gaze: { target, intensity: 0.5 },
    face: { expression: 'neutral_idle', intensity: 0.4 },
    speech: request.speech
      ? { text: String(request.speech).slice(0, 500), delivery: 'neutral' }
      : null,
  };

  return {
    schema_version: SCHEMA_VERSION,
    summary: instruction || 'Use the deterministic idle behavior.',
    beats: [beat],
  };
}
