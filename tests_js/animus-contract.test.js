import test from 'node:test';
import assert from 'node:assert/strict';
import { validateActorPlan } from '../src/animus/contract.js';

function validPlan() {
  return {
    schema_version: '0.1',
    summary: 'Wave to the viewer.',
    beats: [{
      id: 'wave-1',
      at_ms: 0,
      duration_ms: 1500,
      body: { action: 'gesture', gesture: 'wave', style: 'warm', intensity: 0.7 },
      gaze: { target: 'camera', intensity: 0.6 },
      face: { expression: 'warm_amused', intensity: 0.5 },
      speech: null,
    }],
  };
}

test('validates a semantic actor plan and injects caller authority', () => {
  const result = validateActorPlan(validPlan(), { actorId: 'kvrc', controlLevel: 'preview' });
  assert.equal(result.ok, true);
  assert.equal(result.value.actor_id, 'kvrc');
  assert.equal(result.value.control_level, 'preview');
});

test('rejects raw keyframes at any depth', () => {
  const candidate = validPlan();
  candidate.beats[0].body.raw_keyframes = [{ frame: 1 }];
  const result = validateActorPlan(candidate, { actorId: 'kvrc', controlLevel: 'suggest' });
  assert.equal(result.ok, false);
  assert.match(result.errors.join('\n'), /raw_keyframes is forbidden/);
});

test('rejects provider attempts to choose execution authority', () => {
  const candidate = { ...validPlan(), control_level: 'perform' };
  const result = validateActorPlan(candidate, { actorId: 'kvrc', controlLevel: 'suggest' });
  assert.equal(result.ok, false);
  assert.match(result.errors.join('\n'), /control_level is not allowed/);
});

test('requires targets for spatial body actions', () => {
  const candidate = validPlan();
  candidate.beats[0].body = { action: 'walk_to', intensity: 0.5 };
  const result = validateActorPlan(candidate, { actorId: 'kvrc', controlLevel: 'suggest' });
  assert.equal(result.ok, false);
  assert.match(result.errors.join('\n'), /target is required for walk_to/);
});
