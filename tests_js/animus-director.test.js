import test from 'node:test';
import assert from 'node:assert/strict';
import { ActorDirector } from '../src/animus/director.js';
import { extractJson } from '../src/animus/providers/openaiCompatible.js';

test('falls back when a provider returns an invalid plan', async () => {
  const director = new ActorDirector({
    providers: [{
      name: 'bad-provider',
      model: 'bad-model',
      async plan() {
        return { schema_version: '0.1', summary: 'Unsafe', keyframes: [] };
      },
    }],
  });

  const plan = await director.plan({
    actorId: 'kvrc',
    controlLevel: 'suggest',
    instruction: 'Wave hello',
    target: 'camera',
  });

  assert.equal(plan.provenance.operator, 'deterministic');
  assert.equal(plan.provenance.fallback, true);
  assert.equal(plan.provenance.prior_failures[0].provider, 'bad-provider');
  assert.equal(plan.beats[0].body.gesture, 'wave');
});

test('accepts fenced JSON without accepting surrounding prose', () => {
  const parsed = extractJson('```json\n{"schema_version":"0.1","summary":"Idle","beats":[]}\n```');
  assert.equal(parsed.summary, 'Idle');
});

test('takes the first balanced object when the model appends trailing output', () => {
  const parsed = extractJson('{"summary":"Plan","beats":[{"id":"b-1"}]}\n{"summary":"Echo"}');
  assert.equal(parsed.summary, 'Plan');
  const withProse = extractJson('{"summary":"A {brace} in \\"text\\""} and then some prose}');
  assert.equal(withProse.summary, 'A {brace} in "text"');
});

test('uses the first valid provider', async () => {
  const director = new ActorDirector({
    providers: [{
      name: 'local-small',
      model: 'Qwen3-4B-Q4_K_M',
      async plan() {
        return {
          schema_version: '0.1',
          summary: 'Look at the camera.',
          beats: [{
            id: 'look-1',
            at_ms: 0,
            duration_ms: 900,
            gaze: { target: 'camera', intensity: 0.7 },
          }],
        };
      },
    }],
  });

  const plan = await director.plan({ actorId: 'kvrc', controlLevel: 'preview', instruction: 'Look here' });
  assert.equal(plan.provenance.operator, 'local-small');
  assert.equal(plan.control_level, 'preview');
});
