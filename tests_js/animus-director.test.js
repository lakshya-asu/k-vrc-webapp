import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FACE_GLYPH_BROWS,
  FACE_GLYPH_EYES,
  FACE_GLYPH_MOODS,
  FACE_GLYPH_MOUTHS,
} from '../src/animus/contract.js';
import { ActorDirector } from '../src/animus/director.js';
import { ACTOR_SYSTEM_PROMPT } from '../src/animus/prompt.js';
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
  assert.equal('face_glyph' in plan.provenance, false);
});

test('the system prompt teaches the face_glyph channel', () => {
  assert.ok(ACTOR_SYSTEM_PROMPT.includes('face_glyph'));
  for (const vocab of [FACE_GLYPH_EYES, FACE_GLYPH_BROWS, FACE_GLYPH_MOUTHS, FACE_GLYPH_MOODS]) {
    assert.ok(ACTOR_SYSTEM_PROMPT.includes(vocab.join(', ')), vocab.join(', '));
  }
  assert.ok(ACTOR_SYSTEM_PROMPT.includes('Never use face and face_glyph in the same beat.'));
  assert.ok(ACTOR_SYSTEM_PROMPT.includes('"face_glyph":{"eyes":"happy_arc"'));
});

test('a model plan that composes the visor is marked model-authored', async () => {
  const director = new ActorDirector({
    providers: [{
      name: 'local-small',
      model: 'Qwen3-4B-Q4_K_M',
      async plan() {
        return {
          schema_version: '0.1',
          summary: 'Grin at the viewer.',
          beats: [{
            id: 'grin-1',
            at_ms: 0,
            duration_ms: 1200,
            face_glyph: { eyes: 'happy_arc', mouth: 'grin_rect', mood: 'warm' },
          }],
        };
      },
    }],
  });

  const plan = await director.plan({ actorId: 'kvrc', controlLevel: 'suggest', instruction: 'Grin' });
  assert.equal(plan.provenance.fallback, false);
  assert.equal(plan.provenance.face_glyph, 'model-authored');
});

test('the provider sends the grammar schema and the schema matches the fixture', async () => {
  const { readFileSync } = await import('node:fs');
  const { ACTOR_PLAN_JSON_SCHEMA } = await import('../src/animus/prompt.js');
  const { createOpenAICompatibleProvider } = await import('../src/animus/providers/openaiCompatible.js');

  const fixture = JSON.parse(readFileSync(
    new URL('../tests/animus_bridge/fixtures/actor_plan_schema.json', import.meta.url), 'utf-8',
  ));
  assert.deepEqual(ACTOR_PLAN_JSON_SCHEMA, fixture);

  let sentBody = null;
  const provider = createOpenAICompatibleProvider({
    fetchImpl: async (url, options) => {
      sentBody = JSON.parse(options.body);
      return {
        ok: true,
        async json() {
          return { choices: [{ message: { content: '{"schema_version":"0.1","summary":"x","beats":[]}' } }] };
        },
      };
    },
  });
  await provider.plan({ instruction: 'Wave' });
  assert.equal(sentBody.response_format.type, 'json_schema');
  assert.deepEqual(sentBody.response_format.json_schema.schema, fixture);
});
