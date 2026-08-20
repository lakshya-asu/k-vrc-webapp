import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  mapPlanToBridgeJobs,
  sanitizeNameHint,
  validateEmbodimentProfile,
  visemeArtifactToPoseRequest,
  visemeArtifactToRequest,
} from '../src/animus/embodiment.js';
import { validateActorPlan } from '../src/animus/contract.js';
import { deterministicActorPlan } from '../src/animus/fallback.js';

const REPO = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const PROFILE_PATH = path.join(REPO, 'src', 'animus', 'embodiment', 'kvrc-testrig.profile.json');
const KVRC_PROFILE_PATH = path.join(REPO, 'src', 'animus', 'embodiment', 'kvrc.profile.json');

function loadProfile() {
  return JSON.parse(readFileSync(PROFILE_PATH, 'utf-8'));
}

function validProfile() {
  const checked = validateEmbodimentProfile(loadProfile());
  assert.equal(checked.ok, true, checked.errors.join('; '));
  return checked.value;
}

function validatedFallbackPlan(request) {
  const candidate = deterministicActorPlan(request);
  const checked = validateActorPlan(candidate, {
    actorId: 'kvrc',
    controlLevel: request.controlLevel ?? 'suggest',
  });
  assert.equal(checked.ok, true, checked.errors.join('; '));
  return checked.value;
}

// Structural mirror of blender/animus_bridge/protocol.py, applied to
// every mapped request so nothing the mapper emits can be refused by
// the bridge on structure.
function assertBridgeShaped(request) {
  assert.match(request.id, /^[^\s]{1,120}$/);
  assert.ok(['perform_take', 'apply_shape_keys'].includes(request.op),
    `mapper emitted unexpected op ${request.op}`);
  const params = request.params;
  assert.deepEqual(
    Object.keys(params).sort(),
    ['frame_end', 'frame_start', 'name_hint', 'object', 'samples'],
  );
  assert.match(params.name_hint, /^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,59}$/);
  assert.ok(Number.isInteger(params.frame_start) && params.frame_start >= 0);
  assert.ok(Number.isInteger(params.frame_end) && params.frame_end <= 100000);
  assert.ok(params.frame_start <= params.frame_end);
  assert.ok(params.frame_end - params.frame_start <= 10000);
  assert.ok(params.samples.length >= 1 && params.samples.length <= 2000);
  for (const sample of params.samples) {
    assert.ok(sample.frame >= params.frame_start && sample.frame <= params.frame_end,
      `sample frame ${sample.frame} outside ${params.frame_start}..${params.frame_end}`);
    if (request.op === 'perform_take') {
      assert.ok('rotation_quaternion' in sample || 'location' in sample);
    } else {
      assert.equal(typeof sample.shape_key, 'string');
      assert.equal(typeof sample.weight, 'number');
      assert.ok(sample.weight >= 0 && sample.weight <= 1);
    }
  }
}

test('the committed test-rig profile validates', () => {
  validProfile();
});

test('profile validation refuses bad data', () => {
  const unknownBone = loadProfile();
  unknownBone.gestures.wave.samples[0].bone = 'tentacle.R';
  assert.equal(validateEmbodimentProfile(unknownBone).ok, false);

  const badWeight = loadProfile();
  badWeight.expressions.presets.warm_amused.smile_width = 1.5;
  assert.equal(validateEmbodimentProfile(badWeight).ok, false);

  const unknownShapeKey = loadProfile();
  unknownShapeKey.expressions.presets.concerned.eyebrow_raise = 0.5;
  assert.equal(validateEmbodimentProfile(unknownShapeKey).ok, false);

  const badDefault = loadProfile();
  badDefault.default_gesture = 'moonwalk';
  assert.equal(validateEmbodimentProfile(badDefault).ok, false);

  const speechMismatch = loadProfile();
  speechMismatch.speech.object = 'KVRC';
  assert.equal(validateEmbodimentProfile(speechMismatch).ok, false);

  const badActionMap = loadProfile();
  badActionMap.body_action_gestures.interact = 'moonwalk';
  assert.equal(validateEmbodimentProfile(badActionMap).ok, false);
});

test('the fallback wave plan maps to body, gaze, face layers and one voice job', () => {
  const profile = validProfile();
  const plan = validatedFallbackPlan({
    instruction: 'Wave to the viewer and say hello.',
    target: 'camera',
    speech: 'Hello, I am K-VRC.',
  });
  const jobs = mapPlanToBridgeJobs(plan, profile);

  assert.deepEqual(jobs.profile, { name: 'kvrc-testrig', version: '0.1.0' });
  assert.deepEqual(jobs.layers.map((layer) => layer.channel), ['body', 'gaze', 'face']);
  assert.equal(jobs.voice_jobs.length, 1);

  const body = jobs.layers[0];
  assert.equal(body.gesture, 'wave');
  assert.equal(body.request.op, 'perform_take');
  assert.equal(body.request.params.object, 'KVRC');
  assert.equal(body.request.params.frame_start, 1);
  assert.equal(body.request.params.frame_end, 48);
  const bones = new Set(body.request.params.samples.map((sample) => sample.bone));
  assert.deepEqual([...bones].sort(), ['arm.R', 'hand.R']);

  const gaze = jobs.layers[1];
  assert.equal(gaze.target, 'camera');
  assert.equal(gaze.request.params.samples[0].bone, 'head');

  const face = jobs.layers[2];
  assert.equal(face.expression, 'neutral_idle');
  assert.equal(face.request.op, 'apply_shape_keys');
  assert.equal(face.request.params.object, 'KVRC_face');

  const voice = jobs.voice_jobs[0];
  assert.equal(voice.text, 'Hello, I am K-VRC.');
  assert.equal(voice.object, 'KVRC_face');
  assert.equal(voice.frame_start, 1);

  for (const layer of jobs.layers) assertBridgeShaped(layer.request);
});

test('beat timing places takes: at_ms 1000 at 24 fps starts on frame 25', () => {
  const profile = validProfile();
  const plan = {
    beats: [{
      id: 'later-1',
      at_ms: 1000,
      duration_ms: 2000,
      body: { action: 'gesture', gesture: 'nod' },
      gaze: { target: 'left', intensity: 1 },
    }],
  };
  const jobs = mapPlanToBridgeJobs(plan, profile);
  const body = jobs.layers[0].request.params;
  assert.equal(body.frame_start, 25);
  assert.equal(body.frame_end, 25 + 24 - 1);
  assert.equal(Math.min(...body.samples.map((sample) => sample.frame)), 25);

  const gaze = jobs.layers[1].request.params;
  assert.equal(gaze.frame_start, 25);
  assert.equal(gaze.frame_end, 25 + 48 - 1);
  // The mapper renormalizes profile quaternions, so check direction
  // and unit length instead of raw profile values.
  const last = gaze.samples[gaze.samples.length - 1].rotation_quaternion;
  assert.ok(Math.abs(last[0] - 0.976) < 0.001 && Math.abs(last[3] - 0.216) < 0.001, last);
  assert.ok(Math.abs(Math.hypot(...last) - 1) < 1e-6);
});

test('unknown names degrade deterministically to profile defaults', () => {
  const profile = validProfile();
  const plan = {
    beats: [{
      id: 'odd-1',
      at_ms: 0,
      duration_ms: 1500,
      body: { action: 'gesture', gesture: 'backflip' },
      gaze: { target: 'the moon' },
      face: { expression: 'existential_dread' },
    }, {
      id: 'odd-2',
      at_ms: 2000,
      duration_ms: 1500,
      body: { action: 'walk_to', target: 'door' },
    }],
  };
  const jobs = mapPlanToBridgeJobs(plan, profile);
  assert.equal(jobs.layers[0].gesture, 'idle');
  assert.equal(jobs.layers[1].target, 'camera');
  assert.equal(jobs.layers[2].expression, 'neutral_idle');
  assert.equal(jobs.layers[3].gesture, 'idle');
  for (const layer of jobs.layers) assertBridgeShaped(layer.request);
});

test('gaze intensity 0 keeps the neutral pose; face intensity scales weights', () => {
  const profile = validProfile();
  const plan = {
    beats: [{
      id: 'soft-1',
      at_ms: 0,
      duration_ms: 2000,
      gaze: { target: 'left', intensity: 0 },
      face: { expression: 'warm_amused', intensity: 0.5 },
    }],
  };
  const jobs = mapPlanToBridgeJobs(plan, profile);
  const gazeSamples = jobs.layers[0].request.params.samples;
  for (const sample of gazeSamples) {
    assert.deepEqual(sample.rotation_quaternion, [1, 0, 0, 0]);
  }
  const faceSamples = jobs.layers[1].request.params.samples;
  const peakSmile = faceSamples.find(
    (sample) => sample.shape_key === 'smile_width' && sample.weight > 0,
  );
  assert.equal(peakSmile.weight, 0.3);
  for (const sample of faceSamples) {
    assert.ok(sample.weight >= 0 && sample.weight <= 1);
  }
});

test('a very short face beat still yields an increasing frame envelope', () => {
  const profile = validProfile();
  const plan = {
    beats: [{
      id: 'blip-1',
      at_ms: 0,
      duration_ms: 100,
      face: { expression: 'surprised' },
    }],
  };
  const jobs = mapPlanToBridgeJobs(plan, profile);
  assertBridgeShaped(jobs.layers[0].request);
  const frames = [...new Set(jobs.layers[0].request.params.samples.map((s) => s.frame))];
  assert.deepEqual(frames, [1, 2]);
});

test('viseme artifacts map field for field onto apply_shape_keys', () => {
  const artifact = {
    kind: 'animus_viseme_take',
    object: 'KVRC',
    name_hint: 'animus_speech',
    frame_start: 1,
    frame_end: 51,
    samples: [{ shape_key: 'mouth_open', frame: 1, weight: 0, at_ms: 0, viseme: 'X' }],
  };
  const request = visemeArtifactToRequest(artifact, { requestId: 'dir-x', object: 'KVRC_face' });
  assert.equal(request.op, 'apply_shape_keys');
  assert.equal(request.params.object, 'KVRC_face');
  assert.equal(request.params.frame_end, 51);
  assert.equal(request.params.samples.length, 1);

  assert.throws(() => visemeArtifactToRequest({ kind: 'something_else' }), /expected 'animus_viseme_take'/);
});

test('name hints are always bridge-legal', () => {
  assert.equal(sanitizeNameHint('greet 1!'), 'greet_1_');
  assert.equal(sanitizeNameHint('.hidden'), 'x.hidden');
  assert.equal(sanitizeNameHint(''), 'take');
  assert.equal(sanitizeNameHint('a'.repeat(80)).length, 60);
});

// --- the real K-VRC profile: mechanical face and speech ---------------

function loadKvrcProfile() {
  return JSON.parse(readFileSync(KVRC_PROFILE_PATH, 'utf-8'));
}

test('the committed K-VRC profile validates without shape keys', () => {
  const checked = validateEmbodimentProfile(loadKvrcProfile());
  assert.equal(checked.ok, true, checked.errors.join('; '));
  assert.equal(checked.value.rig.object, 'KVRCArmature');
  assert.equal('face_object' in checked.value.rig, false);
});

test('shape-key modes still require the shape-key structures', () => {
  const profile = loadKvrcProfile();
  delete profile.expressions.mode;
  const checked = validateEmbodimentProfile(profile);
  assert.equal(checked.ok, false);
  assert.ok(checked.errors.some((error) => error.includes('rig.face_object')));
  assert.ok(checked.errors.some((error) => error.includes('rig.shape_keys')));
});

test('a pose-mode face beat maps to a perform_take on the armature', () => {
  const checked = validateEmbodimentProfile(loadKvrcProfile());
  assert.equal(checked.ok, true, checked.errors.join('; '));
  const plan = validatedFallbackPlan({
    instruction: 'Wave to the viewer and say hello.',
    target: 'camera',
    speech: 'Hello, I am K-VRC.',
  });
  const jobs = mapPlanToBridgeJobs(plan, checked.value);
  const face = jobs.layers.find((layer) => layer.channel === 'face');
  assert.equal(face.request.op, 'perform_take');
  assert.equal(face.request.params.object, 'KVRCArmature');
  assertBridgeShaped(face.request);
  assert.ok(face.request.params.samples.every((sample) => sample.bone === 'Head'));
});

test('a viseme artifact becomes an ear pose take in bone mode', () => {
  const profile = loadKvrcProfile();
  const artifact = {
    kind: 'animus_viseme_take',
    object: 'KVRCArmature',
    name_hint: 'animus_speech',
    frame_start: 1,
    frame_end: 5,
    samples: [
      { shape_key: 'mouth_open', frame: 1, weight: 0 },
      { shape_key: 'smile_width', frame: 1, weight: 0.1 },
      { shape_key: 'mouth_open', frame: 3, weight: 0.85 },
      { shape_key: 'mouth_open', frame: 5, weight: 0 },
    ],
  };
  const request = visemeArtifactToPoseRequest(artifact, profile.speech, { requestId: 'dir-x' });
  assert.equal(request.op, 'perform_take');
  assert.equal(request.params.object, 'KVRCArmature');
  assertBridgeShaped(request);
  assert.equal(request.params.samples.length, 3 * profile.speech.bones.length);
  const silent = request.params.samples.filter((sample) => sample.frame === 1);
  for (const sample of silent) {
    assert.deepEqual(sample.rotation_quaternion, [1, 0, 0, 0]);
  }
  const loud = request.params.samples.find((sample) => sample.frame === 3);
  assert.notDeepEqual(loud.rotation_quaternion, [1, 0, 0, 0]);
  assert.throws(
    () => visemeArtifactToPoseRequest({ kind: 'nope' }, profile.speech),
    /expected 'animus_viseme_take'/,
  );
});
