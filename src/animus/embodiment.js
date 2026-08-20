// Embodiment profile: the only place semantic actor intent becomes
// numeric motion. Every quaternion, frame, and shape-key weight the
// bridge ever receives comes from a validated profile (or from the
// voice pipeline's viseme artifact). The actor model names things;
// it never supplies numbers (decisions A-005, A-006, A-008).

const PROFILE_KIND = 'animus_embodiment_profile';
const PROFILE_SCHEMA_VERSION = '0.1';

// Mirrors blender/animus_bridge/protocol.py limits so a mapped request
// can never be structurally refused by the bridge.
const MAX_FRAME = 100000;
const MAX_FRAME_SPAN = 10000;
const MAX_SAMPLES = 2000;
const NAME_HINT_RE = /^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,59}$/;

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isFrame(value) {
  return Number.isInteger(value) && value >= 1 && value <= MAX_FRAME;
}

function isQuaternion(value) {
  return Array.isArray(value) && value.length === 4 && value.every(isFiniteNumber);
}

function round6(value) {
  return Math.round(value * 1e6) / 1e6;
}

export function sanitizeNameHint(text, fallback = 'take') {
  let hint = String(text ?? '').replace(/[^A-Za-z0-9_.\-]/g, '_').slice(0, 60);
  if (hint && !/^[A-Za-z0-9_]/.test(hint)) hint = `x${hint}`.slice(0, 60);
  if (!hint || !NAME_HINT_RE.test(hint)) return fallback;
  return hint;
}

function validateGestureSamples(gesture, name, bones, errors) {
  if (!Array.isArray(gesture.samples) || gesture.samples.length === 0) {
    errors.push(`gestures.${name}.samples must be a non-empty list`);
    return;
  }
  if (gesture.samples.length > MAX_SAMPLES) {
    errors.push(`gestures.${name}.samples exceeds ${MAX_SAMPLES} items`);
  }
  gesture.samples.forEach((sample, index) => {
    const path = `gestures.${name}.samples[${index}]`;
    if (!isObject(sample)) {
      errors.push(`${path} must be an object`);
      return;
    }
    if (!bones.has(sample.bone)) {
      errors.push(`${path}.bone '${sample.bone}' is not a rig bone`);
    }
    if (!isFrame(sample.frame) || sample.frame > gesture.frames) {
      errors.push(`${path}.frame must be an integer from 1 to gestures.${name}.frames`);
    }
    const hasRotation = 'rotation_quaternion' in sample;
    const hasLocation = 'location' in sample;
    if (!hasRotation && !hasLocation) {
      errors.push(`${path} needs rotation_quaternion or location`);
    }
    if (hasRotation && !isQuaternion(sample.rotation_quaternion)) {
      errors.push(`${path}.rotation_quaternion must be 4 finite numbers`);
    }
    if (hasLocation
      && !(Array.isArray(sample.location) && sample.location.length === 3
        && sample.location.every(isFiniteNumber))) {
      errors.push(`${path}.location must be 3 finite numbers`);
    }
  });
}

function channelMode(section, key, allowed, fallback, errors) {
  if (!isObject(section) || !('mode' in section)) return fallback;
  if (!allowed.includes(section.mode)) {
    errors.push(`${key}.mode must be one of ${[...allowed].sort().join(', ')}`);
    return fallback;
  }
  return section.mode;
}

export function validateEmbodimentProfile(profile) {
  const errors = [];
  if (!isObject(profile)) {
    return { ok: false, errors: ['profile must be an object'], value: null };
  }
  if (profile.kind !== PROFILE_KIND) {
    errors.push(`profile.kind must be '${PROFILE_KIND}'`);
  }
  if (profile.schema_version !== PROFILE_SCHEMA_VERSION) {
    errors.push(`profile.schema_version must be '${PROFILE_SCHEMA_VERSION}'`);
  }
  if (typeof profile.profile !== 'string' || !profile.profile) {
    errors.push('profile.profile must name the profile');
  }
  if (typeof profile.profile_version !== 'string' || !profile.profile_version) {
    errors.push('profile.profile_version is required');
  }
  if (!isObject(profile.source) || typeof profile.source.license !== 'string') {
    errors.push('profile.source.license is required');
  }
  if (!Number.isInteger(profile.fps) || profile.fps < 1 || profile.fps > 240) {
    errors.push('profile.fps must be an integer from 1 to 240');
  }

  // Mechanical rigs (rigid robots with no morph targets) may map the
  // face channel to a bone pose and speech to viseme-driven bone
  // motion. Shape-key structures are then optional; nothing is faked.
  const faceMode = channelMode(
    profile.expressions, 'expressions', ['shape_keys', 'pose'], 'shape_keys', errors,
  );
  const speechMode = channelMode(
    profile.speech, 'speech', ['shape_keys', 'bone'], 'shape_keys', errors,
  );
  const needsShapeKeys = faceMode === 'shape_keys' || speechMode === 'shape_keys';

  const rig = profile.rig;
  let bones = new Set();
  let shapeKeys = new Set();
  if (!isObject(rig)) {
    errors.push('profile.rig must be an object');
  } else {
    if (typeof rig.object !== 'string' || !rig.object) errors.push('rig.object is required');
    if (!Array.isArray(rig.bones) || rig.bones.length === 0
      || rig.bones.some((bone) => typeof bone !== 'string' || !bone)) {
      errors.push('rig.bones must be a non-empty list of bone names');
    } else {
      bones = new Set(rig.bones);
    }
    if ((needsShapeKeys || rig.face_object != null)
      && (typeof rig.face_object !== 'string' || !rig.face_object)) {
      errors.push('rig.face_object is required');
    }
    if (needsShapeKeys || rig.shape_keys != null) {
      if (!Array.isArray(rig.shape_keys) || rig.shape_keys.length === 0
        || rig.shape_keys.some((key) => typeof key !== 'string' || !key)) {
        errors.push('rig.shape_keys must be a non-empty list of shape key names');
      } else {
        shapeKeys = new Set(rig.shape_keys);
      }
    }
  }

  const gestures = profile.gestures;
  if (!isObject(gestures) || Object.keys(gestures).length === 0) {
    errors.push('profile.gestures must be a non-empty object');
  } else {
    for (const [name, gesture] of Object.entries(gestures)) {
      if (!isObject(gesture)) {
        errors.push(`gestures.${name} must be an object`);
        continue;
      }
      if (!NAME_HINT_RE.test(gesture.name_hint ?? '')) {
        errors.push(`gestures.${name}.name_hint must be a valid bridge name hint`);
      }
      if (!isFrame(gesture.frames) || gesture.frames > MAX_FRAME_SPAN) {
        errors.push(`gestures.${name}.frames must be an integer from 1 to ${MAX_FRAME_SPAN}`);
      } else {
        validateGestureSamples(gesture, name, bones, errors);
      }
    }
    if (typeof profile.default_gesture !== 'string' || !(profile.default_gesture in gestures)) {
      errors.push('profile.default_gesture must name a defined gesture');
    }
    if (!isObject(profile.body_action_gestures)) {
      errors.push('profile.body_action_gestures must be an object');
    } else {
      for (const [action, name] of Object.entries(profile.body_action_gestures)) {
        if (!(name in gestures)) {
          errors.push(`body_action_gestures.${action} names unknown gesture '${name}'`);
        }
      }
    }
  }

  const gaze = profile.gaze;
  if (!isObject(gaze)) {
    errors.push('profile.gaze must be an object');
  } else {
    if (!bones.has(gaze.bone)) errors.push(`gaze.bone '${gaze.bone}' is not a rig bone`);
    if (!NAME_HINT_RE.test(gaze.name_hint ?? '')) errors.push('gaze.name_hint must be a valid bridge name hint');
    if (!isQuaternion(gaze.neutral)) errors.push('gaze.neutral must be 4 finite numbers');
    if (!Number.isInteger(gaze.ease_frames) || gaze.ease_frames < 1 || gaze.ease_frames > 240) {
      errors.push('gaze.ease_frames must be an integer from 1 to 240');
    }
    if (!isObject(gaze.targets) || Object.keys(gaze.targets).length === 0) {
      errors.push('gaze.targets must be a non-empty object');
    } else {
      for (const [name, quat] of Object.entries(gaze.targets)) {
        if (!isQuaternion(quat)) errors.push(`gaze.targets.${name} must be 4 finite numbers`);
      }
      if (!(gaze.default_target in gaze.targets)) {
        errors.push('gaze.default_target must name a defined target');
      }
    }
  }

  const expressions = profile.expressions;
  if (!isObject(expressions) || !isObject(expressions.presets)
    || Object.keys(expressions.presets).length === 0) {
    errors.push('profile.expressions.presets must be a non-empty object');
  } else {
    if (!NAME_HINT_RE.test(expressions.name_hint ?? '')) {
      errors.push('expressions.name_hint must be a valid bridge name hint');
    }
    if (faceMode === 'pose') {
      if (!bones.has(expressions.bone)) {
        errors.push(`expressions.bone '${expressions.bone}' is not a rig bone`);
      }
      if (!isQuaternion(expressions.neutral)) {
        errors.push('expressions.neutral must be 4 finite numbers');
      }
      for (const [name, preset] of Object.entries(expressions.presets)) {
        if (!isQuaternion(preset)) {
          errors.push(`expressions.presets.${name} must be 4 finite numbers in pose mode`);
        }
      }
    } else {
      for (const [name, preset] of Object.entries(expressions.presets)) {
        if (!isObject(preset) || Object.keys(preset).length === 0) {
          errors.push(`expressions.presets.${name} must be a non-empty object`);
          continue;
        }
        for (const [key, weight] of Object.entries(preset)) {
          if (!shapeKeys.has(key)) {
            errors.push(`expressions.presets.${name}.${key} is not a rig shape key`);
          }
          if (!isFiniteNumber(weight) || weight < 0 || weight > 1) {
            errors.push(`expressions.presets.${name}.${key} must be a number from 0 to 1`);
          }
        }
      }
    }
    if (!(expressions.default_expression in expressions.presets)) {
      errors.push('expressions.default_expression must name a defined preset');
    }
  }

  const speech = profile.speech;
  if (!isObject(speech)) {
    errors.push('profile.speech must be an object');
  } else {
    if (speechMode === 'bone') {
      if (isObject(rig) && speech.object !== rig.object) {
        errors.push('speech.object must equal rig.object in bone mode');
      }
      if (!Array.isArray(speech.bones) || speech.bones.length === 0) {
        errors.push('speech.bones must be a non-empty list in bone mode');
      } else {
        speech.bones.forEach((entry, index) => {
          const path = `speech.bones[${index}]`;
          if (!isObject(entry)) {
            errors.push(`${path} must be an object`);
            return;
          }
          if (!bones.has(entry.bone)) {
            errors.push(`${path}.bone '${entry.bone}' is not a rig bone`);
          }
          if (!isQuaternion(entry.neutral)) errors.push(`${path}.neutral must be 4 finite numbers`);
          if (!isQuaternion(entry.peak)) errors.push(`${path}.peak must be 4 finite numbers`);
        });
      }
      if ('driver' in speech && (typeof speech.driver !== 'string' || !speech.driver)) {
        errors.push('speech.driver must be a viseme weight name');
      }
    } else if (isObject(rig) && speech.object !== rig.face_object) {
      errors.push('speech.object must equal rig.face_object');
    }
    if (!NAME_HINT_RE.test(speech.name_hint ?? '')) {
      errors.push('speech.name_hint must be a valid bridge name hint');
    }
    if (typeof speech.voice !== 'string' || !speech.voice) errors.push('speech.voice is required');
    if ('backend' in speech && speech.backend !== 'kokoro' && speech.backend !== 'chatterbox') {
      errors.push("speech.backend must be 'kokoro' or 'chatterbox' when present");
    }
    if ('tts' in speech && !isObject(speech.tts)) {
      errors.push('speech.tts must be an object when present');
    }
  }

  if (errors.length > 0) return { ok: false, errors, value: null };
  return { ok: true, errors: [], value: profile };
}

function msToFrame(atMs, fps) {
  return 1 + Math.round((atMs * fps) / 1000);
}

function durationToFrames(durationMs, fps) {
  return Math.max(2, Math.round((durationMs * fps) / 1000));
}

function nlerp(from, to, t) {
  const mixed = from.map((value, index) => value + (to[index] - value) * t);
  const length = Math.hypot(...mixed);
  if (length === 0) return [1, 0, 0, 0];
  return mixed.map((value) => round6(value / length));
}

function requestId(beatId, channel) {
  return `dir-${sanitizeNameHint(beatId, 'beat')}-${channel}`.slice(0, 120);
}

function resolveGestureName(body, profile) {
  if (body.action === 'gesture') {
    return body.gesture in profile.gestures ? body.gesture : profile.default_gesture;
  }
  return profile.body_action_gestures[body.action] ?? profile.default_gesture;
}

function mapBodyBeat(beat, profile) {
  const gestureName = resolveGestureName(beat.body, profile);
  const gesture = profile.gestures[gestureName];
  const base = msToFrame(beat.at_ms, profile.fps);
  const offset = base - 1;
  const samples = gesture.samples.map((sample) => {
    const moved = { bone: sample.bone, frame: sample.frame + offset };
    if ('rotation_quaternion' in sample) moved.rotation_quaternion = sample.rotation_quaternion;
    if ('location' in sample) moved.location = sample.location;
    return moved;
  });
  return {
    beat_id: beat.id,
    channel: 'body',
    gesture: gestureName,
    request: {
      id: requestId(beat.id, 'body'),
      op: 'perform_take',
      params: {
        object: profile.rig.object,
        name_hint: sanitizeNameHint(gesture.name_hint, 'gesture'),
        frame_start: base,
        frame_end: offset + gesture.frames,
        samples,
      },
    },
  };
}

function mapGazeBeat(beat, profile) {
  const gaze = profile.gaze;
  const targetName = beat.gaze.target in gaze.targets ? beat.gaze.target : gaze.default_target;
  const intensity = beat.gaze.intensity ?? 1;
  const aimed = nlerp(gaze.neutral, gaze.targets[targetName], intensity);
  const base = msToFrame(beat.at_ms, profile.fps);
  const durFrames = durationToFrames(beat.duration_ms, profile.fps);
  const end = base + durFrames - 1;
  const ease = Math.min(gaze.ease_frames, durFrames - 1);
  const frames = new Map();
  frames.set(base, gaze.neutral.map(round6));
  frames.set(Math.min(base + ease, end), aimed);
  frames.set(end, aimed);
  const samples = [...frames.entries()].map(([frame, quat]) => ({
    bone: gaze.bone,
    frame,
    rotation_quaternion: quat,
  }));
  return {
    beat_id: beat.id,
    channel: 'gaze',
    target: targetName,
    request: {
      id: requestId(beat.id, 'gaze'),
      op: 'perform_take',
      params: {
        object: profile.rig.object,
        name_hint: sanitizeNameHint(`${gaze.name_hint}_${targetName}`, 'gaze'),
        frame_start: base,
        frame_end: end,
        samples,
      },
    },
  };
}

function mapFaceBeat(beat, profile) {
  const expressions = profile.expressions;
  const name = beat.face.expression in expressions.presets
    ? beat.face.expression
    : expressions.default_expression;
  const preset = expressions.presets[name];
  const intensity = beat.face.intensity ?? 1;
  const base = msToFrame(beat.at_ms, profile.fps);
  const durFrames = durationToFrames(beat.duration_ms, profile.fps);
  const end = base + durFrames - 1;
  const ramp = Math.max(1, Math.min(4, Math.floor(durFrames / 3)));

  // Ramp in, hold, ramp out; a beat too short for the ramp collapses to
  // peak-at-start, zero-at-end. Later writes win on frame collisions.
  const frameWeights = new Map();
  if (end - base >= 2 * ramp + 1) {
    frameWeights.set(base, 0);
    frameWeights.set(base + ramp, 1);
    frameWeights.set(end - ramp, 1);
    frameWeights.set(end, 0);
  } else {
    frameWeights.set(base, 1);
    frameWeights.set(end, 0);
  }

  const nameHint = sanitizeNameHint(`${expressions.name_hint}_${name}`, 'face');

  if (expressions.mode === 'pose') {
    // Mechanical face: the mood is a bone pose (head tilt), eased by
    // the same envelope shape-key expressions use.
    const samples = [...frameWeights.entries()].map(([frame, envelope]) => ({
      bone: expressions.bone,
      frame,
      rotation_quaternion: nlerp(
        expressions.neutral, preset, Math.min(1, Math.max(0, intensity * envelope)),
      ),
    }));
    return {
      beat_id: beat.id,
      channel: 'face',
      expression: name,
      request: {
        id: requestId(beat.id, 'face'),
        op: 'perform_take',
        params: {
          object: profile.rig.object,
          name_hint: nameHint,
          frame_start: base,
          frame_end: end,
          samples,
        },
      },
    };
  }

  const samples = [];
  for (const [frame, envelope] of frameWeights.entries()) {
    for (const [shapeKey, weight] of Object.entries(preset)) {
      const value = Math.min(1, Math.max(0, weight * intensity * envelope));
      samples.push({ shape_key: shapeKey, frame, weight: round6(value) });
    }
  }
  return {
    beat_id: beat.id,
    channel: 'face',
    expression: name,
    request: {
      id: requestId(beat.id, 'face'),
      op: 'apply_shape_keys',
      params: {
        object: profile.rig.face_object,
        name_hint: nameHint,
        frame_start: base,
        frame_end: end,
        samples,
      },
    },
  };
}

// A validated actor plan in, bridge work out. Returns pose and
// shape-key layers ready to send, plus voice jobs the runner feeds to
// the voice pipeline (speech becomes a bridge request only after the
// pipeline produces a viseme artifact).
export function mapPlanToBridgeJobs(plan, profile) {
  const layers = [];
  const voiceJobs = [];
  for (const beat of plan.beats) {
    if (beat.body != null) layers.push(mapBodyBeat(beat, profile));
    if (beat.gaze != null) layers.push(mapGazeBeat(beat, profile));
    if (beat.face != null) layers.push(mapFaceBeat(beat, profile));
    if (beat.speech != null) {
      voiceJobs.push({
        beat_id: beat.id,
        text: beat.speech.text,
        delivery: beat.speech.delivery ?? 'neutral',
        frame_start: msToFrame(beat.at_ms, profile.fps),
        fps: profile.fps,
        object: profile.speech.object,
        name_hint: sanitizeNameHint(profile.speech.name_hint, 'speech'),
        voice: profile.speech.voice,
        lang: profile.speech.lang ?? 'a',
        speed: profile.speech.speed ?? 1.0,
        seed: profile.speech.seed ?? 0,
        backend: profile.speech.backend ?? 'kokoro',
        tts_opts: { ...(profile.speech.tts ?? {}) },
        stem: sanitizeNameHint(`${beat.id}_speech`, 'speech'),
      });
    }
  }
  return {
    profile: { name: profile.profile, version: profile.profile_version },
    layers,
    voice_jobs: voiceJobs,
  };
}

// For rigs with no shape keys (speech.mode 'bone'): the driver viseme
// weight (default 'mouth_open') becomes, frame for frame, an nlerp
// between each configured bone's neutral and peak pose, sent as one
// atomic perform_take. The voice pipeline still owns the timing; the
// profile still owns every number.
export function visemeArtifactToPoseRequest(artifact, speech, { requestId: id, object } = {}) {
  if (artifact?.kind !== 'animus_viseme_take') {
    throw new Error(`artifact kind is '${artifact?.kind}', expected 'animus_viseme_take'`);
  }
  const driver = speech.driver ?? 'mouth_open';
  const driven = new Map();
  const loudest = new Map();
  for (const sample of artifact.samples) {
    loudest.set(sample.frame, Math.max(loudest.get(sample.frame) ?? 0, sample.weight));
    if (sample.shape_key === driver) driven.set(sample.frame, sample.weight);
  }
  const frames = [...loudest.keys()].sort((a, b) => a - b);
  const bones = speech.bones;
  let stride = 1;
  while (frames.length
    && Math.ceil(frames.length / stride) * bones.length > MAX_SAMPLES) {
    stride += 1;
  }
  const samples = [];
  frames.forEach((frame, index) => {
    if (index % stride && frame !== frames[frames.length - 1]) return;
    const weight = Math.min(1, Math.max(0, driven.get(frame) ?? loudest.get(frame)));
    for (const bone of bones) {
      samples.push({
        bone: bone.bone,
        frame,
        rotation_quaternion: nlerp(bone.neutral, bone.peak, weight),
      });
    }
  });
  return {
    id: id ?? 'speech-take',
    op: 'perform_take',
    params: {
      object: object ?? speech.object,
      name_hint: artifact.name_hint,
      frame_start: artifact.frame_start,
      frame_end: artifact.frame_end,
      samples,
    },
  };
}

// Same field-for-field mapping speech_client.py uses: a viseme take
// artifact becomes one atomic apply_shape_keys request.
export function visemeArtifactToRequest(artifact, { requestId: id, object } = {}) {
  if (artifact?.kind !== 'animus_viseme_take') {
    throw new Error(`artifact kind is '${artifact?.kind}', expected 'animus_viseme_take'`);
  }
  return {
    id: id ?? 'speech-take',
    op: 'apply_shape_keys',
    params: {
      object: object ?? artifact.object,
      name_hint: artifact.name_hint,
      frame_start: artifact.frame_start,
      frame_end: artifact.frame_end,
      samples: artifact.samples,
    },
  };
}
