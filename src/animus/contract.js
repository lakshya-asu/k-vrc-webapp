export const SCHEMA_VERSION = '0.1';

export const CONTROL_LEVELS = Object.freeze([
  'suggest',
  'preview',
  'perform',
]);

export const BODY_ACTIONS = Object.freeze([
  'idle',
  'walk_to',
  'turn_to',
  'gesture',
  'interact',
  'wait',
]);

export const BODY_STYLES = Object.freeze([
  'neutral',
  'warm',
  'cold',
  'confident',
  'careful',
  'energetic',
  'tired',
]);

const TOP_LEVEL_KEYS = new Set(['schema_version', 'summary', 'beats']);
const BEAT_KEYS = new Set(['id', 'at_ms', 'duration_ms', 'body', 'gaze', 'face', 'speech']);
const BODY_KEYS = new Set(['action', 'target', 'gesture', 'style', 'intensity']);
const GAZE_KEYS = new Set(['target', 'intensity']);
const FACE_KEYS = new Set(['expression', 'intensity']);
const SPEECH_KEYS = new Set(['text', 'delivery']);
const FORBIDDEN_KEYS = new Set([
  'code',
  'python',
  'keyframe',
  'keyframes',
  'raw_keyframes',
  'fcurve',
  'fcurves',
]);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function addUnknownKeyErrors(value, allowed, path, errors) {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) errors.push(`${path}.${key} is not allowed`);
  }
}

function findForbiddenKeys(value, path, errors) {
  if (Array.isArray(value)) {
    value.forEach((item, index) => findForbiddenKeys(item, `${path}[${index}]`, errors));
    return;
  }
  if (!isObject(value)) return;

  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_KEYS.has(key.toLowerCase())) {
      errors.push(`${path}.${key} is forbidden`);
    }
    findForbiddenKeys(child, `${path}.${key}`, errors);
  }
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function validateIntensity(value, path, errors) {
  if (!isFiniteNumber(value) || value < 0 || value > 1) {
    errors.push(`${path} must be a number from 0 to 1`);
  }
}

function validateOptionalText(value, path, errors, maxLength = 120) {
  if (value === undefined) return;
  if (typeof value !== 'string' || value.trim().length === 0 || value.length > maxLength) {
    errors.push(`${path} must be a non-empty string no longer than ${maxLength} characters`);
  }
}

function validateBody(body, path, errors) {
  if (!isObject(body)) {
    errors.push(`${path} must be an object or null`);
    return;
  }
  addUnknownKeyErrors(body, BODY_KEYS, path, errors);
  if (!BODY_ACTIONS.includes(body.action)) {
    errors.push(`${path}.action must be one of: ${BODY_ACTIONS.join(', ')}`);
  }
  validateOptionalText(body.target, `${path}.target`, errors);
  validateOptionalText(body.gesture, `${path}.gesture`, errors, 60);
  if (body.style !== undefined && !BODY_STYLES.includes(body.style)) {
    errors.push(`${path}.style must be one of: ${BODY_STYLES.join(', ')}`);
  }
  if (body.intensity !== undefined) validateIntensity(body.intensity, `${path}.intensity`, errors);

  if (['walk_to', 'turn_to', 'interact'].includes(body.action) && !body.target) {
    errors.push(`${path}.target is required for ${body.action}`);
  }
  if (body.action === 'gesture' && !body.gesture) {
    errors.push(`${path}.gesture is required for gesture`);
  }
}

function validateGaze(gaze, path, errors) {
  if (!isObject(gaze)) {
    errors.push(`${path} must be an object or null`);
    return;
  }
  addUnknownKeyErrors(gaze, GAZE_KEYS, path, errors);
  validateOptionalText(gaze.target, `${path}.target`, errors);
  if (!gaze.target) errors.push(`${path}.target is required`);
  if (gaze.intensity !== undefined) validateIntensity(gaze.intensity, `${path}.intensity`, errors);
}

function validateFace(face, path, errors) {
  if (!isObject(face)) {
    errors.push(`${path} must be an object or null`);
    return;
  }
  addUnknownKeyErrors(face, FACE_KEYS, path, errors);
  validateOptionalText(face.expression, `${path}.expression`, errors, 80);
  if (!face.expression) errors.push(`${path}.expression is required`);
  if (face.intensity !== undefined) validateIntensity(face.intensity, `${path}.intensity`, errors);
}

function validateSpeech(speech, path, errors) {
  if (!isObject(speech)) {
    errors.push(`${path} must be an object or null`);
    return;
  }
  addUnknownKeyErrors(speech, SPEECH_KEYS, path, errors);
  validateOptionalText(speech.text, `${path}.text`, errors, 500);
  validateOptionalText(speech.delivery, `${path}.delivery`, errors, 80);
  if (!speech.text) errors.push(`${path}.text is required`);
}

export function validateActorPlan(candidate, authority = {}) {
  const errors = [];
  if (!isObject(candidate)) {
    return { ok: false, errors: ['plan must be an object'], value: null };
  }

  findForbiddenKeys(candidate, 'plan', errors);
  addUnknownKeyErrors(candidate, TOP_LEVEL_KEYS, 'plan', errors);

  if (candidate.schema_version !== SCHEMA_VERSION) {
    errors.push(`plan.schema_version must be ${SCHEMA_VERSION}`);
  }
  validateOptionalText(candidate.summary, 'plan.summary', errors, 200);

  if (!Array.isArray(candidate.beats) || candidate.beats.length === 0) {
    errors.push('plan.beats must contain at least one beat');
  } else if (candidate.beats.length > 8) {
    errors.push('plan.beats may contain at most 8 beats');
  } else {
    candidate.beats.forEach((beat, index) => {
      const path = `plan.beats[${index}]`;
      if (!isObject(beat)) {
        errors.push(`${path} must be an object`);
        return;
      }
      addUnknownKeyErrors(beat, BEAT_KEYS, path, errors);
      validateOptionalText(beat.id, `${path}.id`, errors, 60);
      if (!beat.id) errors.push(`${path}.id is required`);
      if (!Number.isInteger(beat.at_ms) || beat.at_ms < 0 || beat.at_ms > 60000) {
        errors.push(`${path}.at_ms must be an integer from 0 to 60000`);
      }
      if (!Number.isInteger(beat.duration_ms) || beat.duration_ms < 100 || beat.duration_ms > 30000) {
        errors.push(`${path}.duration_ms must be an integer from 100 to 30000`);
      }

      const channels = ['body', 'gaze', 'face', 'speech'].filter((key) => beat[key] != null);
      if (channels.length === 0) errors.push(`${path} must use at least one actor channel`);
      if (beat.body != null) validateBody(beat.body, `${path}.body`, errors);
      if (beat.gaze != null) validateGaze(beat.gaze, `${path}.gaze`, errors);
      if (beat.face != null) validateFace(beat.face, `${path}.face`, errors);
      if (beat.speech != null) validateSpeech(beat.speech, `${path}.speech`, errors);
    });
  }

  const controlLevel = authority.controlLevel ?? 'suggest';
  if (!CONTROL_LEVELS.includes(controlLevel)) {
    errors.push(`authority.controlLevel must be one of: ${CONTROL_LEVELS.join(', ')}`);
  }
  validateOptionalText(authority.actorId, 'authority.actorId', errors, 80);
  if (!authority.actorId) errors.push('authority.actorId is required');

  if (errors.length > 0) return { ok: false, errors, value: null };

  return {
    ok: true,
    errors: [],
    value: {
      schema_version: SCHEMA_VERSION,
      actor_id: authority.actorId,
      control_level: controlLevel,
      summary: candidate.summary.trim(),
      beats: candidate.beats,
    },
  };
}
