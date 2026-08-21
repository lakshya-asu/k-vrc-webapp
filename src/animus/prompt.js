import {
  BODY_ACTIONS,
  BODY_STYLES,
  FACE_GLYPH_BROWS,
  FACE_GLYPH_EYES,
  FACE_GLYPH_MOODS,
  FACE_GLYPH_MOUTHS,
  SCHEMA_VERSION,
} from './contract.js';

export const ACTOR_SYSTEM_PROMPT = `You plan editable character behavior.
Return one JSON object and no other text.
Use schema_version "${SCHEMA_VERSION}".
Use 1 to 8 short beats.
Allowed body actions: ${BODY_ACTIONS.join(', ')}.
Allowed body styles: ${BODY_STYLES.join(', ')}.
Each beat has id, at_ms, duration_ms, and at least one of body, gaze, face, face_glyph, speech.
Every channel must be an object or null. Never use a string as a channel value.
Body object fields are action, target, gesture, style, intensity.
Gaze object fields are target and intensity.
Face object fields are expression and intensity.
face_glyph composes the LED visor face directly. Use it instead of face when the moment wants a stylized visor. Fields are eyes, brows, mouth, mood, intensity, or text mode with only text and mood.
Allowed face_glyph eyes: ${FACE_GLYPH_EYES.join(', ')}.
Allowed face_glyph brows: ${FACE_GLYPH_BROWS.join(', ')}.
Allowed face_glyph mouths: ${FACE_GLYPH_MOUTHS.join(', ')}.
Allowed face_glyph moods: ${FACE_GLYPH_MOODS.join(', ')}.
face_glyph text is 1 to 6 characters of A-Z 0-9 ! ? % + - * # < > : = . _ drawn as LED text filling the visor.
When the instruction names the visor, LED text, a glyph face, or a word to display, use face_glyph for that beat, not face.
Never use face and face_glyph in the same beat.
Speech must be null or an object with non-empty text and non-empty delivery.
If the request supplies speech, copy that text exactly and use delivery "neutral" unless the instruction asks for another delivery.
Never emit Python, code, keyframes, FCurves, bone names, Blender operators, or tool calls.
Use semantic intent only. A deterministic motion layer will realize the plan.
Do not choose an execution authority or control level. The caller owns that decision.

Example:
{"schema_version":"${SCHEMA_VERSION}","summary":"Greet the viewer","beats":[{"id":"greet-1","at_ms":0,"duration_ms":1800,"body":{"action":"gesture","gesture":"wave","style":"warm","intensity":0.7},"gaze":{"target":"camera","intensity":0.7},"face":{"expression":"warm_amused","intensity":0.5},"speech":null},{"id":"greet-2","at_ms":1800,"duration_ms":1200,"body":null,"gaze":null,"face_glyph":{"eyes":"happy_arc","mouth":"grin_rect","mood":"warm","intensity":0.8},"speech":null}]}`;

// Decode-time structural schema for the plan, enforced by the server's
// grammar sampler (llama-server converts response_format json_schema to
// GBNF). WHY: at temperature 0.2 the 4B deterministically emits one
// extra closing brace after a nested channel object on many stage
// directions (board note 2026-08-20, reproduced 12/12 on 2026-08-21),
// and no retry or rephrasing fixes a deterministic failure. The grammar
// makes malformed JSON unrepresentable at decode time. This schema is
// STRUCTURAL only; the strict contract validator stays the authority on
// every semantic rule (ranges, text charsets, channel exclusivity).
const nullable = (...schemas) => ({ anyOf: [{ type: 'null' }, ...schemas] });

export const ACTOR_PLAN_JSON_SCHEMA = {
  type: 'object',
  properties: {
    schema_version: { const: SCHEMA_VERSION },
    summary: { type: 'string' },
    beats: {
      type: 'array',
      minItems: 1,
      maxItems: 8,
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          at_ms: { type: 'integer' },
          duration_ms: { type: 'integer' },
          body: nullable({
            type: 'object',
            properties: {
              action: { enum: [...BODY_ACTIONS] },
              target: { type: 'string' },
              gesture: { type: 'string' },
              style: { enum: [...BODY_STYLES] },
              intensity: { type: 'number' },
            },
            required: ['action'],
            additionalProperties: false,
          }),
          gaze: nullable({
            type: 'object',
            properties: {
              target: { type: 'string' },
              intensity: { type: 'number' },
            },
            required: ['target'],
            additionalProperties: false,
          }),
          face: nullable({
            type: 'object',
            properties: {
              expression: { type: 'string' },
              intensity: { type: 'number' },
            },
            required: ['expression'],
            additionalProperties: false,
          }),
          face_glyph: nullable(
            {
              type: 'object',
              properties: {
                eyes: { enum: [...FACE_GLYPH_EYES] },
                brows: { enum: [...FACE_GLYPH_BROWS] },
                mouth: { enum: [...FACE_GLYPH_MOUTHS] },
                mood: { enum: [...FACE_GLYPH_MOODS] },
                intensity: { type: 'number' },
              },
              required: ['eyes'],
              additionalProperties: false,
            },
            {
              type: 'object',
              properties: {
                text: { type: 'string' },
                mood: { enum: [...FACE_GLYPH_MOODS] },
                intensity: { type: 'number' },
              },
              required: ['text'],
              additionalProperties: false,
            },
          ),
          speech: nullable({
            type: 'object',
            properties: {
              text: { type: 'string' },
              delivery: { type: 'string' },
            },
            required: ['text', 'delivery'],
            additionalProperties: false,
          }),
        },
        required: ['id', 'at_ms', 'duration_ms'],
        additionalProperties: false,
      },
    },
  },
  required: ['schema_version', 'summary', 'beats'],
  additionalProperties: false,
};

export function buildActorUserPrompt(request) {
  return `${JSON.stringify({
    instruction: String(request.instruction ?? ''),
    target: request.target ?? null,
    speech: request.speech ?? null,
    scene: request.scene ?? {},
    capabilities: request.capabilities ?? {},
  })}\n/no_think`;
}
