import { BODY_ACTIONS, BODY_STYLES, SCHEMA_VERSION } from './contract.js';

export const ACTOR_SYSTEM_PROMPT = `You plan editable character behavior.
Return one JSON object and no other text.
Use schema_version "${SCHEMA_VERSION}".
Use 1 to 8 short beats.
Allowed body actions: ${BODY_ACTIONS.join(', ')}.
Allowed body styles: ${BODY_STYLES.join(', ')}.
Each beat has id, at_ms, duration_ms, and at least one of body, gaze, face, speech.
Every channel must be an object or null. Never use a string as a channel value.
Body object fields are action, target, gesture, style, intensity.
Gaze object fields are target and intensity.
Face object fields are expression and intensity.
Speech must be null or an object with non-empty text and non-empty delivery.
If the request supplies speech, copy that text exactly and use delivery "neutral" unless the instruction asks for another delivery.
Never emit Python, code, keyframes, FCurves, bone names, Blender operators, or tool calls.
Use semantic intent only. A deterministic motion layer will realize the plan.
Do not choose an execution authority or control level. The caller owns that decision.

Example:
{"schema_version":"${SCHEMA_VERSION}","summary":"Greet the viewer","beats":[{"id":"greet-1","at_ms":0,"duration_ms":1800,"body":{"action":"gesture","gesture":"wave","style":"warm","intensity":0.7},"gaze":{"target":"camera","intensity":0.7},"face":{"expression":"warm_amused","intensity":0.5},"speech":null}]}`;

export function buildActorUserPrompt(request) {
  return `${JSON.stringify({
    instruction: String(request.instruction ?? ''),
    target: request.target ?? null,
    speech: request.speech ?? null,
    scene: request.scene ?? {},
    capabilities: request.capabilities ?? {},
  })}\n/no_think`;
}
