"""Model-authored actor plans over the director's OpenAI-compatible endpoint.

Python port of src/animus/prompt.js + src/animus/providers/openaiCompatible.js
+ the retry policy in scripts/animus-director.mjs. Same endpoint, same
prompts, same provenance shape, so a plan authored here is
indistinguishable in the receipt from one authored by the Node director.

Environment (same names the director uses):

    ANIMUS_LLM_BASE_URL   default http://127.0.0.1:8081/v1
    ANIMUS_MODEL          default Qwen3-4B-Q4_K_M
    ANIMUS_LLM_ATTEMPTS   default 3 (the 4B sometimes emits bad JSON;
                          retries are caller policy)
    ANIMUS_TIMEOUT_MS     default 30000, per attempt

Stdlib only: urllib, no new dependencies.
"""

import json
import os
import sys
import urllib.error
import urllib.request

from .contract import (
    BODY_ACTIONS,
    BODY_STYLES,
    FACE_GLYPH_BROWS,
    FACE_GLYPH_EYES,
    FACE_GLYPH_MOODS,
    FACE_GLYPH_MOUTHS,
    SCHEMA_VERSION,
    validate_actor_plan,
)
from .fallback import deterministic_actor_plan

DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_MODEL = "Qwen3-4B-Q4_K_M"
DEFAULT_ATTEMPTS = 3
DEFAULT_TIMEOUT_MS = 30000

# Mirrors ACTOR_SYSTEM_PROMPT in src/animus/prompt.js field for field.
ACTOR_SYSTEM_PROMPT = f"""You plan editable character behavior.
Return one JSON object and no other text.
Use schema_version "{SCHEMA_VERSION}".
Use 1 to 8 short beats.
Allowed body actions: {', '.join(BODY_ACTIONS)}.
Allowed body styles: {', '.join(BODY_STYLES)}.
Each beat has id, at_ms, duration_ms, and at least one of body, gaze, face, face_glyph, speech.
Every channel must be an object or null. Never use a string as a channel value.
Body object fields are action, target, gesture, style, intensity.
Gaze object fields are target and intensity.
Face object fields are expression and intensity.
face_glyph composes the LED visor face directly. Use it instead of face when the moment wants a stylized visor. Fields are eyes, brows, mouth, mood, intensity, or text mode with only text and mood.
Allowed face_glyph eyes: {', '.join(FACE_GLYPH_EYES)}.
Allowed face_glyph brows: {', '.join(FACE_GLYPH_BROWS)}.
Allowed face_glyph mouths: {', '.join(FACE_GLYPH_MOUTHS)}.
Allowed face_glyph moods: {', '.join(FACE_GLYPH_MOODS)}.
face_glyph text is 1 to 6 characters of A-Z 0-9 ! ? % + - * # < > : = . _ drawn as LED text filling the visor.
When the instruction names the visor, LED text, a glyph face, or a word to display, use face_glyph for that beat, not face.
Never use face and face_glyph in the same beat.
Speech must be null or an object with non-empty text and non-empty delivery.
If the request supplies speech, copy that text exactly and use delivery "neutral" unless the instruction asks for another delivery.
Never emit Python, code, keyframes, FCurves, bone names, Blender operators, or tool calls.
Use semantic intent only. A deterministic motion layer will realize the plan.
Do not choose an execution authority or control level. The caller owns that decision.

Example:
{{"schema_version":"{SCHEMA_VERSION}","summary":"Greet the viewer","beats":[{{"id":"greet-1","at_ms":0,"duration_ms":1800,"body":{{"action":"gesture","gesture":"wave","style":"warm","intensity":0.7}},"gaze":{{"target":"camera","intensity":0.7}},"face":{{"expression":"warm_amused","intensity":0.5}},"speech":null}},{{"id":"greet-2","at_ms":1800,"duration_ms":1200,"body":null,"gaze":null,"face_glyph":{{"eyes":"happy_arc","mouth":"grin_rect","mood":"warm","intensity":0.8}},"speech":null}}]}}"""


def _nullable(*schemas):
    return {"anyOf": [{"type": "null"}, *schemas]}


# Decode-time structural schema for the plan, enforced by the server's
# grammar sampler (llama-server converts response_format json_schema to
# GBNF). WHY: at temperature 0.2 the 4B deterministically emits one
# extra closing brace after a nested channel object on many stage
# directions (board note 2026-08-20, reproduced 12/12 on 2026-08-21),
# and no retry or rephrasing fixes a deterministic failure. The grammar
# makes malformed JSON unrepresentable at decode time. This schema is
# STRUCTURAL only; the strict contract validator stays the authority on
# every semantic rule (ranges, text charsets, channel exclusivity).
# Mirrors ACTOR_PLAN_JSON_SCHEMA in src/animus/prompt.js.
ACTOR_PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "summary": {"type": "string"},
        "beats": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "at_ms": {"type": "integer"},
                    "duration_ms": {"type": "integer"},
                    "body": _nullable(
                        {
                            "type": "object",
                            "properties": {
                                "action": {"enum": list(BODY_ACTIONS)},
                                "target": {"type": "string"},
                                "gesture": {"type": "string"},
                                "style": {"enum": list(BODY_STYLES)},
                                "intensity": {"type": "number"},
                            },
                            "required": ["action"],
                            "additionalProperties": False,
                        }
                    ),
                    "gaze": _nullable(
                        {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string"},
                                "intensity": {"type": "number"},
                            },
                            "required": ["target"],
                            "additionalProperties": False,
                        }
                    ),
                    "face": _nullable(
                        {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"},
                                "intensity": {"type": "number"},
                            },
                            "required": ["expression"],
                            "additionalProperties": False,
                        }
                    ),
                    "face_glyph": _nullable(
                        {
                            "type": "object",
                            "properties": {
                                "eyes": {"enum": list(FACE_GLYPH_EYES)},
                                "brows": {"enum": list(FACE_GLYPH_BROWS)},
                                "mouth": {"enum": list(FACE_GLYPH_MOUTHS)},
                                "mood": {"enum": list(FACE_GLYPH_MOODS)},
                                "intensity": {"type": "number"},
                            },
                            "required": ["eyes"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "mood": {"enum": list(FACE_GLYPH_MOODS)},
                                "intensity": {"type": "number"},
                            },
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    ),
                    "speech": _nullable(
                        {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "delivery": {"type": "string"},
                            },
                            "required": ["text", "delivery"],
                            "additionalProperties": False,
                        }
                    ),
                },
                "required": ["id", "at_ms", "duration_ms"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["schema_version", "summary", "beats"],
    "additionalProperties": False,
}


def build_actor_user_prompt(request):
    """Mirrors buildActorUserPrompt in src/animus/prompt.js."""
    payload = {
        "instruction": str(request.get("instruction") or ""),
        "target": request.get("target"),
        "speech": request.get("speech"),
        "scene": request.get("scene") or {},
        "capabilities": request.get("capabilities") or {},
    }
    return json.dumps(payload, separators=(",", ":")) + "\n/no_think"


def extract_json(text):
    """First balanced JSON object; port of extractJson in openaiCompatible.js.

    Small models sometimes append a second object or trailing prose
    after a valid plan.
    """
    trimmed = str(text or "").strip()
    if trimmed.lower().startswith("```json"):
        trimmed = trimmed[7:].strip()
    elif trimmed.startswith("```"):
        trimmed = trimmed[3:].strip()
    if trimmed.endswith("```"):
        trimmed = trimmed[:-3].strip()
    start = trimmed.find("{")
    if start < 0:
        raise ValueError("provider returned no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(trimmed)):
        char = trimmed[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(trimmed[start : index + 1])
    raise ValueError("provider returned no complete JSON object")


def request_model_plan(request, base_url=None, model=None, timeout_ms=None):
    """One chat completion; same body the JS provider sends."""
    base = str(base_url or os.environ.get("ANIMUS_LLM_BASE_URL") or DEFAULT_BASE_URL)
    base = base.rstrip("/")
    model_name = model or os.environ.get("ANIMUS_MODEL") or DEFAULT_MODEL
    timeout = (
        timeout_ms
        if timeout_ms is not None
        else int(os.environ.get("ANIMUS_TIMEOUT_MS") or DEFAULT_TIMEOUT_MS)
    )
    body = json.dumps(
        {
            "model": model_name,
            "temperature": 0.2,
            "max_tokens": 600,
            # Grammar-enforced structure (see ACTOR_PLAN_JSON_SCHEMA for
            # why plain json_object was not enough for the 4B).
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "actor_plan",
                    "schema": ACTOR_PLAN_JSON_SCHEMA,
                },
            },
            "messages": [
                {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": build_actor_user_prompt(request)},
            ],
        }
    ).encode("utf-8")
    http_request = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout / 1000.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"provider returned HTTP {error.code}") from error
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise RuntimeError(f"provider unreachable: {error}") from error
    choices = payload.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    return extract_json(message.get("content"))


def model_actor_plan(request, authority, base_url=None, model=None,
                     attempts=None, timeout_ms=None, requester=None):
    """Model-authored plan with the director's retry-then-fallback policy.

    Every candidate goes through the strict contract validator. A plan
    that fails validation is recorded and retried; when every attempt
    fails, the deterministic fallback performs instead and the failure
    list lands LOUDLY on stderr and in provenance.prior_failures. The
    fallback is never silent.
    """
    if requester is None:
        requester = request_model_plan
    tries = (
        attempts
        if attempts is not None
        else max(1, int(os.environ.get("ANIMUS_LLM_ATTEMPTS") or DEFAULT_ATTEMPTS))
    )
    model_name = model or os.environ.get("ANIMUS_MODEL") or DEFAULT_MODEL
    failures = []
    for index in range(tries):
        operator = "local-small" if index == 0 else f"local-small-retry{index}"
        try:
            candidate = requester(
                request, base_url=base_url, model=model_name, timeout_ms=timeout_ms
            )
        except (RuntimeError, ValueError) as error:
            failures.append({"provider": operator, "reason": str(error)})
            continue
        checked = validate_actor_plan(candidate, authority)
        if checked["ok"]:
            plan = dict(checked["value"])
            plan["provenance"] = {
                "operator": operator,
                "model": model_name,
                "fallback": False,
            }
            if any(
                beat.get("face_glyph") is not None for beat in plan["beats"]
            ):
                # The model composed its own visor face; the receipt
                # says so explicitly (hand-authored glyphs are marked
                # "augmented" instead by the lanes that add them).
                plan["provenance"]["face_glyph"] = "model-authored"
            if failures:
                plan["provenance"]["prior_failures"] = failures
            return plan
        failures.append(
            {"provider": operator, "reason": "; ".join(checked["errors"])}
        )

    print(
        "animus_actor: MODEL BRAIN FELL BACK to the deterministic plan "
        f"after {tries} attempt(s): "
        + " | ".join(f"{item['provider']}: {item['reason']}" for item in failures),
        file=sys.stderr,
    )
    fallback_candidate = deterministic_actor_plan(request)
    checked = validate_actor_plan(fallback_candidate, authority)
    if not checked["ok"]:
        raise RuntimeError(
            "deterministic fallback violated the actor contract: "
            + "; ".join(checked["errors"])
        )
    plan = dict(checked["value"])
    plan["provenance"] = {
        "operator": "deterministic",
        "model": None,
        "fallback": True,
        "prior_failures": failures,
    }
    return plan
