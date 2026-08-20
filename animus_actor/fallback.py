"""Deterministic actor plan, Python port of src/animus/fallback.js.

Same keyword table, same beat shape, same numbers. Given the same
request it must produce the same plan the JS fallback produces, so the
cross-language fixture test can hold both to one contract.
"""

from .contract import SCHEMA_VERSION


def _includes_any(text, words):
    return any(word in text for word in words)


def deterministic_actor_plan(request):
    """One contract-valid beat from an instruction. No model, no GPU."""
    instruction = str(request.get("instruction") or "").strip()
    lower = instruction.lower()
    target = str(request.get("target") or "camera").strip() or "camera"

    body = {"action": "gesture", "gesture": "talk", "style": "neutral", "intensity": 0.45}
    if _includes_any(lower, ("wave", "hello", "greet")):
        body = {"action": "gesture", "gesture": "wave", "style": "warm", "intensity": 0.65}
    elif _includes_any(lower, ("walk", "move", "go to")):
        body = {"action": "walk_to", "target": target, "style": "neutral", "intensity": 0.5}
    elif _includes_any(lower, ("look", "face", "turn")):
        body = {"action": "turn_to", "target": target, "style": "careful", "intensity": 0.4}
    elif _includes_any(lower, ("wait", "stop", "hold")):
        body = {"action": "wait", "style": "neutral", "intensity": 0.2}

    speech = request.get("speech")
    beat = {
        "id": "fallback-1",
        "at_ms": 0,
        "duration_ms": 1800,
        "body": body,
        "gaze": {"target": target, "intensity": 0.5},
        "face": {"expression": "neutral_idle", "intensity": 0.4},
        "speech": (
            {"text": str(speech)[:500], "delivery": "neutral"} if speech else None
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "summary": instruction or "Use the deterministic idle behavior.",
        "beats": [beat],
    }
