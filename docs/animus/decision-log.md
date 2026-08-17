# Animus proof decision log

Decisions are append-only. A replacement decision names the decision it
supersedes.

## A-001: extend K-VRC

Date: 2026-08-17.

Decision: build the first Animus proof on a K-VRC feature branch instead of a
new repository.

Reason: K-VRC already has a character, baked clips, a face renderer, voice,
chat, and experimental animation code. It is the declared Animus precursor.
A second repository would duplicate the character and integration surface.

## A-002: one provider-neutral actor contract

Date: 2026-08-17.

Decision: local models, Codex Luna, Hermes Codex Luna, and deterministic logic
must all produce the same actor-plan schema.

Reason: model choice is an operational concern. Blender safety, embodiment
mapping, and editable output must not change when the planner changes.

## A-003: caller-owned authority

Date: 2026-08-17.

Decision: the caller supplies `suggest`, `preview`, or `perform`. The model
output has no authority field.

Reason: a planner should not decide whether its own output mutates a scene.

## A-004: Qwen3-4B Q4_K_M for the small local proof

Date: 2026-08-17.

Decision: use the official Apache-2.0 Qwen3-4B GGUF at Q4_K_M for the first
local planner.

Reason: the official source documents llama.cpp, Windows, OpenAI-compatible
serving, and Hermes. The file is 2.5GB and leaves much more room than the
existing 27B model.

Installed artifact:

- bytes: `2497280256`
- SHA-256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`
- local hash matches the official Hugging Face LFS object

## A-005: JSON plans instead of native tool calls

Date: 2026-08-17.

Decision: the small model returns one JSON plan. It does not call Blender MCP
tools directly.

Reason: our validator must own the trust boundary. This also avoids coupling
the proof to model-specific tool-call templates and parsers.

## A-006: Actions and NLA are the human boundary

Date: 2026-08-17.

Decision: generated motion becomes a new Action and a named NLA strip.

Reason: the animator can inspect, accept, blend, mute, move, edit, or delete a
take without destroying existing work.

## A-007: deterministic behavior is always present

Date: 2026-08-17.

Decision: invalid JSON, timeout, endpoint failure, and schema failure fall back
to deterministic behavior.

Reason: the character should remain usable when every model path is down.
