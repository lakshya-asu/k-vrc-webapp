# Animus proof of concept

This branch turns K-VRC into the first Animus test character.

The proof has one rule: every brain produces the same small actor plan. The
brain never writes keyframes and never receives execution authority. A
validator checks the plan. A deterministic motion layer maps semantic intent
to clips, constraints, Actions, and NLA strips.

## Documents

- [Architecture](architecture.md)
- [Inference and operator profiles](inference-profiles.md)
- [Proof plan and acceptance tests](poc-plan.md)
- [Current K-VRC truth pass](truth-pass.md)
- [Research basis](research-basis.md)
- [Decision log](decision-log.md)
- [Actor-plan JSON Schema](contracts/actor-plan.schema.json)
- [Operator runbook](operator-runbook.md)
- [Codex Luna task prompt](codex-luna-task.md)
- [Run receipt template](run-receipt-template.md)

## Current code

- `src/animus/contract.js` validates the actor plan and injects caller-owned
  authority.
- `src/animus/director.js` tries configured planners and falls back to a
  deterministic plan.
- `src/animus/providers/openaiCompatible.js` talks to a local or remote
  OpenAI-compatible chat endpoint.
- `scripts/animus-poc.mjs` produces a plan from the command line.
- `scripts/start-animus-model.ps1` starts the small local model on port 8081.

## First local check

The fallback path needs no model:

```powershell
npm run animus:poc -- --fallback-only --instruction "Wave to the viewer"
```

The small-model path expects the server from `start-animus-model.ps1`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-animus-model.ps1
npm run animus:poc -- --instruction "Wave, look at the camera, and say hello"
```

The result is a plan. It does not alter Blender yet.

## Proof boundary

The first milestone proves planning, validation, provenance, fallback, and an
editable Blender result. It does not prove general animation generation,
automatic rigging, or unsupervised scene control.
