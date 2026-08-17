# Animus operator instructions

This repository is K-VRC. The `feat/animus-poc` branch is the first Project
Animus proof.

## Read first

1. `docs/animus/README.md`
2. `docs/animus/architecture.md`
3. `docs/animus/operator-runbook.md`
4. `docs/animus/contracts/actor-plan.schema.json`
5. `docs/animus/decision-log.md`

## Binding rules

- The model decides semantic intent. It never writes Python, raw keyframes,
  FCurves, bone names, or Blender operators.
- Validate every actor plan before a Blender mutation.
- The caller owns `suggest`, `preview`, or `perform` authority. A model cannot
  promote itself.
- Create a new Action and named NLA strip. Never replace the active Action.
- Return a mutation receipt with operator and artifact identifiers.
- Stop on ambiguous actor, missing rig profile, invalid plan, unsupported
  capability, or incomplete mutation receipt.
- Keep local-small, Codex Luna, Hermes Codex Luna, and deterministic behavior
  behind the same actor contract.

## Local commands

```powershell
npm ci
npm run test:animus
npm run build
npm run animus:poc -- --fallback-only --instruction "Wave to the viewer"
```

Start the small model only when the shared-resource rules permit it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-animus-model.ps1
```

Before a model download, shared server change, or GPU-heavy job, read and claim
the resource in `C:\Users\jainl\flux-work\boards\_claims.md`.

## Harness provenance

Use one exact operator name:

- `local-small`
- `local-large`
- `codex-luna`
- `hermes-codex-luna`
- `deterministic`
