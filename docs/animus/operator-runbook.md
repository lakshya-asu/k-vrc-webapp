# Animus operator runbook

This file is the common brief for Codex Luna, Hermes Codex Luna, and a human
operator.

## Goal

Turn one user instruction into an editable K-VRC take. Never write raw
keyframes from the language model.

## Inputs

- user instruction
- caller-selected control level
- scene summary
- actor capability profile
- optional target and speech text

## Procedure

1. Inspect the current actor, scene, frame range, active Action, and NLA tracks.
2. Read `docs/animus/contracts/actor-plan.schema.json`.
3. Produce a semantic actor plan.
4. Validate it before any scene mutation.
5. In `suggest` mode, return the plan and stop.
6. In `preview` mode, create a new temporary Action and named NLA strip.
7. In `perform` mode, create a new Action and named NLA strip. Never replace
   the active Action.
8. Apply supported channels through typed MCP tools.
9. Validate the result and return a receipt.

## Receipt

Return:

```json
{
  "ok": true,
  "actor_id": "kvrc",
  "operator": "codex-luna",
  "action_name": "ANIMUS_kvrc_greet_001",
  "nla_track": "ANIMUS_body",
  "nla_strip": "greet_001",
  "warnings": []
}
```

## Stop conditions

Stop without mutating the scene when:

- the target actor is ambiguous
- the rig profile is missing
- the plan fails validation
- the requested action is unsupported
- the current file has an unsaved conflict that the user must resolve
- the MCP add-on returns an incomplete mutation receipt

Do not retry a mutation unless the receipt proves that no artifact was created.

## Provenance names

- `local-small`
- `local-large`
- `codex-luna`
- `hermes-codex-luna`
- `deterministic`

Use these exact values in receipts and NLA metadata.
