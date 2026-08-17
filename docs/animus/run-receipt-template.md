# Animus run receipt template

Store one receipt for every retained generated take.

```json
{
  "schema_version": "0.1",
  "timestamp_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "source_commit": "full Git commit",
  "actor_id": "kvrc",
  "request_id": "user-provided or generated identifier",
  "control_level": "suggest",
  "operator": "local-small",
  "model": "Qwen3-4B-Q4_K_M",
  "actor_plan_sha256": "hash of canonical plan JSON",
  "embodiment_profile": "kvrc-0.1",
  "blender": {
    "version": "not-run",
    "action_name": null,
    "nla_track": null,
    "nla_strip": null
  },
  "validation": {
    "ok": true,
    "warnings": []
  },
  "artifacts": []
}
```

Never put credentials, private prompts, or unrelated scene data in a receipt.
