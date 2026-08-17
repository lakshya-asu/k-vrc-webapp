# Codex Luna operator task

Open a Codex Luna task in this worktree and use this request:

```text
Read AGENTS.md and docs/animus/operator-runbook.md first.
Operate K-VRC through the Animus actor-plan contract.
Start in suggest mode unless I explicitly select preview or perform.
Inspect the current scene and actor capabilities before planning.
Never emit or execute raw Python, keyframes, FCurves, bone names, or Blender
operators from the language model.
Use typed Blender MCP operations only.
Create a new Action and named NLA strip for every retained take.
Return the plan, operator provenance, created identifiers, and validation
warnings.

My instruction: <replace this line>
```

The task cannot execute Blender until the typed Animus MCP add-on from P1 is
installed. Before that milestone, it should return and validate plans only.
