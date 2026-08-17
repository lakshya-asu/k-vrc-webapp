# Inference and operator profiles

Status: 2026-08-17.

Every profile produces the same actor-plan contract. None writes raw Blender
state. The executor does not care which planner produced a valid plan.

## Profile 1: local-small

Default model for the proof: `Qwen/Qwen3-4B-GGUF`, Q4_K_M.

Primary-source facts:

- The official model card lists 4.0 billion parameters.
- The official GGUF repository lists the Q4_K_M file at 2.5GB.
- The repository is Apache-2.0.
- Qwen documents llama.cpp, Windows installation, an OpenAI-compatible server,
  and Hermes configuration for this exact GGUF.
- Native context is 32,768 tokens. The proof starts at 8,192 because actor
  requests are small and a larger cache is unnecessary.

Sources:

- <https://huggingface.co/Qwen/Qwen3-4B-GGUF>
- <https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md>
- <https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md>

The proof does not use model-native tool calls. It asks for one JSON actor
plan. This keeps execution behind our validator and avoids coupling safety to
a chat-template parser.

Run it on port 8081. Port 8080 remains reserved for the existing local fleet
profile.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-animus-model.ps1
```

If another process owns the GPU, a slower CPU-only check is available:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-animus-model.ps1 -GpuLayers 0 -ContextSize 4096
$env:ANIMUS_TIMEOUT_MS = "60000"
npm run animus:poc -- --instruction "Wave to the viewer"
```

The model file is local and is not committed to Git.

Installed file receipt:

- Path: `C:\Users\jainl\local-llm\models\Qwen3-4B-Q4_K_M.gguf`
- Bytes: `2497280256`
- SHA-256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`
- The byte count and SHA-256 match the official Hugging Face LFS object
  checked on 2026-08-17.

## Profile 2: Codex Luna operator

A dedicated Codex Luna task can operate the proof when requested. It reads the
same architecture and operator runbook, creates or validates an actor plan,
then calls the typed Blender MCP tools.

This is a human-started operator tier. The K-VRC web app does not embed Codex
credentials or try to start a Codex task itself.

Required behavior:

1. Read the current scene and embodiment capabilities.
2. Produce an actor plan that passes the checked-in contract.
3. Respect the caller's control level.
4. Create a new Action and NLA strip.
5. Return identifiers and validation warnings.
6. Stop on ambiguity or a failed mutation receipt.

Record `operator: codex-luna` in provenance.

## Profile 3: Hermes with Codex Luna

Hermes can run the same on-demand operator loop. It must load this repository's
operator runbook and use the same MCP surface. It is an alternate harness, not
a separate animation architecture.

Record `operator: hermes-codex-luna` in provenance.

Hermes must not silently switch its unrelated work lanes or global default
model. Animus should use a named profile or task-local model selection.

## Profile 4: deterministic

The deterministic planner handles a small vocabulary with no model:

- idle
- wave or greet
- walk or move to a named target
- look, face, or turn to a named target
- wait or stop
- speak while using a neutral talk gesture

This profile is always present. It is also the final fallback after invalid
JSON, timeout, endpoint failure, or schema failure.

Record `operator: deterministic` and `fallback: true` in provenance.

## Profile 5: local-large

The existing Qwen3.8-27B profile remains an optional higher-quality planner.
It is not required for the proof. It nearly fills the 16GB GPU in the measured
fleet configuration, so it conflicts with learned motion and learned rigging
jobs.

Use the same JSON contract and validator. Do not create a second large-model
code path.

## Routing order

For an interactive local proof:

1. `local-small`
2. deterministic fallback

For a directed animation session:

1. Codex Luna or Hermes Codex Luna
2. `local-small`
3. deterministic fallback

For a scripted test or continuous integration:

1. deterministic only

## What remains unverified

- Qwen3-4B quality and latency on this exact machine need a recorded smoke
  test after the download completes.
- GPU memory use depends on llama.cpp build, context, cache types, and request.
  No memory figure is asserted before measurement.
- Codex Luna and Hermes need an end-to-end Blender MCP trial after the typed
  add-on exists.
- No provider is yet approved for unattended `perform` mode.
