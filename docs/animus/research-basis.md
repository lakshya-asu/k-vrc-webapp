# Research basis

Date: 2026-08-17.

The long research reports live in the shared Janus research directory:

- `C:\Users\jainl\flux-work\janus\research\animus-blender-mcp.md`
- `C:\Users\jainl\flux-work\janus\research\animus-motion-generation.md`
- `C:\Users\jainl\flux-work\janus\research\animus-rigging-and-voice.md`

This file records the decisions that the proof imports from them.

## Blender bridge

- [PRIMARY][high] `ahujasid/blender-mcp` has an active add-on plus socket
  architecture and runs queued `bpy` work through `bpy.app.timers` on the main
  thread.
- [PRIMARY][high] Its MCP surface has no typed pose-bone, Action, NLA,
  armature, constraint, driver, or shape-key mutation tools.
- [PRIMARY][high] Animation is possible only through unrestricted Python
  execution.
- [INFERRED][high] Animus will use a narrow typed actor surface and will not
  expose raw Python to the actor model.

Sources:

- <https://github.com/ahujasid/blender-mcp>
- <https://docs.blender.org/api/current/info_gotchas_threading.html>
- <https://docs.blender.org/api/current/bpy.app.timers.html>

## Motion

- [PRIMARY][high] HumanML3D motion derives from AMASS. AMASS is restricted to
  noncommercial scientific research, education, and noncommercial art.
- [PRIMARY][high] Kimodo SOMA-RP v1.1 emits rotations, positions, foot
  contacts, and BVH. Its documented CPU-text-encoder configuration uses less
  than 3GB GPU memory.
- [INFERRED][high] Licensed clips plus procedural state, IK, gaze, and blending
  are the first product tier. Kimodo is an optional serialized GPU job.

Sources:

- <https://amass.is.tue.mpg.de/license.html>
- <https://github.com/EricGuo5513/HumanML3D>
- <https://github.com/nv-tlabs/kimodo>
- <https://www.ianxmason.com/100style/>
- <https://mocap.cs.cmu.edu/>

## Rigging and embodiment

- [PRIMARY][high] Rigify requires a positioned meta-rig. Mixamo asks a human
  to place landmarks. Neither is a universal unattended auto-rigger.
- [PRIMARY][high] UniRig publishes weights and documents an 8GB GPU minimum.
- [PRIMARY][high] SkinTokens publishes weights and documents a 14GB inference
  minimum.
- [INFERRED][high] The proof starts from the already-rigged K-VRC character.
  Learned rigging stays an optional serialized job.
- [INFERRED][high] 2D cutouts, pixel art, Grease Pencil, and 3D rigs require
  separate embodiment adapters behind one semantic actor contract.

Sources:

- <https://docs.blender.org/manual/en/latest/addons/rigging/rigify/index.html>
- <https://helpx.adobe.com/creative-cloud/help/mixamo-rigging-animation.html>
- <https://github.com/VAST-AI-Research/UniRig>
- <https://github.com/VAST-AI-Research/SkinTokens>
- <https://github.com/facebookresearch/AnimatedDrawings>

## Voice

- [PRIMARY][high] Kokoro-82M is Apache-2.0, has 82 million parameters, and
  publishes local usage examples.
- [PRIMARY][high] Piper is local and actively maintained, but its current
  engine is GPL and every voice model has separate terms.
- [PRIMARY][high] Rhubarb is MIT and produces timed mouth cues from final
  speech audio.
- [INFERRED][high] The first voice path is CPU-local Kokoro plus Rhubarb, with
  Piper as a license-reviewed fallback.

Sources:

- <https://huggingface.co/hexgrad/Kokoro-82M>
- <https://github.com/OHF-Voice/piper1-gpl>
- <https://github.com/DanielSWolf/rhubarb-lip-sync>
