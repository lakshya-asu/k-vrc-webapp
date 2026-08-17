# Animus proof architecture

Status: design baseline for `feat/animus-poc`, 2026-08-17.

## Design rules

1. The actor brain decides what to do.
2. The motion layer decides how the rig does it.
3. A model never writes Python, keyframes, FCurves, bone names, or Blender
   operators.
4. Execution authority belongs to the caller. A provider cannot promote a
   suggestion into an applied scene change.
5. Every generated take is reversible and has provenance.
6. Every provider can fail without making the character unusable.

## Flow

```text
User, script, or operator
        |
        v
Actor request
  instruction, scene summary, actor capabilities, caller control level
        |
        v
ActorDirector
  local-small, Codex Luna, Hermes Luna, or deterministic fallback
        |
        v
Actor-plan validator
  schema, ranges, allowlists, forbidden raw-control fields
        |
        v
Motion realizer
  semantic action -> clip, IK, constraint, expression, speech
        |
        v
Blender MCP adapter
  new Action -> named NLA strip -> validation receipt
        |
        v
Animator
  accept, blend, mute, move, edit, or delete
```

## Authority model

The caller supplies one control level:

- `suggest`: return the actor plan only.
- `preview`: build a temporary take for review. Do not replace active work.
- `perform`: apply a validated take through the Blender adapter.

The plan emitted by a model has no control-level field. The validator adds the
caller's level after validation. Unknown top-level fields are rejected. This
prevents a model from granting itself more authority.

## Actor plan

An actor plan contains one to eight timed beats. A beat may use body, gaze,
face, and speech channels. Body actions remain semantic:

- `idle`
- `walk_to`
- `turn_to`
- `gesture`
- `interact`
- `wait`

The contract rejects keys named `code`, `python`, `keyframe`, `keyframes`,
`raw_keyframes`, `fcurve`, or `fcurves` at any depth.

The checked-in JSON Schema is the interchange form. The JavaScript validator
is the executable boundary for the proof.

## Planner providers

All planners implement one method:

```js
await provider.plan(request, { signal })
```

The provider returns an untrusted candidate. `ActorDirector` validates it,
attaches provenance, and returns it. A timeout, HTTP failure, invalid JSON, or
contract failure moves to the next provider. If all providers fail, a small
deterministic planner returns a safe plan.

## Motion realization

The proof starts with K-VRC's existing animation clips. A semantic gesture is
mapped to a known clip. Look-at and simple contacts use constraints. Speech
uses a local or configured TTS provider. Lip sync becomes a separate face
track.

The next implementation slice adds a Blender add-on with typed operations:

- inspect the actor and capability map
- create a new Action
- apply validated pose samples
- create and name an NLA strip
- add or animate allowlisted constraints
- apply face or viseme values
- validate and return created identifiers

The add-on must execute `bpy` work on Blender's main thread. Socket threads may
only parse and queue requests.

## Artifact and provenance rules

Every realized take records:

- actor ID
- request ID
- control level
- operator name
- model name or deterministic fallback
- source plan hash
- embodiment profile version
- generated Action and NLA names
- validation warnings

The proof never overwrites the active Action. Generated content uses a new,
named NLA strip.

## Failure policy

- Invalid model output is discarded.
- Missing target or unsupported capability becomes a visible validation error.
- An unavailable model falls back to deterministic behavior.
- An unavailable motion generator falls back to licensed clips.
- An unavailable voice falls back to text and silent facial idle.
- Blender connection loss returns a failed receipt and does not retry a scene
  mutation blindly.

## Hardware policy

The small actor model and K-VRC clip realization are the default proof path.
Learned motion and learned rigging are separate jobs. They may unload the actor
model while they use the GPU. No architecture component assumes that the actor
LLM and a large motion model are resident together.
