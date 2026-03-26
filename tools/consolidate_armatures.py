"""
K-VRC Armature Consolidation Script
------------------------------------
Run this inside Blender: Scripting tab → Open → select this file → Run Script

What it does:
  - Finds the main armature (KVRCArmature, no number suffix)
  - Moves every animation from the 88 duplicate armatures onto the main one (NLA tracks)
  - Deletes the duplicate armature objects and duplicate meshes
  - Keeps all animation data — nothing is lost
"""

import bpy, re

# ── Find main armature (no number at the end of the name) ──────
main_arm = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE' and not re.search(r'\d+$', obj.name):
        main_arm = obj
        break

if not main_arm:
    print("ERROR: no main armature found (expected one named without a number suffix)")
    raise SystemExit

print(f"Main armature: {main_arm.name}")

if not main_arm.animation_data:
    main_arm.animation_data_create()

# ── Transfer animations from each duplicate to the main armature
seen  = set()
added = 0
to_delete = []

for obj in bpy.data.objects:
    if obj.type != 'ARMATURE' or obj == main_arm:
        continue
    to_delete.append(obj)

    if not (obj.animation_data and obj.animation_data.action):
        continue

    action = obj.animation_data.action
    base   = re.sub(r'\.\d+$', '', action.name)  # strip .001 / .002 suffixes
    action.use_fake_user = True                   # protect from garbage collection

    if base in seen:
        continue
    seen.add(base)

    # Rename to clean name if the clean name isn't already taken
    if action.name != base and base not in bpy.data.actions:
        action.name = base

    # Add as NLA track on main armature
    track       = main_arm.animation_data.nla_tracks.new()
    track.name  = action.name
    strip       = track.strips.new(action.name, 0, action)
    strip.action_frame_start = action.frame_range[0]
    strip.action_frame_end   = action.frame_range[1]
    added += 1
    print(f"  + {action.name}")

# ── Collect duplicate meshes (Love_Death_Robots001, 002, ...) ──
for obj in bpy.data.objects:
    if (obj.type == 'MESH'
            and obj.name.startswith('Love_Death_Robots')
            and obj.name != 'Love_Death_Robots'):
        to_delete.append(obj)

# ── Delete all duplicates ──────────────────────────────────────
bpy.ops.object.select_all(action='DESELECT')
for obj in to_delete:
    obj.select_set(True)
bpy.ops.object.delete()

print(f"\nDone!")
print(f"  {added} animations transferred to '{main_arm.name}'")
print(f"  {len(to_delete)} duplicate objects deleted")
print("  Open the NLA Editor to verify all clips are present")
print("  Then re-export: File > Export > glTF 2.0 (.glb)")
