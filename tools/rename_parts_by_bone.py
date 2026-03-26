"""
K-VRC Rename Mesh Parts by Dominant Bone
-----------------------------------------
Run AFTER separating the mesh by loose parts.

Each separated piece still carries vertex group data from the rig.
This script renames every piece to match whichever bone influences
it the most — so "Love_Death_Robots.047" becomes "Head" or "LeftArm", etc.

Run inside Blender: Scripting tab → Open → select this file → Run Script
"""

import bpy

renamed = 0
skipped = 0

for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    if not obj.name.startswith('Love_Death_Robots'):
        continue
    if not obj.vertex_groups:
        skipped += 1
        continue

    # Count how many vertices each vertex group influences
    group_counts = {}
    for v in obj.data.vertices:
        for g in v.groups:
            gname = obj.vertex_groups[g.group].name
            group_counts[gname] = group_counts.get(gname, 0) + 1

    if not group_counts:
        skipped += 1
        continue

    dominant = max(group_counts, key=group_counts.get)
    old_name = obj.name
    obj.name = dominant          # Blender auto-appends .001 if name is taken
    print(f"  {old_name}  →  {obj.name}")
    renamed += 1

print(f"\nDone — {renamed} parts renamed, {skipped} skipped (no vertex groups)")
print("Check the Outliner to review names, then re-export the GLB")
