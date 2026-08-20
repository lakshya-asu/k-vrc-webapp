"""The developer test rig: a small humanoid biped, plain bpy data.

One canonical skeleton shared by the acceptance harnesses and the actor
stage, so the test character is a figure with a pelvis, spine, chest,
head, clavicle-arm-hand chains and two legs instead of a tower of
stacked bones. The seven legacy bone names the embodiment profile and
fixtures reference (root, spine, head, arm.L/R, hand.L/R) keep their
names; the biped adds chest, clavicles, legs and feet around them.

Proportions are a ~1.65 m figure facing -Y, standing on Z = 0.
"""

import bpy

# name -> (head, tail, parent). Order matters: parents come first.
BIPED_BONES = {
    "root": ((0.0, 0.0, 0.95), (0.0, 0.0, 1.05), None),
    "spine": ((0.0, 0.0, 1.05), (0.0, 0.0, 1.25), "root"),
    "chest": ((0.0, 0.0, 1.25), (0.0, 0.0, 1.45), "spine"),
    "head": ((0.0, 0.0, 1.45), (0.0, 0.0, 1.65), "chest"),
    "clavicle.L": ((0.03, 0.0, 1.42), (0.15, 0.0, 1.40), "chest"),
    "arm.L": ((0.15, 0.0, 1.40), (0.34, 0.0, 1.02), "clavicle.L"),
    "hand.L": ((0.34, 0.0, 1.02), (0.40, 0.0, 0.90), "arm.L"),
    "clavicle.R": ((-0.03, 0.0, 1.42), (-0.15, 0.0, 1.40), "chest"),
    "arm.R": ((-0.15, 0.0, 1.40), (-0.34, 0.0, 1.02), "clavicle.R"),
    "hand.R": ((-0.34, 0.0, 1.02), (-0.40, 0.0, 0.90), "arm.R"),
    "upperleg.L": ((0.09, 0.0, 0.95), (0.09, 0.0, 0.52), "root"),
    "lowerleg.L": ((0.09, 0.0, 0.52), (0.09, 0.0, 0.12), "upperleg.L"),
    "foot.L": ((0.09, 0.0, 0.12), (0.09, -0.14, 0.03), "lowerleg.L"),
    "upperleg.R": ((-0.09, 0.0, 0.95), (-0.09, 0.0, 0.52), "root"),
    "lowerleg.R": ((-0.09, 0.0, 0.52), (-0.09, 0.0, 0.12), "upperleg.R"),
    "foot.R": ((-0.09, 0.0, 0.12), (-0.09, -0.14, 0.03), "lowerleg.R"),
}

BIPED_BONE_NAMES = list(BIPED_BONES)


def covers(bone_names):
    """True when every requested bone exists in the biped layout."""
    return all(name in BIPED_BONES for name in bone_names)


def build_biped_armature(object_name):
    """Create and link the biped armature object; returns the object."""
    armature = bpy.data.armatures.new(f"{object_name}_rig")
    obj = bpy.data.objects.new(object_name, armature)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (head, tail, parent) in BIPED_BONES.items():
        bone = armature.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        if parent is not None:
            bone.parent = armature.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj
