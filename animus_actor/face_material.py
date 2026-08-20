"""Bind rendered face frames to the visor material. Runs INSIDE Blender.

The webapp drives the visor by swapping the screen mesh's material for
an emissive canvas texture (attachFaceScreen in the webapp's
faceScreen.js, emissiveIntensity 1.8, flipY false). This is the same
move for offline renders: the profile's stage.face_screen names the
screen object (kvrc.glb imports it as 'screen'); its material slots are
replaced with one emissive material driven by the take's PNG image
sequence, frame-locked to the scene so viseme frames line up with the
voice track.

Orientation note: glTF UVs have a top-left origin, which the Blender
importer converts to Blender's bottom-left convention, so the PNGs
(canvas top row first) sample upright without any flip -- the same
reason the webapp sets faceTexture.flipY = false.

Kept importable without bpy so the actor unit tests can load the
module; every function takes bpy as its first argument.
"""

import os

MATERIAL_NAME = "AnimusFaceScreen"
DEFAULT_EMISSION_STRENGTH = 2.0


class FaceBindError(RuntimeError):
    pass


def list_frame_files(frames_dir):
    files = sorted(
        name
        for name in os.listdir(frames_dir)
        if name.startswith("face_") and name.endswith(".png")
    )
    if not files:
        raise FaceBindError(f"no face_*.png frames in {frames_dir}")
    return files


def bind_face_screen(bpy, profile, frames_dir, frame_start=1):
    """Drive the profile's screen object with the frame sequence.

    Returns a report dict (object, material, image, frame count) that
    lands in the stage report so the receipt shows the binding.
    """
    config = (profile.get("stage") or {}).get("face_screen") or {}
    object_name = config.get("object")
    if not object_name:
        raise FaceBindError("profile.stage.face_screen.object is not set")
    screen = bpy.data.objects.get(object_name)
    if screen is None:
        raise FaceBindError(f"screen object '{object_name}' not in the scene")

    files = list_frame_files(frames_dir)
    first = os.path.join(frames_dir, files[0])
    image = bpy.data.images.load(first)
    image.source = "SEQUENCE"

    material = bpy.data.materials.new(MATERIAL_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (400, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (100, 0)
    shader.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    shader.inputs["Roughness"].default_value = 0.4
    strength = float(
        config.get("emission_strength", DEFAULT_EMISSION_STRENGTH)
    )
    shader.inputs["Emission Strength"].default_value = strength
    # Optional visor glass: a clearcoat over the LED face so the
    # screen catches the studio lights (reel-polish brief, directive 1).
    coat = float(config.get("coat", 0.0))
    if coat > 0.0:
        shader.inputs["Coat Weight"].default_value = coat
        shader.inputs["Coat Roughness"].default_value = float(
            config.get("coat_roughness", 0.06)
        )

    texture = nodes.new("ShaderNodeTexImage")
    texture.location = (-260, 0)
    texture.image = image
    texture.interpolation = "Closest"  # the webapp uses NearestFilter
    texture.image_user.frame_duration = len(files)
    texture.image_user.frame_start = int(frame_start)
    texture.image_user.frame_offset = 0
    texture.image_user.use_auto_refresh = True
    texture.image_user.use_cyclic = False

    links.new(texture.outputs["Color"], shader.inputs["Emission Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    # The webapp replaces the visor mesh's material outright; same here.
    screen.data.materials.clear()
    screen.data.materials.append(material)

    return {
        "object": object_name,
        "material": material.name,
        "image": os.path.basename(first),
        "frames_dir": frames_dir,
        "frame_count": len(files),
        "frame_start": int(frame_start),
        "emission_strength": strength,
    }
