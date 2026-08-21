"""The reusable stage look: lighting, set, materials, bloom.

Applied by animus_actor/stage.py when the profile's stage.render
carries a "look" section. Without that section the stage renders
exactly as before (legacy key+fill lights, imported materials
untouched), so old profiles and the acceptance harness see no change.

The look encodes the reel-polish brief (docs/animus/reel-polish-brief.md):

    lighting   warm three-point studio (key, fill, rim) with soft
               area shadows; per-scene warmth still comes from the
               profile's key_color/fill_color
    set        a backdrop plane with gentle falloff to dark plus a
               ground plane that catches the character's soft contact
               shadow
    materials  clearcoat glossy orange shells, metallic joints, a
               subtle roughness variation so the paint reads real,
               and the canon white helmet racing stripes (a shader
               decal in the Head mesh's Generated coordinates: the
               GLB ships no textures, so the stripes cannot come from
               the asset)
    visor      optional clearcoat over the emissive face screen so
               the glass catches the key light
    bloom      compositor Glare (EEVEE Next has no legacy bloom
               toggle), threshold above 1 so only emissive pixels and
               speculars bloom
    grade      AgX view transform with a configurable look

Kept importable without bpy (every scene-touching function takes bpy)
so the unit tests can exercise the preset resolution logic.
"""

DEFAULT_LOOK = {
    "preset": "studio_v2",
    # Lighting energies (W). key_color/fill_color still come from the
    # legacy render config so per-scene warmth keeps working.
    # Values below are the grade picked against reference 2 from the
    # 2026-08-20 test-frame ladder (eevee-v2-a .. -d in the polish
    # session receipts): warm key, dark falloff, pooled backdrop.
    "key_energy": 700.0,
    "fill_energy": 150.0,
    # Wash guard (scene-5 lesson, 2026-08-20 board note): a per-scene
    # render config may raise key/fill energy for mood, but past these
    # ceilings the orange shell blows out toward white under studio_v2's
    # AgX Punchy grade (scene 5 shipped at key 1150 / fill 340 and
    # visibly washed). The preset clamps what profiles request; a scene
    # that truly needs more must override the ceilings knowingly.
    "key_energy_max": 1000.0,
    "fill_energy_max": 300.0,
    "rim_energy": 1000.0,
    "rim_color": [0.85, 0.92, 1.0],
    # Set geometry + colors.
    "backdrop": True,
    "backdrop_color": [0.10, 0.065, 0.045],
    "backdrop_distance": 3.2,
    "floor": True,
    "floor_color": [0.035, 0.022, 0.016],
    # A spot pooled on the backdrop behind the character so the
    # background falls off dark at the edges (reference 2).
    "backdrop_pool_energy": 150.0,
    # Materials.
    "materials": True,
    "stripes": True,
    "stripe_offsets": [0.435, 0.565],
    "stripe_halfwidth": 0.022,
    "stripe_z_min": 0.30,
    "orange_roughness": 0.28,
    "orange_coat": 0.8,
    "black_metallic": 0.85,
    "black_roughness": 0.35,
    "wear": 0.35,
    # Visor glass: clearcoat over the emissive face material.
    "visor_coat": 1.0,
    "visor_coat_roughness": 0.06,
    # Bloom via compositor glare.
    "bloom": True,
    "bloom_strength": 0.12,
    "bloom_threshold": 1.1,
    # Grade.
    "view_look": "AgX - Punchy",
    "exposure": -0.1,
    # EEVEE quality.
    "samples": 96,
    "raytracing": True,
    "shadow_softness_boost": True,
}

PRESETS = {
    "studio_v2": {},
}


def resolve_look(render_cfg):
    """The effective look dict, or None when the profile has no look.

    render_cfg["look"] may be True (all defaults), a preset name, or a
    dict of overrides (optionally naming a preset). Unknown preset
    names fail loudly; silent fallbacks would ship the wrong look.
    """
    raw = (render_cfg or {}).get("look")
    if not raw:
        return None
    look = dict(DEFAULT_LOOK)
    if raw is True:
        return look
    if isinstance(raw, str):
        raw = {"preset": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"stage.render.look must be true, a name, or a dict, not {raw!r}")
    preset = raw.get("preset", "studio_v2")
    if preset not in PRESETS:
        raise ValueError(
            f"unknown look preset '{preset}'; known: {sorted(PRESETS)}"
        )
    look.update(PRESETS[preset])
    for key, value in raw.items():
        if key == "preset":
            look["preset"] = preset
            continue
        if key not in DEFAULT_LOOK:
            raise ValueError(f"unknown look key '{key}'")
        look[key] = value
    return look


# --- lighting -------------------------------------------------------------


def clamped_energy(requested, maximum):
    """The effective light energy: the request, capped by the preset.

    A falsy maximum disables the clamp (a scene overriding
    key_energy_max/fill_energy_max to 0 or null opts out knowingly).
    """
    value = float(requested)
    if maximum:
        return min(value, float(maximum))
    return value


def build_lighting(bpy, render_cfg, look):
    """Warm three-point studio lighting with soft area shadows."""
    scene = bpy.context.scene

    key = bpy.data.objects.new("KeyLight", bpy.data.lights.new("KeyLight", "AREA"))
    key.data.energy = clamped_energy(
        render_cfg.get("key_energy", look["key_energy"]),
        look.get("key_energy_max"),
    )
    key.data.size = 2.6
    key.data.use_shadow = True
    key.location = tuple(render_cfg.get("key_location", (2.1, -2.6, 2.9)))
    key.data.color = tuple(render_cfg.get("key_color", (1.0, 0.88, 0.72)))
    scene.collection.objects.link(key)

    fill = bpy.data.objects.new("FillLight", bpy.data.lights.new("FillLight", "AREA"))
    fill.data.energy = clamped_energy(
        render_cfg.get("fill_energy", look["fill_energy"]),
        look.get("fill_energy_max"),
    )
    fill.data.size = 3.5
    fill.data.use_shadow = True
    fill.location = tuple(render_cfg.get("fill_location", (-2.6, -1.8, 1.3)))
    fill.data.color = tuple(render_cfg.get("fill_color", (0.85, 0.88, 1.0)))
    scene.collection.objects.link(fill)

    rim = bpy.data.objects.new("RimLight", bpy.data.lights.new("RimLight", "AREA"))
    rim.data.energy = float(look["rim_energy"])
    rim.data.size = 1.2
    rim.data.use_shadow = True
    rim.location = (-1.4, 2.6, 2.4)
    rim.data.color = tuple(look["rim_color"])
    scene.collection.objects.link(rim)

    # Aim every light at the character's chest height.
    aim = bpy.data.objects.new("LookLightTarget", None)
    aim.location = (0.0, 0.0, 1.0)
    scene.collection.objects.link(aim)
    for light in (key, fill, rim):
        track = light.constraints.new(type="TRACK_TO")
        track.target = aim

    if look.get("backdrop") and look.get("backdrop_pool_energy"):
        pool = bpy.data.objects.new(
            "BackdropPool", bpy.data.lights.new("BackdropPool", "SPOT")
        )
        pool.data.energy = float(look["backdrop_pool_energy"])
        pool.data.spot_size = 1.5
        pool.data.spot_blend = 1.0
        pool.data.use_shadow = False
        pool.location = (0.0, -1.5, 2.2)
        pool.data.color = tuple(render_cfg.get("key_color", (1.0, 0.88, 0.72)))
        scene.collection.objects.link(pool)
        track = pool.constraints.new(type="TRACK_TO")
        aim2 = bpy.data.objects.new("BackdropAim", None)
        aim2.location = (0.0, float(look["backdrop_distance"]), 1.3)
        scene.collection.objects.link(aim2)
        track.target = aim2
    return key, fill, rim


# --- set geometry ---------------------------------------------------------


def _plain_material(bpy, name, color, roughness=0.85):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.2
    return material


def build_set(bpy, look):
    """Floor plane plus a backdrop with vertical falloff to dark."""
    scene = bpy.context.scene
    made = []

    if look.get("floor"):
        mesh = bpy.data.meshes.new("LookFloor")
        s = 30.0
        mesh.from_pydata(
            [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)],
            [],
            [(0, 1, 2, 3)],
        )
        floor = bpy.data.objects.new("LookFloor", mesh)
        floor.data.materials.append(
            _plain_material(bpy, "LookFloorMat", look["floor_color"], 0.8)
        )
        scene.collection.objects.link(floor)
        made.append(floor.name)

    if look.get("backdrop"):
        mesh = bpy.data.meshes.new("LookBackdrop")
        w, h = 30.0, 16.0
        y = float(look["backdrop_distance"])
        mesh.from_pydata(
            [(-w, y, 0.0), (w, y, 0.0), (w, y, h), (-w, y, h)],
            [],
            [(0, 1, 2, 3)],
        )
        backdrop = bpy.data.objects.new("LookBackdrop", mesh)
        material = bpy.data.materials.new("LookBackdropMat")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
        # Vertical falloff: the backdrop color at the floor line eases
        # to near-black overhead, so the frame edges fall off dark.
        coords = nodes.new("ShaderNodeTexCoord")
        separate = nodes.new("ShaderNodeSeparateXYZ")
        links.new(coords.outputs["Generated"], separate.inputs["Vector"])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "EASE"
        base = look["backdrop_color"]
        ramp.color_ramp.elements[0].position = 0.05
        ramp.color_ramp.elements[0].color = (*base, 1.0)
        ramp.color_ramp.elements[1].position = 0.55
        ramp.color_ramp.elements[1].color = (
            base[0] * 0.06,
            base[1] * 0.06,
            base[2] * 0.06,
            1.0,
        )
        links.new(separate.outputs["Z"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        backdrop.data.materials.append(material)
        scene.collection.objects.link(backdrop)
        made.append(backdrop.name)
    return made


# --- materials ------------------------------------------------------------


def _wear_roughness(nodes, links, bsdf, base_roughness, wear):
    """Subtle noise-driven roughness variation (reference 2's worn paint)."""
    if wear <= 0:
        bsdf.inputs["Roughness"].default_value = base_roughness
        return
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 4.0
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.0
    ramp.inputs["From Max"].default_value = 1.0
    ramp.inputs["To Min"].default_value = max(0.02, base_roughness - 0.10 * wear)
    ramp.inputs["To Max"].default_value = min(1.0, base_roughness + 0.22 * wear)
    links.new(noise.outputs["Fac"], ramp.inputs["Value"])
    links.new(ramp.outputs["Result"], bsdf.inputs["Roughness"])


def upgrade_materials(bpy, look):
    """Clearcoat orange shells, metallic joints. Touches scene data only."""
    report = []
    orange = bpy.data.materials.get("Orange")
    if orange is not None and orange.use_nodes:
        nodes = orange.node_tree.nodes
        links = orange.node_tree.links
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Coat Weight"].default_value = float(look["orange_coat"])
            bsdf.inputs["Coat Roughness"].default_value = 0.08
            _wear_roughness(
                nodes, links, bsdf, float(look["orange_roughness"]), float(look["wear"])
            )
            report.append("Orange: clearcoat gloss + wear roughness")
    black = bpy.data.materials.get("Black")
    if black is not None and black.use_nodes:
        bsdf = next(
            (n for n in black.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None
        )
        if bsdf is not None:
            bsdf.inputs["Metallic"].default_value = float(look["black_metallic"])
            bsdf.inputs["Roughness"].default_value = float(look["black_roughness"])
            report.append("Black: machined metal")
    return report


def add_helmet_stripes(bpy, look, helmet_object="Head"):
    """The canon twin white racing stripes as a shader decal.

    The GLB carries no textures, so the stripes are painted in the
    shader: two bands in the Head mesh's Generated X coordinate
    (Generated coords are computed from the undeformed mesh, so the
    decal sticks to the helmet under armature motion). The Head gets
    its own copy of the orange material; the shared Orange used by the
    body panels is left alone.
    """
    if not look.get("stripes"):
        return None
    head = bpy.data.objects.get(helmet_object)
    if head is None or head.type != "MESH":
        return None
    source = None
    for slot_index, slot in enumerate(head.material_slots):
        if slot.material is not None:
            source = (slot_index, slot.material)
            break
    if source is None:
        return None
    slot_index, base = source
    material = base.copy()
    material.name = "OrangeHelmetStriped"
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return None

    coords = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Generated"], separate.inputs["Vector"])

    half = float(look["stripe_halfwidth"])
    zmin = float(look["stripe_z_min"])
    band_nodes = []
    for center in look["stripe_offsets"]:
        lo = nodes.new("ShaderNodeMath")
        lo.operation = "GREATER_THAN"
        lo.inputs[1].default_value = float(center) - half
        links.new(separate.outputs["X"], lo.inputs[0])
        hi = nodes.new("ShaderNodeMath")
        hi.operation = "LESS_THAN"
        hi.inputs[1].default_value = float(center) + half
        links.new(separate.outputs["X"], hi.inputs[0])
        band = nodes.new("ShaderNodeMath")
        band.operation = "MULTIPLY"
        links.new(lo.outputs[0], band.inputs[0])
        links.new(hi.outputs[0], band.inputs[1])
        band_nodes.append(band)
    both = nodes.new("ShaderNodeMath")
    both.operation = "ADD"
    both.use_clamp = True
    links.new(band_nodes[0].outputs[0], both.inputs[0])
    links.new(band_nodes[1].outputs[0], both.inputs[1])

    # Keep the stripes off the visor bezel: only above the z cut.
    above = nodes.new("ShaderNodeMath")
    above.operation = "GREATER_THAN"
    above.inputs[1].default_value = zmin
    links.new(separate.outputs["Z"], above.inputs[0])
    mask = nodes.new("ShaderNodeMath")
    mask.operation = "MULTIPLY"
    links.new(both.outputs[0], mask.inputs[0])
    links.new(above.outputs[0], mask.inputs[1])

    base_color = tuple(bsdf.inputs["Base Color"].default_value)
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = base_color
    mix.inputs["B"].default_value = (0.92, 0.91, 0.88, 1.0)
    links.new(mask.outputs[0], mix.inputs["Factor"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    head.material_slots[slot_index].material = material
    return material.name


# --- engine, bloom, grade -------------------------------------------------


def configure_render(bpy, look):
    """EEVEE Next quality, compositor bloom, and the color grade."""
    scene = bpy.context.scene
    if scene.render.engine == "BLENDER_EEVEE_NEXT":
        scene.eevee.taa_render_samples = int(look["samples"])
        if look.get("raytracing") and hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = True
        if look.get("shadow_softness_boost") and hasattr(scene.eevee, "shadow_ray_count"):
            scene.eevee.shadow_ray_count = 2
            scene.eevee.shadow_step_count = 4

    view = scene.view_settings
    try:
        view.look = look["view_look"]
    except TypeError:
        # The named look is not in this Blender's OCIO config; the
        # default grade is a safe fallback for the view transform only.
        pass
    view.exposure = float(look["exposure"])

    if look.get("bloom"):
        scene.use_nodes = True
        scene.render.use_compositing = True
        tree = scene.node_tree
        tree.nodes.clear()
        layers = tree.nodes.new("CompositorNodeRLayers")
        glare = tree.nodes.new("CompositorNodeGlare")
        glare.glare_type = "BLOOM"
        if hasattr(glare, "quality"):
            glare.quality = "HIGH"
        # Blender 4.5 exposes bloom knobs as node inputs.
        for name, value in (
            ("Threshold", float(look["bloom_threshold"])),
            ("Strength", float(look["bloom_strength"])),
        ):
            if name in glare.inputs:
                glare.inputs[name].default_value = value
        composite = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(layers.outputs["Image"], glare.inputs["Image"])
        tree.links.new(glare.outputs["Image"], composite.inputs["Image"])


def apply_look(bpy, profile, render_cfg, look):
    """Everything above, in order. Returns a report for the receipt."""
    report = {"preset": look["preset"]}
    build_lighting(bpy, render_cfg, look)
    report["set"] = build_set(bpy, look)
    if look.get("materials"):
        report["materials"] = upgrade_materials(bpy, look)
        stripes = add_helmet_stripes(bpy, look)
        if stripes:
            report["stripes"] = stripes
    configure_render(bpy, look)
    report["bloom"] = bool(look.get("bloom"))
    return report
