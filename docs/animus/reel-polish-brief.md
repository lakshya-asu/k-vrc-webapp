# Reel polish brief (from Lakshya's reference images, 2026-08-20)

Lakshya rejected the first reel's presentation: "lighting and stuff is
not good", transitions "pathetic", wants high-fidelity model, a much
better face screen, and the actor producing its OWN face emotions, not
limited to the preset library.

## Reference 1 (stylized render, front three-quarter)

Vivid orange K-VRC on a flat light-gray backdrop. Glossy toy-like
shells with strong smooth specular highlights. WHITE RACING STRIPE
down the helmet centerline. Soft elliptical contact shadow under the
feet. Dark visor with two soft white oval eyes and a small mouth.
Clean, product-shot feel.

## Reference 2 (cinematic PBR, the fidelity target)

Close three-quarter of K-VRC in a warm brown room, dusty atmosphere.
Materials: orange paint with WEAR: edge scuffs, roughness variation,
subtle dirt; black joints matte with machined detail; visor glass has
real reflections. Face is a CRT/LED DOT-MATRIX: two big warm-white
pixelated round eyes with bloom, small cyan chevron mouth glyph,
visible pixel grid and slight chromatic fringe. Lighting: warm key
from upper left, gentle rim on the helmet edge, background falls off
dark. Floor is worn stone catching the character shadow. This is the
overall grade to hit.

## Reference 3 (pose shot)

Same stylized glossy look as ref 1 on a warm beige backdrop, long soft
cast shadow on the ground plane, character in a walking pose looking
at a butterfly on its finger. Face: happy round eyes + smile curve.
Helmet has the twin white stripes.

## Reference 4 (face expression sheet, the face-screen target)

Six frontal busts on black, deep red-orange shells, dramatic low-key
lighting with strong rim. Visor faces are crisp white LED glyphs with
visible vertical-scanline/pixel texture and slight glow:
1. round eyes + small v smile (neutral-happy)
2. ^ ^ closed happy eyes + flat mouth
3. angry slanted brows over eyes + gritted flat mouth
4. big open grin (large rounded-square mouth) + bar eyes
5. bored half-lidded eyes + small o mouth
6. the word "WTF" filling the screen
Lesson: faces are COMPOSED GLYPHS (eyes, brows, mouth, or short text),
not fixed library entries. The screen can say words.

## Directives derived

1. Materials: clearcoat glossy orange, edge-wear option, metallic
   joints, real visor reflections. Add the helmet racing stripes
   (texture or geometry decal) to match the character's canon look.
2. Lighting: warm studio three-point (key, fill, rim) plus a backdrop
   with gentle falloff and a ground plane catching a soft contact
   shadow. Slight bloom. Optional gentle film grade. Per-scene warmth
   still varies but within this look.
3. Face screen v2: visible LED pixel grid, bloom, slight chromatic
   fringe. New generative glyph composer: eyes, brows, mouth as
   parametric glyphs plus short text mode. The actor brain may emit a
   face_glyph beat (validated vocabulary + text length cap) so the
   character authors its own expressions. Preset library remains as
   the fallback vocabulary.
4. Transitions: smooth. Video crossfades with eased timing, audio
   equal-power crossfades aligned to them, no hard audio cuts.
5. Renders: quality first. If EEVEE Next with bloom/AO/soft shadows
   cannot reach ref 2, test Cycles on one scene and compare cost.
