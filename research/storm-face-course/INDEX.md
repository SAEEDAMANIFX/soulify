# Storm Facial Rigging — Knowledge Base Index
Captured 2026-07-02 from studio.blender.org (course + project pages, ALL 55 lessons, ALL comments with expanded replies).

| File | Content |
|------|---------|
| 00-course-overview.md | Course structure, philosophy, 3-curve principle, references (Art of Moving Points) |
| 01-model-preparation.md | Ch1 (6 lessons): pre-rig checklist, separation, scale/symmetry, world align, lip straightening |
| 02-initial-bones.md | Ch2 (5): neck/head, jaw, MSTR-mouth mask, stretch bones, nose follow + CloudRig integration thread |
| 03-lips.md | Ch3 (9): ribbon mesh, empties, stretch→DEF chain, constraints, weights cascade, lip zipper |
| 04-eyes.md | Ch4 (10): eye targets/space switch, iris/pupil keys, eyelid ribbon, damped-track rationale, auto-blink, follow |
| 05-brows.md | Ch5 (8): shape-key workflow, TMP geometry, 6 directions, auto driver script, GN linked shapekeys |
| 06-cheeks-teeth-tongue.md | Ch6+7 (5): cheeks keys+lattice, teeth lattice, tongue FK (KISS) |
| 07-mouthcorners-deformers-widgets.md | Ch8+9+10 (11): corrective stack order, combination keys, squash, pucker/compress/sneer, DSP widgets |
| 08-project-storm.md | Blog posts + production logs: the philosophy, hybrid architecture, Blender 5.0 enabling features |

## Downloadable assets referenced (CC-BY)
- bone_mirror_subtargets.py (eyelids-local-controls page) — already bundled in our skill
- shapekeys_automatic_driver_assignment.py (applying-shape-keys / splitting-weights / local-controls pages) — already bundled in our skill
- GN-linked_shapekeys.blend (splitting-weights / cheeks-splitting-weights pages)
- Storm character .blend (course intro page / characters library) — final rig reference; contains the neck-up preservation script
- Dependencies: Pose Shape Keys addon (Blender Studio ext.), "Add Selected Bones to Vertex Group" (Mochi_lin), Blender 5.0+

## What could NOT be captured
- The videos themselves (6h53m, subscriber stream) — but Saeed has access; subtitles exist on all videos since Mar 2026.
- The full architecture is reconstructable from: these notes + the facial-rigging-blender skill + the downloadable Storm .blend.

## Top 10 build rules distilled (for SmartRig Pro face module)
1. Hybrid architecture wins: ribbons (mouth, eyes) + shape keys or ribbons (brows) + lattices (push) + correctives (Pose Shape Keys style).
2. Stretch/pointer bones give DEF bones rotation, not just location (lips). Eyes use Damped Track instead — preserves spherical slide.
3. One DEF bone per edge loop = micro control.
4. Weight strategy = masking cascade: head → face_up/low → jaw → MSTR-mouth → individual DEF, always subtract from parent mask.
5. Corrective order matters: initial XY → combination (X∧Y) → mouth-open correctives → squash.
6. Final Storm fix: CTL-Lip_corn children of P- parent bones (not Copy Location) — prevents false shape-key triggering with jaw open.
7. Corrective sculpt rule: move vertices along the driving axis for smooth blends; Corrective Smooth = last resort.
8. DSP display bones + Override Transforms (Blender 5.0) = controls ride the mesh, no dependency cycles.
9. Validation checks our wizard must run: Deform checkbox on, vertex-group names == bone names, TMP mesh vertex parity, DEF-Jaw tail placement, duplicated ribbons carry vertex groups (weights do NOT symmetrize).
10. KISS: tongue = FK + squash/stretch; don't over-engineer any module.
