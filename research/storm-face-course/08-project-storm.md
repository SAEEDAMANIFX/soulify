# Project Storm — Blog Posts & Production Logs
Source: studio.blender.org/projects/storm/ — captured 2026-07-02.

## Blog 1: "Proposal: Facial Rigging with shape keys" (24 Feb 2025) — THE PHILOSOPHY
Rik's original proposal. Core ideas (this is the intellectual foundation of the whole system):

- Industry reality: **the overwhelming majority of feature-film facial rigs (last 10 years, AAA studios incl. Sony Imageworks) are blendshape/shape-key based** — minimize computer interpolation, have shapes DESIGNED by an artist.
- **The control-bone illusion:** animator moves a bone in 3D space, but the bone is bound to the mesh and merely triggers shape keys via drivers/expressions — "moving in space" is an illusion. Infinite shapes combinable while bones follow correctly.
- **Workflow for generating shapes:** duplicate face mesh → single deformation bone weight-painted to region → animate the bone through range of motion (±X/Y/Z) → extract all key poses as shape keys ("gets 85% to final shapes") → tweak/sculpt/mask → split into local shapes (weights splitting) → assign to control bones via drivers. **Partially automatable.**
- Non-destructive iteration: add a new shape key as a "layer" and tweak on it; Rik's script updates shape keys within a collection without losing drivers.
- Bones remain the base for skull, jaw, eyes, eyelids, nose, ears (static/simple areas).
- **Hook-bone attachment (no dependency cycles):** hook bones with location constraint to a single-vert vertex group → drive the DISPLAY location (Override Transform) of control bones which drive shape keys.
- 55 comments, overwhelming community demand. Notable: game engines don't support this well (Rik: "focus is high-quality movie character rigging"; alembic/USD partial路); KSDN's pain points: no shape-key folders, sculpt performance drops with shape keys, shape keys break on topology change, can't transfer between characters (bone rigs can).

## Blog 2: "Project 'Storm' has started!" (23 May 2025)
- Goal: artist-friendly facial rig, expressions sculpted, non-destructive workflow. Concept art: Vivien Lulkowski + Julien Kaspar; expression sheets/sculpts guide the rig ("engineer the facial rig where it can confidently strike these poses").
- **After talking to industry riggers: "there is rarely a single perfect solution — the right approach depends on character style, pipeline, and personal preference. In some areas bones turned out more flexible and reliable."** ← direct validation of SmartRig's modular multi-recipe design.
- Custom scripts made during prototyping: enhanced shape-key operators, weight-painting improvements, driver automation.
- Override Transforms updated (attach controls to vertices, gizmo at visual location).
- Division of labor: Rik = face; Demeter Dzadik = body via CloudRig; merged at the end.

## Blog 3: "Storm has been released!" (18 Nov 2025) — THE ARCHITECTURE SUMMARY
- Storm = ready-to-animate character for **Blender 5.0**; body = CloudRig; face = custom by Rik. **Pose library included (facial, hand, full-body poses) via asset shelf.**
- Body: FK/IK switch preserving world transform, hinge, parenting, tweak bones, foot roll, pole parenting; spine redesign — IK hips with upper+lower hip controls to curve spine; squash & stretch in CloudRig UI panel.
- **Face rig = HYBRID:** eyes + mouth primarily ribbons (with corrective shape keys on top); brows + cheeks shape-key based (with local deform bones that FOLLOW the shape-key deformations); lattices for extra push everywhere.
- Guide ribbon definition (verbatim-ish): a spline/thinned ribbon mesh running along the face surface, a deformable "track" controlling sliding/stretching of skin (mouth corners, eyelids, cheeks); two-sided surface for natural skin rolling + volume preservation; features like lip-zip and auto-blink isolated from facial geometry.
- **Rik's lesson learned: "different facial features benefit from different rigging solutions"; next rig he'd use ribbons for brows too + Pose Shape Keys correctives.**
- Two armatures (face + body) merged into one object at the end so animators interact with a single armature. Built with Blender Studio Asset Pipeline.
- **Blender 5.0 features created FOR this workflow:** shape-key tree-view UI + multi-select (Pratik Borhade); new operators Copy/Update/New-from-Objects + 'flipped' option + Duplicate (Hans Goudey); Pose Shape Key addon (PSD workflow, auto driver assignment from pose); Override Transforms gizmo update (Wayde Moss, Christoph Lendenfeld).
- Comment: Rik confirms course release "hopefully early 2026" (released 4 Mar 2026).

## Blog 4: "Course: Facial Rigging now online!" (4 Mar 2026)
Course announcement (content matches course overview). Open question in comments: multi-artist iteration on same character (modeller+rigger simultaneously) — Studio Asset Pipeline docs exist.

## Production logs (weekly, short)
- #316 Shape Keys: mouth corners first via Pose Shape Key addon (in-house by Demeter); Blender 5.0 updates temporarily broke the addon; continued with brows/cheeks/lips keys.
- #324 Widgets: used CloudRig's premade bone widget library for clean controls.
- #332 Promo Video: final tweaks/polish/bugfix + promo showing expressions.
- #336 Ready for Release: released alongside Blender 5.0.

## Strategic notes for SmartRig Pro
1. Storm itself is downloadable (characters library) — the final rig is our reference implementation; several course Q&As say "check the final Storm rig" for fixes (P- bones for lip corners, DSP bones, collection structure).
2. The hybrid conclusion (ribbons for mouth/eyes, shape keys or ribbons for brows) IS our module/recipe matrix — validated by Blender Studio's own R&D journey.
3. Blender 5.0 native features (shape-key flipped ops, Override Transforms gizmos, Pose Shape Keys extension) are the enabling APIs our face module should target.
4. The "85% automatable" claim from Rik's proposal is our product thesis for the face module: SmartRig automates the mechanical 85%, artist sculpts the last 15%.
5. Pose library shipped via asset shelf = exactly our planned Pose Library feature.
