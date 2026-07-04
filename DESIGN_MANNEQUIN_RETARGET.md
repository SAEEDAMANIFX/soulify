# Soulify — Mannequin Retargeting Architecture (approved 2026-07-02)

## Why (the strategic pivot)

Bottom-up geometric analysis of arbitrary internet garments is an endless
whack-a-mole: every asset violates a new assumption (slits, lying-down
imports, tilted collars, A-pose cuffs, detached buttons, tulle layers...).
The 14 hard-won lessons in `_src/smartrig_pro/LESSONS.md` all exist because
we tried to understand the GARMENT alone.

The insight: store assets are designed WORN. The missing information is the
implied wearer, not the cloth. So invert the problem:

**Put a canonical mannequin INSIDE the garment, bind the garment to it with
Blender's proven native tools, then morph the mannequin into the user's
character — the garment follows. One universal algorithm, no load-bearing
classification, and the garment leaves the pipeline RIGGED so the user goes
straight to animation.**

This is the architecture behind professional conforming-clothing pipelines
(Character Creator / DAZ / Marvelous avatar retarget), and Soulify already
owns the hard part: automatic rigging of arbitrary characters.

## Pipeline

```
garment mesh ──> 1. GARMENT SKELETON      (tube centerlines: torso axis from
                                           wrapping rings, sleeve axes from
                                           cuff rings, leg axes for pants)
              ──> 2. MANNEQUIN            (procedural: stick joints + Skin
                                           modifier + subsurf; radii taken
                                           from the garment interior so it
                                           wears the garment snugly)
              ──> 3. BIND                 (Surface Deform / weight transfer,
                                           native + battle-tested; small
                                           loose parts ride along rigidly)
              ──> 4. RETARGET             (snap mannequin joints onto the
                                           character's joints — Soulify autorig
                                           or the AI pose net; proportions and
                                           pose morph together)
              ──> 5. POLISH (optional)    (existing conform for cleanup +
                                           Drape cloth pass)
              ──> 6. RIG HANDOFF          (transfer character rig weights to
                                           the garment: it animates forever)
```

## What existing work feeds in

- Ring/tube analysis (skirt.py + garment.py) -> stage 1 centerlines.
- AI pose landmarks (detect.py) -> stage 4 target joints.
- Two-zone conform -> stage 5 cleanup.
- Drape operator + accessory pinning -> stage 5 polish, stage 3 rigid parts.
- LESSONS.md regression matrix -> automated test suite (every past asset).

## Phases

1. **Mannequin-in-garment** (this phase): `mannequin.py` — garment skeleton
   extraction + procedural skin-modifier mannequin posed inside the garment.
   Deliverable: press a button, see the mannequin wearing the garment.
2. **Bind + retarget**: Surface Deform bind, joint-snap morph to the target
   character (positions first, rotations for posed characters after).
3. **Rig handoff + test matrix**: weight transfer to the character rig;
   automated regression runs over the asset library (skirt, bra, wedding
   gown, shirt, pants, kandura).

## Non-goals

- No bundled neural garment models (SMPL-bound, GPU-heavy, fail on stylized
  characters — evaluated and rejected twice).
- Cloth simulation stays OPTIONAL polish, never the load-bearing fit.


## AI Bone Placement v2 (approved direction, next session)

Detection is now the accuracy bottleneck (collapsed shoulders, mid-depth-plane
Y). Upgrade plan - all on onnxruntime so it accelerates on NVIDIA (CUDA/
TensorRT), Apple Silicon (CoreML) and Windows GPUs (DirectML); CPU fallback:

1. **Multi-view triangulation** (biggest win, zero new models): render the
   character from FRONT + SIDE (+3/4 optional), run the pose net per view,
   triangulate each keypoint from the known cameras -> true 3D joints, no
   more mid-depth-plane Y guess, no collapsed laterals. The render+backproject
   machinery in detect.py already exists - generalize to N views.
2. **Model ensemble**: add a strong permissively-licensed pretrained 2D pose
   ONNX (RTMPose ~13 MB, Apache-2.0) next to smartrig_pose.onnx: RTMPose
   carries realistic bodies, Saeed's model carries stylized ones - pick per
   keypoint by confidence.
3. **Sanity layer stays** (shoulder-between-neck-and-elbow etc.) - nets
   propose, geometry disposes.
Execution providers enabled in detect._session (v1.25.4).


## FINAL ARCHITECTURE SYNTHESIS (from MetaTailor tech breakdown, 2026-07-02)
The user's research confirms the endgame. MetaTailor's core secret: it
REQUIRES a rigged character and uses HER OWN SKIN WEIGHTS as the transfer
medium - no joint detection, no invented binding. Map to Soulify:

1. RIG FIRST (confirmed twice today) - Soulify's core competency.
2. PLACE the garment (our placement/orientation/classification: done).
3. **WEIGHT TRANSFER body -> garment** (Blender's native Data Transfer,
   NEAREST_POLY): the garment inherits the character's weights; add an
   Armature modifier with HER rig -> 'self-adaptive' = the body's own
   deformation carries the cloth; animation-ready forever. THIS replaces all
   invented binding math (segment weights, Surface Deform rides).
4. DESIGN PRESERVATION = OUR EDGE over MetaTailor: ARAP finish pass + rigid
   stiff panels (collar/cuffs/plackets/buttons) per the v1.27.7 audit
   (48% edge distortion must become <10%).
5. Layer ordering = per-garment Ease offsets + an order index (shirt over
   pants); physics secondary motion = the existing skirt jiggle/wind systems.
6. Engine bridges (Unreal/Unity) = later, after quality.

Next session order: (A) Data-Transfer binding + armature hookup,
(B) ARAP + stiff panels, (C) regression matrix with the distortion metric.

============================================================================
# V2 - THE GARMENT-FIT MANNEQUIN PIPELINE (Saeed's full spec, 2026-07-03)
============================================================================

Saeed's words (translated): the addon must know the INNER LAYER of any
garment; compute the VOLUME of its tubes; from that build a mannequin
that FITS the garment; rig it and BIND the garment to it; then MATCH the
mannequin's bones to the character's bones; weights get compared and the
best result wins; in PRO the user gets a step list - see the mannequin,
see the bones, see the weights, edit anything, then Fit; layers can be
REGISTERED manually; buttons/Solidify/true multi-layer garments handled.

## 1. Inner-layer detection (the foundation)
A garment surface has an INSIDE (faces the body) and an OUTSIDE.
- Solidify modifier: evaluate the modifier - the inner shell is the
  offset surface whose normals point INTO the tube volume. Trivial case.
- True multi-shell meshes (jacket + lining, collar band, plackets):
  connected components (we have `_loose_components`) + NESTING test:
  voxelize (VoxelBind machinery), component A is INSIDE component B when
  rays from A's cells hit B in most directions (parity/winding). Order
  components by nesting depth = LAYER INDEX (0 = innermost = lining).
- Single-shell cloth: every face is both sides; "inner" = the surface
  hit first when marching from the tube AXIS outward.
- MANUAL OVERRIDE (Saeed): 'Register Selected as Layer 0/1/2' exactly
  like the v1.32 part registration - SRF_Layer0/1/2 vertex groups, the
  user's word is final. Buttons/hardware stay SRF_Rigid.

## 2. Tube volume -> the FIT mannequin
Per part domain (sleeve L/R, torso, legs - worn-state domains proven in
v1.36.0): march the tube axis (`_trace_tube` exists), at each station
measure the INNER-layer cross-section ring (median radius + center =
`_ring_stats`). That radius sequence IS the mannequin's flesh:
- mannequin joints = tube stations (shoulder/elbow/wrist... from the
  traced centerline, not guesses)
- Skin-modifier radii = inner radius - wearing ease (fabric never
  intersects its mannequin by construction)
- complete the human silhouette (head/hands/feet/legs, v1.36.3) for
  parts the garment does not cover, proportional or from the character.

## 3. Rig + bind
- Armature from the mannequin joints (same names as the character map:
  spine1/2, arm_l, fore_l, leg_l, shin_l...) - `build_garment_rig` DNA.
- BIND = VoxelBind weights, garment verts -> MANNEQUIN bones. The bones
  sit INSIDE the tubes the garment defines, so occlusion seeding and
  geodesic domains are ideal here; LAYERS separate by construction
  (voxel air gaps + layer index masks: heat never crosses layers).

## 4. Bone match + morph (the placement)
- mannequin bone -> character bone: 1:1 by name; `snap_rig_to_joints`
  poses the mannequin rig onto the character joints (posed ORG- bones,
  v1.28.1 lesson). The garment RIDES its bind. The collar encircles the
  mannequin neck by construction -> lands on the character neck exactly
  (kills the v1.36.3 collar failure structurally).
- per-band girth: mannequin tube radii vs character limb radii give the
  exact radial scale per station (replaces auto_size_targets bands).

## 5. The weight JUDGE (compare, best result wins)
Candidates: VoxelBind (default), body-class transfer (experimental),
segment weights (fallback). Judge = the metrics we built this session:
- torso/limb bleed (dominant-bone test per domain)
- design damage (edge_distortion, local-affine)
- pose probe: rotate arm/leg 40 deg, measure off-part displacement
Winner per PART, not global - e.g. VoxelBind sleeves + transfer torso.
All numbers shown in the PRO panel.

## 6. PRO step list (the v1.29 wizard pattern, proven UX)
Steps: 1 Mannequin (show/adjust radii sliders) -> 2 Bones (show rig,
drag joints) -> 3 Weights (per-bone weight-paint preview + judge
numbers) -> 4 Layers/Rigid (register selections) -> 5 FIT. Simple mode
= one click through the same pipeline with judge defaults.

## Build order (next sessions)
A. inner layer + nesting (solidify/multi-shell/single) + layer register
B. tube-volume mannequin (radii from inner layer) + rig + VoxelBind bind
C. bone match/morph as the DEFAULT one-click path (retire the LBS warp
   to fallback)
D. judge + PRO step list.
