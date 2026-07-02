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
