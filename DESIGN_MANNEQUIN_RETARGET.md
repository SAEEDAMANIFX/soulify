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
