# Chapter 4 — Eyes (10 lessons)
Captured 2026-07-02, all comments incl. expanded replies.

## 4.01 Initial bones
Desc: Initial eye bones; assign both eyes to armature deformation. Eye targets (TGT) with **space switch: head space / body space / world space**.

Comments:
- **YULIYA + Nezonic (course gap):** at 5:30 DEF eye bones must be **duplicated** (not just moved), and **TGT eye bones parented to MSTR-Eye_target** — video doesn't say this explicitly.
- Artiom: script installation confusion → **Rik: the 'assign automatic drivers' script isn't installed — it's run when creating shape keys in ch5 Brows.**

## 4.02 Eye highlights
Desc: Rigging artificial highlights for the animator to pose, with custom properties to dial in follow of the eye. (No comments.)

## 4.03 Iris and pupil
Desc: **Shape keys control iris/pupil size + Shrinkwrap modifiers maintain the spherical eye shape**; shape keys hooked to control bones for viewport manipulation.
Comments: praise only.

## 4.04 Initial weight painting
Desc: Define the area deformed by the master eye bones → clean structured eyelid weighting later.

Comments:
- **Rik weight-smoothing tip:** smoothing only happens on the vertex selection — expand selection first with **Ctrl + Numpad +/− in weight paint mode**.
- Users struggle with jump cuts/timelapses in videos (feedback theme).

## 4.05 Eyelids ribbon mesh
Desc: Ribbon mesh for the eyelids (like lips) — guide for individual eyelid bones. Basis for auto-blink and eyelids-follow.
Comments: one user missed "Selected Bones to Vertex Group" addon dependency (recurring confusion).

## 4.06 Eyelids deformation bones
Desc: DEF eyelid bones, each influencing a single edge loop, like lips. Uses **Damped Track** constraints to vertex targets.

Comments (key design rationale):
- **Yunior: why no empties like the lips? Can I use Stretch-To?** → **Rik: lip empties exist only for the lip-zipper; eyes don't need it — use mesh vertices directly as targets. Stretch bones don't hold spherical deformation → intersections with spherical eyeballs. Damped-track setup adheres to the eye's spherical shape (slide over the eye). If eyes are non-spherical, stretch-to might work.**
- Video timelapse at 4:18-4:21 hides the damped-track target assignment — Rik said he'd update the video.

## 4.07 Eyelids local controls
Desc: Local control bones for the eye ribbon per 3-curve principle; subtract/divide weights of main upper/lower eyelid controls.
**Downloadable script on this page: `bone_mirror_subtargets.py` (922 bytes, CC-BY)** — mirrors constraint subtargets (we already bundle it in our skill).

Comments:
- **Nurbek (important rig-reading insight):** local In/Out bones seem to smoothly follow both parents in final Storm rig with no constraints — **it's a display-bone illusion: DSP bones override widget transforms (covered in final 'Control widgets' video).**
- **Rik: Symmetrize (right-click in Edit Mode) mirrors all selected bones with .L/.R suffix.**

## 4.08 Eyelids weight painting
Desc: Subtract weights from master eye bone; time-consuming tweaking but clean upper/lower eyelid deformation. (No comments.)

## 4.09 Auto blink
Desc: Custom properties + a new guide mesh extracted from the eyelid ribbon = **blink target blendable between upper and lower eyelid**.

Comments (symmetrizing pitfalls):
- **Marshall Peterman:** duplicating HLP autoblink mesh and scaling X=-1 doesn't mirror vertex groups → **Rik: the .L/.R vertex groups should already exist on the duplicated ribbon (duplicated at 1:12 from video 05) — duplication carries all vertex groups, so no mirroring needed.** Confirmed as the fix.
- **Beatriz:** blink damped-track bones went inside the eyeball after symmetrize — **weights don't transfer when symmetrizing the ribbon; assign manually to each point on the blink ribbon.**

## 4.10 Eyelids follow
Desc: Eyelids naturally follow/slide with eye rotation; simple constraint system + animator-adjustable follow amount.

Comments:
- **Leonard (bug in video):** Rik copied the "Highlight Follow" driver instead of "Eyelid Follow" — Rik confirmed, adding a correction note.

## Key takeaways for SmartRig Pro (eyes module)
1. Eyes chain: DEF eye bones (duplicated) + TGT bones under MSTR-Eye_target with head/body/world space switch → iris/pupil shape keys + Shrinkwrap → eyelid ribbon (3-curve) → DEF per edge loop with **Damped Track to mesh vertices (NOT stretch-to — preserves spherical slide)** → local CTLs → auto-blink guide mesh + blend property → eyelids-follow constraint with influence slider.
2. Critical automation rules: duplicate (don't move) DEF eye bones; duplicate ribbon inherits vertex groups (rely on this, skip mirroring); symmetrize needs .L/.R suffixes; weights never symmetrize with the ribbon mesh — assign programmatically per point.
3. bone_mirror_subtargets.py (CC-BY) legitimately reusable in our codebase.
4. DSP display bones (ch10) create the "smooth follow" illusion — animator-facing polish, zero constraints.
5. Iris/pupil sizing = shape keys driven by control bones — matches our Deform Toolkit driver wizard.
