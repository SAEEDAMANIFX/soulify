# SmartRig Pro — Development Notes (problems & solutions)

A Blender 5.x Python add-on (author: Saeed) that places markers on any-sized
character, builds a **Rigify** metarig fitted to the markers (no face), lets the
user add Rigify samples + a **Short Skirt (cloth)** sample, then generate and
**skin** the rig. UI is an ARP-style tabbed panel: **Rig / Skin / Misc**.

This file records every hard-won problem and its fix so they are never lost.
Keep it in the repo. See also `LESSONS.md` (short rules).

---

## Golden workflow / dev discipline
- Editing a `.py` file does **not** affect Blender until the add-on is
  **reinstalled** (rezip / copy to addons dir + `addon_disable`/`addon_enable`).
  Always: edit → reinstall → **verify numerically** in Blender, not by eye.
- The Edit tool can silently **truncate large files** (`ui.py`, `metarig.py`,
  `properties.py`, `skirt.py` were all hit). After editing a big file, verify it
  parses (`ast.parse`). Prefer writing big files via a Blender `io.open` write.
- **Scene fixes are not code fixes.** Cleaning duplicate rigs, binding, applying
  collision etc. change the `.blend`, not the add-on. They are LOST on reopen
  unless the user **saves the .blend**. Tell the user to save.

## Bone roll
- **Foot roll must be FLAT** (local X horizontal, `x_axis.z == 0`), Z down/back —
  like the stock Rigify human foot. Do NOT `align_roll(BACK)` on the foot: a
  splayed foot tilts X upward and the IK foot control looks wrong. Helper
  `setflat()` in `metarig.py` `_fit_core` makes X horizontal. Verify by adding a
  stock `armature_human_metarig_add()` and comparing `foot.L` axes, then deleting
  it. L/R must be mirror-symmetric.

## Short Skirt (cloth) sample
- Built from the **real skirt geometry**, not a guessed circle. Source modes:
  - **Manual** – a starter ring you then edit by hand.
  - **Separate Mesh** – pick the skirt object with the eyedropper.
  - **Merged with Body** – select the skirt faces in Edit Mode → *Register Skirt
    Selection* (stored in the `SR_Skirt` vertex group).
- `_skirt_grid()` slices the geometry by Z from **top (waist) to bottom (hem)**
  and by angular sector into columns. **Anchor each row's Z to the exact slice
  height** (use the verts only for X/Y) — otherwise rows take the band's *mean*
  Z and the chain shrinks inward (bones don't reach the top/hem).
- Each column is one `limbs.simple_tentacle` chain, parented to `spine`, rolled
  like the thigh. Set `tweak_layers_extra=False` (and primary/secondary) to kill
  the "empty tweak layer list" warning.
- Columns/Rows update **live** via a property `update` callback (`_skirt_update`
  → `skirt.live_rebuild`), mesh-driven modes only, never overwriting manual edits.

## Skinning / Bind (the Skin tab)
- Bind body with **automatic weights**. `parent_set(ARMATURE_AUTO)` only works
  with a proper window/area **context override** AND with the Rigify **DEF bone
  collection made visible** (heat weighting ignores hidden bones). Both are
  handled in `bind_mesh._parent_auto`.
- **Split Parts:** the body is solved with the **skirt bones' `use_deform`
  turned OFF**, so the body can never receive skirt weights. The skirt (separate
  object) is solved with **all non-skirt deform bones OFF** → smooth Heat weights
  on the skirt only. (Do NOT crude-proximity-weight the skirt — it tears it.)
- **Body distortion on bind** had several causes, all fixed:
  1. **Double bind** — re-binding added a *second* Armature modifier. Fix:
     `_clean()` removes existing armature modifiers + stale `DEF-` groups + parent
     before binding.
  2. **Preserve Volume** defaulted ON → dual-quat bulging. Default is now OFF.
  3. **Wrong rig targeted** — the scene had two rigs (`RIG-SR_Metarig` and
     `RIG-SR_Metarig.001`); `_generated_rig()` returned the orphan. Fix the scene:
     delete the orphan, rename the live rig to `RIG-SR_Metarig`, set
     `SR_Metarig.data.rigify_target_rig`.
  4. **Leftover Floor-collision junk** — old `SK_FLOOR` constraints (132) +
     `SKC_*` bones driven by drivers pushed bones at rest, distorting BOTH body
     and skirt. Fix: `remove_skirt_collision()` strips all `SK_FLOOR/SK_FOLLOW`
     constraints, `SKC_*` bones and their drivers. At rest, displacement must be 0.

## Skirt motion — leg follow (Pierrick Picaut "dynamic cloth" technique)
- Reference: youtu.be/kCn4sProaek — *leg rotation drives coat rotation*. Cloth is
  a separate bone chain that **rotates** with the leg (waist stays, hem swings) →
  no stretching. This replaced the earlier ARP-style Floor-plane collision, which
  translated deform bones independently and caused **shear/distortion when the
  legs moved**.
- `add_skirt_collision()` (leg-follow) puts a **Copy Rotation (LOCAL, mix ADD)**
  on each column's FK control root, target = nearest thigh, axes X+Z, influence =
  per-column facing weight (front/side columns react, back = 0) × Follow Strength.
- **Skirt went INTO the body instead of out** = the Rigify-generated skirt
  control roll is **flipped 180°** vs the thigh (control X = [-1,0,0] vs thigh
  X = [1,0,0]). Local Copy Rotation then inverts the swing. Fix: auto-set
  `invert_x/invert_z` from the sign of `thigh.x_axis · control.x_axis` (and z).
  Verified: with flipped roll, the front hem now moves OUT (-Y).

## ARP "Kilt" collision — how it really works (reference only, not used now)
Analyzed from a live ARP rig. ARP uses **Floor constraints on per-leg planes that
rotate with the leg**, plus drivers — NOT spheres:
- `kilt_N_dt` Damped-Tracks `kilt_N_tar`; `c_kilt_N_*` are FK children of dt.
- Per leg: `legN_proj` (Locked Track → leg), `kilt_legN_floortar` (Copy Rotation
  from proj, at the thigh), `..._off` (location driven).
- Each `kilt_N_tar` has two FLOOR constraints (FLOOR_Y, use_rotation), one per leg.
- Drivers: floor influence `(1 - rot_diff*(2-spread)) * collide`; floor offset
  `collide_dist * (min(1, rot_diff*2) + falloff)`.
We chose the simpler, distortion-free **leg-follow rotation** instead.

## Reset / Cancel
- `markers.full_cleanup()` removes EVERYTHING the add-on made: markers, metarig,
  all `RIG-SR_Metarig*` / `SR_Reference` / `SR_Rig`, `SKC_*`, `SK_FLOOR/SK_FOLLOW`,
  orphan `WGT-*` widgets, and **unbinds** our meshes (armature mods + `DEF-`/
  `SR_Skirt` groups). It never touches non-SmartRig rigs (e.g. Auto-Rig Pro's
  "rig"). Both "Reset" and "Cancel / Start Over" call it.

## Pending
- **Dynamic jiggle (spring bones)** on the skirt for secondary motion — not yet
  implemented; needs a frame-change spring handler (the Pierrick "dynamic bones"
  layer on top of leg-follow).
