# SmartRig Pro — Lessons (do NOT repeat these mistakes)

## How Auto-Rig Pro's Kilt (cloth) collision REALLY works — analysed from a live ARP rig
ARP does NOT use Limit-Distance spheres. It uses **Floor constraints on per-leg
collision planes that rotate with the leg**, driven dynamically. Architecture:

- Bones per column: `kilt_N_dt.l` (deform), `kilt_N_tar.l` (collision target),
  `c_kilt_N_1..5.l` (FK control chain, children of the dt bone).
- `kilt_N_dt.l` has a **Damped Track** -> `kilt_N_tar.l`. The mesh follows dt /
  controls; the target is the thing that gets collided, dt just aims at it.
- Per leg: `legN_dt.x` (leg proxy), `legN_proj.x` (**Locked Track** -> legN_dt.x,
  projects leg direction), `kilt_legN_floortar.x` (**Copy Rotation** from
  legN_proj.x — the floor plane, sits at the thigh surface), and
  `kilt_legN_floortar_off.x` (offset child whose `.location` is driven).
- Each `kilt_N_tar` gets two **FLOOR** constraints (`floor_location='FLOOR_Y'`,
  `use_rotation=True`), one per leg, targeting that leg's `floortar_off`. The
  target can't cross the leg's plane -> planar constrained collision.
- Master control `c_kilt_master.x` holds custom props: `collide`, `collide_spread`,
  `collide_dist`, `collide_dist_falloff`.
- **Drivers (26):**
  - Floor influence per column/leg: `(1 - rot_diff*(2-collide_spread)) * collide`
    where `rot_diff` = ROTATION_DIFF of a per-column projection vs the leg -> only
    columns facing the moving leg are pushed.
  - Floor offset distance: `collide_dist * (min(1, rot_diff*2) + collide_dist_falloff)`
    with `rot_diff` = leg swing amount -> the plane pushes out more as the leg lifts.

**Why it's better than spheres:** planar (no corner clipping), follows leg
rotation, per-column weighting, push scales with leg motion. To replicate in the
Rigify path: build tar/dt/control tiers, leg floor-plane bones rotating with the
thigh, Floor constraints on the targets, and the two driver families above.


## Foot bone roll MUST be flat (X horizontal) — fixed v1.6.3
**Rule:** In `metarig.py` → `_fit_core`, the **foot** bones must be rolled so their
local **X (side) axis is horizontal** (`x_axis.z == 0`), with Z pointing down/back —
exactly like the standard Rigify human metarig.

**Do NOT** roll the foot with `align_roll(BACK)`. When the character's foot splays
outward / diagonally, aligning Z to BACK tilts the X axis up (seen as
`X = [0.8, 0, 0.6]`, roll ≈ -37°). That makes the IK foot control tilted and looks
unprofessional. Saeed explicitly flagged this.

**Correct approach** — `setflat(name)` helper (already in `_fit_core`):
```
yb = (tail - head).normalized()
xh = yb.cross(Vector((0,0,1)))   # horizontal, ⟂ to the bone
zd = xh.cross(yb)                # desired Z
if zd.z > 0: zd = -zd            # Z must point down/back (Rigify convention)
bone.align_roll(zd)
```
Applied to `foot.*` only. thigh/shin keep `BACK`; toe/heel keep `UP`. The
mirror-roll pass (`R = -L`) keeps both sides symmetric.

**Always verify numerically (don't eyeball):** add the real Rigify metarig in-scene
(`bpy.ops.object.armature_human_metarig_add()`), read its `foot.L` x_axis/z_axis as
the ground truth, delete it, then confirm SR_Metarig's `foot.L` has `x_axis.z ≈ 0`
and that `foot.R` is the mirror (Y flipped, roll negated).

## General discipline
- File edits don't reach Blender until the addon is reinstalled (rezip/copytree +
  addon_disable/enable). Always rebuild → reinstall → verify numerically.
- The Edit tool can silently truncate large files (ui.py, metarig.py, properties.py
  have all been hit). After editing a big file, verify it parses (`ast.parse`) from
  Blender's side; prefer doing large edits via Blender `io.open` writes.
