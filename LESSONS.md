# SmartRig Pro — Lessons (do NOT repeat these mistakes)

## CRITICAL: collision uses Python drivers — needs "Auto Run Python Scripts" ON
The skirt collision is built from Python-expression drivers that read the leg
bones (TRANSFORMS variables). If a .blend is OPENED while Blender's "Auto Run
Python Scripts" (Preferences > Save & Load) is OFF, Blender loads the rig
**untrusted**: the drivers still hold their rest value but their depsgraph
dependency relations are NOT built, so they STOP updating when the legs move ->
"collision doesn't work" even though sk_kilt=1 and the drivers/`SKC_master` all
still exist. This is NOT a bug in our rig. Fixes: (1) the addon now forces
`bpy.context.preferences.filepaths.use_scripts_auto_execute=True` + saves prefs in
`_ensure_drivers_trusted()` (called by add_skirt_collision); (2) for the CURRENTLY
open file you must RELOAD it once (save, then reopen) so the trusted depsgraph
rebuilds the driver relations. Setting the pref mid-session does NOT re-arm the
already-loaded rig's driver relations; only a reload does. NEVER call
`rig.animation_data_clear()` on the generated rig in tests — it wipes ALL the
collision drivers too.



## Skirt collision — FULL EVOLUTION LOG (attempt → result → why) — read this first
Every approach we tried for the short-skirt leg collision, in order, so we never
repeat a dead end. CURRENT shipped = #7 (COMPASS + per-row bend).

1. **Floor constraints on the DEF bones directly** — translated deform bones
   independently → shear/distortion when legs moved. Rejected.
2. **Copy-Rotation leg-follow** (control copies the thigh's 3-axis rotation) —
   columns swing TOWARD the leg and CROSS each other; Limit Rotation can't cap a
   combined rotation. Rejected. Also: Rigify control roll is flipped 180° vs the
   thigh, so the naive copy sent the skirt INTO the body.
3. **Radial-outward drivers (v1.18)** — each column rotates outward only, by the
   nearest leg's local rx/rz. Never crosses, directional, strong. Dropped then for
   "no true clearance", BUT this idea (direction from leg motion) is what the final
   compass model is built on — it was on the right track.
4. **ARP "Kilt" floor+tar+dt mechanism (v1.19.3–1.19.9)** — per-leg floor PLANE
   (follows the leg) + per-column tar (Floor-collided) + dt (Damped Track), control
   rides dt. Gave true clearance. Bugs found & fixed: floor bone must point ALONG
   the leg not +Z (else +0.67 rest jump); Damped Track must be TRACK_Y not
   TRACK_NEGATIVE_Y (else dt flips, +0.68 jump); influence needs PROXIMITY gating
   (v1.19.5) so the far side stays put; then a DIRECTION gate (v1.19.9). Still two
   real flaws: the INFINITE floor plane pulled same-side back/side columns, and its
   push was strong forward but WEAK sideways. Superseded.
5. **ARP literal ROTATION_DIFFERENCE gating (v1.19.6)** — fixed per-column
   reference bones + rotation-difference drivers + offset-growth, exactly like ARP.
   Tested on our rig: 3× WEAKER push and the side kick broke. Why: rotation_diff is
   roll-sensitive, and stacking it on our already-directional floor plane = a
   double gate in series → weak. ARP gets away with it because their whole kilt is
   co-calibrated in their generator. Reverted. DO NOT retry on our floor design.
6. **Euler-channel direction gate** — read the thigh's local rot_x/rot_z to gate
   direction. Failed for SIDE/abduction: euler channels don't separate abduction
   from flexion (IK side kick read as forward). Replaced by the knee-swing compass.
7. **COMPASS model + per-row progressive bend (v1.19.12 → v1.19.15) — CURRENT.**
   See the section below. Knee-minus-hip horizontal vector = compass needle (works
   FK & IK, any direction, roll-free); each column rotates outward by the component
   along its own outward; both legs blended by proximity; SKC_dt split per row so
   the column bends progressively (cloth drape). Strong all directions, far side &
   back stay put, no crossing, scales to any column/row count.

## Skirt collision = COMPASS model (v1.19.12+) — CURRENT shipped mechanism
The floor-plane approach below was replaced. The floor plane's push strength
varied by direction (strong forward, weak side — user: "side kick no response")
and its infinite plane pulled same-side back/side columns. The robust answer
(matches what Saeed described as ARP's "compass bone"): each column RIDES its
`SKC_dt` bone (control re-parented onto dt). A driver rotates dt OUTWARD by how
much the nearest leg's KNEE swings toward that column:
`amount = AMP * clamp(0, (knee_xy - hip_xy - rest_offset) · column_outward) * (dist/0.12) * spread * collide`.
The knee-minus-hip horizontal vector is the compass needle — it points the way
the leg kicks (forward/back/in/out) for ANY pose, FK OR IK, with no dependence on
euler axes (which fail to separate abduction from flexion — that was the
side-kick bug). Each column reacts only to the component along its own outward,
so a side kick moves only the side columns, forward only the front, etc. It only
swings outward (clamp >=0) so it never crosses. Each column BLENDS BOTH legs'
compasses weighted by proximity (wL=dR/(dL+dR), wR=dL/(dL+dR)), so no centre
column is dead and the whole front arc lifts smoothly on a one-leg forward kick
(strong over the leg, tapering across). It scales to ANY column/row count because
every term (outward, swing axis, weights) is per-column from its own position.
Driver var type is `'TRANSFORMS'`
(LOC_X/LOC_Y, WORLD_SPACE) — NOT `'TRANSFORM_CHANNEL'` (that enum doesn't exist).
`SKC_dt` is SPLIT into one segment per ROW (`SKC_dt.CC.RR`), chained, and EACH row
control rides its own segment. Each segment rotates by `AMP/nrows`, so the bend
accumulates down the chain into a smooth progressive drape toward the hem (cloth),
not a rigid swing — like ARP's `c_kilt_CC_01..06`. Distributing across N rows
roughly halves the hem throw, so `AMP=5.5` (verified: 5-row column bends
0.012→0.037→0.074→0.122→0.181 root→hem, no crossing). Scales to any rows.
The dt swing axis (local rot_X vs rot_Z) + sign is picked from dt's rest matrix:
+rotX moves the hem along +Zlocal, +rotZ along -Xlocal. Verified: fwd/back/side
all push 0.13–0.25; far side = 0; back = 0 on a forward kick; rest = 0; no
crossing even at fwd+up 70°. Master props: Collide (on/off), Swing (was
collide_dist), Strength (was collide_spread). `collide_dist_falloff` is now
UNUSED (it was floor clearance) and dropped from the UI.

## (SUPERSEDED by the compass model — kept for reference) ARP "Kilt" floor+tar+dt mechanism (v1.19.3–1.19.9)
NOTE: this was the shipped model from v1.19.3 to v1.19.11; it is NO LONGER used
(see the compass section above). It replicated Auto-Rig Pro's kilt limb and gave
REAL clearance (the cloth is pushed away from the leg), per-leg/per-column weighting,
push-scales-with-leg-motion, and no self-intersection in any direction. Validated
numerically on the live rig: rest displacement ~0.023; leg fwd pushes front cols,
back pushes back cols, side pushes side cols (push-only, outward); angular gaps
between columns stay POSITIVE in fwd/back/side/fwd+up (no crossing).

`add_skirt_collision(rig, props)` builds, per leg + per column:
- **`SKC_floor.L/R`** — a floor PLANE bone at mid-thigh, **parented to DEF-thigh**
  so it follows the leg. CRITICAL: the bone must point **along the leg direction**
  (`fl.tail = mid + legdir*0.12`), so its **Y-axis = leg direction** (down at
  rest). A FLOOR_Y plane then sits roughly horizontal at rest (hem is below it →
  no push) and tilts as the leg swings → pushes only when the leg moves. Pointing
  the bone +Z (Y=up) makes the floor shove the whole skirt UP at rest (0.67 jump).
- **`SKC_tar.NN`** — per-column collision target at the hem, parented to the hips.
  Two **FLOOR** constraints (`floor_location='FLOOR_Y'`, `use_rotation=True`), one
  per leg, `offset = dist*(1+falloff)`, `influence = facing × spread`.
- **Facing weight = XY PROXIMITY DIFFERENCE, not linear X** (v1.19.5 fix). A column
  is pushed by a leg only if it is clearly NEARER that leg than the other:
  `fL = max(0,(dR-dL)/(dR+dL))`, `fR = max(0,(dL-dR)/(dL+dR))` (dL,dR = XY distance
  from the column head to each thigh head), ×1.7 gain, then ×spread. Columns at the
  centre (equidistant) and the whole opposite side get 0, so the far side stays
  rock-stable when one leg moves. The earlier linear `0.5±0.5·bx` gave the centre
  0.5 from BOTH legs, so motion bled across and the user saw "the skirt pulled to
  the other side / no stability on the other side". Verified: lift L leg -> every
  right-side column moves 0.000, centre <=0.025, only left cols near the leg push;
  symmetric for the R leg.
- **TWO gates, not one** (v1.19.9). The infinite FLOOR plane, when it tilts with
  the leg, also lifts SAME-SIDE back/side columns (user: "pulled from the side /
  from behind"). Fix: gate the floor influence by BOTH (a) PROXIMITY = which leg's
  side the column is on (skip the far side), and (b) DIRECTION = the leg's LOCAL
  swing dotted with the column's outward. The direction term reads the thigh's
  `ROT_X` (forward/back) and `ROT_Z` (side) via TRANSFORMS driver vars (LOCAL
  space, roll-FREE — unlike ROTATION_DIFFERENCE): influence =
  `clamp01(GAIN*(rx*oyn + rz*oxn*sidesign))*spread*collide`. Forward kick (rx<0)
  only lifts front cols (oyn<0); back kick only back; side kick only side. Offset
  stays static (true clearance). Verified: fwd kick -> back-left = 0.000, far side
  = 0.000, only front-left push; symmetric. This is what makes it professional.
- **Do NOT replace this with ARP's literal ROTATION_DIFFERENCE gating** (tested
  v1.19.6, reverted). ARP drives each Floor influence by ROTATION_DIFF(fixed
  per-column reference bone, live leg) and grows the offset with leg swing. We
  reverse-engineered it onto our rig: 30 SKC_ref bones + rotdiff drivers. Result
  was 3x WEAKER push (0.14 -> 0.05) and the SIDE kick broke. Reason: our floor
  PLANE already follows the leg, so it alone gates front/back/up by GEOMETRY;
  adding a rotdiff influence gate on top = a SECOND gate in series -> the column
  only pushes when BOTH the plane tilts into it AND rd is small, which rarely
  fully coincide -> weak. ARP can use rotdiff because their whole kilt (planes,
  offsets, reference orientations) is co-calibrated by their generator. For our
  design the winning split is: PROXIMITY gates the SIDE (L/R), the tilting plane
  gates the DIRECTION. Keep it.
- **`SKC_dt.NN`** — per-column bone from waist→hem, parented to the hips, with a
  **Damped Track → SKC_tar.NN**. CRITICAL: `track_axis='TRACK_Y'` (the bone's +Y
  points toward its tail=hem=target; TRACK_NEGATIVE_Y aims the wrong way and flips
  dt 180° → 0.68 rest jump).
- The column control **`skirt.NN.00` is RE-PARENTED onto `SKC_dt.NN`** so it RIDES
  the collision while FK still works on top. The original parent is stored in the
  pose-bone prop **`sk_origparent`**; `remove_skirt_collision` reads it, restores
  the parent in edit mode BEFORE deleting the SKC_ bones, and clears `sk_kilt`.
- Marker `rig["sk_kilt"]=1`. **4 ARP live settings live on a master bone**
  `SKC_master` as keyframeable custom properties (`collide`, `collide_dist`,
  `collide_dist_falloff`, `collide_spread`, each with id_properties_ui min/max).
  Every Floor constraint's `offset` and `influence` is **driven** from those props
  (offset `= dist*(1+fall)`, influence `= min(1,max(0,facing*spread))*collide`,
  facing baked per column). So tweaks are instant AND animatable — exactly ARP's
  `c_kilt_master`. The N-panel **Item tab → "Short Skirt"** (`SMARTRIG_PT_skirt_item`)
  draws these 4 props so the animator adjusts them while posing without selecting
  the bone. The addon Skin-tab sliders feed the same master props via
  `live_kilt_tune` (it writes the bone props, NOT the constraints — drivers own
  those). Verified: collide=0 → push drops to rest baseline; bigger collide_dist →
  bigger push; removal deletes SKC_master + all 52 drivers cleanly.

Earlier models and why they were dropped: **Copy-Rotation leg-follow** copies the
full 3-axis leg rotation → columns swing toward the leg and CROSS. **Radial-outward
drivers** (v1.18) never crossed but only ROTATE outward (no true clearance / the
cloth doesn't actually stay off the leg). The ARP floor mechanism gives real
planar collision and matches what the user asked for ("اضافتنا تعمل بهذة الطريقة").


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


## Skirt "Follow Body" = Surface Deform (v1.19.31) — clings to the body when sitting
First tried bone Copy Rotation (front/side -> ORG-thigh, back -> ORG-spine, LOCAL
ADD, slider-driven influence). REJECTED: it only ROTATES the columns, giving a
stiff skirt that sticks straight out when sitting — not clinging. The user wanted
it to cling "like a Weight Transfer / Surface Deform". Correct tool = a **Surface
Deform** modifier on the SEPARATE skirt mesh, bound to the body mesh, stacked AFTER
the Armature modifier; its `strength` is driven by the `follow_body` slider (and
also set directly in `live_follow_tune` so it works even when python drivers are
disabled). 0 = skirt rig only; 1 = skirt follows the body surface (drapes over the
lap when seated). Bind in REST pose (`bpy.ops.object.surfacedeform_bind` with a
context override, skirt active). Only works for a SEPARATE skirt mesh (Surface
Deform needs a different target mesh; a MERGED skirt is part of the body). The body
must actually deform (legs posed) for the cling to show — needs the trusted-driver
file state like collision.

## Skirt jiggle = live spring (v1.19.26)
Secondary motion ("feels like simulation"): one `SKC_jig.CC` bone per column is
INSERTED above the column root (pivot at waist, spans to hem) and the column root
re-parents onto it, so swinging the jig sways the whole column (hem most, waist
fixed). A `frame_change_post` handler `skirt_jiggle_handler` runs a damped spring
PER jig bone: goal = the no-jiggle tail driven by the (non-jiggling) parent;
`v += (goal-p)*stiffness; v *= (1-damping); p += v`; length-constrained; the bone
is posed to aim at `p`. Underdamped defaults (stiffness 0.40, damping 0.25) give a
believable overshoot+settle (verified: hem 0→0.42 overshoot→0.25→settle). Settings
live on the RIG OBJECT (`jiggle`, `jiggle_amount`, `jiggle_stiffness`,
`jiggle_damping`) — NOT a bone — so jiggle works with OR without collision and is
keyframeable; the handler reads them. State (`_JIG_STATE`) resets on backward/jumped
frames so scrubbing doesn't fake motion. Jig bones go in the hidden "Skirt (MCH)"
collection. `Bake to Keyframes` keys the jig rotations over the frame range and sets
`rig["sk_jiggle_baked"]` so the handler skips that rig (re-Apply to go live again).
The handler is re-armed in `register()` if any rig has `sk_jiggle`. Pitfalls:
parent the jig to the NON-jiggling hips (no chain feedback); only ONE jig per column
keeps the 15 springs independent so one read pass is stable (no per-bone depsgraph
updates needed).

## Smart (structure-aware) skirt skinning (v1.19.17)
The skirt is a known grid (columns = angular sectors, rows = heights), so we skin it
from that structure instead of a generic heat map. `_smart_skirt_weights(obj, rig,
vids)`: per vertex, find the 2 nearest COLUMNS by azimuth (center = avg of the
top-row bone heads) and blend them linearly by angular distance (so weight never
bleeds past the adjacent column); within each column, weight the nearest 1-2 row
SEGMENTS by inverse `_seg_dist`. Weights sum to 1, use ONLY `DEF-skirt.*` bones.
Toggle `skin_smart_skirt` (default ON) in the Skin tab; falls back to heat/proximity
if off or if no skirt grid. Verified: each vertex lands on its 2 columns × 1-2 rows,
sum=1, zero body-bone weights — cleaner than heat on the thin cloth.

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
