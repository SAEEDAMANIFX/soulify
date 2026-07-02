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


## Skirt recognition: slit borders + rotated imports (v1.19.145-146)
**Symptom:** Skirt_5_OBJ (game-asset pleated skirt) got all 8 column tops at one
uniform Z in mid-air (36.92 of a 45-tall mesh); the waist area had no bones.

Two independent recognition bugs:
1. **A boundary loop is not automatically a rim.** The mesh's only border was a
   vertical side SLIT (22 verts all at angle ~-145 deg). `_rim_rings` took it as the
   waist, and `_rim_rz` found every angular wedge empty -> fell back to the
   whole-ring median Z for every column. Fix: `_ring_wraps()` - a rim must occupy
   >=8/12 angular sectors around the bbox centre; non-wrapping loops (slits, seams,
   holes) are dropped. New kind `CLOSED` (borders exist, none wraps) -> rim-span
   over Z-bands. UI label: "closed tube (slit)".
2. **Analyse topology in WORLD space, never local.** The object carried FBX-style
   rotation X=90 (local Y = world up), so every `v.co.z` test in `analyze_skirt`,
   `_rim_rings` and `_skirt_grid_topo` was sideways: the "top band" was a world
   BOTTOM slice. Fix: `bm.transform(ob.matrix_world)` right after `bm.from_mesh`,
   then use `v.co` directly (no more `mw @ v.co` - beware double-transform).

**Verified numerically (Skirt_5_OBJ, h=44.96):** kind CLOSED, waist band z
42.72-44.96, hem 0-2.25, column tops 42.8-44.5 (per-angle, follows the slanted
top), bottoms 0.1-1.7, bone->surface distance mean 1.05 max 3.22 (~2% of h).
Rule reminder: any future geometry test on the skirt object must run on the
world-transformed bmesh.


## Let's Fit - automatic garment fitting (v1.20.0, garment.py)
New feature: fit ANY clothing mesh onto the character (Let's Fit button, entry
panel + compact row while rigging). Pipeline: analyze -> classify -> place ->
conform. Hard-won details:

1. **Two rings are not enough to place a garment** - matching only waist+hem
   rings anchored a skirt on the HEAD (top ring can hug any body slice). Match
   the garment's FULL radius profile (12 slices) against the body profile, with
   asymmetric cost: tighter-than-body x6 (impossible without stretch), looser
   x0.3 (A-line/thobe are fine loose).
2. **Profile cost alone can't tell hips from knees** (both 'loose everywhere' for
   a tube). Classify the garment by top-opening ratio (top_r / max profile r):
   >=0.6 = bottom garment -> anchor at HIPS (max torso R in 35-65% height);
   <0.6 = top/dress -> anchor at NECK (min R in 75-93%). Then refine +-5% height
   x 4 scales with the profile cost.
3. **Density kills percentiles**: A-pose hands (dense finger topology) hang at
   hip height and dominated any per-slice radius percentile (hip R read 0.33
   instead of ~0.15 -> skirt scaled 2x too big). Torso radius must be the
   MEDIAN OF 8 ANGULAR WEDGE MEDIANS - hands occupy ~2 wedges, the median
   ignores them. (Same family as the skirt bbox-vs-median lesson.)
4. Conform stack: SRF_Wrap (Shrinkwrap OUTSIDE, offset = ease% of body h) ->
   SRF_Smooth (Corrective Smooth LENGTH_WEIGHTED) -> SRF_Touchup (Shrinkwrap
   0.9x ease, re-fixes what smoothing pushed back in). All live; Ease/Smoothing/
   Scale/Height sliders have update=_fit_live_update (mandatory - LESSONS #3).
   Scale/Height nudges recompute matrix_world from stored srf_base_matrix
   (never accumulate). Remove Fit restores srf_orig_matrix.
5. Verified on Skirt_5_OBJ + GEO-body_female_stylized: 'bottom (waist->hips),
   scale x0.0124, at 57% of body height', hem just below knee, 0.0% penetrating
   samples. The 45-unit rotated CLOSED-topology skirt needed the v1.19.145-146
   world-space/slit fixes - Let's Fit reuses skirt._rim_rings.


## Let's Fit v2: snug band + AI recognition (v1.20.1-1.20.3)
1. **Shrinkwrap OUTSIDE never pulls a floating garment IN** - the test skirt's
   waistband hovered 6.8 cm off the body. Fix: SRF_Snug = Shrinkwrap ON_SURFACE
   with vertex-group falloff (w=1 at the top ring -> 0 at 30% down), FIRST in
   the stack. Waistband gap after: median 1.4 cm (~= ease). The band hugs, the
   hem stays free. Remove/Apply handle the vgroup too.
2. **Character recognition via the addon's OWN pose net**: _ai_landmarks(body)
   runs detect.detect() (models/smartrig_pose.onnx) - hip_l/hip_r/pelvis ->
   hips anchor, neck -> neck anchor; kconf>=0.3, cached on body['srf_ai'],
   other meshes hidden from the render (clothes confuse the net). Geometric
   wedge-median landmarks remain the fallback when onnxruntime is missing.
   Verified: hips AI 0.912 vs geometric 0.92 on the stylized female (conf 0.72).
3. **onnxruntime on Windows Blender lands in the USER site-packages**
   (C:/Users/<u>/AppData/Roaming/Python/Python313/site-packages) but Blender
   sets site.ENABLE_USER_SITE=False -> import fails even after a successful
   pip install. detect.has_runtime() now appends site.getusersitepackages()
   to sys.path before retrying. onnxruntime 1.27 has cp313 wheels (Blender 5.1).
4. **Clothing recognition = opening signature** (classify_garment): all wrapping
   boundary rings (each tested around ITS OWN centre - a leg opening doesn't
   wrap the garment axis): waist-ratio>=0.6 + >=2 small low rings = pants/
   shalwar; >=0.6 otherwise = skirt; <0.6 + >=2 lateral rings = shirt; else
   dress/thobe. Purely topological, scale-invariant, no model needed. A learned
   garment classifier can be trained later with the same pipeline as
   smartrig_pose.onnx if signatures prove insufficient.


## Let's Fit v3: Preserve Shape mode (v1.20.4, default ON)
User requirement: volumed clothes (pleats, folds, double walls) must fit with
ZERO penetration but WITHOUT destroying the designed shape. Per-vertex
shrinkwrap can't do that (it dents pleats locally and collapses double walls).

Solution - conform_shape(): a SMOOTH OFFSET FIELD on a 14 height-bands x 10
angular-wedges grid (angle measured around the body axis at each vert's z):
  - cell whose innermost vert penetrates (< ease): the WHOLE cell moves
    radially out by (ease - min_gap) -> folds translate together, shape intact;
  - floating cell inside the anchor band (top 30%, falloff weight): whole cell
    pulls IN by (min_gap - ease)*w -> waistband hugs, wall thickness intact;
  - all other cells: zero -> the hem hangs EXACTLY as designed.
Field blurred (angle wraps, passes = 1 + smoothing//8), sampled bilinearly,
written to shape key SRF_Fit (Basis added if the mesh had no keys; K_KEYS
marks ownership). Apply Fit bakes the key; Remove deletes it; sliders recompute
the field live (0.1 s for 8.5k verts).

Verified (Skirt_5 pleated, preserve ON): penetration 0.0%, hem deviation
0.00 cm (median AND max - pleats untouched), all-verts median 0.02 cm, max
4.44 cm confined to the snug band. Non-preserve (shrinkwrap stack) remains as
the option for skin-tight conforms.


## Let's Fit v4: closing the gap for real (v1.20.5-1.20.6)
User: 'still a gap between the skirt and the body'. Two fixes:
1. **Iterate the offset field** (3 rounds of measure -> field -> blur -> apply):
   one pass under-corrects because the blur dilutes band cells and a cell's
   min-gap vert understates the rest (elliptical body vs circular skirt left
   2.4-2.7 cm at front/back). bands 14->20 so the slanted rim gets its own rows.
2. **Strict per-vertex no-penetration clamp AFTER the field**: blur mixing
   strong band pulls with neighbouring pushes left 1.9% verts inside the body
   (visible poke-through). Final pass moves ONLY still-inside verts onto
   surface + floor, where floor = max(ease, 0.002 * body_h) (~3 mm) - render-
   safe contact, zero z-fighting.
Verified all around the waist: EVERY 45-deg wedge min gap = 0.32 cm (= the
floor -> true contact), medians 0.4-1.6 cm = fabric thickness/drape, 0.0%
penetration, hem deviation still 0.00. When the user reports a gap, measure
the INNER-WALL min per wedge - the outer wall median reads as 'gap' but is
actually cloth thickness.


## Let's Fit v5: the rim itself must touch (v1.20.7-1.20.9)
User circled a gap at the waistband RIM. Three layered causes/fixes:
1. **Mixed cells can only push**: with a user Scale nudge shrinking the garment
   below body size, a cell containing both penetrating verts (hip) and the rim
   (outside) pushes EVERYTHING further out - no force ever closes the rim.
   Fix: per-vertex band target = body surface + floor + rel[i], where rel[i] =
   the vert's DESIGN gap above its cell's innermost vert (recorded on iteration
   0 before any conform). Walls keep thickness, band hugs at ANY scale.
2. **rel preserves designed FLARE too** - the rim flares outward by design and
   stayed 1.2-1.8 cm off. Cap rel in the band at percentile-60 (~wall
   thickness): keeps the double wall, kills the flare.
3. **The very top edge**: rel fades to 0 over the top 6% of the garment so the
   rim presses onto the body like a real elastic waistband, thickness returning
   gradually below.
Rim-edge gap after: min = floor (0.32) all around, medians 0.4-1.06 cm (was
0.9-1.8); penetration 0.0%; hem deviation still 0.00 cm. Note: Fit/Refit resets
Scale/Height nudges to neutral by design.


## Let's Fit v6: the Wedding-Dress gauntlet (v1.20.10-1.20.19)
A 154k-vert layered ruffle ball gown (user reference: lace bodice -> waist
belt -> tiered skirt) exposed six recognition/conform gaps. All fixed and
regression-checked against Skirt_5:

1. **Open-back necklines never wrap** -> the highest wrapping ring was an inner
   layer. If the top wrapping ring sits >15% below the mesh top, anchor on the
   actual top Z-band ring; `top_band=True` = no real collar -> anchor at CHEST
   (AI keypoint), not neck.
2. **Layer rings are not legs**: pants detection now requires OFFSET rings
   (ring centre >0.6r off the garment axis). Concentric ring stacks = layers.
3. **AUTO-ORIENTATION (the big one)**: the gown was imported LYING DOWN (ring
   plane normal horizontal). Garment axis = size-weighted, sign-aligned mean of
   wrapping-ring plane normals (SVD per ring, now returned by _all_rings);
   rotate axis->Z about the bbox centre; then the SMALLEST significant ring
   (the wearing opening: waist/neck/tube) must sit in the TOP half, else flip
   180. Only significant rings vote (>=40 verts AND r >= 0.15*half-extent) -
   Skirt_5's 22-vert slit flipped the standing skirt before this filter.
4. **Envelope conform, not nearest-surface**: closest_point_on_mesh pulls
   fabric into the between-legs channel and tears gowns. Renv[band][wedge] =
   body max radius per cell -> garments hang over the leg SILHOUETTE. Faster
   too (no BVH queries in the loop).
5. **Arms out of the envelope**: A-pose arms/hands share slices with torso and
   hips; keep only the radial cluster connected to the innermost surface (cut
   at the first air gap > 4% body height). Where arms CONNECT (shoulder
   junction, above the AI chest line) exempt per-cell: env > 1.35 x band
   median -> NaN (free). Radial push there only builds a fake shelf - necklines
   rest on the shoulder slope vertically.
6. **Snug only what was designed against the body**: wrel weight from the
   design offset (rel <= 2*cap full snug, >= 4*cap free) - without it the
   band pull sucked every ruffle in the top 40% onto the torso (flat disc).
7. **Dresses are fitted at the WAIST**: if a concentric waist ring exists in
   the garment's 30-80% height (belt/waist seam), scale+anchor by it at the
   body waist (min R in 55-72%); neck/chest anchors are fallbacks.
Final: 'dress/thobe (waist, AI), auto-oriented' - bodice over the bust, cap
sleeves on the shoulder slope, tiers from the waist, no shelf, no tearing.


## Let's Fit v7: bust + sleeves at the shoulder junction (v1.20.20-1.20.21)
User (side-view annotation): arm stabbing through the cap sleeve; earlier the
bust poked through the bodice. Both were fallout of the shoulder-cell
exemption:
1. **Bust is not an arm**: the exemption (env > 1.35 x band median above the
   chest line) also freed the BREAST bulge cells -> bodice stayed behind the
   bust. Arms are always LATERAL: exempt only wedges within 50 deg of +-X;
   front/back cells keep their push (bodice now covers the bust).
2. **Exempt != unconstrained**: with lateral cells fully freed, the sleeve sank
   into the deltoid. Keep Rfull[band][wedge] (cluster max INCLUDING the arm)
   and, in exempt cells only, clamp each vertex individually to Rfull + floor -
   no cell-wide push (that was the 25 cm shelf), but a sleeve can never sink
   into the shoulder ball. Measured: lateral shoulder zone 0.0% penetration,
   front bust 0.4% worst 2.1cm (lace edge - user can add ease).
Never apply the Rfull clamp outside exempt cells: at hip height it would
balloon skirts out to the hanging hands.


## Let's Fit v8: staircase blocks = per-cell constants (v1.20.22)
Close-ups showed the lace bodice carved into angular blocks splitting every
~36 deg (= wedge width) and the layered lace crushed to one shell. Two causes,
one principle - EVERY per-vertex operation must read SMOOTH fields:
1. The final constraints (envelope e, arm Rfull) were read as per-CELL
   constants -> radius steps at each cell border. Now a combined constraint
   grid (envelope | arm-envelope in exempt cells) is sampled BILINEARLY with
   NaN-aware weights (_samp_nan).
2. The strict clamp collapsed every inside vert to the same shell (e+floor),
   flattening multi-layer lace. Now it lands each vert at its own DESIGNED
   offset: r = e + floor + min(rel[i], 6% span) - same trick as the band
   target; layer order and lace relief survive.
Metric: bodice edge-step (|offset(a)-offset(b)| per mesh edge, z>0.95) median
0.03 cm, p95 0.64 cm - was multi-cm blocks. Rule: cell grids are for
AGGREGATION; anything applied back to vertices must be interpolated.


## Let's Fit v9: two-zone conform - the professional answer (v1.20.23)
User: 'the shoulder/chest part is not professional'. Root truth: RADIAL
conform is the wrong model for the upper body - bust, shoulder slope, armpit
and sleeves are VERTICAL terrain. Final architecture:

  ABOVE the body waistline (min torso R in 55-72% height, +-blend band):
    per-vertex nearest-SURFACE conform along the surface NORMAL:
      inside -> out to surface + floor + rel (structure preserving);
      outside & designed-close (wrel) -> pulled to the same target x0.85;
      designed-away flare/ruffles (wrel->0) -> untouched;
    then 4+ Laplacian passes over the OFFSET VECTORS along mesh edges
    (np.add.at on the edge array - fast at 154k verts) -> silky fabric;
    final anti-penetration re-check per vertex.
  BELOW the waistline: the radial envelope machinery (legs channel, hem
  freedom) exactly as in v5-v8. 6% z blend band joins the zones seamlessly.

Metrics on the gown bodice: edge-step median 0.016 cm / p95 0.33 cm,
penetration 0.1%, 3.7 s total at 154k verts. Visual: bodice follows the bust,
belt+bow at the waist, cap sleeve resting on the shoulder = reference photo.
Lesson: pick the conform BASIS by anatomy (surface-normal above the waist,
radial below), and smooth offsets - never positions - so design detail
survives.


## Let's Fit v10: structure basis must match the conform basis (v1.20.24)
User circled: cap sleeve glued flat onto the deltoid + z-fighting specks on
the chest. Cause: the upper (surface) zone was still using RADIAL rel as its
design structure - radial layer offsets do not survive a surface-normal
projection, so lace layers landed co-shell and the sleeve's designed air gap
read as 'designed close'.
Fix: measure the design structure IN THE SAME BASIS the zone conforms in:
rel_s[i] = design surface-distance above the cell-min surface distance
(pass 1 gathers signed nearest-surface distances of the placed design;
cap_s = p60; wrel_s fades pull for designed-air fabric). Push target and the
final clamp use floor + min(rel_s, 10% span).
Result: upper-zone surface distances spread p5 0.62 / p50 1.29 / p95 3.29 cm
(layered, arched) instead of one shell; penetration 0.3%. General rule:
radial zone -> radial rel; surface zone -> surface rel.


## Let's Fit v11: 'as imported' is the spec (v1.20.25)
User showed the TurboSquid product render: the asset is ALREADY perfectly
draped on its implied mannequin - the goal is that exact look transferred to
the character, not a re-draping. Upper zone therefore:
  - NO pull/hug at all (pulling is what wrinkled the pristine lace);
  - push ONLY where the character is bigger than the implied mannequin,
    spread as a low-frequency INFLATION: 10+2*passes Laplacian on the offset
    field, then a residual re-push pass (smoothing dilutes the push where it
    was needed most) + 3 more feathering passes;
  - contact comes from the design itself; the sash/waist snug (lower zone)
    anchors the dress.
Result: upper deviation from the imported design p50 = 1.01 cm (the bust
inflation only), penetration 0.2%, smooth lace back seam, arched sleeves.
Principle: when the asset ships pre-draped, fitting = scale + place + minimal
smooth inflation. Deform as little as possible, never re-drape.


## Let's Fit v12: slider lag -> debounced conform (v1.20.26)
Sliders on a 150k-vert gown froze Blender: every drag tick re-ran the full
conform (~4 s). Split the callback:
  - apply_nudges (matrix scale/height) runs INSTANTLY -> live visual feedback;
  - conform_shape is DEBOUNCED: token counter + bpy.app.timers (0.4 s); each
    tick bumps the token, only the closure holding the LATEST token executes.
Slider ticks now 0 ms; the heavy pass runs once after release. Note: user's
liked result for the wedding gown = Scale 0.97, Height -0.05, Smoothing 40.


## Let's Fit v13: reference correspondence (v1.20.27)
Preserve Shape broke at user Scale 0.89: back fabric sank PAST the body's
medial axis, its nearest surface flipped to the FRONT, and the push ejected it
out the wrong side (collapsed bare back + floating shard).
Fix: garment->body correspondence (nearest point LW, normal NW, rel_s, zone
membership) is computed ONCE from the REFERENCE placement (K_BASE, the
Scale-1.0 auto-fit) where it is unambiguous, and reused for any nudge; only
the CURRENT depth (p - LW).NW decides how far to push along the stored NW.
Bonus: the residual + final clamp passes no longer re-query the BVH at all
(plane math on stored correspondence) -> faster too. Verified at the user's
exact tuned state (Ease 1%, Smooth 40, Scale 0.89, Height -0.05): back zone
0.0% penetration, bodice smooth over the bust.

## v1.20.28 — Edit Mode on a HIDDEN rig crashes every skirt toggle
Collision / jiggle / masters add & remove all enter Edit Mode on the generated
rig. If the user hid the rig (eye icon), `view_layer.objects.active = rig`
"succeeds" but `context.active_object` is still None -> `mode_set` raises
"Context missing active object". The chest-jiggle path was patched long ago;
the 6 skirt paths were not. Fix: one shared `skirt._edit_rig(rig)` helper
(leave mode -> hide_set(False) + hide_viewport=False -> select -> activate ->
mode_set EDIT, never raises, returns bool). EVERY future Edit-Mode entry on the
rig must go through it. Verified: add/remove collision+jiggle with the rig
hidden — 0 stale SKC bones, parents restored, 26 columns rebuilt.

## v1.20.29 — accessibility sentinel + no bare `objects.active = rig`
`view_layer.objects.active = rig` RAISES when the rig's collection is excluded
from the View Layer ("ViewLayer does not contain object"). Audited every bare
assignment in skirt.py: removed the redundant ones, wrapped the rest, unified
chest jiggle on `_edit_rig`. Convention: add/remove return **-1 = rig not
accessible** (operators report `_NO_ACCESS`), 0 = nothing to do. Verified with
the rig in an excluded collection: all 6 paths return -1, no traceback; hidden
rig still auto-unhides and builds.

## v1.20.30 — skirt masters deformed the BODY
`eb.new()` defaults `use_deform=True`; the region masters (`skirt_master*`)
were never set False, so the heat-map body bind assigned ~800 body verts to
them -> grabbing a master dragged the body. Fix: (a) masters created with
`use_deform=False`; (b) bind's split filter now disables EVERY skirt-related
deform bone (DEF-skirt, skirt_master, skirt FK/tweak, SKC_) during the body
bind, not just DEF-skirt.*. RULE: every new helper/control bone must set
`use_deform = False` at creation. Verified: master +0.2 m -> skirt 0.2012 m,
body 0.0000 m, collision+jiggle+masters coexisting.
