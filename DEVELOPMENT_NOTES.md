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

## Skirt leg collision — CURRENT = COMPASS model (v1.19.15)
The current shipped skirt collision is the **compass model** (knee-swing direction
per column + per-row progressive bend). See LESSONS.md → "Skirt collision — FULL
EVOLUTION LOG" for the complete history of every approach tried and why each was
kept or dropped. The ARP floor mechanism documented just below was the shipped
model v1.19.3–1.19.11 and is now SUPERSEDED (kept for reference only).

## (SUPERSEDED v1.19.12) ARP "Kilt" floor+tar+dt mechanism (was shipped v1.19.3–1.19.11)
Replicated from a live ARP rig. ARP uses **Floor constraints on per-leg planes
that rotate with the leg** (NOT spheres). Our old `add_skirt_collision` built:
- `SKC_floor.L/R` — floor plane at mid-thigh, **parented to DEF-thigh**, bone
  pointing **along the leg** so its Y-axis = leg direction (FLOOR_Y tilts with the
  leg; horizontal-ish at rest → no rest push).
- `SKC_tar.NN` — per-column target at the hem (parented to hips) with two FLOOR
  constraints (FLOOR_Y, use_rotation), one per leg; `offset = dist*(1+falloff)`,
  `influence = facing_weight * spread`.
- `SKC_dt.NN` — waist→hem bone (parented to hips), **Damped Track → tar**,
  `track_axis='TRACK_Y'`. The column control `skirt.NN.00` is **re-parented onto
  dt** (original parent saved in `sk_origparent`) so it rides the collision and FK
  still layers on top.
- 4 live settings (`skirt_collide / _dist / _spread / _falloff`) re-tune offset +
  influence with no rebuild via `live_kilt_tune`. Marker prop `rig["sk_kilt"]`.

**Two bugs that cost the most (see LESSONS.md):** (1) floor bone must point ALONG
the leg, not +Z, or it shoves the skirt up ~0.67 at rest; (2) Damped Track must be
`TRACK_Y` (toward the tail/hem), not `TRACK_NEGATIVE_Y`, or dt flips and the hem
jumps ~0.68. Verified numerically: rest disp ~0.023, correct per-direction push,
no crossing (angular gaps stay positive fwd/back/side/fwd+up).
The earlier **leg-follow rotation** and **radial-outward** models are superseded:
they only rotated the cloth and gave no true clearance.

## Bone dynamics — how CGDive "Rig Creator" does it (studied from the addon + a rig)
Rig Creator's **Physics** (`GN_Fisicas`) is a **baked overlap**, NOT a live spring:
1. Build a parallel `FS_` bone chain (use_deform off, connected chain).
2. Bake pass 1: `FS` Copy-Transforms the originals → bakes the animation onto `FS`.
3. Remove copies; add **Damped Track** (or Stretch To) from each `FS` bone to the
   NEXT one, `influence = power_overlapping` → introduces lag / follow-through.
4. Bake pass 2: originals Copy-Transform the (now lagging) `FS` → bakes the
   secondary motion back onto the real bones; delete `FS`.
Params: frame_start/end/step, power_overlapping, use_stretch_to, power_stretch.
Pros: no live handler, plays back fast, works on any animated chain. Cons: it's a
bake (re-bake if the animation changes); needs existing animation.
For SmartRig skirt dynamics we can add a similar "Bake Skirt Overlap" operator on
the skirt chains, on top of the leg-collision drivers. (rig_creator & kinetica
have NO continuous spring system.)

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


## v1.83 - v1.91 (2026-07-10) - Waist-down professional automation

- **Register Waist from Loop**: select an edge loop (Edit Mode) -> waist ring lands EXACTLY on it,
  stored on the metarig (`sr_waist_loop`); Columns/Rows subdivide ON the loop, 0.00mm drift.
  SMART bone count: mesh density can auto-raise Columns/Rows (user gets an INFO message).
- **Leg containment**: per-column `SKC_leg` DAMPED_TRACK -> `SKC_tgt` targets riding ORG-thigh at the
  fabric knee-ring rest point (zero rest error). Direction-gated engagement props (eL/eR) + knee-rise
  term; cubed leg proximity; a raised/side-kicked leg never exits the cloth, walking stays calm.
- **Gravity**: `SKC_hang` (pelvis-pitch gated) + `SKC_hangS` (below-knee, always on) root-parented
  references - panels hang straight down instead of sticking out with pitched hips or spanning
  taut diagonals between asymmetric knees.
- **Follow Body = part-of-the-body mode**: Surface Deform binds to `SR_SitProxy` (body copy minus all
  arm/hand verts - binding to the full body grabbed hand faces and spiked). Mask graph-smoothed,
  full below the waist; EVERY waist-down automation driver multiplies by (1 - follow) -> one slider
  crossfades bones-automation <-> pure body deform. Sitting = raise the slider.
- **Corrective smooth**: KAN_WaistSmooth (waist->hem, sleeves excluded, tied to Cloth Smoothing
  slider, 0.5x10 - higher settings balloon). KAN_AntiPen stays LAST: no body penetration ever.
- **Smart modifier stack**: canonical Armature > SitFollow > Smooth > WaistSmooth > AntiPen; live
  ERROR + Fix button when the user drops a modifier inside the stack; auto-fixed at Generate.
- **Self-healing weights**: stale DEF-skirt grids rebuilt; torn-chest (arm weight on torso fabric)
  and UNDER-WEIGHTED fabric (<0.6 total) auto-detected -> analytic polish at Generate. Weight-eating
  passes now carry GEOMETRIC guards for every other fabric zone (sleeve regression lesson).
- **Bone organization**: all SKC_* machinery -> hidden Skirt (MCH); DEF/ORG/MCH hidden; controls in
  Skirt (Master)/(FK)/(Tweak) + Sleeves collections with Rig Layers rows. Zero orphan bones.
- Floor system built then REMOVED per design decision (sitting handled by Follow Body instead).

## v1.92 - v1.94.1 (2026-07-17) - Face System foundation (Storm architecture, FaceIt UX)

New module `face.py` - the first slice of the design-doc Face System (section 5):
- **UX = FaceIt, rig = Storm**: register/auto-detect meshes -> auto landmarks ->
  user adjusts -> one-click build. The rig underneath stays LIVE (animator rig),
  ARKit-52 / visemes will later be BAKED FROM it (not instead of it).
- **Auto landmark detection** (`detect_landmarks`): eye centers from the eyeball
  meshes; centerline front profile -> nose / lips / chin as descending frontness
  maxima with banded search (chin band must start 0.45*ipd below the lips or the
  lower lip wins); jaw pivot (TMJ) at eye_z - 0.22*D, 58% head depth; ears =
  centroid of side-most head verts. Everything scales by IPD - no absolute units.
- **Markers**: `SR_FaceMarkers` empties, same colored GPU glow as body markers
  (wizard.py `_face_marker_items`, 0.55x size), .R follows .L via COPY_LOCATION
  invert_x, center markers locked to the centerline. Labels drawn when selected.
- **Build Face Base** (steps UI card 1-2-3): DEF-jaw (pivot->chin) + CTL-jaw
  (roll forced local X = +world X so +X rotation opens DOWN), master-mouth
  placeholder, master/DEF/CTL eyes (DAMPED_TRACK aim at CTL-eye targets under a
  CTL-eyes master), DEF-ear.L/R. Attaches to the generated rig (ORG-head/DEF-head)
  or builds standalone SR_FaceRig. Eyeballs rigid-bound (1.0 group, like ARP).
- **Analytic mask weights** (no painting): jaw field = smoothstep band around the
  pivot->mouth plane x back-fade x under-chin fade x LATERAL CAP at the pivots
  (without the x-cap the neck/shoulder side verts grabbed 0.5 jaw weight)
  x radial safety. Ears = radial falloff, own side only. On a bound rig the
  fields are CARVED out of DEF-head (re-runnable: previous carve undone first).
- **Verified numerically** (installed addon, not live edits): chin arc 0.056 =
  expected 0.0559 at 18 deg, NOTHING outside the head moves (max 0.00000),
  eyeball rotates about a FIXED center (0.027 surface / 0.000 center move),
  ear 12mm at marker with nose/eye/other-ear at 0. L/R displacement symmetry 0.
- **TRAP fixed**: `_facial_autodetect` picked ORPHAN eye meshes (asset-library
  duplicates parked at the origin, not in the scene) -> landmarks collapsed to
  z~0. `_valid_eye` now requires scene membership + head-height proximity, and
  heals the slots. Same family as the "orphan body" trap.
- OpenGL viewport renders do NOT capture the POST_PIXEL glow overlay - check
  glow live in the viewport, verify placement numerically.

NEXT (in order): lips module (ribbon + zipper, Storm ch.3) on master-mouth,
eyelids (ribbon + auto-blink + follow, ch.4), brows/cheeks shape keys + GN
weight split (ch.5-6), teeth/tongue from Asset Library, mouth correctives,
then the expression/viseme/ARKit library + Rhubarb lipsync.

## v1.95.0 (2026-07-17) - SR_FaceGrid: FaceIt-style landmark grid, auto-generated

Studied the installed FaceIt extension live: its `facial_landmarks` mesh (41
half-face verts) is a bone-placement TEMPLATE - `rigging/rig_data.py` maps every
vertex index to head/tail of specific Rigify-face bones (lips .T/.B + corners,
8-vert eye rings -> lids, brow arcs, chin/jaw chains). The landmark layout IS
the deform skeleton; its density at lips/eyes/brows is what enables the ARKit
expressions later. Saeed's read was exactly right.

Soulify's version (`build_face_grid`, own topology, no FaceIt data copied):
32-vert half-face wire mesh (chin chain, lip half-loop, nose->forehead chain,
face outline, nostril+cheek, 8-vert eye ring, brow arc) POSITIONED from the
detected anchors and RAY-PROJECTED onto the head (front rays; jawline/temple
radial rays from the skull axis) - all 32 verts land dist=0 on the surface,
zero manual placement (FaceIt needs a 4-state manual flow here). Mirror
modifier (clip+merge) -> edit L, see R. `GRID_IDX` = semantic name->index map;
`grid_points()` returns L+R world positions; `_lm_ref()` lets the grid refine
chin/lips/corners/brows over the anchor markers at Build. UI step 3 of 5.
The lips/eyelids/brows modules will read their joints from this grid.

## v1.96.0 (2026-07-17) - Fit system REMOVED (Saeed's decision)

The whole garment-FITTING system is gone: `garment.py` (lets_fit / fit_apply /
fit_drape / live_fit_tune), `mannequin.py` (mannequin_match / garment_mannequin),
`fit_wizard.py` (fitwiz_* + SRFM_ markers), the FIT ui_tab (tabs are now
RIG / ANIMATE), the fit-only properties (garment_object, fit_body_object,
mann_*, garment_ease/smooth/scale/height/preserve, fitwiz_*, fit_started) and
their update callbacks, and the fit-marker glow/label hooks in wizard.py.
NOT touched: garment RIGGING (kandura.py, skirt.py) - "garment" mentions there
are its own logic; voxelbind; fit.py (legacy BODY engine, different thing).
Verified on the installed build: fit ops unregistered, RIG/SKIN/ANIM + face
flow all still work.

## v1.98.0 - v1.98.1 (2026-07-17) - Storm parity: widgets, naming, fingers, hair

Compared our face rig against the REAL Storm rig (storm_v1.1.blend linked as
library; 1102 bones, 552 face bones; prefixes DEF/DSP/P/CTL/MSTR/STR/TGT):
- **Storm facts adopted**: DEF-Jaw is HORIZONTAL (pivot at the TMJ, tail
  straight forward at pivot height) and IS the animator control (WGT-Jaw,
  THEME04, scale 1.37). Eye look: MSTR-Eye_target (ARMATURE space-switch)
  -> P-Eye_target (WGT-Eyes_Target) -> TGT-Eye.L/R (WGT-Circle, THEME13).
  MSTR-Mouth (WGT-Mouth, THEME09). MSTR-Eye.L/R (WGT-Cube). Palettes:
  THEME09 ctl / THEME04 jaw+eyetarget / THEME13 eye tgts / THEME10 masters /
  THEME02 teeth/tongue / THEME03 jawline+nose.
- **face_widgets.py**: 16 widget wire shapes extracted from the CC-BY Storm rig
  and embedded as data (455 verts total); ensure()/assign() create LOCAL
  copies tagged "sr_wgt". TRAP fixed in v1.98.1: bpy.data.objects.get() was
  returning the LIBRARY-linked WGT of the same name -> widgets broke if the
  library moved. ensure() now ignores non-local objects.
- **Our bones renamed to Storm convention**: CTL-Jaw (horizontal), MSTR-Mouth,
  MSTR-Eye_target, TGT-Eye.L/R, MSTR-Eye.L/R (+ backward-compat removal).
- **Fingers**: no finger data -> the Rigify TEMPLATE fingers were left floating
  in mid air (user screenshot). metarig.py now REMOVES all template finger
  bones when none are placed (same policy as the face-bones removal) with a
  console note. AI path: arp_ai remembers the last-good ai_tools_path in
  user CONFIG (soulify_ai_path.txt) - per-scene props died with new files.
- **Hair**: skin_hair slot + autodetect ("hair" in name, near head) + rigid
  bind to DEF-head. Verified: hair follows a 20-deg head turn (0.126 disp).
- **INCIDENT (scene, not addon)**: repeated bind/cleanup cycles left body.001
  with METERS data + 0.01 object scale (100x shrink). Root: Scale Fix applies
  the scale destructively; later cleanup restored a stale matrix. Character
  Check card (backlog #1) must detect data-vs-scale inconsistency. Scene fixed
  by re-basing all four meshes under the rig with identity/0.01 as data units
  demand.
- Verified end-to-end after all fixes: jaw 1184 verts, chin 0.065 opens DOWN,
  nose/chest 0; eyes follow MSTR-Eye_target; hair follows head; zero rest
  deformation on the eye shell; all widgets local.

## v1.99.0 - v1.99.3 (2026-07-17) - Full Storm control layout + STORM_SPEC

- 42 face controls with Storm's exact names/widgets/palettes, positioned from
  SR_FaceGrid: MSTR-Face_upp/low, Brow_all+in/mid/out, Lid_upp/low, Cheek_all+
  in/mid/out, Lips_main/local1/local2/corners (MCH 50% jaw delta-follow),
  MSTR-Nose. Every local control drives a DEF twin (child bone) whose weights
  are carved PROPORTIONALLY from the existing deform weights (CAP 0.65,
  rebuild-safe strip+renormalize). CH-storm appended beside the character.
- STORM_SPEC: per-bone direction/Z-roll/length(/IPD)/widget/scale/palette
  measured from the real Storm rig and applied as a post-pass - hand-tuned
  orientations looked amateur (flat halo widgets); measured axes fixed it.
- CRITICAL LESSON (v1.99.2 collapse -> 1.99.3): COPY_TRANSFORMS pairs MUST
  share the rest orientation, or the mesh bends AT REST. DEF-jaw now copies
  CTL-Jaw's edit transform exactly; corner-follow rebuilt as delta-follow
  (LOCAL_OWNER_ORIENT/LOCAL/BEFORE, inf 0.5). REGRESSION TEST after every
  face build: evaluated rest mesh == base mesh (max disp 0.00000).
- Storm's face masters are legitimately HORIZONTAL halo circles around the
  head - not a bug.

## v1.99.4 - v1.99.5 (2026-07-17) - widget cs_rot/cs_tr + Register Selected Loop

- "Widgets flipped/lying flat" root cause: bone axes matched Storm EXACTLY,
  but Storm orients most widgets via custom_shape_ROTATION (triangles X+90,
  lower ones +180 Z, eye circles X+90, Eyes_Target Y+90, Jaw X-0.43) and
  offsets via custom_shape_TRANSLATION (bone-space, scales with bone length -
  transfer RAW). STORM_SPEC now carries both; never transfer a widget without
  its cs_rot/cs_tr.
- Register Selected Loop (face): select the mouth or an eye-socket edge loop
  in Edit Mode -> auto-classified (mouth / eye.L / eye.R), markers + grid
  verts snap exactly onto the loop, face auto-rebuilds. Verified with the
  user's 48-vert mouth loop. Face ops force OBJECT mode first (Edit-Mode
  session crashed VertexGroup.add).
