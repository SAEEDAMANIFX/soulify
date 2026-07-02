# SmartRig Pro — Developer Handover

**Project:** SmartRig Pro — Blender (4.0+, developed against Blender 5.1) Python auto-rigging addon, body + face + garment fitting (Let's Fit), Rigify-based. Author: Saeed.
**Current version:** `bl_info["version"] = (1, 20, 27)` in `_src/smartrig_pro/__init__.py`.
**Source of truth (edit here):** `C:\Users\hasht\Claude\Projects\smart rig\_src\smartrig_pro\`
**Installed copy (never edit directly):** `bpy.utils.user_resource('SCRIPTS')/addons/smartrig_pro`

Companion docs you MUST read before touching code:
- `_src/smartrig_pro/DEVELOPMENT_NOTES.md` — problem/solution log (collision evolution, reset behaviour, dev discipline).
- `_src/smartrig_pro/LESSONS.md` — "do NOT repeat these mistakes" (collision evolution log, drivers/autorun, foot roll, general discipline).
- The bundled knowledge skill **smartrig-pro** and the session memory notes:
  `smartrig-skirt-edgeflow`, `smartrig-skirt-rebuild-rest-coords`, `smartrig-lift-blowup-rotation`, `smartrig-live-tune-needs-update-callback`.

---

## 1. Architecture overview

`__init__.py` registers (and `importlib.reload`s, so hot-reinstall works) these modules in order:

| Module | Responsibility |
|---|---|
| `properties.py` | Single `Scene.smartrig` PropertyGroup: all UI state + live `update=` callbacks (`_skirt_update`, `_jiggle_update`, `_jiggle_force_update`, `_skirt_collide_update`, `_skirt_follow_update`, …). |
| `utils.py` | Helpers, incl. `read_world_coords` (EVALUATED mesh) and `read_rest_coords` (base `obj.data.vertices` @ matrix_world — see Lessons). `REF_NAME` = SR_Reference. |
| `icons.py` / `assets/` | Custom preview icons + guide images for the marker wizard. |
| `detect.py`, `finger_ai.py`, `finger_render_ai.py`, `models/*.onnx` | Neural (ONNX) joint-proportion / finger detection. |
| `markers.py` | Marker empties (spine_root, ankle.L, ball.L, foottip.L, …), guided click-placement, `full_cleanup()` (complete reset — removes markers, SR_Metarig, RIG-*, SKC_*, orphan WGT-*, unbinds meshes; never touches non-SmartRig rigs). |
| `fingers_manual.py` | Manual palm/finger/toe click placement. |
| `fit.py` | `compute_joints(props)` — fits joint positions from markers + mesh geometry. |
| `metarig.py` | Builds the Rigify metarig (`META_NAME = "SR_Metarig"`), `_generated_rig()`, Generate wrapper. |
| `generate.py`, `skinning.py` | Rigify generation + smart weighting / Bind (Skin tab). |
| `skirt.py` | **The active subsystem.** ~3300 lines: analysis, bone placement, Rigify tentacle chains, leg collision, jiggle/wind/blow-up solver, masters, follow-body, anti-penetration, bind/bake/remove. Details in §2–3. |
| `wizard.py`, `ui.py` | N-panel UI (`View3D > Sidebar > SmartRig`) + guided wizard flow. |

**UI flow (`ui.py: SMARTRIG_PT_panel.draw`):**
1. Entry gate: only a big **"Let's Rig"** toggle (`props.rig_started`) until pressed.
2. Question: **"What are you rigging?"** → `smartrig.pick_mode` sets `CHARACTER` or `PARTS`.
3. Tabs (`props.ui_tab`, incl. SKIN) + mode switch (`props.rig_mode`).
   - **Character:** guided marker placement (body → feet ball/tip → hands yes/no → palm/fingers) → Build Rigify Metarig → bone-roll tools / Options / Rigify samples → Generate.
   - **Parts (`_draw_parts`):** standalone accessories, no body/markers. Only **Cloth → Skirt** is functional; *Appendages (tail/ears/wings)*, *Props*, *Face-only* are shown as "planned". Includes Generate Rig / Back to Metarig / Delete Rig-Start Over.
4. `_draw_skirt_settings(box, context, props, standalone=)` is shared by Character and Parts panels.
5. Extra Item-tab panels (`SMARTRIG_PT_skirt_item`, `SMARTRIG_PT_chest_item`) expose live/keyframable collision + jiggle + wind + Blow Up + follow + anti-pen sliders while posing.

"Cancel / Start Over" and "Reset" both call `markers.full_cleanup()` — a full, clean delete.

---

## 2. Skirt pipeline (analyze → route → place → emit → generate)

All in `skirt.py`. Bone naming: `skirt.CC.RR` (column.row). Vertex-group marker: `SR_Skirt` (non-deform; actual deform comes from `DEF-skirt.*` weights).

**Input** — `skirt_verts_world(props)`: SEPARATE mesh (eyedropper `skirt_object`) → `utils.read_rest_coords` (**REST, never evaluated** — see Lessons); or MERGED → `SR_Skirt` vertex group on `target_mesh`; or MANUAL → `build_manual_skirt` starter ring from body cross-sections.

**Analyze** — `analyze_skirt(ob)` classifies via bmesh boundary loops + quad ratio:
- `TUBE` (2 borders, ≥60% quads), `OPEN` (1 border: wrap/open-front), `LAYERED` (>2 borders: tiers/lining), `MERGED` (no borders), `MESSY` (<60% quads).

**Route** — `build_skirt(props)` picks the placement strategy, in order:
1. **edge-flow** — `_skirt_grid_topo(ob, cols, rows, symmetric, front_ang)` for TUBE/OPEN/LAYERED: walks the mesh's REAL vertical edge loops from waist boundary verts down to the hem (quad edge-loop step: next edge shares no face with the incoming edge, descending Z), resamples each to rows+1 points. Self-check: bails (returns None) unless ≥3 columns and ≥60% of walks descend ≥50% of mesh height (catches solidified/double-wall meshes).
2. **rim-span** — `_skirt_grid_between(co, cols, rows, waist, hem, cx, cy, front)`: columns run EXACTLY from waist rim to hem rim per angle (row 0 on waist rim, last row on hem rim, middle rows follow median mesh radius at interpolated Z). Rims from `_rim_rings(ob)` = topological boundary loops (highest-Z loop = waist, lowest = hem), falling back to 5% Z-bands for folded/thick hems. This is the "skirt starts at the waist OPENING, not the waistband top" fix (v1.19.144).
3. **angular** — `_skirt_grid(co, cols, rows, front_ang)`: robust fallback over full Z extent. Rejects radius outliers per Z-slice, centre = **bbox midpoint** (not median), columns at EXACT even target angles from Front, sampling only the median radius in a wedge.

The chosen `kind` and `method` are stored on the metarig (`mo["sr_skirt_kind"]`, `sr_skirt_method`, `sr_skirt_cols_built`) and shown in the UI ("Detected: clean tube → edge-flow").

**Emit** — `_emit_chains(mo, grid, rows)`: unhides SR_Metarig (Generate hides it — a hidden object can't enter Edit Mode), deletes old `skirt.*` bones, creates one chain per column parented to `spine`/`spine.001`, roll aligned so local Z points radially outward (panels swing radially, never cross), tags each root `rigify_type = "limbs.simple_tentacle"`.

**Live rebuild** — `live_rebuild(context)`: changing Columns/Rows (`_skirt_update`) rebuilds in place for mesh-driven modes only (never MANUAL), preserving Edit Mode.

**Generate** — normal Rigify Generate (Character mode "Add Short Skirt" adds to the body metarig; Parts mode builds a standalone metarig first). Post-generate deform path (memorize this): `tweak_skirt.CC.RR` (positions) → `ORG-skirt` (COPY_TRANSFORMS + **STRETCH_TO next tweak**) → `DEF-skirt` → mesh. **The skirt shape is defined by tweak POSITIONS, not bone rotations.**

**Standalone (Parts mode)** — `SMARTRIG_OT_rig_skirt_standalone` (`smartrig.rig_skirt_standalone`, "Build Skirt Metarig"): requires SEPARATE source; creates a fresh SR_Metarig stripped to one `spine` root (`basic.super_copy`) at the skirt's bbox-centre top, then `build_skirt`. **Metarig only** — user tweaks, then presses Generate Rig separately (same flow as Character mode).

### Post-generate feature layers (all removable, all live-tunable)
- **Leg collision** ("compass model", v1.19.12+): `add_skirt_collision`; `SKC_master` bone custom props (`collide`, `collide_dist`, `collide_spread`, `collide_dist_falloff`) drive Floor-constraint/driver setup; `_resolve_colliders` maps any leg bone name to `DEF-thigh.*`. Live via `live_kilt_tune` / `live_tune`. Constraints prefixed `SK_FLOOR/SK_FOLLOW/SK_LIMIT/SK_DT/SK_RIDE`, helper bones `SKC_*`. **Requires "Auto Run Python Scripts" ON** (Python drivers).
- **Jiggle** (live spring): `skirt_jiggle_handler(scene)` on `frame_change_post` (@persistent, always registered in `register()`; no-ops without jiggle rigs). Spring integrates `SKC_jig.CC.RR` (skirt) and `SKC_jigB.*` (chest — separate `chest_*` params) toward the animated goal, with wind/gust/billow/gravity forces. State in `_JIG_STATE`; resets on frame jumps. Bakeable to keyframes (`SMARTRIG_OT_bake_jiggle` / `chest_bake`; baked flag disables the live solver).
- **Blow Up** (`jiggle_wind_lift`, 0–20): inside the same handler but **NOT a jig rotation** — TRANSLATES `tweak_skirt.*` bones up + radially outward, progressive by row (`rf = row/5`, uncapped so the tip tweak lifts past the hem). See Lessons.
- **Region masters**: `add_skirt_masters` — a global `skirt_master` (compass-dial widget) + N sector masters (squircle widgets) snapped onto the evaluated cloth surface; column tops re-parented per sector. `_organize_skirt_bones` sorts everything into coloured bone collections.
- **Follow Body** (sit/cling): Surface Deform modifier, blend = `skirt_follow_body`. SEPARATE skirts only.
- **Anti-Penetration**: Shrinkwrap (outside) with `skirt_antipen_offset`. SEPARATE only. `_order_skirt_deformers`/`skirt_mods_order_ok` + `SMARTRIG_OT_skirt_fix_order` police modifier order.
- **Corrective Smooth**, **Bind/Unbind** (`bind_mesh` with structure-aware `_smart_skirt_weights`), **Remove Skirt** (`remove_skirt` + `check_skirt_integrity`).

---

## 3. Key properties (`Scene.smartrig`, properties.py)

Placement (all with `update=_skirt_update` → live rebuild):
- `skirt_source`: `SEPARATE` | `MERGED` | `MANUAL`. SEPARATE = full feature set (collision+jiggle+follow+anti-pen); MERGED = collision+jiggle only; MANUAL = starter ring.
- `skirt_object` (eyedropper), `skirt_columns` (default 8, min 4), `skirt_rows` (default 2), `skirt_length` (MANUAL only).
- `skirt_front_axis` (enum, **default '-Y'** = Blender character front): columns at even angles from this axis → always a front-centre + back-centre column, equal L/R (i ↔ cols−i mirror).
- `skirt_symmetric` (default True): symmetric PLACEMENT angles only — geometry is never averaged (asymmetric skirts keep their true shape).

Collision: `skirt_collide`, `skirt_collide_dist`, `skirt_collide_spread`, `skirt_collide_falloff`, `skirt_collider_l/r` (default DEF-thigh.L/R), `skirt_follow` (leg follow), `skirt_limit_deg`.

Jiggle/forces (ALL have `update=_jiggle_update` or `_jiggle_force_update` — mandatory, see Lessons): `skirt_jiggle`, `jiggle_amount/stiffness/damping`, `jiggle_gravity`, `jiggle_wind`, `jiggle_wind_dir/turb/speed/billow`, `jiggle_wind_lift` (Blow Up), `skirt_jiggle_segments`; chest equivalents `chest_*`.

Extras: `skirt_use_masters`/`skirt_masters`, `skirt_smooth`/`_factor`/`_iter`, `skirt_follow_body`, `skirt_antipen_offset`.

---

## 4. Build / verify workflow (DO NOT change this)

The addon is developed **live** against a running Blender exposing MCP tools (`mcp__Blender__*`):

1. **Bump the version** in `_src/smartrig_pro/__init__.py` (patch number) — this is how you prove the running copy is your copy.
2. **Copy** the source tree into `bpy.utils.user_resource('SCRIPTS') + "/addons/smartrig_pro"` (copytree, overwrite).
3. **Disable then re-enable** the addon (`bpy.ops.preferences.addon_disable/enable(module="smartrig_pro")`) — the `importlib.reload` loop in `register()` picks up all modules.
4. **Verify NUMERICALLY** via `mcp__Blender__execute_blender_code`: measure bone head/tail positions and mesh deformation in cm; then screenshots for the visual check. "Looks right" is not verification.
5. Confirm the loaded version matches what you shipped before trusting any result.

Hard rules:
- **NEVER call `rig.animation_data_clear()`** — it wipes Rigify's drivers and destroys the rig.
- When verifying live-tune changes from Python: call `bpy.context.view_layer.update()` AFTER setting a prop and BEFORE reading `pose.bones[...].tail`, or you read the stale pose and wrongly conclude the fix failed.
- Don't run scene-mutating operators just to inspect; prefer read-only measurement.
- Users need **"Auto Run Python Scripts"** enabled or the collision drivers silently do nothing.

---

## 5. Non-obvious lessons (hard-won — do not relearn these)

1. **REST coords, not evaluated** (v1.19.122): any bone/marker fitting that can run AFTER the mesh is rigged must read `utils.read_rest_coords` (base vertices), never `read_world_coords` (evaluated). Reading the deformed mesh fits bones to the momentary jiggle/blow-up pose → intermittent misplaced bones on Columns/Rows rebuild.
2. **Blow Up = tweak TRANSLATION, not jig rotation**: the deform follows tweak POSITIONS (ORG STRETCH_TO between consecutive tweaks). Rotating `SKC_jig`/FK/`ORG` only moved the hem row; translating `tweak_skirt.*` moves all rows. Set `tb.location = tb.matrix.to_3x3().inverted() @ world_off`; reset to 0 when blow≈0; row factor uncapped so the tip lifts past the hem.
3. **Live sliders need `update=` callbacks**: the jiggle solver only runs on `frame_change_post`; without `update=_jiggle_force_update` (which re-runs `skirt_jiggle_handler`) a force slider does nothing on a paused frame — users read this as "broken". Every force prop must carry the callback.
4. **Edge-flow first, angular as fallback**: clean quad skirts get bones fitted to real vertical edge loops (`_skirt_grid_topo`); angular median sampling produces uneven per-column curvature and is only for messy/merged meshes. Max clean columns = number of waist verts.
5. **Ring centre = bbox midpoint, NOT median**: dense vertex clusters (thick waistbands) pull the median metres off-axis and skew every column angle. Applies to `_skirt_grid`, `_skirt_grid_topo`, and the standalone root bone.
6. **Columns at EXACT even target angles from the Front axis**, reading only the radius in a wedge — guarantees front/back-centre columns and equal L/R; clusters can't drag a column off its angle. `skirt_symmetric` never averages geometry (that wrongly symmetrized asymmetric clothing — removed).
7. **START/END = topology rims, not Z extent** (v1.19.144): the waist is the waist BOUNDARY LOOP, which can sit BELOW a folded waistband's top. `_rim_rings` + `_skirt_grid_between` build columns waist-rim → hem-rim. Snapping endpoints to the nearest rim VERT (ignoring angle) distorts bones — take only radius+Z from the rim, keep the column angle (`_anchor_ends` / `_rim_rz`).
8. **Unhide the metarig before Edit Mode** (`_emit_chains`): Generate hides SR_Metarig; a hidden object can't be made active → "Context missing active object".
9. **Foot bone roll must be flat** (X horizontal) — LESSONS.md, fixed v1.6.3.
10. **Full cleanup discipline**: Reset/Cancel must remove everything SmartRig created and unbind meshes, but never touch foreign rigs (e.g. Auto-Rig Pro's). Already implemented — keep it that way when adding features (register removal in `full_cleanup` / `remove_skirt`).
11. **Verified skirt matrix** (v1.19.144): A-line, pencil, circle, mermaid, pleated, wrap (OPEN), tiered (LAYERED), thick double-wall — all classify correctly, keep front+back centre columns, full waist→hem span. Re-run this matrix after any placement change.

---

## 6. Suggested next steps / open items

- **Parts-mode planned categories**: *Appendages (tail / ears / wings)*, *Props (rigid objects)*, *Face-only* are placeholder boxes in `_draw_parts` ("planned"). The skirt standalone flow (build metarig → Generate) is the template; tails map naturally onto the same `limbs.simple_tentacle` emit path.
- **Full cloth-simulation mode**: a real Cloth-sim option (vs the current bone-based collision/jiggle) was offered to the user but never built. The current bone approach is deliberate (ARP-Kilt-style); if adding sim, keep it as a separate optional mode.
- **GitHub upload pending**: the project is not yet in a remote repo. Set one up (exclude `__pycache__`, consider LFS or exclusion for `models/*.onnx`).
- **Stale doc**: DEVELOPMENT_NOTES.md "Pending" still lists dynamic jiggle as unimplemented — it shipped (v1.19.26+, live spring + bake). Update when next editing.
- **Face module**: face rigging is part of the product vision; the `anthropic-skills:facial-rigging-blender` skill (Blender Studio Storm workflow) is available as the reference for building it.
- **Skirt polish candidates**: OPEN/LAYERED edge-flow is "best-effort" (primary layer only — multi-layer skirts get one chain set); MERGED skirts have no rim detection (no boundary loops) so they always use angular sampling; `skirt_front_axis` covers only the 4 cardinal axes.

*Handover written 2026-07-01 against v1.19.144.*


---

## 7. Update 2026-07-01 — v1.19.145 -> v1.20.27: skirt recognition fixes + LET'S FIT

### Skirt recognition (v1.19.145-146)
- `_ring_wraps()` in skirt.py: a boundary loop is a rim only if it occupies
  >=8/12 angular sectors around the bbox centre. Slits/holes are ignored; new
  kind `CLOSED` (borders exist, none wraps) -> rim-span over Z-bands.
- ALL topology analysis (`analyze_skirt`, `_rim_rings`, `_skirt_grid_topo`) now
  runs on `bm.transform(matrix_world)` — WORLD space. Imported meshes carry
  object rotations (FBX X=90) and every local-Z test was sideways.

### Let's Fit — automatic garment fitting (NEW module `garment.py`, ~800 lines)
UI: big "Let's Fit" toggle under "Let's Rig" (entry panel) + compact row while
rigging; panel = Garment/Body eyedroppers, Preserve Shape checkbox, Fit/Refit,
live Tune sliders (Ease %, Smoothing, Scale, Height — debounced), Apply/Remove.
Pipeline (all in `garment.py`, LESSONS.md has the full 13-lesson evolution):
1. RECOGNIZE: `_all_rings` (wrapping boundary rings + SVD plane normals),
   `classify_garment` (opening signature: pants = >=2 OFFSET low rings; shirt =
   >=2 offset lateral rings; skirt/dress by top-ring ratio 0.6), auto-ORIENT
   (ring-normal axis -> vertical, smallest significant ring on top; rings must
   have >=40 verts & r >= 0.15*half-extent so slits can't flip it),
   `_ai_landmarks` = detect.py pose ONNX (hips/neck/chest/shoulders, cached on
   body['srf_ai'], other meshes hidden during render; onnxruntime found via
   user-site sys.path fix in detect.has_runtime).
2. PLACE (`auto_place`): anchor by garment class — bottoms at hips, dress with
   a concentric mid waist-ring at the body waist, band-tops (strapless/open
   back) at chest, collared tops at neck; scale from anchor-ring vs body
   radius; full radius-profile cost refines (+-height, x-scale); K_ORIG/K_BASE
   matrices stored for Remove/nudges.
3. CONFORM `conform_shape` (Preserve Shape ON, shape key SRF_Fit):
   TWO ZONES — above body waistline: per-vertex nearest-surface conform along
   stored REFERENCE correspondence (computed at K_BASE placement so Scale
   nudges can't flip sides), push-only 'inflation' + Laplacian-smoothed offset
   vectors + residual re-push; below: radial body-ENVELOPE (max radius per
   band/wedge cell, arm clusters cut at the air gap, shoulder-junction lateral
   cells exempt from cell push but per-vert clamped to the arm envelope),
   3 field iterations, bilinear NaN-aware constraint sampling, per-vertex
   design offsets (rel / rel_s) preserve layers/thickness everywhere.
   Preserve OFF = modifier stack SRF_Snug/Wrap/Smooth/Touchup (shrinkwrap).
4. Live tuning: nudges instant (matrix from K_BASE), heavy conform debounced
   0.4 s via bpy.app.timers token pattern.
Verified: Skirt_5 (CLOSED pleated, waistband hug 1.4 cm, hem deviation 0.00),
bra top (chest), TurboSquid layered wedding gown (154k verts, auto-oriented
from lying-down, waist-ring anchored, bodice/sleeves smooth, 0% back pen at
user tune Ease 1% / Smooth 40 / Scale 0.89 / Height -0.05).

### Next steps (agreed with Saeed)
- Cloth-sim "Drape" button after geometric fit (Marvelous-level finish).
- Bind fitted garments to the rig (skirt flow as template).
- Test pants + sleeved-shirt classification paths on real assets.
- onnxruntime auto-install button in UI (currently pip + user-site path fix).
