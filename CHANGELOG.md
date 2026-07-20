# Soulify - Changelog

> Each version is also tagged in git (`git tag`) and kept as a versioned zip.
> To roll back safely: `git checkout v<X.Y.Z>` (source) or install the matching
> `soulify_v<X.Y.Z>.zip`. Never delete old tags/zips.

## v2.9.0 - Clean, link-ready character collection (2026-07-20)

- **Improved** "Organize Character (Link-Ready)": build-only objects that an
  animator never needs - the Rigify metarig, ALL body/face markers, marker
  collections, grids, guides, reference - are now moved to a SEPARATE
  `SR-build-<name>` collection that is a SIBLING of `CH-<name>` (not a child) and
  EXCLUDED. So Appending / Linking `CH-<name>` gives a clean character: rig + GEO
  meshes + WGT widgets + FUNCTIONAL deform helpers (lattices/ribbons/hooks) only,
  with no marker/metarig junk dragged along. The rig is renamed to `RIG-<name>`;
  the v2.8.0 stamp keeps it recognised after the rename.

## v2.8.1 - Distinguish MULTIPLE rigs (context-aware) (2026-07-20)

- **Added**: `_generated_rig()` is now context-aware for scenes with MORE THAN ONE
  Soulify character. It prefers the rig you are working on - the active armature,
  the rig deforming the active mesh, or a selected rig - before the global stamp
  scan. So with two characters, selecting one makes every Soulify tool act on it.
  Combined with the v2.8.0 stamp, rigs renamed to meaningful names
  (e.g. `RIG-Harold`) are still recognised. Verified with two stamped rigs.

## v2.8.0 - Portable / robust rig recognition (send, append, link) (2026-07-20)

Recognise a Soulify rig ANYWHERE, the way Auto-Rig Pro does - so you can send the
.blend to a friend who has Soulify, or Append / Link it into another file, and
every Soulify tool still finds the rig for animation / editing.

- **Added**: every generated rig is STAMPED with a `soulify_rig` custom property
  (+ `soulify_metarig`). `metarig._generated_rig()` now recognises a rig by that
  stamp as a final fallback - so it survives a RENAME, an Append (`.001` suffix),
  a Link, a broken Rigify metarig link, or being opened on another machine.
  Prefers a local rig, else accepts a linked one.
- Existing rigs are stamped on the next Generate; a one-off stamp was applied to
  the current project's rig. All face / eye / animation tools resolve the rig via
  `_generated_rig()`, so they inherit this recognition automatically.
- Verified: renaming the rig to `RIG-SR_Metarig.001` AND clearing the Rigify link
  still resolves the rig by stamp.

## v2.7.2 - Rig panel survives marker deletion (2026-07-20)

- **Fixed**: the Rig panel showed the initial "Place Body Markers / Select your
  character first" state whenever the body markers were absent, even if the rig
  was already generated - because the `not has_markers` early-return ran BEFORE
  the "metarig exists -> show rig tools" branch. Now the START state only shows
  when there are no markers AND no `SR_Metarig`, so a generated rig always shows
  its tools (Marker Tools, Rigify, Face/Eye) regardless of marker presence.

## v2.7.1 - Fix stray marker glow at world origin (2026-07-20)

- **Fixed**: the marker overlay (`wizard._draw_glow`) drew its glow for markers
  that merely EXIST, testing `hide_get()` (the eye icon) instead of
  `visible_get()`. Stale body markers parked in an EXCLUDED collection at the
  world origin therefore drew a gold "disc" on the floor. Now it uses
  `visible_get()`, so excluded / hidden markers never draw. (A restart clears any
  leaked in-memory draw handler from a prior live-reload.)

## v2.7.0 - Professional Ribbon Eye Rig (2026-07-20)

Complete rewrite of the eyelid rig (module `eye_sample.py`) + ARP-style widgets.

### Added
- **Ribbon eyelid system.** Deform bones slide over the cornea (Damped-Track,
  constant length = rotate about the eye centre, no poke-through). Upper & lower
  lids close onto ONE analytic smooth seam -> the lid closes with NO kink
  (`_seam_pt`). Guide ribbon mesh built from the registered loop (extruded).
- **Orange tweak circles now RIDE the lid** on blink (child of the moving target)
  and stay grab-able to sculpt the lid shape at any state.
- **ARP-style eyes target widget**: a "peanut" master (two lobes + concave waist)
  with an L/R circle inside each lobe, all facing the camera.
- **Optional blink-driven corrective shape keys** for the full-close: buttons
  `Sculpt Closed`, `Edit Closed` (both L/R), `Finish Correction (Conform)`, and
  `Mirror L->R / R->L`. Correctives are OFF at rest, fade in only as the eye
  closes. (Lid MOTION itself needs no shape keys.)
- Lid bone count raised to 8/8 (still adjustable in the panel).

### Changed / Fixed
- **Weights reworked**: partition-of-unity along the lid + smoothstep fall-off +
  8x smoothing. Stale Rigify head weight (DEF-spine.006) is now STRIPPED off the
  lid verts (it was holding the lids half-open = a slit).
- **Seal**: upper/lower skin margins now CROSS (`OVERLAP`, tapered) so the eye
  seals fully, corner-to-corner (columns reach the canthi; corner bones dropped
  from the weighting).
- **Corrective Smooth** modifier de-facets the wide-open / full-close skin, but is
  MASKED to EXCLUDE the meeting line (smoothing it re-opened the seal). Result:
  sealed close AND smooth extremes, no shape keys.
- Fixed the RIGHT-eye lid-line widget zig-zag (points now ordered by u).
- `clear_eye_rig` forces Object mode first (VertexGroup edits fail in Edit mode).

### Notes
- Eyelashes are a separate mesh (body.002/003); drive them with a Surface Deform
  bound to the face in the OPEN pose - they are NOT part of the lid rig.
- The disc at the world origin is the rig's normal ROOT control, not a leftover.

