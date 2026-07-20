# Soulify - Changelog

> Each version is also tagged in git (`git tag`) and kept as a versioned zip.
> To roll back safely: `git checkout v<X.Y.Z>` (source) or install the matching
> `soulify_v<X.Y.Z>.zip`. Never delete old tags/zips.

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

