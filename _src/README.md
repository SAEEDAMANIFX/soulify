# Soulify

A Blender add-on that auto-rigs any character with a marker wizard, builds a
**Rigify** metarig fitted to the markers, adds Rigify samples and a **Short Skirt
(cloth)** sample, and skins the mesh — all from one ARP-style **Rig / Skin / Misc**
panel in the 3D-View sidebar (`N` → SmartRig).

Author: Saeed · Repo: https://github.com/SAEEDAMANIFX/smartrig-pro

## Install
1. Zip the `smartrig_pro/` folder (or use the provided `smartrig_pro.zip`).
2. Blender → Edit → Preferences → Add-ons → Install… → pick the zip → enable
   **Soulify**. Rigify must be enabled too.

## Features
- **Rig tab:** guided marker placement (body → feet → hands), per-finger picker,
  professional bone roll (flat foot), Build Rigify Metarig (no face), editable
  metarig, Rigify samples (collapsible per group), **Short Skirt (cloth)** sample,
  Re-fit to markers, Back to Metarig (Edit Mode), Re-generate.
- **Skin tab:** Bind engine (Heat / Envelope), Split Parts (body ignores skirt
  bones, skirt follows its own), Preserve Volume, Bind / Unbind, **Skirt Dynamics**
  (leg-follow: leg rotation rotates the skirt, no distortion).
- **Misc tab:** spine/neck/clavicles/mirror + finger detection settings.
- **Display** controls (In Front, Axes, Names, Wireframe, Display As) and
  **Align & Wireframe** tools.

## Important usage notes
- After fixing a scene (clean rigs, bind), **SAVE the .blend** — scene changes are
  not stored in the add-on and are lost on reopen.
- Workflow: Build Metarig → (samples / skirt) → **Generate** → Skin: **Bind** →
  Skirt Dynamics → **Apply Leg Follow** → move the legs with the IK foot.
- **Cancel / Start Over** fully cleans everything SmartRig created (and unbinds
  the meshes) without touching other rigs.

## Docs
- `smartrig_pro/DEVELOPMENT_NOTES.md` — every problem & its solution.
- `smartrig_pro/LESSONS.md` — short do-not-repeat rules.

## Version history (highlights)
- 1.16.x — leg-follow auto-invert (skirt swings OUT not into body); full scene
  cleanup on Reset/Cancel; bind removes double modifiers; Preserve Volume off.
- 1.15.x — leg-follow (Pierrick-style) replaces Floor collision (no shear);
  restored bind/unbind; heat-bind skirt to its own bones (no tearing).
- 1.13–1.14 — ARP-style skirt collision experiments (Floor planes + drivers).
- 1.11–1.12 — tabbed Rig/Skin/Misc panel; dedicated Skin (bind) section.
- 1.8–1.10 — Short Skirt from real geometry (Manual/Separate/Merged), live
  Columns/Rows, smart binding, custom skirt icon, per-group collapsible samples.
- 1.6–1.7 — flat foot roll; Re-fit/Back-to-Metarig; Display section; skirt sample.
- ≤1.5 — marker wizard, metarig build, fingers, bone-roll tools.
