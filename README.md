# Soulify

**Fit · Rig · Animate — give it a soul.**

A Blender add-on that takes a character from raw mesh to animation-ready:
fits clothing onto the body, auto-rigs the character with a marker wizard
(built on **Rigify**), rigs garments (skirt/cloth), skins everything, and hosts
the animation tools — all from one **Fit | Rig | Animate** panel in the 3D-View
sidebar (`N` → **Soulify**).

Author: Saeed · Repo: https://github.com/SAEEDAMANIFX/smartrig-pro

> **Note:** the addon's internal package/folder is still named `smartrig_pro`
> (kept for backward compatibility with existing scenes — operator IDs are
> `smartrig.*`). The addon's display name, panel and sidebar tab are **Soulify**.

## Install
1. Use the provided `soulify_v1.22.0.zip` (or zip the `_src/smartrig_pro/` folder).
2. Blender → Edit → Preferences → Add-ons → Install… → pick the zip → enable
   **Soulify**. Rigify must be enabled too.
3. Upgrading? Remove the old version first, then restart Blender before installing.

## The three phases (v1.21+ UI)
- **FIT** — automatic garment fitting: pick a clothing mesh (skirt, shirt,
  pants, thobe…), press Fit Garment, tune live (ease/smooth/scale/height),
  Apply. AI landmarks + two-zone preserve-shape conform.
- **RIG** — Build | Skin sections:
  - *Build:* guided marker placement (body → feet → hands), per-finger picker,
    Build Rigify Metarig, Rigify samples, **Short Skirt (cloth)** rig from real
    geometry (edge-flow detection, region masters, leg collision, jiggle,
    Follow Body, Anti-Penetration), Re-fit / Back to Metarig / Re-generate.
  - *Skin:* Bind engine, Split Parts (body ignores skirt bones), smart skirt
    weights, Bind / Unbind.
- **ANIMATE** — cloth & secondary motion (jiggle bake, chest jiggle, wind &
  gravity — live sliders in N-panel → Item), plus the upcoming systems:
  Locomotion (drive bone), Action Packs, Animation Layers, Lipsync,
  Pose Library, Ground Adaptation.
- **Simple | Pro** switch: Simple shows only the essential steps; Pro reveals
  bone roll, align, display and advanced options.

## Docs
- `DESIGN_GARMENT_FACE.md` — the full expansion design (garment modules, face
  system, ANIMATE tab, UI).
- `research/storm-face-course/` — Blender Studio "Advanced Facial Rigging"
  knowledge base (55 lessons + community Q&A).
- `_src/smartrig_pro/DEVELOPMENT_NOTES.md` — every problem & its solution.
- `_src/smartrig_pro/LESSONS.md` — short do-not-repeat rules.

## Version history (highlights)
- **1.22.0** — renamed to **Soulify**; docs updated.
- **1.21.0** — UI restructure: FIT / RIG / ANIMATE phase tabs, Simple/Pro
  level, Build|Skin sub-tabs, ANIMATE tab (cloth bake + planned systems).
- 1.20.x — Let's Fit automatic garment fitting (AI landmarks, auto-orient,
  preserve-shape conform); skirt edge-flow recognition (world-space,
  slit-proof); smart skirt skinning; live jiggle + wind/gravity/Blow-Up;
  Follow Body (Surface Deform); Anti-Penetration; region masters.
- 1.16–1.19 — leg-follow auto-invert; full cleanup on Reset; bone collections
  + master widget; Chest Jiggle.
- 1.8–1.15 — Short Skirt from real geometry; tabbed panel; bind/unbind engines.
- ≤1.7 — marker wizard, metarig build, fingers, bone-roll tools, flat foot roll.
