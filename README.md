# Soulify — Give it a soul

**Blender addon: automatic character rigging (body + face) with a Storm-grade control layout.**

> ## ⚠️ حالة تجريبية — Experimental
> **هذا المشروع في وضع تجربة وتطوير نشط وفيه أخطاء معروفة.**
> This project is under ACTIVE development and is EXPERIMENTAL. It contains
> known bugs and unfinished systems. Use at your own risk and please report
> issues.

## What it does
- Marker-based body auto-rigging on top of Rigify (AI-assisted proportions).
- Face System (in progress): auto landmark detection, FaceIt-style landmark
  grid, Storm-style face controls (jaw, eyes, eyelids, brows, cheeks, lips,
  nose) with analytic weights.
- Garment rigging (skirt / kandura / sleeves), weight-editing tools, IK/FK.

## Status / known gaps
- Face deformation is currently bone-based; the lips ribbon + zipper,
  auto-blink and shape-key recipes (Storm ch.3-8) are the next milestones.
- Strong mouth-corner pulls smear the lips until the ribbon lands.
- Character Check (unapplied scale / duplicate copies detection) not yet built.
- See `DEVELOPMENT_NOTES.md` and `UX_AUDIT_FULL_RIG.md` for the full ledger.

## Credits
- Face control widget shapes are extracted from the **Storm** character rig by
  **Blender Studio** (studio.blender.org/projects/storm/), licensed **CC-BY**.
  Storm © Blender Studio — thank you for the amazing course and rig.
- Built with Rigify (Blender).

## License note
The AI packages (`AI/`) and any third-party proprietary sources are NOT part
of this repository and are never distributed.
