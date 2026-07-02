# Chapters 6 & 7 — Cheeks (3) + Teeth and Tongue (2)
Captured 2026-07-02. No comments on any of these 5 lessons.

## 6.01 Main shape keys (Cheeks)
Desc: Same method as brows — main shapes first via temporary armature + duplicate face mesh. See brows chapter for depth.

## 6.02 Lattice deformers (Cheeks)
Desc: In addition to shape keys, a **lattice deformer driving only rotation and scale transform channels** for animator flexibility. **Shape keys = main deformer + lattice = extra deformation on top. "Best of both worlds."**

## 6.03 Splitting weights (Cheeks)
Desc: Split main cheek shape keys into smaller chunks like local brow shapes (3-curve principle).
Downloads (same as brows): `shapekeys_automatic_driver_assignment.py` + `GN-linked_shapekeys.blend` (both CC-BY).

## 7.01 Teeth
Desc: **Lattice deformers as base, driven by control bones** — animator can push a facial pose when needed (bendy teeth for cartoon poses).

## 7.02 Tongue
Desc: **"Keep it simple stupid!"** — tongues easily get over-engineered. Setup = **FK chain + squash & stretch for free**. Simple, predictable, lightweight is key.

## Key takeaways for SmartRig Pro
1. Cheeks = brows pipeline reused (shape keys + GN split) + lattice (rot/scale only) — automation shares code with brows module.
2. Teeth = lattice + control bones; Tongue = FK + squash/stretch. Both simple, fully automatable with zero user weight painting.
3. Rik's tongue philosophy ("KISS") should be a SmartRig design principle across modules — resist over-engineering.
