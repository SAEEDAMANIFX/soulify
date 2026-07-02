# Chapter 5 — Brows (8 lessons)
Captured 2026-07-02, all comments incl. expanded replies.

## 5.01 Why shape keys
Desc: Instead of ribbon mesh (lips/eyes), the traditional way: shape keys for brow deformation + a simple script to automatically assign the many shape keys to the face rig. Straightforward, great control and iteration management.

Comments (important design guidance):
- **Nada Shareef:** will there be a ribbon setup video for brows? → **Rik: not planned, but the setup wouldn't differ much from lips/eyelids — divide main brow bones into smaller bones, then corrective shape keys (like ch8 mouth corners) for furrow lines when brows move inward. "Extract the ribbon mesh from the brows and follow the same steps as the lips."** ← This is the recipe for our alternative bone-based brow module.

## 5.02 Deformation and eye sockets
Desc: DEF bones first — function as **'tweak' bones for small adjustments only**; broad shapes driven by the shape keys created later.

Comments:
- **Matthew Vanscoy + Nick Fisher (feature request):** sample .blend per chapter — main Storm file is rigged differently enough to be unreliable as reference. Rik: will check.

## 5.03 Temporary geometry
Desc: Duplicate face mesh WITHOUT modifiers → clean model sharing same vertex data/count = base for shape keys. TMP meshes not used in final rig.

Comments:
- **Recurring bug (Matthew, Miles):** can't subtract weights on the TMP heads → **cause: Deform checkbox off on the root bone** (same class of bug as ch3 Nick Fisher). Workaround some used: add tiny 0.01 weight to base group. Real fix: enable Deform.

## 5.04 Creating x and z directions
Desc: Additional brow deformations using **temporary armatures**; the x-negative shape needs extra time. (No comments.)

## 5.05 Creating Shape keys
Desc: After successful 6-direction deformations, extract target shape keys stored for one side (left), properly named, then apply to face mesh.

Comments:
- **tone watson (debug case):** "New from objects" shape-key generation failed → **cause: cloned mesh differed from primary mesh (a mask modifier was accidentally applied to the clone before duplicating)**. Vertex data/count must match exactly.

## 5.06 Applying shape keys
Desc: Add the 6 shape keys to the rigged face mesh. **Blender 5.0 new operators generate/update shape keys for both L and R simultaneously.** Python script auto-assigns drivers to the right bone directions.
**Download on this page: `shapekeys_automatic_driver_assignment.py` (3.9 KB, CC-BY).** (No comments.)

## 5.07 Splitting weights
Desc: Split main brow shapes into smaller regions via 3-curve principle, using a **Geometry Nodes setup emulating 'linked shape keys'**.
**Downloads: `shapekeys_automatic_driver_assignment.py` + `GN-linked_shapekeys.blend` (116.9 KB, CC-BY).**
Append via: File > Append > GN-linked_shapekeys.blend > NodeTree > linked_shapekeys.

Comments:
- Yunior (2026-07-02): modifier doesn't appear after append (open).

## 5.08 Local controls
Desc: Assign extracted local shape keys to face mesh; python script assigns the right driver to the right axis automatically; then adjust weights/shapes as needed.
**Download: `shapekeys_automatic_driver_assignment.py`.**

Comments:
- **Dmitry (Blender 5.1 issue):** vertex-group dropdown missing in the GN modifier weight selection — **workaround: type the vertex group name manually.** Possibly a 5.1 UI regression.
- "Pose shape keys panel" — it's an addon; appears as an extra tab in the shape keys panel (Pose Shape Keys addon by Blender Studio).

## Key takeaways for SmartRig Pro (brows module)
1. Shape-key recipe: TMP mesh (no modifiers, same vertex count) → temp armatures pose 6 directions (±X, ±Z, etc.) → extract keys one side → Blender 5.0 operators mirror L/R → auto driver assignment script → GN linked-shapekeys to split into local regions (3-curve).
2. The two bundled scripts + GN .blend are **CC-BY** — legally reusable inside SmartRig Pro with attribution. GN-linked_shapekeys.blend is downloadable and appendable — our addon can ship it.
3. Both brow recipes confirmed by Rik: (A) shape keys as taught; (B) ribbon/bones like lips + correctives for furrow — exactly our modular "recipe per region" plan.
4. Validation checks for our wizard: Deform checkbox on, TMP mesh vertex parity, no stray modifiers on clones.
5. Dependencies: Pose Shape Keys addon (Blender Studio), "Add Selected Bones to Vertex Group" (Mochi_lin), Blender 5.0+ shape-key mirror operators.
