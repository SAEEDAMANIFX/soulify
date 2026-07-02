# Chapter 3 — Lips (9 lessons)
Captured 2026-07-02, all comments incl. expanded replies. The heart of the course.

## 3.01 3 Curve principle
Desc: Theory very important for rigging the lips — also used later for eyes, brows and cheeks. Don't skip.
Comments: Kenzie ("so that's how Pixar does it — deformations are universal, even in anime for extreme expressions"), Luciano Muñoz (praise, 10 likes).

## 3.02 Ribbon mesh
Desc: Create the ribbon mesh for the lips applying the 3-curve principle. The ribbon is the guide for individual lip bones → smooth, predictable lip deformation.

Comments:
- **Nada Shareef:** lips not smooth, ribbon has a notch (M-shape) — OK to edit ribbon shape to be smoother? (echoes Rik's ch1 advice: straighten lips.)
- **Kenzie (snapping workflow):** if vertex snapping struggles: hide mesh/MSTR-mouth for a better angle; or: select target vertex → Cursor to Selected → Edit Mode select bone → Selection to Cursor.
- Video glitch at 8:18 reported (freeze — just unedited end of video, Rik notified).

## 3.03 Helper empties
Desc: Empties connect the ribbon mesh to the armature — points the deformation bones aim at. Also used for the lip zipper.

Comments:
- **Benjamin Kiener:** why empties instead of temporary bones (easier to symmetrize)? → **Rik: bones would work fine too. Empties chosen for simplicity — they only function as pointers, and quick vertex parenting beats using constraints.** (For SmartRig automation: use whichever is programmatically cleaner.)
- Kenzie: Batch Rename — press + next to Find/Replace to add a second rename operation (Set Name).

## 3.04 Stretch bones
Desc: Pointer bones starting from the jaw base, each pointing (Stretch-To) toward its designated helper empty.

Comments:
- **Dustin S Haynes: had an AI write a Python script to automate the tedious setup — parenting empties to nearest vertex, setting Stretch-To targets by name, parenting DEF bones under stretch bones (naming convention).** ← exactly what SmartRig Pro will do natively.
- Miles Hobbs: batch rename vs symmetrize → Rik: symmetrize works too.
- Todor: Batch Rename is native (add/remove rename parameters with +/- icons), no addon.

## 3.05 Deformation bones
Desc: DEF bones parented to their pointer (stretch) bones; each lip DEF bone influences a **single edge loop** of the lips → micro-adjust ability for the animator.

Comments:
- **Denys (key concept Q):** why stretch bones — why not parent DEF directly to empties? → **Rik: stretch bones point toward the ribbon, so DEF bones get location AND rotational influence. Parented straight to empties they'd only get location.**
- **Rik tip:** hold **Alt** when applying bone color / armature-defined change to apply to all selected bones (not just active).
- Kenzie's mental model: "mouth as a balloon; a few bones tell other bones 'make it this big' — more complex than shape keys but far more expressive."

## 3.06 Constraints
Desc: Hook up DEF bones properly: all control bones must follow and scale with parents while keeping their own freedom — animator can manipulate each bone while respecting hierarchy.

Comments:
- **Baffye (debug case):** MSTR-Lip_upp/low not moving with DEF lip bones → cause: **DEF-Jaw tail was at the chin instead of opposite the lips** (bone placement matters for constraint spaces).
- mackenzee: scaling issue (s scales evenly) — unresolved in comments.

## 3.07 Local control bones
Desc: Local control bones (CTL-lip_local) for the ribbon mesh; use 3-curve principle to subtract/divide weights from main upper/lower lip bones. **Requires addon "Add Selected Bones to Vertex Group" by Mochi_lin** (introduced in ch2 'Mouth' video).

Comments:
- **Kenzie (pitfall):** Auto-Normalize stays on in weight-paint mode after assigning 1.0 — "locks" painting; turn it off, then smooth transitions between the 3 upper groups (local_up / local_up.L / local_up.R).
- Kenzie snapping fix: Magnet on, Snap base=Active, target=Vertex, target selection=all, hover over ribbon then G.
- Kenzie observation: "eyelids might just be two extra mouth setups?" — correct intuition, ch4 confirms.

## 3.08 Weight painting
Desc: Subtract weights from MSTR-mouth and assign to each DEF bone → smooth deformation + global/local/micro control.

Comments (rich debugging thread):
- **Rik's masking workflow:** in Edit Mode select lower-face + inner mouth faces → Ctrl+I invert → H hide → loop selections now only hit visible geometry → Alt+H when done.
- **Nick Fisher (important bug + self-solve):** weights didn't fall off like the video — **his DEF-Lip_upper bone had Deform checkbox off, so Auto-Normalize (which only normalizes between deforming bones) took no weight from MSTR-Mouth.** Fix: enable Deform on the bone.
- **Yunior:** CTL-LIP_CORN pinchy when moving along cheek → **Rik: corrective shape keys added later (chapter 8) address this.**
- **Saù tips:** vertex groups auto-sort by name/hierarchy via the "v" icon; **K shortcut opens a pie menu for group management in weight paint**.
- Balam: forgot to subtract weights from original groups as he went — no clean retroactive fix mentioned.
- Dustin: MSTR bones should NOT retain stray weights (his MSTR-FaceLower influence caused messy result).

## 3.09 Lip zipper
Desc: Custom properties driving the lip zipper for left and right individually using simple but effective driver expressions.

Comments:
- **Bone collections confusion (multiple users):** Rik created bone collections early but never showed re-assigning new bones to collections; users got lost. **Miles Hobbs: downloaded the Storm model and copied the collection structure from it.**
- **Andrew Brandon: converted the double-vertex-group zipper setup to the new Geometry Attribute — "pretty much the same".** (Modernization hint for our implementation.)

## Key takeaways for SmartRig Pro (lips module)
1. Full chain: MSTR-mouth mask → ribbon mesh (3-curve) → helper empties (vertex-parented) → stretch/pointer bones from jaw base (Stretch-To empties) → DEF bones (1 per edge loop) parented to stretch bones → CTL local bones → weight subtraction cascade → lip zipper via custom props + drivers.
2. Stretch bones exist to give DEF bones rotation, not just location — don't shortcut this in automation.
3. Users repeatedly hit: Deform checkbox off, Auto-Normalize locking, vertex-group naming mismatches, DEF-Jaw tail placement — all preventable with automated validation checks in our addon.
4. A commenter already scripted this setup with AI assistance — validates that the whole lips chapter is automatable (our core thesis).
5. Consider Geometry Attributes instead of double vertex groups for zipper (Andrew Brandon).
6. Corrective shape keys (ch8) are the designed answer to mouth-corner pinching, not more weight tweaking.
