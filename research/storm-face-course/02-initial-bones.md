# Chapter 2 — Initial Bones (5 lessons)
Captured 2026-07-02, all comments incl. expanded replies.

## 2.01 Neck and head
Desc: Creating the neck and head bones and weight painting the face mesh to it.

Comments:
- **Karsten K:** what brush settings for weight painting? → **Rik: default weight paint settings + shortcuts — hold SHIFT to blur, CTRL to subtract. Quick switching between brush modes without the T panel.**
- **con.grennan:** adding this face rig to existing Rigify rig? → Kenzie: has parented a face armature to the body armature before.
- **Waruta (gold thread — integrating with CloudRig/body rig):** how to add course face rig to existing CloudRig while preserving metarig regeneration?
  - **Demeter Dzadik (Blender Studio):** For Storm they used a complex **post-generation script that preserves the generated rig from the neck up** — Rik manually rigged above the neck and it survived body re-generation. Downside: "code held together by prayer", slow; only for Python-comfortable users. **The script is in Storm's .blend file.** Recommended alternative: create bones+constraints in the metarig and keep re-generating (modern CloudRig copies Regular bones over) — cleaner but you rig "half-blind" since the metarig doesn't deform the character.
  - Waruta: used a separate face armature during the course; considered duplicating face controls in body metarig to drive them.
  - **Demeter:** the Raw Copy deform-checkbox issue Waruta hit was a CloudRig bug, since fixed — update CloudRig.
  - Romain Clement: workaround — weight paint against metarig raw-copy bone, then after generation switch the Armature modifier target to the generated rig; works since DEF names are preserved.

## 2.02 Jaw
Desc: Dividing the head weights into the upper face, lower face and the jaw bone.

Comments:
- Kenzie: painting at 3:31 looks bad with PAINT alone — follow every pass with BLUR to match the video (it was clipped/voiceover).
- **Yunior José:** did you weight paint part of the head off-screen when selecting Face_upp vertex group? → **Rik: at 1:24 vertices are assigned in Edit Mode to Face_upp group, then falloff smoothed in weight paint mode. The size of the upper_face region depends on your model/preferences. Stretch bones covered in 'Extra Bones' video.**
- Yunior suggests screenshot uploads in comments → Rik: will discuss internally.

## 2.03 Mouth
Desc: Setting up weights for the mouth by creating the **master mouth bone (MSTR-mouth)** — functions as an initial mask for eventually weight painting the lip bones.

Comments:
- Kenzie: Vertex Selection mode overrides vertex-group assignments — locks painting (pitfall at 3:02).
- **Romain Clement (snouted characters):** wolf snout — MSTR-mouth falloff copied from Storm causes lips poking through the nose when rotating up; how far should mouth weights reach when facial landmarks are close? → **Rik: valid question; Blender Studio is currently experimenting with snouted characters and mouth rigging for these cases — no consensus yet; may be added to the course later.**
- Wilmer Borda: interesting bone usage → Kenzie: extreme face poses possible with bones, more than shape keys.

## 2.04 Extra bones
Desc: Creating extra bones: **Ears, Jaw line, and stretch bones for the upper and lower face (str-face_up / str-face_low)**.

Comments:
- Kenzie: **vertex groups must be named exactly as the bones** (pitfall).
- GlowGamer: naming with ".L" enables the symmetry command → Rik: "Blender is full of hidden greatness".

## 2.05 Nose
Desc: Bone setup for the nose: **nose base, nostrils, nose bridge** + a simple **'nose follow' setup adjustable by the animator**.

Comments:
- **Nada Shareef:** longer realistic nose, extra nose-bridge bone — should it follow str-face_low or str-face_up? Wants 50/50 of both. (unanswered as of capture; solution = two Copy Transforms/Rotation constraints with 0.5 influence, or Armature constraint)
- **Kenzie (practical tune):** Copy Rotation alone didn't keep the nose put — added **Limit Rotation −5..+5 and reduced Copy Rotation influence to 0.3** for "just enough wiggle when talking".
- Nemanja Jevtic: couldn't pick bone for Copy Transforms — was X-ray issue, couldn't pick bone through mesh.

## Key takeaways for SmartRig Pro
1. Head weights split hierarchy: head → face_upp/face_low → jaw → MSTR-mouth mask → lip bones. Masking cascade is the core weighting strategy — automatable.
2. Stretch bones (str-face_up/low) are the parents that everything follows (nose bridge question confirms).
3. Nose follow = Copy Rotation from jaw/face bones with adjustable influence — expose as animator slider (users tune 0.3 + Limit Rotation).
4. Snouted characters (محل سؤالنا للحيوانات): even Blender Studio has no solved recipe yet — flag as "experimental" in our module, don't promise.
5. Integration with body rigs (our case!): Storm used a post-generation preserve-neck-up script stored in the .blend — for SmartRig the face module should build directly into our own rig, avoiding that fragility entirely. Selling point vs CloudRig workflow.
