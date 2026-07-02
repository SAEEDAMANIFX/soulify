# Chapters 8, 9, 10 — Mouthcorner Shape Keys (5) + Additional Deformers (5) + Control Widgets (1)
Captured 2026-07-02, all comments.

## 8.01 Concept
Desc: Theory of corrective shapes for mouth corners. **The ORDER of corrective shape keys is important — some are related to one another.** Uses the **Pose Shape Keys addon** (Blender Studio, on extensions platform).

## 8.02 Initial shape keys
Desc: First pose shape key using Pose Shape Keys addon; initial shape keys for all axes (mouth corner X/Y directions).

## 8.03 Combination shape keys
Desc: Combination shapes built from the initial shapes — **activated only when both X and Y are posed in a particular direction** (corner up+out, etc.).

## 8.04 Mouth open corrective shape keys
Desc: When the mouth opens, the corner shape keys don't hold up → for each mouth corner shape key create a correction for the opened-mouth state.

Comments (valuable):
- **Karsten K: how to blend smoothly between open-mouth shapes?** → **Rik: keep topology intact — when adjusting an x_pos corrective, vertices should move mostly in X; the more Y drift, the harder the blend. Corrective Smooth modifier with a weight map = last resort.**
- **tone watson (bug + fix from final Storm rig):** SK-MC_y_neg.L wrongly triggered by jaw opening → **in the final Storm rig, CTL-Lip_corn.L/R no longer use Copy Location but are children of additional P- (parent) bones — that fixes it.** Rik: will verify and adjust the video. (Marshall confirmed the fix works.)

## 8.05 Mouth squash
Desc: Extra: mouth squash when jaw rotates further UPWARD — animator can compress lips/lower face when clenching teeth → organic fleshiness.

## 9.01 Cheek puff
Desc: 'Cheek puff' shape key from a temporary armature + a lattice so the animator can use rot/scale/location in every direction. "Flexibility and designed deformations go hand in hand."

## 9.02 Additional lattices
Desc: More lattices across the official Storm rig — purely **secondary deformations to push a pose further**. Add your own lattices per character/style as needed.

## 9.03 Lip pucker
Desc: "Should be in every high quality rig." Easy with Pose Shape Keys addon once the method is clear.

## 9.04 Lip compress
Desc: Opposite of lip pucker, same method.

## 9.05 Nostril shape keys
Desc: Pose Shape Keys corrective for nostrils moving upward — the **'sneer'** (brows and nostrils pulled toward each other, compression). Shape emulates skin bulging; subtle→wrinkly depending on style.

## 10.01 Widgets and override transforms
Desc: Bone widgets for brow controls, main → local. **DSP (display) bones + widget override transforms (new in Blender 5.0) make controls follow mesh deformations seamlessly** (previously they'd stay behind).

Comments:
- **Achyut Shakya (open request):** publishing/linking workflow video — final cleanup of collections, shape keys, bone collections before publishing (Storm v1.1 is very clean). Not yet answered.
- Multiple requests for a **body rigging course** as follow-up.

## Key takeaways for SmartRig Pro
1. Corrective stack order matters: initial XY shapes → combination (X∧Y) → mouth-open correctives → squash. Our Corrective Wizard must encode this dependency order.
2. **P- parent bones instead of Copy Location for CTL-Lip_corn** (final Storm fix) — adopt directly in our build, skip the course's older approach.
3. Corrective sculpt rule: move vertices along the driving axis to keep blends smooth — encode as guidance/validation in our Corrective Shape Key Wizard.
4. Lattice philosophy: shape keys = designed deformation; lattices = secondary push. Cheek/teeth/extra lattices all follow this.
5. DSP widget override transforms (Blender 5.0) = the "controls ride the mesh" magic — required for professional feel of our face rig.
6. Publishing/cleanup step (collections, shape keys order, linking) is a real user pain — SmartRig can automate a "Publish Rig" button. Course gap = our opportunity.
