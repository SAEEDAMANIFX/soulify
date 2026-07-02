# Advanced Facial Rigging (Blender Studio) — Course Overview
Source: https://studio.blender.org/training/facial-rigging/ — Author: Rik Schutte
Captured: 2026-07-02. 11 chapters, 55 videos, 6h 53m. Requires Blender 5.0+.

## Course description (verbatim)
In this course, we are going to cover every method and workflow used to create the flexible and expressive facial rigging system for Storm, the latest character by Blender Studio.

Rik Schutte has been working on this advanced course, spending months preparing, writing, and recording every step of the process. In order to use the latest functionality, Blender 5.0 or later is required.

This is a video course that goes over the step-by-step process of how to create a high-quality facial rig in Blender. We'll be using different techniques for facial deformation as well as Blender's native tools and free add-ons. The course contains over 40 videos and additional files, scripts, and add-ons.

Note: advanced rigging course — assumes solid understanding of facial rigging and Blender. **This course is not about rigging automation** — almost everything is made from scratch to grasp underlying concepts.

### References
- Workflows inspired by facial rigs commonly seen in the animation industry.
- Many principles/deformation concepts from the book **"The Art of Moving Points" by Brian Tindall**.
- Rik spoke with several industry riggers and animators about principles for effective facial rigs.

### Two core methods
1. **Ribbon Guide Meshes** — smooth predictable interpolation between neighboring bones, following the **"three curve principle"** (from the book). Used for lips, eyelids.
2. **Series of shape keys** — same three-curve principle in traditional form. Used for brows and cheeks on Storm.
- Rik's note: in future versions he would likely use Ribbon Guide Mesh for the brows too (more animator flexibility). Both workflows are universally applicable and conceptually equally strong.

### Version note
The course reflects a slightly **updated version** of Storm's facial rig vs the original release: cleaner naming conventions, small bone adjustments, optimizations. No noticeable changes for the animator.

## Chapter structure (55 lessons)
0. Course introduction: 01 Introduction (Free)
1. Model preparation (6): Summary / Separating objects / Scale, symmetry and resolution / Aligning to world orientation / Modeling adjustments / Managing the outliner
2. Initial bones (5): Neck and head / Jaw / Mouth / Extra bones / Nose
3. Lips (9): 3-curve principle / Ribbon mesh / Helper empties / Stretch bones / Deformation bones / Constraints / Local control bones / Weight painting / Lip zipper
4. Eyes (10): Initial bones / Eye highlights / Iris and pupil / Initial weight painting / Eyelids ribbon mesh / Eyelids deformation bones / Eyelids local controls / Eyelids weight painting / Auto blink / Eyelids follow
5. Brows (8): Why shape keys? / Deformation and eye sockets / Temporary geometry / Creating X and Z directions / Creating shape keys / Applying shape keys / Splitting weights / Local controls
6. Cheeks (3): Main shape keys / Lattice deformers / Splitting weights
7. Teeth and tongue (2): Teeth / Tongue
8. Mouthcorner shape keys (4): Concept / Initial shape keys / Combination shape keys / Mouth open corrective shape keys
9. Additional deformers (6): Mouth squash / Cheek puff / Additional lattices / Lip pucker / Lip compress / Nostril shape keys
10. Control widgets (1): Widgets and override transforms

## Intro lesson — comments (Q&A)
- **Shivagurunath Senthil:** Does this face rig support exporting DEF-bones to game engines keeping animation intact?
  - **Rik:** Course focused on film character animation, but the bone structure should transfer to a game engine with some minor tweaks.
- Praise comments (Duncan Rudd, Robin Ruud, derek henry).

## Notes for SmartRig Pro
- Storm face model is downloadable from the course introduction page (Rik confirms in ch1 comments).
- Subtitles were added to every video (March 2026) — useful for transcript-based study later.
