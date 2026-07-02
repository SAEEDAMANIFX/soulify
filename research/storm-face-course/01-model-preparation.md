# Chapter 1 — Model Preparation (6 lessons)
Captured 2026-07-02 with all comments (replies expanded).

## 1.01 Summary
Desc: Go over the different areas to check to make sure the model is clean and ready for rigging; why it's important to check geometry, resolution, and scale before rigging.

Comments:
- Uncle_Bib: taking the class to upgrade VTube avatar expressions.
- **Carina Ilea:** deaf user asked for subtitles → **Rik:** checked → subtitles added to every video (confirmed Mar 27). Alexandre Laclare: requests subtitle size option (hard to read on 27" / iPad).
- **Aitor Fernández:** how to start clean with just model, textures, shading? → **Rik: download the Storm face from the introduction page of this course.**
- Leon Seal G / Sheldon J Smith / Nicolas D'Amore: long-awaited, no more deconstructing Maya tutorials.

## 1.02 Separating objects
Desc: Separating objects, especially the ones that require shape keys.
Comments: GlowGamer (praise only).

## 1.03 Scale, symmetry and resolution
Desc: Checking the scale, symmetry and resolution of the model to make sure they are ready for rigging.

Comments:
- **Caryl Deyn Korma:** issues using Multires instead of Subdivision? → **Rik: you could, but you'd likely lose performance — Multires stores more data on the different layers, whereas Subdivision only calculates the subdivision level.**
- Wesley Pena: thanks for the book recommendation (Art of Moving Points).

## 1.04 Aligning to world orientation
Desc: Importance of aligning objects to world orientation, particularly the eyes.

Comments:
- **Romain Clement:** does world-axis alignment apply to anthropomorphic animal characters (align top of muzzle horizontally)? → **Rik: generally not a must — don't intervene with the character's initial design. Don't force it into world orientation, but if it's slightly off, aligning to world gives clean transformations for rigging and animation.**

## 1.05 Modeling adjustments
Desc: Importance of checking the model on topology, neutrality and clean edgeloops.

Comments:
- Kenzie W Townsend: fell for the jaw-open horror demo (~3:06); notes Auto Normalize was on by default; suggests screencast keys.
- **Nada Shareef:** character has a shape key for mouth open — is simple bone deformation better? → Kenzie: shape-key-only mouth is fine for low-vert / 2D-toon open-closed style; Rik's technique better when you want more "life".
- **Romain Clement (key Q&A):** for snouted characters (wolf) with jowls giving lips a wobbly shape — straighten lips at rest? He struggled with lip deformation quality until asking. → **Rik: yes — straighten the lips as much as possible; avoids fighting unwanted curvature in the model itself.**
- **Nacho de Andrés:** suggests select-linked by face-sets instead of seams for selections → Rik agrees.
- Tamim Ahmed: how to get that face shadow display? → Kenzie: Solid mode Viewport Shading → Object Color Attribute + shadow toggle.

## 1.06 Managing the outliner
Desc: Start of setting up collections to organize and manage the rigging project.
Comments: none.

## Key takeaways for SmartRig Pro face module
1. Pre-rig checklist automatable: scale applied, symmetry check, resolution/subdiv (prefer Subdivision over Multires for performance), world orientation (esp. eyes), clean edge loops, neutral expression, straightened lips at rest.
2. Objects needing shape keys must be separated first — our auto-rigger should verify/offer separation (head/eyes/teeth/tongue).
3. Collections organization is part of the workflow from the start.
