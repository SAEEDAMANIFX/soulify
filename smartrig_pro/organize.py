"""Soulify - professional rig organization.

Replicates the Blender-Studio / CloudRig bone-collection layout observed in
Snow v4.2 + Storm: a nested tree where the animator sees ONLY the controls
(Body IK + Face global/local); every mechanism layer lives under one hidden
"Rigging" branch.  Works on any Soulify-generated rig (body-only or
body+face) and is safe to re-run at any time.

Tree (visibility in brackets):

    Root                        [on]
    Body                        [on]
        Torso                   [on]   (Torso, Torso (Tweak)[off])
        Arms                    [on]   (Arm.* (IK)[on] / (FK)[off] / (Tweak)[off])
        Legs                    [on]   (same)
        Fingers                 [on]   (Fingers[on], Fingers (Detail)[off])
    Face                        [on]
        Face Upper              [on]
            Brows               [on]   (Brows global[on], Brows local[on])
            Eyes                [on]   (Eyes global[on], Eyes local[on], Eyes_micro[off])
            Nose / Cheeks / Ears[on]
            Upper Master        [off]
            Lattices            [off]
        Face Lower              [on]
            Mouth               [on]   (Mouth global[on], Mouth local[on], Mouth micro[off])
            Jawline / Teeth / Tongue [on]
            Lower Master        [off]
    Rigging                     [OFF - the whole branch]
        DEF / MCH / ORG
        Face MCH
        Face Display
"""
import bpy

# (parent, child, child_visible)  -- parents created on demand, children
# only re-parented when they exist on the rig.
_TREE = [
    ("Body", "Torso", True),
    ("Torso", "Torso (Tweak)", False),
    ("Body", "Arms", True),
    ("Arms", "Arm.L (IK)", True), ("Arms", "Arm.L (FK)", False),
    ("Arms", "Arm.L (Tweak)", False),
    ("Arms", "Arm.R (IK)", True), ("Arms", "Arm.R (FK)", False),
    ("Arms", "Arm.R (Tweak)", False),
    ("Body", "Legs", True),
    ("Legs", "Leg.L (IK)", True), ("Legs", "Leg.L (FK)", False),
    ("Legs", "Leg.L (Tweak)", False),
    ("Legs", "Leg.R (IK)", True), ("Legs", "Leg.R (FK)", False),
    ("Legs", "Leg.R (Tweak)", False),
    ("Body", "Fingers", True),
    ("Fingers", "Fingers (Detail)", False),
    ("Face", "Face Upper", True),
    ("Face Upper", "Brows", True),
    ("Brows", "Brows global", True), ("Brows", "Brows local", True),
    ("Face Upper", "Eyes", True),
    ("Eyes", "Eyes global", True), ("Eyes", "Eyes local", True),
    ("Eyes", "Eyes_micro", False),
    ("Face Upper", "Nose", True), ("Face Upper", "Cheeks", True),
    ("Face Upper", "Ears", True),
    ("Face Upper", "Upper Master", False),
    ("Face Upper", "Lattices", False),
    ("Face", "Face Lower", True),
    ("Face Lower", "Mouth", True),
    ("Mouth", "Mouth global", True), ("Mouth", "Mouth local", True),
    ("Mouth", "Mouth micro", False),
    ("Face Lower", "Jawline", True), ("Face Lower", "Teeth", True),
    ("Face Lower", "Tongue", True),
    ("Face Lower", "Lower Master", False),
    ("Rigging", "DEF", True), ("Rigging", "MCH", True),
    ("Rigging", "ORG", True),
    ("Rigging", "Face MCH", True), ("Rigging", "Face Display", True),
]

# group collections we may create; their own default visibility
_GROUPS = {"Body": True, "Arms": True, "Legs": True, "Fingers": True,
           "Torso": True, "Face": True, "Face Upper": True,
           "Face Lower": True, "Brows": True, "Eyes": True, "Mouth": True,
           "Rigging": False}

# top-level display order
_ORDER = ["Root", "Body", "Face", "Rigging"]

# legacy empty collections that may linger from earlier builds
_PURGE_OK = {"Face (Primary)", "Face (Secondary)", "Face (MCH)",
             "Layer 1", "Layer 2"}


def organize(rig):
    """Nest + set default visibility on `rig`'s bone collections.
    Returns a small report dict. Safe to re-run."""
    arm = rig.data
    colls = arm.collections_all
    made, moved = 0, 0

    def _get(name):
        return colls.get(name)

    def _ensure_group(name):
        nonlocal made
        c = colls.get(name)
        if c is None:
            c = arm.collections.new(name)
            made += 1
        return c

    # 1) build the tree
    for parent_name, child_name, child_vis in _TREE:
        child = _get(child_name)
        if child is None and parent_name in _GROUPS and child_name in _GROUPS:
            # group under group (e.g. Arms under Body): create lazily only
            # when the parent side has real children later
            continue
        if child is None:
            continue
        parent = _ensure_group(parent_name)
        if child.parent is not parent:
            try:
                child.parent = parent
                moved += 1
            except Exception:
                pass
        child.is_visible = bool(child_vis)

    # group-under-group links (Body>Torso etc. handled above only when the
    # child existed; now link created groups into their parents)
    _GROUP_LINKS = [("Body", "Torso"), ("Body", "Arms"), ("Body", "Legs"),
                    ("Body", "Fingers"), ("Face", "Face Upper"),
                    ("Face", "Face Lower"), ("Face Upper", "Brows"),
                    ("Face Upper", "Eyes"), ("Face Lower", "Mouth")]
    for pn, cn in _GROUP_LINKS:
        c = _get(cn)
        if c is None:
            continue
        p = _ensure_group(pn)
        if c.parent is not p:
            try:
                c.parent = p
                moved += 1
            except Exception:
                pass

    # 2) group visibility defaults
    for name, vis in _GROUPS.items():
        c = _get(name)
        if c is not None:
            c.is_visible = bool(vis)

    # 3) purge empty leftovers (no bones anywhere below, no children)
    def _bones_below(c):
        n = len(c.bones)
        for ch in c.children:
            n += _bones_below(ch)
        return n

    purged = 0
    for c in list(arm.collections_all):
        try:
            if len(c.children) == 0 and len(c.bones) == 0 and \
                    (c.name in _PURGE_OK or c.parent is None):
                arm.collections.remove(c)
                purged += 1
        except Exception:
            pass
    # second pass: groups we made that ended up empty
    for c in list(arm.collections_all):
        try:
            if c.name in _GROUPS and _bones_below(c) == 0:
                arm.collections.remove(c)
                purged += 1
        except Exception:
            pass

    # 4) top-level ordering: Root, Body, Face, Rigging, rest
    roots = [c for c in arm.collections]
    for i, name in enumerate(_ORDER):
        c = colls.get(name)
        if c is not None and c.parent is None:
            try:
                c.child_number = min(i, len(arm.collections) - 1)
            except Exception:
                pass

    return {"created": made, "moved": moved, "purged": purged,
            "roots": [c.name for c in arm.collections]}


class SMARTRIG_OT_organize_rig(bpy.types.Operator):
    """Organize the generated rig's bone collections into the professional
    nested layout (Body / Face / Rigging) and hide all mechanism layers"""
    bl_idname = "smartrig.organize_rig"
    bl_label = "Organize Rig Collections"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = None
        ob = context.active_object
        if ob is not None and ob.type == 'ARMATURE':
            rig = ob
        if rig is None:
            from . import face as _face
            rig = _face._target_rig()
        if rig is None:
            self.report({'ERROR'}, "No generated rig found")
            return {'CANCELLED'}
        rep = organize(rig)
        self.report({'INFO'},
                    "Organized: %d moved, %d groups, %d purged" %
                    (rep["moved"], rep["created"], rep["purged"]))
        return {'FINISHED'}


_classes = (SMARTRIG_OT_organize_rig,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
