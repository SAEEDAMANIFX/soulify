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


class SMARTRIG_OT_face_rig_check(bpy.types.Operator):
    """Verify the rig is healthy AFTER binding, BEFORE loading expressions:
    drives every face function (jaw, eyes, blink, lips, brows, teeth,
    tongue...) and MEASURES the mesh response + corner accuracy.
    Results show below the button"""
    bl_idname = "smartrig.face_rig_check"
    bl_label = "Rig Check (after binding)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import json
        from mathutils import Vector
        from . import face as _face
        from . import expressions as _ex
        sc = context.scene
        props = sc.smartrig
        rig = _face._target_rig()
        body = getattr(props, "target_mesh", None)
        if rig is None or body is None:
            self.report({'ERROR'}, "Build + bind the face rig first")
            return {'CANCELLED'}
        sc.frame_set(0)
        # detach the expressions action while probing - keyed rest values
        # would overwrite the manual test poses and fake 0.0 readings
        old_action = None
        if rig.animation_data is not None and rig.animation_data.action:
            old_action = rig.animation_data.action
            rig.animation_data.action = None
        meshes = [ob for ob in sc.objects if ob.type == 'MESH'
                  and not ob.name.startswith(("HLP-", "WGT", "SR_", "GEO-"))
                  and any(m.type == 'ARMATURE' and m.object is rig
                          for m in ob.modifiers)]
        dg = context.evaluated_depsgraph_get()

        def cos(ob):
            context.view_layer.update()
            dg.update()
            ev = ob.evaluated_get(dg).to_mesh()
            c = [v.co.copy() for v in ev.vertices]
            ob.evaluated_get(dg).to_mesh_clear()
            return c

        base = {ob.name: cos(ob) for ob in meshes}

        def probe(bone, loc_arm=None, locl=None, rot=None):
            pb = rig.pose.bones.get(bone)
            if pb is None:
                return None
            old_loc = pb.location.copy()
            old_mode = pb.rotation_mode
            old_rot = None
            if rot is not None:
                pb.rotation_mode = 'XYZ'
                old_rot = pb.rotation_euler.copy()
                pb.rotation_euler = rot
            if loc_arm is not None:
                pb.location = pb.bone.matrix_local.to_3x3().inverted() \
                    @ Vector(loc_arm)
            if locl is not None:
                pb.location = locl
            mx = 0.0
            for ob in meshes:
                c = cos(ob)
                b = base[ob.name]
                if len(c) != len(b):
                    continue
                mx = max(mx, max((c[i] - b[i]).length for i in range(len(c))))
            pb.location = old_loc
            if old_rot is not None:
                pb.rotation_euler = old_rot
            pb.rotation_mode = old_mode
            return mx * 1000.0

        bset = rig.data.bones
        S = 0.06
        if bset.get("FK-Eye.L") and bset.get("FK-Eye.R"):
            S = (bset["FK-Eye.L"].head_local
                 - bset["FK-Eye.R"].head_local).length
        sign = _ex._probe_jaw_sign(context, rig, body)

        checks = [
            ("Jaw opens mouth", dict(bone="CTL-Jaw",
                                     rot=(sign * 0.35, 0, 0)), 5.0),
            ("Eye aim (target)", dict(bone="P-Eye_target",
                                      loc_arm=(0.3 * S, 0, 0)), 0.3),
            ("Blink L", dict(bone="PRP-Eye_blink.L",
                             locl=(0, -1.0 / 30.0, 0)), 1.0),
            ("Blink R", dict(bone="PRP-Eye_blink.R",
                             locl=(0, -1.0 / 30.0, 0)), 1.0),
            ("Mouth corner L", dict(bone="CTL-Lips_corn.L",
                                    loc_arm=(0, 0, 0.08 * S)), 1.0),
            ("Mouth corner R", dict(bone="CTL-Lips_corn.R",
                                    loc_arm=(0, 0, 0.08 * S)), 1.0),
            ("Upper lip", dict(bone="CTL-Lips_main_upp",
                               loc_arm=(0, 0, 0.05 * S)), 0.5),
            ("Lower lip", dict(bone="CTL-Lips_main_low",
                               loc_arm=(0, 0, -0.05 * S)), 0.5),
            ("Brow inner L", dict(bone="CTL-Brow_in.L",
                                  loc_arm=(0, 0, 0.08 * S)), 0.5),
            ("Brow master L", dict(bone="CTL-Brow_all.L",
                                   loc_arm=(0, 0, 0.10 * S)), 0.5),
            ("Cheek L", dict(bone="CTL-Cheek_all.L",
                             loc_arm=(0, 0, 0.05 * S)), 0.3),
            ("Nose", dict(bone="MSTR-Nose", loc_arm=(0, 0, 0.04 * S)), 0.3),
            ("Upper teeth", dict(bone="MSTR-Teeth_upp",
                                 loc_arm=(0, 0, 0.05 * S)), 0.3),
            ("Lower teeth", dict(bone="MSTR-Teeth_low",
                                 loc_arm=(0, 0, 0.05 * S)), 0.3),
            ("Tongue", dict(bone="MSTR-Tongue",
                            loc_arm=(0, -0.05 * S, 0)), 0.3),
        ]
        lines = []
        for label, kw, minmm in checks:
            mm = probe(**kw)
            if mm is None:
                lines.append([label, "bone missing", False])
            else:
                lines.append([label, "%.1f mm" % mm, bool(mm >= minmm)])

        # corner accuracy vs the registered/projected landmarks
        try:
            grid = bpy.data.objects.get(_face.GRID_NAME)
            gm = grid.matrix_world

            def gp(n):
                i = _face.GRID_IDX.get(n)
                return (gm @ grid.data.vertices[i].co) if i is not None \
                    else None

            for gname, bname, lim in (
                    ("mouth_corner.L", "CTL-Lips_corn.L", 6.0),
                    ("eye_out.L", "DEF-Eyelid_out.L", 6.0),
                    ("eye_in.L", "DEF-Eyelid_in.L", 6.0)):
                p = gp(gname)
                db = rig.data.bones.get(bname)
                if p is not None and db is not None:
                    mm = ((rig.matrix_world @ db.head_local) - p).length * 1000
                    lines.append(["Corner %s" % gname, "%.1f mm off" % mm,
                                  bool(mm <= lim)])
        except Exception:
            pass

        if old_action is not None:
            rig.animation_data.action = old_action
            sc.frame_set(0)
        sc["sr_rig_check"] = json.dumps(lines)
        npass = sum(1 for l in lines if l[2])
        self.report({'INFO'} if npass == len(lines) else {'WARNING'},
                    "Rig check: %d/%d passed" % (npass, len(lines)))
        return {'FINISHED'}


_classes = (SMARTRIG_OT_organize_rig, SMARTRIG_OT_face_rig_check)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
