"""Build a Rigify human meta-rig and fit it exactly to the SmartRig markers.
Body, limbs, fingers and thumb are snapped to the computed joints; the face is
removed (no face markers were placed). The result is a standard, editable Rigify
metarig -> add Samples if wanted, then press Rigify 'Generate Rig'."""
import bpy
import numpy as np
from mathutils import Vector
from . import utils, fit, markers

META_NAME = "SR_Metarig"

# Rigify samples, grouped + nice labels/icons + hover descriptions
SAMPLE_GROUPS = [
    ("Limbs", [
        ("limbs.super_finger", "Finger", 'HAND'),
        ("limbs.super_palm", "Palm", 'GROUP_BONE'),
        ("limbs.arm", "Arm (IK/FK)", 'CON_KINEMATIC'),
        ("limbs.leg", "Leg (IK/FK)", 'CON_KINEMATIC'),
        ("limbs.super_limb", "Generic Limb", 'CON_KINEMATIC'),
        ("limbs.simple_tentacle", "Tentacle / Tail", 'IPO_EASE_IN_OUT'),
        ("limbs.spline_tentacle", "Spline Tentacle", 'FORCE_CURVE'),
        ("limbs.paw", "Paw", 'CON_KINEMATIC'),
    ]),
    ("Spine & Head", [
        ("spines.super_spine", "Super Spine", 'BONE_DATA'),
        ("spines.basic_spine", "Basic Spine", 'BONE_DATA'),
        ("spines.super_head", "Head / Neck", 'OUTLINER_OB_ARMATURE'),
        ("spines.basic_tail", "Tail", 'IPO_EASE_IN_OUT'),
    ]),
    ("Face", [
        ("faces.super_face", "Full Face", 'USER'),
        ("face.skin_eye", "Eye", 'HIDE_OFF'),
        ("face.skin_jaw", "Jaw", 'USER'),
        ("face.basic_tongue", "Tongue", 'USER'),
    ]),
    ("Basic", [
        ("basic.super_copy", "Single Control", 'BONE_DATA'),
        ("basic.copy_chain", "Copy Chain (FK)", 'LINKED'),
        ("basic.pivot", "Pivot", 'PIVOT_CURSOR'),
        ("basic.raw_copy", "Raw Copy", 'BONE_DATA'),
    ]),
]

SAMPLE_DESC = {
    "limbs.super_finger": "Finger chain with curl, spread and bend controls (master + per-joint).",
    "limbs.super_palm": "Palm / metacarpal fan that spreads the fingers.",
    "limbs.arm": "IK/FK arm with hand, pole target and snapping.",
    "limbs.leg": "IK/FK leg with foot roll, pole target and snapping.",
    "limbs.super_limb": "Generic IK/FK limb base (used for arms and legs).",
    "limbs.simple_tentacle": "Simple FK chain for a tail, tentacle or rope.",
    "limbs.spline_tentacle": "Spline-IK tentacle for smooth curvy chains.",
    "limbs.paw": "Digitigrade animal paw / leg.",
    "spines.super_spine": "Full body spine: hips, torso, chest, neck and head with IK/FK.",
    "spines.basic_spine": "Simple FK/IK spine chain.",
    "spines.super_head": "Head and neck controls (can attach to a spine).",
    "spines.basic_tail": "FK tail chain with a master control.",
    "faces.super_face": "Complete face rig (eyes, brows, lids, lips, jaw, etc.).",
    "face.skin_eye": "Single eye rig (aim + lids).",
    "face.skin_jaw": "Jaw rig.",
    "face.basic_tongue": "Tongue chain.",
    "basic.super_copy": "One control bone with a chosen widget (props, accessories).",
    "basic.copy_chain": "An FK chain that simply copies the bones.",
    "basic.pivot": "A pivot / socket control.",
    "basic.raw_copy": "Copies the bone with no widget or transformation.",
}


def _mirror(v):
    return Vector((-v.x, v.y, v.z))


def _ensure_rigify():
    if "rigify" not in bpy.context.preferences.addons.keys():
        import addon_utils
        try:
            addon_utils.enable("rigify", default_set=True, persistent=True)
        except Exception:
            pass
    return "rigify" in bpy.context.preferences.addons.keys()


def _fit_core(mo, props, J, h, ground, yc):
    """Position + roll all metarig bones from the joints J (shared by
    build_metarig and refit_metarig). Assumes `mo` exists with identity
    transform; preserves any extra sample bones."""
    bpy.context.view_layer.objects.active = mo
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for _o in bpy.context.selected_objects:
        _o.select_set(False)
    mo.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = mo.data.edit_bones

    def setb(name, head, tail):
        b = eb.get(name)
        if b is None or head is None or tail is None:
            return
        b.head = Vector(head); b.tail = Vector(tail)

    # ---- spine: pelvis -> neck (4) , neck -> head_base (2) , head ----
    pelvis = Vector(J["pelvis"]); neck = Vector(J["neck_base"])
    head_b = Vector(J["head_base"]); head_t = Vector(J["head_top"])

    def chain(p0, p1, n):
        return [p0.lerp(p1, i / n) for i in range(n + 1)]
    torso = chain(pelvis, neck, 4)
    setb("spine", torso[0], torso[1])
    setb("spine.001", torso[1], torso[2])
    setb("spine.002", torso[2], torso[3])
    setb("spine.003", torso[3], torso[4])
    nck = chain(neck, head_b, 2)
    setb("spine.004", nck[0], nck[1])
    setb("spine.005", nck[1], nck[2])
    setb("spine.006", head_b, head_t)

    # ---- limbs (left then mirrored right) ----
    def limbs(suf, mir):
        f = _mirror if mir else (lambda v: v)
        ch, sh = J["clavicle.L"]; ua0, ua1 = J["upper_arm.L"]
        fa0, fa1 = J["forearm.L"]; ha0, ha1 = J["hand.L"]
        setb("shoulder" + suf, f(Vector(ch)), f(Vector(sh)))
        setb("upper_arm" + suf, f(Vector(ua0)), f(Vector(ua1)))
        setb("forearm" + suf, f(Vector(fa0)), f(Vector(fa1)))
        setb("hand" + suf, f(Vector(ha0)), f(Vector(ha1)))
        th0, th1 = J["thigh.L"]; s0, s1 = J["shin.L"]
        ft0, ft1 = J["foot.L"]; t0, t1 = J["toe.L"]
        setb("thigh" + suf, f(Vector(th0)), f(Vector(th1)))
        setb("shin" + suf, f(Vector(s0)), f(Vector(s1)))
        setb("foot" + suf, f(Vector(ft0)), f(Vector(t0)))
        setb("toe" + suf, f(Vector(t0)), f(Vector(t1)))
        ank = Vector(ft0); hy = ank.y + 0.06 * h
        hw = 0.035 * h
        setb("heel.02" + suf, f(Vector((ank.x - hw, hy, ground))),
             f(Vector((ank.x + hw, hy, ground))))
        setb("pelvis" + suf, f(pelvis), f(Vector(J["thigh.L"][0])))
    limbs(".L", False)
    limbs(".R", True)

    # ---- breast bones: snap to the mesh-detected apex (if found) ----
    for _bn in ("breast.L", "breast.R"):
        if _bn in J:
            setb(_bn, Vector(J[_bn][0]), Vector(J[_bn][1]))

    # ---- ensure a slight elbow/knee bend so Rigify can solve the IK pole ----
    def ensure_bend(upper, lower, ydir):
        ub = eb.get(upper); lb = eb.get(lower)
        if not ub or not lb:
            return
        a = (ub.tail - ub.head); b = (lb.tail - lb.head)
        if a.length < 1e-6 or b.length < 1e-6:
            return
        if a.normalized().cross(b.normalized()).length < 0.08:   # nearly straight
            mid = ub.tail.copy()
            mid.y += ydir * 0.03 * a.length
            ub.tail = mid; lb.head = mid
    for suf in (".L", ".R"):
        ensure_bend("upper_arm" + suf, "forearm" + suf, 1.0)    # elbow -> back
        ensure_bend("thigh" + suf, "shin" + suf, -1.0)          # knee -> front

    # ---- fingers / thumb / metacarpals ----
    hand_m = J.get("fingers_manual", {}) or {}
    palm_m = J.get("palm_manual", {}) or {}
    palm_chains = [c for c in palm_m.values() if len(c) >= 2]
    FMAP = {"index": ("f_index", "palm.01"), "middle": ("f_middle", "palm.02"),
            "ring": ("f_ring", "palm.03"), "pinky": ("f_pinky", "palm.04")}

    def fingers(suf, mir):
        f = _mirror if mir else (lambda v: v)
        used = set()
        for fn, (fpre, palmname) in FMAP.items():
            ch = hand_m.get(fn)
            if not ch or len(ch) < 2:
                continue
            for k in range(min(3, len(ch) - 1)):
                setb("%s.0%d%s" % (fpre, k + 1, suf), f(ch[k]), f(ch[k + 1]))
            if palm_chains:
                cand = [i for i in range(len(palm_chains)) if i not in used]
                if cand:
                    i = min(cand, key=lambda ii: (palm_chains[ii][-1] - ch[0]).length)
                    used.add(i)
                    setb(palmname + suf, f(palm_chains[i][0]), f(ch[0]))
        th = hand_m.get("thumb")
        if th and len(th) >= 2:
            for k in range(min(3, len(th) - 1)):
                setb("thumb.0%d%s" % (k + 1, suf), f(th[k]), f(th[k + 1]))
    fingers(".L", False)
    fingers(".R", True)

    # ---- set Rigify-convention bone rolls (bones were just moved) ----
    FRONT = Vector((0.0, -1.0, 0.0)); BACK = Vector((0.0, 1.0, 0.0)); UP = Vector((0.0, 0.0, 1.0))

    def setz(name, zt):
        b = eb.get(name)
        if b is not None:
            try:
                b.align_roll(zt)
            except Exception:
                pass

    def setflat(name):
        """Roll the bone so its local X (side) axis is perfectly horizontal,
        with Z pointing down/back - matches the pro Rigify foot regardless of
        how the foot splays."""
        b = eb.get(name)
        if b is None:
            return
        yb = (b.tail - b.head)
        if yb.length < 1e-6:
            return
        yb = yb.normalized()
        xh = yb.cross(Vector((0.0, 0.0, 1.0)))
        if xh.length < 1e-5:
            return
        xh.normalize()
        zd = xh.cross(yb).normalized()
        if zd.z > 0.0:
            zd = -zd
        try:
            b.align_roll(zd)
        except Exception:
            pass
    for b in eb:
        n = b.name
        if n.startswith("spine"):
            setz(n, FRONT)
        elif n.startswith("shoulder"):
            setz(n, UP)
        elif n.startswith(("upper_arm", "forearm", "hand")):
            setz(n, FRONT)
        elif n.startswith("foot"):
            setflat(n)
        elif n.startswith(("thigh", "shin")):
            setz(n, BACK)
        elif n.startswith(("toe", "heel", "pelvis", "breast")):
            setz(n, UP)
    fit._orient_fingers_pro(eb, arm=mo)

    # ---- enforce perfect L/R roll symmetry (mirror roll: R = -L) ----
    for b in eb:
        if b.name.endswith(".L"):
            rb = eb.get(b.name[:-2] + ".R")
            if rb is not None:
                rb.roll = -b.roll

    # ---- remove the whole face rig (no face markers were placed) ----
    def _descendants(b):
        out = []
        for c in b.children:
            out.append(c); out.extend(_descendants(c))
        return out
    face = eb.get("face")
    if face:
        for b in [face] + _descendants(face):
            try:
                eb.remove(b)
            except Exception:
                pass

    bpy.ops.object.mode_set(mode='OBJECT')
    # Finger rotation axis = EXPLICIT 'X' so Rigify KEEPS our shared-axis roll
    # (in _orient_fingers_pro all four fingers share one bend axis). 'automatic'
    # would re-align each finger to its OWN plane and scatter them again
    # ('index goes sideways'). ensure_finger_curl adds the scale-curl drivers.
    try:
        mo2 = bpy.data.objects.get(META_NAME)
        for _pb in (mo2.pose.bones if mo2 else []):
            if _pb.name.startswith(("f_", "thumb")) and _pb.name.endswith((".01.L", ".01.R")):
                rp = getattr(_pb, "rigify_parameters", None)
                if rp is not None and hasattr(rp, "primary_rotation_axis"):
                    rp.primary_rotation_axis = 'X'
    except Exception as _e:
        print("SmartRig finger axis guard:", _e)


def build_metarig(props):
    mesh = props.target_mesh
    if mesh is None or mesh.type != 'MESH':
        return None, "Select your character mesh first."
    if not _ensure_rigify():
        return None, "Rigify add-on is not available/enabled."
    J, err, h = fit.compute_joints(props)
    if err:
        return None, err

    co = utils.read_world_coords(mesh)
    ground = float(co[:, 2].min()); top = float(co[:, 2].max())
    ch_h = top - ground
    yc = float(np.median(co[:, 1]))

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    old = bpy.data.objects.get(META_NAME)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    before = set(o.name for o in bpy.data.objects)
    bpy.ops.object.armature_human_metarig_add()
    new = [o for o in bpy.data.objects if o.name not in before and o.type == 'ARMATURE']
    if not new:
        return None, "Could not add the Rigify human metarig."
    mo = new[0]
    mo.name = META_NAME; mo.data.name = META_NAME
    mo.show_in_front = True
    # X-Mirror ON by default: symmetric edit-mode + pose editing of the metarig
    try:
        mo.data.use_mirror_x = True
        mo.pose.use_mirror_x = True
    except Exception:
        pass

    # pre-fit the WHOLE metarig (so any unmapped bones land roughly right)
    s = (ch_h / 1.98) if ch_h > 1e-4 else 1.0
    mo.scale = (s, s, s)
    mo.location = (0.0, yc, ground)
    bpy.context.view_layer.objects.active = mo
    for o in bpy.context.selected_objects:
        o.select_set(False)
    mo.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    _fit_core(mo, props, J, h, ground, yc)
    # hide markers + leave guided flow so the panel shows the Rigify section
    try:
        markers.set_markers_hidden(True)
        props.markers_hidden = True
        props.guide_active = False
        ref = bpy.data.objects.get(utils.REF_NAME)
        if ref is not None:
            ref.hide_set(True)
    except Exception:
        pass
    return mo, None


def set_spine_neck(props):
    """Rebuild the metarig spine LIVE with a variable torso (spine_count) and neck
    (neck_count) count, preserving the pelvis/neck/head positions and re-parenting
    every dependent bone (shoulders, head, breast...) to the nearest new spine
    bone. basic_spine on the first torso bone, super_head on the first neck bone."""
    import re as _re
    from mathutils import Vector
    mo = bpy.data.objects.get(META_NAME)
    if mo is None:
        return None, "Build the metarig first."
    T = max(2, int(props.spine_count))
    N = max(1, int(props.neck_count))
    # CRITICAL: if the user changes the count WHILE still in Edit Mode, their
    # unsaved edits live only in edit_bones and are NOT yet flushed to
    # bone.head_local. Read them now and commit to Object Mode FIRST, otherwise
    # we'd rebuild from the stale pre-edit positions (= edits "revert"). We were
    # in edit mode, so leave the user back in edit mode at the end.
    was_edit = (mo.mode == 'EDIT')
    if mo.mode != 'OBJECT':
        bpy.context.view_layer.objects.active = mo
        try:
            bpy.ops.object.mode_set(mode='OBJECT')   # flush edit_bones -> bones
        except Exception:
            pass
    bones = mo.data.bones
    sp_re = _re.compile(r"^spine(\.\d+)?$")
    spine_names = [b.name for b in bones if sp_re.match(b.name)]
    if not spine_names:
        return None, "No spine chain found."
    spine_names.sort(key=lambda x: 0 if x == "spine" else int(x.split(".")[1]))
    # ---- identify the 3 regions; PRESERVE the user's manual edits ----
    # torso = bones before the super_head root; neck = super_head root .. before
    # the head; head = the last bone. We keep each region's existing POLYLINE
    # (so a tilted head / curved neck is kept) and only re-subdivide the region
    # whose count changed. An unchanged region (and the head) stays byte-exact.
    neck_first = next((n for n in spine_names
                       if mo.pose.bones[n].rigify_type == 'spines.super_head'), None)
    head_name = spine_names[-1]
    if neck_first in spine_names:
        ni = spine_names.index(neck_first)
    else:
        ni = max(1, len(spine_names) - 2)
        neck_first = spine_names[ni]
    torso_names = spine_names[:ni] or spine_names[:1]
    neck_names = spine_names[ni:-1] or [neck_first]
    neck_base = bones[neck_first].head_local.copy()
    head_base = bones[head_name].head_local.copy()
    head_top = bones[head_name].tail_local.copy()
    # region boundary polylines from the CURRENT bones (keeps manual shape)
    torso_poly = [bones[n].head_local.copy() for n in torso_names] + [neck_base]
    neck_poly = [bones[n].head_local.copy() for n in neck_names] + [head_base]

    def resample(poly, n):
        """n+1 points along `poly`, spaced by arc length. If the count is
        unchanged, return the existing points EXACTLY (no drift)."""
        if n + 1 == len(poly):
            return [p.copy() for p in poly]
        seglen = [(poly[i + 1] - poly[i]).length for i in range(len(poly) - 1)]
        total = sum(seglen)
        if total < 1e-9:
            return [poly[0].lerp(poly[-1], i / n) for i in range(n + 1)]
        out = [poly[0].copy()]
        for k in range(1, n):
            d = total * k / n
            acc = 0.0
            for i, sl in enumerate(seglen):
                if acc + sl >= d or i == len(seglen) - 1:
                    t = (d - acc) / sl if sl > 1e-9 else 0.0
                    out.append(poly[i].lerp(poly[i + 1], min(max(t, 0.0), 1.0)))
                    break
                acc += sl
        out.append(poly[-1].copy())
        return out

    torso = resample(torso_poly, T)            # T+1 pts, last == neck_base
    neck = resample(neck_poly, N)              # N+1 pts, first == neck_base, last == head_base

    # dependents: non-spine bones parented to a spine bone (remember old parent head)
    deps = [(b.name, b.parent.head_local.copy()) for b in bones
            if b.parent and sp_re.match(b.parent.name) and not sp_re.match(b.name)]

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = mo
    bpy.ops.object.mode_set(mode='EDIT')
    eb = mo.data.edit_bones
    for n in spine_names:
        if eb.get(n):
            eb.remove(eb[n])

    segs = [(torso[i], torso[i + 1]) for i in range(T)]
    segs += [(neck[i], neck[i + 1]) for i in range(N)]
    segs += [(head_base, head_top)]            # head bone (kept exactly)
    created = []
    prev = None
    for i, (hh, tt) in enumerate(segs):
        nm = "spine" if i == 0 else "spine.%03d" % i
        b = eb.new(nm); b.head = hh; b.tail = tt
        if prev is not None:
            b.parent = prev
            # The super_head ROOT (first neck bone, index T) must be DISCONNECTED
            # so connected_children_names() stops there: basic_spine then claims
            # only the torso, super_head claims neck+head (matches stock metarig).
            b.use_connect = (i != T)
        b.align_roll(Vector((0.0, -1.0, 0.0)))   # Z faces FRONT (-Y), Rigify spine convention
        prev = b; created.append(nm)
    # re-parent dependents to the nearest new spine bone (by head distance)
    for dn, oldph in deps:
        db = eb.get(dn)
        if db is None:
            continue
        best = min(created, key=lambda nm: (eb[nm].head - oldph).length)
        db.parent = eb[best]; db.use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    # CLEAR any leftover rigify_type first — pose-bone rigify_type persists by
    # NAME, so a recreated bone (e.g. old spine.004 = super_head) would keep its
    # stale type and produce a second, broken spine rig. Wipe, then assign fresh.
    for nm in created:
        try:
            mo.pose.bones[nm].rigify_type = ''
        except Exception:
            pass
    # rigify types + key params
    sp = mo.pose.bones[created[0]]
    sp.rigify_type = 'spines.basic_spine'
    try:
        sp.rigify_parameters.pivot_pos = max(1, min(T - 1, T // 2))
    except Exception:
        pass
    nh = mo.pose.bones[created[T]]            # first neck bone
    nh.rigify_type = 'spines.super_head'
    try:
        nh.rigify_parameters.connect_chain = True
    except Exception:
        pass
    # silence Rigify's "empty tweak layer list" warning (no extra layer refs)
    for pb in (sp, nh):
        for attr in ("tweak_layers_extra", "primary_layers_extra",
                     "secondary_layers_extra", "fk_layers_extra"):
            try:
                setattr(pb.rigify_parameters, attr, False)
            except Exception:
                pass
    # put the user back in Edit Mode if that's where they were
    if was_edit:
        try:
            bpy.context.view_layer.objects.active = mo
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass
    return mo, None


def refit_metarig(props):
    """Move the EXISTING metarig's bones to the current markers without
    recreating the armature - so added samples and the generated-rig link
    are preserved."""
    mo = bpy.data.objects.get(META_NAME)
    if mo is None:
        return None, "No metarig yet. Build it first."
    mesh = props.target_mesh
    if mesh is None or mesh.type != 'MESH':
        return None, "Select your character mesh first."
    J, err, h = fit.compute_joints(props)
    if err:
        return None, err
    co = utils.read_world_coords(mesh)
    ground = float(co[:, 2].min())
    yc = float(np.median(co[:, 1]))
    _fit_core(mo, props, J, h, ground, yc)
    return mo, None


class SMARTRIG_OT_toggle_group(bpy.types.Operator):
    bl_idname = "smartrig.toggle_sample_group"
    bl_label = "Toggle Sample Group"
    bl_description = "Show / hide this group of Rigify samples"
    bl_options = {'INTERNAL'}
    group: bpy.props.StringProperty(default="")

    def execute(self, context):
        p = context.scene.smartrig
        cur = [g for g in p.samples_expanded.split(",") if g]
        if self.group in cur:
            cur.remove(self.group)
        else:
            cur.append(self.group)
        p.samples_expanded = ",".join(cur)
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_add_sample(bpy.types.Operator):
    bl_idname = "smartrig.add_sample"
    bl_label = "Add Rigify Sample"
    bl_options = {'REGISTER', 'UNDO'}
    metarig_type: bpy.props.StringProperty(default="")

    @classmethod
    def description(cls, context, properties):
        return SAMPLE_DESC.get(properties.metarig_type,
                               "Add this Rigify rig sample to the metarig")

    def execute(self, context):
        meta = bpy.data.objects.get(META_NAME)
        if meta is None and context.active_object and context.active_object.type == 'ARMATURE':
            meta = context.active_object
        if meta is None:
            self.report({'ERROR'}, "Build the Rigify metarig first.")
            return {'CANCELLED'}
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for o in context.selected_objects:
            o.select_set(False)
        meta.select_set(True)
        context.view_layer.objects.active = meta
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            bpy.ops.armature.metarig_sample_add(metarig_type=self.metarig_type)
        except Exception as e:
            self.report({'WARNING'}, "Could not add sample: %s" % e)
            return {'CANCELLED'}
        self.report({'INFO'}, "Added sample '%s'. Position it, then Generate." % self.metarig_type)
        return {'FINISHED'}


class SMARTRIG_OT_build_metarig(bpy.types.Operator):
    bl_idname = "smartrig.build_metarig"
    bl_label = "Build Rigify Metarig"
    bl_description = ("Create a Rigify human meta-rig fitted exactly to your markers "
                      "(body, limbs, fingers; no face). Add samples, then Generate Rig.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.smartrig.target_mesh is not None or \
            bpy.data.objects.get("spine_root") is not None

    def execute(self, context):
        props = context.scene.smartrig
        if props.target_mesh is None:
            ao = context.active_object
            if ao and ao.type == 'MESH':
                props.target_mesh = ao
        mo, err = build_metarig(props)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        # drop into Edit Mode on the new metarig so the user can tweak bones now
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            mo.hide_set(False)
            bpy.ops.object.select_all(action='DESELECT')
            mo.select_set(True)
            context.view_layer.objects.active = mo
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass
        # SMART fingers: real AI placement first (joints + palm + ROLLS),
        # geometric knuckle detector as fallback
        try:
            from . import skirt as _sk, arp_ai as _ai
            meta_ob = bpy.data.objects.get(META_NAME)
            mesh_ob = context.scene.smartrig.target_mesh
            if meta_ob is not None and mesh_ob is not None:
                _n = 0
                try:
                    _n = _ai.auto_place(meta_ob, mesh_ob)
                except Exception as _e:
                    print("SmartRig AI place failed:", _e)
                if _n:
                    print("SmartRig AI placed %d finger chains (joints+rolls)"
                          % _n)
                else:
                    _ref = _sk.refine_finger_joints_meta(meta_ob, mesh_ob)
                    if _ref:
                        print("SmartRig refined finger joints:", _ref)
        except Exception as _e:
            print("SmartRig finger refine skipped:", _e)
        self.report({'INFO'}, "Metarig built (Edit Mode). Tweak bones, add samples, then Generate.")
        return {'FINISHED'}



def assign_orphan_bones(rig):
    """Safety net: every bone MUST live in a bone collection, otherwise it can
    never be hidden by the Rig Layers toggles (it stays visible even when the
    user turns everything off). Assign any orphan to a sensible collection by
    name. Returns how many were fixed."""
    if rig is None:
        return 0
    arm = rig.data

    def getc(name):
        return next((c for c in arm.collections_all if c.name == name), None)
    torso = getc("Torso")
    ttweak = getc("Torso (Tweak)") or torso
    defc = getc("DEF"); orgc = getc("ORG"); mchc = getc("MCH")
    n = 0
    for b in arm.bones:
        if len(b.collections) > 0:
            continue
        nm = b.name
        tgt = None
        if nm.startswith("DEF-"):
            tgt = defc
        elif nm.startswith("ORG-"):
            tgt = orgc
        elif nm.startswith("MCH") or nm.startswith("VIS"):
            tgt = mchc
        elif nm.startswith("tweak"):
            tgt = ttweak
        else:
            tgt = torso          # torso / hips / chest / spine_fk / neck / head ...
        if tgt is None:
            tgt = torso or ttweak
        if tgt is not None:
            tgt.assign(b); n += 1
    return n


def _generated_rig():
    """Return the rig generated from the SR metarig, if any."""
    meta = bpy.data.objects.get(META_NAME)
    if meta is not None and meta.data is not None:
        tr = getattr(meta.data, "rigify_target_rig", None)
        if tr is not None:
            return tr
    return bpy.data.objects.get("RIG-" + META_NAME)


class SMARTRIG_OT_generate(bpy.types.Operator):
    bl_idname = "smartrig.generate"
    bl_label = "Generate Rig"
    bl_description = ("Generate (or re-generate) the final Rigify rig from the metarig. "
                      "You can keep editing the metarig and press this again any time.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(META_NAME) is not None

    def execute(self, context):
        meta = bpy.data.objects.get(META_NAME)
        if meta is None:
            self.report({'ERROR'}, "No metarig found. Build the metarig first.")
            return {'CANCELLED'}
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        # pre-flight: catch broken skirt columns BEFORE Rigify fails cryptically,
        # and tell the user exactly how to fix it.
        try:
            from . import skirt as _sk
            probs = _sk.check_skirt_integrity(meta)
            if probs:
                self.report({'ERROR'}, _sk.skirt_integrity_message(probs))
                return {'CANCELLED'}
        except Exception:
            pass
        # make sure the metarig is visible, selected and active
        meta.hide_set(False)
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            pass
        meta.select_set(True)
        context.view_layer.objects.active = meta
        # remember which post-generate features were on, so we can RESTORE them
        # (Rigify regenerate rebuilds the rig and wipes jiggle/follow/anti-pen).
        _prev = _generated_rig()
        _had = {k: (bool(_prev.get(k)) if _prev else False)
                for k in ("sk_jiggle", "sk_follow", "sk_antipen")}
        # NO DUPLICATE RIGS: point Rigify at the ONE existing RIG-* so it
        # re-generates INTO it (not a new RIG-SR_Metarig.001); remove any strays.
        try:
            existing = [o for o in bpy.data.objects
                        if o.type == 'ARMATURE'
                        and o.name.startswith("RIG-" + META_NAME)]
            existing.sort(key=lambda o: o.name)      # RIG-SR_Metarig before .001
            keep = existing[0] if existing else None
            for o in existing[1:]:                   # kill duplicates
                bpy.data.objects.remove(o, do_unlink=True)
            if keep is not None:
                keep.name = "RIG-" + META_NAME
                if hasattr(meta.data, "rigify_target_rig"):
                    meta.data.rigify_target_rig = keep
        except Exception as _e:
            print("SmartRig dedup rigs:", _e)
        try:
            bpy.ops.pose.rigify_generate()
        except Exception as e:
            self.report({'ERROR'}, "Generate failed: %s" % e)
            return {'CANCELLED'}
        # after generate Rigify hides the metarig; record + report
        context.scene.smartrig.rig_generated = True
        # rigging phase done -> auto-UNLOCK the character mesh so the user can
        # select the body again (the lock only matters while placing markers)
        if context.scene.smartrig.lock_mesh:
            context.scene.smartrig.lock_mesh = False
        # professional round-trip: HIDE the metarig, REVEAL + activate the rig so
        # only one is visible at a time (Generate <-> Back to Metarig toggle).
        rig = _generated_rig()
        try:
            meta.hide_set(True)
            meta.hide_viewport = False     # keep it un-hidden in the outliner filter
        except Exception:
            pass
        if rig is not None:
            try:
                rig.hide_set(False)
                bpy.ops.object.select_all(action='DESELECT')
                rig.select_set(True)
                context.view_layer.objects.active = rig
            except Exception:
                pass
            # ARP-style default: NO rubber limbs. Rigify ships IK_Stretch=1.0
            # (the limb stretches infinitely to reach the IK target - reads as
            # a broken "jump/stretch" on a clothed character). Animators can
            # still raise the slider per-limb in the N-panel when wanted.
            try:
                for _pb in rig.pose.bones:
                    if "IK_Stretch" in _pb.keys():
                        _pb["IK_Stretch"] = 0.0
            except Exception:
                pass
            # Finger scale-curl drivers: Rigify emits its OWN when the fingers
            # have a clean primary axis (they all use local X consistently, as
            # they should). ONLY rebuild them if Rigify actually left them
            # MISSING - never override Rigify's consistent drivers, because our
            # per-bone axis detection would mix X/Z axes on a single finger and
            # break the curl (the metarig roll is right but the rig "won't work").
            try:
                from . import skirt as _sk
                if _sk.finger_curl_missing(rig):
                    _sk.ensure_finger_curl(rig)
            except Exception as _e:
                print("SmartRig finger curl:", _e)
        for a in (context.window.screen.areas if context.window else []):
            a.tag_redraw()
        # apply ARP-style skirt collision + region masters on the fresh rig
        p = context.scene.smartrig
        extras = []
        try:
            rg = _generated_rig()
            has_skirt = rg is not None and any(b.name.startswith("DEF-skirt.")
                                               for b in rg.pose.bones)
            if has_skirt:
                from . import skirt as _sk, fit as _fit
                if getattr(p, "skirt_collide", False):
                    _h = None
                    try:
                        _, _e, _h = _fit.compute_joints(p)
                    except Exception:
                        _h = None
                    ncol = _sk.add_skirt_collision(rg, p, _h)
                    if ncol:
                        extras.append("collision")
                # region masters go ON TOP of the collision (re-parent dt roots)
                if getattr(p, "skirt_use_masters", True):
                    nm = _sk.add_skirt_masters(rg, p)
                    if nm:
                        extras.append("%d masters" % nm)
                # restore live skirt features that regenerate wiped
                if _had["sk_jiggle"] and _sk.add_skirt_jiggle(rg, p):
                    extras.append("jiggle")
                if _had["sk_follow"]:
                    _sk.add_skirt_follow_body(rg, p)
                if _had["sk_antipen"]:
                    _sk.add_skirt_antipen(rg, p)
        except Exception as e:
            print("SmartRig skirt extras failed:", e)
        # kandura sleeve automation: roll-up (tashmeer) master per arm +
        # sleeve-end hand follow + cuff ring riding the sleeve END
        try:
            rg = _generated_rig()
            if rg is not None and any(b.name.startswith("DEF-kan_sleeve.")
                                      for b in rg.pose.bones):
                from . import kandura as _kn
                nr = _kn.add_sleeve_rollup(rg, p)
                if nr:
                    extras.append("%d sleeve roll-up master%s"
                                  % (nr, "s" if nr > 1 else ""))
        except Exception as e:
            print("SmartRig kandura sleeve extras failed:", e)
        # safety net: no bone left without a collection (so Rig Layers can hide all)
        try:
            assign_orphan_bones(_generated_rig())
        except Exception as e:
            print("SmartRig orphan-bone assign failed:", e)
        # friendly deform names so users don't get lost: DEF-spine top -> DEF-head,
        # neck segments -> DEF-neck(.NNN) (bones + weight groups; metarig untouched)
        try:
            from . import skirt as _sk
            _sk.rename_head_neck_defs(_generated_rig(), verbose=True)
        except Exception as e:
            print("SmartRig head/neck rename failed:", e)
        if extras:
            self.report({'INFO'}, "Rig generated with skirt %s." % " + ".join(extras))
        else:
            self.report({'INFO'}, "Rig generated. Use 'Back to Metarig' to edit & re-generate.")
        return {'FINISHED'}


class SMARTRIG_OT_back_to_metarig(bpy.types.Operator):
    bl_idname = "smartrig.back_to_metarig"
    bl_label = "Back to Metarig"
    bl_description = ("Hide the generated rig and show the metarig again so you can add "
                      "more samples or tweak bones, then Generate again")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(META_NAME) is not None

    def execute(self, context):
        meta = bpy.data.objects.get(META_NAME)
        if meta is None:
            self.report({'ERROR'}, "No metarig found.")
            return {'CANCELLED'}
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        rig = _generated_rig()
        if rig is not None:
            try:
                rig.hide_set(True)
            except Exception:
                pass
        meta.hide_set(False)
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            pass
        meta.select_set(True)
        context.view_layer.objects.active = meta
        # drop straight into Edit Mode so the user can tweak bones immediately
        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass
        for a in context.window.screen.areas:
            a.tag_redraw()
        self.report({'INFO'}, "Editing metarig (Edit Mode) - tweak bones or add samples, then Generate.")
        return {'FINISHED'}


class SMARTRIG_OT_refit_metarig(bpy.types.Operator):
    bl_idname = "smartrig.refit_metarig"
    bl_label = "Re-fit Metarig to Markers"
    bl_description = ("Move the existing metarig bones to match the current marker "
                      "positions. Keeps any samples you added; no need to rebuild.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(META_NAME) is not None

    def execute(self, context):
        props = context.scene.smartrig
        if props.target_mesh is None:
            ao = context.active_object
            if ao and ao.type == 'MESH':
                props.target_mesh = ao
        mo, err = refit_metarig(props)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        self.report({'INFO'}, "Metarig re-fitted to the current markers.")
        return {'FINISHED'}


class SMARTRIG_OT_toggle_metarig(bpy.types.Operator):
    bl_idname = "smartrig.toggle_metarig"
    bl_label = "Hide / Show Metarig"
    bl_description = ("Hide the metarig so it doesn't clutter the scene while you "
                      "edit the markers. Press again to show it.")

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(META_NAME) is not None

    def execute(self, context):
        mo = bpy.data.objects.get(META_NAME)
        if mo is None:
            self.report({'INFO'}, "No metarig yet.")
            return {'CANCELLED'}
        try:
            mo.hide_set(not mo.hide_get())
        except Exception:
            mo.hide_viewport = not mo.hide_viewport
        for a in (context.window.screen.areas if context.window else []):
            a.tag_redraw()
        return {'FINISHED'}


classes = (SMARTRIG_OT_toggle_group, SMARTRIG_OT_add_sample, SMARTRIG_OT_build_metarig,
           SMARTRIG_OT_generate,           SMARTRIG_OT_back_to_metarig, SMARTRIG_OT_refit_metarig,
           SMARTRIG_OT_toggle_metarig)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
