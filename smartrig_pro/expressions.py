"""Soulify - Expressions & Phonemes (FaceIt-style, editable any time).

Design (mirrors FaceIt's non-destructive workflow):

* One Action "SR_Expressions" on the generated rig holds ONE POSE PER FRAME
  (frame 10, 20, 30 ...).  Every pose is keyed with CONSTANT interpolation on
  the full expression bone-set, so frames never bleed into each other.
* A scene-level list (name / frame / category) drives the UI; clicking an
  item jumps the timeline to that pose -> the animator SEES the expression
  and can simply adjust controls and press "Save Edit" - editable ANY time.
* "Bake to Shape Keys" converts every expression into a shape key on every
  mesh deformed by the rig (skipping meshes an expression doesn't move).
  Baking is repeatable: re-edit a pose, bake again, keys are replaced.
  "Remove Baked Keys" reverses it - nothing is destructive.

Pose amounts are defined in FACE UNITS (S = inter-eye distance, W = mouth
width measured on the actual rig), and world-direction offsets are converted
into each bone's local space via its rest matrix - so the same battery works
on any character, any size.

The jaw-open rotation SIGN is probed numerically on the actual mesh (open =
chin moves down), never assumed.
"""
import bpy
from mathutils import Vector, Euler

ACTION_NAME = "SR_Expressions"
FRAME_STEP = 10
FRAME_START = 10

# ---------------------------------------------------------------- presets --
# Each preset: (name, category, pose)
#   pose = {bone: {"loc": (x,y,z) ARMATURE-space delta in units of S,
#                  "rot": (x,y,z) local euler radians,
#                  "locl": (x,y,z) raw LOCAL-channel values (no conversion)}}
# .L entries are auto-mirrored to .R when the preset name has no _L/_R side.
BLINK_Y = -1.0 / 30.0          # TMP Blink = -30 * PRP-Eye_blink.location.y


def _mirror_vec(v):
    return (-v[0], v[1], v[2])


def build_presets(S, W, jaw_sign):
    J = jaw_sign
    up = lambda a: (0, 0, a)
    down = lambda a: (0, 0, -a)
    fwd = lambda a: (0, -a, 0)        # face looks down -Y
    out_l = lambda a: (a, 0, 0)       # .L side lateral = +X
    in_l = lambda a: (-a, 0, 0)

    def v3(*vs):
        x = [0.0, 0.0, 0.0]
        for v in vs:
            x[0] += v[0]; x[1] += v[1]; x[2] += v[2]
        return tuple(c * S for c in x)

    E = []

    def add(name, cat, pose):
        E.append((name, cat, pose))

    # ---- jaw / mouth ----
    add("jawOpen", 'EXPR', {"CTL-Jaw": {"rot": (J * 0.35, 0, 0)}})
    add("mouthSmile_L", 'EXPR',
        {"CTL-Lips_corn.L": {"loc": v3(out_l(0.10), up(0.09))}})
    add("mouthSmile_R", 'EXPR',
        {"CTL-Lips_corn.R": {"loc": v3(_mirror_vec(out_l(0.10)), up(0.09))}})
    add("mouthSmile", 'EXPR',
        {"CTL-Lips_corn.L": {"loc": v3(out_l(0.10), up(0.09))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(out_l(0.10)), up(0.09))}})
    add("mouthFrown", 'EXPR',
        {"CTL-Lips_corn.L": {"loc": v3(in_l(0.03), down(0.08))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(in_l(0.03)), down(0.08))}})
    add("mouthPucker", 'EXPR',
        {"CTL-Lips_corn.L": {"loc": v3(in_l(0.12))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(in_l(0.12)))},
         "MSTR-Mouth": {"loc": v3(fwd(0.08))}})
    add("mouthFunnel", 'EXPR',
        {"CTL-Jaw": {"rot": (J * 0.12, 0, 0)},
         "CTL-Lips_corn.L": {"loc": v3(in_l(0.10))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(in_l(0.10)))},
         "MSTR-Mouth": {"loc": v3(fwd(0.06))}})
    add("mouthLeft", 'EXPR', {"MSTR-Mouth": {"loc": v3(out_l(0.10))}})
    add("mouthRight", 'EXPR',
        {"MSTR-Mouth": {"loc": v3(_mirror_vec(out_l(0.10)))}})
    add("mouthPress", 'EXPR',
        {"CTL-Lips_main_upp": {"loc": v3(down(0.02))},
         "CTL-Lips_main_low": {"loc": v3(up(0.02))}})
    add("upperLipUp", 'EXPR', {"CTL-Lips_main_upp": {"loc": v3(up(0.05))}})
    add("lowerLipDown", 'EXPR', {"CTL-Lips_main_low": {"loc": v3(down(0.05))}})
    # ---- brows ----
    add("browRaise", 'EXPR',
        {"CTL-Brow_all.L": {"loc": v3(up(0.10))},
         "CTL-Brow_all.R": {"loc": v3(up(0.10))}})
    add("browInnerUp", 'EXPR',
        {"CTL-Brow_in.L": {"loc": v3(up(0.08))},
         "CTL-Brow_in.R": {"loc": v3(up(0.08))}})
    add("browDown_L", 'EXPR',
        {"CTL-Brow_all.L": {"loc": v3(down(0.06), in_l(0.02))}})
    add("browDown_R", 'EXPR',
        {"CTL-Brow_all.R": {"loc": v3(down(0.06), _mirror_vec(in_l(0.02)))}})
    add("browDown", 'EXPR',
        {"CTL-Brow_all.L": {"loc": v3(down(0.06), in_l(0.02))},
         "CTL-Brow_all.R": {"loc": v3(down(0.06), _mirror_vec(in_l(0.02)))}})
    add("browOuterUp_L", 'EXPR', {"CTL-Brow_out.L": {"loc": v3(up(0.08))}})
    add("browOuterUp_R", 'EXPR', {"CTL-Brow_out.R": {"loc": v3(up(0.08))}})
    # ---- eyes ----
    add("eyeBlink_L", 'EXPR', {"PRP-Eye_blink.L": {"locl": (0, BLINK_Y, 0)}})
    add("eyeBlink_R", 'EXPR', {"PRP-Eye_blink.R": {"locl": (0, BLINK_Y, 0)}})
    add("eyeBlink", 'EXPR',
        {"PRP-Eye_blink.L": {"locl": (0, BLINK_Y, 0)},
         "PRP-Eye_blink.R": {"locl": (0, BLINK_Y, 0)}})
    add("eyeLookUp", 'EXPR', {"P-Eye_target": {"loc": v3(up(0.25))}})
    add("eyeLookDown", 'EXPR', {"P-Eye_target": {"loc": v3(down(0.25))}})
    add("eyeLookLeft", 'EXPR', {"P-Eye_target": {"loc": v3(out_l(0.35))}})
    add("eyeLookRight", 'EXPR',
        {"P-Eye_target": {"loc": v3(_mirror_vec(out_l(0.35)))}})
    # ---- cheeks / nose ----
    add("cheekPuff", 'EXPR',
        {"CTL-Cheek_puff.L": {"loc": v3(out_l(0.08), fwd(0.03))},
         "CTL-Cheek_puff.R": {"loc": v3(_mirror_vec(out_l(0.08)), fwd(0.03))}})
    add("cheekSquint", 'EXPR',
        {"CTL-Cheek_all.L": {"loc": v3(up(0.05))},
         "CTL-Cheek_all.R": {"loc": v3(up(0.05))}})
    add("noseSneer", 'EXPR', {"MSTR-Nose": {"loc": v3(up(0.035))}})

    # ---- phonemes / visemes (Preston-Blair-ish set) ----
    add("viseme_AI", 'VISEME', {"CTL-Jaw": {"rot": (J * 0.28, 0, 0)}})
    add("viseme_E", 'VISEME',
        {"CTL-Jaw": {"rot": (J * 0.08, 0, 0)},
         "CTL-Lips_corn.L": {"loc": v3(out_l(0.09))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(out_l(0.09)))}})
    add("viseme_O", 'VISEME',
        {"CTL-Jaw": {"rot": (J * 0.18, 0, 0)},
         "CTL-Lips_corn.L": {"loc": v3(in_l(0.10))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(in_l(0.10)))},
         "MSTR-Mouth": {"loc": v3(fwd(0.06))}})
    add("viseme_U", 'VISEME',
        {"CTL-Jaw": {"rot": (J * 0.08, 0, 0)},
         "CTL-Lips_corn.L": {"loc": v3(in_l(0.14))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(in_l(0.14)))},
         "MSTR-Mouth": {"loc": v3(fwd(0.09))}})
    add("viseme_MBP", 'VISEME',
        {"CTL-Lips_main_upp": {"loc": v3(down(0.015))},
         "CTL-Lips_main_low": {"loc": v3(up(0.015))}})
    add("viseme_FV", 'VISEME',
        {"CTL-Jaw": {"rot": (J * 0.05, 0, 0)},
         "CTL-Lips_main_low": {"loc": v3(up(0.045), (0, 0.03, 0))}})
    add("viseme_L", 'VISEME',
        {"CTL-Jaw": {"rot": (J * 0.15, 0, 0)},
         "MSTR-Tongue": {"loc": v3(up(0.04), fwd(0.03))}})
    add("viseme_WQ", 'VISEME',
        {"CTL-Jaw": {"rot": (J * 0.05, 0, 0)},
         "CTL-Lips_corn.L": {"loc": v3(in_l(0.12))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(in_l(0.12)))},
         "MSTR-Mouth": {"loc": v3(fwd(0.07))}})
    add("viseme_SDT", 'VISEME', {"CTL-Jaw": {"rot": (J * 0.06, 0, 0)}})
    add("viseme_TH", 'VISEME',
        {"CTL-Jaw": {"rot": (J * 0.10, 0, 0)},
         "MSTR-Tongue": {"loc": v3(fwd(0.05))}})
    add("viseme_R", 'VISEME',
        {"CTL-Jaw": {"rot": (J * 0.07, 0, 0)},
         "CTL-Lips_corn.L": {"loc": v3(in_l(0.06))},
         "CTL-Lips_corn.R": {"loc": v3(_mirror_vec(in_l(0.06)))}})
    return E


# ------------------------------------------------------------------ helpers
def _rig():
    from . import face as _face
    return _face._target_rig()


def _face_units(rig):
    """S = inter-eye distance, W = mouth width, measured on the rig."""
    b = rig.data.bones
    eL, eR = b.get("FK-Eye.L"), b.get("FK-Eye.R")
    cL, cR = b.get("CTL-Lips_corn.L"), b.get("CTL-Lips_corn.R")
    S = (eL.head_local - eR.head_local).length if (eL and eR) else 0.06
    W = (cL.head_local - cR.head_local).length if (cL and cR) else S
    return max(S, 1e-4), max(W, 1e-4)


def _arm_to_local(pb, vec):
    """ARMATURE-space delta -> the bone's local location channel."""
    try:
        m = pb.bone.matrix_local.to_3x3().inverted()
    except Exception:
        return vec
    v = m @ Vector(vec)
    return (v.x, v.y, v.z)


def _probe_jaw_sign(context, rig, mesh):
    """Open = chin moves DOWN.  Try +0.35 on CTL-Jaw X; measure the mean
    world-Z delta of the DEF-Jaw vertex group.  Returns +1.0 or -1.0."""
    pb = rig.pose.bones.get("CTL-Jaw")
    if pb is None or mesh is None:
        return 1.0
    vg = mesh.vertex_groups.get("DEF-Jaw")
    if vg is None:
        return 1.0
    gi = vg.index
    idx = [v.index for v in mesh.data.vertices
           if any(g.group == gi and g.weight > 0.3 for g in v.groups)]
    if not idx:
        return 1.0
    idx = idx[:400]
    dg = context.evaluated_depsgraph_get()

    def _mean_z():
        context.view_layer.update()
        dg.update()
        ev = mesh.evaluated_get(dg).to_mesh()
        z = sum(ev.vertices[i].co.z for i in idx) / len(idx)
        mesh.evaluated_get(dg).to_mesh_clear()
        return z

    old_mode = pb.rotation_mode
    pb.rotation_mode = 'XYZ'
    pb.rotation_euler = (0, 0, 0)
    z0 = _mean_z()
    pb.rotation_euler = (0.35, 0, 0)
    z1 = _mean_z()
    pb.rotation_euler = (0, 0, 0)
    pb.rotation_mode = old_mode
    context.view_layer.update()
    return 1.0 if z1 < z0 else -1.0


def _pose_bone_set(presets):
    s = set()
    for _, _, pose in presets:
        s.update(pose.keys())
    return sorted(s)


def _key_bone(action, rig, pb, frame, loc, rot):
    """Insert CONSTANT keys for location + rotation on `pb` at `frame`."""
    pb.location = loc
    if pb.rotation_mode == 'QUATERNION':
        e = Euler(rot, 'XYZ')
        pb.rotation_quaternion = e.to_quaternion()
        rpath, n = "rotation_quaternion", 4
    else:
        pb.rotation_euler = rot
        rpath, n = "rotation_euler", 3
    pb.keyframe_insert("location", frame=frame, group=pb.name)
    pb.keyframe_insert(rpath, frame=frame, group=pb.name)


def _action_fcurves(act):
    """All fcurves of an action - works on legacy AND slotted (4.4+/5.x)
    actions, where `Action.fcurves` no longer exists."""
    if hasattr(act, "fcurves"):
        yield from act.fcurves
        return
    for layer in act.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


def _set_constant(action):
    for fc in _action_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'


def _apply_preset_frame(action, rig, preset, frame, S):
    name, cat, pose = preset
    for bn, d in pose.items():
        pb = rig.pose.bones.get(bn)
        if pb is None:
            continue
        loc = d.get("locl")
        if loc is None:
            loc = _arm_to_local(pb, d.get("loc", (0, 0, 0)))
        rot = d.get("rot", (0, 0, 0))
        _key_bone(action, rig, pb, frame, loc, rot)


def _rest_key_all(action, rig, bones, frame):
    for bn in bones:
        pb = rig.pose.bones.get(bn)
        if pb is None:
            continue
        _key_bone(action, rig, pb, frame, (0, 0, 0), (0, 0, 0))


def _reset_pose(rig, bones):
    for bn in bones:
        pb = rig.pose.bones.get(bn)
        if pb is None:
            continue
        pb.location = (0, 0, 0)
        if pb.rotation_mode == 'QUATERNION':
            pb.rotation_quaternion = (1, 0, 0, 0)
        else:
            pb.rotation_euler = (0, 0, 0)


# ---------------------------------------------------------------- data (UI)
class SR_ExpressionItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    frame: bpy.props.IntProperty()
    category: bpy.props.EnumProperty(items=[
        ('EXPR', "Expression", ""), ('VISEME', "Phoneme", "")])


def _on_index(self, context):
    sc = context.scene
    i = sc.sr_face_expr_index
    items = sc.sr_face_expressions
    if 0 <= i < len(items):
        sc.frame_current = items[i].frame


class SMARTRIG_UL_expressions(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname):
        row = layout.row(align=True)
        ic = 'MONKEY' if item.category == 'EXPR' else 'SYNTAX_OFF'
        row.label(text=item.name, icon=ic)
        row.label(text=str(item.frame))


# --------------------------------------------------------------- operators
class SMARTRIG_OT_expr_generate(bpy.types.Operator):
    """Generate the editable expression + phoneme battery (one pose per
    frame in the SR_Expressions action).  Re-running rebuilds all poses"""
    bl_idname = "smartrig.expr_generate"
    bl_label = "Generate Expressions & Phonemes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = _rig()
        if rig is None:
            self.report({'ERROR'}, "Build the face rig first")
            return {'CANCELLED'}
        props = context.scene.smartrig
        mesh = getattr(props, "target_mesh", None)
        S, W = _face_units(rig)
        jaw_sign = _probe_jaw_sign(context, rig, mesh)
        presets = build_presets(S, W, jaw_sign)
        bones = _pose_bone_set(presets)

        act = bpy.data.actions.get(ACTION_NAME)
        if act is not None:
            # full rebuild: recreating the action is the reliable wipe on
            # slotted actions (5.x has no Action.fcurves to clear)
            bpy.data.actions.remove(act)
        act = bpy.data.actions.new(ACTION_NAME)
        act.use_fake_user = True
        if rig.animation_data is None:
            rig.animation_data_create()
        rig.animation_data.action = act
        try:  # slotted actions: keyframe_insert auto-creates the slot
            if act.slots and rig.animation_data.action_slot is None:
                rig.animation_data.action_slot = act.slots[0]
        except Exception:
            pass

        # neutral frame 0 + rest keys on EVERY expression frame (isolation)
        _rest_key_all(act, rig, bones, 0)
        items = context.scene.sr_face_expressions
        items.clear()
        f = FRAME_START
        for preset in presets:
            _rest_key_all(act, rig, bones, f)
            _apply_preset_frame(act, rig, preset, f, S)
            it = items.add()
            it.name, it.category, it.frame = preset[0], preset[1], f
            f += FRAME_STEP
        _set_constant(act)
        _reset_pose(rig, bones)
        context.scene.frame_current = 0
        context.scene.sr_face_expr_index = 0
        rig.data["sr_expr_units"] = (S, W, jaw_sign)
        self.report({'INFO'},
                    "%d expressions + phonemes on '%s' (S=%.3f jaw=%+d)"
                    % (len(presets), ACTION_NAME, S, int(jaw_sign)))
        return {'FINISHED'}


class SMARTRIG_OT_expr_save_edit(bpy.types.Operator):
    """Save the CURRENT pose into the active expression's frame (edit an
    expression any time: pick it in the list, pose controls, press this)"""
    bl_idname = "smartrig.expr_save_edit"
    bl_label = "Save Expression Edit"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        rig = _rig()
        act = bpy.data.actions.get(ACTION_NAME)
        i = sc.sr_face_expr_index
        if rig is None or act is None or not (
                0 <= i < len(sc.sr_face_expressions)):
            self.report({'ERROR'}, "Generate the battery first")
            return {'CANCELLED'}
        frame = sc.sr_face_expressions[i].frame
        n = 0
        for pb in rig.pose.bones:
            if pb.name.startswith(("DEF-", "DSP-", "ORG-", "MCH-")):
                # allow DEF micro controls too - they ARE animator handles
                if not pb.custom_shape:
                    continue
            ident = (abs(pb.location.length) > 1e-6 or
                     (pb.rotation_mode == 'QUATERNION' and
                      abs(pb.rotation_quaternion.angle) > 1e-4) or
                     (pb.rotation_mode != 'QUATERNION' and
                      Vector(pb.rotation_euler).length > 1e-6))
            tag = 'pose.bones["%s"]' % pb.name
            has_key = any(tag in fc.data_path for fc in _action_fcurves(act))
            if not ident and not has_key:
                continue
            rot = (pb.rotation_euler if pb.rotation_mode != 'QUATERNION'
                   else pb.rotation_quaternion.to_euler('XYZ'))
            _key_bone(act, rig, pb, frame, tuple(pb.location), tuple(rot))
            n += 1
        _set_constant(act)
        self.report({'INFO'}, "Saved %d bones into '%s' (frame %d)"
                    % (n, sc.sr_face_expressions[i].name, frame))
        return {'FINISHED'}


class SMARTRIG_OT_expr_bake(bpy.types.Operator):
    """Bake every expression/phoneme to shape keys on all meshes deformed
    by the rig.  Repeatable: existing baked keys are replaced"""
    bl_idname = "smartrig.expr_bake"
    bl_label = "Bake to Shape Keys"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        rig = _rig()
        act = bpy.data.actions.get(ACTION_NAME)
        if rig is None or act is None or not len(sc.sr_face_expressions):
            self.report({'ERROR'}, "Generate the battery first")
            return {'CANCELLED'}
        meshes = [ob for ob in sc.objects if ob.type == 'MESH'
                  and not ob.name.startswith(("HLP-", "WGT", "SR_", "GEO-"))
                  and any(m.type == 'ARMATURE' and m.object is rig
                          for m in ob.modifiers)]
        dg = context.evaluated_depsgraph_get()
        old_frame = sc.frame_current

        def _eval_cos(ob):
            ev = ob.evaluated_get(dg).to_mesh()
            cos = [v.co.copy() for v in ev.vertices]
            ob.evaluated_get(dg).to_mesh_clear()
            return cos

        sc.frame_set(0)
        base = {ob.name: _eval_cos(ob) for ob in meshes}
        made = 0
        for item in sc.sr_face_expressions:
            sc.frame_set(item.frame)
            for ob in meshes:
                cos = _eval_cos(ob)
                b = base[ob.name]
                # shape-key data size == ORIGINAL vertex count; skip any
                # mesh whose evaluated count differs (generative modifiers)
                if len(cos) != len(b) or len(cos) != len(ob.data.vertices):
                    continue
                maxd = max((cos[i] - b[i]).length for i in range(len(cos))) \
                    if cos else 0.0
                if maxd < 1e-6:
                    continue
                if ob.data.shape_keys is None:
                    ob.shape_key_add(name="Basis", from_mix=False)
                kb = ob.data.shape_keys.key_blocks.get(item.name)
                if kb is None:
                    kb = ob.shape_key_add(name=item.name, from_mix=False)
                for i, co in enumerate(cos):
                    kb.data[i].co = co
                kb.value = 0.0
                baked = set(ob.data.get("sr_expr_baked", []))
                baked.add(item.name)
                ob.data["sr_expr_baked"] = sorted(baked)
                made += 1
        sc.frame_set(old_frame)
        self.report({'INFO'}, "Baked %d shape keys" % made)
        return {'FINISHED'}


class SMARTRIG_OT_expr_unbake(bpy.types.Operator):
    """Remove all shape keys created by Bake (fully reversible)"""
    bl_idname = "smartrig.expr_unbake"
    bl_label = "Remove Baked Keys"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        n = 0
        for ob in context.scene.objects:
            if ob.type != 'MESH':
                continue
            names = list(ob.data.get("sr_expr_baked", []))
            if not names or ob.data.shape_keys is None:
                continue
            for nm in names:
                kb = ob.data.shape_keys.key_blocks.get(nm)
                if kb is not None:
                    ob.shape_key_remove(kb)
                    n += 1
            del ob.data["sr_expr_baked"]
        self.report({'INFO'}, "Removed %d baked keys" % n)
        return {'FINISHED'}


# --------------------------------------------------------------------- UI
def draw_panel(layout, context):
    sc = context.scene
    box = layout.box()
    box.label(text="Expressions & Phonemes", icon='MONKEY')
    r = box.row(); r.scale_y = 1.4
    have = bool(len(sc.sr_face_expressions)) and \
        bpy.data.actions.get(ACTION_NAME) is not None
    r.operator("smartrig.expr_generate",
               text=("Regenerate Battery" if have else
                     "Generate Expressions & Phonemes"),
               icon='ADD')
    if have:
        box.template_list("SMARTRIG_UL_expressions", "",
                          sc, "sr_face_expressions",
                          sc, "sr_face_expr_index", rows=6)
        box.label(text="Pick one, pose controls, then:", icon='INFO')
        r = box.row(align=True); r.scale_y = 1.2
        r.operator("smartrig.expr_save_edit", icon='FILE_TICK')
        r = box.row(align=True)
        r.operator("smartrig.expr_bake", icon='SHAPEKEY_DATA')
        r.operator("smartrig.expr_unbake", icon='X')


_classes = (SR_ExpressionItem, SMARTRIG_UL_expressions,
            SMARTRIG_OT_expr_generate, SMARTRIG_OT_expr_save_edit,
            SMARTRIG_OT_expr_bake, SMARTRIG_OT_expr_unbake)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.sr_face_expressions = bpy.props.CollectionProperty(
        type=SR_ExpressionItem)
    bpy.types.Scene.sr_face_expr_index = bpy.props.IntProperty(
        default=0, update=_on_index)


def unregister():
    for attr in ("sr_face_expressions", "sr_face_expr_index"):
        try:
            delattr(bpy.types.Scene, attr)
        except Exception:
            pass
    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
