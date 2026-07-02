"""FIT WIZARD - step-by-step garment fitting (Saeed's design, v1.29.0).

Same philosophy that made the rig marker wizard reliable: AUTOMATION MAKES
THE FIRST GUESS, THE USER CORRECTS IT, the engine receives EXACT inputs.

  Step 1  PLACE   - position the garment over the ACTUAL rigged character
                    (the character is the anatomy reference: its joints are
                    known exactly); front/side view buttons + auto-place.
  Step 2  MARKERS - joint markers appear PRE-FILLED from the garment
                    analysis; the user drags only the wrong ones
                    (e.g. the wrist marker onto the true cuff).
  Step 3  EXTRAS  - register rigid extras (belt, pockets, buttons, flowers,
                    ornaments) into the SRF_Rigid vertex group so they move
                    as solid pieces; small loose parts are auto-rigid.
  Step 4  FIT     - one click: markers override the analysis and the match
                    engine (warp + design preservation) does everything.
"""
import bpy
from mathutils import Vector

MARKER_COL = "SRF_FitMarkers"
MARKER_PREFIX = "SRFM_"
VG_RIGID = "SRF_Rigid"

# marker set shown per garment: only joints the analysis produced
_ORDERED = ("neck", "chest", "pelvis",
            "shoulder_l", "elbow_l", "wrist_l",
            "shoulder_r", "elbow_r", "wrist_r",
            "hip_l", "knee_l", "ankle_l",
            "hip_r", "knee_r", "ankle_r")


def _garment(context):
    return context.scene.smartrig.garment_object


def _role(key):
    """Same colour system as the character markers."""
    if key.endswith("_l"):
        return 'left'
    if key.endswith("_r"):
        return 'right'
    return 'center'


def _isolate(context, g):
    """Everything disappears - only the garment stays (Saeed's spec)."""
    hidden = []
    for ob in context.view_layer.objects:
        if ob is not g and not ob.hide_get():
            try:
                ob.hide_set(True)
                hidden.append(ob.name)
            except Exception:
                pass
    context.scene["srf_wiz_hidden"] = hidden


def _restore(context):
    for nm in list(context.scene.get("srf_wiz_hidden", [])):
        ob = bpy.data.objects.get(nm)
        if ob is not None:
            try:
                ob.hide_set(False)
            except Exception:
                pass
    if "srf_wiz_hidden" in context.scene:
        del context.scene["srf_wiz_hidden"]


def _marker_col(create=False):
    col = bpy.data.collections.get(MARKER_COL)
    if col is None and create:
        col = bpy.data.collections.new(MARKER_COL)
        bpy.context.scene.collection.children.link(col)
    return col


def clear_markers():
    col = bpy.data.collections.get(MARKER_COL)
    if col is not None:
        for ob in list(col.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(col)


def marker_joints():
    """{joint: world Vector} read from the wizard's marker empties. Empty
    dict when the wizard is not in use - the analysis stays in charge."""
    col = bpy.data.collections.get(MARKER_COL)
    if col is None or col.hide_viewport:
        return {}
    out = {}
    for ob in col.objects:
        if ob.name.startswith(MARKER_PREFIX):
            key = ob.name[len(MARKER_PREFIX):].split(".")[0]
            out[key] = ob.matrix_world.translation.copy()
    return out


class SMARTRIG_OT_fitwiz_start(bpy.types.Operator):
    """Start the step-by-step Fit Wizard (place the garment over the
    character, correct the markers, register extras, fit)"""
    bl_idname = "smartrig.fitwiz_start"
    bl_label = "Start Fit Wizard"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        if props.garment_object is None or props.fit_body_object is None:
            self.report({'ERROR'}, "Pick the garment and the body first.")
            return {'CANCELLED'}
        col = bpy.data.collections.get(MARKER_COL)
        if col is not None:
            col.hide_viewport = False
        props.fitwiz_step = 1
        g = props.garment_object
        # SAEED'S SPEC: everything disappears - only the garment - and the
        # view snaps to FRONT (same entrance as the character marker wizard)
        _isolate(context, g)
        from . import markers as _mk
        _mk.set_front_view(context)
        for ob in context.selected_objects:
            ob.select_set(False)
        g.select_set(True)
        context.view_layer.objects.active = g
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_view(bpy.types.Operator):
    """Look at the character from the front or the side while placing"""
    bl_idname = "smartrig.fitwiz_view"
    bl_label = "Wizard View"
    bl_options = {'REGISTER'}

    axis: bpy.props.EnumProperty(items=[('FRONT', "Front", ""),
                                        ('LEFT', "Left", "")])

    def execute(self, context):
        try:
            bpy.ops.view3d.view_axis(type=self.axis)
        except Exception:
            pass
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_markers(bpy.types.Operator):
    """Build (or rebuild) the joint markers, PRE-FILLED by the automatic
    garment analysis - drag any marker that looks wrong"""
    bl_idname = "smartrig.fitwiz_markers"
    bl_label = "Show Markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import mannequin
        props = context.scene.smartrig
        g = props.garment_object
        if g is None:
            self.report({'ERROR'}, "Pick the garment first.")
            return {'CANCELLED'}
        jt = mannequin.garment_skeleton(g)
        if jt is None:
            self.report({'ERROR'}, "Could not analyze the garment.")
            return {'CANCELLED'}
        g["srf_wiz_label"] = str(jt.get("label", "garment"))
        clear_markers()
        col = _marker_col(create=True)
        # size relative to the garment
        bb = [g.matrix_world @ Vector(c) for c in g.bound_box]
        h = max(max(p.z for p in bb) - min(p.z for p in bb), 1e-3)
        made = 0
        for key in _ORDERED:
            v = jt.get(key)
            if not isinstance(v, Vector):
                continue
            em = bpy.data.objects.new(MARKER_PREFIX + key, None)
            # SAME SYSTEM as the character markers: tiny core, the coloured
            # GPU glow (wizard overlay) is the visual; roles share colours
            role = _role(key)
            em.empty_display_type = 'PLAIN_AXES'
            em.empty_display_size = 0.012 * h
            em.location = v
            em.show_name = False
            em.show_in_front = True
            em.color = {'center': (0.2, 0.9, 1.0, 1.0),
                        'left': (1.0, 0.8, 0.1, 1.0),
                        'right': (0.55, 0.45, 0.2, 1.0)}[role]
            col.objects.link(em)
            made += 1
        # right side mirrors the left automatically (drag LEFT markers),
        # exactly like the character wizard
        for key in _ORDERED:
            if not key.endswith("_r"):
                continue
            r_ob = bpy.data.objects.get(MARKER_PREFIX + key)
            l_ob = bpy.data.objects.get(MARKER_PREFIX + key[:-2] + "_l")
            if r_ob is None or l_ob is None:
                continue
            r_ob.lock_location = (True, True, True)
            for c in list(r_ob.constraints):
                r_ob.constraints.remove(c)
            con = r_ob.constraints.new('COPY_LOCATION')
            con.name = "SRF Mirror"
            con.target = l_ob
            con.invert_x = True
            con.target_space = 'WORLD'
            con.owner_space = 'WORLD'
        props.fitwiz_step = 2
        self.report({'INFO'}, "%d markers - drag the wrong ones" % made)
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_extras(bpy.types.Operator):
    """Register rigid extras: belt, pockets, buttons, flowers... select
    their vertices in Edit Mode then press 'Register Selected'"""
    bl_idname = "smartrig.fitwiz_extras"
    bl_label = "Extras Step"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        g = props.garment_object
        if g is None:
            return {'CANCELLED'}
        if g.vertex_groups.get(VG_RIGID) is None:
            g.vertex_groups.new(name=VG_RIGID)
        props.fitwiz_step = 3
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_register(bpy.types.Operator):
    """Add the selected vertices (Edit Mode) to the rigid extras - each
    connected piece will move as ONE solid object during the fit"""
    bl_idname = "smartrig.fitwiz_register"
    bl_label = "Register Selected as Rigid"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        g = _garment(context)
        if g is None or g.mode != 'EDIT':
            self.report({'ERROR'},
                        "Enter Edit Mode on the garment and select the "
                        "extra piece (belt / pocket / button...)")
            return {'CANCELLED'}
        vg = g.vertex_groups.get(VG_RIGID)
        if vg is None:
            vg = g.vertex_groups.new(name=VG_RIGID)
        g.vertex_groups.active_index = vg.index
        bpy.ops.object.vertex_group_assign()
        self.report({'INFO'}, "Registered - it will stay solid")
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_go(bpy.types.Operator):
    """FIT: the markers override the automatic analysis and the full match
    engine runs (warp + design preservation + live garment rig)"""
    bl_idname = "smartrig.fitwiz_go"
    bl_label = "Fit! (wizard)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        if props.garment_object is not None \
                and props.garment_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        _restore(context)          # the character comes back for the fit
        r = bpy.ops.smartrig.mannequin_match()
        if 'FINISHED' in r:
            col = bpy.data.collections.get(MARKER_COL)
            if col is not None:
                col.hide_viewport = True     # kept for a later refit
            props.fitwiz_step = 0
        return r


class SMARTRIG_OT_fitwiz_cancel(bpy.types.Operator):
    """Leave the wizard and remove its markers"""
    bl_idname = "smartrig.fitwiz_cancel"
    bl_label = "Cancel Wizard"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        clear_markers()
        _restore(context)
        context.scene.smartrig.fitwiz_step = 0
        return {'FINISHED'}


_CLASSES = (SMARTRIG_OT_fitwiz_start, SMARTRIG_OT_fitwiz_view,
            SMARTRIG_OT_fitwiz_markers, SMARTRIG_OT_fitwiz_extras,
            SMARTRIG_OT_fitwiz_register, SMARTRIG_OT_fitwiz_go,
            SMARTRIG_OT_fitwiz_cancel)


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
