"""
fingers_manual.py — reliable MANUAL finger / toe markers (any count, any character).

Workflow: 'Add Hand Fingers' / 'Add Foot Fingers' -> click each JOINT on the body
(base to tip), one click per joint. 'Done -> Next' (or Enter) starts the next digit;
'Finish' (or Esc) ends. The LEFT side mirrors LIVE to the RIGHT (a copy-location
constraint), so editing / undo stays mirrored. 'Hide Skeleton' lets you focus.
Edit afterwards by selecting & dragging a marker; 'Straighten' / 'Align Bases' help.

Markers: empties  fm.<part>.<side>.<name>.<j>   part in {hand,foot}, side in {L,R}.
Build reads the LEFT markers and mirrors to the right - no detection.
"""
import bpy
from mathutils import Vector
from bpy_extras import view3d_utils
from . import utils, markers

ORDER = {"hand": ["thumb", "index", "middle", "ring", "pinky"],
         "foot": ["toe1", "toe2", "toe3", "toe4", "toe5"],
         "palm": ["palm1", "palm2", "palm3", "palm4"]}
PREFIX = {"hand": "finger", "foot": "toe", "palm": "palm"}

# a distinct colour per digit (thumb->pinky / toe1->toe5)
FINGER_COLOR = {
    "thumb": (1.0, 0.30, 0.30), "index": (1.0, 0.62, 0.15), "middle": (1.0, 0.93, 0.20),
    "ring": (0.35, 1.0, 0.45), "pinky": (0.35, 0.70, 1.0),
    "toe1": (1.0, 0.30, 0.30), "toe2": (1.0, 0.62, 0.15), "toe3": (1.0, 0.93, 0.20),
    "toe4": (0.35, 1.0, 0.45), "toe5": (0.35, 0.70, 1.0),
    "palm1": (0.8, 0.55, 1.0), "palm2": (0.6, 0.6, 1.0), "palm3": (0.55, 0.8, 1.0),
    "palm4": (0.5, 0.95, 0.9),
}


def color_for(name):
    if name in FINGER_COLOR:
        return FINGER_COLOR[name]
    import colorsys, hashlib
    hue = (int(hashlib.md5(name.encode()).hexdigest(), 16) % 360) / 360.0
    return colorsys.hsv_to_rgb(hue, 0.7, 1.0)


def _h_of(mesh):
    co = utils.read_world_coords(mesh)
    return float(co[:, 2].max() - co[:, 2].min())


def _name(part, side, nm, j):
    return "fm.%s.%s.%s.%d" % (part, side, nm, j)


def list_fingers(part, side="L"):
    out = {}
    pre = "fm.%s.%s." % (part, side)
    for o in bpy.data.objects:
        if o.name.startswith(pre):
            try:
                _, _, _, nm, j = o.name.split(".")
            except ValueError:
                continue
            out.setdefault(nm, []).append((int(j), o))
    return {nm: [o for _, o in sorted(v)] for nm, v in out.items()}


def next_finger_name(part, side="L"):
    have = set(list_fingers(part, side).keys())
    for nm in ORDER[part]:
        if nm not in have:
            return nm
    i = 6
    while ("%s%d" % (PREFIX[part], i)) in have:
        i += 1
    return "%s%d" % (PREFIX[part], i)


def valid_name(part, side, nm):
    """True only if `nm` legitimately belongs to `part` (prevents a finger name
    leaking into the palm part, or vice-versa)."""
    import re
    if not nm:
        return False
    if nm in ORDER.get(part, []):
        return True
    if nm in list_fingers(part, side):
        return True
    return bool(re.match(r"^%s\d+$" % PREFIX.get(part, ""), nm))


def _mirror_marker(part, nm, j, h):
    """ensure the right-side counterpart exists and live-mirrors the left one."""
    ln = _name(part, "L", nm, j)
    rn = _name(part, "R", nm, j)
    r = bpy.data.objects.get(rn) or markers._new_empty(rn)
    lo = bpy.data.objects.get(ln)
    if lo:
        r.location = Vector((-lo.location.x, lo.location.y, lo.location.z))
    markers._style(r, 0.03 * h, 'finger')
    markers._add_mirror_constraint(r, ln)        # COPY_LOCATION, invert X -> live mirror


def _free_empty(part, side, nm, j, loc, h):
    name = _name(part, side, nm, j)
    o = bpy.data.objects.get(name) or markers._new_empty(name)
    o.location = loc
    markers._style(o, 0.03 * h, 'finger')        # small plain-axes (selectable)
    c = color_for(nm)
    o.color = (c[0], c[1], c[2], 1.0)            # per-digit colour
    return o


def add_joint(mesh, part, side, nm, loc, mirror=True):
    h = _h_of(mesh)
    j = len(list_fingers(part, side).get(nm, []))
    _free_empty(part, side, nm, j, Vector(loc), h)
    if mirror and side == "L":
        _mirror_marker(part, nm, j, h)


def remove_finger(part, side, nm):
    for s in ("L", "R"):
        for o in list_fingers(part, s).get(nm, []):
            bpy.data.objects.remove(o, do_unlink=True)


def has_manual(part=None, side="L"):
    if part is None:
        return any(bool(list_fingers(p, side)) for p in ("hand", "foot", "palm"))
    return len(list_fingers(part, side)) > 0


def manual_chains_world(part, side="L"):
    return {nm: [o.location.copy() for o in chain]
            for nm, chain in list_fingers(part, side).items() if len(chain) >= 2}


def straighten(part, side, nm):
    chain = list_fingers(part, side).get(nm)
    if not chain or len(chain) < 3:
        return
    p0 = chain[0].location.copy(); pn = chain[-1].location.copy()
    n = len(chain) - 1
    for i, o in enumerate(chain):
        o.location = p0.lerp(pn, i / n)


def align_bases(part, side):
    """Align the knuckle (base) joints to one level. For the HAND the thumb is
    EXCLUDED - only the four fingers line up at the knuckles."""
    fingers = list_fingers(part, side)
    names = [nm for nm in fingers if not (part == "hand" and nm == "thumb")]
    bases = [fingers[nm][0] for nm in names if fingers[nm]]
    if len(bases) < 2:
        return
    z = sum(b.location.z for b in bases) / len(bases)
    for b in bases:
        b.location = Vector((b.location.x, b.location.y, z))


def _push():
    try:
        bpy.ops.ed.undo_push(message="SmartRig finger marker")
    except Exception:
        pass


# --------------------------------------------------------------- operators
class SMARTRIG_OT_finger_place(bpy.types.Operator):
    bl_idname = "smartrig.finger_place"
    bl_label = "Place Joints"
    bl_description = ("Click each joint on the body (base->tip). Enter=next digit, "
                      "Esc=finish. Move onto the panel to pause. Left mirrors to right.")
    bl_options = {'REGISTER', 'UNDO'}
    fname: bpy.props.StringProperty(default="")
    side: bpy.props.StringProperty(default="L")
    part: bpy.props.StringProperty(default="hand")

    def invoke(self, context, event):
        p = context.scene.smartrig
        self.mesh = p.target_mesh or context.active_object
        if self.mesh is None or self.mesh.type != 'MESH':
            self.report({'ERROR'}, "Select your character first.")
            return {'CANCELLED'}
        # stay inside the guided flow (so the 'rig hands?' step still appears);
        # just release the body-marker front-view lock
        try:
            markers.lock_front_view(context, False)
        except Exception:
            pass
        p.placing = False
        self._h = _h_of(self.mesh)
        self.area = context.area
        self._entered = False
        part_changed = (p.finger_part != self.part)
        p.finger_part = self.part
        if self.fname and valid_name(self.part, self.side, self.fname):
            p.finger_current = self.fname
        elif part_changed or not valid_name(self.part, self.side, p.finger_current):
            # only use a name that belongs to THIS part (no cross-part leaks)
            p.finger_current = next_finger_name(self.part, self.side)
        p.finger_placing = True
        # zoom the view onto the hand (palm/fingers) or foot (toes) for precise clicks
        try:
            if self.part in ("palm", "hand"):
                markers.set_hand_view_focus(context, "wrist.L")
            elif self.part == "foot":
                markers.set_top_view_focus(context, "ankle.L")
        except Exception:
            pass
        try:
            context.window.cursor_modal_set('CROSSHAIR')
        except Exception:
            pass
        for a in context.window.screen.areas:
            a.tag_redraw()
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Click joints of '%s' (base->tip). Enter=next, Esc=finish." % p.finger_current)
        return {'RUNNING_MODAL'}

    def _pause(self, context):
        try:
            context.window.cursor_modal_restore()
        except Exception:
            pass
        context.scene.smartrig.finger_placing = False
        for a in context.window.screen.areas:
            a.tag_redraw()

    def modal(self, context, event):
        p = context.scene.smartrig
        if not p.finger_placing:
            self._pause(context)
            return {'CANCELLED'}
        if event.type == 'Z' and event.value == 'PRESS' and (event.ctrl or event.oskey):
            self.report({'INFO'}, "Use Backspace to remove the last joint (undo is paused while placing).")
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE':
            if markers._over_panel(self, event):
                if self._entered:
                    self._pause(context)
                    return {'FINISHED'}
            else:
                self._entered = True
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if len(list_fingers(self.part, self.side).get(p.finger_current, [])) < 2:
                self.report({'WARNING'}, "Place at least 2 joints first.")
                return {'RUNNING_MODAL'}
            p.finger_current = next_finger_name(self.part, self.side)
            self.report({'INFO'}, "Next: '%s'." % p.finger_current)
            return {'RUNNING_MODAL'}
        if event.type in {'BACK_SPACE', 'DEL'} and event.value == 'PRESS':
            chain = list_fingers(self.part, self.side).get(p.finger_current, [])
            if chain:
                j = len(chain) - 1
                ro = bpy.data.objects.get(_name(self.part, "R", p.finger_current, j))
                if ro:
                    bpy.data.objects.remove(ro, do_unlink=True)
                bpy.data.objects.remove(chain[-1], do_unlink=True)
                _push()
                self.report({'INFO'}, "Removed last joint of '%s'." % p.finger_current)
                for a in context.window.screen.areas:
                    a.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'ESC' and event.value == 'PRESS':
            p.finger_placing = False
            p.finger_current = ""
            self._pause(context)
            return {'FINISHED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            region = context.region; rv3d = context.region_data
            if region is None or rv3d is None or region.type != 'WINDOW':
                return {'PASS_THROUGH'}
            coord = (event.mouse_region_x, event.mouse_region_y)
            o = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            d = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            hit = markers._volume_hit(self.mesh, o, d)
            if hit is None:
                self.report({'WARNING'}, "Click ON the body.")
                return {'RUNNING_MODAL'}
            add_joint(self.mesh, self.part, self.side, p.finger_current, hit,
                      mirror=p.mirror)
            _push()
            for a in context.window.screen.areas:
                a.tag_redraw()
            # AUTO-ADVANCE: palm = 2 joints (1 bone), hand finger = 4 joints (3 bones).
            # The user can still press Enter to advance early (cartoon hands).
            target = {'palm': 2, 'hand': 4}.get(self.part)
            if target is not None:
                n = len(list_fingers(self.part, self.side).get(p.finger_current, []))
                if n >= target:
                    p.finger_current = next_finger_name(self.part, self.side)
                    self.report({'INFO'}, "Auto-advance -> '%s'." % p.finger_current)
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


class SMARTRIG_OT_finger_next(bpy.types.Operator):
    bl_idname = "smartrig.finger_next"
    bl_label = "Done - Next"
    bl_description = "Finish this digit and start the next one"
    bl_options = {'REGISTER', 'UNDO'}
    side: bpy.props.StringProperty(default="L")
    part: bpy.props.StringProperty(default="hand")

    def execute(self, context):
        p = context.scene.smartrig
        cur = p.finger_current
        if cur and len(list_fingers(self.part, self.side).get(cur, [])) < 2:
            self.report({'WARNING'}, "Place at least 2 joints for '%s' first." % cur)
            return {'CANCELLED'}
        p.finger_current = next_finger_name(self.part, self.side)
        return bpy.ops.smartrig.finger_place('INVOKE_DEFAULT', fname=p.finger_current,
                                             side=self.side, part=self.part)


class SMARTRIG_OT_finger_resume(bpy.types.Operator):
    bl_idname = "smartrig.finger_resume"
    bl_label = "Resume"
    bl_description = "Resume placing joints"
    side: bpy.props.StringProperty(default="L")

    def execute(self, context):
        p = context.scene.smartrig
        nm = p.finger_current or next_finger_name(p.finger_part or "hand", self.side)
        return bpy.ops.smartrig.finger_place('INVOKE_DEFAULT', fname=nm, side=self.side,
                                             part=(p.finger_part or "hand"))


class SMARTRIG_OT_finger_finish(bpy.types.Operator):
    bl_idname = "smartrig.finger_finish"
    bl_label = "Finish"
    bl_description = "Stop placing"

    def execute(self, context):
        p = context.scene.smartrig
        p.finger_placing = False
        p.finger_current = ""
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_finger_select(bpy.types.Operator):
    bl_idname = "smartrig.finger_select"
    bl_label = "Select Digit Markers"
    bl_description = ("Select all of this digit's markers (left side) so you can Align "
                      "them with the X / Y / Z buttons without picking each one by hand.")
    bl_options = {'REGISTER', 'UNDO'}
    fname: bpy.props.StringProperty()
    side: bpy.props.StringProperty(default="L")
    part: bpy.props.StringProperty(default="hand")

    def execute(self, context):
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        try:
            for o in context.selected_objects:
                o.select_set(False)
        except Exception:
            pass
        chain = list_fingers(self.part, self.side).get(self.fname, [])
        for o in chain:
            try:
                o.hide_set(False)
                o.hide_select = False
                o.select_set(True)
            except Exception:
                pass
        if chain:
            context.view_layer.objects.active = chain[0]
        for a in context.window.screen.areas:
            a.tag_redraw()
        self.report({'INFO'}, "Selected %d markers of '%s' - now press Align X/Y/Z." % (len(chain), self.fname))
        return {'FINISHED'}


class SMARTRIG_OT_finger_remove(bpy.types.Operator):
    bl_idname = "smartrig.finger_remove"
    bl_label = "Remove"
    bl_description = "Delete this digit's markers (both sides)"
    bl_options = {'REGISTER', 'UNDO'}
    fname: bpy.props.StringProperty()
    side: bpy.props.StringProperty(default="L")
    part: bpy.props.StringProperty(default="hand")

    def execute(self, context):
        remove_finger(self.part, self.side, self.fname)
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_finger_straighten(bpy.types.Operator):
    bl_idname = "smartrig.finger_straighten"
    bl_label = "Straighten"
    bl_description = "Put this digit's joints on a straight line (base -> tip)"
    bl_options = {'REGISTER', 'UNDO'}
    fname: bpy.props.StringProperty()
    side: bpy.props.StringProperty(default="L")
    part: bpy.props.StringProperty(default="hand")

    def execute(self, context):
        straighten(self.part, self.side, self.fname)
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_finger_align_bases(bpy.types.Operator):
    bl_idname = "smartrig.finger_align_bases"
    bl_label = "Align Bases (same level)"
    bl_description = "Move every digit's base joint to the same level"
    bl_options = {'REGISTER', 'UNDO'}
    side: bpy.props.StringProperty(default="L")
    part: bpy.props.StringProperty(default="hand")

    def execute(self, context):
        align_bases(self.part, self.side)
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_finger_hide(bpy.types.Operator):
    bl_idname = "smartrig.finger_hide"
    bl_label = "Hide / Show Digit"
    bl_description = "Hide/show this digit's markers (both sides) to focus on the others"
    fname: bpy.props.StringProperty()
    side: bpy.props.StringProperty(default="L")
    part: bpy.props.StringProperty(default="hand")

    def execute(self, context):
        chainL = list_fingers(self.part, self.side).get(self.fname, [])
        new = not (chainL[0].hide_get() if chainL else False)
        for s in ("L", "R"):
            for o in list_fingers(self.part, s).get(self.fname, []):
                o.hide_set(new)
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_place_foot(bpy.types.Operator):
    bl_idname = "smartrig.place_foot"
    bl_label = "Place Foot (Top View)"
    bl_description = ("Switch to TOP view focused on the foot, then click the BALL of the "
                      "foot and the TOE TIP (2 clicks). Defines foot+toe bones; right mirrors.")
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        p = context.scene.smartrig
        self.mesh = p.target_mesh or context.active_object
        if self.mesh is None or self.mesh.type != 'MESH':
            self.report({'ERROR'}, "Select your character first.")
            return {'CANCELLED'}
        # stay inside the guided flow; just release the body-marker view lock
        try:
            markers.lock_front_view(context, False)
        except Exception:
            pass
        p.placing = False
        co = utils.read_world_coords(self.mesh)
        self._h = float(co[:, 2].max() - co[:, 2].min())
        self._ground = float(co[:, 2].min())
        self._pts = []
        markers.set_top_view_focus(context, "ankle.L")
        try:
            context.window.cursor_modal_set('CROSSHAIR')
        except Exception:
            pass
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "TOP view: click the BALL of the foot.")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'Z' and event.value == 'PRESS' and (event.ctrl or event.oskey):
            return {'RUNNING_MODAL'}
        if event.type == 'ESC' and event.value == 'PRESS':
            self._restore(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            region = context.region; rv3d = context.region_data
            if region is None or rv3d is None or region.type != 'WINDOW':
                return {'PASS_THROUGH'}
            coord = (event.mouse_region_x, event.mouse_region_y)
            o = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            d = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            # standard click ray -> works from ANY angle (click on the visible foot)
            hit = markers._volume_hit(self.mesh, o, d)
            if hit is None:
                self.report({'WARNING'}, "Click ON the foot.")
                return {'RUNNING_MODAL'}
            p = context.scene.smartrig
            if len(self._pts) == 0:
                markers.place_marker(p, "ball.L", hit, self._h)   # create NOW (visible)
                self._pts.append(hit)
                _push()
                for a in context.window.screen.areas:
                    a.tag_redraw()
                self.report({'INFO'}, "Ball placed. Now click the TOE TIP.")
                return {'RUNNING_MODAL'}
            markers.place_marker(p, "foottip.L", hit, self._h)    # create NOW
            _push()
            self.report({'INFO'}, "Foot markers placed. Build to use them.")
            self._restore(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}

    def _restore(self, context):
        try:
            context.window.cursor_modal_restore()
        except Exception:
            pass
        for a in context.window.screen.areas:
            a.tag_redraw()


def snap_fingers_to_palm(side):
    """Make each finger BASE coincide with its NEAREST palm END (their midpoint), so
    the finger connects to the palm like Rigify. Robust to marker order."""
    palms = [ch for _, ch in list_fingers("palm", side).items() if len(ch) >= 2]
    nonthumb = [ch for fn, ch in list_fingers("hand", side).items()
                if fn != "thumb" and len(ch) >= 1]
    used = set()
    for fch in nonthumb:
        base = fch[0]
        cand = [k for k in range(len(palms)) if k not in used]
        if not cand:
            break
        k = min(cand, key=lambda kk: (palms[kk][-1].location - base.location).length)
        used.add(k)
        mid = (palms[k][-1].location + base.location) * 0.5
        palms[k][-1].location = mid     # palm end
        base.location = mid             # finger base


class SMARTRIG_OT_snap_palm(bpy.types.Operator):
    bl_idname = "smartrig.snap_palm"
    bl_label = "Snap Fingers ↔ Palm"
    bl_description = ("Join each finger base to its palm end (one point), so they connect "
                      "like Rigify. Order: 1st finger↔palm1, 2nd↔palm2, ...")
    bl_options = {'REGISTER', 'UNDO'}
    side: bpy.props.StringProperty(default="L")

    def execute(self, context):
        snap_fingers_to_palm(self.side)
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_align_selected(bpy.types.Operator):
    bl_idname = "smartrig.align_selected"
    bl_label = "Align Selected (one level)"
    bl_description = ("Put all the SELECTED markers on one level (same height by default). "
                      "Change the axis in the redo panel (bottom-left).")
    bl_options = {'REGISTER', 'UNDO'}
    axis: bpy.props.EnumProperty(
        name="Axis", default='Z',
        items=[('Z', "Z", ""), ('X', "X", ""), ('Y', "Y", "")])
    orient: bpy.props.EnumProperty(
        name="Orientation", default='GLOBAL',
        items=[('GLOBAL', "World", "World X/Y/Z axes"),
               ('NORMAL', "Normal", "The finger's own direction (won't distort a tilted finger)"),
               ('BOX', "Box", "The selection's oriented bounding box")])

    def execute(self, context):
        import numpy as np
        from mathutils import Vector as _V
        sel = [o for o in context.selected_objects if o.type == 'EMPTY']
        if len(sel) < 2:
            self.report({'WARNING'}, "Select 2 or more markers first.")
            return {'CANCELLED'}
        P = np.array([[o.location.x, o.location.y, o.location.z] for o in sel], dtype=float)
        c = P.mean(0)
        ai = {'X': 0, 'Y': 1, 'Z': 2}[self.axis]
        if self.orient == 'GLOBAL' or len(sel) < 3:
            frame = np.eye(3)               # world axes
        else:
            cov = np.cov((P - c).T)         # NORMAL/BOX -> local (PCA) frame
            w, v = np.linalg.eigh(cov)
            order = w.argsort()[::-1]       # X=longest (finger length), Y/Z=perp
            frame = v[:, order].T
        a = frame[ai]
        a = a / (np.linalg.norm(a) or 1.0)
        cproj = float(c.dot(a))
        for o, p in zip(sel, P):
            d = float(p.dot(a)) - cproj     # flatten the selection along this axis
            q = p - d * a
            o.location = _V((float(q[0]), float(q[1]), float(q[2])))
        for ar in context.window.screen.areas:
            ar.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_toggle_skeleton(bpy.types.Operator):
    bl_idname = "smartrig.toggle_skeleton"
    bl_label = "Hide / Show Skeleton"
    bl_description = "Hide the skeleton so you can focus on placing markers"

    def execute(self, context):
        arm = bpy.data.objects.get(utils.REF_NAME)
        if arm is None:
            self.report({'INFO'}, "No skeleton yet.")
            return {'CANCELLED'}
        arm.hide_set(not arm.hide_get())
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


classes = (SMARTRIG_OT_finger_place, SMARTRIG_OT_finger_next, SMARTRIG_OT_finger_resume,
           SMARTRIG_OT_finger_finish, SMARTRIG_OT_finger_remove, SMARTRIG_OT_finger_select,
           SMARTRIG_OT_finger_straighten, SMARTRIG_OT_finger_align_bases,
           SMARTRIG_OT_finger_hide, SMARTRIG_OT_align_selected, SMARTRIG_OT_snap_palm,
           SMARTRIG_OT_place_foot, SMARTRIG_OT_toggle_skeleton)


# ---- keep finger markers INSIDE the mesh, even while the user drags them ----
from bpy.app.handlers import persistent

_CLAMPING = False


@persistent
def _clamp_inside(scene, depsgraph):
    """If a (left) finger/toe marker is dragged OUTSIDE the body, pull it back to the
    medial line of the nearest part so joints never end up outside the mesh."""
    global _CLAMPING
    if _CLAMPING:
        return
    p = getattr(scene, "smartrig", None)
    mesh = p.target_mesh if p else None
    if mesh is None or mesh.type != 'MESH':
        return
    mw = mesh.matrix_world
    mwi = mw.inverted()
    n3 = mw.to_3x3()
    changed = []
    for o in bpy.data.objects:
        if not o.name.startswith("fm.") or ".L." not in o.name:
            continue                                  # only left; right is constrained
        try:
            ok, loc, nrm, _idx = mesh.closest_point_on_mesh(mwi @ o.location)
        except Exception:
            continue
        if not ok:
            continue
        wl = mw @ loc
        wn = (n3 @ nrm).normalized()
        if (o.location - wl).dot(wn) > 1e-5:          # outside the surface
            hit = markers._volume_hit(mesh, o.location + wn * 0.001, -wn)
            changed.append((o, hit if hit is not None else (wl - wn * 0.003)))
    if changed:
        _CLAMPING = True
        for o, loc in changed:
            o.location = loc
        _CLAMPING = False


def register():
    for c in classes:
        bpy.utils.register_class(c)
    if _clamp_inside not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_clamp_inside)


def unregister():
    if _clamp_inside in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_clamp_inside)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
