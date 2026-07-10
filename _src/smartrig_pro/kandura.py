"""Kandura (Emirati thobe) module — MANUAL placement workflow (v1.37).

No auto-detection, no edge-loop registration. The user presses one of the
three "Add Bones" buttons and gets a rough starter layout to place by hand:

  - Add Waist-Down Bones : skirt-engine grid (Columns x Rows) around the
                           lower half of the garment -> skirt.* bones
                           (inherits ALL skirt automations downstream).
  - Add Sleeve Bones     : one chain per arm along the body arm bones
                           (Upper/Lower counts)      -> kan_sleeve.*
  - Add Collar Bones     : a ring of N bones around the neck -> kan_collar.*
  - Add Cuff Bones       : a ring of N bones around each sleeve END
                           (wrist opening)             -> kan_cuff.{L,R}.*

Each button drops the bones into the METARIG and opens it in Edit Mode with
the new bones selected, ready for manual placement.

Align to Surface:
  - Toggle  : turns Blender FACE snapping on/off, so dragged bone points
              stick to the garment surface while moving.
  - Align Selected Now: projects the selected bone heads/tails onto the
              nearest point of the kandura mesh in one click.
"""
import bpy
from mathutils import Vector

BONE_SLEEVE = "kan_sleeve"
BONE_COLLAR = "kan_collar"
BONE_CUFF = "kan_cuff"


def kandura_object(context):
    """The kandura mesh: the explicit picker if set, else the mesh being
    edited, else the active mesh object."""
    props = context.scene.smartrig
    ob = getattr(props, "kandura_object", None)
    if ob is not None and ob.type == 'MESH':
        return ob
    eo = context.edit_object
    if eo is not None and eo.type == 'MESH':
        return eo
    ao = context.active_object
    if ao is not None and ao.type == 'MESH':
        return ao
    return None


def _metarig():
    return bpy.data.objects.get("SR_Metarig")


def _bone_seg(arm_ob, names):
    """World-space (head, tail) of the first existing bone in `names`."""
    for n in names:
        b = arm_ob.data.bones.get(n)
        if b is not None:
            mw = arm_ob.matrix_world
            return (mw @ b.head_local, mw @ b.tail_local)
    return None


def _garment_coords(ob):
    """World-space REST coords of the garment (never the deformed pose)."""
    from . import utils as _ut
    rest = _ut.read_rest_coords(ob)          # ALREADY world coords
    return [Vector(p) for p in rest]


def focus_apply(context, enabled):
    """Hide/show the BODY bones of the metarig so only the kandura bones
    stay visible - lets the user concentrate on garment bone placement.
    Kandura bones = kan_* (+ skirt.* when the waist grid is the kandura's)."""
    mo = _metarig()
    if mo is None:
        return
    kan_skirt = (mo.get("sr_skirt_method") == "kandura")

    def is_kan(nm):
        return nm.startswith("kan_") or (kan_skirt and nm.startswith("skirt."))

    if mo.mode == 'EDIT':
        for b in mo.data.edit_bones:
            if not is_kan(b.name):
                b.hide = enabled
    for b in mo.data.bones:
        if not is_kan(b.name):
            b.hide = enabled
    if bpy.app.version >= (5, 0, 0):   # 5.x: Pose Mode draws PoseBone.hide
        for pb in mo.pose.bones:
            if not is_kan(pb.name):
                pb.hide = enabled


def _enter_metarig_edit(context, select_names=None):
    """Open the metarig in Edit Mode; select only `select_names` if given."""
    mo = _metarig()
    if mo is None:
        return False
    try:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    try:
        mo.hide_set(False)
    except Exception:
        pass
    mo.hide_viewport = False
    bpy.ops.object.select_all(action='DESELECT')
    mo.select_set(True)
    context.view_layer.objects.active = mo
    bpy.ops.object.mode_set(mode='EDIT')
    if select_names:
        want = set(select_names)
        for b in mo.data.edit_bones:
            sel = b.name in want
            b.select = sel
            b.select_head = sel
            b.select_tail = sel
    try:
        focus_apply(context, context.scene.smartrig.kandura_focus)
    except Exception:
        pass
    return True


# ====================================================================
# MIRROR (X-axis) — on/off + one-shot geometric mirror
# ====================================================================

def mirror_apply(context, enabled):
    """Turn armature X-Axis Mirror on/off (live mirroring for .L/.R names
    like kan_sleeve while dragging)."""
    mo = _metarig()
    if mo is not None:
        try:
            mo.data.use_mirror_x = enabled
        except Exception:
            pass
    eo = context.edit_object
    if eo is not None and eo.type == 'ARMATURE' and eo is not mo:
        try:
            eo.data.use_mirror_x = enabled
        except Exception:
            pass


_MIRROR_FAMILIES = ("skirt.", BONE_SLEEVE + ".", BONE_COLLAR + ".",
                    BONE_CUFF + ".")


class SMARTRIG_OT_kandura_mirror_now(bpy.types.Operator):
    bl_idname = "smartrig.kandura_mirror_now"
    bl_label = "Mirror Selected to Other Side"
    bl_description = ("Copy the SELECTED kandura bones (waist grid / sleeves / "
                      "collar) onto their nearest counterparts on the other "
                      "side of the X axis (one-shot geometric mirror - works "
                      "even for bones without .L/.R names)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_ARMATURE'

    def execute(self, context):
        arm = context.edit_object
        eb = arm.data.edit_bones

        def family(name):
            for f in _MIRROR_FAMILIES:
                if name.startswith(f):
                    return f
            return None

        sel = [b for b in eb
               if b.select and not b.hide and family(b.name)]
        if not sel:
            self.report({'WARNING'},
                        "Select kandura bones (skirt / sleeve / collar) first")
            return {'CANCELLED'}
        sel_names = set(b.name for b in sel)
        n = 0
        for b in sel:
            fam = family(b.name)
            mh = Vector((-b.head.x, b.head.y, b.head.z))
            mt = Vector((-b.tail.x, b.tail.y, b.tail.z))
            # skip bones sitting on the centerline - nothing to mirror onto
            if abs(b.head.x) < 1e-5 and abs(b.tail.x) < 1e-5:
                continue
            best, best_d = None, 1e18
            for c in eb:
                if (c is b or c.hide or c.name in sel_names
                        or not c.name.startswith(fam)):
                    continue
                d = (c.head - mh).length_squared + (c.tail - mt).length_squared
                if d < best_d:
                    best, best_d = c, d
            if best is None:
                continue
            best.head = mh
            best.tail = mt
            best.roll = -b.roll
            n += 1
        if n == 0:
            self.report({'WARNING'}, "No counterpart bones found")
            return {'CANCELLED'}
        self.report({'INFO'}, "Mirrored %d bones to the other side" % n)
        return {'FINISHED'}


# ====================================================================
# ALIGN TO SURFACE
# ====================================================================

def align_snap_apply(context, enabled):
    """Turn FACE snapping on/off so dragged bone points stick to surfaces.

    SCOPED TO BONE PLACEMENT: Blender's snap is a GLOBAL tool setting, so
    leaving it on leaked into Pose Mode - grabbing an IK control snapped it
    to the garment surface and the limb "jumped high" (Saeed's bug). Snap is
    now only switched on inside EDIT_ARMATURE; a watcher shuts it off the
    moment the user leaves Edit Mode."""
    ts = context.scene.tool_settings
    if enabled:
        if getattr(context, "mode", "") != 'EDIT_ARMATURE':
            return          # re-applied by _enter_metarig_edit on entry
        context.window_manager["sr_kan_snap"] = 1
        ts.use_snap = True
        try:
            ts.snap_elements = {'FACE'}
        except Exception:
            pass
        for attr, val in (("snap_target", 'CLOSEST'),
                          ("use_snap_align_rotation", False),
                          ("use_snap_translate", True),
                          ("use_snap_rotate", False),
                          ("use_snap_scale", False),
                          ("use_snap_self", False)):
            if hasattr(ts, attr):
                try:
                    setattr(ts, attr, val)
                except Exception:
                    pass
    else:
        ts.use_snap = False
        wm = getattr(context, "window_manager", None)
        if wm is not None and "sr_kan_snap" in wm:
            del wm["sr_kan_snap"]


def _kan_snap_watch(scene, depsgraph=None):
    """Kill the placement snap as soon as the user leaves EDIT_ARMATURE, so
    Pose/Object mode manipulation is never hijacked by FACE snapping."""
    try:
        wm = bpy.context.window_manager
        if not wm.get("sr_kan_snap"):
            return
        if bpy.context.mode != 'EDIT_ARMATURE':
            scene.tool_settings.use_snap = False
            del wm["sr_kan_snap"]
    except Exception:
        pass


class SMARTRIG_OT_kandura_align_now(bpy.types.Operator):
    bl_idname = "smartrig.kandura_align_now"
    bl_label = "Align Selected to Surface"
    bl_description = ("Project the SELECTED bone heads/tails onto the nearest "
                      "point of the kandura mesh surface (one-shot)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_ARMATURE'
                and kandura_object(context) is not None)

    def execute(self, context):
        ob = kandura_object(context)
        arm = context.edit_object
        mw = ob.matrix_world
        mwi = mw.inverted()
        amw = arm.matrix_world
        amwi = amw.inverted()

        def project(world_co):
            res = ob.closest_point_on_mesh(mwi @ world_co)
            # Blender returns (result, location, normal, index)
            if res[0]:
                return mw @ res[1]
            return world_co

        n = 0
        for b in arm.data.edit_bones:
            if b.hide:
                continue
            if b.select_head or b.select:
                b.head = amwi @ project(amw @ b.head)
                n += 1
            if b.select_tail or b.select:
                b.tail = amwi @ project(amw @ b.tail)
                n += 1
        if n == 0:
            self.report({'WARNING'}, "Select bone heads/tails first")
            return {'CANCELLED'}
        self.report({'INFO'}, "Aligned %d bone points to the surface" % n)
        return {'FINISHED'}


# ====================================================================
# RESAMPLING — changing Columns/Rows PRESERVES the manual placement.
# The new grid is re-sampled from the CURRENT bone positions (index-
# parameter lerp: identical counts reproduce the exact same bones).
# ====================================================================

def _resample_open_arc(pts, n_new):
    """PROFESSIONAL open-polyline resample: n_new+1 points spaced EVENLY by
    arc length ALONG the user's placed shape. Endpoints stay exact."""
    import bisect
    if len(pts) < 2 or n_new < 1:
        return None
    cum = [0.0]
    for i in range(len(pts) - 1):
        cum.append(cum[-1] + (pts[i + 1] - pts[i]).length)
    total = cum[-1]
    if total < 1e-9:
        return None
    out = []
    for k in range(n_new + 1):
        s = total * k / float(n_new)
        i = min(bisect.bisect_right(cum, s) - 1, len(pts) - 2)
        seg = cum[i + 1] - cum[i]
        f = 0.0 if seg < 1e-12 else (s - cum[i]) / seg
        out.append(pts[i].lerp(pts[i + 1], f))
    out[0] = pts[0].copy()
    out[-1] = pts[-1].copy()
    return out


def _resample_ring_arc(pts, n_new):
    """PROFESSIONAL closed-ring resample: n_new points spaced EVENLY by arc
    length around the user's placed ring. Point 0 stays exact (no twist)."""
    import bisect
    if len(pts) < 3 or n_new < 3:
        return None
    P = list(pts) + [pts[0]]
    cum = [0.0]
    for i in range(len(P) - 1):
        cum.append(cum[-1] + (P[i + 1] - P[i]).length)
    total = cum[-1]
    if total < 1e-9:
        return None
    out = []
    for k in range(n_new):
        s = total * k / float(n_new)
        i = min(bisect.bisect_right(cum, s) - 1, len(P) - 2)
        seg = cum[i + 1] - cum[i]
        f = 0.0 if seg < 1e-12 else (s - cum[i]) / seg
        out.append(P[i].lerp(P[i + 1], f))
    out[0] = pts[0].copy()
    return out


def _regrid_columns(old, cols, rpz, knee_z=None):
    """Rebuild the waist grid FOLLOWING the user's placed shape (subdivide-
    style): even arc-length spacing, waist/hem/knee anchors exact.
    `rpz` = rows PER ZONE: the KNEE ring is always a boundary — rpz rows
    cover the THIGH zone (waist->knee) and rpz rows the SHIN zone
    (knee->hem). Same counts -> the exact same grid (nothing moves).
    Returns columns list or None (caller falls back to a fresh build)."""
    M = len(old)
    Rt = len(old[0]) - 1                       # total rows in the old grid
    rows_total = 2 * rpz
    # locate the knee ring in the OLD grid (closest ring to the knee z)
    k = None
    if Rt >= 2:
        if knee_z is not None:
            zs = [sum(col[r].z for col in old) / M for r in range(Rt + 1)]
            k = min(range(1, Rt), key=lambda r: abs(zs[r] - knee_z))
        else:
            k = Rt // 2
    if M == cols and Rt == rows_total and k == rpz:
        return old                              # untouched
    # rows: resample each zone separately so the knee ring is PRESERVED
    if Rt == rows_total and k == rpz:
        cols2 = old
    elif k is not None:
        cols2 = []
        for c in old:
            up = _resample_open_arc(c[:k + 1], rpz)
            dn = _resample_open_arc(c[k:], rpz)
            if up is None or dn is None:
                return None
            cols2.append(up + dn[1:])
    else:                                       # no knee ring info
        cols2 = [_resample_open_arc(c, rows_total) for c in old]
        if any(c is None for c in cols2):
            return None
    # columns: resample every ring evenly (column-0 anchor kept)
    if M != cols:
        rings = []
        for r in range(rows_total + 1):
            ring = _resample_ring_arc([col[r] for col in cols2], cols)
            if ring is None:
                return None
            rings.append(ring)
        cols2 = [[rings[r][j] for r in range(rows_total + 1)]
                 for j in range(cols)]
    return cols2


def _read_bone_points(mo):
    """{name: (head, tail)} in armature space, ALWAYS current.
    In Edit Mode reads the LIVE edit bones (data.bones is stale until the
    mode is left - relying on a mode flush proved unreliable from operators)."""
    out = {}
    if mo.mode == 'EDIT':
        for b in mo.data.edit_bones:
            out[b.name] = (Vector(b.head), Vector(b.tail))
    else:
        for b in mo.data.bones:
            out[b.name] = (Vector(b.head_local), Vector(b.tail_local))
    return out


def _existing_waist_grid(mo):
    """The CURRENT (manually placed) skirt grid as columns of points,
    or None if there is no complete kandura skirt grid."""
    import re
    pat = re.compile(r"^skirt\.(\d+)\.(\d+)$")
    pts_map = _read_bone_points(mo)
    data = {}
    for name, ht in pts_map.items():
        m = pat.match(name)
        if m:
            data.setdefault(int(m.group(1)), {})[int(m.group(2))] = ht
    if not data:
        return None
    rows_old = max(max(rs) for rs in data.values()) + 1
    cols_sorted = sorted(data)
    grid = []
    for c in cols_sorted:
        rs = data[c]
        if len(rs) != rows_old or set(rs) != set(range(rows_old)):
            return None                      # incomplete -> fresh build
        pts = [rs[r][0] for r in range(rows_old)]
        pts.append(rs[rows_old - 1][1])
        grid.append(pts)
    return grid if len(grid) >= 3 else None


def _existing_chain(mo, prefix):
    """CURRENT chain bones '<prefix>.NN' as an open point list, or None."""
    import re
    pat = re.compile(r"^%s\.(\d+)$" % re.escape(prefix))
    pts_map = _read_bone_points(mo)
    data = {}
    for name, ht in pts_map.items():
        m = pat.match(name)
        if m:
            data[int(m.group(1))] = ht
    if not data or set(data) != set(range(len(data))):
        return None
    pts = [data[k][0] for k in range(len(data))]
    pts.append(data[len(data) - 1][1])
    return pts


# ====================================================================
# ADD WAIST-DOWN BONES — rough grid, then manual placement
# ====================================================================

def _pt_seg_d(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    L2 = ab.length_squared
    if L2 < 1e-12:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / L2))
    return (p - (a + ab * t)).length


def _drop_sleeve_verts(mo, cos):
    """For the INITIAL rough layout only: drop garment verts that belong to
    the sleeves (closer to the arm bones than to the spine/leg bones), so
    the waist-down rings hug the tube of the thobe instead of flaring out
    to the cuffs. Returns cos unchanged if the body bones are missing."""
    def segs(names):
        out = []
        for nm in names:
            s = _bone_seg(mo, [nm])
            if s is not None:
                out.append(s)
        return out

    arm_segs = segs(["upper_arm.L", "forearm.L", "hand.L",
                     "upper_arm.R", "forearm.R", "hand.R"])
    body_segs = segs(["spine", "spine.001", "spine.002", "spine.003",
                      "thigh.L", "thigh.R", "shin.L", "shin.R"])
    if not arm_segs or not body_segs:
        return cos

    def dmin(p, ss):
        return min(_pt_seg_d(p, a, b) for a, b in ss)

    kept = [p for p in cos if dmin(p, body_segs) <= dmin(p, arm_segs)]
    return kept if len(kept) >= 24 else cos


def _waist_z(cos, gz0, gh):
    """Rough waist height. NEVER above the spine bone: the waist-down grid
    must stay below the hips (spine head), whatever the garment shape."""
    mo = _metarig()
    zs = []
    if mo is not None:
        seg = _bone_seg(mo, ["thigh.L", "thigh.R"])
        if seg is not None:
            zs.append(seg[0].z)
        sp = _bone_seg(mo, ["spine"])
        if sp is not None:
            zs.append(sp[0].z)              # hips - the hard ceiling
    if zs:
        return min(zs)
    return gz0 + 0.55 * gh


def _ring_ellipse(cos, z, band, fallback):
    """Rough ellipse (cx, cy, rx, ry) of the garment verts near height z.
    Falls back to the whole lower bbox if the band is too thin."""
    pts = [p for p in cos if abs(p.z - z) < band]
    if len(pts) < 8:
        pts = fallback
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    rx = max(0.02, (max(xs) - min(xs)) * 0.5)
    ry = max(0.02, (max(ys) - min(ys)) * 0.5)
    return cx, cy, rx, ry


def _flush_edit_mode():
    """Leave Edit Mode so data.bones reflects the user's LATEST manual
    placement. Reading bones while still in Edit Mode returns STALE data
    (edits only flush on mode exit) - that made re-Add 'lose' the layout."""
    try:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass


def add_waist_bones(context):
    """Emit a Columns x Rows skirt grid around the lower half of the garment
    (rough bounding-ellipse placement — the user refines it manually).
    Reuses the skirt engine's _emit_chains so ALL skirt automations
    (collision, jiggle, weights, follow) keep working. Returns (ok, msg)."""
    import math
    from . import skirt as _sk
    props = context.scene.smartrig
    ob = kandura_object(context)
    if ob is None:
        return False, "No kandura mesh"
    mo = _metarig()
    if mo is None:
        return False, "Build the body metarig first"
    _flush_edit_mode()          # read the LATEST manual placement

    cols = max(4, int(props.kandura_columns))
    rpz = max(1, int(props.kandura_rows))       # rows PER ZONE (thigh/shin)
    rows = 2 * rpz                              # total: rpz above + rpz below
    knee_z = None
    shin = _bone_seg(mo, ["shin.L", "shin.R"])
    if shin is not None:
        knee_z = shin[0].z
    spine = _bone_seg(mo, ["spine"])

    # ---- PRESERVE MANUAL PLACEMENT: rebuild from the CURRENT bones ----
    old = None
    if mo.get("sr_skirt_method") == "kandura":
        old = _existing_waist_grid(mo)
    if old is not None and spine is not None:
        # sanity: a waist ring ABOVE the spine = broken/stray grid (e.g.
        # parked at the chest/neck) -> rebuild fresh instead of keeping it
        top = max(col[0].z for col in old)
        if top > spine[0].z + 0.02:
            old = None
    if old is not None:
        new_cols = _regrid_columns(old, cols, rpz, knee_z)
        if new_cols is not None:
            grid = [(c, pts) for c, pts in enumerate(new_cols)]
        else:
            old = None
    if old is None:
        # ---- first build: rough grid around the lower half ----
        cos = _garment_coords(ob)
        if len(cos) < 24:
            return False, "Kandura mesh has too few vertices"
        zs = sorted(p.z for p in cos)
        gz0, gz1 = zs[0], zs[-1]
        gh = max(1e-6, gz1 - gz0)
        wz = _waist_z(cos, gz0, gh)
        # sleeves would inflate the rings sideways - drop them for the layout
        cos = _drop_sleeve_verts(mo, cos)
        lower = [p for p in cos if p.z <= wz + 1e-4]
        if len(lower) < 24:
            return False, "Too few vertices below the waist"
        hem_z = min(p.z for p in lower)
        front = _sk._FRONT_ANG.get(getattr(props, "skirt_front_axis", '-Y'),
                                   _sk._FRONT_ANG['-Y'])
        band = max(0.01, (wz - hem_z) / max(2 * rows, 4))
        grid = []
        # PROFESSIONAL ROW SPLIT for the thigh/shin automation: the KNEE
        # ring is always a boundary (like the elbow rule for sleeves) -
        # rpz rows over the THIGH zone + rpz rows over the SHIN zone.
        if (knee_z is not None
                and hem_z < knee_z - 0.02 and knee_z < wz - 0.02):
            ring_zs = [wz + (knee_z - wz) * r / rpz for r in range(rpz)]
            ring_zs += [knee_z + (hem_z - knee_z) * r / rpz
                        for r in range(rpz + 1)]
        else:
            ring_zs = [wz + (hem_z - wz) * r / rows for r in range(rows + 1)]
        rings = [_ring_ellipse(cos, z, band, lower) for z in ring_zs]
        for c in range(cols):
            ang = front + 2.0 * math.pi * c / cols
            ca, sa = math.cos(ang), math.sin(ang)
            pts = []
            for (cx, cy, rx, ry), z in zip(rings, ring_zs):
                pts.append(Vector((cx + rx * ca, cy + ry * sa, z)))
            grid.append((c, pts))

    _sk._emit_chains(mo, grid, rows)
    mo["sr_skirt_kind"] = "TUBE"
    mo["sr_skirt_method"] = "kandura"
    mo["sr_skirt_cols_built"] = cols
    mo["sr_kandura"] = True
    # downstream skirt tools (collision / jiggle / weights) need these:
    props.skirt_source = 'SEPARATE'
    props.skirt_object = ob
    names = ["%s.%02d.%02d" % (_sk.PREFIX, c, r)
             for c in range(cols) for r in range(rows)]
    return True, names


def _live_rebuild(context, builder, prefix):
    """REAL-TIME rebuild when a count property changes (subdivide-style):
    only if the bones already exist; keeps Edit Mode if the user is in it."""
    mo = _metarig()
    if mo is None:
        return
    names = _read_bone_points(mo)
    if not any(n.startswith(prefix) for n in names):
        return
    was_edit = (mo.mode == 'EDIT')
    try:
        ok, res = builder(context)
    except Exception as e:
        print("SmartRig kandura live rebuild:", e)
        return
    if ok and was_edit:
        try:
            _enter_metarig_edit(context, select_names=res)
        except Exception:
            pass


def live_rebuild_waist(context):
    mo = _metarig()
    if mo is not None and mo.get("sr_skirt_method") == "kandura":
        _live_rebuild(context, add_waist_bones, "skirt.")


def live_rebuild_sleeves(context):
    _live_rebuild(context, add_sleeve_bones, BONE_SLEEVE + ".")


def live_rebuild_collar(context):
    _live_rebuild(context, add_collar_bones, BONE_COLLAR + ".")


def live_rebuild_cuffs(context):
    _live_rebuild(context, add_cuff_bones, BONE_CUFF + ".")


class SMARTRIG_OT_kandura_add_waist(bpy.types.Operator):
    bl_idname = "smartrig.kandura_add_waist"
    bl_label = "Add Waist-Down Bones"
    bl_description = ("Add a Columns x Rows bone grid roughly around the lower "
                      "half of the kandura, then open the metarig in Edit Mode "
                      "so you place the bones MANUALLY (use Align to Surface)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (kandura_object(context) is not None
                and _metarig() is not None)

    def execute(self, context):
        ok, res = add_waist_bones(context)
        if not ok:
            self.report({'ERROR'}, res)
            return {'CANCELLED'}
        _enter_metarig_edit(context, select_names=res)
        align_snap_apply(context, context.scene.smartrig.kandura_align_surface)
        mirror_apply(context, context.scene.smartrig.kandura_mirror)
        self.report({'INFO'},
                    "Waist-down grid added (%d bones) - place them manually"
                    % len(res))
        return {'FINISHED'}


# ====================================================================
# ADD SLEEVE BONES — chains along the body arms, then manual placement
# ====================================================================

def _resample_sleeve_elbow(mo, pts, side, n_up, n_lo):
    """Resample a placed sleeve chain keeping the ELBOW as a HARD boundary
    (the knee-ring rule, arm edition): n_up bones on the upper-arm part,
    n_lo bones on the forearm part. The split point = the point of the
    PLACED polyline closest to the body elbow joint, so the user's manual
    placement is preserved on both sides of the elbow."""
    fa = _read_bone_points(mo).get("forearm." + side)
    if fa is None or len(pts) < 2:
        return _resample_open_arc(pts, n_up + n_lo)
    elbow = fa[0]
    best = (1e18, 0, 0.0)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        L2 = ab.length_squared
        t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, (elbow - a).dot(ab) / L2))
        d = (a + ab * t - elbow).length
        if d < best[0]:
            best = (d, i, t)
    _, i, t = best
    split = pts[i].lerp(pts[i + 1], t)
    up_pts = pts[:i + 1] + [split]
    lo_pts = [split] + pts[i + 1:]
    alen = lambda ps: sum((ps[j + 1] - ps[j]).length for j in range(len(ps) - 1))
    if alen(up_pts) < 1e-4 or alen(lo_pts) < 1e-4:
        return _resample_open_arc(pts, n_up + n_lo)
    up = _resample_open_arc(up_pts, n_up)
    lo = _resample_open_arc(lo_pts, n_lo)
    if up is None or lo is None:
        return _resample_open_arc(pts, n_up + n_lo)
    return up + lo[1:]


def add_sleeve_bones(context):
    """One kan_sleeve chain per arm, laid along the body arm bones as a
    rough start (the user drags them onto the sleeve). Returns (ok, msg)."""
    props = context.scene.smartrig
    mo = _metarig()
    if mo is None:
        return False, "Build the body metarig first"
    n_up = max(1, int(props.kandura_sleeve_upper))
    n_lo = max(1, int(props.kandura_sleeve_lower))
    _flush_edit_mode()          # read the LATEST manual placement

    chains = {}
    for side in ("L", "R"):
        # PRESERVE MANUAL PLACEMENT: resample the CURRENT chain if it exists
        cur = _existing_chain(mo, "%s.%s" % (BONE_SLEEVE, side))
        if cur is not None:
            same = (len(cur) - 1 == n_up + n_lo
                    and int(mo.get("sr_sleeve_up", -1)) == n_up
                    and int(mo.get("sr_sleeve_lo", -1)) == n_lo)
            if same:
                chains[side] = cur              # same counts: untouched
                continue
            # ELBOW = hard boundary: resample each segment separately
            res = _resample_sleeve_elbow(mo, cur, side, n_up, n_lo)
            if res is not None:
                chains[side] = res
                continue
        up = _bone_seg(mo, ["upper_arm." + side])
        lo = _bone_seg(mo, ["forearm." + side])
        if up is None or lo is None:
            continue
        pts = [up[0] + (up[1] - up[0]) * k / n_up for k in range(n_up)]
        pts += [lo[0] + (lo[1] - lo[0]) * k / n_lo for k in range(n_lo + 1)]
        chains[side] = pts
    if not chains:
        return False, "No arm bones (upper_arm/forearm) on the metarig"

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        mo.hide_set(False)
    except Exception:
        pass
    mo.hide_viewport = False
    bpy.context.view_layer.objects.active = mo
    mo.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = mo.data.edit_bones
    for b in [b for b in eb if b.name.startswith(BONE_SLEEVE + ".")]:
        eb.remove(b)
    made, roots = [], []
    for side, pts in chains.items():
        parent = eb.get("upper_arm." + side)
        prev = None
        for r in range(len(pts) - 1):
            name = "%s.%s.%02d" % (BONE_SLEEVE, side, r)
            b = eb.new(name)
            b.head = pts[r]
            b.tail = pts[r + 1]
            if prev is None:
                if parent is not None:
                    b.parent = parent
                    b.use_connect = False
                roots.append(name)
            else:
                b.parent = prev
                b.use_connect = True
            prev = b
            made.append(name)
    bpy.ops.object.mode_set(mode='OBJECT')
    for name in roots:
        pb = mo.pose.bones.get(name)
        if pb is not None:
            try:
                pb.rigify_type = "limbs.simple_tentacle"
            except Exception:
                pass
    mo["sr_sleeve_up"] = n_up
    mo["sr_sleeve_lo"] = n_lo
    mo["sr_kandura"] = True
    return True, made


class SMARTRIG_OT_kandura_add_sleeves(bpy.types.Operator):
    bl_idname = "smartrig.kandura_add_sleeves"
    bl_label = "Add Sleeve Bones"
    bl_description = ("Add one sleeve chain per arm (Upper + Lower bone "
                      "counts) along the body arms, then open the metarig in "
                      "Edit Mode so you place the bones MANUALLY")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _metarig() is not None

    def execute(self, context):
        ok, res = add_sleeve_bones(context)
        if not ok:
            self.report({'ERROR'}, res)
            return {'CANCELLED'}
        _enter_metarig_edit(context, select_names=res)
        align_snap_apply(context, context.scene.smartrig.kandura_align_surface)
        mirror_apply(context, context.scene.smartrig.kandura_mirror)
        self.report({'INFO'},
                    "Sleeve chains added (%d bones) - place them manually"
                    % len(res))
        return {'FINISHED'}


# ====================================================================
# ADD COLLAR BONES — ring around the neck, then manual placement
# ====================================================================

def add_collar_bones(context):
    """A ring of N kan_collar anchor bones around the neck as a rough start.
    Returns (ok, msg)."""
    import math
    props = context.scene.smartrig
    mo = _metarig()
    if mo is None:
        return False, "Build the body metarig first"
    n = max(3, int(props.kandura_collar_count))
    _flush_edit_mode()          # read the LATEST manual placement

    neck = _bone_seg(mo, ["spine.004", "neck", "spine.005"])
    if neck is None:
        return False, "No neck bone (spine.004/neck) on the metarig"
    base = neck[0]

    # PRESERVE MANUAL PLACEMENT: resample the CURRENT ring if it exists
    import re
    pat = re.compile(r"^%s\.(\d+)$" % re.escape(BONE_COLLAR))
    cur = {}
    for name, ht in _read_bone_points(mo).items():
        m = pat.match(name)
        if m:
            cur[int(m.group(1))] = ht
    ring = None
    if cur and set(cur) == set(range(len(cur))) and len(cur) >= 3:
        pairs = [[cur[k][0], cur[k][1]] for k in range(len(cur))]
        if len(pairs) == n:
            ring = pairs                        # same count: untouched
        else:
            heads = _resample_ring_arc([p[0] for p in pairs], n)
            tails = _resample_ring_arc([p[1] for p in pairs], n)
            if heads is not None and tails is not None:
                ring = [[h, t] for h, t in zip(heads, tails)]

    # rough ring size from the garment verts near the neck base (bbox only)
    ob = kandura_object(context)
    cx, cy, rx, ry = base.x, base.y, 0.07, 0.07
    bone_len = 0.05
    if ring is None and ob is not None:
        cos = _garment_coords(ob)
        zs = sorted(p.z for p in cos)
        gh = max(1e-6, zs[-1] - zs[0])
        bone_len = max(0.02, 0.03 * gh)
        band = [p for p in cos if abs(p.z - base.z) < 0.06 * gh]
        if len(band) >= 8:
            cx, cy, rx, ry = _ring_ellipse(band, base.z, 1e9, band)

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        mo.hide_set(False)
    except Exception:
        pass
    mo.hide_viewport = False
    bpy.context.view_layer.objects.active = mo
    mo.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = mo.data.edit_bones
    for b in [b for b in eb if b.name.startswith(BONE_COLLAR + ".")]:
        eb.remove(b)
    parent = eb.get("spine.004") or eb.get("neck") or eb.get("spine.003")
    made = []
    for k in range(n):
        if ring is not None:
            head, tail = ring[k][0], ring[k][1]
        else:
            ang = -0.5 * math.pi + 2.0 * math.pi * k / n  # start at the front
            head = Vector((cx + rx * math.cos(ang),
                           cy + ry * math.sin(ang), base.z))
            tail = head + Vector((0.0, 0.0, bone_len))
        name = "%s.%02d" % (BONE_COLLAR, k)
        b = eb.new(name)
        b.head = head
        b.tail = tail
        if parent is not None:
            b.parent = parent
            b.use_connect = False
        made.append(name)
    bpy.ops.object.mode_set(mode='OBJECT')
    for name in made:
        pb = mo.pose.bones.get(name)
        if pb is not None:
            try:
                pb.rigify_type = "basic.super_copy"
                pb.rigify_parameters.make_deform = True
            except Exception:
                pass
    mo["sr_kandura"] = True
    return True, made


class SMARTRIG_OT_kandura_add_collar(bpy.types.Operator):
    bl_idname = "smartrig.kandura_add_collar"
    bl_label = "Add Collar Bones"
    bl_description = ("Add a ring of collar bones around the neck (count is "
                      "adjustable), then open the metarig in Edit Mode so you "
                      "place the bones MANUALLY")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _metarig() is not None

    def execute(self, context):
        ok, res = add_collar_bones(context)
        if not ok:
            self.report({'ERROR'}, res)
            return {'CANCELLED'}
        _enter_metarig_edit(context, select_names=res)
        align_snap_apply(context, context.scene.smartrig.kandura_align_surface)
        mirror_apply(context, context.scene.smartrig.kandura_mirror)
        self.report({'INFO'},
                    "Collar ring added (%d bones) - place them manually"
                    % len(res))
        return {'FINISHED'}


# ====================================================================
# ADD CUFF BONES — ring around each sleeve END (wrist opening)
# ====================================================================

def add_cuff_bones(context):
    """A ring of N kan_cuff bones around each sleeve END (wrist opening).
    The ring centre + axis come from the END of the PLACED kan_sleeve chain
    (so add + place the sleeves first); without a sleeve chain the body
    forearm tail is used. The radius is measured from the garment verts
    around the wrist. Parented to hand.L/R so it follows the hand.
    Returns (ok, msg)."""
    import math
    import re
    props = context.scene.smartrig
    mo = _metarig()
    if mo is None:
        return False, "Build the body metarig first"
    n = max(3, int(props.kandura_cuff_count))
    _flush_edit_mode()          # read the LATEST manual placement

    ob = kandura_object(context)
    cos = _garment_coords(ob) if ob is not None else []
    gh = 1.0
    if cos:
        zs = sorted(p.z for p in cos)
        gh = max(1e-6, zs[-1] - zs[0])

    pts = _read_bone_points(mo)
    rings = {}
    for side in ("L", "R"):
        # PRESERVE MANUAL PLACEMENT: resample the CURRENT ring if it exists
        pat = re.compile(r"^%s\.%s\.(\d+)$" % (re.escape(BONE_CUFF), side))
        cur = {}
        for name, ht in pts.items():
            m = pat.match(name)
            if m:
                cur[int(m.group(1))] = ht
        if cur and set(cur) == set(range(len(cur))) and len(cur) >= 3:
            pairs = [[cur[k][0], cur[k][1]] for k in range(len(cur))]
            if len(pairs) == n:
                rings[side] = pairs             # same count: untouched
                continue
            heads = _resample_ring_arc([p[0] for p in pairs], n)
            tails = _resample_ring_arc([p[1] for p in pairs], n)
            if heads is not None and tails is not None:
                rings[side] = [[h, t] for h, t in zip(heads, tails)]
                continue
        # centre + axis: END of the placed sleeve chain, else the forearm
        chain = _existing_chain(mo, "%s.%s" % (BONE_SLEEVE, side))
        if chain is not None and len(chain) >= 2:
            centre = chain[-1].copy()
            axis = chain[-1] - chain[-2]
        else:
            fa = _bone_seg(mo, ["forearm." + side])
            if fa is None:
                continue
            centre = fa[1].copy()
            axis = fa[1] - fa[0]
        if axis.length < 1e-9:
            axis = Vector((1.0 if side == "L" else -1.0, 0.0, 0.0))
        axis.normalize()
        # ring size from the garment verts around the wrist opening
        r0 = 0.03 * gh
        if cos:
            band = [p for p in cos
                    if (p - centre).length < 0.10 * gh
                    and abs((p - centre).dot(axis)) < 0.03 * gh]
            if len(band) >= 8:
                cen = Vector((0.0, 0.0, 0.0))
                for p in band:
                    cen += p
                cen /= len(band)
                centre = cen
                r0 = sum(((p - cen) - axis * ((p - cen).dot(axis))).length
                         for p in band) / len(band)
        up = Vector((0.0, 0.0, 1.0))
        if abs(axis.dot(up)) > 0.9:
            up = Vector((0.0, 1.0, 0.0))
        u = axis.cross(up).normalized()
        v = axis.cross(u).normalized()
        blen = max(0.02, 0.03 * gh)
        ring = []
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            head = centre + u * (r0 * math.cos(ang)) + v * (r0 * math.sin(ang))
            ring.append([head, head + axis * blen])
        rings[side] = ring
    if not rings:
        return False, "No sleeve chains or forearm bones to anchor the cuffs"

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        mo.hide_set(False)
    except Exception:
        pass
    mo.hide_viewport = False
    bpy.context.view_layer.objects.active = mo
    mo.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = mo.data.edit_bones
    for b in [b for b in eb if b.name.startswith(BONE_CUFF + ".")]:
        eb.remove(b)
    made = []
    for side, ring in rings.items():
        parent = (eb.get("hand." + side) or eb.get("forearm." + side))
        for k, (head, tail) in enumerate(ring):
            name = "%s.%s.%02d" % (BONE_CUFF, side, k)
            b = eb.new(name)
            b.head = head
            b.tail = tail
            if parent is not None:
                b.parent = parent
                b.use_connect = False
            made.append(name)
    bpy.ops.object.mode_set(mode='OBJECT')
    for name in made:
        pb = mo.pose.bones.get(name)
        if pb is not None:
            try:
                pb.rigify_type = "basic.super_copy"
                pb.rigify_parameters.make_deform = True
            except Exception:
                pass
    mo["sr_kandura"] = True
    return True, made


class SMARTRIG_OT_kandura_add_cuffs(bpy.types.Operator):
    bl_idname = "smartrig.kandura_add_cuffs"
    bl_label = "Add Cuff Bones"
    bl_description = ("Add a ring of cuff bones around each sleeve END "
                      "(wrist opening) - place the sleeves first so the ring "
                      "lands on the real cuff; then adjust MANUALLY")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _metarig() is not None

    def execute(self, context):
        ok, res = add_cuff_bones(context)
        if not ok:
            self.report({'ERROR'}, res)
            return {'CANCELLED'}
        _enter_metarig_edit(context, select_names=res)
        align_snap_apply(context, context.scene.smartrig.kandura_align_surface)
        mirror_apply(context, context.scene.smartrig.kandura_mirror)
        self.report({'INFO'},
                    "Cuff rings added (%d bones) - place them manually"
                    % len(res))
        return {'FINISHED'}


# ====================================================================
# REMOVE
# ====================================================================

# ====================================================================
# POST-GENERATE — SLEEVE ROLL-UP (tashmeer) + sleeve-end hand follow
# ====================================================================

ROLLUP_MASTER = "kan_rollup"


def _kan_joints(rig, side):
    """Rest joints of the generated sleeve chain, shoulder -> cuff tip."""
    import re
    pat = re.compile(r"^ORG-%s\.%s\.(\d+)$" % (BONE_SLEEVE, side))
    segs = {}
    for b in rig.data.bones:
        m = pat.match(b.name)
        if m:
            segs[int(m.group(1))] = b
    if not segs or set(segs) != set(range(len(segs))):
        return None
    joints = [Vector(segs[k].head_local) for k in range(len(segs))]
    joints.append(Vector(segs[len(segs) - 1].tail_local))
    return joints


def _kan_tweak_map(rig, side, joints):
    """{joint index: tweak bone name} matched by nearest rest head."""
    out = {}
    for b in rig.data.bones:
        if not b.name.startswith("tweak_%s.%s" % (BONE_SLEEVE, side)):
            continue
        h = Vector(b.head_local)
        k = min(range(len(joints)), key=lambda i: (joints[i] - h).length)
        if (joints[k] - h).length < 0.02 and k not in out:
            out[k] = b.name
    return out


def _kanr_var(drv, nm, rig, kind, bone, key):
    v = drv.variables.new(); v.name = nm
    if kind in ('LOC', 'ROTX', 'ROTZ'):
        v.type = 'TRANSFORMS'
        t = v.targets[0]; t.id = rig; t.bone_target = bone
        t.transform_type = {'LOC': 'LOC_Y', 'ROTX': 'ROT_X',
                            'ROTZ': 'ROT_Z'}[kind]
        t.transform_space = 'LOCAL_SPACE'
    else:
        v.type = 'SINGLE_PROP'
        t = v.targets[0]; t.id_type = 'OBJECT'; t.id = rig
        t.data_path = 'pose.bones["%s"]["%s"]' % (bone, key)


def remove_sleeve_rollup(rig):
    """Undo add_sleeve_rollup: restore tweak/cuff parents, drop helpers,
    masters and their drivers."""
    if rig is None:
        return
    from . import skirt as _sk
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    restore = {}
    for pb in rig.pose.bones:
        if "kanr_origparent" in pb.keys():
            restore[pb.name] = pb["kanr_origparent"]
    doomed = tuple(n for n in rig.data.bones.keys()
                   if n.startswith(("KANR_dt.", "KANH_dt.", "KANH_tgt.",
                                    "KANC_root.", "KANC_dt.", "KANF_dt.",
                                    ROLLUP_MASTER + ".")))
    if not doomed and not restore:
        return
    ad = rig.animation_data
    if ad is not None:
        for fc in list(ad.drivers):
            if any(('"%s"' % n) in fc.data_path for n in doomed):
                ad.drivers.remove(fc)
    if not _sk._edit_rig(rig):
        return
    eb = rig.data.edit_bones
    for name, pn in restore.items():
        b = eb.get(name)
        if b is not None:
            b.parent = eb.get(pn) if pn else None
    for n in doomed:
        b = eb.get(n)
        if b is not None:
            eb.remove(b)
    bpy.ops.object.mode_set(mode='OBJECT')
    for name in restore:
        pb = rig.pose.bones.get(name)
        if pb is not None and "kanr_origparent" in pb.keys():
            del pb["kanr_origparent"]
    rig["kan_rollup"] = 0


def add_sleeve_rollup(rig, props):
    """ARP-style sleeve ROLL-UP (tashmeer). One master per arm at the cuff:
    drag it UP the forearm and the sleeve gathers accordion-style toward
    the ELBOW, thickening as it bunches (pile/bulge live settings on the
    master). The sleeve END also softly follows the hand (damped track on
    ORG-hand -> works in FK AND IK) so the opening never tears away from
    the wrist, and the kan_cuff ring is re-parented from the hand onto the
    sleeve END so it rides the roll-up instead of clipping into the hand.
    Everything lives in arm space -> layers on top of FK/IK. Returns the
    number of masters made."""
    if rig is None:
        return 0
    from . import skirt as _sk
    _sk._ensure_drivers_trusted()
    remove_sleeve_rollup(rig)

    data = {}
    for side in ("L", "R"):
        joints = _kan_joints(rig, side)
        fb = rig.data.bones.get("ORG-forearm." + side)
        if joints is None or len(joints) < 3 or fb is None:
            continue
        elbow = Vector(fb.head_local)
        ei = min(range(len(joints)), key=lambda i: (joints[i] - elbow).length)
        tipi = len(joints) - 1
        if ei >= tipi:
            continue
        arc = {tipi: 0.0}
        s = 0.0
        for k in range(tipi, 0, -1):
            s += (joints[k] - joints[k - 1]).length
            arc[k - 1] = s
        Lf = arc[ei]                 # cuff -> elbow (hand-follow fade)
        Lt = arc[0]                  # cuff -> TOP of the sleeve (full roll)
        if Lf < 1e-4 or Lt < 1e-4:
            continue
        twmap = _kan_tweak_map(rig, side, joints)
        if tipi not in twmap:
            continue
        data[side] = (joints, twmap, ei, tipi, arc, Lf, Lt)
    if not data:
        return 0

    if not _sk._edit_rig(rig):
        return 0
    eb = rig.data.edit_bones
    orig_parent = {}
    ring_dirs = {}
    for side, (joints, twmap, ei, tipi, arc, Lf, Lt) in data.items():
        # hand-follow jig: rotates the sleeve END about the last joint
        hj = eb.new("KANH_dt." + side)
        hj.head = joints[tipi - 1].copy(); hj.tail = joints[tipi].copy()
        hj.use_deform = False
        # track TARGET on the hand, placed ON the rest line of the sleeve
        # end -> ZERO deformation at rest, follows the hand when it moves
        hand = eb.get("ORG-hand." + side)
        if hand is not None:
            dr = (joints[tipi] - joints[tipi - 1]).normalized()
            hlen = (hand.tail - hand.head).length
            tg = eb.new("KANH_tgt." + side)
            tg.head = joints[tipi] + dr * max(0.03, 0.45 * hlen)
            tg.tail = tg.head + dr * 0.02
            tg.use_deform = False
            tg.parent = hand
        # roll helpers: one per tweak, cuff to the TOP (joint 0 anchors)
        for k in range(1, tipi + 1):
            twn = twmap.get(k)
            if twn is None:
                continue
            tb = eb.get(twn)
            if tb is None:
                continue
            d = (joints[k - 1] - joints[k])
            if d.length < 1e-9:
                continue
            d.normalize()
            hb = eb.new("KANR_dt.%s.%02d" % (side, k))
            hb.head = joints[k].copy()
            hb.tail = joints[k] + d * 0.025      # local +Y = up the sleeve
            hb.use_deform = False
            op = tb.parent
            orig_parent[twn] = op.name if op else ""
            if k == tipi:
                hj.parent = op
                hb.parent = hj
            else:
                hb.parent = op
            tb.parent = hb
        # FOREARM FOLLOW: the sleeve chain is rooted on the UPPER ARM, so
        # without this the below-elbow part keeps going STRAIGHT when the
        # elbow bends (the sleeve slides OFF the arm). A hinge aligned with
        # ORG-forearm is inserted above the first below-elbow control: it
        # copies the forearm's WORLD rotation about the elbow -> the whole
        # lower sleeve (controls, tweaks, rim, master) bends with the arm,
        # FK and IK, while staying zero at rest.
        fob = eb.get("ORG-forearm." + side)
        ctl0 = eb.get("%s.%s.%02d" % (BONE_SLEEVE, side, ei))
        if fob is not None and ctl0 is not None:
            fj = eb.new("KANF_dt." + side)
            fj.head = fob.head.copy()
            fj.tail = fob.tail.copy()
            fj.roll = fob.roll
            fj.use_deform = False
            fj.parent = ctl0.parent
            orig_parent[ctl0.name] = ctl0.parent.name if ctl0.parent else ""
            ctl0.use_connect = False
            ctl0.parent = fj
        # roll-up master: at the cuff, +Y pointing up the forearm
        d0 = (joints[ei] - joints[tipi]).normalized()
        mb = eb.new(ROLLUP_MASTER + "." + side)
        mb.head = joints[tipi].copy()
        mb.tail = joints[tipi] + d0 * min(0.1, max(0.05, 0.3 * Lf))
        mb.use_deform = False
        mb.parent = eb.get("ORG-forearm." + side)
        # cuff ring root: rides the sleeve END (roll-up + hand follow)
        ring = [n for n in rig.data.bones.keys()
                if n.startswith("%s.%s." % (BONE_CUFF, side))
                and not n.startswith(("DEF-", "ORG-"))]
        if ring:
            tip_h = eb.get("KANR_dt.%s.%02d" % (side, tipi))
            cr = eb.new("KANC_root." + side)
            cr.head = joints[tipi].copy()
            cr.tail = joints[tipi] + (joints[tipi] - joints[tipi - 1]).normalized() * 0.04
            cr.use_deform = False
            cr.parent = tip_h if tip_h is not None else hj
            # ARP CLOTH-KILT MODEL, hand edition: one compass helper per rim
            # bone; its local +Y = radially OUTWARD from the cuff opening.
            # The driver pushes ONLY the rim sector the hand kicks toward -
            # the sleeve never lifts or follows the hand back.
            axis = (joints[tipi] - joints[tipi - 1]).normalized()
            cen = Vector((0.0, 0.0, 0.0))
            rbs = [eb.get(n) for n in ring if eb.get(n) is not None]
            for cb in rbs:
                cen += cb.head
            cen = cen / max(1, len(rbs))
            hbn = rig.data.bones.get("ORG-hand." + side)
            HM = hbn.matrix_local.to_3x3() if hbn is not None else None
            for n in ring:
                cb = eb.get(n)
                if cb is None:
                    continue
                orig_parent[n] = cb.parent.name if cb.parent else ""
                o = cb.head - cen
                o = o - axis * o.dot(axis)
                if o.length < 1e-6:
                    cb.parent = cr
                    continue
                o.normalize()
                dt = eb.new("KANC_dt.%s.%s" % (side, n.rsplit(".", 1)[-1]))
                dt.head = cb.head.copy()
                dt.tail = cb.head + o * 0.02
                dt.use_deform = False
                dt.parent = cr
                cb.parent = dt
                if HM is not None:
                    ring_dirs.setdefault(side, {})[dt.name] = (
                        o.dot(HM.col[0]), o.dot(HM.col[2]))
    bpy.ops.object.mode_set(mode='OBJECT')

    for name, pn in orig_parent.items():
        pb = rig.pose.bones.get(name)
        if pb is not None:
            pb["kanr_origparent"] = pn

    made = 0
    for side, (joints, twmap, ei, tipi, arc, Lf, Lt) in data.items():
        mn = ROLLUP_MASTER + "." + side
        mpb = rig.pose.bones.get(mn)
        if mpb is None:
            continue
        mpb.rotation_mode = 'XYZ'
        for key, val, lo, hi, desc in (
                ("pile", 0.02, 0.005, 0.12,
                 "Spacing of the gathered folds (roll-up stack)"),
                ("bulge", 0.0, 0.0, 1.5,
                 "How much the gathered fabric thickens as it rolls up"),
                ("hand_follow", 0.0, 0.0, 1.0,
                 "OPTIONAL soft follow of the hand by the sleeve END "
                 "(default 0 = pure kilt-style collision, no lifting)"),
                ("cuff_collide", 1.0, 0.0, 2.0,
                 "Cloth-kilt hand collision STRENGTH: how hard the rim "
                 "sector the hand bends toward is pushed (never lifts)"),
                ("cuff_dist", 0.07, 0.0, 0.25,
                 "Cloth-kilt hand collision DISTANCE: how far the rim can "
                 "be pushed away from the hand"),
                ("hand_clear", 0.6, 0.0, 1.0,
                 "Sleeve END retreats up the forearm when the wrist bends "
                 "so the cuff never intersects the hand (0 = off)")):
            mpb[key] = float(val)
            try:
                ui = mpb.id_properties_ui(key)
                ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi,
                          description=desc)
            except Exception:
                pass
        con = mpb.constraints.new('LIMIT_LOCATION')
        con.use_min_x = con.use_max_x = True
        con.use_min_z = con.use_max_z = True
        con.use_min_y = con.use_max_y = True
        con.min_y = 0.0; con.max_y = Lt
        con.owner_space = 'LOCAL'; con.use_transform_limit = True
        try:
            mpb.custom_shape = _sk._ensure_master_widget()
            mpb.custom_shape_scale_xyz = (0.5, 0.5, 0.5)
        except Exception:
            pass
        src_b = rig.data.bones.get("%s.%s.00" % (BONE_SLEEVE, side))
        if src_b is not None:
            for coll in src_b.collections:
                try:
                    coll.assign(rig.data.bones[mn])
                except Exception:
                    pass
        # forearm-follow hinge
        fpb = rig.pose.bones.get("KANF_dt." + side)
        if fpb is not None and rig.data.bones.get("ORG-forearm." + side):
            cr2 = fpb.constraints.new('COPY_ROTATION')
            cr2.target = rig; cr2.subtarget = "ORG-forearm." + side
            cr2.target_space = 'WORLD'; cr2.owner_space = 'WORLD'
            cr2.mix_mode = 'REPLACE'
            rig.data.bones["KANF_dt." + side].hide = True
        # hand follow (damped track, fades out as the sleeve rolls up)
        hpb = rig.pose.bones.get("KANH_dt." + side)
        if hpb is not None and rig.data.bones.get("KANH_tgt." + side):
            dt = hpb.constraints.new('DAMPED_TRACK')
            dt.target = rig; dt.subtarget = "KANH_tgt." + side
            dt.head_tail = 0.0; dt.track_axis = 'TRACK_Y'
            drv = dt.driver_add("influence").driver
            drv.type = 'SCRIPTED'
            _kanr_var(drv, "t", rig, 'LOC', mn, "")
            _kanr_var(drv, "cf", rig, 'PROP', mn, "hand_follow")
            drv.expression = "cf*max(0.0, 1.0 - t/%.4f)" % max(1e-4, Lf)
        # NO ballooning: the sleeve stretch segments must not fatten as
        # they compress (the roll-up look is controlled ONLY by "bulge")
        import re as _re
        for opb in rig.pose.bones:
            if _re.match(r"^ORG-%s\.%s\.\d+$" % (BONE_SLEEVE, side), opb.name):
                for c in opb.constraints:
                    if c.type == 'STRETCH_TO':
                        c.volume = 'NO_VOLUME'
        # CASCADE roll-up: each gathered tweak COPIES THE LOCATION of the
        # tweak ABOVE it as the master passes its rest spot - the tip rides
        # bone-for-bone up the LIVE chain (works in any FK/IK pose), and the
        # stack cascades because each target itself moves up in turn.
        # head_tail keeps a small "pile" offset so no segment collapses.
        rank = 0
        for k in range(tipi, 0, -1):
            hn = "KANR_dt.%s.%02d" % (side, k)
            hb = rig.pose.bones.get(hn)
            tgt = twmap.get(k - 1)
            if hb is None or tgt is None:
                continue
            a = arc[k]
            seg = max(1e-4, arc[k - 1] - arc[k])
            tlen = max(1e-4, rig.data.bones[tgt].length)
            con = hb.constraints.new('COPY_LOCATION')
            con.name = "KAN Roll Cascade"
            con.target = rig; con.subtarget = tgt
            drv = con.driver_add("influence").driver
            drv.type = 'SCRIPTED'
            _kanr_var(drv, "t", rig, 'LOC', mn, "")
            drv.expression = ("min(1.0, max(0.0, (t - %.4f)/%.4f))"
                              % (a, seg))
            dht = con.driver_add("head_tail").driver
            dht.type = 'SCRIPTED'
            _kanr_var(dht, "pl", rig, 'PROP', mn, "pile")
            dht.expression = "min(0.9, pl/%.4f)" % tlen
            # optional fold thickening (default 0 = perfectly clean)
            for idx in (0, 2):
                d2 = hb.driver_add("scale", idx).driver
                d2.type = 'SCRIPTED'
                _kanr_var(d2, "t", rig, 'LOC', mn, "")
                _kanr_var(d2, "bg", rig, 'PROP', mn, "bulge")
                _kanr_var(d2, "pl", rig, 'PROP', mn, "pile")
                d2.expression = ("1.0 + bg*min(1.0, max(0.0, t - %.4f)"
                                 "/max(0.02, pl))" % a)
            # HAND CLEARANCE: the sleeve END retreats up the forearm when
            # the wrist bends, so the cuff opening NEVER eats into the hand
            if k == tipi and rig.data.bones.get("ORG-hand." + side):
                d3 = hb.driver_add("location", 1).driver
                d3.type = 'SCRIPTED'
                _kanr_var(d3, "rx", rig, 'ROTX', "ORG-hand." + side, "")
                _kanr_var(d3, "rz", rig, 'ROTZ', "ORG-hand." + side, "")
                _kanr_var(d3, "hc", rig, 'PROP', mn, "hand_clear")
                d3.expression = "hc*0.06*min(1.6, abs(rx) + abs(rz))"
            rig.data.bones[hn].hide = True
            rank += 1
        # cloth-kilt compass: push each rim sector outward by the component
        # of the hand's bend that points AT it (max(0,..) = outward only)
        for dtn, (ox, oz) in ring_dirs.get(side, {}).items():
            dpb = rig.pose.bones.get(dtn)
            if dpb is None:
                continue
            drv = dpb.driver_add("location", 1).driver
            drv.type = 'SCRIPTED'
            _kanr_var(drv, "rx", rig, 'ROTX', "ORG-hand." + side, "")
            _kanr_var(drv, "rz", rig, 'ROTZ', "ORG-hand." + side, "")
            _kanr_var(drv, "cc", rig, 'PROP', mn, "cuff_collide")
            _kanr_var(drv, "cd", rig, 'PROP', mn, "cuff_dist")
            drv.expression = ("cc*min(cd*1.3, cd*max(0.0, %.4f*(-rz) + %.4f*rx))"
                              % (ox, oz))
            rig.data.bones[dtn].hide = True
        for hn in ("KANH_dt." + side, "KANH_tgt." + side,
                   "KANC_root." + side):
            if rig.data.bones.get(hn):
                rig.data.bones[hn].hide = True
        made += 1
    rig["kan_rollup"] = 1 if made else 0
    return made


# ====================================================================
# SLEEVE WEIGHT POLISH — professional kandura binding
# ====================================================================

def polish_sleeve_weights(ob, rig):
    """PROFESSIONAL sleeve binding pass. The generic bind leaves part of the
    sleeve fabric weighted to the BODY arm bones (upper_arm/forearm/hand) -
    those verts stay behind when the sleeve chain rolls up or the cuff
    follows the hand => tearing. This pass:
      1. deletes DEAD vertex groups (no bone of that name on the rig),
      2. on every vert that carries kan_sleeve/kan_cuff weight, transfers
         the body-ARM share to the NEAREST kan_sleeve/kan_cuff bone,
      3. graph-smooths the kan weights inside the sleeve (keeps the torso
         blend at the shoulder seam untouched).
    Returns (n_dead_groups, n_verts_fixed)."""
    if ob is None or rig is None:
        return 0, 0
    bones = set(rig.data.bones.keys())
    dead = [g for g in ob.vertex_groups if g.name not in bones]
    n_dead = len(dead)
    for g in dead:
        ob.vertex_groups.remove(g)

    rw = rig.matrix_world
    mw = ob.matrix_world
    fixed = 0
    for side in ("L", "R"):
        kan_named = [b.name for b in rig.data.bones
                     if b.use_deform
                     and (b.name.startswith("DEF-%s.%s." % (BONE_SLEEVE, side))
                          or b.name.startswith("DEF-%s.%s." % (BONE_CUFF, side)))]
        if not kan_named:
            continue
        segs = []
        for n in kan_named:
            b = rig.data.bones[n]
            segs.append((n, rw @ b.head_local, rw @ b.tail_local))
        arm_named = [n for n in (
            "DEF-upper_arm.%s" % side, "DEF-upper_arm.%s.001" % side,
            "DEF-forearm.%s" % side, "DEF-forearm.%s.001" % side,
            "DEF-hand.%s" % side) if n in bones]
        gi = {g.name: g.index for g in ob.vertex_groups}
        kan_idx = {gi[n] for n in kan_named if n in gi}
        arm_idx = {gi[n] for n in arm_named if n in gi}
        if not kan_idx:
            continue

        def _nearest(pt):
            best, bn = 1e18, None
            for n, a, b in segs:
                ab = b - a
                L2 = ab.length_squared
                t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, (pt - a).dot(ab) / L2))
                d = (a + ab * t - pt).length_squared
                if d < best:
                    best, bn = d, n
            return bn

        sleeve_verts = []
        for v in ob.data.vertices:
            wk = sum(g.weight for g in v.groups if g.group in kan_idx)
            if wk <= 0.02:
                continue
            sleeve_verts.append(v.index)
            wa = sum(g.weight for g in v.groups if g.group in arm_idx)
            if wa <= 1e-4:
                continue
            tgt = _nearest(mw @ v.co)
            if tgt is None:
                continue
            for idx in arm_idx:
                try:
                    ob.vertex_groups[idx].remove([v.index])
                except Exception:
                    pass
            ob.vertex_groups[tgt].add([v.index], wa, 'ADD')
            fixed += 1

        # graph smoothing of the kan groups INSIDE the sleeve only: keeps
        # the shoulder-seam torso blend, evens the roll-up transitions
        sv = set(sleeve_verts)
        nbr = {i: [] for i in sv}
        for e in ob.data.edges:
            a, b = e.vertices
            if a in sv and b in sv:
                nbr[a].append(b)
                nbr[b].append(a)
        gnames = [n for n in kan_named if n in gi]
        for _ in range(2):
            cur = {}
            for n in gnames:
                g = ob.vertex_groups[n]
                col = {}
                for i in sv:
                    try:
                        col[i] = g.weight(i)
                    except RuntimeError:
                        col[i] = 0.0
                cur[n] = col
            for n in gnames:
                g = ob.vertex_groups[n]
                col = cur[n]
                for i in sv:
                    ns = nbr[i]
                    if not ns:
                        continue
                    w = 0.5 * col[i] + 0.5 * sum(col[j] for j in ns) / len(ns)
                    # rescale so the vert's total kan share is preserved
                    g.add([i], w, 'REPLACE')
            # renormalise the kan share per vert (torso share untouched)
            for i in sv:
                v = ob.data.vertices[i]
                wk = sum(gg.weight for gg in v.groups if gg.group in
                         {gi[n] for n in gnames})
                tot = sum(gg.weight for gg in v.groups)
                free = max(0.0, 1.0 - (tot - wk))
                if wk > 1e-6 and abs(wk - free) > 1e-4:
                    s = free / wk
                    for n in gnames:
                        try:
                            w0 = ob.vertex_groups[n].weight(i)
                        except RuntimeError:
                            continue
                        ob.vertex_groups[n].add([i], w0 * s, 'REPLACE')
    ob.data.update()
    return n_dead, fixed


class SMARTRIG_OT_kandura_polish_weights(bpy.types.Operator):
    bl_idname = "smartrig.kandura_polish_weights"
    bl_label = "Polish Sleeve Weights"
    bl_description = ("PROFESSIONAL sleeve binding: move any body-arm weight "
                      "on the sleeve fabric onto the nearest sleeve/cuff bone "
                      "(stops tearing on roll-up), delete dead vertex groups, "
                      "smooth the sleeve weights (shoulder blend untouched)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return kandura_object(context) is not None

    def execute(self, context):
        ob = kandura_object(context)
        rig = None
        for m in ob.modifiers:
            if m.type == 'ARMATURE' and m.object is not None:
                rig = m.object
                break
        if rig is None:
            self.report({'ERROR'}, "Bind the kandura first (no Armature modifier)")
            return {'CANCELLED'}
        nd, nf = polish_sleeve_weights(ob, rig)
        self.report({'INFO'},
                    "Sleeve weights polished: %d verts fixed, %d dead groups removed"
                    % (nf, nd))
        return {'FINISHED'}


# ====================================================================
# KANDURA ANTI-PENETRATION — the body must NEVER poke through the cloth
# ====================================================================

def remove_kandura_antipen(rig):
    n = 0
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        md = ob.modifiers.get("KAN_AntiPen")
        if md is not None:
            try:
                ob.modifiers.remove(md); n += 1
            except Exception:
                pass
    if rig is not None and "kan_antipen" in rig:
        del rig["kan_antipen"]
    return n


def add_kandura_antipen(rig, props):
    """PROFESSIONAL no-clipping layer for the WHOLE kandura: a Shrinkwrap
    in 'OUTSIDE' mode pushes ONLY the verts that end up INSIDE the body
    back out to the surface (+offset) - verts already outside are never
    touched, so the drape the rig produced is preserved. Covers the torso,
    the sleeves AND the cuff-vs-hand case in every pose, FK or IK.
    Topology-safe; ordered right after the Armature modifier."""
    ob = kandura_object(bpy.context)
    body = getattr(props, "target_mesh", None)
    if ob is None or body is None or ob is body:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    remove_kandura_antipen(rig)
    vg = (ob.vertex_groups.get("SR_KanAntiPen")
          or ob.vertex_groups.new(name="SR_KanAntiPen"))
    vg.add([v.index for v in ob.data.vertices], 1.0, 'REPLACE')
    mod = ob.modifiers.new("KAN_AntiPen", 'SHRINKWRAP')
    mod.target = body
    mod.wrap_method = 'NEAREST_SURFACEPOINT'
    mod.wrap_mode = 'OUTSIDE'      # push out ONLY penetrating verts
    mod.offset = float(getattr(props, "kandura_antipen_offset", 0.005))
    mod.vertex_group = "SR_KanAntiPen"
    # order: right AFTER the Armature deform
    names = [m.name for m in ob.modifiers]
    after = None
    for mm in ob.modifiers:
        if mm.type in ('ARMATURE', 'SURFACE_DEFORM'):
            after = mm.name
    if after is not None:
        try:
            win = bpy.context.window
            area = next((a for a in win.screen.areas if a.type == 'VIEW_3D'),
                        None) if win else None
            region = (next((r for r in area.regions if r.type == 'WINDOW'),
                           None) if area else None)
            ov = {"object": ob, "active_object": ob}
            if win: ov["window"] = win
            if area: ov["area"] = area
            if region: ov["region"] = region
            with bpy.context.temp_override(**ov):
                idx = [m.name for m in ob.modifiers].index(after) + 1
                if [m.name for m in ob.modifiers].index("KAN_AntiPen") != idx:
                    bpy.ops.object.modifier_move_to_index(
                        modifier="KAN_AntiPen", index=idx)
        except Exception as e:
            print("SmartRig kandura anti-pen reorder:", e)
    if rig is not None:
        rig["kan_antipen"] = 1
    return 1


def live_kandura_antipen(context):
    try:
        ob = kandura_object(context)
        md = ob.modifiers.get("KAN_AntiPen") if ob else None
        if md is not None:
            md.offset = float(context.scene.smartrig.kandura_antipen_offset)
    except Exception as e:
        print("SmartRig kandura anti-pen tune:", e)


def remove_kandura_bones(mo):
    """Delete every bone the kandura module created from the metarig."""
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        mo.hide_set(False)
    except Exception:
        pass
    bpy.ops.object.select_all(action='DESELECT')
    mo.select_set(True)
    bpy.context.view_layer.objects.active = mo
    bpy.ops.object.mode_set(mode='EDIT')
    eb = mo.data.edit_bones
    n = 0
    kandura_skirt = (mo.get("sr_skirt_method") == "kandura")
    for b in list(eb):
        nm = b.name
        if nm.startswith("kan_") or (kandura_skirt and nm.startswith("skirt.")):
            eb.remove(b)
            n += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    if kandura_skirt:
        for key in ("sr_skirt_kind", "sr_skirt_method", "sr_skirt_cols_built"):
            if key in mo:
                del mo[key]
    if "sr_kandura" in mo:
        del mo["sr_kandura"]
    return n


class SMARTRIG_OT_kandura_remove(bpy.types.Operator):
    bl_idname = "smartrig.kandura_remove"
    bl_label = "Remove Kandura Bones"
    bl_description = ("Delete kandura bones from the metarig (one part or "
                      "everything). Re-generate the rig afterwards to update it")
    bl_options = {'REGISTER', 'UNDO'}

    part: bpy.props.EnumProperty(
        items=[('ALL', "All", "Delete ALL kandura bones"),
               ('WAIST', "Waist-down", "Delete the waist-down grid bones"),
               ('SLEEVES', "Sleeves", "Delete the sleeve chain bones"),
               ('COLLAR', "Collar", "Delete the collar ring bones"),
               ('CUFF', "Cuffs", "Delete the cuff ring bones")],
        default='ALL')

    @classmethod
    def poll(cls, context):
        return _metarig() is not None

    def execute(self, context):
        mo = _metarig()
        if self.part == 'ALL':
            n = remove_kandura_bones(mo)
            self.report({'INFO'},
                        "Removed %d kandura bones - Re-generate the rig" % n)
            return {'FINISHED'}
        prefix = {'WAIST': "skirt.",
                  'SLEEVES': BONE_SLEEVE + ".",
                  'COLLAR': BONE_COLLAR + ".",
                  'CUFF': BONE_CUFF + "."}[self.part]
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            mo.hide_set(False)
        except Exception:
            pass
        mo.hide_viewport = False
        bpy.ops.object.select_all(action='DESELECT')
        mo.select_set(True)
        bpy.context.view_layer.objects.active = mo
        bpy.ops.object.mode_set(mode='EDIT')
        eb = mo.data.edit_bones
        n = 0
        for b in list(eb):
            if b.name.startswith(prefix):
                eb.remove(b)
                n += 1
        bpy.ops.object.mode_set(mode='OBJECT')
        if self.part == 'WAIST' and mo.get("sr_skirt_method") == "kandura":
            for key in ("sr_skirt_kind", "sr_skirt_method",
                        "sr_skirt_cols_built"):
                if key in mo:
                    del mo[key]
        # no kandura bones left at all -> clear the flag
        if not any(b.name.startswith("kan_")
                   or (mo.get("sr_skirt_method") == "kandura"
                       and b.name.startswith("skirt."))
                   for b in mo.data.bones):
            if "sr_kandura" in mo:
                del mo["sr_kandura"]
        if n == 0:
            self.report({'WARNING'}, "No %s bones to remove" % self.part.lower())
            return {'CANCELLED'}
        self.report({'INFO'}, "Removed %d %s bones" % (n, self.part.lower()))
        return {'FINISHED'}


classes = (SMARTRIG_OT_kandura_add_waist,
           SMARTRIG_OT_kandura_add_sleeves,
           SMARTRIG_OT_kandura_polish_weights,
           SMARTRIG_OT_kandura_add_collar,
           SMARTRIG_OT_kandura_add_cuffs,
           SMARTRIG_OT_kandura_align_now,
           SMARTRIG_OT_kandura_mirror_now,
           SMARTRIG_OT_kandura_remove)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    if _kan_snap_watch not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_kan_snap_watch)


def unregister():
    if _kan_snap_watch in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_kan_snap_watch)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
