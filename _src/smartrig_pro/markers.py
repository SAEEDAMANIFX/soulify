import bpy
import numpy as np
from mathutils import Vector, Quaternion
from bpy_extras import view3d_utils
from . import utils
from . import detect

CENTER_MARKERS = ["spine_root", "neck", "head_top"]
LEFT_MARKERS = ["shoulder.L", "elbow.L", "wrist.L", "hip.L", "knee.L", "ankle.L"]
FOOT_MARKERS = ["ball.L", "foottip.L"]          # placed from TOP view (ball + toe tip)
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def _marker_coll():
    c = utils.ensure_collection(utils.MARKERS_COLL)
    try:
        c.color_tag = 'COLOR_04'
    except Exception:
        pass
    return c


def get_marker(name):
    return bpy.data.objects.get(name)


def _new_empty(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        _marker_coll().objects.link(obj)
    return obj


def _style(obj, size, role):
    # tiny empty core (the colourful GPU glow is the visible marker on top of it).
    # keep the black axes VERY small so they hide under the glow but stay selectable.
    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = size * 0.12   # small, unobtrusive; the glow is the visual
    obj.show_name = False
    obj.show_in_front = True
    obj.color = {'center': (0.2, 0.9, 1.0, 1.0),
                 'left': (1.0, 0.8, 0.1, 1.0),
                 'right': (0.55, 0.45, 0.2, 1.0),
                 'finger': (0.3, 1.0, 0.45, 1.0)}[role]
    if role == 'center':
        obj.lock_location = (True, True, False)   # up/down (Z) only
    elif role == 'right':
        obj.lock_location = (True, True, True)     # driven by its .L counterpart
    else:
        obj.lock_location = (False, False, False)  # left side / fingers: free 3D move


def _add_mirror_constraint(right_obj, left_name):
    for c in list(right_obj.constraints):
        right_obj.constraints.remove(c)
    con = right_obj.constraints.new('COPY_LOCATION')
    con.name = "SR Mirror"
    con.target = get_marker(left_name)
    con.use_x = con.use_y = con.use_z = True
    con.invert_x = True
    con.target_space = 'WORLD'
    con.owner_space = 'WORLD'


def _leg_center_guess(co, ground, h, sign):
    """Femur-head guess: fit the leg's cross-section centres sampled below the
    crotch (clean of arms), then read the hip height off the fit."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    side = (x * sign) > 0.02 * h
    zs, xs, ys = [], [], []
    for fr in np.linspace(0.36, 0.18, 6):
        zc = ground + fr * h
        m = side & (np.abs(z - zc) < 0.03 * h)
        if m.sum() > 3:
            zs.append(zc); xs.append(float(x[m].mean())); ys.append(float(y[m].mean()))
    hip_z = ground + 0.53 * h
    if len(zs) >= 2:
        px = np.polyfit(zs, xs, 1); py = np.polyfit(zs, ys, 1)
        return (float(np.polyval(px, hip_z)), float(np.polyval(py, hip_z)), hip_z)
    return (sign * 0.12 * h, float(y.mean()), hip_z)


def _guess_positions(co, heights=None):
    """Geometric marker guess. `heights` (optional) is a dict of ADAPTIVE joint
    heights as fractions of the mesh height, supplied by the neural detector
    (detect.detect_height_fractions). Keys: spine_root, neck, shoulder, hip,
    ankle. When present they replace the fixed anatomical fractions, so the
    markers follow each character's true proportions; the lateral (X) and depth
    (Y) are still solved geometrically below (verified to mm)."""
    H = heights or {}
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    ground, top = float(z.min()), float(z.max())
    h = top - ground
    cen = np.abs(x) < 0.15
    yc = float(np.median(y[cen])) if cen.any() else float(y.mean())

    def band(zc, tol):
        return np.abs(z - zc) < tol * h

    def hz(key, default, window=0.06):
        # Neural heights refine the verified geometric defaults but are CLAMPED
        # to a safe window: a single 384px detection can occasionally swing
        # wildly (bad neck/ankle), so it may nudge - never break - placement.
        v = float(H.get(key, default))
        v = min(default + window, max(default - window, v))
        return ground + v * h

    pos = {}
    pos["spine_root"] = (0.0, yc, hz("spine_root", 0.50))
    pos["neck"] = (0.0, yc, hz("neck", 0.82))
    pos["head_top"] = (0.0, yc, top - 0.01 * h)

    # shoulder: lateral edge of the torso at shoulder height (exclude the arm)
    sh_z = hz("shoulder", 0.80)
    m = band(sh_z, 0.025) & (x > 0) & (x < 0.30 * h)
    sh_x = float(np.percentile(x[m], 70)) if m.any() else 0.16 * h
    pos["shoulder.L"] = (sh_x, yc, sh_z)

    # hip: femur head = median of the PELVIS half-section at hip height, with the
    # arm/forearm excluded by an x cap (the arm sits well outside the pelvis)
    hip_z = hz("hip", 0.53)
    m = band(hip_z, 0.035) & (x > 0.02 * h) & (x < 0.13 * h)
    hip_x = float(np.median(x[m])) if m.sum() > 5 else 0.11 * h
    pos["hip.L"] = (hip_x, yc, hip_z)

    # wrist: hand cluster, then nudged a good way up the arm toward the shoulder
    # so the marker sits on the WRIST JOINT, not down on the hand/fingers
    arm = x > 0.60 * float(x.max())
    if arm.sum() > 5:
        zc = float(np.percentile(z[arm], 22))
        m2 = arm & (np.abs(z - zc) < 0.05 * h)
        wr = co[m2].mean(0)
        wr = wr + (np.array(pos["shoulder.L"], dtype=np.float64) - wr) * 0.18
        pos["wrist.L"] = (float(wr[0]), float(wr[1]), float(wr[2]))
    else:
        pos["wrist.L"] = (0.45 * h, yc, ground + 0.50 * h)

    # ankle joint: just above the ground (NOT up on the shin), at the leg's x
    ankle_z = hz("ankle", 0.06)
    m = band(ankle_z, 0.025) & (x > 0.02 * h)
    ax = float(np.median(x[m])) if m.sum() > 3 else 0.13 * h
    pos["ankle.L"] = (ax, yc, ankle_z)

    # elbow = mid of the arm centre-line; knee = mid of the leg (good starting hints)
    shp = np.array(pos["shoulder.L"]); wrp = np.array(pos["wrist.L"])
    el = shp + (wrp - shp) * 0.5
    pos["elbow.L"] = (float(el[0]), float(el[1]), float(el[2]))
    hpp = np.array(pos["hip.L"]); anp = np.array(pos["ankle.L"])
    kn = hpp + (anp - hpp) * 0.5
    pos["knee.L"] = (float(kn[0]), float(kn[1]), float(kn[2]))
    return pos, h


def _guess_fingertips(co, h, shoulder, wrist, n):
    """Place n fingertip markers spread across the actual FINGER region, by
    measuring the hand's real reach from the mesh (the distance the hand extends
    past the wrist along the arm axis) instead of a fixed short fan. They land
    near the fingertips; the user nudges each onto its exact tip.
    (Reliable per-finger geometric detection on touching hands is unsolved by
    heuristics, so we give an adaptive, well-spread starting point.)"""
    wr = np.array(wrist, dtype=np.float64)
    sh = np.array(shoulder, dtype=np.float64)
    down = wr - sh
    nd = np.linalg.norm(down)
    down = down / nd if nd > 1e-6 else np.array([0.3, 0.0, -1.0])

    # spread axis: world X projected perpendicular to the arm (visible head-on)
    spread = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ down) * down
    ns = np.linalg.norm(spread)
    spread = spread / ns if ns > 1e-6 else np.array([1.0, 0.0, 0.0])

    # measure ONLY the hand: vertices just past the wrist AND within a small
    # radius of it (so legs / torso along the arm axis are never picked up).
    rel = co - wr
    dist = np.linalg.norm(rel, axis=1)
    proj = rel @ down                       # >0 means past the wrist (into hand)
    lat = rel @ spread
    near = (proj > 0.01 * h) & (dist < 0.22 * h)
    if near.sum() > 10:
        reach = float(np.percentile(proj[near], 94))   # wrist -> fingertip dist
        width = float(np.percentile(np.abs(lat[near]), 90))
    else:
        reach = 0.11 * h
        width = 0.05 * h
    reach = float(np.clip(reach, 0.06 * h, 0.18 * h))   # hard sane bounds
    width = float(np.clip(width, 0.02 * h, 0.06 * h))
    tip_d = reach * 0.92                     # sit just shy of the very tip
    half = width * 0.85

    names = FINGER_NAMES[:n]
    out = {}
    for i, nm in enumerate(names):
        f = (i / (n - 1) - 0.5) if n > 1 else 0.0     # -0.5 .. 0.5
        tip = wr + down * tip_d + spread * (f * 2 * half)
        out[nm] = (float(tip[0]), float(tip[1]), float(tip[2]))
    if "thumb" in out:                                # thumb: shorter & out to the side
        t = wr + down * (tip_d * 0.55) + spread * (half * 1.15)
        out["thumb"] = (float(t[0]), float(t[1]), float(t[2]))
    return out


def _geometric_fingertips(co, h, shoulder, wrist, n):
    """Detect the n fingertips DIRECTLY from the hand mesh, anchored at the
    (AI-detected) wrist: the most-distal mesh points of the hand, separated by
    non-maximum-suppression, then ordered thumb->pinky along the hand-spread
    axis. Accurate (~2-5 cm) for spread/open hands; returns None when the hand
    can't be resolved (e.g. a closed fist) so the caller falls back to the fan."""
    wr = np.array(wrist, dtype=np.float64)
    sh = np.array(shoulder, dtype=np.float64)
    down = wr - sh
    nd = np.linalg.norm(down)
    down = down / nd if nd > 1e-6 else np.array([0.3, 0.0, -1.0])
    rel = co - wr
    proj = rel @ down
    dist = np.linalg.norm(rel, axis=1)
    hand = (proj > -0.02 * h) & (dist < 0.16 * h)
    if hand.sum() < n * 3:
        return None
    H = co[hand]; Hd = dist[hand]
    # NMS: collect up to n+1 well-separated distal local maxima
    order = np.argsort(-Hd)
    tips = []
    supp = 0.020 * h
    for idx in order:
        pt = H[idx]
        if all(np.linalg.norm(pt - t) > supp for t in tips):
            tips.append(pt)
        if len(tips) >= n + 1:
            break
    if len(tips) < n:
        return None
    spread = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ down) * down
    ns = np.linalg.norm(spread)
    spread = spread / ns if ns > 1e-6 else np.array([1.0, 0.0, 0.0])
    # drop the most-redundant extra(s) until exactly n remain
    while len(tips) > n:
        nn = [min(np.linalg.norm(tips[i] - tips[j]) for j in range(len(tips)) if j != i)
              for i in range(len(tips))]
        tips.pop(int(np.argmin(nn)))
    tips = sorted(tips, key=lambda t: (t - wr) @ spread)
    names = FINGER_NAMES[:n]
    return {nm: (float(t[0]), float(t[1]), float(t[2])) for nm, t in zip(names, tips)}


def build_markers(props, heights=None):
    co = utils.read_world_coords(props.target_mesh)
    pos, h = _guess_positions(co, heights)
    size = 0.03 * h
    for nm in CENTER_MARKERS:
        o = _new_empty(nm); o.location = Vector(pos[nm]); _style(o, size, 'center')
    for nm in LEFT_MARKERS:
        o = _new_empty(nm); o.location = Vector(pos[nm]); _style(o, size, 'left')
        rn = nm.replace(".L", ".R")
        r = _new_empty(rn)
        p = pos[nm]; r.location = Vector((-p[0], p[1], p[2]))
        _style(r, size * 0.85, 'right')
        if props.mirror:
            _add_mirror_constraint(r, nm)
        else:
            for c in list(r.constraints):
                r.constraints.remove(c)
            r.lock_location = (False, False, False)
    return h


def _remove_fingertips():
    for fn in FINGER_NAMES:
        for s in (".L", ".R"):
            o = bpy.data.objects.get("ftip_%s%s" % (fn, s))
            if o:
                bpy.data.objects.remove(o, do_unlink=True)


def build_fingertip_markers(props):
    """Create fingertip markers on demand (after the user chooses how many)."""
    _remove_fingertips()
    if props.finger_count <= 0:
        return
    co = utils.read_world_coords(props.target_mesh)
    pos, h = _guess_positions(co)
    size = 0.03 * h
    sh = bpy.data.objects.get("shoulder.L")
    wr = bpy.data.objects.get("wrist.L")
    shoulder = tuple(sh.matrix_world.translation) if sh else pos["shoulder.L"]
    wrist = tuple(wr.matrix_world.translation) if wr else pos["wrist.L"]
    tips = _guess_fingertips(co, h, shoulder, wrist, props.finger_count)
    # PRIORITY: 1) geometric detection from the hand mesh (accurate, ~2-5 cm on
    # open hands), 2) AI detected tip if it lands sanely on the hand, 3) the fan.
    geo = None
    try:
        geo = _geometric_fingertips(co, h, shoulder, wrist, props.finger_count) or {}
    except Exception:
        geo = {}
    neural = {}
    try:
        if detect.available():
            neural = detect.detect_fingertips(props.target_mesh)
    except Exception:
        neural = {}
    wr = np.array(wrist, dtype=np.float64)
    gate = 0.28 * h
    n_geo = n_ai = 0
    for fn in FINGER_NAMES[:props.finger_count]:
        p = None
        if fn in geo:                                    # 1) mesh geometry (best)
            p = geo[fn]; n_geo += 1
        elif fn in neural:                               # 2) AI, sanity-gated
            v = np.array((float(neural[fn][0]), float(neural[fn][1]), float(neural[fn][2])))
            if np.linalg.norm(v - wr) <= gate:
                p = (float(v[0]), float(v[1]), float(v[2])); n_ai += 1
        if p is None:                                    # 3) geometric fan
            p = tips.get(fn)
        if p is None:
            continue
        ln = "ftip_%s.L" % fn
        o = _new_empty(ln); o.location = Vector(p); _style(o, size * 0.55, 'finger')
        rn = "ftip_%s.R" % fn
        r = _new_empty(rn); r.location = Vector((-p[0], p[1], p[2])); _style(r, size * 0.5, 'right')
        if props.mirror:
            _add_mirror_constraint(r, ln)
    return n_geo + n_ai


def set_front_view(context):
    try:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                # hide Blender's blue constraint/parent relationship lines
                try:
                    area.spaces.active.overlay.show_relationship_lines = False
                except Exception:
                    pass
                for region in area.regions:
                    if region.type == 'WINDOW':
                        with context.temp_override(area=area, region=region):
                            bpy.ops.view3d.view_axis(type='FRONT')
                return
    except Exception:
        pass


def lock_front_view(context, lock=True):
    """Snap every 3D viewport to FRONT ortho and lock its rotation so the user
    can't accidentally tumble away while placing markers (ARP-style). Panning and
    zooming stay free. Pass lock=False to release."""
    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        sp = area.spaces.active
        try:
            if lock:
                for region in area.regions:
                    if region.type == 'WINDOW':
                        with context.temp_override(area=area, region=region):
                            bpy.ops.view3d.view_axis(type='FRONT')
                sp.region_3d.lock_rotation = True
            else:
                sp.region_3d.lock_rotation = False
        except Exception:
            pass


def all_marker_names():
    names = list(CENTER_MARKERS)
    for n in LEFT_MARKERS + FOOT_MARKERS:
        names += [n, n.replace(".L", ".R")]
    for fn in FINGER_NAMES:
        names += ["ftip_%s.L" % fn, "ftip_%s.R" % fn]
    return names


def all_marker_objects():
    """Every marker empty currently in the scene (body + foot + fingertips +
    manual finger/palm/toe markers)."""
    objs = []
    for nm in all_marker_names():
        o = bpy.data.objects.get(nm)
        if o:
            objs.append(o)
    for o in bpy.data.objects:
        if o.name.startswith("fm."):
            objs.append(o)
    return objs


def set_markers_hidden(hide):
    for o in all_marker_objects():
        try:
            o.hide_set(bool(hide))
        except Exception:
            pass


def set_character_selectable(props, selectable):
    """Lock/unlock click-selection of the character mesh (+ its eyes/children) and
    the generated skeleton, so while editing markers the body can't be grabbed by
    mistake - only the markers stay clickable."""
    names = set()
    mesh = props.target_mesh if props else None
    if mesh:
        names.add(mesh.name)
        try:
            for ch in mesh.children_recursive:
                names.add(ch.name)
        except Exception:
            pass
    # also lock the garment + every mesh bound to our rig, so NONE of the
    # character can be grabbed by mistake while editing markers.
    sk = getattr(props, "skirt_object", None) if props else None
    if sk:
        names.add(sk.name)
    ko = getattr(props, "kandura_object", None) if props else None
    if ko:
        names.add(ko.name)
    try:
        rig = bpy.data.objects.get(utils.RIG_NAME) or bpy.data.objects.get("RIG-SR_Metarig")
        if rig is not None:
            for o in bpy.data.objects:
                if o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == rig
                                            for m in o.modifiers):
                    names.add(o.name)
    except Exception:
        pass
    for nm in names:
        o = bpy.data.objects.get(nm)
        if o:
            try:
                o.hide_select = (not selectable)
            except Exception:
                pass


# bone(head/tail) -> body marker, for "Sync Markers from Skeleton" (bones -> markers)
SYNC_MAP = {
    "spine_root": ("spine_01", "head"),
    "neck":       ("neck_01", "head"),
    "head_top":   ("head", "tail"),
    "shoulder.L": ("upper_arm.L", "head"),
    "elbow.L":    ("forearm.L", "head"),
    "wrist.L":    ("hand.L", "head"),
    "hip.L":      ("thigh.L", "head"),
    "knee.L":     ("shin.L", "head"),
    "ankle.L":    ("foot.L", "head"),
    "ball.L":     ("foot.L", "tail"),
    "foottip.L":  ("toe.L", "tail"),
}


# 3/4 perspective angle the user chose for the hand (quaternion)
HAND_VIEW_QUAT = (0.6563, 0.6293, 0.2881, 0.3005)


def set_hand_view_focus(context, focus_name="wrist.L", dist=None):
    """Zoom to the hand using the user's preferred 3/4 perspective angle, centred
    on the wrist, at a distance that scales with the character height."""
    o = bpy.data.objects.get(focus_name)
    target = o.matrix_world.translation if o else None
    if dist is None:
        try:
            co = utils.read_world_coords(context.scene.smartrig.target_mesh)
            dist = 0.45 * float(co[:, 2].max() - co[:, 2].min())
        except Exception:
            dist = 0.8
    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        sp = area.spaces.active
        try:
            sp.region_3d.view_perspective = 'PERSP'
            sp.region_3d.lock_rotation = False
            sp.region_3d.view_rotation = Quaternion(HAND_VIEW_QUAT)
            if target is not None:
                sp.region_3d.view_location = target
            sp.region_3d.view_distance = dist
        except Exception:
            pass
        return


def set_top_view_focus(context, focus_name="ankle.L", dist=None):
    """Top ortho view, centred & zoomed on the foot, so the user can drop the 2 foot
    markers precisely. Distance scales with the character height (any size)."""
    if dist is None:
        try:
            co = utils.read_world_coords(context.scene.smartrig.target_mesh)
            dist = 0.5 * float(co[:, 2].max() - co[:, 2].min())
        except Exception:
            dist = 0.7
    for area in context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        sp = area.spaces.active
        for region in area.regions:
            if region.type == 'WINDOW':
                with context.temp_override(area=area, region=region):
                    bpy.ops.view3d.view_axis(type='TOP')
        try:
            sp.region_3d.view_perspective = 'ORTHO'      # ortho so clicks map straight down
            sp.region_3d.lock_rotation = False
            o = bpy.data.objects.get(focus_name)
            if o:
                sp.region_3d.view_location = o.matrix_world.translation
                sp.region_3d.view_distance = dist
        except Exception:
            pass
        return


# --------------------------------------------------------------- operators
class SMARTRIG_OT_start(bpy.types.Operator):
    bl_idname = "smartrig.start"
    bl_label = "Start"
    bl_description = ("Place markers automatically and switch to front view. "
                      "Then select & drag any LEFT/center marker to fine-tune (right mirrors live).")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.smartrig.target_mesh is not None

    def execute(self, context):
        build_markers(context.scene.smartrig)
        set_front_view(context)
        self.report({'INFO'}, "Markers placed. Select & drag to adjust; right side mirrors. Then Build.")
        return {'FINISHED'}


def _volume_hit(mesh, origin, direction):
    """March a ray through the mesh; return the world midpoint of the FIRST solid the
    ray enters (entry + first exit). Using only the first part keeps the joint INSIDE
    it - first+last could land in the empty gap between two parts (e.g. between a
    finger and the thigh behind it)."""
    mi = mesh.matrix_world.inverted()
    o_l = mi @ origin
    d_l = (mi.to_3x3() @ direction).normalized()
    hits = []
    cur = o_l.copy()
    for _ in range(24):
        res, loc, nrm, idx = mesh.ray_cast(cur, d_l)
        if not res:
            break
        hits.append(loc.copy())
        cur = loc + d_l * 1e-4
    if not hits:
        return None
    mid = (hits[0] + hits[1]) * 0.5 if len(hits) >= 2 else hits[0]
    return mesh.matrix_world @ mid


GUIDE_SHORT = {"head_top": "HEAD", "neck": "NECK", "spine_root": "PELVIS",
               "shoulder.L": "SHOULDER", "elbow.L": "ELBOW", "wrist.L": "WRIST",
               "hip.L": "HIP", "knee.L": "KNEE", "ankle.L": "ANKLE"}


_GUIDE_PCOLL = None
_GUIDE_ICONS = []
_GUIDE_POS = None      # cached smart-start positions for the guided flow
_GUIDE_H = 1.0


def _create_one_marker(props, name, pos, h):
    """Create a single guide marker (+ its mirror if it's a .L) at its smart-start
    position. If it already exists it is left alone so the user's edits survive."""
    if pos is None or name not in pos:
        return
    size = 0.03 * h
    if bpy.data.objects.get(name) is None:
        role = 'center' if name in CENTER_MARKERS else 'left'
        o = _new_empty(name); o.location = Vector(pos[name]); _style(o, size, role)
    if name.endswith(".L"):
        rn = name.replace(".L", ".R")
        if bpy.data.objects.get(rn) is None:
            p = pos[name]
            r = _new_empty(rn); r.location = Vector((-p[0], p[1], p[2]))
            _style(r, size * 0.85, 'right')
            if props.mirror:
                _add_mirror_constraint(r, name)


def _guide_create_step(context, step):
    """Make the marker for this step appear (one-by-one reveal)."""
    if _GUIDE_POS is None:
        return
    _create_one_marker(context.scene.smartrig, GUIDE_SEQUENCE[step][0], _GUIDE_POS, _GUIDE_H)


def guide_icon(i):
    return _GUIDE_ICONS[i] if 0 <= i < len(_GUIDE_ICONS) else 0


def _guide_clear_previews():
    global _GUIDE_PCOLL, _GUIDE_ICONS
    _GUIDE_ICONS = []
    if _GUIDE_PCOLL is not None:
        try:
            bpy.utils.previews.remove(_GUIDE_PCOLL)
        except Exception:
            pass
        _GUIDE_PCOLL = None


GUIDE_SEQUENCE = [
    ("head_top", "TOP OF HEAD"), ("neck", "NECK"),
    ("shoulder.L", "LEFT SHOULDER"), ("elbow.L", "LEFT ELBOW"), ("wrist.L", "LEFT WRIST"),
    ("spine_root", "PELVIS / SPINE ROOT"),
    ("hip.L", "LEFT HIP"), ("knee.L", "LEFT KNEE"), ("ankle.L", "LEFT ANKLE"),
]


def _step_label(name, i, total):
    return "ADD %s   %d/%d" % (GUIDE_SHORT.get(name, name.upper()), i + 1, total)


def _guide_select(context, step):
    """Select (and make active) the marker for this guide step, so the user can
    grab it (G) immediately and drag it onto the joint shown in the panel."""
    name = GUIDE_SEQUENCE[step][0]
    try:
        for o in context.selected_objects:
            o.select_set(False)
    except Exception:
        pass
    o = bpy.data.objects.get(name)
    if o:
        try:
            o.select_set(True)
            context.view_layer.objects.active = o
        except Exception:
            pass
    for a in context.window.screen.areas:
        if a.type == 'VIEW_3D':
            a.tag_redraw()


# Correct marker spots on the bundled reference picture (assets/marker_reference.png),
# as normalised (u, v) with v measured from the TOP. Detected from the artwork itself.
GUIDE_REF_UV = {
    "head_top":   (0.500, 0.056),
    "neck":       (0.500, 0.185),
    "shoulder.L": (0.596, 0.226),
    "elbow.L":    (0.675, 0.365),
    "wrist.L":    (0.746, 0.488),
    "spine_root": (0.500, 0.479),
    "hip.L":      (0.539, 0.479),
    "knee.L":     (0.568, 0.720),
    "ankle.L":    (0.582, 0.922),
}


def _build_guide(context, props, sequence):
    """Crop the bundled REFERENCE picture (correct marker spots) into one zoomed
    square per joint and load them into a preview collection, so the SmartRig PANEL
    shows 'ADD HEAD' + that exact picture - ARP style. Robust: on failure the panel
    simply shows text and click-placement still works."""
    global _GUIDE_PCOLL, _GUIDE_ICONS
    from bpy.utils import previews as _pv
    import tempfile, os
    import numpy as np
    _GUIDE_ICONS = []
    try:
        ref = os.path.join(os.path.dirname(__file__), "assets", "marker_reference.png")
        full = bpy.data.images.load(ref, check_existing=True)
        Wf, Hf = full.size[0], full.size[1]
        arr = np.array(full.pixels[:], dtype=np.float32).reshape(Hf, Wf, 4)[::-1]   # top-down
        try:
            bpy.data.images.remove(full)
        except Exception:
            pass
        if _GUIDE_PCOLL is None:
            _GUIDE_PCOLL = _pv.new()
        else:
            _GUIDE_PCOLL.clear()
        crop = int(min(Wf, Hf) * 0.26)
        tmpd = tempfile.gettempdir()
        for i, (name, _lab) in enumerate(sequence):
            uc, vc = GUIDE_REF_UV.get(name, (0.5, 0.5))
            cxp = int(uc * Wf); cyp = int(vc * Hf)       # vc already measured from top
            x0 = min(max(cxp - crop // 2, 0), Wf - crop)
            y0 = min(max(cyp - crop // 2, 0), Hf - crop)
            sub = arr[y0:y0 + crop, x0:x0 + crop, :].copy()
            # bake ONLY this joint's dot (base image is clean - no other dots)
            ddx, ddy = cxp - x0, cyp - y0
            yy, xx = np.ogrid[:crop, :crop]
            d2 = (xx - ddx) ** 2 + (yy - ddy) ** 2
            rr = max(5, crop // 30)
            glow = (d2 <= (rr + 8) ** 2) & (d2 > rr ** 2)
            sub[glow] = sub[glow] * 0.30 + np.array([0.2, 0.9, 1.0, 1.0], np.float32) * 0.70
            sub[d2 <= rr ** 2] = np.array([0.85, 0.98, 1.0, 1.0], np.float32)
            ci = bpy.data.images.new("sr_crop_tmp", crop, crop, alpha=True)
            ci.pixels = sub[::-1].ravel()
            cpath = os.path.join(tmpd, "sr_guide_%d.png" % i)
            ci.filepath_raw = cpath; ci.file_format = 'PNG'; ci.save()
            bpy.data.images.remove(ci)
            pvw = _GUIDE_PCOLL.load("step%d" % i, cpath, 'IMAGE')
            _GUIDE_ICONS.append(pvw.icon_id)
        props.guide_total = len(sequence)
        props.guide_step = 0
        props.guide_label = _step_label(sequence[0][0], 0, len(sequence))
        props.guide_request = ""
        props.guide_active = True
    except Exception as e:
        props.guide_active = False
        print("SmartRig: guide images unavailable:", e)


def place_marker(props, name, loc, h):
    """Create/move one marker (+ its .R mirror). Center markers snap to X=0."""
    size = 0.03 * h
    role = 'center' if name in CENTER_MARKERS else 'left'
    o = bpy.data.objects.get(name) or _new_empty(name)
    if role == 'center':
        loc = Vector((0.0, loc.y, loc.z))
        o.lock_location = (True, False, False)
    o.location = loc
    _style(o, size, role)
    if name.endswith(".L"):
        rn = name.replace(".L", ".R")
        r = bpy.data.objects.get(rn) or _new_empty(rn)
        r.location = Vector((-loc.x, loc.y, loc.z))
        _style(r, size * 0.85, 'right')
        if props.mirror:
            _add_mirror_constraint(r, name)


def _guide_goto(context, step):
    """Move the guided flow to a step and update the panel label."""
    p = context.scene.smartrig
    step = max(0, min(step, p.guide_total - 1))
    p.guide_step = step
    p.guide_label = _step_label(GUIDE_SEQUENCE[step][0], step, p.guide_total)
    for a in context.window.screen.areas:
        a.tag_redraw()


def _guide_finish(context):
    """Release the guided state (keep the markers for Build)."""
    p = context.scene.smartrig
    p.guide_active = False
    p.guide_request = ""
    p.placing = False
    _guide_clear_previews()
    lock_front_view(context, False)
    for a in context.window.screen.areas:
        a.tag_redraw()


# ---- shared continuous click-placement (used by Let's Rig and the Resume button) --
def _placement_start(op, context):
    p = context.scene.smartrig
    op.mesh = p.target_mesh
    co = utils.read_world_coords(op.mesh)
    op._h = float(co[:, 2].max() - co[:, 2].min())
    op.area = context.area
    op._entered = False
    try:
        context.window.cursor_modal_set('CROSSHAIR')
    except Exception:
        pass
    p.placing = True
    for a in context.window.screen.areas:
        a.tag_redraw()
    context.window_manager.modal_handler_add(op)
    return {'RUNNING_MODAL'}


def _placement_paused(op, context):
    """Pause clicking and hand control back to the panel buttons."""
    try:
        context.window.cursor_modal_restore()
    except Exception:
        pass
    context.scene.smartrig.placing = False
    for a in context.window.screen.areas:
        a.tag_redraw()


def _over_panel(op, event):
    try:
        ui = next((r for r in op.area.regions if r.type == 'UI'), None)
        return ui is not None and ui.width > 1 and event.mouse_x >= ui.x
    except Exception:
        return False


def _placement_modal(op, context, event):
    p = context.scene.smartrig
    if not p.guide_active:                       # cancelled from the panel
        _placement_paused(op, context)
        return {'CANCELLED'}
    # PAUSE as soon as the cursor reaches the panel, so its buttons work
    if event.type == 'MOUSEMOVE':
        if _over_panel(op, event):
            if op._entered:
                _placement_paused(op, context)
                return {'FINISHED'}
        else:
            op._entered = True
        return {'RUNNING_MODAL'}
    if event.type == 'ESC' and event.value == 'PRESS':
        _placement_paused(op, context)
        return {'FINISHED'}
    if event.type in {'BACK_SPACE', 'DEL'} and event.value == 'PRESS':
        step = max(0, p.guide_step - 1)
        name = GUIDE_SEQUENCE[step][0]
        for n in (name, name.replace(".L", ".R") if name.endswith(".L") else None):
            if n:
                o = bpy.data.objects.get(n)
                if o:
                    bpy.data.objects.remove(o, do_unlink=True)
        _guide_goto(context, step)
        return {'RUNNING_MODAL'}
    if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
        region = context.region
        rv3d = context.region_data
        if region is None or rv3d is None or region.type != 'WINDOW':
            return {'PASS_THROUGH'}
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        hit = _volume_hit(op.mesh, origin, direction)
        if hit is None:
            op.report({'WARNING'}, "Click ON the body.")
            return {'RUNNING_MODAL'}
        step = p.guide_step
        name = GUIDE_SEQUENCE[step][0]
        place_marker(p, name, hit, op._h)
        if step < p.guide_total - 1:
            _guide_goto(context, step + 1)            # advance, keep clicking
            return {'RUNNING_MODAL'}
        op.report({'INFO'}, "All joints placed. Move to the panel and press Build Skeleton.")
        _placement_paused(op, context)
        return {'FINISHED'}
    return {'PASS_THROUGH'}


class SMARTRIG_OT_place_guided(bpy.types.Operator):
    bl_idname = "smartrig.place_guided"
    bl_label = "Let's Rig"
    bl_description = ("Start guided rigging. The Soulify panel then shows each joint "
                      "with a picture: press 'Place this joint' and click it on the body. "
                      "Everything is driven from the panel - head, neck, shoulder, ...")
    bl_options = {'REGISTER'}

    SEQUENCE = [
        ("head_top", "TOP OF HEAD"), ("neck", "NECK"),
        ("shoulder.L", "LEFT SHOULDER"), ("elbow.L", "LEFT ELBOW"), ("wrist.L", "LEFT WRIST"),
        ("spine_root", "PELVIS / SPINE ROOT"),
        ("hip.L", "LEFT HIP"), ("knee.L", "LEFT KNEE"), ("ankle.L", "LEFT ANKLE"),
    ]

    def invoke(self, context, event):
        props = context.scene.smartrig
        # use whatever character the user has SELECTED (no Mesh field needed)
        mesh = None
        ao = context.active_object
        if ao is not None and ao.type == 'MESH':
            mesh = ao
        else:
            sel = [o for o in context.selected_objects if o.type == 'MESH']
            mesh = sel[0] if sel else props.target_mesh
        if mesh is None or mesh.type != 'MESH':
            self.report({'ERROR'}, "Please select your character (a mesh) first, then press Let's Rig.")
            return {'CANCELLED'}
        props.target_mesh = mesh
        # start fresh - the panel drives everything from here (no viewport overlay)
        for nm in all_marker_names():
            o = bpy.data.objects.get(nm)
            if o:
                bpy.data.objects.remove(o, do_unlink=True)
        set_front_view(context)
        lock_front_view(context, True)                     # lock the view to FRONT
        _build_guide(context, props, self.SEQUENCE)        # per-joint pictures in the panel
        props.guide_active = True
        props.hands_decided = False
        props.want_hands = False
        props.lock_mesh = True
        props.guide_step = 0
        props.guide_request = ""
        props.guide_label = _step_label(self.SEQUENCE[0][0], 0, len(self.SEQUENCE))
        self.report({'INFO'}, "Click each joint on the body (head, neck, ...). Move to the panel to pause.")
        return _placement_start(self, context)             # click placement is live immediately

    def modal(self, context, event):
        return _placement_modal(self, context, event)


class SMARTRIG_OT_guide_place(bpy.types.Operator):
    """Continuous click placement: click each joint on the body one after another
    (head, neck, shoulder, ...). The right side mirrors live. The moment the cursor
    moves onto the SmartRig panel, this PAUSES so the panel's Back / Cancel buttons
    work; press 'Start clicking' to resume. Esc also pauses."""
    bl_idname = "smartrig.guide_place"
    bl_label = "Start clicking"
    bl_description = "Click each joint on the body in turn. Move onto the panel to pause"

    @classmethod
    def poll(cls, context):
        p = context.scene.smartrig
        return p.guide_active and p.target_mesh is not None

    def invoke(self, context, event):
        return _placement_start(self, context)

    def modal(self, context, event):
        return _placement_modal(self, context, event)


class SMARTRIG_OT_auto_detect(bpy.types.Operator):
    bl_idname = "smartrig.auto_detect"
    bl_label = "AI Detect"
    bl_description = ("Use the trained neural model to read this character's joint "
                      "proportions, then place markers automatically. Falls back to "
                      "pure geometry if the model isn't available.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.smartrig.target_mesh is not None

    def execute(self, context):
        props = context.scene.smartrig
        heights = None
        conf = 0.0
        if detect.available():
            try:
                heights, conf = detect.detect_height_fractions(props.target_mesh)
            except Exception as e:
                self.report({'WARNING'}, "AI detect failed (%s); using geometry." % e)
                heights = None
        elif not detect.has_model():
            self.report({'WARNING'}, "No AI model bundled; using geometry.")
        elif not detect.has_runtime():
            self.report({'WARNING'}, "onnxruntime not installed in Blender's Python; using geometry.")

        build_markers(props, heights)
        set_front_view(context)
        if heights:
            self.report({'INFO'}, "AI detected proportions (conf %.0f%%): %d joint heights adapted. "
                                  "Drag any marker to fine-tune." % (conf * 100, len(heights)))
        else:
            self.report({'INFO'}, "Markers placed geometrically. Drag any marker to fine-tune.")
        return {'FINISHED'}


class SMARTRIG_OT_add_fingers(bpy.types.Operator):
    bl_idname = "smartrig.add_fingers"
    bl_label = "Add Fingers"
    bl_description = ("Choose how many fingers this character has, then create draggable "
                     "fingertip markers. Works for 3, 4 or 5-finger hands.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("wrist.L") is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, context):
        col = self.layout.column()
        col.label(text="How many fingers does this character have?")
        col.prop(context.scene.smartrig, "finger_count", text="Fingers per hand")
        col.label(text="Tip: 5 = human, 3-4 = stylized/cartoon", icon='INFO')

    def execute(self, context):
        n_ai = build_fingertip_markers(context.scene.smartrig) or 0
        n = context.scene.smartrig.finger_count
        if n > 0 and n_ai:
            self.report({'INFO'}, "Added %d fingertip markers (%d placed by AI). Nudge any onto the exact tip." % (n, n_ai))
        elif n > 0:
            self.report({'INFO'}, "Added %d fingertip markers. Drag them onto the fingertips." % n)
        else:
            self.report({'INFO'}, "Fingers removed.")
        return {'FINISHED'}


def full_cleanup(context):
    """Remove EVERYTHING Soulify created: markers, metarig, generated rigs,
    skirt bones/collision, widgets, and UNBIND our meshes (DEF/SR_Skirt groups +
    armature modifiers). Never touches non-Soulify rigs (e.g. Auto-Rig Pro)."""
    p = context.scene.smartrig
    try:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    p.lock_mesh = False

    def _is_ours(o):
        return o is not None and o.type == 'ARMATURE' and (
            o.name == "SR_Metarig" or o.name.startswith("RIG-SR_Metarig")
            or o.name in (utils.REF_NAME, utils.RIG_NAME))

    arm_objs = set(o for o in bpy.data.objects if _is_ours(o))

    # unbind meshes bound to our armatures (keep the meshes themselves)
    our_meshes = set()
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        for m in list(o.modifiers):
            if m.type == 'ARMATURE' and m.object in arm_objs:
                o.modifiers.remove(m); our_meshes.add(o)
        if o.parent in arm_objs:
            mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
            our_meshes.add(o)
    for o in our_meshes:
        for vg in list(o.vertex_groups):
            if vg.name.startswith("DEF-") or vg.name == "SR_Skirt":
                o.vertex_groups.remove(vg)

    # remove markers + collection
    for o in list(all_marker_objects()):
        bpy.data.objects.remove(o, do_unlink=True)
    c = bpy.data.collections.get(utils.MARKERS_COLL)
    if c:
        try:
            bpy.data.collections.remove(c)
        except Exception:
            pass

    # remove our armatures + their data
    for o in list(arm_objs):
        ad = o.data
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            pass
        if ad and ad.users == 0:
            try:
                bpy.data.armatures.remove(ad)
            except Exception:
                pass

    # purge any ORPHAN armature datablocks we left behind (e.g. SR_Metarig.001 from
    # repeated builds) - their objects are gone but the data lingers with 0 users.
    for a in list(bpy.data.armatures):
        if a.users == 0 and ("SR_Metarig" in a.name or a.name.startswith("RIG-SR")
                             or a.name in (utils.REF_NAME, utils.RIG_NAME)):
            try:
                bpy.data.armatures.remove(a)
            except Exception:
                pass

    # remove ALL widget objects/meshes our rigs created. Rigify keeps them linked
    # in a WGTS_* collection, so users is never 0 -> match by NAME, not users.
    # Prefixes are tied to OUR rig names only (never ARP's "WGT-rig_*" etc.).
    wgt_prefixes = ("WGT-RIG-SR_Metarig", "WGT-SR_Metarig", "WGT-SR_Rig",
                    "WGT-SKC", "WGT-SK_", "WGT-SK", "WGT-" + utils.RIG_NAME,
                    "WGT-" + utils.REF_NAME)
    for o in list(bpy.data.objects):
        nm = o.name
        if nm.startswith("WGT-") and (nm.startswith(wgt_prefixes)
                                      or "SR_Metarig" in nm or "SKC" in nm
                                      or "SK_Master" in nm):
            md = o.data
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
            if md and getattr(md, "users", 1) == 0:
                try:
                    bpy.data.meshes.remove(md)
                except Exception:
                    pass
    # remove the now-empty Rigify widget collections we made (WGTS_SmartRig,
    # WGTS_SR_Metarig, WGTS_RIG-SR_Metarig...). Leave non-SmartRig ones alone.
    for c in list(bpy.data.collections):
        cn = c.name
        if cn.startswith("WGTS_") and ("SmartRig" in cn or "SR_Metarig" in cn
                                       or "SR_Rig" in cn):
            if len(c.objects) == 0:
                try:
                    bpy.data.collections.remove(c)
                except Exception:
                    pass

    # reset state flags
    for attr, val in (("guide_active", False), ("markers_hidden", False),
                      ("rig_generated", False), ("hands_decided", False),
                      ("want_hands", False), ("finger_placing", False),
                      ("placing", False), ("rig_started", False),
                      ("mode_chosen", False)):
        try:
            setattr(p, attr, val)
        except Exception:
            pass


class SMARTRIG_OT_reset(bpy.types.Operator):
    bl_idname = "smartrig.reset"
    bl_label = "Reset"
    bl_description = "Delete markers and any generated skeleton/rig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        full_cleanup(context)
        self.report({'INFO'}, "Everything cleaned: markers, rig, skirt and binding removed.")
        return {'FINISHED'}


class SMARTRIG_OT_guide_back(bpy.types.Operator):
    bl_idname = "smartrig.guide_back"
    bl_label = "Back"
    bl_description = "Step back one joint and remove its marker so you can place it again"

    @classmethod
    def poll(cls, context):
        return context.scene.smartrig.guide_active

    def execute(self, context):
        p = context.scene.smartrig
        step = max(0, p.guide_step - 1)
        name = GUIDE_SEQUENCE[step][0]
        for n in (name, name.replace(".L", ".R") if name.endswith(".L") else None):
            if n:
                o = bpy.data.objects.get(n)
                if o:
                    bpy.data.objects.remove(o, do_unlink=True)
        _guide_goto(context, step)
        return {'FINISHED'}


class SMARTRIG_OT_guide_cancel(bpy.types.Operator):
    bl_idname = "smartrig.guide_cancel"
    bl_label = "Cancel"
    bl_description = "Cancel guided rigging and remove all markers"

    def execute(self, context):
        full_cleanup(context)
        try:
            _guide_finish(context)
        except Exception:
            pass
        self.report({'INFO'}, "Cancelled. Everything cleaned.")
        return {'FINISHED'}


class SMARTRIG_OT_pick_mode(bpy.types.Operator):
    bl_idname = "smartrig.pick_mode"
    bl_label = "Choose Rig Mode"
    bl_description = "Choose what to rig (Character or Parts), then show the tools"
    bl_options = {'INTERNAL'}
    mode: bpy.props.StringProperty(default='CHARACTER')

    def execute(self, context):
        p = context.scene.smartrig
        p.rig_mode = self.mode if self.mode in ('CHARACTER', 'PARTS') else 'CHARACTER'
        p.mode_chosen = True
        for a in getattr(context.window.screen, "areas", []):
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_guide_done(bpy.types.Operator):
    bl_idname = "smartrig.guide_done"
    bl_label = "Done & Build"
    bl_description = "Finish placing markers AND build the skeleton in one step"

    def execute(self, context):
        _guide_finish(context)
        # Done builds the skeleton straight away - no separate Build click needed
        try:
            bpy.ops.smartrig.go('INVOKE_DEFAULT')
        except Exception as e:
            self.report({'WARNING'}, "Markers kept, but build failed: %s" % e)
        return {'FINISHED'}



class SMARTRIG_OT_edit_markers(bpy.types.Operator):
    bl_idname = "smartrig.edit_markers"
    bl_label = "Edit Markers"
    bl_description = ("Show the markers and select them so you can grab (G) and drag them. "
                      "Turn on Live Link to have the skeleton follow as you edit.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.smartrig
        # leave edit mode if we're tweaking the skeleton
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        set_markers_hidden(False)
        p.markers_hidden = False
        objs = all_marker_objects()
        try:
            for o in context.selected_objects:
                o.select_set(False)
            for o in objs:                      # show, make pickable, and SELECT them
                o.hide_select = False
                o.hide_viewport = False
                o.select_set(True)
            if objs:
                context.view_layer.objects.active = objs[0]
        except Exception:
            pass
        p.lock_mesh = True                   # lock the body so it can't be grabbed by mistake
        for a in context.window.screen.areas:
            a.tag_redraw()
        self.report({'INFO'}, "Editing markers: body locked. Click a marker (yellow ring = selected), grab (G) to move.")
        return {'FINISHED'}


class SMARTRIG_OT_toggle_markers(bpy.types.Operator):
    bl_idname = "smartrig.toggle_markers"
    bl_label = "Show / Hide Markers"
    bl_description = "Show or hide all the placement markers (they auto-hide when you Build)"

    def execute(self, context):
        p = context.scene.smartrig
        new_hidden = not p.markers_hidden
        set_markers_hidden(new_hidden)
        p.markers_hidden = new_hidden
        for a in context.window.screen.areas:
            a.tag_redraw()
        self.report({'INFO'}, "Markers hidden." if new_hidden else "Markers shown.")
        return {'FINISHED'}


class SMARTRIG_OT_reset_markers(bpy.types.Operator):
    bl_idname = "smartrig.reset_markers"
    bl_label = "Reset Markers to Default"
    bl_description = ("Put the body markers back to their automatic geometric positions "
                      "(undo all your manual nudges). Finger/toe markers are kept.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.smartrig.target_mesh is not None

    def execute(self, context):
        props = context.scene.smartrig
        build_markers(props)                 # re-derive body markers at default spots
        set_markers_hidden(False)
        props.markers_hidden = False
        for a in context.window.screen.areas:
            a.tag_redraw()
        self.report({'INFO'}, "Body markers reset to default positions.")
        return {'FINISHED'}


class SMARTRIG_OT_sync_from_skeleton(bpy.types.Operator):
    bl_idname = "smartrig.sync_from_skeleton"
    bl_label = "Sync Markers from Skeleton"
    bl_description = ("Pull the markers back onto the current skeleton bones, so manual "
                      "bone edits flow back to the markers (the reverse of Build).")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(utils.REF_NAME) is not None

    def execute(self, context):
        props = context.scene.smartrig
        arm = bpy.data.objects.get(utils.REF_NAME)
        if arm is None:
            self.report({'ERROR'}, "No skeleton (build it first).")
            return {'CANCELLED'}
        co = utils.read_world_coords(props.target_mesh) if props.target_mesh else None
        h = float(co[:, 2].max() - co[:, 2].min()) if co is not None else 1.0
        mw = arm.matrix_world
        n = 0
        for mk_name, (bone_name, end) in SYNC_MAP.items():
            b = arm.data.bones.get(bone_name)
            if b is None:
                continue
            loc = mw @ (b.head_local if end == "head" else b.tail_local)
            place_marker(props, mk_name, loc, h)
            n += 1
        set_markers_hidden(False)
        props.markers_hidden = False
        for a in context.window.screen.areas:
            a.tag_redraw()
        self.report({'INFO'}, "Synced %d markers from the skeleton." % n)
        return {'FINISHED'}


def metarig_sync_map(arm):
    """marker_name -> (bone_name, 'head'|'tail') for the live SR_Metarig. The
    spine/neck/head bones are resolved DYNAMICALLY (their count can vary)."""
    import re as _re
    sp_re = _re.compile(r"^spine(\.\d+)?$")
    spine = sorted([b.name for b in arm.data.bones if sp_re.match(b.name)],
                   key=lambda x: 0 if x == "spine" else int(x.split(".")[1]))
    m = {}
    if spine:
        m["spine_root"] = (spine[0], "head")
        m["head_top"] = (spine[-1], "tail")
        nf = next((n for n in spine
                   if arm.pose.bones.get(n) and
                   arm.pose.bones[n].rigify_type == 'spines.super_head'), None)
        if nf:
            m["neck"] = (nf, "head")
    limb = {"shoulder": ("upper_arm", "head"), "elbow": ("forearm", "head"),
            "wrist": ("hand", "head"), "hip": ("thigh", "head"),
            "knee": ("shin", "head"), "ankle": ("foot", "head"),
            "ball": ("foot", "tail"), "foottip": ("toe", "tail")}
    for mk, (bone, end) in limb.items():
        for s in (".L", ".R"):
            m[mk + s] = (bone + s, end)
    return m


def bones_to_markers(props, arm):
    """Push the metarig's current bone endpoints OUT to their markers
    (bone -> marker). Reads edit_bones while the metarig is in Edit Mode so live
    bone edits propagate immediately. Returns how many markers were updated."""
    co = utils.read_world_coords(props.target_mesh) if props.target_mesh else None
    h = float(co[:, 2].max() - co[:, 2].min()) if co is not None else 1.0
    mw = arm.matrix_world
    edit = (bpy.context.mode == 'EDIT_ARMATURE' and bpy.context.edit_object == arm)
    src = arm.data.edit_bones if edit else arm.data.bones
    n = 0
    for mk, (bone, end) in metarig_sync_map(arm).items():
        b = src.get(bone)
        if b is None:
            continue
        if edit:
            loc = mw @ (b.head.copy() if end == "head" else b.tail.copy())
        else:
            loc = mw @ (b.head_local.copy() if end == "head" else b.tail_local.copy())
        place_marker(props, mk, loc, h)
        n += 1
    return n


# ---- Live Link: a PERSISTENT timer (survives addon reloads & file loads), so
# it can never silently stop the way a modal operator does. It polls cheaply and
# only syncs when scene.smartrig.live_link is ON. ----
_LL = {"mk_applied": None, "bn_applied": None, "mk_last": None, "bn_last": None,
       "changed_at": 0.0, "applied": True, "busy": False}


def _ll_marker_sig():
    vals = []
    for nm in all_marker_names():
        o = bpy.data.objects.get(nm)
        if o:
            vals.append((nm, tuple(round(c, 5) for c in o.location)))
    for o in bpy.data.objects:
        if o.name.startswith("fm.") and ".L." in o.name:
            vals.append((o.name, tuple(round(c, 5) for c in o.location)))
    return tuple(sorted(vals))


def _ll_bone_sig():
    arm = bpy.data.objects.get('SR_Metarig')
    if arm is None:
        return ()
    edit = (bpy.context.mode == 'EDIT_ARMATURE' and bpy.context.edit_object == arm)
    src = arm.data.edit_bones if edit else arm.data.bones
    vals = []
    for b in src:
        hv, tv = (b.head, b.tail) if edit else (b.head_local, b.tail_local)
        vals.append((b.name, tuple(round(c, 5) for c in hv) +
                     tuple(round(c, 5) for c in tv)))
    return tuple(sorted(vals))


def _ll_reset_baseline():
    _LL["mk_applied"] = _LL["mk_last"] = _ll_marker_sig()
    _LL["bn_applied"] = _LL["bn_last"] = _ll_bone_sig()
    _LL["applied"] = True


def _live_link_tick():
    """Persistent bpy.app.timer. Returns the next poll interval (seconds)."""
    import time
    interval = 0.2
    try:
        scn = getattr(bpy.context, "scene", None)
        p = getattr(scn, "smartrig", None) if scn else None
        if p is None or not getattr(p, "live_link", False):
            _LL["applied"] = True
            return 0.4
        if _LL["busy"] or p.placing or p.finger_placing or p.guide_active:
            return interval
        arm = bpy.data.objects.get('SR_Metarig')
        ref = bpy.data.objects.get(utils.REF_NAME)
        if arm is None and ref is None:
            return interval
        if bpy.context.mode not in ('OBJECT', 'EDIT_ARMATURE'):
            return interval
        if _LL["mk_applied"] is None:
            _ll_reset_baseline()
            return interval
        mk = _ll_marker_sig()
        bn = _ll_bone_sig()
        now = time.time()
        if mk != _LL["mk_last"] or bn != _LL["bn_last"]:
            _LL["mk_last"], _LL["bn_last"] = mk, bn
            _LL["changed_at"] = now
            _LL["applied"] = False
            return interval
        if _LL["applied"] or (now - _LL["changed_at"]) <= 0.25:
            return interval
        # settled: who moved? propagate the OTHER way.
        _LL["applied"] = True
        mk_changed = (mk != _LL["mk_applied"])
        bn_changed = (bn != _LL["bn_applied"])
        _LL["busy"] = True
        try:
            if arm is not None and bn_changed and \
                    (not mk_changed or bpy.context.mode == 'EDIT_ARMATURE'):
                bones_to_markers(p, arm)            # BONE is master
            elif mk_changed:
                if arm is not None:
                    from . import metarig
                    metarig.refit_metarig(p)        # MARKER is master
                elif ref is not None:
                    from . import fit
                    fit.build_reference(p)
        finally:
            _LL["busy"] = False
        _ll_reset_baseline()
    except Exception as e:
        print("SmartRig live link tick:", e)
    return interval


def _ensure_live_timer():
    try:
        if not bpy.app.timers.is_registered(_live_link_tick):
            bpy.app.timers.register(_live_link_tick, first_interval=0.3, persistent=True)
    except Exception as e:
        print("SmartRig live timer register failed:", e)


class SMARTRIG_OT_live_link(bpy.types.Operator):
    bl_idname = "smartrig.live_link"
    bl_label = "Live Link"
    bl_description = ("Toggle two-way live sync: drag a MARKER and the bone follows; "
                      "edit a BONE (Edit Mode) and its marker follows. Markers always "
                      "track the bones, so Re-fit never loses your edits.")

    def execute(self, context):
        p = context.scene.smartrig
        p.live_link = not p.live_link
        if p.live_link:
            _ll_reset_baseline()
            _ensure_live_timer()
            self.report({'INFO'}, "Live Link ON - markers <-> bones sync both ways.")
        else:
            self.report({'INFO'}, "Live Link OFF.")
        for a in (context.window.screen.areas if context.window else []):
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_choose_hands(bpy.types.Operator):
    bl_idname = "smartrig.choose_hands"
    bl_label = "Rig Hands?"
    bl_description = "Choose whether to rig the hands (palm + fingers) after the feet"
    do_hands: bpy.props.BoolProperty(default=True)

    def execute(self, context):
        p = context.scene.smartrig
        p.want_hands = self.do_hands
        p.hands_decided = True
        if self.do_hands:
            p.finger_part = "palm"
            try:
                set_hand_view_focus(context, "wrist.L")
            except Exception:
                pass
        for a in context.window.screen.areas:
            a.tag_redraw()
        return {'FINISHED'}


class SMARTRIG_OT_back_to_markers(bpy.types.Operator):
    bl_idname = "smartrig.back_to_markers"
    bl_label = "Back to Markers"
    bl_description = "Show the markers again so you can keep editing them (hides the metarig/skeleton)"

    def execute(self, context):
        p = context.scene.smartrig
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        set_markers_hidden(False)
        p.markers_hidden = False
        for nm in ("SR_Metarig", utils.REF_NAME, utils.RIG_NAME):
            o = bpy.data.objects.get(nm)
            if o is not None:
                try:
                    o.hide_set(True)
                except Exception:
                    pass
        p.lock_mesh = True
        for a in context.window.screen.areas:
            a.tag_redraw()
        self.report({'INFO'}, "Markers shown - edit them, then rebuild.")
        return {'FINISHED'}


classes = (SMARTRIG_OT_start, SMARTRIG_OT_auto_detect, SMARTRIG_OT_place_guided,
           SMARTRIG_OT_guide_place, SMARTRIG_OT_guide_back,
           SMARTRIG_OT_guide_done, SMARTRIG_OT_guide_cancel, SMARTRIG_OT_pick_mode,
           SMARTRIG_OT_add_fingers, SMARTRIG_OT_reset,
           SMARTRIG_OT_toggle_markers, SMARTRIG_OT_reset_markers,
           SMARTRIG_OT_sync_from_skeleton, SMARTRIG_OT_live_link,
           SMARTRIG_OT_edit_markers, SMARTRIG_OT_choose_hands,
           SMARTRIG_OT_back_to_markers)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    _ensure_live_timer()         # persistent Live Link poller (acts only when ON)


def unregister():
    _guide_clear_previews()
    try:
        if bpy.app.timers.is_registered(_live_link_tick):
            bpy.app.timers.unregister(_live_link_tick)
    except Exception:
        pass
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
