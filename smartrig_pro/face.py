"""Soulify Face System — foundation (Storm-course architecture, FaceIt-style UX).

Roadmap (design doc section 5): modular face slots, each region a recipe.
This module ships the FOUNDATION:
  * geometric auto-detection of the face landmarks (FaceIt-style: automatic
    first, the user adjusts the markers before building),
  * the initial bones of the Blender Studio "Advanced Facial Rigging" course
    chapter 2 (jaw pivot -> chin, master-mouth placeholder),
  * analytic base mask weights (jaw carved out of the head weights) verified
    numerically: nothing outside the head moves, L/R symmetry exact.

Regions to follow on this base: lips (ribbon + zipper), eyes (auto-blink +
follow), brows/cheeks (shape keys + weight split), teeth/tongue, correctives,
then the expression / ARKit-52 / viseme library generated FROM this rig.

Conventions (Storm): DEF- deform, CTL- control, MCH- mechanism, master-*.
Front of the character faces -Y (same assumption as the body pipeline).
"""

import bpy
import numpy as np
from mathutils import Vector

from . import utils

FACE_COLL = "SR_FaceMarkers"
FACE_RIG_NAME = "SR_FaceRig"
GRID_NAME = "SR_FaceGrid"

# Semantic map of the face grid (HALF face, +X = .L side, mirror modifier
# shows the .R side live). Every vertex is a future bone joint - FaceIt-style:
# the layout density at lips/eyes/brows is what makes expressions & visemes
# possible later. Center chain first, then the .L side.
# Clean FaceIt-style half-face template (v1.99.16, rebuilt from Saeed's
# reference video). Coordinates in FACE UNITS: chin=(0,0), brow line v~1.0,
# +u = .L side. Placed FLAT first (like FaceIt), edited in a locked front
# view, then PROJECTED onto the head. Loops + short struts only - no long
# crossing diagonals (the v1.99.15 net read as spaghetti).
# Clean FaceIt-style half-face template (v1.99.17: TRIMMED to FaceIt's
# density on Saeed's request - ~70 visible points). Coordinates in FACE
# UNITS: chin=(0,0), brow line v~1.0, +u = .L side. Placed FLAT first,
# edited in a locked front view, then PROJECTED onto the head. Closed loops
# + short struts only - no crossing diagonals.
FACE_TEMPLATE = {
    # center column (12)
    "chin_bot": (0, -0.07), "chin": (0, 0.0), "chin_top": (0, 0.09),
    "lip_B": (0, 0.17),
    "lip_T": (0, 0.33), "nose_base": (0, 0.42), "nose_tip": (0, 0.52),
    "nose_bridge": (0, 0.80), "brow_c": (0, 0.98), "forehead_c": (0, 1.44),
    # face outline (6)
    "jaw_low.L": (0.30, 0.02), "jaw_mid.L": (0.55, 0.20),
    "jaw_up.L": (0.70, 0.45), "ear_low.L": (0.78, 0.66),
    "temple.L": (0.79, 0.98), "forehead_side.L": (0.58, 1.30),
    # brow arc (3)
    "brow_in.L": (0.14, 1.00), "brow_mid.L": (0.41, 1.09),
    "brow_out.L": (0.65, 0.96),
    # eyelid ring (8)
    "eye_in.L": (0.245, 0.82), "lid_T_in.L": (0.31, 0.874),
    "lid_T.L": (0.40, 0.888), "lid_T_out.L": (0.49, 0.874),
    "eye_out.L": (0.555, 0.82), "lid_B_out.L": (0.49, 0.766),
    "lid_B.L": (0.40, 0.752), "lid_B_in.L": (0.31, 0.766),
    # eye socket ring (8) - FaceIt has these between the eyelid and the
    # brow (top half) and on the upper cheek (bottom half)
    "socket_in.L": (0.175, 0.82), "socket_T_in.L": (0.265, 0.922),
    "socket_T.L": (0.40, 0.948), "socket_T_out.L": (0.535, 0.922),
    "socket_out.L": (0.625, 0.82), "socket_B_out.L": (0.535, 0.718),
    "socket_B.L": (0.40, 0.692), "socket_B_in.L": (0.265, 0.718),
    # nose wing + cheeks (3)
    "nose_side.L": (0.115, 0.55),
    "cheek_up.L": (0.40, 0.60), "cheek_low.L": (0.27, 0.33),
    # lips outer + inner (5)
    "lip_T.L": (0.11, 0.30), "mouth_corner.L": (0.225, 0.20),
    "lip_B.L": (0.11, 0.165),
    # chin line (FaceIt): marionette box under the mouth corners
    "chin_side.L": (0.13, 0.03),
}
GRID_IDX = {n: i for i, n in enumerate(FACE_TEMPLATE)}
_TE = [
    # outline
    ("chin_bot", "jaw_low.L"), ("jaw_low.L", "jaw_mid.L"),
    ("jaw_mid.L", "jaw_up.L"), ("jaw_up.L", "ear_low.L"),
    ("ear_low.L", "temple.L"), ("temple.L", "forehead_side.L"),
    ("forehead_side.L", "forehead_c"),
    # center column
    ("chin_bot", "chin"), ("chin", "chin_top"), ("chin_top", "lip_B"),
    ("lip_T", "nose_base"), ("nose_base", "nose_tip"),
    ("nose_tip", "nose_bridge"), ("nose_bridge", "brow_c"),
    ("brow_c", "forehead_c"),
    # outer lip ring
    ("lip_T", "lip_T.L"), ("lip_T.L", "mouth_corner.L"),
    ("lip_B", "lip_B.L"), ("lip_B.L", "mouth_corner.L"),

    # chin line (marionette box)
    ("mouth_corner.L", "chin_side.L"), ("chin_side.L", "chin_top"),
    ("chin_side.L", "jaw_low.L"),
    # nose wing
    ("nose_tip", "nose_side.L"), ("nose_side.L", "nose_base"),
    # smile line + cheek
    ("nose_side.L", "cheek_low.L"), ("cheek_low.L", "mouth_corner.L"),
    ("socket_B.L", "cheek_up.L"), ("cheek_up.L", "cheek_low.L"),
    # eye socket ring + spokes
    ("socket_in.L", "socket_T_in.L"), ("socket_T_in.L", "socket_T.L"),
    ("socket_T.L", "socket_T_out.L"), ("socket_T_out.L", "socket_out.L"),
    ("socket_out.L", "socket_B_out.L"), ("socket_B_out.L", "socket_B.L"),
    ("socket_B.L", "socket_B_in.L"), ("socket_B_in.L", "socket_in.L"),
    ("eye_in.L", "socket_in.L"), ("lid_T.L", "socket_T.L"),
    ("eye_out.L", "socket_out.L"), ("lid_B.L", "socket_B.L"),
    # eyelid ring
    ("eye_in.L", "lid_T_in.L"), ("lid_T_in.L", "lid_T.L"),
    ("lid_T.L", "lid_T_out.L"), ("lid_T_out.L", "eye_out.L"),
    ("eye_out.L", "lid_B_out.L"), ("lid_B_out.L", "lid_B.L"),
    ("lid_B.L", "lid_B_in.L"), ("lid_B_in.L", "eye_in.L"),
    # bridge
    ("socket_in.L", "nose_bridge"),
    # brow arc
    ("brow_c", "brow_in.L"), ("brow_in.L", "brow_mid.L"),
    ("brow_mid.L", "brow_out.L"), ("brow_out.L", "temple.L"),
]
GRID_EDGES = [(GRID_IDX[a], GRID_IDX[b]) for a, b in _TE]
# outline verts projected RADIALLY from the skull axis; the rest frontally
GRID_RADIAL = {GRID_IDX[n] for n in
               ("jaw_low.L", "jaw_mid.L", "jaw_up.L", "ear_low.L",
                "temple.L", "forehead_side.L")}

# landmark -> (role) ; .L markers get a mirrored, constraint-driven .R twin
CENTER_LM = ["face_nose", "face_lip_up", "face_lip_low", "face_chin"]
SIDE_LM = ["face_eye.L", "face_brow.L", "face_mouth_corner.L", "face_jaw.L",
           "face_ear.L"]
ALL_LM = CENTER_LM + [n for s in SIDE_LM for n in (s, s[:-2] + ".R")]


def _sstep(t):
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------ detection
def _valid_eye(ob, body, top_z, hgt):
    """An eye mesh candidate must be IN THE SCENE and sit near the body's head.
    Guards against orphan / asset-library duplicates parked at the origin.
    Uses the GEOMETRY (bbox center), never the object origin - imported
    characters keep their origin at the world zero."""
    if ob is None or ob.type != 'MESH' or ob is body:
        return False
    # NEVER our own helpers/widgets (the eyelid RIBBON once passed as an
    # eyeball and scattered the whole face build)
    if ob.name.startswith(("HLP-SR-", "WGT", "SR_")) or ob.get("sr_wgt"):
        return False
    if ob.name not in bpy.context.scene.objects:
        return False
    try:
        c = sum((ob.matrix_world @ Vector(b) for b in ob.bound_box),
                Vector()) / 8.0
        z = float(c.z)
    except Exception:
        z = float(ob.matrix_world.translation.z)
    return (top_z - 0.35 * hgt) < z < (top_z + 0.05 * hgt)


def _sphere_fit(pts):
    """Least-squares sphere through pts -> (center(3,), radius). Robust for
    PARTIAL shells (cornea 'outer eye' covers): the fitted center is the TRUE
    eyeball pivot, where a centroid would sit on the shell surface."""
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + (c ** 2).sum(), 1e-12)))
    return c, r


def _eye_meshes(props, body):
    """Validated eyeball meshes (slots first, then scene meshes named *eye*).
    Heals the props slots when it finds better candidates. Returns a list of
    0-2 objects sorted L (x>0) first."""
    bco = utils.read_rest_coords(body)
    top_z = float(bco[:, 2].max())
    hgt = top_z - float(bco[:, 2].min())

    cands = []
    for ob in (getattr(props, "skin_eye_l", None),
               getattr(props, "skin_eye_r", None)):
        if _valid_eye(ob, body, top_z, hgt) and ob not in cands:
            cands.append(ob)
    if len(cands) < 2:
        named = [ob for ob in bpy.context.scene.objects
                 if "eye" in ob.name.lower()
                 and _valid_eye(ob, body, top_z, hgt) and ob not in cands]
        named.sort(key=lambda o: o.name)
        cands += named
    cands = sorted(cands[:2],
                   key=lambda o: -float(o.matrix_world.translation.x))
    if len(cands) >= 2:              # heal the slots with the validated meshes
        try:
            props.skin_eye_l, props.skin_eye_r = cands[0], cands[1]
        except Exception:
            pass
    return cands


def _eye_center_of(pts):
    """Best eyeball center for a point cloud: sphere fit when it looks like a
    sphere/shell, else the centroid."""
    cen = pts.mean(axis=0)
    if len(pts) >= 12:
        try:
            c, r = _sphere_fit(pts)
            span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            # sane fit: radius comparable to the cloud size, center nearby
            if 0.15 * span < r < 1.5 * span and \
                    float(np.linalg.norm(c - cen)) < 1.2 * span:
                return c
        except Exception:
            pass
    return cen


def _eye_centers(props, body):
    """World-space eye centers (L, R). Returns (eL, eR, sure)."""
    cands = _eye_meshes(props, body)
    eyes = [utils.read_rest_coords(ob) for ob in cands]
    if len(eyes) == 1 and len(eyes[0]) > 8:
        co = eyes[0]
        xm = float(co[:, 0].mean())
        l, r = co[co[:, 0] > xm], co[co[:, 0] <= xm]
        if len(l) > 4 and len(r) > 4:    # single mesh holding both eyeballs
            eyes = [l, r]
    if len(eyes) >= 2:
        cl = _eye_center_of(eyes[0])
        cr = _eye_center_of(eyes[1])
        if cl[0] < cr[0]:
            cl, cr = cr, cl
        return np.asarray(cl), np.asarray(cr), True

    # ---- no eye meshes: proportional guess off the head silhouette ----
    co = utils.read_rest_coords(body)
    top = float(co[:, 2].max())
    hgt = top - float(co[:, 2].min())
    zs = np.linspace(top - 0.02 * hgt, top - 0.30 * hgt, 30)
    width = []
    for z in zs:
        m = np.abs(co[:, 2] - z) < 0.01 * hgt
        width.append(float(co[m][:, 0].max() - co[m][:, 0].min()) if m.sum() else 0.0)
    width = np.array(width)
    head_w = float(width[:12].max())            # widest slice of the skull
    neck_i = int(np.argmin(np.where(width > 0, width, 9e9)))
    neck_z = float(zs[neck_i])
    head_h = top - neck_z
    eye_z = top - 0.45 * head_h
    ipd = 0.32 * head_w
    m = np.abs(co[:, 2] - eye_z) < 0.02 * head_h
    y_front = float(co[m][:, 1].min()) if m.sum() else float(co[:, 1].min())
    eL = np.array([+ipd / 2.0, y_front + 0.12 * head_w, eye_z])
    eR = np.array([-ipd / 2.0, y_front + 0.12 * head_w, eye_z])
    return eL, eR, False


def detect_landmarks(props, body):
    """Compute all face landmark positions from the mesh geometry.
    Returns (dict name -> [x,y,z] world, ipd, sure_eyes)."""
    eL, eR, sure = _eye_centers(props, body)
    eyeC = (eL + eR) / 2.0
    ipd = float(np.linalg.norm(eL - eR))
    if ipd < 1e-6:
        raise RuntimeError("Eye centers coincide - check the eye meshes")

    cb = utils.read_rest_coords(body)
    head = cb[cb[:, 2] > eyeC[2] - 2.5 * ipd]
    if len(head) < 16:
        raise RuntimeError("No head geometry found above the eye line")
    y_front = float(head[:, 1].min())
    y_back = float(head[:, 1].max())
    depth = y_back - y_front

    # centerline front profile: nose / lips / chin = frontmost point per band
    c = cb[np.abs(cb[:, 0]) < 0.25 * ipd]

    def front_max(zhi, zlo):
        m = (c[:, 2] >= zlo) & (c[:, 2] <= zhi)
        if not m.sum():
            return None
        v = c[m]
        i = int(np.argmin(v[:, 1]))
        return float(v[i, 2]), float(v[i, 1])

    nose = front_max(eyeC[2] - 0.05 * ipd, eyeC[2] - 1.2 * ipd)
    nose_z = nose[0] if nose else eyeC[2] - 0.55 * ipd
    lips = front_max(nose_z - 0.20 * ipd, nose_z - 1.0 * ipd)
    lip_z = lips[0] if lips else nose_z - 0.45 * ipd
    chin = front_max(lip_z - 0.45 * ipd, lip_z - 1.6 * ipd)
    chin_z, chin_y = chin if chin else (lip_z - 0.8 * ipd, y_front + 0.2 * ipd)
    m = (np.abs(c[:, 1] - chin_y) < 0.5 * ipd) & \
        (c[:, 2] < chin_z) & (c[:, 2] > chin_z - 1.2 * ipd)
    chin_bot_z = float(c[m][:, 2].min()) if m.sum() else chin_z - 0.35 * ipd

    Minv = body.matrix_world.inverted()
    M3 = Minv.to_3x3()

    def snap_front(x, z):
        o = Minv @ Vector((x, y_front - 4.0 * ipd, z))
        d = (M3 @ Vector((0.0, 1.0, 0.0))).normalized()
        hit, loc, _n, _i = body.ray_cast(o, d)
        if hit:
            v = body.matrix_world @ loc
            return [v.x, v.y, v.z]
        return [x, y_front, z]

    D = eyeC[2] - chin_bot_z                    # face height: eyes -> chin
    mouth_z = lip_z - 0.06 * ipd
    mw = 0.95 * ipd                             # mouth width ~ IPD
    jz = eyeC[2] - 0.22 * D                     # TMJ height
    sl = head[np.abs(head[:, 2] - jz) < 0.3 * ipd]
    half_w = float(np.abs(sl[:, 0]).max()) if len(sl) else 0.6 * ipd

    # ear: centroid of the side-most head verts around the eye line
    em = (head[:, 0] > 0.82 * half_w) & \
         (np.abs(head[:, 2] - (eyeC[2] - 0.2 * ipd)) < 1.2 * ipd)
    if em.sum() >= 4:
        ear = head[em].mean(axis=0)
        ear_pos = [float(ear[0]), float(ear[1]), float(ear[2])]
    else:
        ear_pos = [0.95 * half_w, y_front + 0.62 * depth, eyeC[2] - 0.2 * ipd]

    L = {
        "face_nose": snap_front(0.0, nose_z),
        "face_lip_up": snap_front(0.0, mouth_z + 0.10 * ipd),
        "face_lip_low": snap_front(0.0, mouth_z - 0.10 * ipd),
        "face_chin": snap_front(0.0, (chin_z + chin_bot_z) / 2.0),
        "face_mouth_corner.L": snap_front(+mw / 2.0, mouth_z),
        "face_brow.L": snap_front(float(eL[0]), eyeC[2] + 0.55 * ipd),
        "face_eye.L": [float(v) for v in eL],
        "face_jaw.L": [0.80 * half_w, y_front + 0.58 * depth, jz],
        "face_ear.L": ear_pos,
    }
    return L, ipd, sure


def template_landmarks(props, body):
    """FaceIt-style TEMPLATE: the face markers laid out by classic facial
    proportions on the head region (top of the mesh / rig head bone) and
    snapped to the FRONT surface. NO geometric detection - the user drags
    every marker onto his character's face, then registers the mouth/eye
    loops. Returns (dict name -> [x,y,z] world, ipd)."""
    cb = utils.read_rest_coords(body)
    ground = float(cb[:, 2].min()); top = float(cb[:, 2].max())
    hgt = top - ground
    # head band: rig head bone when available, else the top 13% of the mesh
    z_lo = None
    for rn in ("SR_Metarig", "SR_Rig"):
        rig = bpy.data.objects.get(rn)
        if rig is not None and rig.type == 'ARMATURE':
            hb = rig.data.bones.get("head") or rig.data.bones.get("spine.006")
            if hb is not None:
                mw = rig.matrix_world
                z_lo = float((mw @ hb.head_local).z)
                break
    if z_lo is None or z_lo > top - 0.04 * hgt:
        z_lo = top - 0.13 * hgt
    head = cb[cb[:, 2] > z_lo]
    if len(head) < 16:
        raise RuntimeError("No head geometry found - check the character mesh")
    headH = top - z_lo
    half_w = float(np.percentile(np.abs(head[:, 0]), 98))
    y_front = float(head[:, 1].min()); y_back = float(head[:, 1].max())
    depth = y_back - y_front
    ipd = 0.42 * (2.0 * half_w)                  # eye centers ~ 2/5 head width

    Minv = body.matrix_world.inverted()
    M3 = Minv.to_3x3()

    def snap_front(x, z):
        o = Minv @ Vector((x, y_front - 2.0 * headH, z))
        d = (M3 @ Vector((0.0, 1.0, 0.0))).normalized()
        hit, loc, _n, _i = body.ray_cast(o, d)
        if hit:
            v = body.matrix_world @ loc
            return [v.x, v.y, v.z]
        return [x, y_front, z]

    eye_z = z_lo + 0.55 * headH                  # classic: eyes at mid-face
    eye_x = 0.5 * ipd
    chin_z = z_lo + 0.06 * headH
    mouth_z = z_lo + 0.28 * headH
    nose_z = z_lo + 0.42 * headH
    jz = eye_z - 0.22 * (eye_z - chin_z)         # TMJ height (same as detect)

    L = {
        "face_nose": snap_front(0.0, nose_z),
        "face_lip_up": snap_front(0.0, mouth_z + 0.10 * ipd),
        "face_lip_low": snap_front(0.0, mouth_z - 0.10 * ipd),
        "face_chin": snap_front(0.0, chin_z),
        "face_mouth_corner.L": snap_front(0.475 * ipd, mouth_z),
        "face_brow.L": snap_front(eye_x, eye_z + 0.55 * ipd),
        "face_eye.L": [eye_x, y_front + 0.28 * depth, eye_z],
        "face_jaw.L": [0.80 * half_w, y_front + 0.58 * depth, jz],
        "face_ear.L": [0.95 * half_w, y_front + 0.62 * depth,
                       eye_z - 0.2 * ipd],
    }
    return L, ipd


def _marker_coll():
    coll = utils.ensure_collection(FACE_COLL)
    try:
        coll.color_tag = 'COLOR_06'
    except Exception:
        pass
    return coll


def get_marker(name):
    ob = bpy.data.objects.get(name)
    if ob is not None and ob.name in [o.name for o in _marker_coll().objects]:
        return ob
    return ob


def markers_present():
    return all(bpy.data.objects.get(n) is not None
               for n in ("face_chin", "face_jaw.L", "face_lip_up"))


def place_markers(L, ipd):
    """(Re)create the face marker empties. .R twins mirror .L live."""
    coll = _marker_coll()
    for ob in list(coll.objects):
        if ob.name == GRID_NAME:        # never wipe the landmark net
            continue
        bpy.data.objects.remove(ob, do_unlink=True)

    def mk(name, co, role):
        ob = bpy.data.objects.new(name, None)
        coll.objects.link(ob)
        # tiny empty core - the colourful GPU glow (wizard overlay) is the
        # visible marker, exactly like the body markers
        ob.empty_display_type = 'PLAIN_AXES'
        ob.empty_display_size = 0.05 * ipd
        ob.show_in_front = True
        ob.location = Vector(co)
        ob.color = {'center': (0.2, 0.9, 1.0, 1.0),
                    'left': (1.0, 0.8, 0.1, 1.0),
                    'right': (0.55, 0.45, 0.2, 1.0)}[role]
        if role == 'center':
            ob.lock_location = (True, False, False)     # keep on the centerline
        elif role == 'right':
            ob.lock_location = (True, True, True)       # driven by the .L twin
        return ob

    for name in CENTER_LM:
        mk(name, L[name], 'center')
    for name in SIDE_LM:
        lob = mk(name, L[name], 'left')
        rname = name[:-2] + ".R"
        co = L[name]
        rob = mk(rname, (-co[0], co[1], co[2]), 'right')
        con = rob.constraints.new('COPY_LOCATION')
        con.name = "SR Mirror"
        con.target = lob
        con.invert_x = True
        con.target_space = con.owner_space = 'WORLD'


def _lm(name):
    ob = bpy.data.objects.get(name)
    if ob is None:
        raise RuntimeError("Face marker '%s' is missing - run Detect again" % name)
    return np.array(ob.matrix_world.translation)


# marker -> grid-vertex refinement (the user-polished grid wins when present)
_GRID_REFINE = {
    "face_chin": "chin", "face_lip_up": "lip_T", "face_lip_low": "lip_B",
    "face_nose": "nose_tip", "face_mouth_corner.L": "mouth_corner.L",
    "face_mouth_corner.R": "mouth_corner.R",
    "face_brow.L": "brow_mid.L", "face_brow.R": "brow_mid.R",
}


def _lm_ref(name):
    """Landmark position, refined by the face grid when one exists."""
    g = _GRID_REFINE.get(name)
    if g:
        gp = grid_points()
        if g in gp:
            return gp[g]
    return _lm(name)


# ------------------------------------------------------------------ face grid
def _make_grid_object(pts, ipd_hint):
    """(Re)create SR_FaceGrid from world-space pts (FACE_TEMPLATE order)."""
    old = bpy.data.objects.get(GRID_NAME)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    om = bpy.data.meshes.get(GRID_NAME)
    if om is not None and om.users == 0:
        bpy.data.meshes.remove(om)
    me = bpy.data.meshes.new(GRID_NAME)
    me.from_pydata(pts, GRID_EDGES, [])
    me.update()
    ob = bpy.data.objects.new(GRID_NAME, me)
    _marker_coll().objects.link(ob)
    mod = ob.modifiers.new("Mirror", 'MIRROR')
    mod.use_axis = (True, False, False)
    mod.use_clip = True
    mod.merge_threshold = 0.001 * max(ipd_hint, 0.01)
    ob.show_in_front = True
    ob.display_type = 'WIRE'
    ob.color = (1.0, 0.55, 0.1, 1.0)
    return ob


def flat_grid_pts(chin_pos, scale, y0, scale_z=None):
    """FACE_TEMPLATE laid out FLAT (XZ plane at depth y0), FaceIt-style.
    scale = WIDTH, scale_z = HEIGHT (defaults to scale = uniform)."""
    sz = scale if scale_z is None else scale_z
    return [[u * scale, y0, chin_pos.z + v * sz]
            for u, v in FACE_TEMPLATE.values()]


def project_grid(body, ob):
    """Snap every grid vertex onto the head: front ray -> radial ray ->
    closest surface point (a vertex must NEVER stay floating)."""
    co = utils.read_rest_coords(body)
    mw = ob.matrix_world
    pts = [list(mw @ v.co) for v in ob.data.vertices]
    zs = [p[2] for p in pts]
    z_lo = min(zs)
    head = co[co[:, 2] > z_lo - 0.02]
    if len(head) < 8:
        raise RuntimeError("No head geometry at the grid height")
    yc = float(np.median(head[:, 1]))
    head_r = max(float(np.abs(head[:, 0]).max()) * 2.0, 0.1)
    Minv = body.matrix_world.inverted()
    M3 = Minv.to_3x3()
    y_front = float(head[:, 1].min())
    missed = 0
    for i, p in enumerate(pts):
        if i in GRID_RADIAL:
            # Side landmarks (ear / temple / jaw / forehead side) belong on
            # the SIDE of the head.  The flat net lays them on the FRONT
            # plane, so a radial ray cast from that plane is dominated by
            # the frontal component and collapses them onto the FRONT of
            # the face (the ear ends up next to the nose, which drags the
            # cheek / jawline / mouth-master controls in with it).  Cast a
            # LATERAL ray straight inward at mid-depth so the point lands on
            # the true side silhouette instead.
            side = 1.0 if p[0] >= 0.0 else -1.0
            o = Vector((side * 3.0 * head_r, yc, p[2]))
            dr = Vector((-side, 0.0, 0.0))
        else:
            o = Vector((p[0], y_front - 2.0 * head_r, p[2]))
            dr = Vector((0.0, 1.0, 0.0))
        hit, loc, _n, _i = body.ray_cast(Minv @ o, (M3 @ dr).normalized())
        if not hit and i in GRID_RADIAL:
            # thin slice - fall back to a radial ray from the head centre
            d = Vector((p[0], p[1] - yc, 0.0))
            if d.length > 1e-6:
                d.normalize()
                o2 = Vector((0.0, yc, p[2])) + d * (3.0 * head_r)
                hit, loc, _n, _i = body.ray_cast(Minv @ o2,
                                                 (M3 @ (-d)).normalized())
        if not hit and i not in GRID_RADIAL:
            d = Vector((p[0], p[1] - yc, 0.0))
            if d.length > 1e-6:
                d.normalize()
                o2 = Vector((0.0, yc, p[2])) + d * (3.0 * head_r)
                hit, loc, _n, _i = body.ray_cast(Minv @ o2,
                                                 (M3 @ (-d)).normalized())
        if not hit:
            ok2, loc2, _n2, _i2 = body.closest_point_on_mesh(
                Minv @ Vector((p[0], p[1], p[2])))
            if ok2:
                hit, loc = True, loc2
        if hit:
            wp = body.matrix_world @ loc
            pts[i] = [wp.x, wp.y, wp.z]
        else:
            missed += 1
    inv = mw.inverted()
    for i, p in enumerate(pts):
        ob.data.vertices[i].co = inv @ Vector(p)
    ob.data.update()
    return missed


def markers_from_grid(body):
    """Create/update the 14 anchor face markers FROM the projected grid, so
    Build Face Base works exactly as before (grid is the source of truth)."""
    gp = grid_points()
    lids = [gp[n + ".L"] for n in ("eye_in", "lid_T_in", "lid_T", "lid_T_out",
                                   "eye_out", "lid_B_out", "lid_B", "lid_B_in")]
    eyeC = np.mean(np.array(lids), axis=0)
    rx = 0.5 * float(np.linalg.norm(gp["eye_out.L"] - gp["eye_in.L"]))
    eyeC = eyeC + np.array([0.0, 0.55 * rx, 0.0])       # push to eyeball depth
    ipd = 2.0 * abs(float(eyeC[0]))
    ear = 0.5 * (gp["ear_low.L"] + gp["temple.L"])
    L = {
        "face_nose": [float(v) for v in gp["nose_tip"]],
        "face_lip_up": [float(v) for v in gp["lip_T"]],
        "face_lip_low": [float(v) for v in gp["lip_B"]],
        "face_chin": [float(v) for v in gp["chin"]],
        "face_mouth_corner.L": [float(v) for v in gp["mouth_corner.L"]],
        "face_brow.L": [float(v) for v in gp["brow_mid.L"]],
        "face_eye.L": [float(v) for v in eyeC],
        "face_jaw.L": [float(v) for v in gp["ear_low.L"]],
        "face_ear.L": [float(v) for v in ear],
    }
    place_markers(L, max(ipd, 0.02))
    # the NET is the visible UI in the FaceIt flow - the anchor markers are
    # derived data for Build, keep them hidden (the glow overlay skips hidden)
    for nm in ALL_LM:
        o = bpy.data.objects.get(nm)
        if o is not None:
            try:
                o.hide_set(True)
            except Exception:
                pass
    return ipd


def build_face_grid(props, context):
    """LEGACY path (markers already exist, e.g. via face_template/face_detect):
    lay the clean template flat from the marker anchors, then project."""
    body = getattr(props, "target_mesh", None) or context.active_object
    if body is None or body.type != 'MESH':
        raise RuntimeError("Pick the character mesh first")
    chin = _lm("face_chin")
    eyeL = _lm("face_eye.L")
    scale = max(float(eyeL[0]) / 0.40, 0.02)      # eye center sits at u=0.40
    y0 = float(min(_lm("face_nose")[1], chin[1]))
    pts = flat_grid_pts(Vector((0.0, y0, float(chin[2]))), scale, y0)
    ob = _make_grid_object(pts, 0.42 * 2.0 * scale)
    missed = project_grid(body, ob)
    return ob, missed


def grid_points():
    """name -> world position for every grid vertex (L half + mirrored R).
    Returns {} when no grid exists."""
    ob = bpy.data.objects.get(GRID_NAME)
    if ob is None:
        return {}
    mw = ob.matrix_world
    out = {}
    for name, i in GRID_IDX.items():
        v = mw @ ob.data.vertices[i].co
        out[name] = np.array([v.x, v.y, v.z])
        if name.endswith(".L"):
            out[name[:-2] + ".R"] = np.array([-v.x, v.y, v.z])
    return out


# ------------------------------------------------------------------ build
def _target_rig():
    """The rig the face bones attach to: the generated Rigify rig if present,
    else the active/selected armature, else None (standalone face rig)."""
    try:
        from . import metarig as _mr
        rig = _mr._generated_rig()
        if rig is not None:
            return rig
    except Exception:
        pass
    ob = bpy.context.active_object
    if ob is not None and ob.type == 'ARMATURE':
        return ob
    for ob in bpy.context.selected_objects:
        if ob.type == 'ARMATURE':
            return ob
    return None


def _head_parent_name(rig):
    for n in ("ORG-head", "DEF-head", "head"):
        if n in rig.data.bones:
            return n
    return None


def build_base(props, context):
    """Initial face bones (Storm ch.2): DEF-jaw pivot->chin + CTL-jaw + the
    master-mouth placeholder. Attaches to the body rig when present, else
    creates a standalone SR_FaceRig with a head bone."""
    body = getattr(props, "target_mesh", None) or context.active_object
    if body is None or body.type != 'MESH':
        raise RuntimeError("Pick the character mesh first")

    jawL, jawR = _lm("face_jaw.L"), _lm("face_jaw.R")
    chin = _lm_ref("face_chin")
    lip_up, lip_low = _lm_ref("face_lip_up"), _lm_ref("face_lip_low")
    eyeC = (_lm("face_eye.L") + _lm("face_eye.R")) / 2.0
    P = (jawL + jawR) / 2.0
    mouth_mid = (lip_up + lip_low) / 2.0
    ipd = float(np.linalg.norm(_lm("face_eye.L") - _lm("face_eye.R")))

    rig = _target_rig()
    standalone = rig is None
    if standalone:
        old = bpy.data.objects.get(FACE_RIG_NAME)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        arm = bpy.data.armatures.new(FACE_RIG_NAME)
        rig = bpy.data.objects.new(FACE_RIG_NAME, arm)
        context.scene.collection.objects.link(rig)

    prev_active = context.view_layer.objects.active
    prev_mode = bpy.context.mode
    context.view_layer.objects.active = rig
    rig.hide_set(False)
    rig.hide_viewport = False
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones

    # idempotent: rebuild our bones from scratch (old + Storm names)
    ours = ["DEF-jaw", "CTL-jaw", "CTL-Jaw", "master-mouth", "MSTR-Mouth",
            "CTL-eyes", "MSTR-Eye_target"]
    for s in (".L", ".R"):
        ours += ["master-eye" + s, "MSTR-Eye" + s, "DEF-eye" + s,
                 "CTL-eye" + s, "TGT-Eye" + s, "DEF-ear" + s]
    for n in ours:
        if n in eb:
            eb.remove(eb[n])

    parent_name = None
    if standalone:
        cb = utils.read_rest_coords(body)
        top_z = float(cb[cb[:, 2] > eyeC[2] - 2.5 * ipd][:, 2].max())
        if "head" in eb:
            eb.remove(eb["head"])
        hb = eb.new("head")
        hb.head = Vector((0.0, P[1], P[2] - 0.25 * ipd))
        hb.tail = Vector((0.0, P[1], top_z))
        hb.use_deform = True
        parent_name = "head"
    else:
        parent_name = _head_parent_name(rig)

    def nb(name, head, tail, deform):
        b = eb.new(name)
        b.head = Vector([float(v) for v in head])
        b.tail = Vector([float(v) for v in tail])
        b.use_deform = deform
        if parent_name and parent_name in eb:
            b.parent = eb[parent_name]
            b.use_connect = False
        return b

    # Storm's DEF-Jaw is HORIZONTAL: pivot at the TMJ, tail straight forward
    # at pivot height (rotation about local X is identical - the pivot and the
    # axis are what matter - but the axes match Storm and WGT-Jaw sits right)
    jaw_tail = np.array([0.0, float(chin[1]) + 0.15 * (float(P[1]) - float(chin[1])),
                         float(P[2])])
    dj = nb("DEF-jaw", P, jaw_tail, True)
    cj = nb("CTL-Jaw", P, jaw_tail, False)
    mm = eb.new("MSTR-Mouth")
    mm.head = Vector([float(v) for v in mouth_mid])
    mm.tail = Vector([float(v) for v in mouth_mid + np.array([0.0, 0.0, 0.10 * ipd])])
    mm.use_deform = False
    mm.parent = dj
    mm.use_connect = False

    # roll: local X = +world X so a POSITIVE X rotation opens the jaw DOWN
    import math
    for b in (dj, cj):
        b.align_roll(Vector((0.0, 0.0, 1.0)))
        if (rig.matrix_world.to_3x3() @ b.x_axis).x < 0.0:
            b.roll += math.pi

    # ---- eyes: master + deform (aims at its target) + targets (Storm ch.4
    # simplified: DAMPED_TRACK now, Armature-constraint space switch later) ----
    eye_meshes = _eye_meshes(props, body)
    eye_r = {}
    for ob in eye_meshes:
        c = utils.read_rest_coords(ob)
        clouds = [c]
        if len(eye_meshes) == 1 and float(c[:, 0].max()) > 0.1 * ipd \
                and float(c[:, 0].min()) < -0.1 * ipd:
            xm = float(c[:, 0].mean())
            l, r_ = c[c[:, 0] > xm], c[c[:, 0] <= xm]
            if len(l) > 4 and len(r_) > 4:
                clouds = [l, r_]
        for cl in clouds:
            ec = _eye_center_of(cl)
            rr = float(np.linalg.norm(cl - ec, axis=1).max())
            eye_r[".L" if ec[0] >= 0 else ".R"] = max(rr, 0.15 * ipd)

    tgt_dist = 6.0 * ipd
    ct = eb.new("MSTR-Eye_target")
    ct.head = Vector((0.0, float(eyeC[1]) - tgt_dist, float(eyeC[2])))
    ct.tail = ct.head + Vector((0.0, 0.0, 0.6 * ipd))
    ct.use_deform = False
    if parent_name and parent_name in eb:
        ct.parent = eb[parent_name]

    for s, sgn in ((".L", 1.0), (".R", -1.0)):
        ec = _lm("face_eye" + s)
        r = eye_r.get(s, 0.5 * ipd)
        me_b = eb.new("MSTR-Eye" + s)
        me_b.head = Vector([float(v) for v in ec])
        me_b.tail = me_b.head + Vector((0.0, 0.0, 1.2 * r))
        me_b.use_deform = False
        if parent_name and parent_name in eb:
            me_b.parent = eb[parent_name]
        de = eb.new("DEF-eye" + s)
        de.head = Vector([float(v) for v in ec])
        de.tail = de.head + Vector((0.0, -1.4 * r, 0.0))
        de.use_deform = True
        de.parent = me_b
        te = eb.new("TGT-Eye" + s)
        te.head = Vector((float(ec[0]), float(ec[1]) - tgt_dist, float(ec[2])))
        te.tail = te.head + Vector((0.0, 0.0, 0.4 * ipd))
        te.use_deform = False
        te.parent = ct

    # ---- ears: vertical deform bone at the ear centroid ----
    for s in (".L", ".R"):
        try:
            ear = _lm("face_ear" + s)
        except RuntimeError:
            continue
        de = eb.new("DEF-ear" + s)
        de.head = Vector((float(ear[0]), float(ear[1]), float(ear[2]) - 0.35 * ipd))
        de.tail = Vector((float(ear[0]), float(ear[1]), float(ear[2]) + 0.55 * ipd))
        de.use_deform = True
        if parent_name and parent_name in eb:
            de.parent = eb[parent_name]

    # bone collections
    try:
        col_c = utils.bone_collection(rig.data, "Face")
        col_m = utils.bone_collection(rig.data, "Face (MCH)")
        ctl = ["CTL-Jaw", "MSTR-Mouth", "MSTR-Eye_target",
               "TGT-Eye.L", "TGT-Eye.R", "MSTR-Eye.L", "MSTR-Eye.R"]
        mch = ["DEF-jaw", "DEF-eye.L", "DEF-eye.R", "DEF-ear.L", "DEF-ear.R"]
        for bname in ctl:
            if bname in eb:
                col_c.assign(eb[bname])
        for bname in mch:
            if bname in eb:
                col_m.assign(eb[bname])
        col_m.is_visible = False
    except Exception:
        pass

    bpy.ops.object.mode_set(mode='POSE')
    pb = rig.pose.bones["DEF-jaw"]
    for c in list(pb.constraints):
        pb.constraints.remove(c)
    con = pb.constraints.new('COPY_TRANSFORMS')
    con.name = "SR Face Jaw"
    con.target = rig
    con.subtarget = "CTL-Jaw"
    cjp = rig.pose.bones["CTL-Jaw"]
    cjp.rotation_mode = 'XYZ'
    for s in (".L", ".R"):
        if "DEF-eye" + s not in rig.pose.bones:
            continue
        pe = rig.pose.bones["DEF-eye" + s]
        for c in list(pe.constraints):
            pe.constraints.remove(c)
        con = pe.constraints.new('DAMPED_TRACK')
        con.name = "SR Eye Aim"
        con.target = rig
        con.subtarget = "TGT-Eye" + s
        con.track_axis = 'TRACK_Y'
    bpy.ops.object.mode_set(mode='OBJECT')

    # ---- Storm widgets + palettes (shapes from the CC-BY Storm rig) ----
    from . import face_widgets as _fw
    jaw_len = float(np.linalg.norm(jaw_tail - P))
    _fw.assign(rig, "CTL-Jaw", "WGT-Jaw", 1.37, "THEME04")
    _fw.assign(rig, "MSTR-Mouth", "WGT-Mouth", 10.5, "THEME09")
    _fw.assign(rig, "MSTR-Eye_target", "WGT-Eyes_Target", 8.0, "THEME04")
    for s2 in (".L", ".R"):
        r2 = eye_r.get(s2, 0.5 * ipd)
        _fw.assign(rig, "TGT-Eye" + s2, "WGT-Circle",
                   1.2 * r2 / (0.4 * ipd), "THEME13")
        _fw.assign(rig, "MSTR-Eye" + s2, "WGT-Cube", 1.17, None)

    # ---- rigid-bind the eyeball meshes to their DEF-eye bones ----
    # A single object can hold BOTH eyeballs (e.g. an 'outer eye' cornea
    # shell): its verts are split per side into DEF-eye.L / DEF-eye.R.
    for ob in eye_meshes:
        c = utils.read_rest_coords(ob)
        for m in list(ob.modifiers):
            if m.type == 'ARMATURE':
                ob.modifiers.remove(m)
        for vg in list(ob.vertex_groups):
            if vg.name.startswith("DEF-"):
                ob.vertex_groups.remove(vg)
        spans_both = (float(c[:, 0].max()) > 0.1 * ipd and
                      float(c[:, 0].min()) < -0.1 * ipd)
        if spans_both and len(eye_meshes) == 1:
            xm = float(c[:, 0].mean())
            li = [i for i in range(len(c)) if c[i, 0] > xm]
            ri = [i for i in range(len(c)) if c[i, 0] <= xm]
            gl = ob.vertex_groups.new(name="DEF-eye.L")
            gl.add(li, 1.0, 'REPLACE')
            gr = ob.vertex_groups.new(name="DEF-eye.R")
            gr.add(ri, 1.0, 'REPLACE')
        else:
            s = ".L" if c.mean(axis=0)[0] >= 0 else ".R"
            vg = ob.vertex_groups.new(name="DEF-eye" + s)
            vg.add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
        if ob.parent != rig:
            mw = ob.matrix_world.copy()
            ob.parent = rig
            ob.matrix_world = mw
        mod = ob.modifiers.new("Armature", 'ARMATURE')
        mod.object = rig

    if prev_active is not None:
        context.view_layer.objects.active = prev_active

    return rig, standalone


# ------------------------------------------------------------------ weights
def base_weights(props, context, rig, standalone):
    """Analytic base mask weights (Storm ch.2 equivalent, computed not painted):
    jaw field below the pivot->mouth plane, capped laterally at the pivots,
    faded behind the ear / under the chin. On a bound body rig the jaw is
    CARVED OUT of the existing head weights; standalone writes head = rest."""
    body = getattr(props, "target_mesh", None) or context.active_object
    jawL, jawR = _lm("face_jaw.L"), _lm("face_jaw.R")
    chin = _lm_ref("face_chin")
    lip_up, lip_low = _lm_ref("face_lip_up"), _lm_ref("face_lip_low")
    eyeL, eyeR = _lm("face_eye.L"), _lm("face_eye.R")
    eyeC = (eyeL + eyeR) / 2.0
    ipd = float(np.linalg.norm(eyeL - eyeR))
    P = (jawL + jawR) / 2.0
    # split plane passes just UNDER the mouth line: the upper lip stays with
    # the skull (Storm), only the lower lip rides the jaw
    M = lip_low + 0.30 * (lip_up - lip_low)
    jaw_x = abs(float(jawL[0]))

    w = utils.read_rest_coords(body)
    n = len(w)

    d = (M - P).astype(float)
    d[0] = 0.0
    d /= np.linalg.norm(d)
    nrm = np.array([0.0, -d[2], d[1]])
    if nrm[2] > 0:
        nrm = -nrm
    s = (w - P) @ nrm
    band = 0.11 * ipd
    wj = _sstep(np.clip((s + band) / (2.0 * band), 0.0, 1.0))
    back = np.clip((w[:, 1] - P[1]) / (0.8 * ipd), 0.0, 1.0)
    wj *= (1.0 - back * back)
    chin_bot = chin[2] - 0.15 * ipd
    under = _sstep(np.clip(((chin_bot - 0.35 * ipd) - w[:, 2]) / (0.5 * ipd), 0.0, 1.0))
    wj *= (1.0 - under)
    xcap = _sstep(np.clip((np.abs(w[:, 0]) - (jaw_x + 0.15 * ipd)) / (0.35 * ipd), 0.0, 1.0))
    wj *= (1.0 - xcap)
    mid = (P + chin) / 2.0
    R = float(np.linalg.norm(chin - P))
    r = np.linalg.norm(w - mid, axis=1)
    rad = np.clip((r - 1.0 * R) / (0.45 * R), 0.0, 1.0)
    wj *= (1.0 - rad * rad)

    # ---- ear weight fields: radial falloff around each ear marker ----
    fields = [("DEF-jaw", wj)]
    for s in (".L", ".R"):
        try:
            ear = _lm("face_ear" + s)
        except RuntimeError:
            continue
        if ("DEF-ear" + s) not in rig.data.bones:
            continue
        dist = np.linalg.norm(w - ear, axis=1)
        we = 1.0 - _sstep(np.clip((dist - 0.45 * ipd) / (0.55 * ipd), 0.0, 1.0))
        we *= (np.sign(ear[0]) * w[:, 0]) > 0.3 * abs(ear[0])  # own side only
        fields.append(("DEF-ear" + s, we))

    head_group = None
    if not standalone:
        for cand in ("DEF-head", "head", "DEF-spine.006"):
            if body.vertex_groups.get(cand) is not None:
                head_group = cand
                break

    groups = {}
    for gname, _f in fields:
        g = body.vertex_groups.get(gname)
        if g is None:
            g = body.vertex_groups.new(name=gname)
        groups[gname] = g

    if head_group is not None:
        # ---- carve mode: move weight from the head group to the face bones ----
        gh = body.vertex_groups[head_group]
        idxs = {gname: groups[gname].index for gname, _f in fields}
        hi = gh.index
        me = body.data
        all_idx = set(idxs.values())
        for i, v in enumerate(me.vertices):
            old_h = 0.0
            old_f = 0.0
            for g in v.groups:
                if g.group == hi:
                    old_h = g.weight
                elif g.group in all_idx:
                    old_f += g.weight
            total = old_h + old_f            # undo any previous carve first
            if total <= 0.0:
                continue
            rem = total
            for gname, f in fields:
                fw = min(float(f[i]), rem)
                if fw > 1e-4:
                    groups[gname].add([i], fw, 'REPLACE')
                    rem -= fw
                else:
                    groups[gname].remove([i])
            gh.add([i], rem, 'REPLACE')
    else:
        # ---- standalone: compute the head field too ----
        zcut = eyeC[2] - 2.5 * ipd
        wh = _sstep(np.clip((w[:, 2] - (zcut - 0.6 * ipd)) / (1.2 * ipd), 0.0, 1.0))
        rem = wh.copy()
        gh = body.vertex_groups.get("head")
        if gh is None:
            gh = body.vertex_groups.new(name="head")
        idx = list(range(n))
        gh.remove(idx)
        for gname, f in fields:
            groups[gname].remove(idx)
        for i in range(n):
            r_i = float(rem[i])
            for gname, f in fields:
                fw = min(float(f[i]), r_i)
                if fw > 1e-4:
                    groups[gname].add([i], fw, 'REPLACE')
                    r_i -= fw
            if r_i > 1e-4:
                gh.add([i], r_i, 'REPLACE')
        has_mod = any(m.type == 'ARMATURE' and m.object == rig
                      for m in body.modifiers)
        if not has_mod:
            mod = body.modifiers.new(rig.name, 'ARMATURE')
            mod.object = rig

    return int((wj > 0.5).sum())



# ------------------------------------------------------------- Storm controls

# Per-bone spec measured from the REAL Storm rig (bone direction, Z axis for
# the roll, length as a fraction of the IPD, widget + scale + palette). Our
# bones keep THEIR grid positions but take Storm's orientation and sizing -
# the widgets were modeled for these axes.
STORM_SPEC = {'MSTR-Face_upp': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.2642, 'wgt': 'WGT-Circle', 'scale': [10.7, 10.7, 10.7], 'palette': 'THEME10', 'cs_rot': [0.0, 0.0, 0.0], 'cs_tr': [0.0, 0.013, 0.0]}, 'MSTR-Face_low': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.2522, 'wgt': 'WGT-Circle', 'scale': [11.2, 11.2, 11.2], 'palette': 'THEME10', 'cs_rot': [0.0, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.0]}, 'MSTR-Mouth': {'dir': [0.0, 0.0, 1.0], 'z': [-0.0, -1.0, -0.0], 'len_ipd': 0.0912, 'wgt': 'WGT-Mouth', 'scale': [9.46, 9.46, 9.46], 'palette': 'THEME09', 'cs_rot': [0.0, 0.0, 0.0], 'cs_tr': [0.0, 0.011, 0.067]}, 'MSTR-Nose': {'dir': [-0.0, 0.2427, 0.9701], 'z': [0.0, -0.9701, 0.2427], 'len_ipd': 0.0256, 'wgt': 'WGT-Circle', 'scale': [19.0, 19.0, 19.0], 'palette': 'THEME03', 'cs_rot': [0.0, 0.0, 0.0], 'cs_tr': [0.0, 0.0, -0.014]}, 'CTL-Jaw': {'dir': [-0.0, -1.0, 0.0], 'z': [0.0, -0.0, 1.0], 'len_ipd': 1.0714, 'wgt': 'WGT-Jaw', 'scale': [1.37, 1.37, 1.37], 'palette': 'THEME04', 'cs_rot': [-0.4276, 0.0, 0.0], 'cs_tr': [0.0, 0.003, 0.01]}, 'MSTR-Eye_target': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.1299, 'wgt': 'WGT-Eyes_Target', 'scale': [8.3, 8.3, 8.3], 'palette': 'THEME04', 'cs_rot': [0.0, 1.5708, 0.0], 'cs_tr': [0.0, 0.0, 0.0]}, 'TGT-Eye.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.0393, 'wgt': 'WGT-Circle', 'scale': [7.7, 7.7, 7.7], 'palette': 'THEME13', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.0]}, 'TGT-Eye.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.0393, 'wgt': 'WGT-Circle', 'scale': [7.7, 7.7, 7.7], 'palette': 'THEME13', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.0]}, 'MSTR-Eye.L': {'dir': [0.0, -1.0, 0.0], 'z': [0.0, 0.0, 1.0], 'len_ipd': 0.0962, 'wgt': 'WGT-Cube', 'scale': [7.64, 7.64, 7.64], 'palette': 'DEFAULT', 'cs_rot': [0.0, 0.0, 0.0], 'cs_tr': [0.0, 0.021, 0.0]}, 'MSTR-Eye.R': {'dir': [0.0, -1.0, 0.0], 'z': [0.0, 0.0, 1.0], 'len_ipd': 0.0962, 'wgt': 'WGT-Cube', 'scale': [-7.64, 7.64, 7.64], 'palette': 'DEFAULT', 'cs_rot': [0.0, -0.0, -0.0], 'cs_tr': [-0.0, 0.021, 0.0]}, 'CTL-Lips_main_upp': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0019, 'wgt': 'WGT-lips_main', 'scale': [1.0, -0.55, 1.0], 'palette': 'THEME09', 'cs_rot': [0.0, 0.0, 0.0], 'cs_tr': [0.0, 0.006, 0.0]}, 'CTL-Lips_main_low': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0019, 'wgt': 'WGT-lips_main', 'scale': [1.0, 0.75, 1.0], 'palette': 'THEME09', 'cs_rot': [0.0, 0.0, 0.0], 'cs_tr': [0.0, -0.003, -0.002]}, 'CTL-Lips_corn.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.6997, -0.7145, -0.0], 'len_ipd': 0.0227, 'wgt': 'WGT-Mouth_corner', 'scale': [5.4, 4.079, 4.079], 'palette': 'THEME09', 'cs_rot': [-1.7453, 0.0, 1.5708], 'cs_tr': [0.003, 0.0, 0.011]}, 'CTL-Lips_corn.R': {'dir': [0.0, 0.0, 1.0], 'z': [-0.6997, -0.7145, -0.0], 'len_ipd': 0.0227, 'wgt': 'WGT-Mouth_corner', 'scale': [-5.4, 4.079, 4.079], 'palette': 'THEME09', 'cs_rot': [-1.7453, -0.0, -1.5708], 'cs_tr': [-0.003, 0.0, 0.011]}, 'CTL-Lid_upp.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.019, 'wgt': None, 'scale': [1.0, 1.0, 1.0], 'palette': 'DEFAULT', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.0]}, 'CTL-Lid_upp.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.019, 'wgt': None, 'scale': [1.0, 1.0, 1.0], 'palette': 'DEFAULT', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.0]}, 'CTL-Lid_low.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.019, 'wgt': None, 'scale': [1.0, 1.0, 1.0], 'palette': 'DEFAULT', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.0]}, 'CTL-Lid_low.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.019, 'wgt': None, 'scale': [1.0, 1.0, 1.0], 'palette': 'DEFAULT', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.0]}, 'CTL-Brow_all.L': {'dir': [0.0, 0.0917, 0.9958], 'z': [0.1602, -0.9829, 0.0905], 'len_ipd': 0.0606, 'wgt': 'WGT-brow', 'scale': [8.73, 9.3, 12.2], 'palette': 'THEME09', 'cs_rot': [-1.5708, 0.1396, 0.0], 'cs_tr': [-0.002, -0.002, 0.005]}, 'CTL-Cheek_all.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.342, -0.9397, 0.0], 'len_ipd': 0.0588, 'wgt': 'WGT-Cheek', 'scale': [8.15, 8.15, 8.15], 'palette': 'THEME09', 'cs_rot': [-1.7279, 0.1222, 0.0], 'cs_tr': [0.0, 0.005, 0.004]}, 'CTL-Brow_in.L': {'dir': [0.0, 0.0917, 0.9958], 'z': [0.1602, -0.9829, 0.0905], 'len_ipd': 0.0606, 'wgt': 'WGT-triangle', 'scale': [1.71, 1.71, 1.71], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.013, 0.004]}, 'CTL-Cheek_in.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.0588, 'wgt': 'WGT-triangle', 'scale': [1.17, 1.17, 1.17], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.006]}, 'CTL-Brow_mid.L': {'dir': [0.0, 0.0917, 0.9958], 'z': [0.1602, -0.9829, 0.0905], 'len_ipd': 0.0606, 'wgt': 'WGT-triangle', 'scale': [1.71, 1.71, 1.71], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.013, 0.004]}, 'CTL-Cheek_mid.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.0588, 'wgt': 'WGT-triangle', 'scale': [1.17, 1.17, 1.17], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.006]}, 'CTL-Brow_out.L': {'dir': [0.0, 0.0917, 0.9958], 'z': [0.1602, -0.9829, 0.0905], 'len_ipd': 0.0606, 'wgt': 'WGT-triangle', 'scale': [1.71, 1.71, 1.71], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.013, 0.004]}, 'CTL-Cheek_out.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.0588, 'wgt': 'WGT-triangle', 'scale': [1.17, 1.17, 1.17], 'palette': 'THEME09', 'cs_rot': [1.5708, -0.6981, 3.1416], 'cs_tr': [0.0, 0.0, 0.006]}, 'CTL-Lips_local1_upp.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0263, 'wgt': 'WGT-triangle', 'scale': [1.0, 1.0, 1.0], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.001]}, 'CTL-Lips_local2_upp.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0263, 'wgt': 'WGT-triangle', 'scale': [1.0, 1.0, 1.0], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.001]}, 'CTL-Lips_local1_low.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0263, 'wgt': 'WGT-triangle', 'scale': [1.0, 1.0, 1.0], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.001]}, 'CTL-Lips_local2_low.L': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0263, 'wgt': 'WGT-triangle', 'scale': [1.0, 1.0, 1.0], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.001]}, 'CTL-Brow_all.R': {'dir': [0.0, 0.0917, 0.9958], 'z': [-0.1602, -0.9829, 0.0905], 'len_ipd': 0.0606, 'wgt': 'WGT-brow', 'scale': [-8.73, 9.3, 12.2], 'palette': 'THEME09', 'cs_rot': [-1.5708, -0.1396, -0.0], 'cs_tr': [0.002, -0.002, 0.005]}, 'CTL-Cheek_all.R': {'dir': [0.0, 0.0, 1.0], 'z': [-0.342, -0.9397, 0.0], 'len_ipd': 0.0588, 'wgt': 'WGT-Cheek', 'scale': [8.15, 8.15, 8.15], 'palette': 'THEME09', 'cs_rot': [-1.7279, -0.1222, 0.0], 'cs_tr': [0.0, 0.005, 0.004]}, 'CTL-Brow_in.R': {'dir': [0.0, 0.0917, 0.9958], 'z': [-0.1602, -0.9829, 0.0905], 'len_ipd': 0.0606, 'wgt': 'WGT-triangle', 'scale': [1.71, 1.71, 1.71], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.013, 0.004]}, 'CTL-Cheek_in.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.0588, 'wgt': 'WGT-triangle', 'scale': [1.17, 1.17, 1.17], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.006]}, 'CTL-Brow_mid.R': {'dir': [0.0, 0.0917, 0.9958], 'z': [-0.1602, -0.9829, 0.0905], 'len_ipd': 0.0606, 'wgt': 'WGT-triangle', 'scale': [1.71, 1.71, 1.71], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.013, 0.004]}, 'CTL-Cheek_mid.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.0588, 'wgt': 'WGT-triangle', 'scale': [1.17, 1.17, 1.17], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.006]}, 'CTL-Brow_out.R': {'dir': [0.0, 0.0917, 0.9958], 'z': [-0.1602, -0.9829, 0.0905], 'len_ipd': 0.0606, 'wgt': 'WGT-triangle', 'scale': [1.71, 1.71, 1.71], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.013, 0.004]}, 'CTL-Cheek_out.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, 0.0], 'len_ipd': 0.0588, 'wgt': 'WGT-triangle', 'scale': [1.17, 1.17, 1.17], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.6981, 3.1416], 'cs_tr': [0.0, 0.0, 0.006]}, 'CTL-Lips_local1_upp.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0263, 'wgt': 'WGT-triangle', 'scale': [1.0, 1.0, 1.0], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.001]}, 'CTL-Lips_local2_upp.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0263, 'wgt': 'WGT-triangle', 'scale': [1.0, 1.0, 1.0], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 0.0], 'cs_tr': [0.0, 0.0, 0.001]}, 'CTL-Lips_local1_low.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0263, 'wgt': 'WGT-triangle', 'scale': [1.0, 1.0, 1.0], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.001]}, 'CTL-Lips_local2_low.R': {'dir': [0.0, 0.0, 1.0], 'z': [0.0, -1.0, -0.0], 'len_ipd': 0.0263, 'wgt': 'WGT-triangle', 'scale': [1.0, 1.0, 1.0], 'palette': 'THEME09', 'cs_rot': [1.5708, 0.0, 3.1416], 'cs_tr': [0.0, 0.0, 0.001]}}

# eyelid controls: measured from Storm's P-Eyelid_mid_upp/low (tiny bone,
# huge rotated widget arc lying along the lid)
for _sd, _sy in ((".L", 1.0), (".R", -1.0)):
    STORM_SPEC["CTL-Lid_upp" + _sd] = {
        "dir": [-0.028 * _sy, 0.0, 1.0], "z": [0.028 * _sy, -1.0, 0.001],
        "len_ipd": 0.0083, "wgt": "WGT-Eyelid",
        "scale": [56.5, 56.5, 56.5], "palette": 'THEME09',
        "cs_rot": [0.0, 1.7977 * _sy, 0.0], "cs_tr": [0.0, 0.002, 0.002]}
    STORM_SPEC["CTL-Lid_low" + _sd] = {
        "dir": [-0.021 * _sy, 0.0, 1.0], "z": [0.021 * _sy, -1.0, 0.0],
        "len_ipd": 0.0095, "wgt": "WGT-Eyelid",
        "scale": [33.1, 33.1, 45.4], "palette": 'THEME09',
        "cs_rot": [3.1416, 1.8151 * _sy, 0.0], "cs_tr": [0.0, 0.0, 0.003]}



def _apply_storm_spec(rig, ipd):
    """Post-pass: re-orient + re-size every face bone per STORM_SPEC and
    re-assign its widget with Storm's exact scale/palette."""
    from . import face_widgets as _fw
    from mathutils import Vector as _V
    eb = rig.data.edit_bones          # caller must be in EDIT mode
    for name, sp in STORM_SPEC.items():
        b = eb.get(name)
        if b is None:
            continue
        d = _V(sp["dir"]).normalized()
        ln = max(sp["len_ipd"] * ipd, 1e-4)
        b.tail = b.head + d * ln
        try:
            b.align_roll(_V(sp["z"]))
        except Exception:
            pass
    # constraint pairs MUST share the rest orientation, or COPY_TRANSFORMS
    # bends the mesh at rest: keep DEF-jaw identical to CTL-Jaw
    cj, dj = eb.get("CTL-Jaw"), eb.get("DEF-jaw")
    if cj is not None and dj is not None:
        dj.head = cj.head.copy()
        dj.tail = cj.tail.copy()
        dj.roll = cj.roll


def _assign_storm_widgets(rig):
    """Storm's widgets rely on custom_shape ROTATION + TRANSLATION (the
    triangles are rotated 90 deg, the lower-lip ones flipped 180) - dropping
    them leaves every shape lying flat / facing the wrong way."""
    from . import face_widgets as _fw
    for name, sp in STORM_SPEC.items():
        pb = rig.pose.bones.get(name)
        if pb is None:
            continue
        if sp["wgt"]:
            pal = sp["palette"] if sp["palette"] != 'DEFAULT' else None
            _fw.assign(rig, name, sp["wgt"], tuple(sp["scale"]), pal)
        try:
            pb.custom_shape_rotation_euler = sp.get("cs_rot", (0, 0, 0))
            pb.custom_shape_translation = sp.get("cs_tr", (0, 0, 0))
        except Exception:
            pass

# The full Storm face-control layout (brows, eyelids, cheeks, lips with
# locals + corners, nose), positioned from the SR_FaceGrid, each control
# carrying a DEF- twin (child of the control) whose weights are carved
# radially out of the surrounding deform weights. Bone-based recipe - the
# shape-key recipes layer on top of the same controls later.

FACE_LOCAL_DEFS = []      # filled by _control_table; used to strip on rebuild


def _control_table(gp, ipd, P_jaw, head_key):
    """[(ctl_name, pos, widget, scale, palette, parent_key, def_radius)]
    parent_key: 'head' | 'jaw' | 'face_up' | 'face_low' | 'mouth' |
    'eye.L' | 'eye.R' | 'corn.L' | 'corn.R' | another ctl name."""
    import numpy as _np

    def lerp(a, b, t):
        return a + (b - a) * t

    T = []
    u = ipd
    brow_c = gp["brow_c"]
    mouth_mid = (gp["lip_T"] + gp["lip_B"]) / 2.0

    # masters (Storm: WGT-Circle, THEME10)
    T.append(("MSTR-Face_upp", _np.array([0.0, float(P_jaw[1]), float(brow_c[2])]),
              "WGT-Circle", 5.4, 'THEME10', 'head', 0.0))
    T.append(("MSTR-Face_low", _np.array([0.0, float(P_jaw[1]), float(mouth_mid[2])]),
              "WGT-Circle", 5.4, 'THEME10', 'head', 0.0))

    for sd in (".L", ".R"):
        # ---- brows: all + in/mid/out ----
        T.append(("CTL-Brow_all" + sd, gp["brow_mid" + sd], "WGT-brow",
                  (9.2 if sd == ".L" else -9.2, 9.2, 9.2), 'THEME09',
                  'face_up', 0.0))
        for part in ("in", "mid", "out"):
            T.append(("CTL-Brow_%s%s" % (part, sd), gp["brow_%s%s" % (part, sd)],
                      "WGT-triangle", 1.8, 'THEME09',
                      "CTL-Brow_all" + sd, 0.38 * u))
        # ---- eyelids ----
        T.append(("CTL-Lid_upp" + sd, gp["lid_T" + sd], "WGT-Eyelid",
                  (5.0 if sd == ".L" else -5.0, 5.0, 5.0), 'THEME09',
                  'eye' + sd, 0.25 * u))
        T.append(("CTL-Lid_low" + sd, gp["lid_B" + sd], "WGT-Eyelid",
                  (5.0 if sd == ".L" else -5.0, -5.0, 5.0), 'THEME09',
                  'eye' + sd, 0.25 * u))
        # ---- cheeks: all + in/mid/out ----
        cheek_all = lerp(gp["cheek_up" + sd], gp["cheek_low" + sd], 0.5)
        T.append(("CTL-Cheek_all" + sd, cheek_all, "WGT-Cheek",
                  (8.3 if sd == ".L" else -8.3, 8.3, 8.3), 'THEME09',
                  'face_low', 0.0))
        cin = lerp(gp["nose_side" + sd], gp["cheek_up" + sd], 0.55)
        cout = lerp(gp["cheek_up" + sd], gp["ear_low" + sd], 0.42)
        for part, pos in (("in", cin), ("mid", gp["cheek_up" + sd]), ("out", cout)):
            T.append(("CTL-Cheek_%s%s" % (part, sd), pos, "WGT-triangle",
                      1.8, 'THEME09', "CTL-Cheek_all" + sd, 0.42 * u))
        # ---- lips locals + corners (UPPER = red, LOWER = blue, count
        #      configurable via face_lip_ctls) ----
        _nl = int(getattr(bpy.context.scene.smartrig, "face_lip_ctls", 2))
        for lvl, cen_key in (("upp", "lip_T"), ("low", "lip_B")):
            _pal = 'THEME01' if lvl == "upp" else 'THEME04'
            for k in range(1, _nl + 1):
                t = k / (_nl + 1.0)
                pos = lerp(gp[cen_key], gp["mouth_corner" + sd], t)
                T.append(("CTL-Lips_local%d_%s%s" % (k, lvl, sd), pos,
                          "WGT-triangle", 1.4, _pal,
                          "CTL-Lips_main_" + lvl, 0.16 * u))
        T.append(("CTL-Lips_corn" + sd, gp["mouth_corner" + sd],
                  "WGT-Mouth_corner", (3.7 if sd == ".L" else -3.7, 3.7, 3.7),
                  'THEME09', 'corn' + sd, 0.22 * u))

    # ---- lips mains (center) ----
    T.append(("CTL-Lips_main_upp", gp["lip_T"], "WGT-lips_main",
              (8.3, -4.6, 8.3), 'THEME01', 'mouth', 0.30 * u))
    T.append(("CTL-Lips_main_low", gp["lip_B"], "WGT-lips_main",
              (8.3, 6.2, 8.3), 'THEME04', 'jaw', 0.30 * u))
    # ---- nose ----
    T.append(("MSTR-Nose", gp["nose_base"], "WGT-Circle", 5.8, 'THEME03',
              'face_low', 0.40 * u))
    return T


def build_controls(props, context, rig, parent_name, ipd, P_jaw):
    """Create the Storm control layout + DEF twins from the face grid."""
    from . import face_widgets as _fw
    gp = grid_points()
    if not gp:
        return 0
    T = _control_table(gp, ipd, P_jaw, parent_name)

    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = rig
    rig.hide_set(False)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones

    # idempotent: drop previous controls + DEF twins + corner MCHs
    names = [row[0] for row in T]
    drop = names + ["DEF" + n[3:] for n in names if n.startswith("CTL")] + \
           ["MCH-Lips_corn.L", "MCH-Lips_corn.R", "DEF-Nose"]
    # the count may have CHANGED: purge every existing lip local + twin
    drop += [b.name for b in eb
             if b.name.startswith(("CTL-Lips_local", "DEF-Lips_local"))]
    for n in drop:
        if n in eb:
            eb.remove(eb[n])

    # corner MCH parents (skull-based, jaw applied 50% via constraint)
    mchs = {}
    for sd in (".L", ".R"):
        m = eb.new("MCH-Lips_corn" + sd)
        pos = gp["mouth_corner" + sd]
        m.head = Vector([float(v) for v in pos])
        m.tail = m.head + Vector((0.0, 0.0, 0.04 * ipd))
        m.use_deform = False
        if parent_name and parent_name in eb:
            m.parent = eb[parent_name]
        mchs["corn" + sd] = m.name

    def parent_of(key):
        return {'head': parent_name, 'jaw': "DEF-jaw",
                'face_up': "MSTR-Face_upp", 'face_low': "MSTR-Face_low",
                'mouth': "MSTR-Mouth", 'eye.L': "MSTR-Eye.L",
                'eye.R': "MSTR-Eye.R", 'corn.L': mchs.get("corn.L"),
                'corn.R': mchs.get("corn.R")}.get(key, key)

    blen = 0.06 * ipd
    made = []
    for name, pos, wgt, scale, pal, pkey, rad in T:
        b = eb.new(name)
        b.head = Vector([float(v) for v in pos])
        b.tail = b.head + Vector((0.0, 0.0, blen))
        b.use_deform = False
        pn = parent_of(pkey)
        if pn and pn in eb:
            b.parent = eb[pn]
        made.append((name, wgt, scale, pal, rad))
        if rad > 0.0:
            dn = "DEF" + name[3:] if name.startswith("CTL") else                  "DEF-" + name.split("-", 1)[1]
            d = eb.new(dn)
            d.head = b.head.copy()
            d.tail = b.tail.copy()
            d.use_deform = True
            d.parent = b

    # bone collections
    try:
        col_c = utils.bone_collection(rig.data, "Face")
        col_m = utils.bone_collection(rig.data, "Face (MCH)")
        for name, _w, _s, _p, rad in made:
            if name in eb:
                col_c.assign(eb[name])
        for n in list(mchs.values()):
            col_m.assign(eb[n])
        for name, _w, _s, _p, rad in made:
            dn = "DEF" + name[3:] if name.startswith("CTL") else                  "DEF-" + name.split("-", 1)[1]
            if dn in eb:
                col_m.assign(eb[dn])
    except Exception:
        pass

    _apply_storm_spec(rig, ipd)

    bpy.ops.object.mode_set(mode='POSE')
    # jaw follows 50% on the corner MCHs (Storm's MSTR-lips mechanism)
    for sd in (".L", ".R"):
        pb = rig.pose.bones.get("MCH-Lips_corn" + sd)
        if pb is None:
            continue
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        con = pb.constraints.new('COPY_TRANSFORMS')
        con.name = "SR Corner Jaw 50"
        con.target = rig
        con.subtarget = "CTL-Jaw"
        con.target_space = 'LOCAL_OWNER_ORIENT'
        con.owner_space = 'LOCAL'
        con.mix_mode = 'BEFORE'
        con.influence = 0.5
    # widgets + palettes (table defaults, then Storm's exact spec on top)
    for name, wgt, scale, pal, _rad in made:
        _fw.assign(rig, name, wgt, scale, pal)
    _assign_storm_widgets(rig)
    bpy.ops.object.mode_set(mode='OBJECT')
    if prev_active is not None:
        context.view_layer.objects.active = prev_active
    return len(made)


def _local_def_names():
    out = []
    for sd in (".L", ".R"):
        for part in ("in", "mid", "out"):
            out += ["DEF-Brow_%s%s" % (part, sd), "DEF-Cheek_%s%s" % (part, sd)]
        out += ["DEF-Lid_upp" + sd, "DEF-Lid_low" + sd, "DEF-Lips_corn" + sd]
        _nl = int(getattr(bpy.context.scene.smartrig, "face_lip_ctls", 2))
        for lvl in ("upp", "low"):
            for k in range(1, _nl + 1):
                out.append("DEF-Lips_local%d_%s%s" % (k, lvl, sd))
    out += ["DEF-Lips_main_upp", "DEF-Lips_main_low", "DEF-Nose"]
    return out


def local_weights(props, context, rig):
    """Radial weights for every local DEF twin, carved PROPORTIONALLY from
    the deform weights already on each vertex (head/jaw/neck...) so totals
    stay normalized. Rebuild-safe: previous local groups are stripped and
    the remaining deform weights renormalized first."""
    body = getattr(props, "target_mesh", None) or context.active_object
    gp = grid_points()
    if not gp:
        return 0
    eyeL, eyeR = _lm("face_eye.L"), _lm("face_eye.R")
    ipd = float(np.linalg.norm(eyeL - eyeR))
    w = utils.read_rest_coords(body)
    me = body.data
    deform = {b.name for b in rig.data.bones if b.use_deform}
    locals_ = [n for n in _local_def_names() if n in deform]

    # ---- strip old local groups + renormalize the rest ----
    lidx = set()
    for n in locals_:
        g = body.vertex_groups.get(n)
        if g is not None:
            lidx.add(g.index)
    gidx = {g.index: g for g in body.vertex_groups}
    dgi = {g.index for g in body.vertex_groups if g.name in deform}
    if lidx:
        for v in me.vertices:
            tot_l = sum(g.weight for g in v.groups if g.group in lidx)
            if tot_l <= 1e-6:
                continue
            rest = [(g.group, g.weight) for g in v.groups
                    if g.group in dgi and g.group not in lidx]
            tot_r = sum(x[1] for x in rest)
            if tot_r > 1e-6:
                f = (tot_r + tot_l) / tot_r
                for gi, gw in rest:
                    gidx[gi].add([v.index], gw * f, 'REPLACE')
            for gi in lidx:
                if gi in gidx:
                    gidx[gi].remove([v.index])

    # ---- control centers + radii (match _control_table) ----
    def lerp(a, b, t):
        return a + (b - a) * t
    fields = {}
    u = ipd
    for sd in (".L", ".R"):
        for part in ("in", "mid", "out"):
            fields["DEF-Brow_%s%s" % (part, sd)] = (gp["brow_%s%s" % (part, sd)], 0.38 * u)
        fields["DEF-Lid_upp" + sd] = (gp["lid_T" + sd], 0.25 * u)
        fields["DEF-Lid_low" + sd] = (gp["lid_B" + sd], 0.25 * u)
        cin = lerp(gp["nose_side" + sd], gp["cheek_up" + sd], 0.55)
        cout = lerp(gp["cheek_up" + sd], gp["ear_low" + sd], 0.42)
        fields["DEF-Cheek_in" + sd] = (cin, 0.42 * u)
        fields["DEF-Cheek_mid" + sd] = (gp["cheek_up" + sd], 0.42 * u)
        fields["DEF-Cheek_out" + sd] = (cout, 0.42 * u)
        _nl = int(getattr(bpy.context.scene.smartrig, "face_lip_ctls", 2))
        for lvl, cen_key in (("upp", "lip_T"), ("low", "lip_B")):
            for k in range(1, _nl + 1):
                t = k / (_nl + 1.0)
                fields["DEF-Lips_local%d_%s%s" % (k, lvl, sd)] = (
                    lerp(gp[cen_key], gp["mouth_corner" + sd], t), 0.16 * u)
        fields["DEF-Lips_corn" + sd] = (gp["mouth_corner" + sd], 0.22 * u)
    fields["DEF-Lips_main_upp"] = (gp["lip_T"], 0.30 * u)
    fields["DEF-Lips_main_low"] = (gp["lip_B"], 0.30 * u)
    fields["DEF-Nose"] = (lerp(gp["nose_base"], gp["nose_tip"], 0.6), 0.40 * u)

    # cap: a local never takes more than this fraction of a vertex
    CAP = 0.65
    n_assigned = 0
    for name in locals_:
        cen, rad = fields.get(name, (None, 0.0))
        if cen is None:
            continue
        g = body.vertex_groups.get(name)
        if g is None:
            g = body.vertex_groups.new(name=name)
        dist = np.linalg.norm(w - np.asarray(cen, dtype=float), axis=1)
        f = 1.0 - _sstep(np.clip(dist / max(rad, 1e-6), 0.0, 1.0))
        f *= CAP
        gi_new = g.index
        dgi2 = {gg.index for gg in body.vertex_groups
                if gg.name in deform and gg.index != gi_new}
        gmap = {gg.index: gg for gg in body.vertex_groups}
        idxs = np.where(f > 1e-3)[0]
        for i in idxs:
            v = me.vertices[int(i)]
            fi = float(f[i])
            tot = sum(gg.weight for gg in v.groups if gg.group in dgi2)
            if tot <= 1e-6:
                continue                    # no deform weight here to share
            for gg in list(v.groups):
                if gg.group in dgi2 and gg.weight > 0.0:
                    gmap[gg.group].add([int(i)], gg.weight * (1.0 - fi),
                                       'REPLACE')
            g.add([int(i)], fi * tot, 'REPLACE')
            n_assigned += 1
    return n_assigned


# ------------------------------------------------------------------ operators
class SMARTRIG_OT_face_detect(bpy.types.Operator):
    bl_idname = "smartrig.face_detect"
    bl_label = "Detect Face"
    bl_description = ("Analyze the head geometry and place the face markers "
                      "automatically (eyes, brows, nose, lips, mouth corners, "
                      "jaw pivot, chin). Adjust any marker, then Build")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.smartrig
        ob = getattr(props, "target_mesh", None) or context.active_object
        return ob is not None and ob.type == 'MESH'

    def execute(self, context):
        props = context.scene.smartrig
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        body = getattr(props, "target_mesh", None) or context.active_object
        try:
            from . import skirt as _sk
            _sk._facial_autodetect(props, context)   # fill the eye slots by name
        except Exception:
            pass
        try:
            L, ipd, sure = detect_landmarks(props, body)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        place_markers(L, ipd)
        if sure:
            self.report({'INFO'}, "Face markers placed from the head geometry "
                        "- adjust any marker, then Build Face Base")
        else:
            self.report({'WARNING'}, "No eye meshes found - markers are a "
                        "proportional guess, please adjust them")
        return {'FINISHED'}


# professional widget sizing: max WORLD size per control, in IPD units -
# the addon knows the character's size and the widgets follow it
_WGT_CAPS = (
    ("CTL-Jaw", 0.55), ("MSTR-Face_upp", 0.60), ("MSTR-Face_low", 0.60),
    ("MSTR-Eye_target", 0.45), ("CTL-Brow_all", 0.30), ("CTL-Cheek_all", 0.25),
    ("MSTR-Mouth", 0.30), ("MSTR-Eye.", 0.22), ("MSTR-Nose", 0.18),
    ("CTL-Eye_target", 0.16), ("CTL-", 0.16), ("MSTR-", 0.25),
)


def organize_face_layers(rig):
    """PROFESSIONAL bone layers for the face (Saeed's spec): primary controls
    under Face (Primary), tweaks under Face (Secondary), everything under
    Face; DEF twins + MCH helpers into a HIDDEN collection. Matches the
    body's Rig Layers discipline (v1.91)."""
    data = rig.data

    def coll(name, visible=True):
        c = data.collections.get(name)
        if c is None:
            c = data.collections.new(name)
        c.is_visible = visible
        return c

    c_face = coll("Face")
    c_pri = coll("Face (Primary)")
    c_sec = coll("Face (Secondary)")
    c_mch = coll("Face (MCH)", visible=False)
    PRIMARY = ("MSTR-", "CTL-Jaw", "CTL-Brow_all", "CTL-Cheek_all",
               "CTL-Eye_target")
    n = 0
    for b in data.bones:
        nm = b.name
        is_ctl = nm.startswith(("CTL-", "MSTR-"))
        is_help = (nm.startswith(("MCH-Lips_corn", "DEF-Brow_", "DEF-Lid_",
                                  "DEF-Cheek_", "DEF-Lips_", "DEF-Nose",
                                  "DEF-Eye_target", "DEF-jaw", "DEF-ear",
                                  "DEF-eye"))
                   or nm in ("DEF-Jaw",))
        if not (is_ctl or is_help):
            continue
        # remove from every collection first (no strays in Torso etc.)
        for c in list(b.collections):
            try:
                c.unassign(b)
            except Exception:
                pass
        if is_ctl:
            c_face.assign(b)
            if any(nm.startswith(p) for p in PRIMARY):
                c_pri.assign(b)
            else:
                c_sec.assign(b)
        else:
            c_mch.assign(b)
        n += 1
    return n


def separate_lip_widgets(rig, ipd):
    """Thin lips = upper/lower lip controls overlap on the mouth line: nudge
    the DRAWN widgets up/down (bones + weights stay put) so upper (red)
    reads above the line and lower (blue) below."""
    off = 0.14 * ipd
    n = 0
    for pb in rig.pose.bones:
        nm = pb.name
        if not nm.startswith(("CTL-Lips_local", "CTL-Lips_main")):
            continue
        up = "_upp" in nm
        try:
            t = pb.custom_shape_translation
            pb.custom_shape_translation = (t[0], off if up else -off, t[2])
            n += 1
        except Exception:
            pass
    return n


def normalize_widget_sizes(rig, ipd):
    """Clamp every face control's DRAWN size to a character-proportional cap
    (Storm's cs_scale numbers were tuned for the Storm character - on a
    different head they explode; Saeed: widgets must fit the character)."""
    import numpy as np
    n = 0
    for pb in rig.pose.bones:
        if not pb.name.startswith(("CTL-", "MSTR-")):
            continue
        shp = pb.custom_shape
        if shp is None or shp.type != 'MESH' or pb.length < 1e-6:
            continue
        try:
            vs = np.array([list(v.co) for v in shp.data.vertices])
            if not len(vs):
                continue
            rad = float(np.abs(vs).max())
            cs = max(abs(v) for v in pb.custom_shape_scale_xyz)
            world = pb.length * cs * rad
            cap_mult = next((m for pre, m in _WGT_CAPS
                             if pb.name.startswith(pre)), 0.2)
            cap = cap_mult * ipd
            if world > cap and world > 1e-9:
                f = cap / world
                s = pb.custom_shape_scale_xyz
                pb.custom_shape_scale_xyz = (s[0] * f, s[1] * f, s[2] * f)
                # pull the shape's offset in proportionally too
                try:
                    t = pb.custom_shape_translation
                    pb.custom_shape_translation = (t[0] * f, t[1] * f, t[2] * f)
                except Exception:
                    pass
                n += 1
        except Exception:
            pass
    return n


def _face_rigs():
    """Every armature the face flow cares about (metarig / generated / face)."""
    out = []
    for nm in ("SR_Metarig", "SR_Rig", FACE_RIG_NAME):
        ob = bpy.data.objects.get(nm)
        if ob is not None and ob.type == 'ARMATURE':
            out.append(ob)
    try:
        from . import metarig as _mr
        g = _mr._generated_rig()
        if g is not None and g not in out:
            out.append(g)
    except Exception:
        pass
    return out


def set_rigs_hidden(hide):
    n = 0
    for ob in _face_rigs():
        try:
            ob.hide_set(bool(hide)); n += 1
        except Exception:
            pass
    return n


class SMARTRIG_OT_toggle_bones(bpy.types.Operator):
    bl_idname = "smartrig.toggle_bones"
    bl_label = "Show / Hide Bones"
    bl_description = ("Quickly hide or show every rig (metarig, generated "
                      "rig, face rig) - e.g. while placing the face markers")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rigs = _face_rigs()
        if not rigs:
            self.report({'INFO'}, "No rig in the scene yet")
            return {'CANCELLED'}
        hide = not all(ob.hide_get() for ob in rigs)
        set_rigs_hidden(hide)
        self.report({'INFO'}, "Bones hidden" if hide else "Bones visible")
        return {'FINISHED'}


class SMARTRIG_OT_face_objects_detect(bpy.types.Operator):
    bl_idname = "smartrig.face_objects_detect"
    bl_label = "Auto Detect Face Objects"
    bl_description = ("Fill the face object slots automatically (FaceIt-style "
                      "register): eyes from the eyeball detector, teeth / "
                      "tongue / brows / eyelashes / hair by name. Fix any "
                      "slot by hand after")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        found = []
        try:
            from . import skirt as _sk
            _sk._facial_autodetect(props, context)      # eyes (validated)
            if props.skin_eye_l: found.append("Eye L")
            if props.skin_eye_r: found.append("Eye R")
        except Exception:
            pass
        body = getattr(props, "target_mesh", None)

        def _real_mesh(o):
            """A character mesh - NEVER a rig widget / helper object."""
            if o is None or o.type != 'MESH' or o is body:
                return False
            n = o.name
            if n.startswith(("WGT", "WGTS", "SR_", "CTL", "MCH", "DEF", "ORG")):
                return False
            for c in o.users_collection:
                if c.name.startswith(("WGT", "WGTS")):
                    return False
            # EXCLUDED collections (e.g. an "idle pose" copy) are not in the
            # view layer - their duplicates must never win a slot
            if o.name not in context.view_layer.objects:
                return False
            return True

        scene_meshes = [o for o in context.scene.objects if _real_mesh(o)]
        # the eye detector can grab widgets too (orphan-mesh trap): heal
        for attr in ("skin_eye_l", "skin_eye_r"):
            cur = getattr(props, attr, None)
            if cur is not None and not _real_mesh(cur):
                setattr(props, attr, None)
        if props.skin_eye_l is None or props.skin_eye_r is None:
            eye = next((o for o in scene_meshes if "eye" in o.name.lower()
                        and "brow" not in o.name.lower()
                        and "lash" not in o.name.lower()), None)
            if eye is not None:
                if props.skin_eye_l is None:
                    props.skin_eye_l = eye
                if props.skin_eye_r is None:
                    # combined-eye meshes are handled downstream by the
                    # sphere-fit L/R split - same mesh in both slots is fine
                    props.skin_eye_r = eye
                found.append("Eyes")

        def by_name(*keys, exclude=()):
            for o in scene_meshes:
                n = o.name.lower()
                if any(k in n for k in keys) and not any(e in n for e in exclude):
                    return o
            return None

        combined = by_name("teeth_and_tongue", "teeth_tongue")
        t_up = by_name("teeth", exclude=("low", "bottom", "tongue")) \
            if combined is None else combined
        t_low = (by_name("teeth") if combined is None else combined)
        low_named = by_name("lower teeth", "teeth_low", "bottom teeth")
        if low_named is not None:
            t_low = low_named
        for attr, ob, label in (("skin_teeth_up", t_up, "Teeth Up"),
                                ("skin_teeth_low", t_low, "Teeth Low"),
                                ("skin_tongue", combined or by_name("tongue", "toung"), "Tongue"),
                                ("skin_brows", by_name("brow"), "Brows"),
                                ("skin_lashes", by_name("lash"), "Eyelashes"),
                                ("skin_hair", by_name("hair", exclude=("brow", "lash")), "Hair")):
            if ob is not None and getattr(props, attr, None) is None:
                try:
                    setattr(props, attr, ob); found.append(label)
                except Exception:
                    pass
        if combined is not None:
            self.report({'WARNING'}, "One combined teeth+tongue mesh found - "
                        "assigned to all three slots (split it for best jaw "
                        "results). Detected: " + ", ".join(found))
        elif found:
            self.report({'INFO'}, "Registered: " + ", ".join(found))
        else:
            self.report({'WARNING'}, "Nothing new detected - assign the "
                        "slots manually")
        return {'FINISHED'}


# fixed vertex-group names for parts MERGED into the face mesh (FaceIt-style
# component registration). Presence of the group = the part is registered.
FACE_PART_VG = {
    'BROWS': "SR_brows", 'LASHES': "SR_lashes",
    'TEETH_UP': "SR_teeth_up", 'TEETH_LOW': "SR_teeth_low",
    'TONGUE': "SR_tongue", 'EYE_L': "SR_eye_l", 'EYE_R': "SR_eye_r",
    'HAIR': "SR_hair",
}
FACE_PART_SLOT = {
    'EYE_L': "skin_eye_l", 'EYE_R': "skin_eye_r",
    'TEETH_UP': "skin_teeth_up", 'TEETH_LOW': "skin_teeth_low",
    'TONGUE': "skin_tongue", 'BROWS': "skin_brows",
    'LASHES': "skin_lashes", 'HAIR': "skin_hair",
}


def face_part_host(props, part):
    """The mesh object that carries this part's SR_* vertex group (in-mesh
    registration).  Searched EVERYWHERE the user may have registered it:
    the main face mesh, any slot object (combined teeth_and_tongue case),
    then any scene mesh.  None = not registered in-mesh."""
    vgn = FACE_PART_VG.get(part)
    if not vgn:
        return None
    cands = []
    body = getattr(props, "target_mesh", None)
    if body is not None:
        cands.append(body)
    for attr in FACE_PART_SLOT.values():
        ob = getattr(props, attr, None)
        if ob is not None and ob not in cands:
            cands.append(ob)
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob not in cands and \
                not ob.name.startswith(("WGT", "SR_", "HLP-", "GEO-")):
            cands.append(ob)
    for ob in cands:
        if ob.type == 'MESH' and ob.vertex_groups.get(vgn) is not None:
            return ob
    return None


class SMARTRIG_OT_face_register_part(bpy.types.Operator):
    bl_idname = "smartrig.face_register_part"
    bl_label = "Register Selection As"
    bl_description = ("The part is MERGED into the face mesh? Select its "
                      "vertices in Edit Mode and register the selection as "
                      "that part (stored as a vertex group, exactly like "
                      "FaceIt) - works for brows, lashes, teeth, tongue, "
                      "eyes")
    bl_options = {'REGISTER', 'UNDO'}
    part: bpy.props.EnumProperty(
        name="Part",
        items=(('BROWS', "Brows", ""), ('LASHES', "Eyelashes", ""),
               ('TEETH_UP', "Teeth Up", ""), ('TEETH_LOW', "Teeth Low", ""),
               ('TONGUE', "Tongue", ""), ('EYE_L', "Eye L", ""),
               ('EYE_R', "Eye R", ""), ('HAIR', "Hair", "")),
        default='BROWS')

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH'
                and context.edit_object is not None
                and context.edit_object.type == 'MESH')

    def execute(self, context):
        ob = context.edit_object
        import bmesh
        bm = bmesh.from_edit_mesh(ob.data)
        idx = [v.index for v in bm.verts if v.select]
        if len(idx) < 3:
            self.report({'ERROR'}, "Select the part's vertices first "
                        "(Edit Mode, e.g. hover + L for a linked island)")
            return {'CANCELLED'}
        vg_name = FACE_PART_VG[self.part]
        bpy.ops.object.mode_set(mode='OBJECT')
        vg = ob.vertex_groups.get(vg_name)
        if vg is None:
            vg = ob.vertex_groups.new(name=vg_name)
        else:
            vg.remove(range(len(ob.data.vertices)))     # re-register = replace
        vg.add(idx, 1.0, 'REPLACE')
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, "%s registered: %d vertices on '%s' (vertex "
                    "group %s)" % (self.part.title().replace("_", " "),
                                   len(idx), ob.name, vg_name))
        return {'FINISHED'}


class SMARTRIG_OT_face_register_slot(bpy.types.Operator):
    bl_idname = "smartrig.face_register_slot"
    bl_label = "Register"
    bl_description = ("Register THIS part from what you have selected - "
                      "works both ways, your choice:\n"
                      "- Object Mode: the selected mesh object goes into "
                      "this slot\n"
                      "- Edit Mode: the selected VERTICES are registered as "
                      "this part (for parts merged into the face mesh)")
    bl_options = {'REGISTER', 'UNDO'}
    part: bpy.props.EnumProperty(
        items=[(k, k.title().replace("_", " "), "") for k in FACE_PART_SLOT],
        default='EYE_L')

    def execute(self, context):
        props = context.scene.smartrig
        label = self.part.title().replace("_", " ")
        # ---- Edit Mode: vertex-selection registration (merged parts) ----
        if context.mode == 'EDIT_MESH':
            try:
                return bpy.ops.smartrig.face_register_part(part=self.part)
            except Exception as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
        # ---- Object Mode: selected object -> this slot ----
        body = getattr(props, "target_mesh", None)
        cand = [o for o in context.selected_objects
                if o.type == 'MESH' and o is not body
                and not o.name.startswith(("WGT", "WGTS", "SR_"))
                and o.name in context.view_layer.objects]
        ao = context.active_object
        ob = ao if (ao in cand) else (cand[0] if cand else None)
        if ob is None:
            self.report({'ERROR'}, "Select the part's mesh object first "
                        "(or its vertices in Edit Mode)")
            return {'CANCELLED'}
        setattr(props, FACE_PART_SLOT[self.part], ob)
        self.report({'INFO'}, "%s registered: %s" % (label, ob.name))
        return {'FINISHED'}


class SMARTRIG_OT_face_unregister_part(bpy.types.Operator):
    bl_idname = "smartrig.face_unregister_part"
    bl_label = "Unregister Part"
    bl_description = "Remove this in-mesh registration (deletes its SR_* vertex group)"
    bl_options = {'REGISTER', 'UNDO'}
    part: bpy.props.EnumProperty(
        items=[(k, k.title().replace("_", " "), "") for k in FACE_PART_SLOT],
        default='BROWS')

    def execute(self, context):
        props = context.scene.smartrig
        host = face_part_host(props, self.part)
        if host is None:
            return {'CANCELLED'}
        vg = host.vertex_groups.get(FACE_PART_VG[self.part])
        if vg is not None:
            host.vertex_groups.remove(vg)
            self.report({'INFO'}, "%s unregistered (was on '%s')"
                        % (self.part.title().replace("_", " "), host.name))
        return {'FINISHED'}


class SMARTRIG_OT_face_register_selected(bpy.types.Operator):
    bl_idname = "smartrig.face_register_selected"
    bl_label = "Register Selected"
    bl_description = ("FaceIt-style register: SELECT the face part meshes in "
                      "the viewport (separate eyes, separate upper/lower "
                      "teeth, tongue, brows, eyelashes...) and this sorts "
                      "them into the right slots automatically - eyes L/R by "
                      "position, teeth upper/lower by height")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        props = context.scene.smartrig
        body = getattr(props, "target_mesh", None)
        sel = [o for o in context.selected_objects
               if o.type == 'MESH' and o is not body
               and not o.name.startswith(("WGT", "WGTS", "SR_"))]
        if not sel:
            self.report({'ERROR'}, "Select the face part meshes first "
                        "(eyes / teeth / tongue / brows / lashes)")
            return {'CANCELLED'}

        def mean_x(o):
            import numpy as _np
            co = utils.read_world_coords(o)
            return float(_np.mean(co[:, 0]))

        def mean_z(o):
            import numpy as _np
            co = utils.read_world_coords(o)
            return float(_np.mean(co[:, 2]))

        done = []
        rest = []
        for o in sel:
            n = o.name.lower()
            if "tongue" in n or "toung" in n:
                props.skin_tongue = o; done.append("Tongue: " + o.name)
            elif "brow" in n:
                props.skin_brows = o; done.append("Brows: " + o.name)
            elif "lash" in n:
                props.skin_lashes = o; done.append("Eyelashes: " + o.name)
            elif "hair" in n:
                props.skin_hair = o; done.append("Hair: " + o.name)
            else:
                rest.append(o)
        eyes = [o for o in rest if "eye" in o.name.lower()]
        teeth = [o for o in rest if "teeth" in o.name.lower()
                 or "tooth" in o.name.lower()]
        other = [o for o in rest if o not in eyes and o not in teeth]
        # eyes: L/R by name suffix first, else by X position (+X = .L)
        if len(eyes) >= 2:
            eyes.sort(key=mean_x)                # min x = R, max x = L
            props.skin_eye_r, props.skin_eye_l = eyes[0], eyes[-1]
            done.append("Eye L: %s / Eye R: %s" % (eyes[-1].name, eyes[0].name))
        elif len(eyes) == 1:
            nl = eyes[0].name.lower()
            if nl.endswith((".r", "_r", " r")) or "right" in nl:
                props.skin_eye_r = eyes[0]; done.append("Eye R: " + eyes[0].name)
            elif nl.endswith((".l", "_l", " l")) or "left" in nl:
                props.skin_eye_l = eyes[0]; done.append("Eye L: " + eyes[0].name)
            else:                                # combined both-eyes mesh
                props.skin_eye_l = props.skin_eye_r = eyes[0]
                done.append("Eyes (combined): " + eyes[0].name)
        # teeth: upper/lower by name first, else by height
        if len(teeth) >= 2:
            teeth.sort(key=mean_z)               # min z = lower
            props.skin_teeth_low, props.skin_teeth_up = teeth[0], teeth[-1]
            done.append("Teeth Up: %s / Low: %s" % (teeth[-1].name, teeth[0].name))
        elif len(teeth) == 1:
            nl = teeth[0].name.lower()
            if any(k in nl for k in ("low", "bottom", "inf")):
                props.skin_teeth_low = teeth[0]
                done.append("Teeth Low: " + teeth[0].name)
            elif any(k in nl for k in ("up", "top", "sup")):
                props.skin_teeth_up = teeth[0]
                done.append("Teeth Up: " + teeth[0].name)
            else:
                props.skin_teeth_up = props.skin_teeth_low = teeth[0]
                done.append("Teeth (combined): " + teeth[0].name)
        if other:
            self.report({'WARNING'}, "Registered: %s | Unrecognized (rename "
                        "or use the dropdowns): %s"
                        % ("; ".join(done) if done else "nothing",
                           ", ".join(o.name for o in other)))
        elif done:
            self.report({'INFO'}, "Registered: " + "; ".join(done))
        else:
            self.report({'WARNING'}, "Nothing recognized - use the dropdowns")
        return {'FINISHED'}


class SMARTRIG_OT_face_place(bpy.types.Operator):
    bl_idname = "smartrig.face_place"
    bl_label = "Face Markers"
    bl_description = ("FaceIt-style placement: the landmark net follows the "
                      "mouse - CLICK on the CHIN, move the mouse to match the "
                      "FACE WIDTH, click to confirm. Then refine the flat net "
                      "in the locked front view and press Project Landmarks")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.smartrig
        ob = getattr(props, "target_mesh", None) or context.active_object
        return ob is not None and ob.type == 'MESH'

    def _mouse_plane(self, context, event):
        from bpy_extras import view3d_utils as v3u
        region = context.region
        rv3d = context.region_data
        co = (event.mouse_region_x, event.mouse_region_y)
        return v3u.region_2d_to_location_3d(
            region, rv3d, co, Vector((0.0, self.y0, self.z_guess)))

    def invoke(self, context, event):
        props = context.scene.smartrig
        self.body = getattr(props, "target_mesh", None) or context.active_object
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run from a 3D viewport")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        co = utils.read_rest_coords(self.body)
        top = float(co[:, 2].max())
        hgt = top - float(co[:, 2].min())
        head = co[co[:, 2] > top - 0.15 * hgt]
        self.y0 = float(head[:, 1].min()) - 0.01 * hgt   # just in front
        self.z_guess = top - 0.12 * hgt
        half_w = float(np.percentile(np.abs(head[:, 0]), 98))
        self.scale = max(half_w / 0.79, 1e-4)      # width
        self.scale_z = self.scale                  # height (FaceIt: separate)
        self.chin = Vector((0.0, self.y0, self.z_guess))
        pts = flat_grid_pts(self.chin, self.scale, self.y0)
        self.ob = _make_grid_object(pts, 0.84 * self.scale)
        set_rigs_hidden(True)                            # bones out of the way
        from . import markers as _mk
        _mk.lock_front_view(context, True)               # FaceIt: front locked
        self.stage = 'PLACE'
        context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set(
            "Face Markers  -  CLICK on the CHIN   (Esc / Right-click: cancel)")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _rebuild(self):
        pts = flat_grid_pts(self.chin, self.scale, self.y0, self.scale_z)
        for i, p in enumerate(pts):
            self.ob.data.vertices[i].co = Vector(p)
        self.ob.data.update()

    def _finish(self, context):
        context.window.cursor_modal_restore()
        context.area.header_text_set(None)

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            p = self._mouse_plane(context, event)
            if p is not None:
                if self.stage == 'PLACE':
                    self.chin = Vector((0.0, self.y0, p.z))
                else:
                    # FaceIt: WIDTH and HEIGHT follow the mouse separately -
                    # aim at the top-side corner of the head
                    self.scale = max(abs(p.x) / 0.79, 1e-4)
                    self.scale_z = max((p.z - self.chin.z) / 1.44, 1e-4)
                    try:
                        context.area.header_text_set(
                            "Face Markers  -  Width %.3f   Height %.3f   "
                            "(aim at the TOP-SIDE of the head, CLICK to "
                            "confirm, Esc: cancel)"
                            % (2 * 0.79 * self.scale, 1.51 * self.scale_z))
                    except Exception:
                        pass
                self._rebuild()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self.stage == 'PLACE':
                self.stage = 'SCALE'
                context.area.header_text_set(
                    "Face Markers  -  move the mouse to the TOP-SIDE of the "
                    "head: horizontal = WIDTH, vertical = HEIGHT, then CLICK "
                    "  (Esc: cancel)")
                return {'RUNNING_MODAL'}
            # confirmed: drop into Edit Mode on the flat net (front stays locked)
            self._finish(context)
            bpy.ops.object.select_all(action='DESELECT')
            self.ob.select_set(True)
            context.view_layer.objects.active = self.ob
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'INFO'}, "Refine the net (G/S/R, R follows L), then "
                        "press Project Landmarks")
            return {'FINISHED'}
        if event.type in ('RIGHTMOUSE', 'ESC'):
            self._finish(context)
            bpy.data.objects.remove(self.ob, do_unlink=True)
            from . import markers as _mk
            _mk.lock_front_view(context, False)
            set_rigs_hidden(False)
            return {'CANCELLED'}
        return {'PASS_THROUGH'}


class SMARTRIG_OT_face_project(bpy.types.Operator):
    bl_idname = "smartrig.face_project"
    bl_label = "Project Landmarks"
    bl_description = ("Project the flat landmark net onto the head surface "
                      "(FaceIt-style), create the anchor markers from it and "
                      "unlock the view. Register the mouth/eye loops or Build "
                      "Face Base next")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(GRID_NAME) is not None

    def execute(self, context):
        props = context.scene.smartrig
        body = getattr(props, "target_mesh", None) or context.active_object
        if body is None or body.type != 'MESH':
            self.report({'ERROR'}, "Pick the character mesh first")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        ob = bpy.data.objects.get(GRID_NAME)
        try:
            missed = project_grid(body, ob)
            markers_from_grid(body)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        from . import markers as _mk
        _mk.lock_front_view(context, False)
        if missed:
            self.report({'WARNING'}, "%d point(s) could not be projected - "
                        "move them in Edit Mode" % missed)
        else:
            self.report({'INFO'}, "Landmarks projected onto the face - "
                        "register the mouth/eye loops, then Build Face Base")
        return {'FINISHED'}


class SMARTRIG_OT_face_template(bpy.types.Operator):
    bl_idname = "smartrig.face_template"
    bl_label = "Face Markers"
    bl_description = ("Place the FaceIt-style face marker template on the "
                      "head (classic facial proportions, snapped to the "
                      "front surface - no auto-detection). Drag every marker "
                      "onto YOUR character's face (the R side mirrors the L "
                      "side), register the mouth / eye loops, then Build")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.smartrig
        ob = getattr(props, "target_mesh", None) or context.active_object
        return ob is not None and ob.type == 'MESH'

    def execute(self, context):
        props = context.scene.smartrig
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        body = getattr(props, "target_mesh", None) or context.active_object
        try:
            L, ipd = template_landmarks(props, body)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        place_markers(L, ipd)
        # the bones get in the way of the face markers: hide them now,
        # Build Face Base brings them back automatically
        set_rigs_hidden(True)
        self.report({'INFO'}, "Face template placed - drag each marker onto "
                    "the face (R follows L), then register the mouth/eye "
                    "loops and Build. Bones hidden (toggle in the panel)")
        return {'FINISHED'}


class SMARTRIG_OT_face_build_base(bpy.types.Operator):
    bl_idname = "smartrig.face_build_base"
    bl_label = "Build Face Base"
    bl_description = ("Build the face foundation from the markers: jaw bone "
                      "(pivot at the ear, tip at the chin) + jaw control + "
                      "analytic jaw/head weights. The lips / eyes / brows "
                      "modules build on top of this")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return markers_present()

    def execute(self, context):
        props = context.scene.smartrig
        storm_info = None
        try:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            if not grid_points():
                build_face_grid(props, context)      # FaceIt-style: automatic
            rig, standalone = build_base(props, context)
            nj = base_weights(props, context, rig, standalone)
            eyeL = _lm("face_eye.L")
            eyeR = _lm("face_eye.R")
            ipd = float(np.linalg.norm(eyeL - eyeR))
            P_jaw = (_lm("face_jaw.L") + _lm("face_jaw.R")) / 2.0
            parent_name = None if standalone else _head_parent_name(rig)
            if standalone:
                parent_name = "head"
            # ---- FULL Storm face (v2.0): the complete Storm control /
            # ribbon / lattice / driver system, retargeted by RBF ----
            if bool(getattr(props, "face_storm_full", True)):
                try:
                    import sys as _sys
                    from . import storm_face as _sf
                    if _sf.spec_available():
                        storm_info = _sf.build_full(
                            _sys.modules[__name__], props, context)
                        nc = storm_info["bones"]
                except Exception as _e:
                    import traceback
                    traceback.print_exc()
                    self.report({'WARNING'},
                                "Storm face failed (%s) - using the "
                                "simple layout" % _e)
                    storm_info = None
            if storm_info is None:
                nc = build_controls(props, context, rig, parent_name, ipd,
                                    P_jaw)
                local_weights(props, context, rig)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        # character-proportional widget sizes (Storm numbers explode on
        # other heads)
        if storm_info is None:
            try:
                nw = normalize_widget_sizes(rig, ipd)
                if nw:
                    print("Soulify: normalized %d face widget sizes" % nw)
            except Exception as _e:
                print("Soulify widget normalize failed:", _e)
            try:
                separate_lip_widgets(rig, ipd)
            except Exception as _e:
                print("Soulify lip widget separation failed:", _e)
            try:
                no = organize_face_layers(rig)
                print("Soulify: organized %d face bones into Rig Layers" % no)
            except Exception as _e:
                print("Soulify face layers failed:", _e)
        # RIG VIEW: the tools disappear, the animator rig is what you see
        try:
            from . import markers as _mk2
            _mk2.set_markers_hidden(True)        # body markers away too
        except Exception:
            pass
        g = bpy.data.objects.get(GRID_NAME)
        if g is not None:
            g.hide_set(True)                     # landmark net away
        meta = bpy.data.objects.get("SR_Metarig")
        if meta is not None:
            meta.hide_set(True)                  # metarig bones away
        try:
            rig.hide_set(False)                  # the rig with the controls
        except Exception:
            pass
        where = "standalone rig '%s'" % rig.name if standalone else \
                "rig '%s'" % rig.name
        if storm_info is not None:
            self.report({'INFO'},
                        "FULL Storm face on %s: %d bones, %d constraints, "
                        "%d drivers, %d lattices, %s"
                        % (where, storm_info["bones"],
                           storm_info["constraints"], storm_info["drivers"],
                           storm_info["lattice_mods"],
                           ", ".join(storm_info["parts"]) or "no extra parts"))
        else:
            self.report({'INFO'}, "Face rig built on %s - %d jaw verts, %d "
                        "Storm controls. CTL-Jaw = mouth, MSTR-Eye_target = "
                        "look" % (where, nj, nc))
        return {'FINISHED'}


class SMARTRIG_OT_face_grid(bpy.types.Operator):
    bl_idname = "smartrig.face_grid"
    bl_label = "Generate Face Grid"
    bl_description = ("Generate the detailed landmark grid (lips loop, eye "
                      "rings, brow arcs, jawline) from the markers, projected "
                      "onto the head. Refine it in Edit Mode - the R side "
                      "mirrors the L side. Every vertex is a future bone "
                      "joint for the lips / eyelids / brows modules")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return markers_present()

    def execute(self, context):
        props = context.scene.smartrig
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            ob, missed = build_face_grid(props, context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if missed:
            self.report({'WARNING'}, "Face grid: %d point(s) missed the mesh "
                        "- move them in Edit Mode" % missed)
        else:
            self.report({'INFO'}, "Face grid projected onto the head - "
                        "refine in Edit Mode, then Build Face Base")
        return {'FINISHED'}


class SMARTRIG_OT_face_loop_register(bpy.types.Operator):
    bl_idname = "smartrig.face_loop_register"
    bl_label = "Register Selected Loop"
    bl_description = ("Select an edge loop on the face mesh (Edit Mode, "
                      "Alt+Click) - the MOUTH loop or an EYELID loop - and "
                      "this snaps the landmarks EXACTLY onto it, then "
                      "rebuilds the face rig")
    bl_options = {'REGISTER', 'UNDO'}
    region: bpy.props.EnumProperty(
        name="Loop",
        items=(('AUTO', "Auto", "Classify by distance"),
               ('MOUTH', "Mouth", "The selected loop is the mouth"),
               ('EYE_L', "Eyelid L", "Use the LEFT-side verts of the selection"),
               ('EYE_R', "Eyelid R", "Use the RIGHT-side verts of the selection")),
        default='AUTO')

    @classmethod
    def poll(cls, context):
        return markers_present()

    def execute(self, context):
        props = context.scene.smartrig
        body = getattr(props, "target_mesh", None) or context.active_object
        if body is None or body.type != 'MESH':
            self.report({'ERROR'}, "Pick the character mesh first")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        me = body.data
        idx = [v.index for v in me.vertices if v.select]
        if len(idx) < 6:
            self.report({'ERROR'}, "Select the loop in Edit Mode first "
                        "(Alt+Click), then run this")
            return {'CANCELLED'}
        co = np.array([list(body.matrix_world @ me.vertices[i].co)
                       for i in idx])
        cen = co.mean(axis=0)

        # ---- classify: explicit from the pressed button, else by distance ----
        if self.region == 'MOUTH':
            region = "mouth"
        elif self.region == 'EYE_L':
            region = "eye.L"
        elif self.region == 'EYE_R':
            region = "eye.R"
        else:
            mouth_mid = (_lm("face_lip_up") + _lm("face_lip_low")) / 2.0
            eyeL, eyeR = _lm("face_eye.L"), _lm("face_eye.R")
            dists = {"mouth": np.linalg.norm(cen - mouth_mid),
                     "eye.L": np.linalg.norm(cen - eyeL),
                     "eye.R": np.linalg.norm(cen - eyeR)}
            region = min(dists, key=dists.get)

        grid = bpy.data.objects.get(GRID_NAME)

        def set_grid(name, p):
            if grid is None or name not in GRID_IDX:
                return
            v = grid.data.vertices[GRID_IDX[name]]
            v.co = grid.matrix_world.inverted() @ Vector(
                (float(p[0]), float(p[1]), float(p[2])))

        def set_marker(name, p, keep_x=False):
            ob = bpy.data.objects.get(name)
            if ob is None:
                return
            x = ob.matrix_world.translation.x if keep_x else float(p[0])
            ob.matrix_world.translation = (x, float(p[1]), float(p[2]))

        if region == "mouth":
            iR = int(np.argmin(co[:, 0]))
            iL = int(np.argmax(co[:, 0]))
            width = co[iL, 0] - co[iR, 0]
            midb = np.abs(co[:, 0] - (co[iL, 0] + co[iR, 0]) / 2.0) < 0.25 * width
            top = co[midb][co[midb][:, 2] >= np.median(co[midb][:, 2])]
            bot = co[midb][co[midb][:, 2] < np.median(co[midb][:, 2])]
            lip_T = top.mean(axis=0)
            lip_B = bot.mean(axis=0)
            set_marker("face_lip_up", lip_T, keep_x=True)
            set_marker("face_lip_low", lip_B, keep_x=True)
            set_marker("face_mouth_corner.L", co[iL])
            set_grid("lip_T", [0.0, lip_T[1], lip_T[2]])
            set_grid("lip_B", [0.0, lip_B[1], lip_B[2]])
            set_grid("mouth_corner.L", co[iL])
            # locals along the top/bottom arcs
            for name, base, t in (("lip_T.L", lip_T, 0.5),
                                  ("lip_B.L", lip_B, 0.5)):
                tgt = base + (co[iL] - base) * t
                j = int(np.argmin(np.linalg.norm(co - tgt, axis=1)))
                set_grid(name, co[j])
            msg = "Mouth loop registered (%d verts)" % len(idx)
        else:
            sd = ".L" if region == "eye.L" else ".R"
            sign = 1.0 if sd == ".L" else -1.0
            # ONLY this side's verts: a selection containing BOTH eyelid
            # loops centred the ring at x=0 and stretched it across the
            # whole face (the "marker broke" bug)
            sub = co[co[:, 0] * sign > 0.0]
            if len(sub) < 6:
                self.report({'ERROR'}, "No %s-side eyelid verts in the "
                            "selection - select the %s eyelid loop"
                            % (sd[1], sd[1]))
                return {'CANCELLED'}
            cen_s = sub.mean(axis=0)
            # signed X: the template angles are L-oriented for both sides
            ang = np.degrees(np.arctan2(sub[:, 2] - cen_s[2],
                                        (sub[:, 0] - cen_s[0]) * sign))
            ring = {"eye_out": 0.0, "lid_T_out": 45.0, "lid_T": 90.0,
                    "lid_T_in": 135.0, "eye_in": 180.0, "lid_B_in": -135.0,
                    "lid_B": -90.0, "lid_B_out": -45.0}
            for name, a in ring.items():
                d = np.abs(((ang - a + 180.0) % 360.0) - 180.0)
                j = int(np.argmin(d))
                p = sub[j]
                if sd == ".R":     # grid stores the L side; mirror into it
                    p = np.array([-p[0], p[1], p[2]])
                set_grid(name + ".L", p)
            msg = "Eye%s loop registered (%d verts)" % (sd, len(sub))
            # X-mirror modifier on the face mesh: one real side = both eyes
            if any(m.type == 'MIRROR' and m.use_axis[0]
                   for m in body.modifiers):
                if grid is not None:
                    grid["sr_loop_eye_L"] = True
                    grid["sr_loop_eye_R"] = True
                msg += " - mirrored mesh: both eyes covered"

        if grid is not None:
            grid.data.update()
            grid["sr_loop_" + region.replace(".", "_")] = True   # step ✓ badge
        # rebuild the face rig on the new landmarks
        try:
            bpy.ops.smartrig.face_build_base()
        except Exception:
            pass
        self.report({'INFO'}, msg + " - face rebuilt")
        return {'FINISHED'}


def _ensure_selectable(context, ob):
    """Make `ob` selectable NO MATTER how it was hidden: un-exclude and
    un-hide every layer collection that contains it, link it to the scene
    root if it is in no view-layer collection at all, and clear object-level
    hiding.  (Build hides helpers with hide_viewport / collection tricks -
    'not in View Layer' crashes came from selecting without this.)"""
    def _walk(lc):
        hits = []
        try:
            if ob.name in lc.collection.objects:
                hits.append(lc)
        except Exception:
            pass
        for ch in lc.children:
            hits.extend(_walk(ch))
        return hits

    hits = _walk(context.view_layer.layer_collection)
    for lc in hits:
        try:
            lc.exclude = False
            lc.hide_viewport = False
            lc.collection.hide_viewport = False
        except Exception:
            pass
    if ob.name not in context.view_layer.objects:
        try:
            context.scene.collection.objects.link(ob)
        except Exception:
            pass
    ob.hide_viewport = False
    try:
        ob.hide_set(False)
    except Exception:
        pass
    context.view_layer.update()


class SMARTRIG_OT_face_back_to_edit(bpy.types.Operator):
    bl_idname = "smartrig.face_back_to_edit"
    bl_label = "Back to Edit Landmarks"
    bl_description = ("Bring the landmark net back (bones hidden) to adjust "
                      "it, then press Rebuild Face Base to regenerate")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(GRID_NAME) is not None

    def execute(self, context):
        g = bpy.data.objects.get(GRID_NAME)
        _ensure_selectable(context, g)
        set_rigs_hidden(True)
        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
        bpy.ops.object.select_all(action='DESELECT')
        g.select_set(True)
        context.view_layer.objects.active = g
        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass
        self.report({'INFO'}, "Adjust the net (G/S/R), register loops if "
                    "needed, then Rebuild Face Base")
        return {'FINISHED'}


class SMARTRIG_OT_face_clear(bpy.types.Operator):
    bl_idname = "smartrig.face_clear"
    bl_label = "Remove Face Markers"
    bl_description = "Delete the SR_FaceMarkers collection and its markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
        coll = bpy.data.collections.get(FACE_COLL)
        if coll is not None:
            for ob in list(coll.objects):
                bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.collections.remove(coll)
        # clean restart: unlock the view, bring the bones back
        try:
            from . import markers as _mk
            _mk.lock_front_view(context, False)
        except Exception:
            pass
        set_rigs_hidden(False)
        self.report({'INFO'}, "Face markers cleared - press Face Markers to "
                    "start fresh")
        return {'FINISHED'}


CLASSES = (SMARTRIG_OT_face_detect, SMARTRIG_OT_face_objects_detect,
           SMARTRIG_OT_face_register_selected, SMARTRIG_OT_face_register_part,
           SMARTRIG_OT_face_register_slot, SMARTRIG_OT_face_unregister_part,
           SMARTRIG_OT_face_place,
           SMARTRIG_OT_face_project, SMARTRIG_OT_face_template,
           SMARTRIG_OT_toggle_bones,
           SMARTRIG_OT_face_build_base,
           SMARTRIG_OT_face_grid, SMARTRIG_OT_face_loop_register,
           SMARTRIG_OT_face_back_to_edit,
           SMARTRIG_OT_face_clear)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
