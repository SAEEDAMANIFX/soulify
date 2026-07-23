"""Soulify Face (new) - PART 2 = the MOUTH, register-driven (mirror of the EYE).

Goal: a cinematic, professional, EASY mouth rig, built exactly the proven way
the eye was (register a loop + drop two movable corner markers + pick the bone
count, then Build).  It reuses the eye's widget / helper toolkit so both parts
feel like one system.

Registration (all on the character face body):
  1. Mouth loop   -> select the lip EDGE LOOP (Edit mode, Alt+click) -> the whole
                     lip contour (upper + lower margin).  Stored as SR_mouthloop.
  2. Corners L/R  -> two movable coloured markers (BLUE = left commissure,
                     RED = right commissure).  Grab (G) each, drop it exactly on
                     the mouth corner, then Build.

Build Mouth Rig produces:
  - MASTER      : MSTR-Mouth (moves / rotates the whole mouth, parented to the
                  head) + per-lip masters CTL-LipM_upp / CTL-LipM_low (open /
                  shape a whole lip) drawn as a blue line tracing the lip.
  - LIP RIBBON  : upper & lower lips are sampled into K columns each, spread
                  EVENLY by arc-length between the two corners (upper == lower
                  so column i of the upper pairs with column i of the lower and
                  they meet EXACTLY on seal).  Each column carries an orange
                  TWEAK circle the animator grabs to sculpt phonemes / shapes.
  - OPEN / SEAL : one master slider CTL-MouthOpen drives 'mopen' in [-1, +1]:
                  +1 = mouth open (jaw drop: lower lip swings down + back on an
                  arc, upper lip lifts a touch), 0 = rest, -1 = lips pressed /
                  sealed (upper & lower cross onto each other so the lips close
                  with no gap - the same anti-slit overlap trick as the eyelids).
  - CORNERS     : DEF-lip_corn.L/.R anchors + pink corner TWEAK circles the
                  animator pulls up/out (smile) or down/in (frown).
  - SLIDE       : every lip deform bone is a SPOKE from a mouth 'socket' (a pivot
                  behind the lips, ~ the teeth cylinder axis) out to its lip
                  point, and DAMPED-TRACKs its tweak -> keeps its length ->
                  rotates about the socket = the lip SLIDES over the teeth
                  instead of stretching or poking through.
  - BIND        : optional automatic lip skinning, BOUNDED to the mouth region
                  (recognised from the registered loop) - never bleeds into the
                  chin, cheeks or nose.
Colour-coded: L = blue / R = red / centre = yellow / tweaks = orange / corners
= pink.  Naming uses 'lip' so it never collides with the eye's 'lid' bones.
"""
import bpy
import numpy as np
from mathutils import Vector
from . import utils
from . import eye_sample as _eye      # reuse the eye's widget + math toolkit

MOUTH_COLL = "Face - Mouth"
N_LIP = 6                             # fallback columns per lip if no prop set
MCORNER_COLL = "SR Mouth Corners"
_C_BLUE = (0.10, 0.45, 1.0)           # lip master lines / left
_C_RED = (1.0, 0.15, 0.15)            # right
_C_ORANGE = (1.0, 0.50, 0.04)         # per-point tweak circles
_C_YELLOW = (1.0, 0.85, 0.10)         # mouth master
_C_PINK = (1.0, 0.35, 0.62)           # corner tweaks (smile / frown)
_C_CYAN = (0.15, 0.85, 1.0)           # open / seal slider


# --------------------------------------------------------------------- helpers
def _body(context):
    return _eye._body(context)


def _target_rig(context):
    return _eye._target_rig(context)


def _mouth_loop(body):
    """World coords of the registered lip loop (vertex group SR_mouthloop)."""
    vg = body.vertex_groups.get("SR_mouthloop")
    if vg is None:
        return None
    gi = vg.index
    out = []
    for v in body.data.vertices:
        for g in v.groups:
            if g.group == gi:
                out.append(list(body.matrix_world @ v.co))
                break
    return np.array(out) if out else None


def _mcorner_pts(body):
    """(left, right) corner world coords. Priority:
      1. movable markers SR_mcorner.L / SR_mcorner.R (snapped to nearest vertex),
      2. the loop's horizontal extremes (fallback, handled by the caller).
    left = the +X commissure, right = the -X commissure."""
    l = bpy.data.objects.get("SR_mcorner.L")
    r = bpy.data.objects.get("SR_mcorner.R")
    if l is not None and r is not None and body is not None:
        pl = l.matrix_world.translation
        pr = r.matrix_world.translation
        if pl.length > 1e-4 and pr.length > 1e-4 and (pl - pr).length > 1e-4:
            a = _eye._snap_vert(body, pl)
            b = _eye._snap_vert(body, pr)
            if a[0] < b[0]:                 # ensure a = +X (left), b = -X (right)
                a, b = b, a
            return a, b
    return None


# ---------------------------------------------------------------- corner markers
def _mcorner_coll():
    c = bpy.data.collections.get(MCORNER_COLL)
    if c is None:
        c = bpy.data.collections.new(MCORNER_COLL)
        bpy.context.scene.collection.children.link(c)
    return c


def _mcorner_marker(name, rgb, size):
    """A small coloured diamond the animator grabs and drops on a mouth corner."""
    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != 'MESH':
        me = bpy.data.meshes.new(name)
        V = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        F = [(0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
             (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)]
        me.from_pydata(V, [], F)
        me.update()
        ob = bpy.data.objects.new(name, me)
        _mcorner_coll().objects.link(ob)
        ob["sr_mouth_corner"] = True
    ob.data.materials.clear()
    ob.data.materials.append(_eye._corner_mat("SRmk_" + name, rgb))
    ob.color = (rgb[0], rgb[1], rgb[2], 1.0)
    ob.scale = (size, size, size)
    ob.show_in_front = True
    ob.show_name = True
    ob.hide_viewport = False
    try:
        ob.hide_set(False)
    except Exception:
        pass
    return ob


def _hide_mcorner_markers(hidden):
    for nm in ("SR_mcorner.L", "SR_mcorner.R"):
        o = bpy.data.objects.get(nm)
        if o is not None:
            try:
                o.hide_viewport = hidden
                o.hide_set(hidden)
            except Exception:
                pass


def _registration_state(context):
    body = _body(context)
    st = {}
    st["loop"] = (body is not None and
                  body.vertex_groups.get("SR_mouthloop") is not None)
    st["corners"] = (bpy.data.objects.get("SR_mcorner.L") is not None and
                     bpy.data.objects.get("SR_mcorner.R") is not None)
    return st


# --------------------------------------------------------------- modern widgets
def _mwgt(name, kind):
    """Modern mouth control widgets, drawn in the X-Z plane (normal +Y) so they
    face the camera when the control's roll puts local Z = world up.
      frame   = a rounded rectangle (the whole-mouth master outline)
      handle  = a bold rounded square (the corner smile / frown grip)
      jawhook = a shallow chin hook (the jaw control)"""
    ob = bpy.data.objects.get(name)
    if ob is not None:
        return ob

    def _round_rect(hx, hz, r, seg=6):
        # rounded rectangle, half-extents hx/hz, corner radius r, in X-Z
        r = min(r, hx * 0.99, hz * 0.99)
        cxs = [(hx - r, hz - r), (-(hx - r), hz - r),
               (-(hx - r), -(hz - r)), (hx - r, -(hz - r))]
        a0 = [0.0, np.pi * 0.5, np.pi, np.pi * 1.5]
        pts = []
        for (cx, cz), a in zip(cxs, a0):
            for k in range(seg + 1):
                ang = a + (np.pi * 0.5) * k / seg
                pts.append((cx + r * np.cos(ang), 0.0, cz + r * np.sin(ang)))
        return pts

    if kind == "frame":
        V = _round_rect(1.0, 1.0, 0.35, seg=5)
        E = [(i, (i + 1) % len(V)) for i in range(len(V))]
    elif kind == "handle":
        V = _round_rect(1.0, 1.0, 0.5, seg=5)
        E = [(i, (i + 1) % len(V)) for i in range(len(V))]
    elif kind == "jawhook":
        # a shallow smile-shaped chin hook (opens upward), with short upturned
        # ends so it reads as a jaw cradle under the chin.
        m = 20
        V = []
        for k in range(m + 1):
            t = -1.0 + 2.0 * k / m
            x = t
            z = -0.55 + 0.55 * (t * t)          # parabola dipping to -0.55
            V.append((x, 0.0, z))
        # upturned end ticks
        V.append((-1.0, 0.0, 0.15))
        V.append((1.0, 0.0, 0.15))
        E = [(i, i + 1) for i in range(m)]
        E.append((0, m + 1))
        E.append((m, m + 2))
    elif kind in ("ringup", "ringdn"):
        # a circle shifted along local +Z / -Z, so on a nearly-flat mouth the
        # upper-lip and lower-lip tweak circles sit in two clean, separated rows
        # instead of piling on top of each other. The shift scales with the
        # widget, so it stays proportional at any mouth size.
        seg = 24
        off = 1.8 if kind == "ringup" else -1.8
        V = [(np.cos(2 * np.pi * i / seg), 0.0, np.sin(2 * np.pi * i / seg) + off)
             for i in range(seg)]
        E = [(i, (i + 1) % seg) for i in range(seg)]
    else:                                       # fallback: a plain ring
        seg = 24
        V = [(np.cos(2 * np.pi * i / seg), 0.0, np.sin(2 * np.pi * i / seg))
             for i in range(seg)]
        E = [(i, (i + 1) % seg) for i in range(seg)]
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in V], E, [])
    me.update()
    ob = bpy.data.objects.new(name, me)
    ob["sr_wgt"] = True
    col = bpy.data.collections.get("WGTS_soulify")
    if col is None:
        col = bpy.data.collections.new("WGTS_soulify")
        bpy.context.scene.collection.children.link(col)
        col.hide_viewport = True
        col.hide_render = True
    col.objects.link(ob)
    return ob


# --------------------------------------------------------------------- geometry
def _mouth_frame(loop, corners):
    """Build a stable, upright local frame for the mouth.

    IMPORTANT: a lip loop is often a thin, wide, nearly-FLAT ring (a wide mouth
    can be only a few mm tall).  An SVD 'plane normal' of such a loop is
    unstable - the smallest-variance axis comes out VERTICAL, not out of the
    face - which rotates the whole rig 90 deg (jaw flies to the forehead, the
    control circles go edge-on).  So we anchor the frame to WORLD axes instead,
    which is what a face rig wants anyway:
      fwd  = out of the face (world -Y, de-tilted by the corner axis),
      right= corner -> corner (world X-ish),
      up   = fwd x right (world +Z-ish, upper lip > 0).
    Returns c, right, up, fwd, (pL, pR)."""
    c = loop.mean(axis=0)
    if corners is not None:
        pL, pR = corners
    else:
        xs = loop[:, 0]
        pL = loop[int(np.argmax(xs))]
        pR = loop[int(np.argmax(-xs))]
    pL = np.asarray(pL, float)
    pR = np.asarray(pR, float)
    right = pL - pR
    right = right / (np.linalg.norm(right) + 1e-9)
    fwd = np.array([0.0, -1.0, 0.0])           # the face looks down -Y
    # keep fwd perpendicular to the corner axis (in case the head is rolled)
    fwd = fwd - right * float(np.dot(fwd, right))
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    up = np.cross(fwd, right)
    up = up / (np.linalg.norm(up) + 1e-9)
    if up[2] < 0:
        up = -up
        fwd = np.cross(right, up)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(up, fwd)
    right = right / (np.linalg.norm(right) + 1e-9)
    return c, right, up, fwd, (pL, pR)


def build_mouth_rig(context):
    body = _body(context)
    if body is None or body.type != 'MESH':
        raise RuntimeError("Pick the character face mesh first (Target Mesh).")
    rig = _target_rig(context)
    if rig is None:
        raise RuntimeError("No armature found - generate the body rig first.")

    props = context.scene.smartrig
    n_upp = max(2, int(getattr(props, "mouth_lip_upper_count", N_LIP)))
    n_low = max(2, int(getattr(props, "mouth_lip_lower_count", N_LIP)))
    # upper == lower so the lips pair up 1:1 and meet exactly on seal
    K = min(n_upp, n_low)
    autobind = bool(getattr(props, "mouth_autobind", True))
    band_fac = float(getattr(props, "mouth_bind_band", 0.9))

    loop = _mouth_loop(body)
    if loop is None or len(loop) < 8:
        raise RuntimeError("Register the mouth loop first (Edit mode, Alt+click "
                           "the lip edge loop -> Register Mouth Loop).")
    corners = _mcorner_pts(body)
    c, right, up, fwd, (pL, pR) = _mouth_frame(loop, corners)

    rel = loop - c
    u = rel @ right                        # left(+) <-> right(-)
    vv = rel @ up                          # upper(+) / lower(-)
    ff = rel @ fwd                         # depth (out of face)
    mw = float(u.max() - u.min())          # mouth width
    mh = float(np.abs(vv).max()) * 2.0     # mouth height (full)
    mh = mh or (0.4 * mw)
    # LV = the lip VERTICAL zone. A wide mouth can be only a few mm tall, so the
    # true mh is useless for sizing widgets / offsets / bind reach - floor it to
    # a fraction of the width so the controls stay visible and grabbable and the
    # skin bind actually reaches the lip.
    LV = max(0.5 * mh, 0.16 * mw, 1e-4)
    S = LV                                 # small unit for widgets / offsets

    # corners in u
    u_L = float((pL - c) @ right)
    u_R = float((pR - c) @ right)
    lo_u, hi_u = min(u_L, u_R), max(u_L, u_R)
    # if the markers are missing / degenerate, fall back to the loop extremes
    if (hi_u - lo_u) < 0.6 * (u.max() - u.min()):
        lo_u, hi_u = float(u.min()), float(u.max())
        pR = loop[int(np.argmin(u))]
        pL = loop[int(np.argmax(u))]
    span = max(hi_u - lo_u, 1e-6)

    # socket = a pivot BEHIND the lips (into +Y) ~ the teeth cylinder axis, so
    # the deform spokes point outward and slide the lip over the teeth.
    SOCKET_DEPTH = 1.5
    socket = c - fwd * (SOCKET_DEPTH * 0.5 * mw)

    # ---- ordered margin polylines (ascending u) for upper & lower lips
    margins = {}
    for part, mask in (("upp", vv > 0), ("low", vv < 0)):
        m = mask & (u > lo_u - 1e-6) & (u < hi_u + 1e-6)
        pu = u[m]
        if len(pu) < 2:
            m = mask
            pu = u[m]
        if len(pu) < 2:
            margins[part] = None
            continue
        order = np.argsort(pu)
        margins[part] = dict(us=pu[order], pts=loop[m][order],
                             vs=(vv[m])[order], ffs=(ff[m])[order])

    # ---- tuning (top-level so it's easy to iterate live) --------------------
    BMIX = 0.5          # seam blend (0.5 = midway between upper & lower margins)
    FWD_PUSH = 0.12     # push the seal seam slightly forward
    OVERLAP = 0.12      # upper crosses just below the seam, lower just above ->
                        #   the two lip skins overlap and SEAL with no slit
    # the JAW now does the gross open; the slider's +open side is only a small
    # extra lip-part on top of it (so the animator can part the lips independently
    # of the jaw - e.g. a soft "aah" without dropping the chin).
    OPEN_DROP = 0.45    # extra lower-lip part on full slider-open (fraction of mh)
    OPEN_BACK = 0.20    # lower lip swings back into the mouth on open
    OPEN_LIFT = 0.14    # upper lip lift on full open
    TAPER = lambda frac: 0.75 + 0.25 * float(np.sin(np.pi * frac))

    def _seam_pt(uq):
        """Smooth closed-lip seam point at horizontal position uq (upper & lower
        columns at the same uq land here, so the lips meet with no kink)."""
        mu, ml = margins.get("upp"), margins.get("low")
        if mu is None or ml is None:
            base = ml or mu
            vseam = float(np.interp(uq, base["us"], base["vs"]))
            ffseam = float(np.interp(uq, base["us"], base["ffs"]))
        else:
            vu = float(np.interp(uq, mu["us"], mu["vs"]))
            vl = float(np.interp(uq, ml["us"], ml["vs"]))
            fu = float(np.interp(uq, mu["us"], mu["ffs"]))
            fl = float(np.interp(uq, ml["us"], ml["ffs"]))
            vseam = (1.0 - BMIX) * vu + BMIX * vl
            ffseam = max(fu, fl)
        return c + right * uq + up * vseam + fwd * (ffseam + FWD_PUSH * S)

    info = {"width": round(mw, 4), "height": round(mh, 4),
            "upper": K, "lower": K, "columns": [], "bound": 0}

    prev = context.view_layer.objects.active
    context.view_layer.objects.active = rig
    rig.hide_set(False)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones

    def _drop(pred):
        for b in list(eb):
            if pred(b.name):
                eb.remove(b)
    _drop(lambda n: n.startswith(("MSTR-Mouth", "CTL-Lip", "DEF-lip", "MCH-mtgt",
                                  "CTL-MouthOpen", "MCH-Mouth", "CTL-Jaw",
                                  "DEF-Jaw", "MCH-JawCorn", "MCH-JawRot",
                                  "MCH-JawRet", "MCH-JawSway")))

    def nb(name, head, tail, deform, parent=None):
        b = eb.new(name)
        b.head = Vector([float(v) for v in head])
        b.tail = Vector([float(v) for v in tail])
        b.use_deform = deform
        if parent and parent in eb:
            b.parent = eb[parent]
            b.use_connect = False
        return b

    head_parent = None
    for n in ("DEF-head", "head", "DEF-spine.006", "DEF-neck"):
        if n in rig.data.bones:
            head_parent = n
            break

    made_ring, made_arrow = [], []
    made_ring_up, made_ring_dn = [], []   # upper / lower lip tweak circles
    made_frame, made_handle = [], []   # mouth-frame + corner-handle widgets
    col_wiring = []                  # per column: dict for the pose pass
    corner_follow = []               # (corner DEF bone, its tweak)
    master_wdata = []                # (master bone, world polyline tracing lip)
    bind_data = None
    ribbon_specs = []

    # MSTR-Mouth = the whole-mouth pivot, parented to head.  Its yellow frame
    # widget must ENCLOSE THE WHOLE MOUTH (both corners + upper & lower lip
    # lines) with a clear margin, so the animator reads it as "grab this = move
    # the entire mouth".  Half-width reaches just beyond the commissures
    # (0.5*span) plus a margin; half-height clears both lip master lines
    # (offset +/-0.55*LV) with room to spare.
    nb("MSTR-Mouth", c, c + fwd * (0.6 * S), False, head_parent)
    mstr_hw = 0.5 * span + 0.95 * LV
    mstr_hh = 2.1 * LV
    made_frame.append(("MSTR-Mouth", (mstr_hw, mstr_hh), _C_YELLOW))
    made_jaw = []                    # (bone, size, colour) - jaw / hook widgets
    corner_masters = {}
    jaw_wiring = []                  # (mch-follow bone, factor)

    # ---- JAW: an ARP-style TRANSLATION control (professional).  The animator
    # GRABS a handle sitting at the front of the chin and MOVES it - up/down
    # closes/opens the jaw, left/right sways it - instead of dialling a rotation.
    # The handle's LOCAL translation drives two hidden hinge bones so the chin
    # still swings on a real anatomical arc (down + back), not a straight slide:
    #   MCH-JawSway = a vertical bone at the pivot; twisting it (driven by the
    #                 handle's SIDE motion) sways the whole jaw left/right.
    #   MCH-JawRot  = the pivot->chin hinge (child of MCH-JawSway); rotating it
    #                 about its X (driven by the handle's UP/DOWN motion) opens
    #                 the mouth.  DEF-Jaw is a CHILD of MCH-JawRot (rest = identity
    #                 always -> no rest-mismatch tear), so it deforms the chin.
    # CTL-Jaw itself never deforms and never rotates - it is a pure move-handle.
    # The hinge pivot is an EDITABLE estimate - nudge MCH-JawRot's HEAD in Edit
    # mode if the open arc looks off on a stylised head.
    JAW_BACK, JAW_UP = 1.8, 0.5      # hinge offset behind / above the mouth (x mw)
                                     #   ~ ear/nose height, a realistic lever
    jaw_pivot = c - fwd * (JAW_BACK * mw) + up * (JAW_UP * mw)
    chin = c - up * (2.2 * LV) - fwd * (0.15 * mw)
    # The jaw is a SKULL element: it must hang off the HEAD, NOT off MSTR-Mouth
    # (the whole-mouth shifter).  If it rode MSTR-Mouth, sliding the mouth master
    # would drag the jaw around - wrong.  So anchor the jaw chain + its handle to
    # the head bone.  (The lip CORNERS below stay on MSTR-Mouth so they still move
    # with the mouth master, and only FOLLOW the jaw's open via a constraint.)
    jaw_anchor = head_parent or "MSTR-Mouth"
    # sway bone: points straight UP at the pivot (local Y = world up), so a twist
    # about its own Y rotates the jaw about the vertical = clean left/right sway.
    nb("MCH-JawSway", jaw_pivot, jaw_pivot + up * (1.2 * LV), False, jaw_anchor)
    # hinge: pivot -> chin, child of the sway bone so it inherits the sway.
    nb("MCH-JawRot", jaw_pivot, chin, False, "MCH-JawSway")
    # deform bone rides the hinge (parented, no constraint -> rest identity).
    nb("DEF-Jaw", jaw_pivot, chin, True, "MCH-JawRot")
    # ⭐ JAW RETAIN (ARP jaw_ret_bone / "Soft Lips - Linear Jaw"): a hinge that
    # copies the jaw open at JAW_FOLLOW (50%).  The lower-lip master AND the lip
    # CORNERS both ride this, so they follow the open by the SAME fraction and
    # move TOGETHER - the chin (DEF-Jaw) alone follows 100%.  This is THE fix for
    # the boxy open: before, the lower lip followed 100% while the corner was
    # frozen (0%) -> a vertical wall at each commissure.  Now lip==corner==50%
    # and the skin blends smoothly lip(50%) -> chin(100%).
    # NB: parented to MSTR-Mouth (NOT the jaw chain) so the lower lip follows the
    # WHOLE-MOUTH master for move / SCALE / KISS-pucker, and gets the jaw open on
    # TOP via the COPY_ROTATION below.  (DEF-Jaw/chin stays on the jaw chain, so
    # scaling MSTR-Mouth scales the LIPS only, never the jaw bone.)
    nb("MCH-JawRet", jaw_pivot, chin, False, "MSTR-Mouth")
    jaw_wiring.append(("MCH-JawRet", 0.5))            #   50% jaw follow (Soft Lips)
    # the MOVE-handle: a short bone JUST IN FRONT of the chin, pointing forward
    # (+fwd = -Y toward camera) so its widget faces front and its local axes are
    # clean: local X = sideways, local Z = up, local Y = fwd/back (locked off).
    # Anchored to the head so it stays glued to the chin when the mouth shifts.
    jaw_ctl_head = np.asarray(chin, float) + fwd * (0.10 * mw)
    nb("CTL-Jaw", jaw_ctl_head, jaw_ctl_head + fwd * (1.2 * LV), False,
       jaw_anchor)
    made_jaw.append(("CTL-Jaw", 1.3 * LV, _C_YELLOW))
    info["jaw_pivot"] = [round(float(x), 4) for x in jaw_pivot]
    # travel -> rotation mapping (used by the pose-pass drivers). Tie every
    # distance to LV so it scales with the mouth: loc DOWN by open_travel = full
    # open; loc UP by clench_travel = full clench; loc SIDE by side_travel = full
    # sway.
    jaw_map = dict(open_travel=2.5 * LV, clench_travel=1.0 * LV,
                   side_travel=2.0 * LV, open_max=0.55, clench_max=0.15,
                   sway_max=0.22)

    # per-lip masters (open / shape a whole lip).  UPPER follows the head; LOWER
    # follows the JAW RETAIN (50%) so the bottom lip rides the jaw open at the
    # SAME rate as the corners (moves together, no wall) - not the full 100%.
    masters = {}
    master_parent = {"upp": "MSTR-Mouth", "low": "MCH-JawRet"}
    for part, sgn in (("upp", 1.0), ("low", -1.0)):
        mg = margins.get(part)
        mmid = _eye._interp_pt(0.5 * (lo_u + hi_u), mg["us"], mg["pts"]) if mg \
            else (c + up * sgn * 0.5 * mh)
        mn = "CTL-LipM_%s" % part
        nb(mn, socket, socket + (np.asarray(mmid) - socket) * 1.05, False,
           master_parent[part])
        masters[part] = mn

    # corners: each rides a MCH-JawCorn that copies HALF the jaw rotation (so a
    # smile corner drops a little on open, like real lips). On top of it sits a
    # big corner MASTER (smile / frown handle) and a small tweak child of the
    # master (fine sculpt).  DEF corner spoke follows the tweak.
    for cname, cp, pal in (("L", pL, _C_BLUE), ("R", pR, _C_RED)):
        cp = np.asarray(cp, float)
        # PARTIAL-JAW hinge for this corner (ARP/Rigify style).  A bone at the
        # jaw PIVOT pointing to the chin - the SAME orientation as MCH-JawRot -
        # parented to the mouth master.  A clean LOCAL copy of the jaw open at
        # CORN_FOLLOW swings it on the REAL hinge arc, so the corner DROPS a
        # fraction with the jaw.  (The sideways MCH-JawCorn below can't copy the
        # hinge rotation itself - different local frame, it capped at ~6mm and
        # left the corner frozen = the boxy open with vertical corner walls.)
        chp = "MCH-JawCornP.%s" % cname
        nb(chp, jaw_pivot, chin, False, "MSTR-Mouth")
        jaw_wiring.append((chp, 0.5))                  #   50% - SAME as the lip
        mch = "MCH-JawCorn.%s" % cname
        # the sideways corner bone RIDES the partial hinge (so it inherits the
        # 45% drop) and still moves with the mouth master through it.
        nb(mch, jaw_pivot, cp, False, chp)
        cm = "CTL-Lips_corn.%s" % cname
        nb(cm, cp, cp + fwd * (0.5 * S), False, mch)   # +fwd = point -Y at camera
        made_handle.append((cm, 0.28 * S, pal))        # the ONE corner handle
        # DEF corner = a SHORT bone AT the corner, a CHILD of the corner MASTER
        # directly (the extra fine tweak CTL-LipT_corn was removed per Saeed - it
        # just cluttered the same spot as the master).  NOT a spoke from the
        # centre socket: that old socket-spoke + DAMPED_TRACK held a constant
        # distance to the centre, so dropping the corner on jaw-open slid it
        # INWARD along that sphere -> corners collapsed and wrecked the mouth.  A
        # short child just follows the master rigidly: down/back with the jaw at
        # 50%, horizontal X preserved.
        bn = "DEF-lip_corn.%s" % cname
        nb(bn, cp, cp + fwd * (0.30 * S), True, cm)
        corner_masters[cname] = (mch, cm)

    # ---- lip ribbon columns (upper & lower share the SAME fracs so they pair) -
    if K == 1:
        fracs = [0.5]
    else:
        lo_f, hi_f = 0.04, 0.96
        fracs = [lo_f + (hi_f - lo_f) * j / (K - 1.0) for j in range(K)]
    us_col = [lo_u + f * span for f in fracs]

    bones_by = {}
    for part, sgn in (("upp", 1.0), ("low", -1.0)):
        mg = margins.get(part)
        if mg is None:
            bones_by[part] = []
            continue
        blist = []
        open_pts = []
        for i, (frac, uq) in enumerate(zip(fracs, us_col), 1):
            p_rest = _eye._interp_pt(uq, mg["us"], mg["pts"])
            open_pts.append(p_rest)
            # SEAL: upper crosses just below the seam, lower just above it, so the
            # two lip skins overlap and the closed mouth seals with no slit.
            ov = OVERLAP * TAPER(frac)
            p_seal = _seam_pt(uq) - up * (sgn * ov * S)
            # OPEN: lower lip swings down + back on an arc; upper lifts a touch.
            if sgn < 0:
                p_open = (p_rest - up * (OPEN_DROP * LV)
                          - fwd * (OPEN_BACK * LV))
            else:
                p_open = p_rest + up * (OPEN_LIFT * LV)
            sl = "MCH-mtgtSeal_%s%d" % (part, i)
            op = "MCH-mtgtOpen_%s%d" % (part, i)
            tg = "MCH-mtgt_%s%d" % (part, i)
            nb(sl, p_seal, p_seal + fwd * 0.12 * S, False, masters[part])
            nb(op, p_open, p_open + fwd * 0.12 * S, False, masters[part])
            nb(tg, p_rest, p_rest + fwd * 0.12 * S, False, masters[part])
            tw = "CTL-LipT_%s%d" % (part, i)
            nb(tw, p_rest, p_rest + fwd * (0.35 * S), False, tg)  # +fwd = face cam
            (made_ring_up if part == "upp" else made_ring_dn).append(
                (tw, 0.10 * S, _C_ORANGE))
            dn = "DEF-lip_%s%d" % (part, i)
            nb(dn, socket, p_rest, True, masters[part])
            # ⭐ CORNER FOLLOW (ARP-style control interpolation): this column's
            # tweak FOLLOWS the NEAREST corner control by `cf` - 1 at the
            # commissure, smoothstep to 0 at the mouth centre.  So pulling
            # CTL-Lips_corn drags the actual lip BONES (upper AND lower) with it,
            # the whole lip sweeps into the smile / frown - not just the skin near
            # the corner.  frac 0 = the lo_u corner (.R), frac 1 = hi_u (.L).
            dn_c = min(frac, 1.0 - frac)
            t_cf = max(0.0, 1.0 - 1.7 * dn_c)          # reaches past mid-lip
            cf = t_cf * t_cf * (3.0 - 2.0 * t_cf)
            cbone = "CTL-Lips_corn.R" if frac < 0.5 else "CTL-Lips_corn.L"
            col_wiring.append(dict(tgt=tg, seal=sl, open=op, defb=dn, track=tw,
                                   slider="CTL-MouthOpen", cbone=cbone, cfollow=cf))
            blist.append((float((p_rest - c) @ right), dn, np.asarray(p_rest)))
        bones_by[part] = blist
        info["columns"].append("%s x%d" % (part, len(blist)))

        # blue master line tracing corner -> lip -> corner, OFFSET outward
        # (upper line above the top lip, lower line below the bottom lip) so the
        # two lip masters read as distinct, grabbable handles - not one line on
        # the mouth. Slightly forward so they float just off the skin.
        moff = up * (sgn * 0.55 * LV) + fwd * (0.25 * S)
        u_pl = float((pL - c) @ right)
        u_pr = float((pR - c) @ right)
        lo_cnr, hi_cnr = ((pR, pL) if u_pr <= u_pl else (pL, pR))
        poly = ([list(map(float, np.asarray(lo_cnr) + moff))]
                + [list(map(float, np.asarray(p) + moff)) for p in open_pts]
                + [list(map(float, np.asarray(hi_cnr) + moff))])
        master_wdata.append((masters[part], poly))
        ribbon_specs.append(("HLP-SR-liprib_%s" % part, part,
                             [(i + 1, "MCH-mtgt_%s%d" % (part, i + 1))
                              for i in range(len(open_pts))],
                             list(map(list, open_pts)), list(up), sgn, S))

    # master OPEN / SEAL slider beside the mouth (drag UP = open, DOWN = seal)
    slb = "CTL-MouthOpen"
    side_sign = 1.0 if pL[0] >= pR[0] else -1.0
    sl_head = np.asarray(pL) + right * (0.35 * mw) + fwd * (0.5 * S)
    nb(slb, sl_head, sl_head + up * (1.1 * LV), False, "MSTR-Mouth")
    made_arrow.append((slb, 0.7 * LV, _C_CYAN))

    # weighting geometry (columns reach the corners; the static corner bones are
    # a fallback only if a lip has no columns)
    upp_ctrl = sorted(bones_by.get("upp", []), key=lambda t: t[0])
    low_ctrl = sorted(bones_by.get("low", []), key=lambda t: t[0])
    # the two mouth CORNERS as shared blend anchors (the commissure belongs to
    # BOTH lips), so the corner DEF bones actually get skin and CTL-Lips_corn
    # deforms.  Placed at each corner's true horizontal position.
    cornerL = (float((np.asarray(pL) - c) @ right), "DEF-lip_corn.L")
    cornerR = (float((np.asarray(pR) - c) @ right), "DEF-lip_corn.R")
    bind_data = dict(c=c, right=right, up=up, fwd=fwd, loop=loop,
                     vmax=max(float(np.abs(vv).max()), 0.5 * LV), mh=LV,
                     u_in=lo_u, u_out=hi_u, span=span,
                     # LONG professional reach like ARP (its lip weights fade out
                     # over ~40-76mm on a ~95mm mouth).  Tied to the mouth WIDTH
                     # (0.42*mw) so it auto-scales to ANY character, not a fixed
                     # number.  This is the smooth "distance" ARP has and the old
                     # ~13mm band lacked (felt like a hard, un-professional cutoff).
                     band=max(band_fac * LV, 0.30 * mw, 1e-4),
                     upp=upp_ctrl, low=low_ctrl, jaw="DEF-Jaw",
                     cornerL=cornerL, cornerR=cornerR,
                     cornerL_pt=list(map(float, np.asarray(pL))),
                     cornerR_pt=list(map(float, np.asarray(pR))))

    # bone collections: only controls visible; DEF + MCH hidden
    try:
        def _coll(name, visible):
            data = rig.data
            cc = None
            for bc in data.collections:
                if bc.name == name:
                    cc = bc
                    break
            if cc is None:
                cc = data.collections.new(name)
            try:
                cc.is_visible = visible
            except Exception:
                pass
            return cc
        ctrl_c = _coll(MOUTH_COLL, True)
        def_c = _coll(MOUTH_COLL + " (Deform)", False)
        mch_c = _coll(MOUTH_COLL + " (Mech)", False)
        for b in eb:
            nm = b.name
            if nm.startswith(("MCH-mtgt", "MCH-Mouth", "MCH-JawCorn",
                              "MCH-JawRot", "MCH-JawRet", "MCH-JawSway")):
                mch_c.assign(b)
            elif nm.startswith(("DEF-lip", "DEF-Jaw")):
                def_c.assign(b)
            elif nm.startswith(("MSTR-Mouth", "CTL-Lip", "CTL-MouthOpen",
                                "CTL-Jaw")):
                ctrl_c.assign(b)
    except Exception:
        pass

    # straight gizmos.  Z-up roll = face-front for the flat widgets, AND it gives
    # MCH-JawRot the same clean local frame the old rotating jaw had (local X =
    # the hinge axis, so rot_x opens the mouth).
    for b in eb:
        if b.name.startswith(("MSTR-Mouth", "CTL-Lip", "CTL-MouthOpen",
                              "CTL-Jaw", "MCH-JawRot", "MCH-JawRet",
                              "MCH-JawCornP")):
            try:
                b.align_roll(Vector((0.0, 0.0, 1.0)))
            except Exception:
                pass
    # the sway bone points UP; roll it so local Z faces forward (-Y) -> local X =
    # world +X (a clean horizontal axis), so a twist about its local Y is a pure
    # left/right jaw sway.
    bsw = eb.get("MCH-JawSway")
    if bsw is not None:
        try:
            bsw.align_roll(Vector((0.0, -1.0, 0.0)))
        except Exception:
            pass

    bpy.ops.object.mode_set(mode='POSE')

    # ⚠️ CLEAR STALE CONSTRAINTS first.  Dropping + recreating an edit bone with
    # the SAME name REUSES its pose bone, so constraints from an earlier build
    # (e.g. an old "SR Jaw Follow" on MCH-JawCorn from when the wiring targeted a
    # different bone) SURVIVE the rebuild and silently corrupt the motion (the
    # corner span-collapse-inward bug: a stale COPY_ROTATION spun the sideways
    # corner bone about the wrong local axis).  Wipe every mouth pose bone's
    # constraints so only the ones this build adds below remain.
    for pb in rig.pose.bones:
        if pb.name.startswith(("MSTR-Mouth", "CTL-Lip", "DEF-lip", "MCH-mtgt",
                               "CTL-MouthOpen", "MCH-Mouth", "CTL-Jaw", "DEF-Jaw",
                               "MCH-Jaw")):
            for con in list(pb.constraints):
                pb.constraints.remove(con)

    # JAW (translation-driven): the two hidden hinge bones are rotated by DRIVERS
    # that read CTL-Jaw's LOCAL translation.  DEF-Jaw is parented to MCH-JawRot
    # (deforms the chin with it - no constraint, so rest stays identity).
    OT = float(jaw_map["open_travel"]);  CT = float(jaw_map["clench_travel"])
    ST = float(jaw_map["side_travel"]);  OM = float(jaw_map["open_max"])
    CM = float(jaw_map["clench_max"]);   SM = float(jaw_map["sway_max"])

    def _jaw_driver(bone, chan, expr, loc_axis):
        """Drive rotation_euler[chan] of `bone` from CTL-Jaw's LOCAL `loc_axis`."""
        pb = rig.pose.bones.get(bone)
        if pb is None:
            return
        pb.rotation_mode = 'XYZ'
        try:
            pb.driver_remove("rotation_euler", chan)
        except Exception:
            pass
        fcu = pb.driver_add("rotation_euler", chan)
        drv = fcu.driver
        drv.type = 'SCRIPTED'
        drv.expression = expr
        for v in list(drv.variables):
            drv.variables.remove(v)
        var = drv.variables.new()
        var.name = "d"
        var.type = 'TRANSFORMS'
        t = var.targets[0]
        t.id = rig
        t.bone_target = "CTL-Jaw"
        t.transform_type = loc_axis
        t.transform_space = 'LOCAL_SPACE'

    # UP/DOWN handle motion (local Z) -> hinge open about local X.  On THIS hinge
    # frame a NEGATIVE rot_x drops the chin (open) and a positive one raises it
    # (clench) - verified live.  So: down (d<0) -> negative rot (open) up to -OM;
    # up (d>0) -> positive rot (clench) up to +CM.
    _jaw_driver("MCH-JawRot", 0,
                "(d/%.6f*%.6f) if d < 0 else (d/%.6f*%.6f)"
                % (OT, OM, CT, CM), 'LOC_Z')
    # LEFT/RIGHT handle motion (local X) -> sway about the vertical (JawSway Y).
    # negated so the jaw follows the handle to the SAME side - verified live.
    _jaw_driver("MCH-JawSway", 1, "-d/%.6f*%.6f" % (ST, SM), 'LOC_X')

    # each corner copies HALF the hinge open so the corners follow the jaw a
    # little (natural lip motion); sway they inherit by parenting to MCH-JawSway.
    for mch, fac in jaw_wiring:
        pb = rig.pose.bones.get(mch)
        if pb is None:
            continue
        for con in list(pb.constraints):
            if con.name == "SR Jaw Follow":
                pb.constraints.remove(con)
        con = pb.constraints.new('COPY_ROTATION')
        con.name = "SR Jaw Follow"
        con.target = rig
        con.subtarget = "MCH-JawRot"
        con.target_space = 'LOCAL'
        con.owner_space = 'LOCAL'
        con.influence = fac

    # CTL-Jaw = a pure MOVE handle: no rotation / no scale, forward-back (local Y)
    # locked, so it only travels up/down/left/right.  Bound the travel so open /
    # clench / sway can't be over-driven.
    pj = rig.pose.bones.get("CTL-Jaw")
    if pj is not None:
        pj.rotation_mode = 'XYZ'
        pj.lock_location = (False, True, False)   # X + Z free, Y (fwd/back) off
        pj.lock_rotation = (True, True, True)
        pj.lock_rotation_w = True
        pj.lock_scale = (True, True, True)
        for con in list(pj.constraints):
            if con.name == "SR Jaw Limit":
                pj.constraints.remove(con)
        lim = pj.constraints.new('LIMIT_LOCATION')
        lim.name = "SR Jaw Limit"
        lim.owner_space = 'LOCAL'
        lim.use_min_x = lim.use_max_x = True
        lim.min_x, lim.max_x = -ST, ST           # left / right sway travel
        lim.use_min_y = lim.use_max_y = True
        lim.min_y = lim.max_y = 0.0              # forward / back locked
        lim.use_min_z = lim.use_max_z = True
        lim.min_z, lim.max_z = -OT, CT           # down = open, up = clench
        lim.use_transform_limit = True

    # corner deform bones follow their tweak circle
    for dn, tw in corner_follow:
        pb = rig.pose.bones.get(dn)
        if pb is None:
            continue
        for con in list(pb.constraints):
            if con.name == "SR Follow":
                pb.constraints.remove(con)
        con = pb.constraints.new('DAMPED_TRACK')
        con.name = "SR Follow"
        con.target = rig
        con.subtarget = tw
        con.track_axis = 'TRACK_Y'
        con.influence = 1.0

    # ribbon mechanism per column:
    #   MCH-mtgt = rest + (seal-rest)*max(0,-m) + (open-rest)*max(0,m)
    #   DEF-lip  Damp-Tracks its tweak (keeps length -> slides over the teeth)
    def _slider_infl(con, slb, expr):
        con.influence = 0.0
        try:
            fcu = con.driver_add("influence")
            drv = fcu.driver
            drv.type = 'SCRIPTED'
            drv.expression = expr
            for v in list(drv.variables):
                drv.variables.remove(v)
            var = drv.variables.new()
            var.name = "m"
            var.type = 'SINGLE_PROP'
            tg = var.targets[0]
            tg.id = rig
            tg.data_path = 'pose.bones["%s"]["mopen"]' % slb
        except Exception:
            pass

    for w in col_wiring:
        pb = rig.pose.bones.get(w["tgt"])
        if pb is not None:
            for con in list(pb.constraints):
                pb.constraints.remove(con)
            cs = pb.constraints.new('COPY_LOCATION')
            cs.name = "SR Seal"
            cs.target = rig
            cs.subtarget = w["seal"]
            _slider_infl(cs, w["slider"], "max(0.0, -m)")
            co = pb.constraints.new('COPY_LOCATION')
            co.name = "SR Open"
            co.target = rig
            co.subtarget = w["open"]
            _slider_infl(co, w["slider"], "max(0.0, m)")
        db = rig.pose.bones.get(w["defb"])
        if db is not None:
            for con in list(db.constraints):
                db.constraints.remove(con)
            dt = db.constraints.new('DAMPED_TRACK')
            dt.name = "SR Slide"
            dt.target = rig
            dt.subtarget = w.get("track", w["tgt"])
            dt.track_axis = 'TRACK_Y'
            dt.influence = 1.0
        # CORNER FOLLOW: the tweak ADDS a fraction of the nearest corner control's
        # LOCAL motion, so the lip bone sweeps into a smile / frown with the
        # corner (both bones are Z-up so local axes match -> correct direction).
        cf = float(w.get("cfollow", 0.0))
        cb = w.get("cbone")
        tb = rig.pose.bones.get(w.get("track", ""))
        if tb is not None and cb and cf > 1e-3:
            for con in list(tb.constraints):
                if con.name == "SR Corner Follow":
                    tb.constraints.remove(con)
            cfc = tb.constraints.new('COPY_LOCATION')
            cfc.name = "SR Corner Follow"
            cfc.target = rig
            cfc.subtarget = cb
            cfc.use_offset = True
            cfc.target_space = 'LOCAL'
            cfc.owner_space = 'LOCAL'
            cfc.influence = cf

    # the open / seal slider: drag UP = open (+1), DOWN = seal (-1)
    pbb = rig.pose.bones.get(slb)
    if pbb is not None:
        pbb["mopen"] = 0.0
        try:
            uip = pbb.id_properties_ui("mopen")
            uip.update(min=-1.0, max=1.0, soft_min=-1.0, soft_max=1.0,
                       description="Mouth master: -1 = lips sealed (pressed "
                                   "together), 0 = rest, +1 = open (jaw drop)")
        except Exception:
            pass
        pbb.lock_location = (True, False, True)
        pbb.lock_rotation = (True, True, True)
        pbb.lock_rotation_w = True
        pbb.lock_scale = (True, True, True)
        L = 1.1 * LV
        for con in list(pbb.constraints):
            if con.name == "SR Mouth Limit":
                pbb.constraints.remove(con)
        lim = pbb.constraints.new('LIMIT_LOCATION')
        lim.name = "SR Mouth Limit"
        lim.owner_space = 'LOCAL'
        lim.use_min_y = lim.use_max_y = True
        lim.min_y = -L
        lim.max_y = L
        lim.use_transform_limit = True
        try:
            fcu = pbb.driver_add('["mopen"]')
            drv = fcu.driver
            drv.type = 'SCRIPTED'
            drv.expression = "max(-1.0, min(1.0, ly / %.6f))" % L
            for v in list(drv.variables):
                drv.variables.remove(v)
            var = drv.variables.new()
            var.name = "ly"
            var.type = 'TRANSFORMS'
            t = var.targets[0]
            t.id = rig
            t.bone_target = slb
            t.transform_type = 'LOC_Y'
            t.transform_space = 'LOCAL_SPACE'
        except Exception:
            pass

    # widgets - modern, purpose-built mouth shapes
    ring = _eye._wgt("WGT-SR-eye-ringF", "ring")     # thin circle = per-point tweak
    arrowud = _eye._wgt("WGT-SR-eye-arrowud", "arrowud")  # double chevron = slider
    frame = _mwgt("WGT-SR-mouth-frame", "frame")     # rounded mouth outline
    handle = _mwgt("WGT-SR-mouth-handle", "handle")  # rounded-square corner grip
    jawhook = _mwgt("WGT-SR-mouth-jaw", "jawhook")   # chin hook
    ring_up = _mwgt("WGT-SR-mouth-ringU", "ringup")  # upper-lip tweak (raised)
    ring_dn = _mwgt("WGT-SR-mouth-ringD", "ringdn")  # lower-lip tweak (lowered)

    def _apply(lst, shape):
        for name, size, pal in lst:
            pb = rig.pose.bones.get(name)
            if pb:
                pb.custom_shape = shape
                pb.custom_shape_scale_xyz = (size, size, size)
                pb.use_custom_shape_bone_size = False
                if pal:
                    _eye._pal(rig, name, pal)

    def _apply_xy(lst, shape):
        # non-uniform: size = (width, height) in X-Z
        for name, size, pal in lst:
            pb = rig.pose.bones.get(name)
            if pb:
                pb.custom_shape = shape
                pb.custom_shape_scale_xyz = (size[0], 1.0, size[1])
                pb.use_custom_shape_bone_size = False
                if pal:
                    _eye._pal(rig, name, pal)
    _apply(made_ring, ring)
    _apply(made_ring_up, ring_up)
    _apply(made_ring_dn, ring_dn)
    _apply(made_arrow, arrowud)
    _apply(made_handle, handle)
    _apply(made_jaw, jawhook)
    _apply_xy(made_frame, frame)

    # master blue lid-line widgets tracing each lip
    wcoll = bpy.data.collections.get("WGTS_soulify")
    if wcoll is None:
        wcoll = bpy.data.collections.new("WGTS_soulify")
        bpy.context.scene.collection.children.link(wcoll)
        wcoll.hide_viewport = True
        wcoll.hide_render = True
    for bone_name, poly in master_wdata:
        pb = rig.pose.bones.get(bone_name)
        if pb is None or len(poly) < 2:
            continue
        try:
            binv = rig.data.bones[bone_name].matrix_local.inverted()
        except Exception:
            continue
        V = [tuple(binv @ Vector(p)) for p in poly]
        E = [(i, i + 1) for i in range(len(V) - 1)]
        wname = "WGT-SR-lipline_%s" % bone_name
        old = bpy.data.objects.get(wname)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        wme = bpy.data.meshes.new(wname)
        wme.from_pydata(V, E, [])
        wme.update()
        wob = bpy.data.objects.new(wname, wme)
        wob["sr_wgt"] = True
        wcoll.objects.link(wob)
        pb.custom_shape = wob
        pb.use_custom_shape_bone_size = False
        pb.custom_shape_scale_xyz = (1.0, 1.0, 1.0)
        _eye._pal(rig, bone_name, _C_BLUE)

    bpy.ops.object.mode_set(mode='OBJECT')

    try:
        _build_ribbons(context, rig, ribbon_specs)
    except Exception as e:
        info["ribbon_error"] = str(e)

    if autobind and bind_data is not None:
        try:
            info["bound"] = _bind_mouth(context, rig, body, bind_data)
        except Exception as e:
            info["bind_error"] = str(e)

    _hide_mcorner_markers(True)
    if prev is not None:
        context.view_layer.objects.active = prev
    return info


RIBBON_COLL = "SR Mouth Ribbons"


def _build_ribbons(context, rig, specs):
    """A thin guide-ribbon mesh per lip, bound to that lip's target bones so it
    deforms with the rig (a visible, paintable lip guide; never rendered)."""
    coll = bpy.data.collections.get(RIBBON_COLL)
    if coll is None:
        coll = bpy.data.collections.new(RIBBON_COLL)
        context.scene.collection.children.link(coll)
    coll.hide_render = True
    for name, part, cols, open_pts, up, sgn, S in specs:
        old = bpy.data.objects.get(name)
        if old is not None:
            me = old.data
            bpy.data.objects.remove(old, do_unlink=True)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        pts = [Vector(p) for p in open_pts]
        if len(pts) < 2:
            continue
        upv = Vector(up)
        w = 0.35 * S * float(sgn)
        V, F = [], []
        for k, p in enumerate(pts):
            V.append(p)
            V.append(p + upv * w)
        for k in range(len(pts) - 1):
            a, b = 2 * k, 2 * k + 1
            cc, d = 2 * (k + 1), 2 * (k + 1) + 1
            F.append((a, cc, d, b))
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in V], [], F)
        me.update()
        ob = bpy.data.objects.new(name, me)
        ob["sr_wgt"] = True
        coll.objects.link(ob)
        for idx, (col_i, tgt) in enumerate(cols):
            vg = ob.vertex_groups.new(name=tgt)
            vg.add([2 * idx, 2 * idx + 1], 1.0, 'REPLACE')
        md = ob.modifiers.new(name="SR Ribbon", type='ARMATURE')
        md.object = rig
        ob.parent = rig
        ob.display_type = 'WIRE'
        ob.hide_render = True


# ------------------------------------------------------------------- binding
def _bind_mouth(context, rig, body, sd):
    """Skin ONLY the lip-region verts to the lip/corner bones, bounded to the
    mouth (recognised from the registered loop).  Partition-of-unity along the
    lip x a smoothstep fall-off from the lip margin toward the skin.  1:1 upper/
    lower pairing (via topological smoothing) lets the lips zip together on seal.
    """
    me = body.data
    mw = body.matrix_world
    co = np.array([list(mw @ v.co) for v in me.vertices], float)
    nv = len(me.vertices)

    if not any(m.type == 'ARMATURE' and m.object == rig for m in body.modifiers):
        md = body.modifiers.new(name="SR Armature", type='ARMATURE')
        md.object = rig

    bones = set()
    for _u, bn, _pt in (sd["upp"] + sd["low"]):
        bones.add(bn)
    bones.add("DEF-lip_corn.L")
    bones.add("DEF-lip_corn.R")
    for bn in bones:
        vg = body.vertex_groups.get(bn)
        if vg is not None:
            vg.remove(range(nv))
        else:
            body.vertex_groups.new(name=bn)

    c = sd["c"]
    right, up, fwd = sd["right"], sd["up"], sd["fwd"]
    loop, band, vmax = sd["loop"], sd["band"], sd["vmax"]
    upp = [(u2, bn) for u2, bn, _p in sd["upp"]]
    low = [(u2, bn) for u2, bn, _p in sd["low"]]
    # append the shared corner anchors to BOTH lips so the commissure verts get
    # weighted to the corner bones (CTL-Lips_corn then actually deforms).
    cornerL = sd.get("cornerL")
    cornerR = sd.get("cornerR")
    corners = [c2 for c2 in (cornerR, cornerL) if c2 is not None]
    upp_c = sorted(upp + corners, key=lambda t: t[0])
    low_c = sorted(low + corners, key=lambda t: t[0])
    rel = co - c
    d = np.linalg.norm(rel, axis=1)
    ffv = rel @ fwd
    uu = rel @ right
    vvv = rel @ up

    # DEPTH GATE: the mouth CORNERS (commissures) tuck BACK into the face - on a
    # real head they recede ~1x band behind the mouth-centre plane.  The old gate
    # (ffv > -0.5*band) cut them off, so the commissure verts were never assigned
    # to the lips and fell through to the JAW field (head/jaw split) - the corner
    # went DEAD and the jaw pinched it on open.  The true spatial bound is the 3D
    # dl<=band sphere around the loop below, so this coarse depth pre-filter only
    # needs to keep the receding corners IN and the deep mouth-bag OUT.
    # LONG tangential reach (like ARP) but a SHORT depth zone so the weights sweep
    # far ACROSS the face surface without swallowing the interior mouth-bag.
    _dband = max(2.4 * sd["mh"], 0.16 * sd["span"])     # depth (LV-based, short)
    cand = ((ffv > -_dband) &
            (np.abs(vvv) < max(1.8 * vmax, 1.2 * band)) &
            (uu > sd["u_in"] - max(0.18 * sd["span"], 0.9 * band)) &
            (uu < sd["u_out"] + max(0.18 * sd["span"], 0.9 * band)))

    # ------------------------------------------------------------------ LIP
    # RECOGNITION (topological).  A flat height cut (vvv>=0) splits the mouth on
    # a straight line through its centre - but the two lips TOUCH on a closed
    # mouth, so the lower lip's inner edge sits ABOVE that line and gets
    # mislabelled UPPER = static.  On jaw-open those verts stay stuck while the
    # rest of the lower lip drops -> a boxy, torn, "stuck lower lip" open.
    # Instead we recognise the lips by MESH TOPOLOGY: the registered loop is the
    # lip contact contour; remove it as a separator and the lip-band skin falls
    # into an UPPER component (above the loop) and a LOWER component (below).
    # Every vert is labelled by the lip it is actually attached to, so the whole
    # lower lip - inner edge included - follows the jaw as one clean piece.
    loopset = set()
    _vgm = body.vertex_groups.get("SR_mouthloop")
    if _vgm is not None:
        _gil = _vgm.index
        for v in me.vertices:
            if any(g.group == _gil for g in v.groups):
                loopset.add(int(v.index))
    # commissure verts (nearest mesh vert to each corner) also separate the two
    # lips where the loop opens out at the corners.
    cornersep = set()
    for _cp in (sd.get("cornerL_pt"), sd.get("cornerR_pt")):
        if _cp is not None:
            cornersep.add(int(np.argmin(np.linalg.norm(co - np.asarray(_cp, float),
                                                        axis=1))))
    # lip-band skin verts = candidates within band of the loop.
    band_verts = []
    for vi in np.nonzero(cand)[0]:
        vi = int(vi)
        if float(np.min(np.linalg.norm(loop - co[vi], axis=1))) <= band:
            band_verts.append(vi)
    node = set(band_verts) | loopset
    adj2 = {vi: set() for vi in node}
    for e in me.edges:
        a, b = int(e.vertices[0]), int(e.vertices[1])
        if a in adj2 and b in adj2:
            adj2[a].add(b)
            adj2[b].add(a)
    # multi-source BFS FROM the loop: each loop vert is a seed carrying its own
    # margin side (upper margin vv>=0 -> +1, lower margin vv<0 -> -1).  Every
    # band vert takes the side of the geodesically NEAREST loop vert, flooding
    # outward through the mesh.  Because the two lips meet only ON the loop, a
    # vert above the contact is reached first by an UPPER-margin seed and a vert
    # below by a LOWER-margin seed - so the lower lip's inner edge (which sits
    # just above the mouth centre on a closed mouth) is still labelled LOWER and
    # drops with the jaw.  No flat height cut, no merged components.
    from collections import deque
    # ⭐ split the LOOP into an UPPER arc and a LOWER arc using the two CORNERS
    # (Saeed: "you know where the corner is, upper/lower should be SIMPLE").  The
    # loop is a ring; cut it at the two commissures -> two arcs.  The higher-mean-
    # height arc = UPPER-lip contact (stays), the other = LOWER (drops FULLY on
    # jaw-open).  THIS is what opens the mouth across its whole width: the old
    # vv>=0 height cut left most of the 3mm-tall contact line labelled "upper" =
    # sealed, so it opened only as a slit at the bottom centre.
    loop_adj = {vi: set() for vi in loopset}
    for e in me.edges:
        a, b = int(e.vertices[0]), int(e.vertices[1])
        if a in loop_adj and b in loop_adj:
            loop_adj[a].add(b)
            loop_adj[b].add(a)
    cpts = []
    for _cp in (sd.get("cornerL_pt"), sd.get("cornerR_pt")):
        if _cp is not None:
            cpts.append(int(np.argmin(np.linalg.norm(
                co - np.asarray(_cp, float), axis=1))))
    loop_side = {}
    if len(cpts) == 2 and all(len(loop_adj[cc]) >= 2 for cc in cpts):
        cL, cR = cpts

        def _arc(first):
            path, prev, cur, steps = [], cL, first, 0
            while cur is not None and cur != cR and steps < len(loopset) + 2:
                path.append(cur)
                nxts = [n for n in loop_adj[cur] if n != prev]
                prev, cur = cur, (nxts[0] if nxts else None)
                steps += 1
            return path, (cur == cR)
        nbs = list(loop_adj[cL])
        arc1, ok1 = _arc(nbs[0])
        arc2, ok2 = _arc(nbs[1]) if len(nbs) > 1 else ([], False)
        if arc1 and arc2 and ok1 and ok2:
            m1 = float(np.mean([vvv[v] for v in arc1]))
            m2 = float(np.mean([vvv[v] for v in arc2]))
            upper_arc, lower_arc = (arc1, arc2) if m1 >= m2 else (arc2, arc1)
            for v in upper_arc:
                loop_side[v] = 1
            for v in lower_arc:
                loop_side[v] = -1
    side_of = {}
    dq = deque()
    for vi in loopset:
        if vi in node:
            # arc label if we have it; else fall back to the height sign.
            side_of[vi] = loop_side.get(vi, 1 if vvv[vi] >= 0.0 else -1)
            dq.append(vi)
    while dq:
        x = dq.popleft()
        sx = side_of[x]
        for y in adj2[x]:
            if y not in side_of:
                side_of[y] = sx
                dq.append(y)

    wmap = {}
    w_lid_of = {}
    assigned = set()
    lip_side = {}                     # vi -> +1 upper lip, -1 lower lip
    # RADIAL per-bone weighting (professional - like the corner): every lip column
    # paints a smooth smoothstep gradient around ITS OWN point (strong at the
    # bone, fading out over col_reach), and the overlaps normalise.  So each bone
    # gets a clean ROUND gradient exactly like the corner weight Saeed pointed at,
    # not a flat linear split.
    K_side = max(len(sd["upp"]), len(sd["low"]), 1)
    # STEEP core so the NEAREST column dominates (high peak = strong control),
    # ~1.35x the column spacing - NOT tied to the long band (that over-spread and
    # dropped every peak to ~0.17 = mushy, control felt gone).  The long smooth
    # TAIL comes from the diffusion below + the band region, not from a giant core.
    col_reach = 1.35 * sd["span"] / max(K_side - 1, 1)
    side_pts = {1: [(bn, np.asarray(pt, float)) for (_u, bn, pt) in sd["upp"]],
                -1: [(bn, np.asarray(pt, float)) for (_u, bn, pt) in sd["low"]]}
    for vi in np.nonzero(cand)[0]:
        vi = int(vi)
        dl = float(np.min(np.linalg.norm(loop - co[vi], axis=1)))
        if dl > band:
            continue
        t = dl / band
        w_lid = 1.0 - t * t * (3.0 - 2.0 * t)         # smoothstep: 1 margin->0
        if w_lid <= 1e-4:
            continue
        is_upper = side_of.get(vi, vvv[vi] >= 0.0) == 1
        lip_side[vi] = 1 if is_upper else -1
        m = wmap.setdefault(vi, {})
        p = co[vi]
        pts = side_pts[1 if is_upper else -1]
        for bn, pt in pts:
            d = float(np.linalg.norm(p - pt))
            x = 1.0 - d / col_reach
            if x <= 1e-4:
                continue
            m[bn] = m.get(bn, 0.0) + x * x * (3.0 - 2.0 * x)    # smooth radial
        if not m and pts:                             # beyond every reach: nearest
            bn = min(pts, key=lambda bp: np.linalg.norm(p - bp[1]))[0]
            m[bn] = 1.0
        w_lid_of[vi] = max(w_lid_of.get(vi, 0.0), w_lid)
        assigned.add(vi)

    # smooth the bone distribution to de-facet the deform - but ONLY between verts
    # of the SAME lip (upper<->upper, lower<->lower).  Crossing the seam would
    # bleed lower-lip (jaw) bones onto the upper lip and vice versa; keeping it
    # same-side is what makes the two lips move INDEPENDENTLY - a jaw drop takes
    # the lower lip only, the upper lip stays put.  The shared corner bones keep
    # the commissure continuous.  (The closed-mouth seal comes from the ribbon
    # OVERLAP geometry, not from weight bleed, so nothing is lost here.)
    if assigned:
        nbrs = {vi: set() for vi in assigned}
        for e in me.edges:
            a, b = int(e.vertices[0]), int(e.vertices[1])
            if a in nbrs and b in nbrs and lip_side.get(a) == lip_side.get(b):
                nbrs[a].add(b)
                nbrs[b].add(a)
        # HEAT-MAP-STYLE surface diffusion: spread each bone's weight smoothly
        # ALONG the mesh surface over MANY passes (like Blender's bone-heat), so
        # every bone gets a long, smooth, professional falloff (the "distance"
        # ARP has) instead of a short hard patch.  A moderate self-weight keeps
        # each bone's PEAK at its own point (crisp control) while the tail spreads
        # far.  Same-side only so upper/lower stay independent.
        SELF_W = 3.2
        for _ in range(6):
            new = {}
            for vi in assigned:
                acc = {bn: w * SELF_W for bn, w in wmap[vi].items()}
                for nj in nbrs[vi]:
                    for bn, w in wmap[nj].items():
                        acc[bn] = acc.get(bn, 0.0) + w
                tot = sum(acc.values()) or 1.0
                new[vi] = {bn: w / tot for bn, w in acc.items()}
            wmap = new

    # CORNER emphasis (applied AFTER smoothing so it isn't averaged away): near
    # each commissure the corner DEF bone takes the LEAD from the lip columns so
    # CTL-Lips_corn actually deforms the corner and a smile / frown pulls it.
    # BUT it is a SMOOTH PEAK, not a rigid lock:
    #   * peak is CAPPED at CORNER_PEAK (< 1) so the commissure KEEPS a share of
    #     the neighbouring lip columns - and on the LOWER side those columns ride
    #     the jaw, so the corner FOLLOWS the jaw open instead of pinching, and a
    #     big smile still drags a broad region (the corner never rigidly owns a
    #     block).
    #   * reach c_rad is generous so the influence fades gently over ~1/5 of the
    #     mouth, which reads as a professional smile/frown, not a local tug.
    CORNER_PEAK = 1.0
    corner_defs = []
    if sd.get("cornerL_pt") is not None:
        corner_defs.append((np.asarray(sd["cornerL_pt"], float), "DEF-lip_corn.L"))
    if sd.get("cornerR_pt") is not None:
        corner_defs.append((np.asarray(sd["cornerR_pt"], float), "DEF-lip_corn.R"))
    # MODERATE corner reach: the corner OWNS a clean chunk at each commissure,
    # but does NOT eat the lip columns' weights across the whole lip (that made
    # the individual CTL-LipT_upp/low tweaks weak = un-professional).  The BROAD
    # smile/frown comes from the CONTROL-FOLLOW (the lip tweaks follow the corner
    # bone-wise) deforming via their OWN crisp per-column weights - the ARP way:
    # crisp weights + a follow rig, NOT one bone's weight bleeding everywhere.
    c_rad = max(0.7 * sd["span"], 1.8 * band)   # ~ARP corner_mini reach (76mm)
    for vi in list(wmap.keys()):
        p = co[vi]
        for cpt, cbn in corner_defs:
            dcn = float(np.linalg.norm(p - cpt))
            if dcn >= c_rad:
                continue
            x = 1.0 - dcn / c_rad
            wc = x * x * (3.0 - 2.0 * x) * CORNER_PEAK   # smooth peak, capped
            if wc <= 1e-3:
                continue
            m = wmap[vi]
            s = 1.0 - wc
            for bn in list(m.keys()):
                if bn != cbn:
                    m[bn] *= s
            m[cbn] = m.get(cbn, 0.0) + wc

    jaw_bone = sd.get("jaw", "DEF-Jaw")
    if jaw_bone not in rig.data.bones:
        jaw_bone = None
    jaw_g = None
    if jaw_bone:
        jaw_g = (body.vertex_groups.get(jaw_bone) or
                 body.vertex_groups.new(name=jaw_bone))
        jaw_g.remove(range(nv))

    # write lip weights; the crease remainder of LOWER-lip verts goes to the JAW
    # (so it moves with the jaw = no tear on open), of UPPER-lip verts to the
    # head (static upper face).
    head_rem = {}
    jaw_rem = {}
    for vi, m in wmap.items():
        wl = w_lid_of.get(vi, 1.0)
        tot = sum(m.values()) or 1.0
        for bn, w in m.items():
            ww = (w / tot) * wl
            if ww <= 1e-4:
                continue
            vg = body.vertex_groups.get(bn)
            if vg is not None:
                vg.add([vi], float(ww), 'REPLACE')
        rem = 1.0 - wl
        if vvv[vi] < 0 and jaw_g is not None:
            jaw_rem[vi] = rem
        else:
            head_rem[vi] = rem

    # strip pre-existing head/face/neck skinning off the lip verts (keep DEF-lip
    # AND DEF-Jaw), hand the remainder to head / jaw so the deform fades cleanly.
    al = list(assigned)
    mass = {}
    for g in body.vertex_groups:
        if g.name.startswith("DEF-") and \
           not g.name.startswith(("DEF-lip", "DEF-Jaw")):
            gi = g.index
            tot = 0.0
            for vi in al:
                for gg in me.vertices[vi].groups:
                    if gg.group == gi:
                        tot += gg.weight
                        break
            if tot > 0.0:
                mass[g.name] = tot
    head_bone = max(mass, key=mass.get) if mass else None
    if head_bone is None:
        for gn in ("DEF-head", "head", "DEF-spine.006", "DEF-neck"):
            if gn in rig.data.bones:
                head_bone = gn
                break
    if al:
        for g in list(body.vertex_groups):
            if (g.name.startswith("DEF-") or g.name in ("head", "face")) and \
               not g.name.startswith(("DEF-lip", "DEF-Jaw")):
                g.remove(al)
    hg = (body.vertex_groups.get(head_bone) or
          body.vertex_groups.new(name=head_bone)) if head_bone else None
    for vi, rem in head_rem.items():
        if hg is not None and rem > 1e-4:
            hg.add([vi], float(max(0.0, min(1.0, rem))), 'REPLACE')
    for vi, rem in jaw_rem.items():
        if jaw_g is not None and rem > 1e-4:
            jaw_g.add([vi], float(max(0.0, min(1.0, rem))), 'REPLACE')

    # ---- PROFESSIONAL JAW WEIGHTS (ARP / Rigify style): a SMOOTH gradient over
    # the whole lower jaw - the lower-lip crease, chin, jaw, lower cheeks and the
    # mouth-bag floor - FULL at the chin/jaw, fading smoothly to 0 UP across the
    # mouth line toward the nose, OUT toward the ears, and DOWN into the neck.
    # Every boundary is a smoothstep (no hard box), so opening the jaw has NO
    # crease. Verts split their weight between the jaw (what moves) and the head
    # (what stays), so the whole lower face swings as one piece about the hinge.
    if jaw_g is not None:
        span = sd["span"]
        LV = sd.get("mh", band * 2.0)              # lip vertical zone (from build)

        def _ss(x):
            x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
            return x * x * (3.0 - 2.0 * x)

        v_top, v_topband = 0.8 * LV, 2.6 * LV      # fade up to the nose
        v_bot, v_botband = -9.0 * LV, 3.0 * LV     # fade down into the neck
        s_half, s_band = 1.35 * span, 0.6 * span   # fade out to the ears
        # the chin & jaw RECEDE backward as they drop, so the depth gate must
        # reach well back to catch them (else only the front mouth patch moves =
        # a hard rectangle); but stop before the throat / back of the head.
        d_back = 1.5 * span

        jaw_field = {}
        for vi in range(nv):
            if vi in assigned:
                continue
            fv = float(ffv[vi])
            if fv < -d_back:
                continue
            vY = float(vvv[vi])
            uX = float(uu[vi])
            wv = _ss((v_top - vY) / v_topband) * _ss((vY - v_bot) / v_botband)
            if wv <= 1e-3:
                continue
            ws = _ss((s_half - abs(uX)) / s_band)
            wj = wv * ws
            if wj > 1e-3:
                jaw_field[vi] = min(1.0, wj)

        al_j = list(jaw_field)
        if al_j:
            for g in list(body.vertex_groups):
                if (g.name.startswith("DEF-") or g.name in ("head", "face")) \
                        and not g.name.startswith(("DEF-lip", "DEF-Jaw")):
                    g.remove(al_j)
            for vi, wj in jaw_field.items():
                jaw_g.add([vi], float(wj), 'REPLACE')
                if hg is not None and (1.0 - wj) > 1e-3:
                    hg.add([vi], float(1.0 - wj), 'REPLACE')

    # de-facet the lip deformation at the extremes (full open / seal) with a
    # Corrective Smooth scoped to the lip region, EXCLUDING the margin line so
    # the seal stays crisp (same trick as the eye).
    try:
        _mouth_smooth_modifier(body, assigned)
    except Exception:
        pass
    return len(assigned)


def _mouth_smooth_modifier(body, assigned):
    me = body.data
    nv = len(me.vertices)
    grp = (body.vertex_groups.get("SR_mouth_smooth") or
           body.vertex_groups.new(name="SR_mouth_smooth"))
    grp.remove(range(nv))
    if not assigned:
        return
    margin = set()
    vgm = body.vertex_groups.get("SR_mouthloop")
    if vgm is not None:
        gi = vgm.index
        for v in me.vertices:
            if any(g.group == gi for g in v.groups):
                margin.add(v.index)
    adj = {}
    core = set(assigned)
    for e in me.edges:
        a, b = int(e.vertices[0]), int(e.vertices[1])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    region = set(core)
    cur = set(core)
    rings = []
    for _ in range(2):
        nxt = set()
        for vi in cur:
            nxt |= adj.get(vi, set())
        nxt -= region
        region |= nxt
        rings.append(nxt)
        cur = nxt
    grp.add([v for v in core if v not in margin], 1.0, 'REPLACE')
    if len(rings) > 0 and rings[0]:
        grp.add([v for v in rings[0] if v not in margin], 0.55, 'REPLACE')
    if len(rings) > 1 and rings[1]:
        grp.add([v for v in rings[1] if v not in margin], 0.28, 'REPLACE')
    mod = None
    for m in body.modifiers:
        if m.name == "SR Mouth Smooth":
            mod = m
            break
    if mod is None:
        mod = body.modifiers.new("SR Mouth Smooth", 'CORRECTIVE_SMOOTH')
    mod.vertex_group = "SR_mouth_smooth"
    # DISABLED for now (Saeed: show the REAL weights, no smoothing).  The
    # LENGTH_WEIGHTED corrective smooth fought the jaw open and dragged the lower
    # lip back UP unevenly ("الشفايف لاصقة فوق").  Left inert (factor 0, hidden)
    # so the deformation is driven purely by the bone weights we can inspect.
    mod.factor = 0.0
    mod.iterations = 1
    mod.smooth_type = 'SIMPLE'
    mod.rest_source = 'ORCO'
    mod.show_viewport = False
    mod.show_render = False
    try:
        while body.modifiers[-1].name != "SR Mouth Smooth":
            body.modifiers.move(body.modifiers.find("SR Mouth Smooth"),
                                len(body.modifiers) - 1)
            break
    except Exception:
        pass


# --------------------------------------------------------------------- clear
def clear_mouth_rig(context, also_registration=False):
    rig = _target_rig(context)
    body = _body(context)
    removed = {"bones": 0, "weight_groups": 0, "restored_verts": 0,
               "registrations": 0}
    try:
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    head_bone = None
    if rig is not None:
        for n in ("DEF-head", "head", "DEF-spine.006", "DEF-neck"):
            if n in rig.data.bones:
                head_bone = n
                break

    if body is not None and body.type == 'MESH':
        for m in list(body.modifiers):
            if m.name == "SR Mouth Smooth":
                body.modifiers.remove(m)
        gsm = body.vertex_groups.get("SR_mouth_smooth")
        if gsm is not None:
            body.vertex_groups.remove(gsm)
        dgroups = [g for g in body.vertex_groups
                   if g.name.startswith(("DEF-lip", "DEF-Jaw"))]
        gidx = {g.index for g in dgroups}
        vset = set()
        if gidx:
            for v in body.data.vertices:
                if any(gg.group in gidx for gg in v.groups):
                    vset.add(v.index)
        if head_bone and vset:
            hg = body.vertex_groups.get(head_bone) or \
                body.vertex_groups.new(name=head_bone)
            hg.add(list(vset), 1.0, 'REPLACE')
            removed["restored_verts"] = len(vset)
        for g in list(dgroups):
            body.vertex_groups.remove(g)
            removed["weight_groups"] += 1
        if also_registration:
            g = body.vertex_groups.get("SR_mouthloop")
            if g is not None:
                body.vertex_groups.remove(g)
                removed["registrations"] += 1

    if rig is not None:
        prev = context.view_layer.objects.active
        context.view_layer.objects.active = rig
        rig.hide_set(False)
        bpy.ops.object.mode_set(mode='EDIT')
        eb = rig.data.edit_bones
        pref = ("MSTR-Mouth", "CTL-Lip", "DEF-lip", "MCH-mtgt", "CTL-MouthOpen",
                "MCH-Mouth", "CTL-Jaw", "DEF-Jaw", "MCH-JawCorn", "MCH-JawRot",
                "MCH-JawRet", "MCH-JawSway")
        for b in list(eb):
            if b.name.startswith(pref):
                eb.remove(b)
                removed["bones"] += 1
        bpy.ops.object.mode_set(mode='OBJECT')
        if prev is not None:
            context.view_layer.objects.active = prev

    for part in ("upp", "low"):
        o = bpy.data.objects.get("HLP-SR-liprib_%s" % part)
        if o is not None:
            m = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if m is not None and m.users == 0:
                bpy.data.meshes.remove(m)
    rc = bpy.data.collections.get(RIBBON_COLL)
    if rc is not None and not rc.objects:
        bpy.data.collections.remove(rc)

    if also_registration:
        for nm in ("SR_mcorner.L", "SR_mcorner.R"):
            o = bpy.data.objects.get(nm)
            if o is not None:
                me = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                if me is not None and me.users == 0:
                    bpy.data.meshes.remove(me)
                removed["registrations"] += 1
    return removed


# ------------------------------------------------------------------- operators
class SMARTRIG_OT_mouth_register(bpy.types.Operator):
    bl_idname = "smartrig.mouth_register"
    bl_label = "Register Mouth Loop"
    bl_description = ("In Edit mode on the face, select the lip EDGE LOOP "
                      "(Alt+click) - the whole lip contour (upper + lower "
                      "margin) - then run this.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        body = _body(context)
        if body is None or body.type != 'MESH':
            self.report({'ERROR'}, "Set the Target Mesh (the face) first")
            return {'CANCELLED'}
        was_edit = (context.mode == 'EDIT_MESH')
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        sel = [v.index for v in body.data.vertices if v.select]
        if len(sel) < 8:
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, "Select the lip loop (Edit mode, Alt+click), "
                        "then run this")
            return {'CANCELLED'}
        vg = body.vertex_groups.get("SR_mouthloop")
        if vg is None:
            vg = body.vertex_groups.new(name="SR_mouthloop")
        else:
            vg.remove([v.index for v in body.data.vertices])
        vg.add(sel, 1.0, 'REPLACE')
        self.report({'INFO'}, "Mouth loop registered (%d verts)" % len(sel))
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class SMARTRIG_OT_mouth_corner_marker(bpy.types.Operator):
    bl_idname = "smartrig.mouth_corner_marker"
    bl_label = "Mouth Corner Markers"
    bl_description = ("Spawn two movable coloured markers - BLUE = left corner, "
                      "RED = right corner. Grab (G) each and drop it exactly on "
                      "the mouth corner, then Build. Press again to re-select "
                      "them (your positions are kept).")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        body = _body(context)
        loop = _mouth_loop(body) if body is not None else None
        if loop is None or len(loop) < 8:
            self.report({'ERROR'}, "Register the mouth loop first")
            return {'CANCELLED'}
        exists = (bpy.data.objects.get("SR_mcorner.L") is not None and
                  bpy.data.objects.get("SR_mcorner.R") is not None)
        xs = loop[:, 0]
        p_l = loop[int(np.argmax(xs))]        # +X = left commissure
        p_r = loop[int(np.argmin(xs))]        # -X = right commissure
        size = max(float(np.linalg.norm(loop.max(0) - loop.min(0))) * 0.09,
                   0.004)
        mk_l = _mcorner_marker("SR_mcorner.L", (0.10, 0.45, 1.0), size)
        mk_r = _mcorner_marker("SR_mcorner.R", (0.95, 0.15, 0.15), size)
        if not exists:
            mk_l.location = Vector([float(v) for v in p_l])
            mk_r.location = Vector([float(v) for v in p_r])
        try:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        for o in list(context.selected_objects):
            o.select_set(False)
        mk_l.select_set(True)
        mk_r.select_set(True)
        context.view_layer.objects.active = mk_l
        self.report({'INFO'}, "Corners: move BLUE=left / RED=right onto the "
                    "commissures, then Build (%s)"
                    % ("kept your positions" if exists else "placed a guess"))
        return {'FINISHED'}


class SMARTRIG_OT_mouth_sample(bpy.types.Operator):
    bl_idname = "smartrig.mouth_sample_build"
    bl_label = "Build Mouth Rig"
    bl_description = ("Build a cinematic mouth rig from the registered lip loop "
                      "+ corners: master + per-lip masters, arc-even lip ribbon "
                      "with per-point tweaks, open/seal slider (lips meet), "
                      "smile/frown corners, and bounded lip binding.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            info = build_mouth_rig(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        msg = ("Mouth rig: %s, bound %d verts"
               % (", ".join(info["columns"]), info.get("bound", 0)))
        if info.get("bind_error"):
            msg += " (bind skipped: %s)" % info["bind_error"]
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SMARTRIG_OT_mouth_bind(bpy.types.Operator):
    bl_idname = "smartrig.mouth_bind"
    bl_label = "Re-Bind Lips Only"
    bl_description = ("Rebuild + re-run ONLY the bounded lip skinning (weights "
                      "stay inside the mouth region).")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            info = build_mouth_rig(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Lips bound: %d verts" % info.get("bound", 0))
        return {'FINISHED'}


class SMARTRIG_OT_mouth_clear(bpy.types.Operator):
    bl_idname = "smartrig.mouth_clear"
    bl_label = "Clear Mouth Sample"
    bl_description = ("Delete the whole mouth sample (bones + lip weights) and "
                      "hand the lip verts back to the head bone, so you can "
                      "rebuild from a clean slate.")
    bl_options = {'REGISTER', 'UNDO'}
    also_registration: bpy.props.BoolProperty(
        name="Also clear registrations",
        description="Also remove the mouth-loop + corner registrations",
        default=False)

    def execute(self, context):
        try:
            rm = clear_mouth_rig(context, self.also_registration)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Mouth cleared: %d bones, %d weight groups, "
                    "%d verts back to head%s"
                    % (rm["bones"], rm["weight_groups"], rm["restored_verts"],
                       (", %d registrations" % rm["registrations"])
                       if self.also_registration else ""))
        return {'FINISHED'}


def _setup_mouth_corrective(context, which):
    """Pose the mouth at an extreme (seal or open) and make a slider-driven
    corrective shape key the active, editable key. which in {'SEAL','OPEN'}."""
    rig = _target_rig(context)
    if rig is None or rig.type != 'ARMATURE':
        return None
    body = _body(context)
    if body is None or body.type != 'MESH':
        return None
    slb = "CTL-MouthOpen"
    pbb = rig.pose.bones.get(slb)
    if pbb is None:
        return None
    try:
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    L = 0.04
    for con in pbb.constraints:
        if con.name == "SR Mouth Limit" and con.use_max_y:
            L = float(con.max_y) or L
            break
    me = body.data
    if which == 'SEAL':
        sk_name = "SR_mouth_seal"
        expr = "max(0.0, -m)"
        loc_y = -L
    else:
        sk_name = "SR_mouth_open"
        expr = "max(0.0, m)"
        loc_y = L
    if me.shape_keys is None:
        body.shape_key_add(name="Basis", from_mix=False)
    kb = me.shape_keys.key_blocks
    sk = kb.get(sk_name)
    if sk is None:
        sk = body.shape_key_add(name=sk_name, from_mix=False)
    sk.slider_min = 0.0
    sk.slider_max = 1.0
    dp = 'key_blocks["%s"].value' % sk_name
    try:
        ad = me.shape_keys.animation_data
        if ad is not None:
            for d in list(ad.drivers):
                if d.data_path == dp:
                    ad.drivers.remove(d)
    except Exception:
        pass
    try:
        fcu = me.shape_keys.driver_add(dp)
        drv = fcu.driver
        drv.type = 'SCRIPTED'
        drv.expression = expr
        for v in list(drv.variables):
            drv.variables.remove(v)
        var = drv.variables.new()
        var.name = "m"
        var.type = 'SINGLE_PROP'
        tg = var.targets[0]
        tg.id_type = 'OBJECT'
        tg.id = rig
        tg.data_path = 'pose.bones["%s"]["mopen"]' % slb
    except Exception:
        pass
    try:
        pbb.location = (0.0, loc_y, 0.0)
    except Exception:
        pass
    idx = kb.find(sk_name)
    if idx >= 0:
        body.active_shape_key_index = idx
    sk.value = 1.0
    body.show_only_shape_key = False
    body.use_shape_key_edit_mode = True
    for m in body.modifiers:
        if m.type == 'ARMATURE':
            m.show_in_editmode = True
            m.show_on_cage = True
    for o in list(context.selected_objects):
        o.select_set(False)
    body.select_set(True)
    context.view_layer.objects.active = body
    return rig, body, sk_name


class SMARTRIG_OT_mouth_corrective(bpy.types.Operator):
    bl_idname = "smartrig.mouth_corrective"
    bl_label = "Correct Mouth Shape"
    bl_description = ("Pose the mouth at an extreme (sealed or fully open) and "
                      "create a corrective shape key wired to the slider (fades "
                      "in ONLY at that extreme), then drop into EDIT or SCULPT "
                      "mode to perfect the shape. Press 'Finish Correction' "
                      "after. It rides the slider automatically.")
    bl_options = {'REGISTER', 'UNDO'}
    which: bpy.props.EnumProperty(
        items=(('SEAL', "Sealed", "Perfect the pressed-together lips"),
               ('OPEN', "Open", "Perfect the fully-open mouth")),
        default='SEAL')
    mode: bpy.props.EnumProperty(
        items=(('SCULPT', "Sculpt", "Perfect by sculpting"),
               ('EDIT', "Edit", "Perfect by moving vertices")),
        default='SCULPT')

    def execute(self, context):
        r = _setup_mouth_corrective(context, self.which)
        if r is None:
            self.report({'ERROR'}, "Build the mouth rig first (need the "
                        "open/seal slider + lip mesh)")
            return {'CANCELLED'}
        try:
            bpy.ops.object.mode_set(mode=self.mode)
        except Exception as e:
            self.report({'WARNING'}, "Enter %s Mode manually: %s"
                        % (self.mode.title(), e))
        self.report({'INFO'}, "Mouth %s - %s the corrective, then 'Finish "
                    "Correction'." % (self.which.lower(), self.mode.lower()))
        return {'FINISHED'}


class SMARTRIG_OT_mouth_corrective_finish(bpy.types.Operator):
    bl_idname = "smartrig.mouth_corrective_finish"
    bl_label = "Finish Correction (Conform)"
    bl_description = ("Confirm the mouth correction: leave Edit/Sculpt, turn the "
                      "edit-cage off, and return the mouth to rest. The fix stays "
                      "stored in the slider-driven shape key.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        body = _body(context)
        rig = _target_rig(context)
        try:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        if body is not None and body.type == 'MESH':
            for m in body.modifiers:
                if m.type == 'ARMATURE':
                    m.show_on_cage = False
            body.use_shape_key_edit_mode = False
            try:
                body.active_shape_key_index = 0
            except Exception:
                pass
        if rig is not None:
            pb = rig.pose.bones.get("CTL-MouthOpen")
            if pb is not None:
                pb.location = (0.0, 0.0, 0.0)
        self.report({'INFO'}, "Correction stored + driven by the slider.")
        return {'FINISHED'}


_classes = (SMARTRIG_OT_mouth_register, SMARTRIG_OT_mouth_corner_marker,
            SMARTRIG_OT_mouth_sample, SMARTRIG_OT_mouth_bind,
            SMARTRIG_OT_mouth_clear, SMARTRIG_OT_mouth_corrective,
            SMARTRIG_OT_mouth_corrective_finish)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
