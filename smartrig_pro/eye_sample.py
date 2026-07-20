"""Soulify Face (new) - clean, incremental, part-by-part rebuild.

PART 1 = the EYE, register-driven (no auto-detect guessing, no RBF).

Registration (all on the character):
  1. Eye L / Eye R      -> pick the eyeball MESH (Object mode) or its verts on a
                           combined mesh (Edit mode).
  2. Eyelid L / R       -> select the eyelid EDGE LOOP (Edit mode, Alt+click).
  3. Corners L / R      -> select the TWO corner verts (inner + outer canthus);
                           the addon auto-labels inner (nearer the face mid-line)
                           vs outer.

Build Eye Rig produces a professional, animator-ready eye:
  - AIM/look   : MSTR-Eyes + CTL-Eye_target(+.L/.R) + DEF-eye.L/R DAMPED_TRACK.
  - LIDS       : each deform bone is a SPOKE from the eyeball centre out to a
                 point on the lid loop (radiates like the reference image),
                 spread EVENLY by arc-length. Upper and lower counts are set
                 independently in the panel.
  - CORNERS    : DEF-lid_corner_in/out anchors pinned at the canthi (never move).
  - CONTROLS   : per-lid MASTER control (CTL-LidM_upp/low) that opens / closes /
                 over-opens the whole lid, PLUS a per-point TWEAK control
                 (CTL-LidT_*) on every lid bone so the animator can sculpt the
                 lid shape (angry / sad / surprised). Both upper AND lower.
  - BLINK      : one master CTL-Blink slider per eye. Upper bone i is PAIRED with
                 lower bone i and, on blink, both aim at their shared midpoint
                 (MCH-Lid_close) so the lids meet exactly onto each other.
  - BIND       : optional automatic eyelid skinning, BOUNDED to the eye region
                 only (recognised from the registered loop) - never bleeds into
                 the brows or cheeks.
Colour-coded L = blue / R = red / centre = yellow.
"""
import bpy
import numpy as np
from mathutils import Vector
from . import utils

EYE_COLL = "Face - Eyes"
N_LID = 5                      # fallback deform bones per lid if no prop set
_C_BLUE = (0.10, 0.45, 1.0)    # lid master arcs
_C_ORANGE = (1.0, 0.50, 0.04)  # point tweak circles
_C_YELLOW = (1.0, 0.85, 0.10)  # blink / eye master


# --------------------------------------------------------------------- helpers
def _sphere_fit(pts):
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + (c ** 2).sum(), 1e-12)))
    return c, r


def _body(context):
    props = context.scene.smartrig
    return getattr(props, "target_mesh", None) or context.active_object


def _target_rig(context):
    try:
        from . import metarig as _mr
        r = _mr._generated_rig()
        if r is not None:
            return r
    except Exception:
        pass
    ob = context.active_object
    if ob is not None and ob.type == 'ARMATURE':
        return ob
    arms = [o for o in context.scene.objects if o.type == 'ARMATURE']
    return arms[0] if arms else None


def _eye_obj(context, side):
    nm = context.scene.get("sr_eye" + side)
    return bpy.data.objects.get(nm) if nm else None


def _eye_center(context, side):
    """(centre, radius) of the registered eye."""
    ob = _eye_obj(context, side)
    if ob is None:
        return None
    vg = ob.vertex_groups.get("SR_eye" + side)
    if vg is not None:
        gi = vg.index
        pts = [list(ob.matrix_world @ v.co) for v in ob.data.vertices
               if any(g.group == gi for g in v.groups)]
        if len(pts) >= 6:
            return _sphere_fit(np.array(pts))
    pts = np.array([list(ob.matrix_world @ v.co) for v in ob.data.vertices])
    return _sphere_fit(pts)                       # (centre, radius)


def _lid_loop(body, side):
    """World coords of the registered eyelid loop (vertex group SR_eyelid_*)."""
    vg = body.vertex_groups.get("SR_eyelid" + side)
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


def _snap_vert(body, p):
    """World-position of the body vertex nearest to point p (keeps a corner
    exactly on the mesh even if the marker floats slightly off the surface)."""
    inv = body.matrix_world.inverted()
    lp = inv @ Vector((float(p[0]), float(p[1]), float(p[2])))
    best, bd = None, 1e18
    for v in body.data.vertices:
        d = (v.co - lp).length_squared
        if d < bd:
            bd, best = d, v
    if best is None:
        return np.array([float(p[0]), float(p[1]), float(p[2])], float)
    return np.array(list(body.matrix_world @ best.co), float)


def _corner_pts(body, side):
    """(inner, outer) corner world coords. Priority order:
      1. the movable marker objects SR_corner_in/out<side> (snapped to the
         nearest mesh vertex),
      2. the legacy SR_corner<side> vertex group.
    inner = nearer the face mid-line (smaller |x|)."""
    ins = bpy.data.objects.get("SR_corner_in" + side)
    outs = bpy.data.objects.get("SR_corner_out" + side)
    if ins is not None and outs is not None and body is not None:
        pa = ins.matrix_world.translation
        pb = outs.matrix_world.translation
        # only trust markers that are actually PLACED (not left at the origin,
        # not coincident) - else fall back to the loop extremes below
        if pa.length > 1e-4 and pb.length > 1e-4 and (pa - pb).length > 1e-4:
            a = _snap_vert(body, pa)
            b = _snap_vert(body, pb)
            if abs(a[0]) > abs(b[0]):
                a, b = b, a
            return a, b
    vg = body.vertex_groups.get("SR_corner" + side) if body else None
    if vg is None:
        return None
    gi = vg.index
    pts = []
    for v in body.data.vertices:
        for g in v.groups:
            if g.group == gi:
                pts.append(np.array(list(body.matrix_world @ v.co), float))
                break
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: abs(p[0]))       # inner canthus = nearest mid-line
    return pts[0], pts[-1]


# ---------------------------------------------------------------- eye markers
EYE_CORNER_COLL = "SR Eye Corners"


def _eye_corner_coll():
    c = bpy.data.collections.get(EYE_CORNER_COLL)
    if c is None:
        c = bpy.data.collections.new(EYE_CORNER_COLL)
        bpy.context.scene.collection.children.link(c)
    return c


def _corner_mat(name, rgb):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name)
    m.diffuse_color = (rgb[0], rgb[1], rgb[2], 1.0)   # solid-mode 'Material' color
    try:
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
            for k in ("Emission Color", "Emission"):
                if k in bsdf.inputs:
                    bsdf.inputs[k].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
                    break
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 1.0
    except Exception:
        pass
    return m


def _corner_marker(name, rgb, size):
    """A small coloured diamond mesh the animator can grab and drop on a corner.
    Drawn in front, name shown, so it's easy to see and place."""
    ob = bpy.data.objects.get(name)
    if ob is None or ob.type != 'MESH':
        me = bpy.data.meshes.new(name)
        V = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        F = [(0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
             (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)]
        me.from_pydata(V, [], F)
        me.update()
        ob = bpy.data.objects.new(name, me)
        _eye_corner_coll().objects.link(ob)
        ob["sr_eye_corner"] = True
    ob.data.materials.clear()
    ob.data.materials.append(_corner_mat("SRmk_" + name, rgb))
    ob.color = (rgb[0], rgb[1], rgb[2], 1.0)
    ob.scale = (size, size, size)
    ob.show_in_front = True
    ob.show_name = True
    ob.hide_viewport = False        # re-registering shows them again
    try:
        ob.hide_set(False)
    except Exception:
        pass
    return ob


def _registration_state(context):
    body = _body(context)
    st = {}
    for s in (".L", ".R"):
        st["eye" + s] = _eye_obj(context, s) is not None
        st["lid" + s] = (body is not None and
                         body.vertex_groups.get("SR_eyelid" + s) is not None)
        has_mk = (bpy.data.objects.get("SR_corner_in" + s) is not None and
                  bpy.data.objects.get("SR_corner_out" + s) is not None)
        st["corner" + s] = has_mk or (body is not None and
                            body.vertex_groups.get("SR_corner" + s) is not None)
    return st


# --------------------------------------------------------------------- widgets
def _wgt(name, kind):
    ob = bpy.data.objects.get(name)
    if ob is not None:
        return ob
    me = bpy.data.meshes.new(name)
    if kind == "ring":
        # ring in the X-Z plane so its normal is +Y = the bone's aim axis;
        # since every eye control points forward, the circles face the camera.
        seg = 24
        V = [(np.cos(2 * np.pi * i / seg), 0.0, np.sin(2 * np.pi * i / seg))
             for i in range(seg)]
        E = [(i, (i + 1) % seg) for i in range(seg)]
    elif kind == "cross":
        s = 1.0
        V = [(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0),
             (-s, 0, 0), (s, 0, 0), (0, -s, 0), (0, s, 0)]
        E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (6, 7)]
    elif kind == "diamond":
        V = [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0)]
        E = [(0, 1), (1, 2), (2, 3), (3, 0)]
    elif kind == "arrowud":
        # modern blink control: a smooth rounded double-chevron (upper dome +
        # lower bowl) that forms an open-eye lens - signals the vertical drag
        # that opens/closes the lid, far cleaner than a sharp arrow. Drawn in
        # X-Y (y = up), matching the blink bone's aim so it reads upright.
        def _bz(p0, p1, p2, m):
            return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
                     (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1],
                     0.0) for t in np.linspace(0.0, 1.0, m + 1)]
        m = 8
        V, E = [], []
        for cy, s in ((0.42, 1.0), (-0.42, -1.0)):
            base = len(V)
            arm = (_bz((-0.5, cy - s * 0.22), (-0.20, cy + s * 0.22),
                       (0.0, cy + s * 0.30), m)
                   + _bz((0.0, cy + s * 0.30), (0.20, cy + s * 0.22),
                         (0.5, cy - s * 0.22), m)[1:])
            V.extend(arm)
            E.extend((base + i, base + i + 1) for i in range(len(arm) - 1))
    elif kind == "peanut":
        # ARP-style eyes target: two round lobes joined by a concave waist, in
        # the X-Z plane (normal +Y = bone aim) so it faces the camera. Lobe
        # centres at x = +/-1, so a uniform scale of 0.5*ipd lands them on the
        # two per-eye targets.
        C, R, rn = 1.0, 0.95, 0.55
        h = np.sqrt((R + rn) ** 2 - C ** 2)

        def _t(lcx, fcz):
            d = np.array([0.0 - lcx, fcz], float)
            d /= np.linalg.norm(d)
            return np.array([lcx, 0.0]) + R * d

        tpR, tpL = _t(C, h), _t(-C, h)
        bpR, bpL = _t(C, -h), _t(-C, -h)

        def _a(p, cx, cz=0.0):
            return np.arctan2(p[1] - cz, p[0] - cx)

        def _arc(cx, cz, rad, a0, a1, m):
            return [(cx + rad * np.cos(a0 + (a1 - a0) * k / m), 0.0,
                     cz + rad * np.sin(a0 + (a1 - a0) * k / m))
                    for k in range(m + 1)]
        m = 16
        V = []
        aR0, aR1 = _a(tpR, C), _a(bpR, C)
        if aR1 > aR0:
            aR1 -= 2 * np.pi
        V += _arc(C, 0, R, aR0, aR1, m)                 # right lobe outer arc
        aB0, aB1 = _a(bpR, 0, -h), _a(bpL, 0, -h)
        if aB1 < aB0:
            aB1 += 2 * np.pi
        V += _arc(0, -h, rn, aB0, aB1, m // 2)          # bottom concave waist
        aL0, aL1 = _a(bpL, -C), _a(tpL, -C)
        if aL1 > aL0:
            aL1 -= 2 * np.pi
        V += _arc(-C, 0, R, aL0, aL1, m)                # left lobe outer arc
        aT0, aT1 = _a(tpL, 0, h), _a(tpR, 0, h)
        if aT1 < aT0:
            aT1 += 2 * np.pi
        V += _arc(0, h, rn, aT0, aT1, m // 2)           # top concave waist
        E = [(i, (i + 1) % len(V)) for i in range(len(V))]
    else:  # "arc" - a shallow smile-shaped arc for lid master controls
        seg = 12
        V = [(np.cos(np.pi * (0.15 + 0.7 * i / seg)),
              np.sin(np.pi * (0.15 + 0.7 * i / seg)) * 0.5, 0.0)
             for i in range(seg + 1)]
        E = [(i, i + 1) for i in range(seg)]
    me.from_pydata(V, E, [])
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


def _pal(rig, bone, col):
    """col = a THEME string, or an (r,g,b) tuple for a custom colour."""
    try:
        if isinstance(col, str):
            rig.pose.bones[bone].color.palette = col
            rig.data.bones[bone].color.palette = col
            return
        for cc in (rig.pose.bones[bone].color, rig.data.bones[bone].color):
            cc.palette = 'CUSTOM'
            cc.custom.normal = (col[0], col[1], col[2])
            cc.custom.select = (min(col[0] + 0.25, 1.0), min(col[1] + 0.25, 1.0),
                                min(col[2] + 0.25, 1.0))
            cc.custom.active = (1.0, 1.0, 1.0)
    except Exception:
        pass


def _hide_corner_markers(hidden):
    """Corner markers show only while registering; hidden after Build."""
    for s in (".L", ".R"):
        for nm in ("SR_corner_in" + s, "SR_corner_out" + s):
            o = bpy.data.objects.get(nm)
            if o is not None:
                try:
                    o.hide_viewport = hidden
                    o.hide_set(hidden)
                except Exception:
                    pass


def _arc_even(pts, k):
    """k points spread EVENLY by arc-length along the ordered polyline pts."""
    pts = [np.asarray(p, float) for p in pts]
    if len(pts) < 2:
        return [pts[0] for _ in range(k)] if pts else []
    seg = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
    total = sum(seg) or 1e-9
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)

    def at(frac):
        t = max(0.0, min(1.0, frac)) * total
        for i in range(len(seg)):
            if t <= cum[i + 1] or i == len(seg) - 1:
                d = seg[i] if seg[i] > 1e-9 else 1e-9
                return pts[i] + (pts[i + 1] - pts[i]) * min(max((t - cum[i]) / d, 0.0), 1.0)
        return pts[-1]
    return [at((i + 1) / (k + 1.0)) for i in range(k)]


def _interp_pt(uq, us, pts):
    """Componentwise linear interp of 3-D points pts (sampled at increasing us)
    at query u = uq.  us MUST be ascending."""
    x = np.interp(uq, us, pts[:, 0])
    y = np.interp(uq, us, pts[:, 1])
    z = np.interp(uq, us, pts[:, 2])
    return np.array([x, y, z], float)


def build_eye_rig(context):
    body = _body(context)
    if body is None or body.type != 'MESH':
        raise RuntimeError("Pick the character mesh first (Target Mesh).")
    rig = _target_rig(context)
    if rig is None:
        raise RuntimeError("No armature found - generate the body rig first.")

    props = context.scene.smartrig
    n_upp = max(1, int(getattr(props, "eye_lid_upper_count", N_LID)))
    n_low = max(1, int(getattr(props, "eye_lid_lower_count", N_LID)))
    autobind = bool(getattr(props, "eye_autobind", True))
    band_fac = float(getattr(props, "eye_bind_band", 0.6))

    eyes = {}
    for s in (".L", ".R"):
        cr = _eye_center(context, s)
        if cr is not None:
            eyes[s] = cr
    if not eyes:
        raise RuntimeError("Register at least one eye first "
                           "(select the eyeball mesh -> Register Eye L/R).")

    if ".L" in eyes and ".R" in eyes:
        ipd = float(np.linalg.norm(eyes[".L"][0] - eyes[".R"][0]))
    else:
        ipd = 4.0 * next(iter(eyes.values()))[1]
    ipd = max(ipd, 1e-3)
    mid = np.mean([eyes[s][0] for s in eyes], axis=0)
    tgt_dist = 6.0 * max(next(iter(eyes.values()))[1], 0.02)

    head_parent = None
    for n in ("DEF-head", "head", "DEF-spine.006", "DEF-neck"):
        if n in rig.data.bones:
            head_parent = n
            break

    info = {"eyes": list(eyes.keys()), "ipd": round(ipd, 4), "lids": [],
            "upper": n_upp, "lower": n_low, "bound": 0}

    prev = context.view_layer.objects.active
    context.view_layer.objects.active = rig
    rig.hide_set(False)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones

    # purge previous eye-sample bones (all generations of prefixes)
    def _drop(pred):
        for b in list(eb):
            if pred(b.name):
                eb.remove(b)
    _drop(lambda n: n in ("MSTR-Eyes", "CTL-Eye_target",
                          "CTL-Eye_target.L", "CTL-Eye_target.R",
                          "DEF-eye.L", "DEF-eye.R"))
    _drop(lambda n: n.startswith(("CTL-Lid", "MCH-Lid", "DEF-lid", "CTL-Blink",
                                  "MCH-Rib", "MCH-rib", "MCH-tgt", "MCH-Tgt")))

    def nb(name, head, tail, deform, parent=None):
        b = eb.new(name)
        b.head = Vector([float(v) for v in head])
        b.tail = Vector([float(v) for v in tail])
        b.use_deform = deform
        if parent and parent in eb:
            b.parent = eb[parent]
            b.use_connect = False
        return b

    made_ring, made_cross, made_arc, made_diamond = [], [], [], []
    made_arrow = []                 # CTL-Blink slider = an up/down double arrow
    made_peanut = []                # ARP-style eyes master target
    bind_data = []                  # per side: weighting geometry
    master_wdata = []               # (master bone, world polyline tracing the lid)
    col_wiring = []                 # per column: dict of bone names for pose pass
    corner_follow = []              # (corner DEF bone, its tweak circle)
    ribbon_specs = []               # (obj_name, part, side, [(col_index, mch_tgt)])
    gaze_masters = []               # (master bone, side, fac) for lid-follow

    # MSTR-Eyes = the aim pivot at the eyes (small ring, kept subtle).
    nb("MSTR-Eyes", mid, mid + np.array([0, 0, 0.20 * ipd]), False, head_parent)
    made_ring.append(("MSTR-Eyes", 0.13 * ipd, _C_YELLOW))
    # CTL-Eye_target = the master look-at, OUT IN FRONT. Point it -Y (toward the
    # animator) so its peanut widget faces the camera and reads cleanly.
    tc = np.array([mid[0], mid[1] - tgt_dist, mid[2]])
    nb("CTL-Eye_target", tc, tc + np.array([0, -0.30 * ipd, 0]), False, head_parent)
    made_peanut.append(("CTL-Eye_target", 0.5 * ipd, _C_YELLOW))

    BMIX = 0.62         # closed seam sits toward the LOWER lid (upper travels more)
    FWD_PUSH = 0.40     # push the closed seam forward so lids meet in FRONT of eye
    WIDE = 0.40         # wide-open lift as a fraction of eye radius
    OVERLAP = 0.14      # upper aims just BELOW the seam and lower just ABOVE it, so
                        # the two skin margins cross and the lid SEALS (no slit)

    for s in eyes:
        c, r = eyes[s]
        pal = 'THEME04' if s == ".L" else 'THEME01'
        # AIM: deform bone forward, per-eye target
        nb("DEF-eye" + s, c, c + np.array([0, -1.3 * r, 0]), True, "MSTR-Eyes")
        tcs = np.array([c[0], c[1] - tgt_dist, c[2]])
        nb("CTL-Eye_target" + s, tcs, tcs + np.array([0, -0.30 * ipd, 0]),
           False, "CTL-Eye_target")
        made_ring.append(("DEF-eye" + s, 1.1 * r, pal))
        # per-eye look-at = a clean circle sitting INSIDE its peanut lobe
        made_ring.append(("CTL-Eye_target" + s, 1.0 * r, pal))

        # EYELIDS from the registered loop
        loop = _lid_loop(body, s)
        if loop is None or len(loop) < 6:
            continue
        # frame: forward = C -> loop centroid, up = world Z, right = up x fwd
        fwd = loop.mean(axis=0) - c
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        up0 = np.array([0.0, 0.0, 1.0])
        right = np.cross(up0, fwd)
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(fwd, right)
        rel = loop - c
        u = rel @ right                      # horizontal (inner<->outer)
        vv = rel @ up                        # vertical (upper>0 / lower<0)
        ff = rel @ fwd                       # depth toward the lid opening
        vmax = float(np.max(np.abs(vv))) or (0.5 * r)

        # corners: registered markers, else the loop's horizontal extremes
        cor = _corner_pts(body, s)
        loop_lo, loop_hi = float(u.min()), float(u.max())
        loop_span = loop_hi - loop_lo
        use_loop = cor is None
        if cor is not None:
            inner_p, outer_p = cor
            cu = sorted([float((inner_p - c) @ right), float((outer_p - c) @ right)])
            if (cu[1] - cu[0]) < 0.6 * loop_span:
                use_loop = True
        if use_loop:
            inner_p = loop[int(np.argmin(u))]
            outer_p = loop[int(np.argmax(u))]
        if abs(inner_p[0]) > abs(outer_p[0]):
            inner_p, outer_p = outer_p, inner_p
        u_in = float((inner_p - c) @ right)
        u_out = float((outer_p - c) @ right)
        lo_u, hi_u = min(u_in, u_out), max(u_in, u_out)
        span = max(hi_u - lo_u, 1e-6)

        # ---- ordered margin polylines (ascending u) for upper and lower lids,
        #      so we can sample points / seam heights at ANY horizontal position.
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

        def _seam_pt(uq):
            """Point on the SMOOTH closed-lid seam at horizontal position uq.
            Upper and lower bones at the same uq land on this exact point, so the
            lids meet cleanly; the curve is smooth in uq so the closed lid never
            kinks."""
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
            return c + right * uq + up * vseam + fwd * (ffseam + FWD_PUSH * r)

        # corner ANCHOR bones (never blink) - spoke deform + a circle tweak at
        # the canthus so the animator can nudge the corner.
        cbone = {}
        for cname, cp in (("in", inner_p), ("out", outer_p)):
            cp = np.asarray(cp)
            bn = "DEF-lid_corner_%s%s" % (cname, s)
            nb(bn, c, cp, True, "MSTR-Eyes")
            tw = "CTL-LidT_corner_%s%s" % (cname, s)
            nb(tw, cp, cp + np.array([0.0, -1.0, 0.0]) * 0.35 * r, False, "MSTR-Eyes")
            made_ring.append((tw, 0.065 * r, _C_ORANGE))
            corner_follow.append((bn, tw))
            cbone[cname] = (float((cp - c) @ right), bn, np.asarray(cp))
        info["lids"].append("corners%s" % s)

        # per-lid MASTER controls (open / close / over-open the whole lid).
        masters = {}
        for part, sgn in (("upp", 1.0), ("low", -1.0)):
            mg = margins.get(part)
            mmid = _interp_pt(0.5 * (lo_u + hi_u), mg["us"], mg["pts"]) if mg \
                else (c + up * sgn * r)
            mn = "CTL-LidM_%s%s" % (part, s)
            nb(mn, c, c + (mmid - c) * 1.1, False, "MSTR-Eyes")
            masters[part] = mn
            gaze_masters.append((mn, s, 1.0 if part == "upp" else 0.5))

        # ---- RIBBON columns: for each lid, sample K columns EVEN in horizontal
        #      position between the corners.  Every column carries:
        #        CTL-LidT  (orange tweak the animator grabs)
        #        MCH-tgtClose / MCH-tgtWide (static anchors, children of the tweak)
        #        MCH-tgt   (the live target: Copy-Loc blends open<->close<->wide)
        #        DEF-lid   (deform bone; Damp-Tracks MCH-tgt = slides over cornea)
        counts = {"upp": n_upp, "low": n_low}
        bones_by = {}
        for part, sgn in (("upp", 1.0), ("low", -1.0)):
            mg = margins.get(part)
            if mg is None:
                bones_by[part] = []
                continue
            K = counts[part]
            # spread columns from NEAR the inner canthus to NEAR the outer one
            # (not bunched in the middle) so the lids close right up to both
            # corners with no gap between the last column and the static canthus.
            if K == 1:
                fracs = [0.5]
            else:
                lo_f, hi_f = 0.02, 0.98
                fracs = [lo_f + (hi_f - lo_f) * j / (K - 1.0) for j in range(K)]
            us_col = [lo_u + f * span for f in fracs]
            ribbon_cols = []
            blist = []
            open_pts = []
            for i, (frac, uq) in enumerate(zip(fracs, us_col), 1):
                p_open = _interp_pt(uq, mg["us"], mg["pts"])
                # upper (sgn=+1) aims BELOW the seam, lower (sgn=-1) ABOVE it, so
                # the skin margins overlap and the closed lid seals with no slit.
                # The overlap eases only slightly toward the canthi (stays strong
                # there, since the corners need sealing too) - it never drops to
                # zero, so both corners close as firmly as the centre.
                ov = OVERLAP * (0.75 + 0.25 * float(np.sin(np.pi * frac)))
                p_close = _seam_pt(uq) - up * (sgn * ov * r)
                p_wide = p_open + up * (sgn * WIDE * r)
                open_pts.append(p_open)
                # static close / wide anchors + the blink-driven moving base,
                # ALL children of the lid master.
                cl = "MCH-tgtClose_%s%d%s" % (part, i, s)
                wd = "MCH-tgtWide_%s%d%s" % (part, i, s)
                nb(cl, p_close, p_close + fwd * 0.15 * r, False, masters[part])
                nb(wd, p_wide, p_wide + fwd * 0.15 * r, False, masters[part])
                # live moving base (open; Copy-Loc blends toward close / wide)
                tg = "MCH-tgt_%s%d%s" % (part, i, s)
                nb(tg, p_open, p_open + fwd * 0.15 * r, False, masters[part])
                # orange tweak RIDES the moving base (child of MCH-tgt) so the
                # circle FOLLOWS the lid as it blinks; the animator grabs it to
                # sculpt and that offset is carried through open AND closed.
                tw = "CTL-LidT_%s%d%s" % (part, i, s)
                nb(tw, p_open, p_open + np.array([0.0, -1.0, 0.0]) * 0.35 * r,
                   False, tg)
                made_ring.append((tw, 0.065 * r, _C_ORANGE))
                # deform bone Damp-Tracks the TWEAK (= blink motion + animator nudge)
                dn = "DEF-lid_%s%d%s" % (part, i, s)
                nb(dn, c, p_open, True, masters[part])
                col_wiring.append(dict(tgt=tg, close=cl, wide=wd, defb=dn,
                                       track=tw, blink="CTL-Blink%s" % s))
                ribbon_cols.append((i, tg))
                blist.append((float((p_open - c) @ right), dn, np.asarray(p_open)))
            bones_by[part] = blist
            info["lids"].append(masters[part] + " x%d" % len(blist))

            # master lid-line widget: a polyline tracing canthus -> lid -> canthus.
            # Order the two corners by horizontal position u (NOT inner/outer),
            # else on the RIGHT eye - where the inner canthus has the LARGER u -
            # the line jumps corner->far-end->back and zig-zags. open_pts are
            # already u-ascending, so prepend the low-u corner, append the high-u.
            u_ip = float((np.asarray(inner_p) - c) @ right)
            u_op = float((np.asarray(outer_p) - c) @ right)
            lo_cnr, hi_cnr = ((inner_p, outer_p) if u_ip <= u_op
                              else (outer_p, inner_p))
            poly = ([list(map(float, lo_cnr))] +
                    [list(map(float, p)) for p in open_pts] +
                    [list(map(float, hi_cnr))])
            master_wdata.append((masters[part], poly))
            # ribbon mesh spec (built in object mode, after the pose pass)
            ribbon_specs.append(("HLP-SR-rib_%s%s" % (part, s), part, s,
                                 ribbon_cols, list(map(list, open_pts)),
                                 list(right), list(up), sgn, r))

        # master BLINK control - a draggable vertical slider beside the eye
        blb = "CTL-Blink%s" % s
        side_sign = 1.0 if c[0] >= 0 else -1.0
        bl_L = 1.6 * r
        bl_head = np.asarray(outer_p) + np.array([side_sign * 0.45 * r,
                                                  -0.3 * r, 0.0])
        nb(blb, bl_head, bl_head + np.array([0.0, 0.0, 1.0]) * 0.6 * r,
           False, "MSTR-Eyes")
        made_arrow.append((blb, 0.72 * r, (0.15, 0.85, 1.0)))

        # weighting geometry. The lid COLUMNS now reach right to the canthi, so
        # the corner skin is driven by the (closing) end columns and seals - the
        # STATIC corner bones are kept ONLY as a fallback if a lid has no columns
        # (they would otherwise pin the very corner tip open = a residual slit).
        upp_ctrl = sorted(bones_by.get("upp", []) or [cbone["in"], cbone["out"]],
                          key=lambda t: t[0])
        low_ctrl = sorted(bones_by.get("low", []) or [cbone["in"], cbone["out"]],
                          key=lambda t: t[0])
        bind_data.append(dict(
            c=c, r=r, right=right, up=up, fwd=fwd, loop=loop, vmax=vmax,
            u_in=lo_u, u_out=hi_u, span=span, band=max(band_fac * r, 1e-4),
            upp=upp_ctrl, low=low_ctrl))

    # organise into CTRL / DEF / MCH bone collections so the VISIBLE rig is
    # clean: only the controls show; deform + mechanism bones are hidden.
    try:
        def _coll(name, visible):
            data = rig.data
            c = None
            for bc in data.collections:
                if bc.name == name:
                    c = bc
                    break
            if c is None:
                c = data.collections.new(name)
            try:
                c.is_visible = visible
            except Exception:
                pass
            return c
        ctrl_c = _coll(EYE_COLL, True)
        def_c = _coll(EYE_COLL + " (Deform)", False)
        mch_c = _coll(EYE_COLL + " (Mech)", False)
        for b in eb:
            nm = b.name
            if nm.startswith(("MCH-Lid", "MCH-tgt", "MCH-Tgt", "MCH-Rib",
                              "MCH-rib")):
                mch_c.assign(b)
            elif nm.startswith("DEF-"):
                def_c.assign(b)
            elif nm.startswith(("MSTR-Eyes", "CTL-")):
                ctrl_c.assign(b)
    except Exception:
        pass

    # STRAIGHT gizmos: align every control bone's roll so its local Z is world up
    for b in eb:
        if b.name.startswith(("MSTR-Eyes", "CTL-")):
            try:
                b.align_roll(Vector((0.0, 0.0, 1.0)))
            except Exception:
                pass

    bpy.ops.object.mode_set(mode='POSE')

    # eye aim constraints
    for s in eyes:
        pb = rig.pose.bones.get("DEF-eye" + s)
        if pb is None:
            continue
        for con in list(pb.constraints):
            pb.constraints.remove(con)
        con = pb.constraints.new('DAMPED_TRACK')
        con.name = "SR Eye Aim"
        con.target = rig
        con.subtarget = "CTL-Eye_target" + s
        con.track_axis = 'TRACK_Y'

    # EYELIDS FOLLOW THE GAZE: the lid masters copy a fraction of the eyeball's
    # rotation (look up -> lids up).  Strength = 'lid_follow' on CTL-Eye_target.
    et = rig.pose.bones.get("CTL-Eye_target")
    if et is not None:
        if "lid_follow" not in et:
            et["lid_follow"] = 0.30
        try:
            uid = et.id_properties_ui("lid_follow")
            uid.update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0,
                       description="How much the eyelids follow the eye gaze "
                                   "(0 = off, 1 = full)")
        except Exception:
            pass
    for mn, s, fac in gaze_masters:
        if ("DEF-eye" + s) not in rig.data.bones:
            continue
        pb = rig.pose.bones.get(mn)
        if pb is None:
            continue
        for con in list(pb.constraints):
            if con.name == "SR Lid Follow":
                pb.constraints.remove(con)
        con = pb.constraints.new('COPY_ROTATION')
        con.name = "SR Lid Follow"
        con.target = rig
        con.subtarget = "DEF-eye" + s
        con.target_space = 'LOCAL'
        con.owner_space = 'LOCAL'
        con.mix_mode = 'ADD'
        con.influence = 0.0
        try:
            fcu = con.driver_add("influence")
            drv = fcu.driver
            drv.type = 'SCRIPTED'
            drv.expression = "f * %s" % fac
            for v in list(drv.variables):
                drv.variables.remove(v)
            var = drv.variables.new()
            var.name = "f"
            var.type = 'SINGLE_PROP'
            tg = var.targets[0]
            tg.id = rig
            tg.data_path = 'pose.bones["CTL-Eye_target"]["lid_follow"]'
        except Exception:
            pass

    # corner deform bones follow their tweak circle (animator can nudge canthus)
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

    # ---- the ribbon mechanism, per column:
    #   MCH-tgt  = open  +  (close - open)*max(0,blink)  +  (wide - open)*max(0,-blink)
    #   DEF-lid  Damp-Tracks MCH-tgt  (keeps its length -> rotates around the eye
    #            centre, so the lid SLIDES over the cornea instead of poking it).
    def _blink_infl(con, blb, expr):
        con.influence = 0.0
        try:
            fcu = con.driver_add("influence")
            drv = fcu.driver
            drv.type = 'SCRIPTED'
            drv.expression = expr
            for v in list(drv.variables):
                drv.variables.remove(v)
            var = drv.variables.new()
            var.name = "b"
            var.type = 'SINGLE_PROP'
            tg = var.targets[0]
            tg.id = rig
            tg.data_path = 'pose.bones["%s"]["blink"]' % blb
        except Exception:
            pass

    for w in col_wiring:
        pb = rig.pose.bones.get(w["tgt"])
        if pb is not None:
            for con in list(pb.constraints):
                pb.constraints.remove(con)
            cc = pb.constraints.new('COPY_LOCATION')
            cc.name = "SR Close"
            cc.target = rig
            cc.subtarget = w["close"]
            _blink_infl(cc, w["blink"], "max(0.0, b)")
            cw = pb.constraints.new('COPY_LOCATION')
            cw.name = "SR Wide"
            cw.target = rig
            cw.subtarget = w["wide"]
            _blink_infl(cw, w["blink"], "max(0.0, -b)")
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

    # blink slider handle: drag DOWN to close, UP to open wide
    for s in eyes:
        blb = "CTL-Blink" + s
        pbb = rig.pose.bones.get(blb)
        if pbb is None:
            continue
        pbb["blink"] = 0.0
        try:
            ui = pbb.id_properties_ui("blink")
            ui.update(min=-1.0, max=1.0, soft_min=-1.0, soft_max=1.0,
                      description="Eye master: -1 = wide open, 0 = rest, "
                                  "1 = closed (lids meet)")
        except Exception:
            pass
        pbb.lock_location = (True, False, True)
        pbb.lock_rotation = (True, True, True)
        pbb.lock_rotation_w = True
        pbb.lock_scale = (True, True, True)
        L = 1.6 * max(eyes[s][1], 0.02)
        for con in list(pbb.constraints):
            if con.name == "SR Blink Limit":
                pbb.constraints.remove(con)
        lim = pbb.constraints.new('LIMIT_LOCATION')
        lim.name = "SR Blink Limit"
        lim.owner_space = 'LOCAL'
        lim.use_min_y = lim.use_max_y = True
        lim.min_y = -L
        lim.max_y = L
        lim.use_transform_limit = True
        try:
            fcu = pbb.driver_add('["blink"]')
            drv = fcu.driver
            drv.type = 'SCRIPTED'
            drv.expression = "max(-1.0, min(1.0, -ly / %.6f))" % L
            for v in list(drv.variables):
                drv.variables.remove(v)
            var = drv.variables.new()
            var.name = "ly"
            var.type = 'TRANSFORMS'
            t = var.targets[0]
            t.id = rig
            t.bone_target = blb
            t.transform_type = 'LOC_Y'
            t.transform_space = 'LOCAL_SPACE'
        except Exception:
            pass

    # widgets (forward-facing so circles read cleanly from the front)
    ring = _wgt("WGT-SR-eye-ringF", "ring")
    cross = _wgt("WGT-SR-eye-crossF", "cross")
    diamond = _wgt("WGT-SR-eye-diamond", "diamond")
    arrowud = _wgt("WGT-SR-eye-arrowud", "arrowud")
    peanut = _wgt("WGT-SR-eye-peanut", "peanut")

    def _apply(lst, shape):
        for name, size, pal in lst:
            pb = rig.pose.bones.get(name)
            if pb:
                pb.custom_shape = shape
                pb.custom_shape_scale_xyz = (size, size, size)
                pb.use_custom_shape_bone_size = False
                if pal:
                    _pal(rig, name, pal)
    _apply(made_ring, ring)
    _apply(made_cross, cross)
    _apply(made_diamond, diamond)
    _apply(made_arrow, arrowud)
    _apply(made_peanut, peanut)

    # MASTER lid-line widgets: a blue polyline tracing the real eyelid.
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
        wname = "WGT-SR-lidline_%s" % bone_name
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
        _pal(rig, bone_name, _C_BLUE)

    bpy.ops.object.mode_set(mode='OBJECT')

    # build the guide RIBBON meshes (a thin strip tracing each lid, bound to the
    # column target bones so it deforms with the blink = a visible, editable
    # smooth lid guide).  Purely cosmetic / animator aid - never rendered.
    try:
        _build_ribbons(context, rig, ribbon_specs)
    except Exception as e:
        info["ribbon_error"] = str(e)

    # bounded eyelid binding
    if autobind and bind_data:
        try:
            info["bound"] = _bind_eyelids(context, rig, body, bind_data)
        except Exception as e:
            info["bind_error"] = str(e)

    _hide_corner_markers(True)

    if prev is not None:
        context.view_layer.objects.active = prev
    return info


RIBBON_COLL = "SR Eye Ribbons"


def _build_ribbons(context, rig, specs):
    """Build a thin guide-ribbon mesh per lid, bound to that lid's column target
    bones so it deforms smoothly with the rig (a visible, paintable lid guide)."""
    coll = bpy.data.collections.get(RIBBON_COLL)
    if coll is None:
        coll = bpy.data.collections.new(RIBBON_COLL)
        context.scene.collection.children.link(coll)
    coll.hide_render = True
    for name, part, side, cols, open_pts, right, up, sgn, r in specs:
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
        w = 0.55 * r * float(sgn)      # outer row offset (up for upper, down low)
        V, F = [], []
        for k, p in enumerate(pts):
            V.append(p)                 # inner row (on the lid margin)
            V.append(p + upv * w)       # outer row (into the lid skin)
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
        # weight each column (its two verts) 100% to that column's target bone
        for idx, (col_i, tgt) in enumerate(cols):
            vg = ob.vertex_groups.new(name=tgt)
            vg.add([2 * idx, 2 * idx + 1], 1.0, 'REPLACE')
        md = ob.modifiers.new(name="SR Ribbon", type='ARMATURE')
        md.object = rig
        ob.parent = rig
        ob.display_type = 'WIRE'
        ob.hide_render = True


# ------------------------------------------------------------------- binding
def _blend_u(ctrl, uv):
    """ctrl = sorted [(u, bone, point), ...]; return up to 2 (bone, weight)
    linearly interpolating by horizontal position uv (partition of unity)."""
    if not ctrl:
        return []
    if uv <= ctrl[0][0]:
        return [(ctrl[0][1], 1.0)]
    if uv >= ctrl[-1][0]:
        return [(ctrl[-1][1], 1.0)]
    for i in range(len(ctrl) - 1):
        u0, b0 = ctrl[i][0], ctrl[i][1]
        u1, b1 = ctrl[i + 1][0], ctrl[i + 1][1]
        if u0 <= uv <= u1:
            d = (u1 - u0) or 1e-9
            t = (uv - u0) / d
            if b0 == b1:
                return [(b0, 1.0)]
            return [(b0, 1.0 - t), (b1, t)]
    return [(ctrl[-1][1], 1.0)]


def _bind_eyelids(context, rig, body, bind_data):
    """Skin ONLY the eyelid-region verts to the lid/corner bones, bounded to the
    eye (recognised from the registered loop).  Never reaches brows/cheeks.

    Weighting = partition-of-unity ALONG the lid (linear blend between the two
    nearest lid columns, incl. the corners) x a smoothstep fall-off from the lid
    margin toward the crease.  This is smooth by construction, independent of
    bone count / mesh density, and needs no post-smoothing or shape keys."""
    me = body.data
    mw = body.matrix_world
    co = np.array([list(mw @ v.co) for v in me.vertices], float)
    nv = len(me.vertices)

    if not any(m.type == 'ARMATURE' and m.object == rig for m in body.modifiers):
        md = body.modifiers.new(name="SR Armature", type='ARMATURE')
        md.object = rig

    # collect + clear the bones we (re)assign
    bones = set()
    for sd in bind_data:
        for _u, bn, _pt in (sd["upp"] + sd["low"]):
            bones.add(bn)
    for bn in bones:
        vg = body.vertex_groups.get(bn)
        if vg is not None:
            vg.remove(range(nv))
        else:
            body.vertex_groups.new(name=bn)

    # eyeball verts must NEVER get lid weight (else lids drag the eye surface)
    eye_verts = set()
    for nm in ("SR_eye.L", "SR_eye.R", "SR_eye_l", "SR_eye_r"):
        vg = body.vertex_groups.get(nm)
        if vg is not None:
            gi = vg.index
            for v in me.vertices:
                if any(g.group == gi for g in v.groups):
                    eye_verts.add(v.index)

    # ---- pass 1: accumulate smooth target weights per vert (don't write yet)
    #   * eyeball exclusion is by the REGISTERED eye verts (eye_verts), so the
    #     geometric radius gate can be loose enough to catch the lid-margin skin
    #     that hugs the eyeball (d ~ 0.8..1.0 r) - otherwise those verts stay on
    #     the head and leave an unsealed slit when the eye closes.
    wmap = {}                      # vert -> {bone: distribution weight}
    w_lid_of = {}                  # vert -> margin-falloff mass (0..1)
    assigned = set()
    for sd in bind_data:
        c, r = sd["c"], sd["r"]
        right, up, fwd = sd["right"], sd["up"], sd["fwd"]
        loop, band, vmax = sd["loop"], sd["band"], sd["vmax"]
        upp = [(u2, bn) for u2, bn, _p in sd["upp"]]
        low = [(u2, bn) for u2, bn, _p in sd["low"]]
        rel = co - c
        d = np.linalg.norm(rel, axis=1)
        ff = rel @ fwd
        uu = rel @ right
        vvv = rel @ up
        cand = ((ff > -0.30 * r) & (d < 2.0 * r) & (d > r * 0.80) &
                (np.abs(vvv) < 1.7 * vmax) &
                (uu > sd["u_in"] - 0.15 * sd["span"]) &
                (uu < sd["u_out"] + 0.15 * sd["span"]))
        for vi in np.nonzero(cand)[0]:
            vi = int(vi)
            if vi in eye_verts:
                continue
            dl = float(np.min(np.linalg.norm(loop - co[vi], axis=1)))
            if dl > band:
                continue
            # smoothstep fall-off: 1 on the lid margin -> 0 at the crease
            t = dl / band
            w_lid = 1.0 - t * t * (3.0 - 2.0 * t)
            if w_lid <= 1e-4:
                continue
            # partition of unity ALONG the lid: blend the 2 nearest columns
            ctrl = upp if vvv[vi] >= 0 else low
            m = wmap.setdefault(vi, {})
            for bn, wu in _blend_u(ctrl, float(uu[vi])):
                if wu > 1e-6:
                    m[bn] = m.get(bn, 0.0) + wu
            w_lid_of[vi] = max(w_lid_of.get(vi, 0.0), w_lid)
            assigned.add(vi)

    # ---- pass 2: SMOOTH the bone distribution across neighbouring lid verts.
    #   Blurs the along-lid weights (kills faceting / bunching on the closed lid)
    #   and, at the palpebral fissure, lets upper- and lower-lid verts share each
    #   other's bones so they zip TOGETHER on close instead of leaving a slit.
    #   Re-normalised each pass, so partition of unity is preserved.
    if assigned:
        nbrs = {vi: set() for vi in assigned}
        for e in me.edges:
            a, b = int(e.vertices[0]), int(e.vertices[1])
            if a in nbrs and b in nbrs:
                nbrs[a].add(b)
                nbrs[b].add(a)
        for _ in range(8):
            new = {}
            for vi in assigned:
                acc = dict(wmap[vi])
                for nj in nbrs[vi]:
                    for bn, w in wmap[nj].items():
                        acc[bn] = acc.get(bn, 0.0) + w
                tot = sum(acc.values()) or 1.0
                new[vi] = {bn: w / tot for bn, w in acc.items()}
            wmap = new

    # ---- pass 3: write (smoothed distribution) x (margin falloff); rest to head
    head_rem = {}
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
        head_rem[vi] = 1.0 - wl

    # find the face's MAIN deform bone = the non-lid DEF group holding the most
    # weight over the lid verts (Rigify head = DEF-spine.006). We hand the crease
    # remainder to it - and, critically, we STRIP it (and every other non-lid DEF
    # group) off the lid verts first, else that head weight keeps ~half the mass
    # and the lids only close halfway, leaving a slit.
    al = list(assigned)
    mass = {}
    for g in body.vertex_groups:
        if g.name.startswith("DEF-") and not g.name.startswith(("DEF-lid",
                                                                 "DEF-eye")):
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
        for gn in ("DEF-spine.006", "DEF-head", "head"):
            if gn in rig.data.bones:
                head_bone = gn
                break
    # strip ALL pre-existing head / face / neck skinning off the lid verts
    if al:
        for g in list(body.vertex_groups):
            if (g.name.startswith("DEF-") or g.name in ("head", "face")) and \
               not g.name.startswith(("DEF-lid", "DEF-eye")):
                g.remove(al)
    # give each crease vert its remainder back to the head so deform fades out
    hg = (body.vertex_groups.get(head_bone) or
          body.vertex_groups.new(name=head_bone)) if head_bone else None
    for vi, rem in head_rem.items():
        if hg is not None and rem > 1e-4:
            hg.add([vi], float(max(0.0, min(1.0, rem))), 'REPLACE')

    # weight the EYEBALL verts rigidly to DEF-eye so the aim control rotates them
    for s in (".L", ".R"):
        vg_eye = body.vertex_groups.get("SR_eye" + s)
        if vg_eye is None or ("DEF-eye" + s) not in rig.data.bones:
            continue
        gi = vg_eye.index
        everts = [v.index for v in me.vertices
                  if any(g.group == gi for g in v.groups)]
        if not everts:
            continue
        dg = body.vertex_groups.get("DEF-eye" + s) or \
            body.vertex_groups.new(name="DEF-eye" + s)
        dg.remove(range(nv))
        dg.add(everts, 1.0, 'REPLACE')
        for gn in ("DEF-head", "head", "DEF-spine.006", "DEF-neck", "DEF-face"):
            g = body.vertex_groups.get(gn)
            if g is not None:
                g.remove(everts)

    # de-facet the eyelid deformation on the extremes (wide-open crease / full
    # close) with a Corrective Smooth modifier SCOPED to the eye region - a pure
    # deformation smoother, NOT a shape key.
    try:
        _eye_smooth_modifier(body, assigned)
    except Exception:
        pass
    return len(assigned)


def _eye_smooth_modifier(body, assigned):
    """Scope a Corrective Smooth modifier to the eyelid region (+2 soft rings)
    so extreme open/close poses stay smooth without shape keys. rest_source=ORCO
    means it only smooths the DEFORMATION roughness, never the rest shape."""
    me = body.data
    nv = len(me.vertices)
    grp = (body.vertex_groups.get("SR_eye_smooth") or
           body.vertex_groups.new(name="SR_eye_smooth"))
    grp.remove(range(nv))
    if not assigned:
        return
    # the MARGIN line (palpebral fissure = the registered eyelid loop) must be
    # EXCLUDED: smoothing it averages the upper/lower overlap and RE-OPENS the
    # seal. We smooth the lid SKIN / crease (where the faceting is) but leave the
    # meeting line crisp, so the eye stays sealed on close.
    margin = set()
    for nm in ("SR_eyelid.L", "SR_eyelid.R"):
        vgm = body.vertex_groups.get(nm)
        if vgm is not None:
            gi = vgm.index
            for v in me.vertices:
                if any(g.group == gi for g in v.groups):
                    margin.add(v.index)
    # grow the region 2 rings (tapered weight) so the crease ABOVE the lid and
    # the cheek transition below it are smoothed too, with a soft boundary.
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
    # write weights, but keep the margin line (and its immediate ring) OUT
    grp.add([v for v in core if v not in margin], 1.0, 'REPLACE')
    if len(rings) > 0 and rings[0]:
        grp.add([v for v in rings[0] if v not in margin], 0.55, 'REPLACE')
    if len(rings) > 1 and rings[1]:
        grp.add([v for v in rings[1] if v not in margin], 0.28, 'REPLACE')
    mod = None
    for m in body.modifiers:
        if m.name == "SR Eye Smooth":
            mod = m
            break
    if mod is None:
        mod = body.modifiers.new("SR Eye Smooth", 'CORRECTIVE_SMOOTH')
    mod.vertex_group = "SR_eye_smooth"
    mod.factor = 0.45
    mod.iterations = 12
    mod.smooth_type = 'LENGTH_WEIGHTED'
    mod.rest_source = 'ORCO'
    # keep it LAST so it smooths the armature-deformed result
    try:
        while body.modifiers[-1].name != "SR Eye Smooth":
            body.modifiers.move(body.modifiers.find("SR Eye Smooth"),
                                len(body.modifiers) - 1)
            break
    except Exception:
        pass


# --------------------------------------------------------------------- clear
def clear_eye_rig(context, also_registration=False):
    """Remove the whole eye sample cleanly so it can be rebuilt / re-tried:
      - delete every eye-sample bone,
      - remove the eyelid deform weights AND hand those verts back to the head
        bone (so the face never tears / floats after a wipe),
      - optionally also drop the registrations (SR_eye/eyelid/corner + keys)."""
    rig = _target_rig(context)
    body = _body(context)
    removed = {"bones": 0, "weight_groups": 0, "restored_verts": 0,
               "registrations": 0}

    # never operate on mesh vertex groups while a mesh is in Edit mode
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
        # drop the eye-region Corrective Smooth modifier + its scope group
        for m in list(body.modifiers):
            if m.name == "SR Eye Smooth":
                body.modifiers.remove(m)
        gsm = body.vertex_groups.get("SR_eye_smooth")
        if gsm is not None:
            body.vertex_groups.remove(gsm)
        # deform groups this sample created (NOT the SR_ registration groups)
        dgroups = [g for g in body.vertex_groups
                   if g.name.startswith(("DEF-lid", "DEF-eye"))]
        gidx = {g.index for g in dgroups}
        vset = set()
        if gidx:
            for v in body.data.vertices:
                if any(gg.group in gidx for gg in v.groups):
                    vset.add(v.index)
        # give those verts back to the head so the mesh stays skinned
        if head_bone and vset:
            hg = body.vertex_groups.get(head_bone) or \
                body.vertex_groups.new(name=head_bone)
            hg.add(list(vset), 1.0, 'REPLACE')
            removed["restored_verts"] = len(vset)
        for g in list(dgroups):
            body.vertex_groups.remove(g)
            removed["weight_groups"] += 1
        if also_registration:
            for nm in ("SR_eye.L", "SR_eye.R", "SR_eyelid.L", "SR_eyelid.R",
                       "SR_corner.L", "SR_corner.R", "SR_eye_l", "SR_eye_r"):
                g = body.vertex_groups.get(nm)
                if g is not None:
                    body.vertex_groups.remove(g)
                    removed["registrations"] += 1

    if rig is not None:
        prev = context.view_layer.objects.active
        context.view_layer.objects.active = rig
        rig.hide_set(False)
        bpy.ops.object.mode_set(mode='EDIT')
        eb = rig.data.edit_bones
        pref = ("MSTR-Eyes", "CTL-Eye", "DEF-eye", "CTL-Lid", "DEF-lid",
                "CTL-Blink", "MCH-Lid", "MCH-tgt", "MCH-Tgt", "MCH-Rib",
                "MCH-rib")
        for b in list(eb):
            if b.name.startswith(pref):
                eb.remove(b)
                removed["bones"] += 1
        bpy.ops.object.mode_set(mode='OBJECT')
        if prev is not None:
            context.view_layer.objects.active = prev

    # remove the guide-ribbon meshes (they are rebuilt on every Build)
    for s in (".L", ".R"):
        for part in ("upp", "low"):
            o = bpy.data.objects.get("HLP-SR-rib_%s%s" % (part, s))
            if o is not None:
                m = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                if m is not None and m.users == 0:
                    bpy.data.meshes.remove(m)
    rc = bpy.data.collections.get("SR Eye Ribbons")
    if rc is not None and not rc.objects:
        bpy.data.collections.remove(rc)

    if also_registration:
        for k in ("sr_eye.L", "sr_eye.R"):
            if k in context.scene:
                del context.scene[k]
        for nm in ("SR_corner_in.L", "SR_corner_out.L",
                   "SR_corner_in.R", "SR_corner_out.R"):
            o = bpy.data.objects.get(nm)
            if o is not None:
                me = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                if me is not None and me.users == 0:
                    bpy.data.meshes.remove(me)
                removed["registrations"] += 1

    return removed


# ------------------------------------------------------------------- operators
class SMARTRIG_OT_eye_register(bpy.types.Operator):
    bl_idname = "smartrig.eye_register"
    bl_label = "Register Eye Part"
    bl_description = ("EYE_L/R: in Object mode pick the eyeball mesh. "
                      "LID_L/R: in Edit mode on the face, select the eyelid "
                      "edge loop (Alt+click). CORNER_L/R: select the two corner "
                      "verts (inner + outer canthus). Then run this.")
    bl_options = {'REGISTER', 'UNDO'}
    part: bpy.props.EnumProperty(
        items=(('EYE_L', "Eye L", "Register the LEFT eyeball mesh"),
               ('EYE_R', "Eye R", "Register the RIGHT eyeball mesh"),
               ('LID_L', "Eyelid L", "Register the LEFT eyelid loop"),
               ('LID_R', "Eyelid R", "Register the RIGHT eyelid loop"),
               ('CORNER_L', "Corners L", "Register LEFT inner+outer corners"),
               ('CORNER_R', "Corners R", "Register RIGHT inner+outer corners")),
        default='EYE_L')

    def execute(self, context):
        side = ".L" if self.part.endswith("_L") else ".R"
        if self.part.startswith("EYE"):
            ob = context.active_object
            if ob is None or ob.type != 'MESH':
                self.report({'ERROR'}, "Select the eyeball mesh (Object mode), "
                            "or its verts on a combined mesh (Edit mode)")
                return {'CANCELLED'}
            if context.mode == 'EDIT_MESH':
                bpy.ops.object.mode_set(mode='OBJECT')
                sel = [v.index for v in ob.data.vertices if v.select]
                if len(sel) >= 6:
                    vg = ob.vertex_groups.get("SR_eye" + side) or \
                        ob.vertex_groups.new(name="SR_eye" + side)
                    vg.remove([v.index for v in ob.data.vertices])
                    vg.add(sel, 1.0, 'REPLACE')
                    context.scene["sr_eye" + side] = ob.name
                    bpy.ops.object.mode_set(mode='EDIT')
                    self.report({'INFO'}, "Eye%s registered: %s (%d verts)"
                                % (side, ob.name, len(sel)))
                    return {'FINISHED'}
                bpy.ops.object.mode_set(mode='EDIT')
                self.report({'ERROR'}, "Select this eye's verts first")
                return {'CANCELLED'}
            vg = ob.vertex_groups.get("SR_eye" + side)
            if vg is not None:
                ob.vertex_groups.remove(vg)
            context.scene["sr_eye" + side] = ob.name
            self.report({'INFO'}, "Eye%s registered: %s" % (side, ob.name))
            return {'FINISHED'}

        # LID loop or CORNER verts - both live on the face body mesh
        body = _body(context)
        if body is None or body.type != 'MESH':
            self.report({'ERROR'}, "Set the Target Mesh (the face) first")
            return {'CANCELLED'}
        was_edit = (context.mode == 'EDIT_MESH')
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        sel = [v.index for v in body.data.vertices if v.select]

        if self.part.startswith("CORNER"):
            if len(sel) < 2:
                if was_edit:
                    bpy.ops.object.mode_set(mode='EDIT')
                self.report({'ERROR'}, "Select the TWO eye corners (inner + "
                            "outer canthus) in Edit mode, then run this")
                return {'CANCELLED'}
            vg = body.vertex_groups.get("SR_corner" + side)
            if vg is None:
                vg = body.vertex_groups.new(name="SR_corner" + side)
            else:
                vg.remove([v.index for v in body.data.vertices])
            vg.add(sel, 1.0, 'REPLACE')
            self.report({'INFO'}, "Corners%s registered (%d verts)"
                        % (side, len(sel)))
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            return {'FINISHED'}

        # eyelid loop
        if len(sel) < 6:
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, "Select the eyelid loop (Edit mode, "
                        "Alt+click) on the face, then run this")
            return {'CANCELLED'}
        vg = body.vertex_groups.get("SR_eyelid" + side)
        if vg is None:
            vg = body.vertex_groups.new(name="SR_eyelid" + side)
        else:
            vg.remove([v.index for v in body.data.vertices])
        vg.add(sel, 1.0, 'REPLACE')
        self.report({'INFO'}, "Eyelid%s registered (%d verts)" % (side, len(sel)))
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class SMARTRIG_OT_eye_sample(bpy.types.Operator):
    bl_idname = "smartrig.eye_sample_build"
    bl_label = "Build Eye Rig"
    bl_description = ("Build a clean, professional eye rig from the registered "
                      "eyes + eyelids + corners: aim/look + spoke lids + master "
                      "& per-point controls + paired blink + bounded binding.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            info = build_eye_rig(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        msg = ("Eye rig: eyes %s, upper %d / lower %d, bound %d verts"
               % (", ".join(info["eyes"]), info["upper"], info["lower"],
                  info.get("bound", 0)))
        if info.get("bind_error"):
            msg += " (bind skipped: %s)" % info["bind_error"]
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SMARTRIG_OT_eye_bind(bpy.types.Operator):
    bl_idname = "smartrig.eye_bind"
    bl_label = "Bind Eyelids"
    bl_description = ("Re-run ONLY the bounded eyelid skinning onto the existing "
                      "eye rig (weights stay inside the eye region).")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # rebuild always regenerates + binds; here we just call build with bind
        try:
            info = build_eye_rig(context)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Eyelids bound: %d verts" % info.get("bound", 0))
        return {'FINISHED'}


class SMARTRIG_OT_eye_clear(bpy.types.Operator):
    bl_idname = "smartrig.eye_clear"
    bl_label = "Clear Eye Sample"
    bl_description = ("Delete the whole eye sample (bones + eyelid weights) and "
                      "hand the eyelid verts back to the head bone, so you can "
                      "rebuild or try a different setup from a clean slate.")
    bl_options = {'REGISTER', 'UNDO'}
    also_registration: bpy.props.BoolProperty(
        name="Also clear registrations",
        description="Also remove the eye/eyelid/corner registrations (you would "
                    "then re-select them). Leave off to keep them and just "
                    "rebuild",
        default=False)

    def execute(self, context):
        try:
            rm = clear_eye_rig(context, self.also_registration)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Eye sample cleared: %d bones, %d weight groups, "
                    "%d verts back to head%s"
                    % (rm["bones"], rm["weight_groups"], rm["restored_verts"],
                       (", %d registrations" % rm["registrations"])
                       if self.also_registration else ""))
        return {'FINISHED'}


class SMARTRIG_OT_eye_corner_marker(bpy.types.Operator):
    bl_idname = "smartrig.eye_corner_marker"
    bl_label = "Eye Corner Markers"
    bl_description = ("Spawn two movable coloured markers for this eye - GREEN = "
                      "inner corner, RED = outer corner. Grab (G) each and drop "
                      "it exactly on the canthus, then Build. Press again to "
                      "re-select them (your positions are kept).")
    bl_options = {'REGISTER', 'UNDO'}
    side: bpy.props.StringProperty(default=".L")

    def execute(self, context):
        side = self.side
        body = _body(context)
        in_name = "SR_corner_in" + side
        out_name = "SR_corner_out" + side
        exists = (bpy.data.objects.get(in_name) is not None and
                  bpy.data.objects.get(out_name) is not None)

        # initial placement + marker size from the eyelid loop / eye
        size, p_in, p_out = 0.01, None, None
        loop = _lid_loop(body, side) if body is not None else None
        cr = _eye_center(context, side)
        if loop is not None and len(loop) >= 2:
            xs = loop[:, 0]
            p_lo, p_hi = loop[int(np.argmin(xs))], loop[int(np.argmax(xs))]
            p_in, p_out = ((p_lo, p_hi) if abs(p_lo[0]) <= abs(p_hi[0])
                           else (p_hi, p_lo))
            size = max(float(np.linalg.norm(loop.max(0) - loop.min(0))) * 0.10,
                       0.004)
        elif cr is not None:
            c, r = cr
            size = max(r * 0.22, 0.004)
            off = 0.9 * r if c[0] >= 0 else -0.9 * r
            p_in = np.array([c[0] - off, c[1] - 0.6 * r, c[2]])
            p_out = np.array([c[0] + off, c[1] - 0.6 * r, c[2]])
        else:
            self.report({'ERROR'}, "Register this eye (and its eyelid) first")
            return {'CANCELLED'}

        mk_in = _corner_marker(in_name, (0.10, 0.90, 0.20), size)
        mk_out = _corner_marker(out_name, (0.95, 0.15, 0.15), size)
        if not exists:
            mk_in.location = Vector([float(v) for v in p_in])
            mk_out.location = Vector([float(v) for v in p_out])

        try:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        for o in list(context.selected_objects):
            o.select_set(False)
        mk_in.select_set(True)
        mk_out.select_set(True)
        context.view_layer.objects.active = mk_in
        self.report({'INFO'}, "Corners%s: move GREEN=inner / RED=outer onto the "
                    "canthi, then Build (%s)"
                    % (side, "kept your positions" if exists else "placed a guess"))
        return {'FINISHED'}


def _setup_eye_corrective(context, side):
    """Pose the eye FULLY CLOSED and make a blink-driven corrective shape key the
    active, editable key - shared by the Edit- and Sculpt-mode correctives.
    Returns (rig, body, sk_name) or None on failure."""
    rig = _target_rig(context)
    if rig is None or rig.type != 'ARMATURE':
        return None
    body = _body(context)
    if body is None or body.type != 'MESH':
        return None
    blb = "CTL-Blink" + side
    pbb = rig.pose.bones.get(blb)
    if pbb is None:
        return None
    try:
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    # blink travel L (blink = -ly / L) - so blink = 1 is exactly full close
    L = 0.04
    for con in pbb.constraints:
        if con.name == "SR Blink Limit" and con.use_max_y:
            L = float(con.max_y) or L
            break
    me = body.data
    sk_name = "SR_blink_close" + side
    if me.shape_keys is None:
        body.shape_key_add(name="Basis", from_mix=False)
    kb = me.shape_keys.key_blocks
    sk = kb.get(sk_name)
    if sk is None:
        sk = body.shape_key_add(name=sk_name, from_mix=False)
    sk.slider_min = 0.0
    sk.slider_max = 1.0
    # drive value = max(0, blink) so the fix is absent when open, full at close.
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
        drv.expression = "max(0.0, b)"
        for v in list(drv.variables):
            drv.variables.remove(v)
        var = drv.variables.new()
        var.name = "b"
        var.type = 'SINGLE_PROP'
        tg = var.targets[0]
        tg.id_type = 'OBJECT'
        tg.id = rig
        tg.data_path = 'pose.bones["%s"]["blink"]' % blb
    except Exception:
        pass
    # pose fully closed
    try:
        pbb.location = (0.0, -L, 0.0)
        pbb["blink"] = 1.0
    except Exception:
        pass
    # make the corrective the active, editable key at full strength
    idx = kb.find(sk_name)
    if idx >= 0:
        body.active_shape_key_index = idx
    sk.value = 1.0
    body.show_only_shape_key = False
    body.use_shape_key_edit_mode = True
    # show the armature in edit mode + ON THE CAGE, so you edit the actual CLOSED
    # shape and Blender crazyspace-corrects the edit back into the key.
    for m in body.modifiers:
        if m.type == 'ARMATURE':
            m.show_in_editmode = True
            m.show_on_cage = True
    for o in list(context.selected_objects):
        o.select_set(False)
    body.select_set(True)
    context.view_layer.objects.active = body
    return rig, body, sk_name


class SMARTRIG_OT_eye_corrective(bpy.types.Operator):
    bl_idname = "smartrig.eye_corrective"
    bl_label = "Correct Closed Shape"
    bl_description = ("Pose the eye FULLY CLOSED and create a corrective shape key "
                      "wired to the blink (fades in ONLY as the lid closes), then "
                      "drop into EDIT or SCULPT mode so you can perfect the closed "
                      "shape - kill any gap, corner pinch or eyeball poke-through. "
                      "Edit it, then press 'Finish Correction'. It rides the blink "
                      "automatically.")
    bl_options = {'REGISTER', 'UNDO'}
    side: bpy.props.StringProperty(default=".L")
    mode: bpy.props.EnumProperty(
        items=(('SCULPT', "Sculpt", "Perfect the closed shape by sculpting"),
               ('EDIT', "Edit", "Perfect the closed shape by moving vertices")),
        default='SCULPT')

    def execute(self, context):
        r = _setup_eye_corrective(context, self.side)
        if r is None:
            self.report({'ERROR'}, "Build the eye rig first (need the blink "
                        "control + eyelid mesh)")
            return {'CANCELLED'}
        rig, body, sk_name = r
        try:
            bpy.ops.object.mode_set(mode=self.mode)
        except Exception as e:
            self.report({'WARNING'}, "Enter %s Mode manually: %s"
                        % (self.mode.title(), e))
        self.report({'INFO'},
                    "Eye%s CLOSED - %s the corrective, then press 'Finish "
                    "Correction'. It rides the blink." %
                    (self.side, self.mode.lower()))
        return {'FINISHED'}


class SMARTRIG_OT_eye_corrective_finish(bpy.types.Operator):
    bl_idname = "smartrig.eye_corrective_finish"
    bl_label = "Finish Correction (Conform)"
    bl_description = ("Confirm the closed-eye correction: leave Edit/Sculpt mode, "
                      "turn the edit-cage off, and return the eye to rest. The "
                      "correction stays stored in the blink-driven shape key, so it "
                      "shows up automatically only as the eye closes.")
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
            # park the shape-key panel back on the Basis (value is driven anyway)
            try:
                body.active_shape_key_index = 0
            except Exception:
                pass
        # return the eye(s) to rest so the correction fades out (driver = max(0,b))
        if rig is not None:
            for s in (".L", ".R"):
                pb = rig.pose.bones.get("CTL-Blink" + s)
                if pb is not None:
                    pb.location = (0.0, 0.0, 0.0)
        self.report({'INFO'}, "Correction stored + driven by the blink - it "
                    "appears only as the eye closes. Adjust anytime by re-opening "
                    "Edit/Sculpt.")
        return {'FINISHED'}


class SMARTRIG_OT_eye_corrective_mirror(bpy.types.Operator):
    bl_idname = "smartrig.eye_corrective_mirror"
    bl_label = "Mirror Correction"
    bl_description = ("Mirror the closed-eye corrective from one side to the other "
                      "(across X) so you only sculpt ONE eye. The other side is "
                      "created / overwritten and stays driven by its own blink.")
    bl_options = {'REGISTER', 'UNDO'}
    from_side: bpy.props.StringProperty(default=".L")

    def execute(self, context):
        from mathutils import kdtree
        rig = _target_rig(context)
        body = _body(context)
        if rig is None or body is None or body.type != 'MESH':
            self.report({'ERROR'}, "Build the eye rig first")
            return {'CANCELLED'}
        try:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        me = body.data
        other = ".R" if self.from_side == ".L" else ".L"
        src_name = "SR_blink_close" + self.from_side
        dst_name = "SR_blink_close" + other
        if me.shape_keys is None or src_name not in me.shape_keys.key_blocks:
            self.report({'ERROR'}, "No correction on %s yet - sculpt/edit that "
                        "side first, Finish, then Mirror" % self.from_side)
            return {'CANCELLED'}
        kb = me.shape_keys.key_blocks
        basis = kb[0]
        src = kb[src_name]
        dst = kb.get(dst_name)
        if dst is None:
            dst = body.shape_key_add(name=dst_name, from_mix=False)
        dst.slider_min = 0.0
        dst.slider_max = 1.0
        n = len(basis.data)
        # spatial mirror map: for each vert, the vert nearest its X-flipped rest pos
        kd = kdtree.KDTree(n)
        for i in range(n):
            co = basis.data[i].co
            kd.insert((co[0], co[1], co[2]), i)
        kd.balance()
        # reset the destination key to Basis, then write the mirrored deltas
        for i in range(n):
            dst.data[i].co = basis.data[i].co
        moved = 0
        for i in range(n):
            d = src.data[i].co - basis.data[i].co
            if d.length < 1e-6:
                continue
            bc = basis.data[i].co
            _, j, _ = kd.find((-bc[0], bc[1], bc[2]))
            dst.data[j].co = basis.data[j].co + Vector((-d[0], d[1], d[2]))
            moved += 1
        # wire the destination key's driver to the OTHER side's blink
        blb = "CTL-Blink" + other
        dp = 'key_blocks["%s"].value' % dst_name
        try:
            ad = me.shape_keys.animation_data
            if ad is not None:
                for dd in list(ad.drivers):
                    if dd.data_path == dp:
                        ad.drivers.remove(dd)
        except Exception:
            pass
        try:
            fcu = me.shape_keys.driver_add(dp)
            drv = fcu.driver
            drv.type = 'SCRIPTED'
            drv.expression = "max(0.0, b)"
            for v in list(drv.variables):
                drv.variables.remove(v)
            var = drv.variables.new()
            var.name = "b"
            var.type = 'SINGLE_PROP'
            tg = var.targets[0]
            tg.id_type = 'OBJECT'
            tg.id = rig
            tg.data_path = 'pose.bones["%s"]["blink"]' % blb
        except Exception:
            pass
        self.report({'INFO'}, "Mirrored %s -> %s (%d verts) - driven by the %s "
                    "blink." % (self.from_side, other, moved, other))
        return {'FINISHED'}


_classes = (SMARTRIG_OT_eye_register, SMARTRIG_OT_eye_sample,
            SMARTRIG_OT_eye_bind, SMARTRIG_OT_eye_clear,
            SMARTRIG_OT_eye_corner_marker, SMARTRIG_OT_eye_corrective,
            SMARTRIG_OT_eye_corrective_finish,
            SMARTRIG_OT_eye_corrective_mirror)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
