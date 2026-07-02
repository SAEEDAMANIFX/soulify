"""Mannequin Retargeting - Phase 1 (v1.24.0). See DESIGN_MANNEQUIN_RETARGET.md.

Extract the GARMENT SKELETON (torso axis + limb axes from the garment's own
tube geometry) and build a procedural MANNEQUIN inside the garment: a stick
skeleton fleshed out by a Skin modifier, its radii taken from the garment
interior so it wears the garment snugly. The mannequin gives full
garment-to-body correspondence for free; later phases bind the garment to it
and morph it into the user's character.
"""
import bpy
import math
import numpy as np
from mathutils import Vector

from . import utils

MANN_NAME = "SRF_Mannequin"


# ------------------------------------------------------------ garment bones --

def _trace_tube(co, tip_c, tip_r, toward, cx, z_stop, steps=16):
    """March from a tube opening (cuff / leg hem) toward the torso along the
    tube's OWN centerline - handles sleeves/legs modeled BENT (elbow/knee in
    the design pose). Returns the centerline as a list of Vectors, tip first.
    Stops when the tube opens into the torso (spread jump), when it CROSSES
    the body midline (it drifted onto the other side - the shoulder_l-at-+x
    bug), or when it climbs above the collar."""
    p = np.array([tip_c.x, tip_c.y, tip_c.z], dtype=float)
    d = np.array([toward.x, toward.y, toward.z], dtype=float)
    d /= (np.linalg.norm(d) + 1e-12)
    r = max(tip_r * 2.2, 1e-4)
    sgn = 1.0 if tip_c.x >= cx else -1.0
    lat0 = abs(tip_c.x - cx)
    pts = [Vector(p)]
    for _ in range(steps):
        q = p + d * r * 0.8
        sel = co[np.linalg.norm(co - q, axis=1) < r]
        if len(sel) < 6:
            break
        c = sel.mean(axis=0)
        if sgn * (c[0] - cx) < 0.20 * lat0:        # reached/crossed the midline
            break
        if c[2] > z_stop:                          # climbed into the collar
            break
        nd = c - p
        nl = np.linalg.norm(nd)
        if nl < 1e-9:
            break
        nd /= nl
        d = 0.6 * d + 0.4 * nd                    # follow the bend, stay stable
        d /= (np.linalg.norm(d) + 1e-12)
        p = c
        pts.append(Vector(c))
        spread = float(np.percentile(np.linalg.norm(sel - c, axis=1), 80))
        if spread > 2.6 * tip_r:                  # merged into the torso
            break
    return pts


def _bend_joint(polyline):
    """The design-pose ELBOW/KNEE: the centerline point deviating most from
    the straight tip->root chord (a straight sleeve returns the midpoint)."""
    if len(polyline) < 3:
        return polyline[len(polyline) // 2] if polyline else None
    a, b = polyline[0], polyline[-1]
    ab = (b - a)
    L = ab.length
    if L < 1e-9:
        return polyline[len(polyline) // 2]
    ab /= L
    best, bd = None, -1.0
    for p in polyline[1:-1]:
        d = ((p - a) - ab * ((p - a).dot(ab))).length
        if d > bd:
            bd, best = d, p
    if bd < 0.03 * L:                              # effectively straight
        return a.lerp(b, 0.5)
    return best


def _lower_mode(co, z0, span, cx, cy):
    """How the garment treats the legs:
      'LEGS'  - splits into two tubes below the crotch (pants/shalwar)
      'LOOSE' - one wide column continues down (kandura/thobe/dress/skirt):
                binds to pelvis/spine so KNEE BENDS NEVER TEAR IT
      'NONE'  - garment ends above the crotch (shirt/top)"""
    zc = z0 + 0.28 * span                          # below-crotch test band
    sel = co[np.abs(co[:, 2] - zc) < 0.05 * span]
    if len(sel) < 12:
        return 'NONE'
    xs = np.sort(sel[:, 0] - cx)
    # two clusters w/ a central gap = legs
    mid = np.abs(xs) < 0.25 * (xs.max() - xs.min() + 1e-9)
    if mid.sum() < 0.08 * len(xs):
        return 'LEGS'
    return 'LOOSE'

def garment_skeleton(g_ob):
    """Joints implied by the garment (world space):
      {'pelvis','chest','neck', 'shoulder_l','elbow_l','wrist_l', ...R,
       'hip_l','knee_l','ankle_l', ...R (pants only), 'radii': {...}}
    Torso from the garment's vertical extent + ring axis; sleeves/legs from
    OFFSET rings (cuffs / leg hems) traced back toward the torso."""
    from .garment import _all_rings, analyze_garment
    g = analyze_garment(g_ob)
    if g is None:
        return None
    co = utils.read_rest_coords(g_ob)
    z0, z1 = float(co[:, 2].min()), float(co[:, 2].max())
    span = max(z1 - z0, 1e-6)
    cx = (float(co[:, 0].min()) + float(co[:, 0].max())) * 0.5
    cy = (float(co[:, 1].min()) + float(co[:, 1].max())) * 0.5

    def torso_r(z, q=60):
        sel = co[(np.abs(co[:, 2] - z) < 0.04 * span)
                 & (np.abs(co[:, 0] - cx) < 0.35 * span)]
        if len(sel) < 6:
            return 0.12 * span
        return float(np.percentile(np.hypot(sel[:, 0] - cx, sel[:, 1] - cy), q))

    rings = _all_rings(g_ob)
    tc = g["top_c"]
    max_r = max([r for (_, r) in g["profile"]] + [g["top_r"]])
    offset_rings = []
    for (c, r, n, nrm) in rings:
        if n < 20 or r < 1e-6:
            continue
        if math.hypot(c.x - cx, c.y - cy) > 0.6 * r and r < 0.55 * max_r:
            offset_rings.append((c, r))
    # split by side, keep the biggest per side
    left = [x for x in offset_rings if x[0].x < cx]
    right = [x for x in offset_rings if x[0].x >= cx]
    left = max(left, key=lambda x: x[1]) if left else None
    right = max(right, key=lambda x: x[1]) if right else None

    jt = {"radii": {}}
    z_neck = tc.z
    jt["neck"] = Vector((cx, cy, z_neck))
    jt["pelvis"] = Vector((cx, cy, z0 + 0.02 * span))
    jt["chest"] = Vector((cx, cy, z0 + 0.62 * span))
    jt["radii"]["torso"] = torso_r(z0 + 0.55 * span) * 0.62
    jt["radii"]["chest"] = torso_r(z0 + 0.70 * span) * 0.62
    jt["radii"]["neck"] = max(g["top_r"] * 0.55, 0.02 * span)

    def _limb(side, ring):
        if ring is None:
            return
        c, r = ring
        sgn = -1.0 if c.x < cx else 1.0
        is_leg = c.z < z0 + 0.35 * span and abs(c.x - cx) < 0.30 * span
        if is_leg:
            root_guess = Vector((cx + sgn * jt["radii"]["torso"] * 0.55, cy,
                                 z0 + 0.42 * span))
            names = ("hip", "knee", "ankle")
        else:
            root_guess = Vector((cx + sgn * torso_r(z_neck - 0.12 * span) * 0.9,
                                 cy, z_neck - 0.10 * span))
            names = ("shoulder", "elbow", "wrist")
        tip = Vector(c)
        # trace the REAL tube centerline (handles bent elbows/knees in the
        # design pose); root = where the tube merged into the torso
        line = _trace_tube(co, tip, r, (root_guess - tip), cx,
                           z_neck - 0.05 * span)
        root = line[-1] if len(line) >= 2 else root_guess
        mid = _bend_joint(line) or root.lerp(tip, 0.5)
        jt["%s_%s" % (names[0], side)] = root
        jt["%s_%s" % (names[1], side)] = mid
        jt["%s_%s" % (names[2], side)] = tip
        jt["radii"]["%s_%s" % (names[0], side)] = r * 0.75

    _limb("l", left)
    _limb("r", right)

    # ---- ANATOMICAL torso joints ----
    # A garment's hem is NOT the pelvis (mapping a shirt's hem to the body
    # pelvis crushed the shirt). Anchor anatomy on what the garment really
    # tells us: the collar (neck) and the traced shoulder roots. Human
    # proportions: neck->pelvis ~= 1.55 x shoulder width; chest ~= 0.55 x.
    if "shoulder_l" in jt and "shoulder_r" in jt:
        sw = (jt["shoulder_l"] - jt["shoulder_r"]).length
        if sw > 1e-6:
            jt["chest"] = Vector((cx, cy, z_neck - 0.55 * sw))
            jt["pelvis"] = Vector((cx, cy, z_neck - 1.55 * sw))
    elif g["top_r"] / max_r >= 0.6:
        # bottoms (skirt/pants): the top ring IS the waist -> pelvis
        jt["pelvis"] = Vector((cx, cy, tc.z))
        jt["chest"] = Vector((cx, cy, tc.z + 3.0 * g["top_r"]))
        jt["neck"] = Vector((cx, cy, tc.z + 5.5 * g["top_r"]))

    # ---- lower body: pants legs / loose column (kandura, dress, skirt) ----
    # the class is the PRIOR (a shirt's lower band is also a tube - it must
    # not read as a loose column); the two-cluster split test decides pants
    from .garment import classify_garment
    label, _isb = classify_garment(g, rings)
    split = _lower_mode(co, z0, span, cx, cy)
    if split == 'LEGS':
        mode = 'LEGS'
    elif label in ("skirt", "dress/thobe"):
        mode = 'LOOSE'
    else:
        mode = 'NONE'
    jt["lower_mode"] = mode
    jt["label"] = label
    has_legs = any(k.startswith("hip_") for k in jt)
    if mode == 'LOOSE' and not has_legs:
        # kandura/long dress/skirt: the column hangs free. The mannequin still
        # gets STRAIGHT legs inside it (needed to retarget onto the character)
        # but they are marked FREE: the loose fabric binds to pelvis/spine, so
        # bending the character's knees NEVER tears or drags the garment.
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            hx = cx + sgn * jt["radii"]["torso"] * 0.5
            jt["hip_%s" % side] = Vector((hx, cy, z0 + 0.45 * span))
            jt["knee_%s" % side] = Vector((hx, cy, z0 + 0.22 * span))
            jt["ankle_%s" % side] = Vector((hx, cy, z0 + 0.02 * span))
            jt["radii"]["hip_%s" % side] = jt["radii"]["torso"] * 0.38
        jt["free_legs"] = True
    return jt


# --------------------------------------------------------------- build mesh --

def adjusted_joints(jt, props):
    """Apply the user's LIVE mannequin controls to the base joints:
    arm open (raise/lower both arms), elbow bend, neck length. Volume factors
    are applied at build time (radii)."""
    from mathutils import Matrix as _M
    out = {k: (Vector(v) if isinstance(v, Vector) else v) for k, v in jt.items()}
    # neck length (fraction of torso length)
    if "neck" in out and "chest" in out:
        tl = (out["neck"] - out["chest"]).length
        out["neck"] = out["neck"] + Vector((0, 0, tl * props.mann_neck_len))
    for side, sgn in (("l", -1.0), ("r", 1.0)):
        sh = out.get("shoulder_%s" % side)
        el = out.get("elbow_%s" % side)
        wr = out.get("wrist_%s" % side)
        if sh is not None and el is not None:
            # open/close the whole arm: rotate about the FRONT axis (Y)
            R = _M.Rotation(math.radians(props.mann_arm_open) * -sgn, 3, 'Y')
            el2 = sh + R @ (el - sh)
            out["elbow_%s" % side] = el2
            if wr is not None:
                wr2 = sh + R @ (wr - sh)
                # extra elbow bend: rotate the forearm about the elbow, around
                # the axis perpendicular to the arm in the body plane
                arm = (el2 - sh)
                ax = arm.cross(Vector((0.0, 1.0, 0.0)))
                if ax.length > 1e-9:
                    Rb = _M.Rotation(math.radians(props.mann_elbow_bend) * -sgn,
                                     3, ax.normalized())
                    wr2 = el2 + Rb @ (wr2 - el2)
                out["wrist_%s" % side] = wr2
    return out


def build_mannequin(jt, name=MANN_NAME):
    """Procedural mannequin: joint verts + edges + Skin modifier + subsurf.
    Radii come from the garment, so the flesh fills the clothes."""
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)

    order = ["pelvis", "chest", "neck"]
    for side in ("l", "r"):
        for j in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            k = "%s_%s" % (j, side)
            if k in jt:
                order.append(k)
    idx = {k: i for i, k in enumerate(order)}
    verts = [jt[k] for k in order]
    edges = [(idx["pelvis"], idx["chest"]), (idx["chest"], idx["neck"])]
    for side in ("l", "r"):
        for a, b, root in (("shoulder", "elbow", "chest"),
                           ("elbow", "wrist", None),
                           ("hip", "knee", "pelvis"),
                           ("knee", "ankle", None)):
            ka, kb = "%s_%s" % (a, side), "%s_%s" % (b, side)
            if ka in idx and kb in idx:
                edges.append((idx[ka], idx[kb]))
                if root and (idx[root], idx[ka]) not in edges:
                    edges.append((idx[root], idx[ka]))
    me.from_pydata([v[:] for v in verts], edges, [])
    me.update()

    skin = ob.modifiers.new("SRF_Skin", 'SKIN')
    ob.modifiers.new("SRF_Sub", 'SUBSURF').levels = 2
    ob.data.skin_vertices[0].data[idx["pelvis"]].use_root = True
    rad = jt["radii"]
    base = rad.get("torso", 0.1)
    for k, i in idx.items():
        r = base * 0.5
        if k in ("pelvis", "chest"):
            r = rad.get("torso" if k == "pelvis" else "chest", base)
        elif k == "neck":
            r = rad.get("neck", base * 0.4)
        else:
            j = k.split("_")[0]
            r = rad.get("shoulder_" + k[-1], base * 0.45) \
                if j in ("shoulder", "elbow", "wrist") else base * 0.6
            if j in ("elbow", "wrist"):
                r *= 0.85
            if j in ("knee", "ankle"):
                r *= 0.8
        ob.data.skin_vertices[0].data[i].radius = (r, r)
    ob.display_type = 'SOLID'
    ob["srf_order"] = order                        # stick vert i -> joint name
    ob["srf_joints"] = {k: list(v) for k, v in jt.items()
                        if isinstance(v, Vector)}
    ob["srf_radii"] = {k: float(v) for k, v in jt.get("radii", {}).items()}
    ob["srf_lower_mode"] = jt.get("lower_mode", 'NONE')
    ob["srf_free_legs"] = bool(jt.get("free_legs", False))
    ob["srf_label"] = jt.get("label", "?")
    return ob


# ----------------------------------------------------------- live controls --

_ADJ_TOKEN = [0]


def live_adjust(context):
    """update= callback for the mannequin sliders: re-pose the mannequin and
    re-warp the garment from its DESIGN onto the adjusted pose. Debounced."""
    props = context.scene.smartrig
    g_ob = props.garment_object
    mq = bpy.data.objects.get(MANN_NAME)
    if g_ob is None or mq is None:
        return
    jb = mq.get("srf_joints_base") or mq.get("srf_joints")
    if not jb:
        return
    _ADJ_TOKEN[0] += 1
    tok = _ADJ_TOKEN[0]
    gname, mname = g_ob.name, mq.name

    def _run():
        if tok != _ADJ_TOKEN[0]:
            return None
        g = bpy.data.objects.get(gname)
        m = bpy.data.objects.get(mname)
        if g is None or m is None:
            return None
        p = bpy.context.scene.smartrig
        base = {k: Vector(v) for k, v in
                (m.get("srf_joints_base") or m.get("srf_joints")).items()}
        radii = dict(m.get("srf_radii", {}))
        # volume factors: torso / arms
        for k in list(radii.keys()):
            if k in ("torso", "chest", "neck"):
                radii[k] = radii[k] * p.mann_torso_vol
            else:
                radii[k] = radii[k] * p.mann_arm_vol
        adj = adjusted_joints(base, p)
        adj["radii"] = radii
        adj["lower_mode"] = m.get("srf_lower_mode", 'NONE')
        adj["label"] = m.get("srf_label", "?")
        base_store = {k: list(v) for k, v in base.items()}
        nm = build_mannequin(adj)
        nm["srf_joints_base"] = base_store
        neutral = (abs(p.mann_arm_open) < 1e-3 and abs(p.mann_elbow_bend) < 1e-3
                   and abs(p.mann_neck_len) < 1e-3)
        if not neutral:                            # neutral = keep the fit as-is
            try:
                warp_garment(g, base, adj,
                             loose=(m.get("srf_lower_mode") == 'LOOSE'))
            except Exception as e:
                print("Soulify mannequin live_adjust:", e)
        return None

    bpy.app.timers.register(_run, first_interval=0.30)


# ------------------------------------------------------- the GARMENT RIG ----

RIG_NAME = "SRF_GarmentRig"

# bone -> (head joint, tail joint, parent bone)
_BONES = [("spine1", "pelvis", "chest", None),
          ("spine2", "chest", "neck", "spine1"),
          ("arm_l", "shoulder_l", "elbow_l", "spine2"),
          ("fore_l", "elbow_l", "wrist_l", "arm_l"),
          ("arm_r", "shoulder_r", "elbow_r", "spine2"),
          ("fore_r", "elbow_r", "wrist_r", "arm_r"),
          ("leg_l", "hip_l", "knee_l", "spine1"),
          ("shin_l", "knee_l", "ankle_l", "leg_l"),
          ("leg_r", "hip_r", "knee_r", "spine1"),
          ("shin_r", "knee_r", "ankle_r", "leg_r")]


def build_garment_rig(g_ob, jt, loose=False, coords=None):
    """A REAL armature from the garment's implied skeleton, with the garment
    SKINNED to it (our smooth segment weights). This is the live control the
    user asked for: grab any bone in Pose Mode and the garment follows
    instantly - and Match to Character simply poses these bones."""
    old = bpy.data.objects.get(RIG_NAME)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    arm = bpy.data.armatures.new(RIG_NAME)
    ob = bpy.data.objects.new(RIG_NAME, arm)
    bpy.context.scene.collection.objects.link(ob)   # identity world transform
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode='EDIT')
    made = {}
    for name, hj, tj, parent in _BONES:
        if hj not in jt or tj not in jt:
            continue
        eb = arm.edit_bones.new(name)
        eb.head = jt[hj]
        eb.tail = jt[tj]
        if parent in made:
            eb.parent = made[parent]
        made[name] = eb
    bpy.ops.object.mode_set(mode='OBJECT')

    # skin the garment: smooth inverse-distance weights per bone segment
    me = g_ob.data
    n = len(me.vertices)
    mw = g_ob.matrix_world
    if coords is not None:
        wco = coords                               # e.g. the WARPED positions
    else:
        wco = np.array([(mw @ v.co)[:] for v in me.vertices], dtype=float)
    names, segs = [], []
    for (b, h, t, _p) in _BONES:
        if b in made:
            names.append(b)
            segs.append((np.array(jt[h][:]), np.array(jt[t][:])))
    W = np.zeros((n, len(names)))
    for k, (a0, b0) in enumerate(segs):
        ab = b0 - a0
        L2 = float(ab.dot(ab)) + 1e-12
        t = np.clip(((wco - a0) @ ab) / L2, 0.0, 1.0)
        cl = a0 + t[:, None] * ab
        d2 = np.sum((wco - cl) ** 2, axis=1)
        W[:, k] = 1.0 / (d2 + 1e-8) ** 2.5
    if loose and "pelvis" in jt:
        below = wco[:, 2] < jt["pelvis"].z
        for k, nm in enumerate(names):
            if nm != "spine1":
                W[below, k] = 0.0
    W /= (W.sum(axis=1, keepdims=True) + 1e-12)
    for nm in names:
        vg = g_ob.vertex_groups.get(nm)
        if vg:
            g_ob.vertex_groups.remove(vg)
    vgs = {nm: g_ob.vertex_groups.new(name=nm) for nm in names}
    for k, nm in enumerate(names):
        col = W[:, k]
        for i in np.nonzero(col > 0.001)[0]:
            vgs[nm].add([int(i)], float(col[i]), 'REPLACE')
    for m in list(g_ob.modifiers):
        if m.type == 'ARMATURE' and m.name == "SRF_Arm":
            g_ob.modifiers.remove(m)
    md = g_ob.modifiers.new("SRF_Arm", 'ARMATURE')
    md.object = ob
    return ob


def snap_rig_to_joints(arm_ob, jt_src, jt_dst, gs=1.0):
    """Pose the garment rig so every bone lands on the character's joints.
    Stretch along the bone (Y) by the length ratio; X/Z by the global size
    ratio only (girth is design). Parents first."""
    bpy.context.view_layer.objects.active = arm_ob
    bpy.ops.object.mode_set(mode='POSE')
    for name, hj, tj, _p in _BONES:
        pb = arm_ob.pose.bones.get(name)
        if pb is None or hj not in jt_dst or tj not in jt_dst \
                or hj not in jt_src or tj not in jt_src:
            continue
        a1, b1 = jt_dst[hj], jt_dst[tj]
        v1 = b1 - a1
        if v1.length < 1e-9:
            continue
        rest = pb.bone.matrix_local            # armature space (== world here)
        rest_y = Vector((rest[0][1], rest[1][1], rest[2][1])).normalized()
        q = rest_y.rotation_difference(v1.normalized())
        R = q.to_matrix()
        rest3 = rest.to_3x3()
        basis = R @ rest3
        L0 = (jt_src[tj] - jt_src[hj]).length
        sy = v1.length / max(L0, 1e-9)
        M = basis.to_4x4()
        for r_ in range(3):                    # scale columns: X,Z = gs, Y = sy
            M[r_][0] *= gs
            M[r_][1] *= sy
            M[r_][2] *= gs
        M[0][3], M[1][3], M[2][3] = a1.x, a1.y, a1.z
        pb.matrix = M
        bpy.context.view_layer.update()        # parents affect children
    bpy.ops.object.mode_set(mode='OBJECT')


# --------------------------- MetaTailor-style: ride the mannequin's surface --

def retarget_ride(context, g_ob, jt_src, jt_dst, body, loose=False):
    """How MetaTailor succeeded: NEVER map garment->body directly. Bind the
    garment to the MANNEQUIN's surface (Surface Deform - native, silky), then
    MORPH THE MANNEQUIN into the character by moving its stick joints (they
    ARE its vertices) - the garment rides the morph with perfect, smooth
    correspondence. No bone math, no rays."""
    # 1. mannequin inside the garment - FROZEN to a plain mesh (the skin
    # modifier regenerates topology when its stick verts move, which breaks
    # the Surface Deform binding -> garbage). Frozen mesh + warp morph keeps
    # the topology constant forever.
    mq = build_mannequin(jt_src)
    dg0 = context.evaluated_depsgraph_get()
    frozen = bpy.data.meshes.new_from_object(mq.evaluated_get(dg0))
    mq.modifiers.clear()
    mq.data = frozen
    # 2. bind the garment to the mannequin surface
    for m in list(g_ob.modifiers):
        if m.type == 'SURFACE_DEFORM' and m.name == "SRF_Ride":
            g_ob.modifiers.remove(m)
    md = g_ob.modifiers.new("SRF_Ride", 'SURFACE_DEFORM')
    md.target = mq
    md.falloff = 4.0
    context.view_layer.objects.active = g_ob
    with context.temp_override(object=g_ob, active_object=g_ob):
        bpy.ops.object.surfacedeform_bind(modifier=md.name)
    if not md.is_bound:
        bpy.data.objects.remove(mq, do_unlink=True)
        return False
    # 3. morph the FROZEN mannequin body with the bone-pair warp (easy case:
    # smooth capsule body, no tents possible) - topology constant, SD follows
    me = mq.data
    nm = len(me.vertices)
    wm = np.array([v.co[:] for v in me.vertices], dtype=float)
    segs = []
    for name, hj, tj, _p in _BONES:
        if hj in jt_src and tj in jt_src and hj in jt_dst and tj in jt_dst:
            a0, b0 = jt_src[hj], jt_src[tj]
            a1, b1 = jt_dst[hj], jt_dst[tj]
            v0, v1 = (b0 - a0), (b1 - a1)
            if v0.length < 1e-9 or v1.length < 1e-9:
                continue
            R = mathutils_matrix_to_np(v0.rotation_difference(v1).to_matrix())
            segs.append((np.array(a0[:]), np.array(b0[:]), R,
                         np.array(a1[:]),
                         v1.length / v0.length, np.array(v0.normalized()[:])))
    if not segs:
        bpy.data.objects.remove(mq, do_unlink=True)
        return False
    W = np.zeros((nm, len(segs)))
    for k, (a0, b0, _R, _a1, _ax, _d0) in enumerate(segs):
        ab = b0 - a0
        L2 = float(ab.dot(ab)) + 1e-12
        t = np.clip(((wm - a0) @ ab) / L2, 0.0, 1.0)
        cl = a0 + t[:, None] * ab
        W[:, k] = 1.0 / (np.sum((wm - cl) ** 2, axis=1) + 1e-8) ** 2.5
    W /= (W.sum(axis=1, keepdims=True) + 1e-12)
    outm = np.zeros_like(wm)
    for k, (a0, _b0, R, a1, axial, d0) in enumerate(segs):
        S = np.eye(3) + (axial - 1.0) * np.outer(d0, d0)
        outm += W[:, k:k + 1] * ((wm - a0) @ (R @ S).T + a1)
    for i in range(nm):
        me.vertices[i].co = outm[i]
    me.update()
    # 3.5 THE MetaTailor STEP: the mannequin now WEARS THE CHARACTER'S SKIN -
    # shrinkwrap its generated surface onto the real body, so its surface
    # becomes her surface and the garment rides out to the true silhouette
    sw = mq.modifiers.new("SRF_Skin2Body", 'SHRINKWRAP')
    sw.target = body
    sw.wrap_method = 'NEAREST_SURFACEPOINT'
    sw.wrap_mode = 'ON_SURFACE'
    context.view_layer.update()
    # 3.6 live cleanup on the garment itself: a thin OUTSIDE shrinkwrap vs the
    # real body catches what the mannequin approximation misses
    for m in list(g_ob.modifiers):
        if m.name == "SRF_Clean":
            g_ob.modifiers.remove(m)
    from . import utils as _u
    bco = _u.read_rest_coords(body)
    cl = g_ob.modifiers.new("SRF_Clean", 'SHRINKWRAP')
    cl.target = body
    cl.wrap_mode = 'OUTSIDE'
    cl.wrap_method = 'NEAREST_SURFACEPOINT'
    cl.offset = 0.003 * float(bco[:, 2].max() - bco[:, 2].min())
    # 4. keep it LIVE: the user can nudge any mannequin vertex (a joint) and
    # the garment follows; hide the mannequin visually only
    mq.hide_set(True)
    mq.hide_render = True
    return True


# ------------------------------------- SURFACE RETARGET (the deep solution) --

def retarget_surface(g_ob, jt_src, jt_dst, body, loose=False):
    """Professional garment retarget via BONE-SURFACE coordinates.
    Every garment vertex is encoded in the DESIGN as:
        (bone segment, t along the bone, angle around the bone,
         CLEARANCE above the garment's own inner wall)
    and decoded on the CHARACTER by casting a ray from her bone at the same
    (t, angle) to her REAL surface (evaluated - subsurf included) and standing
    the vertex at surface + clearance. The fabric therefore SPREADS ON THE
    BODY with its designed looseness: sleeves wrap the actual arms, the waist
    cannot balloon, shoulders cannot poke through.
    Replaces the bone-pair warp (blind to the surface - tents & pokes)."""
    me = g_ob.data
    n = len(me.vertices)
    mw = g_ob.matrix_world
    base = [v.co.copy() for v in me.vertices]
    wco = np.array([(mw @ c)[:] for c in base], dtype=float)

    # global size ratio for the clearance (design mm -> character mm)
    if all(k in jt_src and k in jt_dst
           for k in ("shoulder_l", "shoulder_r")):
        gs = (jt_dst["shoulder_l"] - jt_dst["shoulder_r"]).length \
            / max((jt_src["shoulder_l"] - jt_src["shoulder_r"]).length, 1e-9)
    else:
        gs = (jt_dst["neck"] - jt_dst["pelvis"]).length \
            / max((jt_src["neck"] - jt_src["pelvis"]).length, 1e-9)

    radii = jt_src.get("radii", {}) if isinstance(jt_src, dict) else {}
    span_g = float(wco[:, 2].max() - wco[:, 2].min()) or 1.0

    segs = []
    for name, hj, tj, _p in _BONES:
        if hj in jt_src and tj in jt_src and hj in jt_dst and tj in jt_dst:
            a0, b0 = jt_src[hj], jt_src[tj]
            a1, b1 = jt_dst[hj], jt_dst[tj]
            if (b0 - a0).length < 1e-9 or (b1 - a1).length < 1e-9:
                continue
            is_spine = name.startswith("spine")
            side = name[-1] if name[-1] in ("l", "r") else ""
            tube_r = radii.get("shoulder_%s" % side,
                               radii.get("hip_%s" % side, 0.08 * span_g))
            segs.append({"name": name, "a0": a0, "b0": b0, "a1": a1, "b1": b1,
                         "spine": is_spine,
                         "rlim": (max(2.6 * tube_r, 0.045 * span_g)
                                  if not is_spine else 1e9)})
    if not segs:
        return False
    pelvis_z = jt_src["pelvis"].z if "pelvis" in jt_src else -1e9

    # ---- assign each vertex to its DESIGN segment (tube membership) ----
    NB = len(segs)
    D2 = np.full((n, NB), 1e18)
    T = np.zeros((n, NB))
    for k, s in enumerate(segs):
        a0 = np.array(s["a0"][:]); b0 = np.array(s["b0"][:])
        ab = b0 - a0
        L2 = float(ab.dot(ab)) + 1e-12
        t = np.clip(((wco - a0) @ ab) / L2, 0.0, 1.0)
        cl = a0 + t[:, None] * ab
        d2 = np.sum((wco - cl) ** 2, axis=1)
        if not s["spine"]:
            d2[d2 > s["rlim"] ** 2] = 1e18         # limb claims tube fabric only
        if loose and not s["spine"]:
            d2[wco[:, 2] < pelvis_z] = 1e18        # loose column -> spine only
        D2[:, k] = d2
        T[:, k] = t
    # TORSO COLUMN OVERRIDE: fabric inside the torso column belongs to the
    # spine no matter how close a hanging A-pose forearm passes by it -
    # without this the hem/placket got claimed by the forearm and sagged
    # below the wrist (the 0.18-0.8 z outliers).
    if "pelvis" in jt_src:
        px, py = jt_src["pelvis"].x, jt_src["pelvis"].y
        t_r = radii.get("torso", 0.10 * span_g)
        rho = np.hypot(wco[:, 0] - px, wco[:, 1] - py)
        in_col = rho < 1.7 * t_r
        for k, s in enumerate(segs):
            if not s["spine"]:
                D2[in_col, k] = 1e18
    best = np.argmin(D2, axis=1)

    # ---- per segment: local frame, angle bins, inner-wall radius table ----
    dg = bpy.context.evaluated_depsgraph_get()
    bev = body.evaluated_get(dg)
    binv = body.matrix_world.inverted()
    b3 = binv.to_3x3()
    bmw = body.matrix_world
    NT, NA = 12, 16
    out = wco.copy()
    for k, s in enumerate(segs):
        idx = np.nonzero(best == k)[0]
        if len(idx) == 0:
            continue
        a0, b0, a1, b1 = s["a0"], s["b0"], s["a1"], s["b1"]
        y0 = (b0 - a0).normalized()
        y1 = (b1 - a1).normalized()
        R = y0.rotation_difference(y1)
        u0 = y0.orthogonal().normalized()
        v0 = y0.cross(u0)
        L0 = (b0 - a0).length
        L1 = (b1 - a1).length
        tt = T[idx, k]
        P = wco[idx]
        ax0 = np.array(a0[:]) + np.outer(tt, np.array(y0[:])) * L0
        rad = P - ax0
        ru = rad @ np.array(u0[:])
        rv = rad @ np.array(v0[:])
        rr = np.hypot(ru, rv)
        # CLEARANCE BASIS = the SOURCE MANNEQUIN's limb/torso radius, not the
        # fabric's own inner wall (that collapsed sleeves skin-tight: a
        # sleeve's designed looseness lives relative to the implied ARM)
        nmk = s["name"]
        if nmk == "spine1":
            r_src = np.full(len(idx), radii.get("torso", 0.10 * span_g))
        elif nmk == "spine2":
            rc = radii.get("chest", radii.get("torso", 0.10 * span_g))
            rn = radii.get("neck", 0.03 * span_g)
            r_src = rc + (rn - rc) * tt
        else:
            side = nmk[-1]
            r_src = np.full(len(idx),
                            radii.get("shoulder_%s" % side,
                                      radii.get("hip_%s" % side,
                                                0.05 * span_g)))
        # decode on the character
        for j, i in enumerate(idx):
            if rr[j] < 1e-9:
                out[i] = np.array(a1[:]) + np.array(y1[:]) * tt[j] * L1
                continue
            dir0 = Vector((rad[j][0], rad[j][1], rad[j][2])) / rr[j]
            dir1 = (R @ dir0)
            axis1 = a1 + y1 * (tt[j] * L1)
            clear = max(rr[j] - float(r_src[j]), 0.0)
            # ray from the character's bone outward to her REAL surface.
            # The ray STARTS INSIDE the body (on the bone axis) - the FIRST
            # hit is the surface we stand on. Taking the last hit grabbed the
            # OPPOSITE side of the torso and flung sleeves across like wings.
            o_l = binv @ axis1
            d_l = (b3 @ dir1).normalized()
            r_dst = None
            okc, loc, nrm, _fi = bev.ray_cast(o_l, d_l,
                                              distance=span_g * 2.0)
            if okc:
                r_dst = ((bmw @ loc) - axis1).length
            if r_dst is None:
                r_dst = float(r_src[j]) * gs       # off-body: implied radius
            out[i] = np.array((axis1 + dir1 * (r_dst + 0.004 * gs
                                               + clear * gs))[:])

    # heal segment seams: smooth the OFFSETS along mesh edges
    offs = out - wco
    ev = np.empty(2 * len(me.edges), dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    for _ in range(5):
        acc = np.zeros_like(offs); cnt = np.zeros(n)
        np.add.at(acc, ev[:, 0], offs[ev[:, 1]])
        np.add.at(cnt, ev[:, 0], 1.0)
        np.add.at(acc, ev[:, 1], offs[ev[:, 0]])
        np.add.at(cnt, ev[:, 1], 1.0)
        nz = cnt > 0
        offs[nz] = 0.5 * offs[nz] + 0.5 * (acc[nz] / cnt[nz, None])
    out = wco + offs

    # write to SRF_Fit (same reversible slot)
    from .garment import SK_FIT, K_KEYS
    if me.shape_keys is None:
        g_ob.shape_key_add(name="Basis", from_mix=False)
        g_ob[K_KEYS] = True
    sk = me.shape_keys.key_blocks.get(SK_FIT)
    if sk is None:
        sk = g_ob.shape_key_add(name=SK_FIT, from_mix=False)
    sk.slider_min = 0.0
    sk.value = 1.0
    inv = mw.inverted()
    for i in range(n):
        sk.data[i].co = inv @ Vector(out[i])
    return True


# ----------------------------------------------------- phase 2: retargeting --

# skeleton segments used by the warp (present ones only)
_SEGS = [("pelvis", "chest"), ("chest", "neck"),
         ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
         ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
         ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
         ("hip_r", "knee_r"), ("knee_r", "ankle_r")]


def _rig_joints(body):
    """EXACT joints from a Soulify rig when the character is rigged first -
    the recommended flow: Rig, then Fit (no AI guessing at all)."""
    try:
        from . import metarig as _mr
        rig = _mr._generated_rig() or bpy.data.objects.get("SR_Metarig")
        if rig is None or rig.type != 'ARMATURE':
            return None
        mw = rig.matrix_world
        bones = rig.pose.bones if rig.mode == 'POSE' else rig.pose.bones

        def head(*names):
            for nm in names:
                pb = rig.pose.bones.get(nm)
                if pb is not None:
                    return mw @ pb.head
            return None

        out = {}
        pairs = {
            "neck": ("neck", "neck_01", "spine.004", "spine.006"),
            "chest": ("spine.003", "chest", "spine_03", "spine.002"),
            "pelvis": ("spine", "spine_01", "root", "hips"),
            "shoulder_l": ("upper_arm.L", "upper_arm_fk.L", "upper_arm.l"),
            "elbow_l": ("forearm.L", "forearm_fk.L", "forearm.l"),
            "wrist_l": ("hand.L", "hand_fk.L", "hand.l"),
            "shoulder_r": ("upper_arm.R", "upper_arm_fk.R", "upper_arm.r"),
            "elbow_r": ("forearm.R", "forearm_fk.R", "forearm.r"),
            "wrist_r": ("hand.R", "hand_fk.R", "hand.r"),
            "hip_l": ("thigh.L", "thigh_fk.L"), "knee_l": ("shin.L", "shin_fk.L"),
            "ankle_l": ("foot.L", "foot_fk.L"),
            "hip_r": ("thigh.R", "thigh_fk.R"), "knee_r": ("shin.R", "shin_fk.R"),
            "ankle_r": ("foot.R", "foot_fk.R"),
        }
        for ours, cands in pairs.items():
            p = head(*cands)
            if p is not None:
                out[ours] = p
        return out if len(out) >= 8 else None
    except Exception as e:
        print("Soulify _rig_joints:", e)
        return None


def character_joints(body):
    """The character's joints. RIG FIRST (exact - the recommended flow: rig
    the character with Soulify, then Fit), pose net as the fallback."""
    rj = _rig_joints(body)
    if rj is not None:
        return rj
    try:
        from . import detect
        if not detect.available():
            return None
        hidden = []
        for o in bpy.context.scene.objects:
            if o.type == 'MESH' and o is not body and not o.hide_render:
                o.hide_render = True
                hidden.append(o)
        try:
            res = detect.detect(body)
        finally:
            for o in hidden:
                o.hide_render = False
        if not res:
            return None
        pts, kc = res["points"], res["kconf"]
        out = {}
        for ours, theirs in (("shoulder_l", "shoulder_l"), ("elbow_l", "elbow_l"),
                             ("wrist_l", "wrist_l"),
                             ("shoulder_r", "shoulder_r"), ("elbow_r", "elbow_r"),
                             ("wrist_r", "wrist_r"),
                             ("hip_l", "hip_l"), ("knee_l", "knee_l"),
                             ("ankle_l", "ankle_l"),
                             ("hip_r", "hip_r"), ("knee_r", "knee_r"),
                             ("ankle_r", "ankle_r")):
            if theirs in pts and kc.get(theirs, 0.0) >= 0.3:
                out[ours] = pts[theirs]
        # SIDES ARE GEOMETRIC, not semantic: the garment names its left tube by
        # world -x; the net names the CHARACTER's anatomical left (+x when
        # facing -Y). Mismatched names crossed the arms through the body.
        # Re-label every chain by the sign of its mean x.
        for chain in (("shoulder", "elbow", "wrist"), ("hip", "knee", "ankle")):
            lx = [out[j + "_l"].x for j in chain if j + "_l" in out]
            rx = [out[j + "_r"].x for j in chain if j + "_r" in out]
            if lx and rx and (sum(lx) / len(lx)) > (sum(rx) / len(rx)):
                for j in chain:
                    a, b = out.get(j + "_l"), out.get(j + "_r")
                    if a is not None and b is not None:
                        out[j + "_l"], out[j + "_r"] = b, a
        # torso joints must use the SAME definitions as the garment side:
        # neck = narrowest geometric neck (collar line), chest/pelvis by the
        # same shoulder-width proportions used in garment_skeleton.
        from .garment import body_profile
        z0b, z1b, Rb, Cb = body_profile(body)
        bhb = max(z1b - z0b, 1e-6)
        zn = min([z0b + (0.75 + 0.18 * k / 29.0) * bhb for k in range(30)],
                 key=lambda z: Rb(z))
        cN = Cb(zn)
        out["neck"] = Vector((cN.x, cN.y, zn))
        # SANITY: the net sometimes collapses shoulder keypoints to the centre
        # (this male body: shoulders at x~0 while elbows at +-0.4 -> shoulder
        # width 6 cm -> pelvis computed AT the neck -> total crush). A shoulder
        # must sit laterally between the neck and its elbow; rebuild it from
        # the elbow when it doesn't.
        for side in ("l", "r"):
            el = out.get("elbow_" + side)
            sh = out.get("shoulder_" + side)
            if el is None:
                continue
            lat_el = el.x - cN.x
            if sh is None or abs(sh.x - cN.x) < 0.30 * abs(lat_el):
                out["shoulder_" + side] = Vector((
                    cN.x + 0.45 * lat_el, el.y,
                    zn - 0.35 * max(zn - el.z, 0.0)))
        if "shoulder_l" in out and "shoulder_r" in out:
            sw = (out["shoulder_l"] - out["shoulder_r"]).length
            out["chest"] = Vector((cN.x, cN.y, zn - 0.55 * sw))
            out["pelvis"] = Vector((cN.x, cN.y, zn - 1.55 * sw))
        elif "pelvis" in pts:
            out["pelvis"] = pts["pelvis"]
        return out if len(out) >= 6 else None
    except Exception as e:
        print("Soulify mannequin character_joints:", e)
        return None


def warp_garment(g_ob, jt_src, jt_dst, loose=False, body=None):
    """BONE-PAIR SPACE WARP (the match): every garment vertex is skinned to
    the mannequin's skeleton segments by smooth inverse-distance weights; each
    segment carries its fabric with the rigid rotation+scale that maps the
    mannequin bone onto the character bone. Works for any pose difference
    (A->T etc.) because every bone is solved independently.
    loose=True (kandura/dress/skirt): fabric below the pelvis follows ONLY the
    spine segments - knees never drag the garment."""
    me = g_ob.data
    n = len(me.vertices)
    mw = g_ob.matrix_world
    base = [v.co.copy() for v in me.vertices]
    wco = np.array([(mw @ c)[:] for c in base], dtype=float)

    # GLOBAL radial scale: girth is a design property - it scales with overall
    # body size (shoulder width), NEVER with individual bone lengths (that
    # shrank the fabric onto the skin and crumpled it)
    if "shoulder_l" in jt_src and "shoulder_r" in jt_src \
            and "shoulder_l" in jt_dst and "shoulder_r" in jt_dst:
        gs = (jt_dst["shoulder_l"] - jt_dst["shoulder_r"]).length \
            / max((jt_src["shoulder_l"] - jt_src["shoulder_r"]).length, 1e-9)
    elif "pelvis" in jt_src and "chest" in jt_src \
            and "pelvis" in jt_dst and "chest" in jt_dst:
        gs = (jt_dst["chest"] - jt_dst["pelvis"]).length \
            / max((jt_src["chest"] - jt_src["pelvis"]).length, 1e-9)
    else:
        gs = 1.0

    segs = []
    for a, b in _SEGS:
        if a in jt_src and b in jt_src and a in jt_dst and b in jt_dst:
            a0, b0 = jt_src[a], jt_src[b]
            a1, b1 = jt_dst[a], jt_dst[b]
            v0, v1 = (b0 - a0), (b1 - a1)
            if v0.length < 1e-9 or v1.length < 1e-9:
                continue
            R = mathutils_matrix_to_np(v0.rotation_difference(v1).to_matrix())
            axial = v1.length / v0.length
            d0 = np.array((v0.normalized())[:])
            # anisotropic: bone-length ratio ALONG the bone, global size across
            S = gs * np.eye(3) + (axial - gs) * np.outer(d0, d0)
            is_spine = a in ("pelvis", "chest")
            segs.append((np.array(a0[:]), np.array(b0[:]),
                         R @ S, np.array(a1[:]), is_spine))
    if not segs:
        return False
    pelvis_z = jt_src["pelvis"].z if "pelvis" in jt_src else -1e9

    W = np.zeros((n, len(segs)))
    for k, (a0, b0, _, _, is_spine) in enumerate(segs):
        ab = b0 - a0
        L2 = float(ab.dot(ab)) + 1e-12
        t = np.clip(((wco - a0) @ ab) / L2, 0.0, 1.0)
        cl = a0 + t[:, None] * ab
        d2 = np.sum((wco - cl) ** 2, axis=1)
        W[:, k] = 1.0 / (d2 + 1e-8) ** 2.5          # local: no candy-wrapping
    if loose:
        below = wco[:, 2] < pelvis_z
        for k, (_, _, _, _, is_spine) in enumerate(segs):
            if not is_spine:
                W[below, k] = 0.0
    W /= (W.sum(axis=1, keepdims=True) + 1e-12)

    out = np.zeros_like(wco)
    for k, (a0, _, M, a1, _) in enumerate(segs):
        out += W[:, k:k + 1] * ((wco - a0) @ M.T + a1)

    # CLEANUP: per-segment scaling can shrink the girth slightly - push any
    # vert that landed inside the body back out along the surface normal,
    # then feather the pushes (Laplacian) so the fabric stays silky
    if body is not None:
        binv = body.matrix_world.inverted()
        bmw = body.matrix_world
        bco = utils.read_rest_coords(body)
        floor = 0.003 * max(float(bco[:, 2].max() - bco[:, 2].min()), 1e-6)
        push = np.zeros_like(out)
        for i in range(n):
            pl = binv @ Vector(out[i])
            okc, loc, nrm, _ = body.closest_point_on_mesh(pl)
            if not okc:
                continue
            lw = bmw @ loc
            sgn = 1.0 if (pl - loc).dot(nrm) >= 0.0 else -1.0
            d = sgn * (Vector(out[i]) - lw).length
            if d < floor:
                nw = (bmw.to_3x3() @ nrm).normalized()
                push[i] = np.array((lw + nw * floor)[:]) - out[i]
        ev = np.empty(2 * len(me.edges), dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        for _ in range(3):
            acc = np.zeros_like(push); cnt = np.zeros(n)
            np.add.at(acc, ev[:, 0], push[ev[:, 1]])
            np.add.at(cnt, ev[:, 0], 1.0)
            np.add.at(acc, ev[:, 1], push[ev[:, 0]])
            np.add.at(cnt, ev[:, 1], 1.0)
            nz = cnt > 0
            push[nz] = 0.5 * push[nz] + 0.5 * (acc[nz] / cnt[nz, None])
        out += push
        # strict second pass after feathering
        for i in range(n):
            pl = binv @ Vector(out[i])
            okc, loc, nrm, _ = body.closest_point_on_mesh(pl)
            if okc:
                lw = bmw @ loc
                sgn = 1.0 if (pl - loc).dot(nrm) >= 0.0 else -1.0
                if sgn * (Vector(out[i]) - lw).length < floor * 0.6:
                    nw = (bmw.to_3x3() @ nrm).normalized()
                    out[i] = np.array((lw + nw * floor)[:])

    # write to the SRF_Fit shape key (same reversible slot as Let's Fit)
    from .garment import SK_FIT, K_KEYS
    if me.shape_keys is None:
        g_ob.shape_key_add(name="Basis", from_mix=False)
        g_ob[K_KEYS] = True
    sk = me.shape_keys.key_blocks.get(SK_FIT)
    if sk is None:
        sk = g_ob.shape_key_add(name=SK_FIT, from_mix=False)
    sk.slider_min = 0.0
    sk.value = 1.0
    inv = mw.inverted()
    for i in range(n):
        sk.data[i].co = inv @ Vector(out[i])
    return True


def mathutils_matrix_to_np(M):
    return np.array([[M[0][0], M[0][1], M[0][2]],
                     [M[1][0], M[1][1], M[1][2]],
                     [M[2][0], M[2][1], M[2][2]]])


class SMARTRIG_OT_mannequin_match(bpy.types.Operator):
    """THE MATCH: extract the garment's implied skeleton, detect the
    character's joints (any pose), and warp the garment bone-by-bone onto the
    character. Kandura/dresses/skirts stay knee-safe (spine-bound)"""
    bl_idname = "smartrig.mannequin_match"
    bl_label = "Match to Character"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        g_ob = props.garment_object
        body = props.fit_body_object
        if g_ob is None or body is None:
            self.report({'ERROR'}, "Pick the garment and the body first.")
            return {'CANCELLED'}
        jt = garment_skeleton(g_ob)
        if jt is None:
            self.report({'ERROR'}, "Could not read the garment's structure.")
            return {'CANCELLED'}
        dst = character_joints(body)
        if dst is None:
            self.report({'ERROR'},
                        "Character joints not detected (onnxruntime/model?).")
            return {'CANCELLED'}
        # register the standard fit keys so Drape / Remove / sliders work
        from .garment import K_ORIG, K_BASE, K_BODY, K_BODYH
        from . import utils as _u
        bco = _u.read_rest_coords(body)
        if K_ORIG not in g_ob:
            g_ob[K_ORIG] = [v for row in g_ob.matrix_world for v in row]
        g_ob[K_BASE] = [v for row in g_ob.matrix_world for v in row]
        g_ob[K_BODY] = body.name
        g_ob[K_BODYH] = float(bco[:, 2].max() - bco[:, 2].min())
        loose = jt.get("lower_mode") == 'LOOSE'
        # ONE CLICK = the proven warp puts the garment ON the character;
        # THEN a real armature is built on the WORN state (bones on the
        # character's joints, garment skinned to them at rest) - so nothing
        # moves until the USER grabs a bone: instant, GPU-live hand-tweaking.
        # DEFAULT = the stable warp. The surface-coordinate retarget (the true
        # professional engine) is behind a flag until its clearance basis is
        # switched to the mannequin limb radii (see LESSONS: sleeves collapse
        # skin-tight when clearance is measured from the fabric's own inner
        # wall). Enable per scene: scene["srf_experimental_retarget"] = True
        # STABLE DEFAULT = the warp. The MetaTailor-style ride (bind to a
        # frozen mannequin, morph the mannequin) is gated experimental: its
        # remaining gap is Surface-Deform binding quality across a sparse
        # mannequin surface (needs a denser mannequin + falloff pass - see
        # LESSONS). Enable: scene["srf_experimental_retarget"] = True
        ok = False
        if context.scene.get("srf_experimental_retarget"):
            try:
                ok = retarget_ride(context, g_ob, jt, dst, body, loose=loose)
            except Exception as e:
                print("Soulify retarget_ride failed:", e)
        if not ok:
            for mn in ("SRF_Ride", "SRF_Clean"):
                m = g_ob.modifiers.get(mn)
                if m:
                    g_ob.modifiers.remove(m)
            ok = warp_garment(g_ob, jt, dst, loose=loose, body=body)
        if not ok:
            self.report({'ERROR'}, "No matching skeleton segments.")
            return {'CANCELLED'}
        try:
            dg = context.evaluated_depsgraph_get()
            ev = g_ob.evaluated_get(dg).to_mesh()
            wco = np.array([(g_ob.matrix_world @ ev.vertices[i].co)[:]
                            for i in range(len(ev.vertices))], dtype=float)
            g_ob.evaluated_get(dg).to_mesh_clear()
            build_garment_rig(g_ob, dst, loose=loose, coords=wco)
        except Exception as e:
            print("Soulify garment rig:", e)
        g_ob["srf_info"] = "%s matched to %s (%d joints, live rig)" % (
            jt.get("label", "garment"), body.name, len(dst))
        self.report({'INFO'}, g_ob["srf_info"])
        return {'FINISHED'}


class SMARTRIG_OT_garment_mannequin(bpy.types.Operator):
    """Phase 1 of Mannequin Retargeting: extract the garment's implied
    skeleton and build a procedural mannequin WEARING the garment"""
    bl_idname = "smartrig.garment_mannequin"
    bl_label = "Build Garment Mannequin"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        g_ob = props.garment_object
        if g_ob is None or g_ob.type != 'MESH':
            self.report({'ERROR'}, "Pick the garment mesh first.")
            return {'CANCELLED'}
        jt = garment_skeleton(g_ob)
        if jt is None:
            self.report({'ERROR'}, "Could not read the garment's structure.")
            return {'CANCELLED'}
        ob = build_mannequin(jt)
        ob["srf_joints_base"] = dict(ob["srf_joints"])   # for live sliders
        limbs = sum(1 for k in jt if k.startswith(("shoulder", "hip")))
        self.report({'INFO'}, "Mannequin built (%d limb roots)." % limbs)
        context.view_layer.objects.active = ob
        return {'FINISHED'}


_classes = (SMARTRIG_OT_garment_mannequin, SMARTRIG_OT_mannequin_match)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
