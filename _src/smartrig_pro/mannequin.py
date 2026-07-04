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
    Radii come from the garment, so the flesh fills the clothes.
    v1.36.3 (Saeed): the mannequin reads as a PERSON now - head, mitt
    hands (no fingers), full legs and feet are synthesized proportionally
    whenever the garment itself does not imply them."""
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)

    jt = dict(jt)                        # never mutate the caller's dict
    rad = dict(jt.get("radii") or {})
    jt["radii"] = rad
    up = Vector((0.0, 0.0, 1.0))
    fwd = Vector((0.0, -1.0, 0.0))       # characters face -Y here
    T = (jt["neck"] - jt["pelvis"]).length \
        if ("neck" in jt and "pelvis" in jt) else 0.5
    r_nk = float(rad.get("neck", 0.05))
    t_r = float(rad.get("torso", 0.1))
    # HEAD
    if "neck" in jt:
        jt["head"] = jt["neck"] + up * (0.18 * T)
        jt["head_top"] = jt["neck"] + up * (0.34 * T)
        rad["head"] = 1.7 * r_nk
        rad["head_top"] = 1.15 * r_nk
    # MITT HANDS
    for sd in ("l", "r"):
        w, e = jt.get("wrist_" + sd), jt.get("elbow_" + sd)
        if w is not None and e is not None and (w - e).length > 1e-6:
            d = (w - e).normalized()
            jt["hand_" + sd] = w + d * (0.38 * (w - e).length)
    # FULL LEGS when the garment implies none (a shirt): the ankle
    # reaches the GROUND (characters stand at z~0 in this pipeline),
    # not a torso-proportional guess that left hockey-stick shins
    if "hip_l" not in jt and "pelvis" in jt:
        sw_g = (jt["shoulder_l"] - jt["shoulder_r"]).length \
            if ("shoulder_l" in jt and "shoulder_r" in jt) else 2.2 * t_r
        for sd, sgn in (("l", -1.0), ("r", 1.0)):
            hx = jt["pelvis"] + Vector((sgn * max(0.30 * sw_g,
                                                  0.75 * t_r), 0.0, 0.0))
            ankle_z = max(0.045, 0.05 * hx.z)
            jt["hip_" + sd] = hx
            jt["knee_" + sd] = Vector((hx.x, hx.y,
                                       0.5 * (hx.z + ankle_z) + 0.02))
            jt["ankle_" + sd] = Vector((hx.x, hx.y, ankle_z))
    # FEET
    for sd in ("l", "r"):
        a = jt.get("ankle_" + sd)
        if a is not None:
            k_ = jt.get("knee_" + sd)
            fl = 0.5 * (k_ - a).length if k_ is not None else 0.14 * T
            jt["toe_" + sd] = a + fwd * fl + Vector((0.0, 0.0,
                                                     -0.25 * 0.8 * t_r))

    order = ["pelvis", "chest", "neck"]
    if "head" in jt:
        order += ["head", "head_top"]
    for side in ("l", "r"):
        for j in ("shoulder", "elbow", "wrist", "hand",
                  "hip", "knee", "ankle", "toe"):
            k = "%s_%s" % (j, side)
            if k in jt:
                order.append(k)
    idx = {k: i for i, k in enumerate(order)}
    verts = [jt[k] for k in order]
    edges = [(idx["pelvis"], idx["chest"]), (idx["chest"], idx["neck"])]
    if "head" in idx:
        edges.append((idx["neck"], idx["head"]))
        edges.append((idx["head"], idx["head_top"]))
    for side in ("l", "r"):
        for a, b, root in (("shoulder", "elbow", "chest"),
                           ("elbow", "wrist", None),
                           ("wrist", "hand", None),
                           ("hip", "knee", "pelvis"),
                           ("knee", "ankle", None),
                           ("ankle", "toe", None)):
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
        elif k == "head":
            r = rad.get("head", base * 0.8)
        elif k == "head_top":
            r = rad.get("head_top", base * 0.55)
        else:
            j = k.split("_")[0]
            r = rad.get("shoulder_" + k[-1], base * 0.45) \
                if j in ("shoulder", "elbow", "wrist", "hand") \
                else base * 0.6
            if j in ("elbow", "wrist"):
                r *= 0.85
            if j == "hip":
                r = base * 0.8
            if j == "knee":
                r = base * 0.62
            if j == "ankle":
                r = base * 0.45
            if j == "hand":
                ob.data.skin_vertices[0].data[i].radius = (r * 1.05,
                                                           r * 0.45)
                continue
            if j == "toe":
                r0 = base * 0.6 * 0.8
                ob.data.skin_vertices[0].data[i].radius = (r0 * 1.05,
                                                           r0 * 0.4)
                continue
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
        # after a MATCH, hand control belongs to SRF_GarmentRig (Pose Mode) -
        # the slider warp would fight it and re-tent the garment from design
        matched = bpy.data.objects.get(RIG_NAME) is not None
        if not neutral and not matched:            # pre-match shaping only
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
    # VOXELBIND (v1.36.0): part-aware voxel weights - sleeve fabric only
    # receives arm heat, torso only spine, geodesic through fabric with the
    # body as an obstacle, feathered seams. Measured reason: Euclidean
    # weights left 40% of torso fabric ARM-dominated (vhd.exe: 35% - voxel
    # heat alone is part-blind). Fallback = the old segment weights.
    W = None
    try:
        from . import voxelbind
        W = voxelbind.weights(g_ob, wco, jt, names, segs, loose=loose)
        if W is not None:
            print("Soulify VoxelBind: %d verts x %d bones" % W.shape)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Soulify VoxelBind failed -> segment weights:", e)
        W = None
    if W is None:
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
        # THE GENERATED RIG FIRST: its ORG- bones carry the character's
        # CURRENT POSE (arms down, bent, anything). The metarig only knows
        # the rest pose - matching to it while the body is posed detaches
        # the garment from the visible character (v1.28.0 bug).
        bpy.context.view_layer.update()
        rig = _mr._generated_rig() or bpy.data.objects.get("SR_Metarig")
        if rig is None or rig.type != 'ARMATURE':
            return None
        mw = rig.matrix_world

        def head(*names):
            for nm in names:
                pb = rig.pose.bones.get(nm)
                if pb is not None:
                    return mw @ pb.head
            return None

        out = {}
        pairs = {
            "neck": ("ORG-spine.004", "neck", "neck_01", "spine.004",
                     "spine.006"),
            "chest": ("ORG-spine.003", "spine.003", "chest", "spine_03",
                      "spine.002"),
            "pelvis": ("ORG-spine", "spine", "spine_01", "hips"),
            "shoulder_l": ("ORG-upper_arm.L", "upper_arm.L",
                           "upper_arm_fk.L", "upper_arm.l"),
            "elbow_l": ("ORG-forearm.L", "forearm.L", "forearm_fk.L",
                        "forearm.l"),
            "wrist_l": ("ORG-hand.L", "hand.L", "hand_fk.L", "hand.l"),
            "shoulder_r": ("ORG-upper_arm.R", "upper_arm.R",
                           "upper_arm_fk.R", "upper_arm.r"),
            "elbow_r": ("ORG-forearm.R", "forearm.R", "forearm_fk.R",
                        "forearm.r"),
            "wrist_r": ("ORG-hand.R", "hand.R", "hand_fk.R", "hand.r"),
            "hip_l": ("ORG-thigh.L", "thigh.L", "thigh_fk.L"),
            "knee_l": ("ORG-shin.L", "shin.L", "shin_fk.L"),
            "ankle_l": ("ORG-foot.L", "foot.L", "foot_fk.L"),
            "hip_r": ("ORG-thigh.R", "thigh.R", "thigh_fk.R"),
            "knee_r": ("ORG-shin.R", "shin.R", "shin_fk.R"),
            "ankle_r": ("ORG-foot.R", "foot.R", "foot_fk.R"),
        }
        for ours, cands in pairs.items():
            p = head(*cands)
            if p is not None:
                out[ours] = p
        # SIDES ARE GEOMETRIC here too: the rig's .L is the character's
        # anatomical left (+x facing -Y) while the garment labels world -x
        for chain in (("shoulder", "elbow", "wrist"),
                      ("hip", "knee", "ankle")):
            lx = [out[j + "_l"].x for j in chain if j + "_l" in out]
            rx = [out[j + "_r"].x for j in chain if j + "_r" in out]
            if lx and rx and (sum(lx) / len(lx)) > (sum(rx) / len(rx)):
                for j in chain:
                    a, b = out.get(j + "_l"), out.get(j + "_r")
                    if a is not None and b is not None:
                        out[j + "_l"], out[j + "_r"] = b, a
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
        if not detect.has_model():
            return None
        if not detect.has_runtime():
            # ONE CLICK, ANY CHARACTER (v1.35.0): install on demand
            detect.ensure_runtime()
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


# ---------------------------------------------------------------------------
#  DESIGN PRESERVATION (v1.28.0) - the audit fix.
#  The v1.27.7 audit proved LBS position-blending smears local shape:
#  48% of edges distorted beyond +-25%. Two engines fix it:
#    1. STIFF PANELS: small loose components (buttons, buckles, detached
#       cuffs/collars/ornaments) move as ONE rigid body each - never
#       per-vertex - so designed hardware stays crisp.
#    2. ARAP FINISH: iterative local-similarity projection. Every vertex
#       one-ring is pulled back onto a rotated + uniformly-scaled copy of
#       its DESIGN shape (batch Kabsch/Horn via np.linalg.svd), softly
#       constrained to the warped placement. Kills shear/smear, keeps fit.
# ---------------------------------------------------------------------------

def _edge_array(me):
    ev = np.empty(2 * len(me.edges), dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    return ev.reshape(-1, 2)


def _loose_components(n, ev):
    """Connected components (union-find). Returns list of index arrays."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in ev:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(g, dtype=np.int64) for g in groups.values()]


def _rigid_snap(base, cur, c):
    """Best rigid(+uniform scale) transform of the DESIGN onto cur (Kabsch
    + Horn scale) for index set c; writes cur in-place."""
    P, Q = base[c], cur[c]
    pc, qc = P.mean(0), Q.mean(0)
    dP, dQ = P - pc, Q - qc
    U, S, Vt = np.linalg.svd(dP.T @ dQ)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.array([1.0, 1.0, d])
    A = (U * D) @ Vt
    s = float((S * D).sum() / max((dP * dP).sum(), 1e-12))
    cur[c] = qc + (dP @ A) * s


def rigidify_components(base, cur, comps, max_frac=0.05, extra=None):
    """Snap every SMALL loose component - plus every USER-REGISTERED rigid
    cluster (belt/pockets/buttons via the SRF_Rigid group) - to one rigid
    transform of its DESIGN shape. Returns bool mask of stiff verts."""
    n = base.shape[0]
    stiff = np.zeros(n, dtype=bool)
    if len(comps) > 1:
        big = max(len(c) for c in comps)
        for c in comps:
            if len(c) == big or len(c) > max_frac * n or len(c) < 3:
                continue
            _rigid_snap(base, cur, c)
            stiff[c] = True
    for c in (extra or []):
        if len(c) >= 3:
            _rigid_snap(base, cur, c)
            stiff[c] = True
    return stiff


def _group_mask(g_ob, me, name):
    """Bool mask of a registered part group (weight > .5), or None."""
    vg = g_ob.vertex_groups.get(name)
    if vg is None:
        return None
    gi = vg.index
    m = np.zeros(len(me.vertices), dtype=bool)
    for v in me.vertices:
        for ge in v.groups:
            if ge.group == gi and ge.weight > 0.5:
                m[v.index] = True
                break
    return m if m.any() else None


def _rigid_group_clusters(g_ob, me, ev, name="SRF_Rigid"):
    """Connected clusters of the user's SRF_Rigid vertex group (Fit Wizard
    'register extras' step): each cluster moves as ONE rigid body."""
    vg = g_ob.vertex_groups.get(name)
    if vg is None:
        return []
    gi = vg.index
    tag = np.zeros(len(me.vertices), dtype=bool)
    for v in me.vertices:
        for ge in v.groups:
            if ge.group == gi and ge.weight > 0.5:
                tag[v.index] = True
                break
    if tag.sum() < 3:
        return []
    import collections
    adj = collections.defaultdict(list)
    for a, b in ev:
        a, b = int(a), int(b)
        if tag[a] and tag[b]:
            adj[a].append(b)
            adj[b].append(a)
    seen = set()
    clusters = []
    for i in np.nonzero(tag)[0]:
        i = int(i)
        if i in seen:
            continue
        q, comp = [i], []
        seen.add(i)
        while q:
            x = q.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        if len(comp) >= 3:
            clusters.append(np.array(comp, dtype=np.int64))
    return clusters


def arap_refine(base, warped, ev, iters=6, lam=0.10, s_lo=0.5, s_hi=2.0,
                pin=None):
    """ARAP-style finish pass (local-global with Jacobi updates).
    base/warped: (n,3) world coords. pin: verts held exactly at warped."""
    n = base.shape[0]
    if n == 0 or len(ev) == 0:
        return warped
    ctr = np.concatenate([ev[:, 0], ev[:, 1]])
    nbr = np.concatenate([ev[:, 1], ev[:, 0]])
    deg = np.zeros(n)
    np.add.at(deg, ctr, 1.0)
    ok = deg > 0
    pc = np.zeros((n, 3))
    np.add.at(pc, ctr, base[nbr])
    pc[ok] /= deg[ok, None]
    dP = base[nbr] - pc[ctr]
    pnorm = np.zeros(n)
    np.add.at(pnorm, ctr, np.sum(dP * dP, axis=1))
    cur = warped.copy()
    for _ in range(iters):
        qc = np.zeros((n, 3))
        np.add.at(qc, ctr, cur[nbr])
        qc[ok] /= deg[ok, None]
        dQ = cur[nbr] - qc[ctr]
        C = np.zeros((n, 3, 3))
        np.add.at(C, ctr, dP[:, :, None] * dQ[:, None, :])
        U, S, Vt = np.linalg.svd(C[ok])
        d = np.sign(np.linalg.det(np.matmul(U, Vt)))
        D = np.ones((U.shape[0], 3))
        D[:, 2] = d
        A = np.matmul(U * D[:, None, :], Vt)          # dP @ A ~= dQ
        s = (S * D).sum(axis=1) / (pnorm[ok] + 1e-12)
        np.clip(s, s_lo, s_hi, out=s)
        Rf = np.zeros((n, 3, 3))
        Rf[ok] = A * s[:, None, None]
        # prediction of every vertex from each neighbour's frame + its own
        pred = qc[nbr] + np.einsum('ij,ijk->ik', base[ctr] - pc[nbr], Rf[nbr])
        acc = np.zeros((n, 3))
        np.add.at(acc, ctr, pred)
        own = qc + np.einsum('ij,ijk->ik', base - pc, Rf)
        w = deg + 1.0 + lam * deg
        cur[ok] = ((acc + own + (lam * deg)[:, None] * warped)[ok]
                   / w[ok, None])
        if pin is not None:
            cur[pin] = warped[pin]
    return cur


def _adjacency(n, ev):
    """Flat one-ring arrays + degree + a fixed reference neighbour per
    vertex (lowest index) for building consistent tangent frames."""
    ctr = np.concatenate([ev[:, 0], ev[:, 1]])
    nbr = np.concatenate([ev[:, 1], ev[:, 0]])
    deg = np.zeros(n)
    np.add.at(deg, ctr, 1.0)
    ref = np.full(n, n, dtype=np.int64)
    np.minimum.at(ref, ctr, nbr)
    ref[ref >= n] = 0
    return ctr, nbr, deg, ref


def _smooth_base(co, ctr, nbr, deg, iters=25, alpha=0.5, mu=-0.53):
    """Low-frequency base of the design via TAUBIN smoothing (lambda|mu):
    plain Laplacian SHRINKS open boundaries (hem/cuffs pulled in -> the
    detail layer explodes there and re-adds as flared skirts / sleeves on
    hands). Taubin's negative step cancels the shrink."""
    cur = co.copy()
    ok = deg > 0
    dn = np.maximum(deg, 1.0)[:, None]
    for i in range(iters * 2):
        step = alpha if (i % 2 == 0) else mu
        acc = np.zeros_like(cur)
        np.add.at(acc, ctr, cur[nbr])
        lap = acc / dn - cur
        cur[ok] += step * lap[ok]
    return cur


def _ring_normals_radius(co, ctr, nbr, deg, align):
    """Batch one-ring PCA: unit normal (smallest eigenvector, sign-aligned
    with 'align') + mean ring radius, per vertex."""
    n = co.shape[0]
    ok = deg > 0
    c = np.zeros((n, 3))
    np.add.at(c, ctr, co[nbr])
    c[ok] /= deg[ok, None]
    d = co[nbr] - c[ctr]
    C = np.tile(1e-12 * np.eye(3), (n, 1, 1))
    np.add.at(C, ctr, d[:, :, None] * d[:, None, :])
    _, v_ = np.linalg.eigh(C)
    nrm = v_[:, :, 0]
    flip = np.sum(nrm * align, axis=1) < 0.0
    nrm[flip] = -nrm[flip]
    rad = np.zeros(n)
    np.add.at(rad, ctr, np.linalg.norm(d, axis=1))
    rad[ok] /= deg[ok]
    return nrm, rad


def _frames(co, nrm, ref):
    """Orthonormal (t, b, n) rows per vertex; tangent = reference-neighbour
    direction projected off the normal."""
    t = co[ref] - co
    t -= np.sum(t * nrm, axis=1, keepdims=True) * nrm
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    bad = ln[:, 0] < 1e-9
    if bad.any():
        alt = np.cross(nrm[bad], np.array([1.0, 0.0, 0.0]))
        t[bad] = alt
        ln = np.linalg.norm(t, axis=1, keepdims=True)
    t /= np.maximum(ln, 1e-12)
    b = np.cross(nrm, t)
    return np.stack([t, b, nrm], axis=1)


def edge_distortion(g_ob, band=0.25):
    """DESIGN-DAMAGE metric: % of edges whose final length deviates more
    than +-band from the length predicted by the INTENDED local affine
    (fitted per one-ring on the smoothed base). A designed resize -
    isotropic or anisotropic tailoring - does NOT count; only genuine
    destruction of local detail does."""
    me = g_ob.data
    if me.shape_keys is None or len(me.edges) == 0:
        return None
    from .garment import SK_FIT
    kb = me.shape_keys.key_blocks
    fit = kb.get(SK_FIT)
    if fit is None:
        return None
    nv = len(me.vertices)
    a = np.empty(nv * 3)
    kb[0].data.foreach_get("co", a)
    b = np.empty(nv * 3)
    fit.data.foreach_get("co", b)
    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)
    ev = _edge_array(me)
    ctr, nbr, deg, _ = _adjacency(nv, ev)
    sa = _smooth_base(a, ctr, nbr, deg)
    sb = _smooth_base(b, ctr, nbr, deg)
    ok = deg > 0
    ca = np.zeros((nv, 3))
    np.add.at(ca, ctr, sa[nbr])
    ca[ok] /= deg[ok, None]
    cb = np.zeros((nv, 3))
    np.add.at(cb, ctr, sb[nbr])
    cb[ok] /= deg[ok, None]
    dP = sa[nbr] - ca[ctr]
    dQ = sb[nbr] - cb[ctr]
    PtP = np.tile(1e-9 * np.eye(3), (nv, 1, 1))
    PtQ = np.zeros((nv, 3, 3))
    np.add.at(PtP, ctr, dP[:, :, None] * dP[:, None, :])
    np.add.at(PtQ, ctr, dP[:, :, None] * dQ[:, None, :])
    A = np.linalg.solve(PtP, PtQ)          # row-vector: q = p @ A
    e0 = a[ev[:, 1]] - a[ev[:, 0]]
    Ae = 0.5 * (np.einsum('ij,ijk->ik', e0, A[ev[:, 0]])
                + np.einsum('ij,ijk->ik', e0, A[ev[:, 1]]))
    lp = np.linalg.norm(Ae, axis=1)
    lf = np.linalg.norm(b[ev[:, 1]] - b[ev[:, 0]], axis=1)
    m = (lp > 1e-9) & (np.linalg.norm(e0, axis=1) > 1e-9)
    r = lf[m] / lp[m]
    return float(np.mean((r > 1.0 + band) | (r < 1.0 - band)) * 100.0)


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

    # WRIST = the true end of the sleeve: the tube trace can stop a few cm
    # short of the cuff, and everything beyond the source wrist extrapolates
    # (x axial scale) - that is how sleeves ended up riding the hands.
    # Extend each source wrist to the farthest fabric along the forearm axis.
    for side in ("l", "r"):
        e = jt_src.get("elbow_" + side)
        w_ = jt_src.get("wrist_" + side)
        if e is None or w_ is None:
            continue
        L = (w_ - e).length
        if L < 1e-9:
            continue
        dv = (w_ - e) / L
        dn_ = np.array(dv[:])
        rel = wco - np.array(e[:])
        t = rel @ dn_
        rho = np.linalg.norm(rel - t[:, None] * dn_, axis=1)
        # NARROW tube + hard cap: a wide/uncapped mask caught the HEM near
        # the forearm ray, dragged the source wrist to the hem and crushed
        # the sleeve toward the elbow (v1.28.1 first attempt)
        m_ = (t > 0.8 * L) & (t < 1.4 * L) & (rho < 0.35 * L)
        if m_.any():
            tmax = min(float(t[m_].max()), 1.35 * L)
            if tmax > L:
                jt_src["wrist_" + side] = e + dv * tmax

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

    # WIDTH MARKERS (Fit Wizard): chest/waist girth control. When present,
    # each torso band gets its own radial scale = body width at that height
    # / the user-confirmed garment width (clamped around gs so the design
    # girth is never destroyed).
    sc_chest = sc_waist = None
    if body is not None and any(k in jt_src for k in
                                ("chest_w_l", "waist_w_l",
                                 "chest_d_f", "waist_d_f")):
        try:
            # read_rest_coords is already WORLD - the old extra
            # matrix multiply broke the band fallback on rotated bodies
            bw_ = utils.read_rest_coords(body)
            bh_ = float(bw_[:, 2].max() - bw_[:, 2].min())

            # ARMS EXCLUSION (the old A-pose lesson): at waist height the
            # HANDS sit in the band and explode the measured width - keep
            # only the torso column around the joint
            sw_ = (jt_dst["shoulder_l"] - jt_dst["shoulder_r"]).length \
                if ("shoulder_l" in jt_dst and "shoulder_r" in jt_dst) \
                else 0.25 * bh_

            _mcol = bpy.data.collections.get("SRF_FitMarkers")

            def _band_scale(kl, kr, zkey, axis, spankey=None):
                a_, b_ = jt_src.get(kl), jt_src.get(kr)
                if a_ is None or b_ is None or zkey not in jt_dst:
                    return None
                w_g = (a_ - b_).length
                if w_g < 1e-6:
                    return None
                # MORPH TARGETS: marker span (where the user says the
                # fabric must sit) / the garment's design span.
                # ONE-CLICK AUTO (v1.35.0): headless spans travel inside
                # jt_src["spans"] and WIN over a possibly stale marker
                # collection from an older wizard run.
                auto = (jt_src.get("spans") or {}).get(spankey) \
                    if spankey else None
                if not auto and _mcol is not None and spankey:
                    auto = _mcol.get("srf_span_" + spankey)
                if auto:
                    return float(np.clip(w_g / float(auto), 0.6, 1.6))
                j = jt_dst[zkey]
                band = (np.abs(bw_[:, 2] - j.z) < 0.02 * bh_) \
                    & (np.abs(bw_[:, 0] - j.x) < 0.55 * sw_) \
                    & (np.abs(bw_[:, 1] - j.y) < 0.55 * sw_)
                if band.sum() < 8:
                    return None
                w_b = float(bw_[band, axis].max() - bw_[band, axis].min())
                return float(np.clip(w_b / w_g, 0.8 * gs, 1.3 * gs))

            # width (x) and depth (y) morphs are fully INDEPENDENT per band
            sc_chest_w = _band_scale("chest_w_l", "chest_w_r", "chest",
                                     0, "chest_w")
            sc_chest_d = _band_scale("chest_d_f", "chest_d_b", "chest",
                                     1, "chest_d")
            sc_waist_w = _band_scale("waist_w_l", "waist_w_r", "pelvis",
                                     0, "waist_w")
            sc_waist_d = _band_scale("waist_d_f", "waist_d_b", "pelvis",
                                     1, "waist_d")
            sc_chest = (sc_chest_w, sc_chest_d)
            sc_waist = (sc_waist_w, sc_waist_d)
        except Exception as e:
            print("Soulify width markers:", e)

    segs = []
    seg_keys = []
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
            # anisotropic: bone-length ratio ALONG the bone; radial = the
            # band's width-marker scale when available, else global size
            scw = scd = None
            if a == "pelvis" and sc_waist is not None:
                scw, scd = sc_waist
            elif a == "chest" and sc_chest is not None:
                scw, scd = sc_chest
            if scw is not None or scd is not None:
                # anisotropic band: width (world-x) and depth (world-y)
                # morph independently around the bone axis
                u = np.array([1.0, 0.0, 0.0]) - d0 * d0[0]
                u /= max(np.linalg.norm(u), 1e-9)
                v = np.cross(d0, u)
                S = axial * np.outer(d0, d0) \
                    + (scw if scw is not None else gs) * np.outer(u, u) \
                    + (scd if scd is not None else gs) * np.outer(v, v)
            else:
                S = gs * np.eye(3) + (axial - gs) * np.outer(d0, d0)
            is_spine = a in ("pelvis", "chest")
            segs.append((np.array(a0[:]), np.array(b0[:]),
                         R @ S, np.array(a1[:]), is_spine, R))
            seg_keys.append(a)
    if not segs:
        return False
    pelvis_z = jt_src["pelvis"].z if "pelvis" in jt_src else -1e9

    W = np.zeros((n, len(segs)))
    for k, (a0, b0, _, _, is_spine, _) in enumerate(segs):
        ab = b0 - a0
        L2 = float(ab.dot(ab)) + 1e-12
        t = np.clip(((wco - a0) @ ab) / L2, 0.0, 1.0)
        cl = a0 + t[:, None] * ab
        d2 = np.sum((wco - cl) ** 2, axis=1)
        W[:, k] = 1.0 / (d2 + 1e-8) ** 2.5          # local: no candy-wrapping
    if loose:
        below = wco[:, 2] < pelvis_z
        for k, (_, _, _, _, is_spine, _) in enumerate(segs):
            if not is_spine:
                W[below, k] = 0.0
    # TORSO-COLUMN OVERRIDE (v1.31.3): side-torso fabric equidistant to the
    # arm and spine segments was dragged outward by the arm rotation =
    # underarm WINGS. Fabric inside the torso column follows the spine only.
    if "pelvis" in jt_src and "neck" in jt_src:
        p0 = np.array(jt_src["pelvis"][:])
        n0 = np.array(jt_src["neck"][:])
        ax = n0 - p0
        L2x = float(ax.dot(ax)) + 1e-12
        t_ = np.clip(((wco - p0) @ ax) / L2x, 0.0, 1.0)
        rho_ = np.linalg.norm(wco - (p0 + t_[:, None] * ax), axis=1)
        radii_ = jt_src.get("radii") or {}
        t_r = float(radii_.get("torso") or radii_.get("chest") or 0.0)
        if t_r <= 0.0 and "shoulder_l" in jt_src and "shoulder_r" in jt_src:
            t_r = 0.5 * (jt_src["shoulder_l"] - jt_src["shoulder_r"]).length
        if t_r > 0.0:
            core = rho_ < 1.45 * t_r
            m_sl0 = _group_mask(g_ob, me, "SRF_Sleeve")
            if m_sl0 is not None:
                core &= ~m_sl0     # registered sleeves are never torso
            for k, (_, _, _, _, is_spine, _) in enumerate(segs):
                if not is_spine:
                    W[core, k] = 0.0
    # REGISTERED PARTS (Fit Wizard 4/4): the part carries its own binding -
    # no garment-type guessing. Sleeve->arms, Collar->neck seg, Lower->spine.
    def _restrict(mask, keep):
        cols = [k for k, ka in enumerate(seg_keys) if keep(ka)]
        if mask is None or not cols:
            return
        for k in range(len(seg_keys)):
            if k not in cols:
                W[mask, k] = 0.0
    _restrict(_group_mask(g_ob, me, "SRF_Sleeve"),
              lambda ka: ka in ("shoulder_l", "elbow_l",
                                "shoulder_r", "elbow_r"))
    _restrict(_group_mask(g_ob, me, "SRF_Collar"), lambda ka: ka == "chest")
    _restrict(_group_mask(g_ob, me, "SRF_Lower"),
              lambda ka: ka in ("pelvis", "chest"))
    W /= (W.sum(axis=1, keepdims=True) + 1e-12)

    # ---- DESIGN PRESERVATION (v1.28.0): detail-layer transfer ----
    # Decompose the DESIGN into a smooth base + high-frequency detail held
    # in local tangent frames. Only the BASE is warped/cleaned (its stretch
    # is intended tailoring, spread smoothly); the detail - wrinkles, seams,
    # collar crispness - is re-added untouched afterwards. Small loose parts
    # (buttons/ornaments) stay perfectly rigid.
    has_topo = len(me.edges) > 0
    if has_topo:
        ev_all = _edge_array(me)
        ctr, nbr, deg, ref = _adjacency(n, ev_all)
        comps = _loose_components(n, ev_all)
        smooth = _smooth_base(wco, ctr, nbr, deg)
        detail = wco - smooth
        n_align = np.empty(n * 3)
        me.vertices.foreach_get("normal", n_align)
        R3 = mathutils_matrix_to_np(mw.to_3x3())
        n_align = n_align.reshape(-1, 3) @ R3.T
        n_align /= np.maximum(
            np.linalg.norm(n_align, axis=1, keepdims=True), 1e-12)
        nrm0, rad0 = _ring_normals_radius(smooth, ctr, nbr, deg, n_align)
        F0 = _frames(smooth, nrm0, ref)
        bnd_ramp, bnd_parent, bnd_order = _boundary_info(me, n)
        if bnd_order:
            for v in bnd_order:           # copy interior frames outward
                p_ = bnd_parent[v]
                if p_ >= 0:
                    F0[v] = F0[p_]
        dloc = np.einsum('ijk,ik->ij', F0, detail)
        src = smooth
    else:
        ev_all = np.zeros((0, 2), np.int64)
        comps, dloc = [], None
        src = wco

    out = np.zeros_like(wco)
    for k, (a0, _, M, a1, _, _) in enumerate(segs):
        out += W[:, k:k + 1] * ((src - a0) @ M.T + a1)

    # ---- COLLAR RING MAPPING (v1.36.3, Saeed: collar place was wrong) --
    # the generic segment warp leaves the neck OPENING wide and low, on
    # the chest slope. A collar is a RING: map it explicitly onto the
    # character's NECK COLUMN - same design angles, the body's measured
    # neck radius (+6mm ease) at each height, design height offsets
    # scaled by the upper-spine ratio - then PIN it so ARAP blends the
    # yoke fabric smoothly into the ring.
    col_pin = np.zeros(n, dtype=bool)
    try:
        if body is not None and "neck" in jt_src and "neck" in jt_dst \
                and jt_src.get("label") not in ("skirt", "pants"):
            c0 = np.array(jt_src["neck"][:])
            mg = _group_mask(g_ob, me, "SRF_Collar")
            m_col = mg if mg is not None else (wco[:, 2] > c0[2] - 1e-9)
            # only the TRUE collar band rides the ring: fabric radially
            # near the design neck opening (shoulder-top yoke fabric also
            # sits above the neck line - vacuuming it onto the neck ring
            # would strip the shoulders)
            r_nk = float((jt_src.get("radii") or {}).get("neck") or 0.0)
            if r_nk > 0.0:
                rho_d = np.hypot(wco[:, 0] - c0[0], wco[:, 1] - c0[1])
                m_col = m_col & (rho_d < 2.2 * r_nk)
            if m_col.sum() >= 8:
                cN = np.array(jt_dst["neck"][:])
                if "chest" in jt_src and "chest" in jt_dst:
                    L0 = np.linalg.norm(c0 - np.array(jt_src["chest"][:]))
                    L1 = np.linalg.norm(cN - np.array(jt_dst["chest"][:]))
                    sz = float(np.clip(L1 / max(L0, 1e-9), 0.5, 2.0))
                else:
                    sz = 1.0
                rel = wco[m_col] - c0
                th = np.arctan2(rel[:, 1], rel[:, 0])
                hh = rel[:, 2] * sz
                sw_d = (jt_dst["shoulder_l"]
                        - jt_dst["shoulder_r"]).length \
                    if ("shoulder_l" in jt_dst
                        and "shoulder_r" in jt_dst) else 0.36
                bw2 = utils.read_rest_coords(body)
                hs = np.linspace(float(hh.min()), float(hh.max()), 16)
                rs = np.full(16, np.nan)
                for k2 in range(16):
                    zt = cN[2] + hs[k2]
                    band = np.abs(bw2[:, 2] - zt) < 0.012
                    if band.sum() < 6:
                        band = np.abs(bw2[:, 2] - zt) < 0.03
                    if band.sum() < 6:
                        continue
                    rr = np.hypot(bw2[band, 0] - cN[0],
                                  bw2[band, 1] - cN[1])
                    rr = rr[rr < 0.35 * sw_d]
                    if len(rr) >= 4:
                        # the NECK is the innermost ring around the axis:
                        # at collar heights the traps/shoulders also fall
                        # in the band and an 85th pct read 13cm - use a
                        # LOW percentile and a hard sanity cap
                        rs[k2] = min(np.percentile(rr, 25),
                                     0.30 * sw_d)
                oks = ~np.isnan(rs)
                if oks.sum() >= 3 and oks.mean() > 0.4:
                    r_t = np.interp(hh, hs[oks], rs[oks]) + 0.006
                    idx2 = np.nonzero(m_col)[0]
                    out[idx2, 0] = cN[0] + np.cos(th) * r_t
                    out[idx2, 1] = cN[1] + np.sin(th) * r_t
                    out[idx2, 2] = cN[2] + hh
                    col_pin[idx2] = True
    except Exception as e:
        print("Soulify collar ring:", e)

    # AUTO-created collars never rigidify (v1.36.0): unconfirmed rigid
    # clusters disagree with neighbours and TEAR the shoulders; the chest
    # restriction still holds the collar. Wizard/user parts stay rigid.
    extra = []
    if has_topo:
        extra = _rigid_group_clusters(g_ob, me, ev_all)
        if not g_ob.get("srf_auto_parts"):
            extra = extra + _rigid_group_clusters(g_ob, me, ev_all,
                                                  "SRF_Collar")
    stiff = rigidify_components(wco, out, comps, extra=extra) \
        if (comps or extra) else np.zeros(n, dtype=bool)
    if has_topo:
        # heavy ARAP on the base (it is smooth, so this is safe): relaxes
        # LBS blend-zone stretch/pinch into evenly distributed tailoring
        out = arap_refine(src, out, ev_all, iters=30, lam=0.05,
                          s_lo=0.7, s_hi=1.6, pin=(stiff | col_pin))

    # CLEANUP: per-segment scaling can shrink the girth slightly - push any
    # vert that landed inside the body back out along the surface normal,
    # then feather the pushes (Laplacian) so the fabric stays silky
    if body is not None:
        # collisions against the EVALUATED body: the character may be POSED
        # (armature/subsurf) - the rest mesh lies (documented lesson)
        dg_ = bpy.context.evaluated_depsgraph_get()
        bev = body.evaluated_get(dg_)
        binv = bev.matrix_world.inverted()
        bmw = bev.matrix_world
        bco = utils.read_rest_coords(body)
        floor = 0.003 * max(float(bco[:, 2].max() - bco[:, 2].min()), 1e-6)
        # per-vertex floor: reserve room for detail that dips inward, so the
        # re-added wrinkles/seams never end up inside the body
        if dloc is not None:
            # capped: an unreliable boundary frame must never balloon the
            # clearance (that flared hems into skirts in v1.28.0)
            fl = floor + np.minimum(np.maximum(0.0, -dloc[:, 2]), floor)
        else:
            fl = np.full(n, floor)
        push = np.zeros_like(out)
        for i in range(n):
            pl = binv @ Vector(out[i])
            okc, loc, nrm, _ = bev.closest_point_on_mesh(pl)
            if not okc:
                continue
            lw = bmw @ loc
            sgn = 1.0 if (pl - loc).dot(nrm) >= 0.0 else -1.0
            d = sgn * (Vector(out[i]) - lw).length
            if d < fl[i]:
                nw = (bmw.to_3x3() @ nrm).normalized()
                push[i] = np.array((lw + nw * fl[i])[:]) - out[i]
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
            okc, loc, nrm, _ = bev.closest_point_on_mesh(pl)
            if okc:
                lw = bmw @ loc
                sgn = 1.0 if (pl - loc).dot(nrm) >= 0.0 else -1.0
                if sgn * (Vector(out[i]) - lw).length < fl[i] * 0.6:
                    nw = (bmw.to_3x3() @ nrm).normalized()
                    out[i] = np.array((lw + nw * fl[i])[:])

    # ---- RE-ADD THE DETAIL on the cleaned base ----
    if has_topo and dloc is not None:
        # the base must stay SMOOTH: a mild relax removes residual LBS /
        # collision jitter that would otherwise masquerade as detail
        hold = stiff | col_pin
        keep = out[hold].copy() if hold.any() else None
        out = _smooth_base(out, ctr, nbr, deg, iters=4, alpha=0.3)
        if keep is not None:
            out[hold] = keep
        # transport the design normals through the segment rotations to
        # sign-align the warped-base PCA normals
        n_tr = np.zeros((n, 3))
        for k, (_, _, _, _, _, R) in enumerate(segs):
            n_tr += W[:, k:k + 1] * (n_align @ R.T)
        n_tr /= np.maximum(np.linalg.norm(n_tr, axis=1, keepdims=True), 1e-12)
        nrm1, rad1 = _ring_normals_radius(out, ctr, nbr, deg, n_tr)
        F1 = _frames(out, nrm1, ref)
        if bnd_order:
            for v in bnd_order:           # same transport as encode
                p_ = bnd_parent[v]
                if p_ >= 0:
                    F1[v] = F1[p_]
        # ORIENTATION CONTINUITY (ChatGPT: many spikes are a literal 180
        # flip): a decode frame that disagrees with the rotation-
        # transported encode frame flips back
        flip = np.einsum('ij,ij->i', F1[:, :, 2] if F1.ndim == 3 else F1, nrm0) < 0.0 if False else None  # noqa
        s_loc = np.clip(rad1 / np.maximum(rad0, 1e-12), 0.85, 1.2)
        out = out + np.einsum('ij,ijk->ik',
                              dloc * (s_loc * bnd_ramp)[:, None], F1)
        # stiff panels: exact rigid copies of the DESIGN
        rigidify_components(wco, out, comps, extra=extra)
        # final safety: only true penetrations move - detail stays crisp.
        # v1.36.1: lift to the FULL floor and run twice (0.35x once left
        # upper-chest fabric inside the body = ragged intersection)
        if body is not None:
            # capped full-depth projections, closest point recomputed
            # every pass (deep pushes with stale normals overshoot
            # across thin regions like the armpit)
            e_len = np.linalg.norm(wco[ev_all[:, 0]] - wco[ev_all[:, 1]],
                                   axis=1).mean() if len(ev_all) else floor
            # wide enough to clear a deep chest penetration in 3
            # passes, tight enough to never jump across the armpit
            cap = max(2.0 * float(e_len), 3.0 * floor)
            for _pass in range(4):
                # passes 0-2 capped; pass 3 = the DEFINITE push (Gemini):
                # concave pockets (clavicle/armpit) ping-pong under a cap
                definite = _pass == 3
                moved = False
                for i in range(n):
                    pl = binv @ Vector(out[i])
                    okc, loc, nrm, _ = bev.closest_point_on_mesh(pl)
                    if okc:
                        lw = bmw @ loc
                        sgn = 1.0 if (pl - loc).dot(nrm) >= 0.0 else -1.0
                        if sgn * (Vector(out[i]) - lw).length < floor * 0.35:
                            nw = (bmw.to_3x3() @ nrm).normalized()
                            want = np.array((lw + nw * floor)[:])
                            step = want - out[i]
                            sl = float(np.linalg.norm(step))
                            if not definite and sl > cap:
                                step *= cap / sl
                            out[i] = out[i] + step
                            moved = True
                if not moved:
                    break
    elif body is not None and stiff.any():
        rigidify_components(wco, out, comps, extra=extra)

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


def _boundary_info(me, n, rings=3):
    """OPEN-BOUNDARY handling (v1.36.1-2, consensus of our measurements +
    DeepSeek/ChatGPT/Gemini review): one-ring PCA tangent frames are
    ill-conditioned on boundary half-rings (collar/cuffs/hem/placket) and
    flip - re-added detail SPIKED the collar edge. Returns:
      ramp   - soft detail fade (0.4 edge / 0.75 / 0.95 / 1.0) - soft
               because the frames get TRANSPORTED (below), not trusted
      parent - for each vert within `rings` of the boundary, a neighbour
               one ring FARTHER from the edge: interior frames are copied
               outward along these links instead of estimating PCA on a
               half ring (ChatGPT/Gemini: frame propagation)."""
    cnt = {}
    for p in me.polygons:
        vs = p.vertices
        m = len(vs)
        for k in range(m):
            a, b = vs[k], vs[(k + 1) % m]
            e = (a, b) if a < b else (b, a)
            cnt[e] = cnt.get(e, 0) + 1
    front = sorted({v for e, c in cnt.items() if c == 1 for v in e})
    parent = np.full(n, -1, dtype=np.int64)
    if not front:
        return np.ones(n), parent, None
    adj = {}
    for (a, b) in cnt:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    d = np.full(n, float(rings))
    seen = set(front)
    for v in front:
        d[v] = 0.0
    cur = front
    for r in range(1, rings):
        nxt = []
        for v in cur:
            for w in adj.get(v, ()):
                if w not in seen:
                    seen.add(w)
                    d[w] = float(r)
                    nxt.append(w)
        cur = nxt
    # parent = a neighbour strictly farther from the open edge
    order = []
    for v in range(n):
        if d[v] < rings:
            best, bd = -1, d[v]
            for w in adj.get(v, ()):
                if d[w] > bd:
                    best, bd = w, d[w]
            parent[v] = best
            order.append(v)
    order.sort(key=lambda v: -d[v])       # farthest first: chains resolve
    fade = {0.0: 0.4, 1.0: 0.75, 2.0: 0.95}
    ramp = np.ones(n)
    for v in range(n):
        if d[v] < rings:
            ramp[v] = fade.get(d[v], 1.0)
    return ramp, parent, order


def mathutils_matrix_to_np(M):
    return np.array([[M[0][0], M[0][1], M[0][2]],
                     [M[1][0], M[1][1], M[1][2]],
                     [M[2][0], M[2][1], M[2][2]]])


# ---------------------------------------------------------------------------
#  ONE-CLICK AUTO MODE (v1.35.0) - "Fit to Character" with zero setup.
#  The Fit Wizard's pre-fill intelligence (size morph targets, part
#  registration) runs HEADLESSLY when the user never opened the wizard,
#  and empty pickers are filled automatically. Manual picks and wizard
#  markers always win over the automation.
# ---------------------------------------------------------------------------

def auto_pick(props, context):
    """Fill empty Garment/Body pickers automatically. Body = the rig-tab
    target mesh, else the biggest rig-skinned mesh, else the biggest visible
    mesh. Garment = the user's selected/active mesh that is not the body,
    else the unskinned mesh whose bounding box overlaps the body the most
    (tiny meshes like eyes/teeth are filtered by height)."""
    g = props.garment_object
    b = props.fit_body_object
    if g is not None and b is not None:
        return g, b
    meshes = [o for o in context.scene.objects
              if o.type == 'MESH' and o.visible_get()]

    def _vol(o):
        d = o.dimensions
        return float(d.x * d.y * d.z)

    def _skinned(o):
        for m in o.modifiers:
            if m.type == 'ARMATURE' and m.object is not None:
                return True
        return o.parent is not None and o.parent.type == 'ARMATURE'

    if b is None:
        tm = props.target_mesh
        if tm is not None and tm is not g and tm.type == 'MESH':
            b = tm
        else:
            pool = [o for o in meshes if o is not g and _skinned(o)] \
                or [o for o in meshes if o is not g]
            b = max(pool, key=_vol) if pool else None
    if g is None and b is not None:
        bb = [b.matrix_world @ Vector(c) for c in b.bound_box]
        b0 = Vector((min(p.x for p in bb), min(p.y for p in bb),
                     min(p.z for p in bb)))
        b1 = Vector((max(p.x for p in bb), max(p.y for p in bb),
                     max(p.z for p in bb)))
        min_h = 0.08 * max(b1.z - b0.z, 1e-6)

        def _overlap(o):
            cc = [o.matrix_world @ Vector(c) for c in o.bound_box]
            o0 = [min(p[i] for p in cc) for i in range(3)]
            o1 = [max(p[i] for p in cc) for i in range(3)]
            v = 1.0
            for i, (lo, hi) in enumerate(((b0.x, b1.x), (b0.y, b1.y),
                                          (b0.z, b1.z))):
                v *= max(min(hi, o1[i]) - max(lo, o0[i]), 0.0)
            return v

        def _world_h(o):
            cc = [o.matrix_world @ Vector(c) for c in o.bound_box]
            return max(p.z for p in cc) - min(p.z for p in cc)

        cands = [o for o in meshes
                 if o is not b and not _skinned(o)
                 and _world_h(o) >= min_h]
        act = context.view_layer.objects.active
        sel = [o for o in cands if o.select_get()]
        if act is not None and act in sel:
            g = act                     # explicit user intent first
        elif len(sel) == 1:
            g = sel[0]
        else:
            pool = sel or cands
            scored = [(o, _overlap(o)) for o in pool]
            inside = [x for x in scored if x[1] > 0.0]
            scored = inside or scored
            g = max(scored, key=lambda x: x[1])[0] if scored else None
    return g, b


def auto_size_targets(jt, dst, body, g_ob=None):
    """The wizard's chest/waist width+depth MORPH TARGETS, headless.
    Design span = the garment's OWN band extent at the joint height
    (arms excluded) - NOT 2x the percentile ring radius: the percentile
    underestimates the true width, the ratio overshot the 1.6 clip and
    BALLOONED the chest (v1.35.0 screenshot bug). Target span = the
    CHARACTER's band extent + wearing ease, then CONSERVATIVELY clamped
    around gs*design (0.8-1.3, the proven fallback clamp) so a loose
    design stays loose and nothing balloons. Existing keys (wizard
    markers) are never overwritten."""
    if g_ob is None:
        return
    radii = jt.get("radii") or {}
    t_r = float(radii.get("torso") or radii.get("chest") or 0.0)
    # read_rest_coords already returns WORLD coords - transforming again
    # garbled the space on rotated bodies (bands came back empty)
    bw_ = utils.read_rest_coords(body)
    bh_ = float(bw_[:, 2].max() - bw_[:, 2].min())
    sw_ = (dst["shoulder_l"] - dst["shoulder_r"]).length \
        if ("shoulder_l" in dst and "shoulder_r" in dst) else 0.25 * bh_
    ease = 0.03 * bh_                    # wizard snap: half += 0.015*bh
    gco = utils.read_rest_coords(g_ob)
    # global scale, same formula as the warp, for the conservative clamp
    if all(k in jt for k in ("shoulder_l", "shoulder_r")) \
            and all(k in dst for k in ("shoulder_l", "shoulder_r")):
        gs = (dst["shoulder_l"] - dst["shoulder_r"]).length \
            / max((jt["shoulder_l"] - jt["shoulder_r"]).length, 1e-9)
    elif all(k in jt for k in ("pelvis", "chest")) \
            and all(k in dst for k in ("pelvis", "chest")):
        gs = (dst["chest"] - dst["pelvis"]).length \
            / max((jt["chest"] - jt["pelvis"]).length, 1e-9)
    else:
        gs = 1.0

    # SYMMETRIC DEFINITIONS: the garment torso box uses the GARMENT's own
    # shoulder span exactly like the body box uses the body's (a t_r-sized
    # box capped the measurable width at the box itself and re-ballooned)
    sw_g = (jt["shoulder_l"] - jt["shoulder_r"]).length \
        if ("shoulder_l" in jt and "shoulder_r" in jt) else None

    def _design(jkey, axis):
        j = jt[jkey]
        m = np.abs(gco[:, 2] - j.z) < 0.02 * bh_
        if sw_g:           # sleeves live beyond the shoulder roots
            m &= (np.abs(gco[:, 0] - j.x) < 0.55 * sw_g) \
                & (np.abs(gco[:, 1] - j.y) < 0.55 * sw_g)
        if m.sum() < 8:
            return None
        return float(gco[m, axis].max() - gco[m, axis].min())

    def _target(zkey, axis):
        j = dst.get(zkey)
        if j is None:
            return None
        # ARMS EXCLUSION (v1.31.2 lesson): keep the torso column only
        band = (np.abs(bw_[:, 2] - j.z) < 0.02 * bh_) \
            & (np.abs(bw_[:, 0] - j.x) < 0.55 * sw_) \
            & (np.abs(bw_[:, 1] - j.y) < 0.55 * sw_)
        if band.sum() < 8:
            return None
        return float(bw_[band, axis].max() - bw_[band, axis].min()) + ease

    spans = jt.setdefault("spans", {})
    for span_key, zkey, jkey, kl, kr, axis, d in (
            ("chest_w", "chest", "chest",
             "chest_w_l", "chest_w_r", 0, Vector((1.0, 0.0, 0.0))),
            ("chest_d", "chest", "chest",
             "chest_d_f", "chest_d_b", 1, Vector((0.0, 1.0, 0.0))),
            ("waist_w", "pelvis", "pelvis",
             "waist_w_l", "waist_w_r", 0, Vector((1.0, 0.0, 0.0))),
            ("waist_d", "pelvis", "pelvis",
             "waist_d_f", "waist_d_b", 1, Vector((0.0, 1.0, 0.0)))):
        if jkey not in jt or kl in jt or kr in jt:
            continue
        des = _design(jkey, axis)
        tgt = _target(zkey, axis)
        if des is None or tgt is None or des < 1e-6:
            continue
        tgt = float(np.clip(tgt, 0.8 * gs * des, 1.3 * gs * des))
        c = jt[jkey]
        jt[kl] = c - d * (0.5 * tgt)
        jt[kr] = c + d * (0.5 * tgt)
        spans[span_key] = des


def auto_part_groups(g_ob, jt):
    """The wizard's part pre-fill (Sleeve/Collar/Lower), headless.
    Sleeve = fabric outside the torso column above the pelvis, Collar =
    above the neck joint, Lower = below the pelvis (LOOSE columns only -
    a pants' legs must keep following the leg bones). Groups the user or
    wizard already registered are NEVER touched."""
    pel, nk = jt.get("pelvis"), jt.get("neck")
    if pel is None or nk is None:
        return
    me = g_ob.data
    nv = len(me.vertices)
    if nv == 0:
        return
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    R3 = np.array(g_ob.matrix_world.to_3x3())
    w = co.reshape(-1, 3) @ R3.T + np.array(g_ob.matrix_world.translation[:])
    p = np.array(pel[:])
    nkv = np.array(nk[:])
    ax = nkv - p
    L2 = float(ax @ ax) + 1e-12
    tt = np.clip(((w - p) @ ax) / L2, 0.0, 1.0)
    rho = np.linalg.norm(w - (p + tt[:, None] * ax), axis=1)
    radii = jt.get("radii") or {}
    t_r = float(radii.get("torso") or radii.get("chest")
                or 0.22 * math.sqrt(L2))

    fills = {
        "SRF_Sleeve": (rho > 1.45 * t_r) & (w[:, 2] > p[2]),
        "SRF_Collar": w[:, 2] > nkv[2],
        "SRF_Lower": (w[:, 2] < p[2])
        if jt.get("lower_mode") == 'LOOSE' else None,
    }
    for nm, m in fills.items():
        if m is None or g_ob.vertex_groups.get(nm) is not None:
            continue                     # user/wizard already decided
        if not m.any():
            continue
        vg = g_ob.vertex_groups.new(name=nm)
        vg.add([int(i) for i in np.nonzero(m)[0]], 1.0, 'REPLACE')
        g_ob["srf_auto_parts"] = 1       # auto parts: no collar rigidify


class SMARTRIG_OT_mannequin_match(bpy.types.Operator):
    """THE MATCH: extract the garment's implied skeleton, detect the
    character's joints (any pose), and warp the garment bone-by-bone onto the
    character. Kandura/dresses/skirts stay knee-safe (spine-bound)"""
    bl_idname = "smartrig.mannequin_match"
    bl_label = "Match to Character"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        # ONE CLICK, ZERO SETUP (v1.35.0): empty pickers fill themselves
        g_ob, body = auto_pick(props, context)
        if g_ob is None or body is None:
            self.report({'ERROR'},
                        "No garment/character found - select the garment "
                        "or pick both with the eyedroppers.")
            return {'CANCELLED'}
        # write the auto choice back so the UI shows what was matched
        if props.garment_object is not g_ob:
            props.garment_object = g_ob
        if props.fit_body_object is not body:
            props.fit_body_object = body
        jt = garment_skeleton(g_ob)
        if jt is None:
            self.report({'ERROR'}, "Could not read the garment's structure.")
            return {'CANCELLED'}
        # FIT WIZARD OVERRIDE: user-corrected marker empties beat the
        # automatic analysis (same philosophy as the rig marker wizard)
        auto = True
        try:
            from . import fit_wizard as _fw
            mj = _fw.marker_joints()
            if mj:
                jt.update(mj)
                auto = False
        except Exception as e:
            print("Soulify fit wizard markers:", e)
        dst = character_joints(body)
        if dst is None:
            self.report({'ERROR'},
                        "Character joints not detected - rig the character "
                        "(Rig tab) or check the console (onnxruntime).")
            return {'CANCELLED'}
        if auto:
            # ONE-CLICK AUTO: the wizard's pre-fill intelligence, headless
            try:
                auto_size_targets(jt, dst, body, g_ob)
            except Exception as e:
                print("Soulify auto size:", e)
            try:
                auto_part_groups(g_ob, jt)
            except Exception as e:
                print("Soulify auto parts:", e)
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
        # sliders are OFFSETS from the freshly-built base: stale values from a
        # previous garment ballooned the mannequin (Neck -0.5 / Volume 1.4 on
        # rebuild). Neutralize on every build.
        props.mann_arm_open = 0.0
        props.mann_elbow_bend = 0.0
        props.mann_neck_len = 0.0
        props.mann_torso_vol = 1.0
        props.mann_arm_vol = 1.0
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
