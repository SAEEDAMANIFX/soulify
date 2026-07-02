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
    ob["srf_joints"] = {k: list(v) for k, v in jt.items()
                        if isinstance(v, Vector)}
    ob["srf_lower_mode"] = jt.get("lower_mode", 'NONE')
    ob["srf_free_legs"] = bool(jt.get("free_legs", False))
    ob["srf_label"] = jt.get("label", "?")
    return ob


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
        limbs = sum(1 for k in jt if k.startswith(("shoulder", "hip")))
        self.report({'INFO'}, "Mannequin built (%d limb roots)." % limbs)
        context.view_layer.objects.active = ob
        return {'FINISHED'}


_classes = (SMARTRIG_OT_garment_mannequin,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
