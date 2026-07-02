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
        # the limb ROOT: where the tube meets the torso - lateral edge of the
        # torso at just-below-neck height (sleeves) or the pelvis (legs)
        is_leg = c.z < z0 + 0.35 * span and abs(c.x - cx) < 0.30 * span
        if is_leg:
            root = Vector((cx + sgn * jt["radii"]["torso"] * 0.55, cy,
                           jt["pelvis"].z))
            names = ("hip", "knee", "ankle")
        else:
            root = Vector((cx + sgn * torso_r(z_neck - 0.12 * span) * 0.9,
                           cy, z_neck - 0.10 * span))
            names = ("shoulder", "elbow", "wrist")
        tip = Vector(c)
        mid = root.lerp(tip, 0.5)
        jt["%s_%s" % (names[0], side)] = root
        jt["%s_%s" % (names[1], side)] = mid
        jt["%s_%s" % (names[2], side)] = tip
        jt["radii"]["%s_%s" % (names[0], side)] = r * 0.75

    _limb("l", left)
    _limb("r", right)
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
    ob["srf_joints"] = {k: list(v) for k, v in jt.items() if k != "radii"}
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
