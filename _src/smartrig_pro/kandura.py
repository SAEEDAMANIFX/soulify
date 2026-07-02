"""Kandura (Emirati thobe) module — region REGISTRATION step.

Beginner-friendly flow (mirrors the Short Skirt 'Register' idea, but for a
full thobe):
  1. Open the kandura mesh in Edit Mode.
  2. Alt+Click an edge loop, press the matching Register button:
       - Waist  : one horizontal loop at the waist   -> SR_KAN_waist
       - Sleeves: the two cuff loops (both at once)  -> SR_KAN_sleeve_L / _R
                  (split automatically by world X sign)
       - Neck   : the neck-opening loop              -> SR_KAN_neck
  3. Each region shows a checkmark once registered.

The vertex groups are the smart-bone-placement anchors for the next step
(waist-down skirt grid, sleeve chains with roll-up, neck ring).
"""
import bpy
import bmesh

VG_WAIST = "SR_KAN_waist"
VG_SLEEVE_L = "SR_KAN_sleeve_L"
VG_SLEEVE_R = "SR_KAN_sleeve_R"
VG_NECK = "SR_KAN_neck"
VG_PLACKET = "SR_KAN_placket"
VG_BUTTONS = "SR_KAN_buttons"
VG_POCKET = "SR_KAN_pocket"
VG_SPOCKET_L = "SR_KAN_pocket_side_L"
VG_SPOCKET_R = "SR_KAN_pocket_side_R"

REGIONS = {
    'WAIST': (VG_WAIST,),
    'SLEEVES': (VG_SLEEVE_L, VG_SLEEVE_R),
    'NECK': (VG_NECK,),
    'PLACKET': (VG_PLACKET,),
    'BUTTONS': (VG_BUTTONS,),
    'POCKET': (VG_POCKET,),
    'SIDE_POCKETS': (VG_SPOCKET_L, VG_SPOCKET_R),
}


def kandura_object(context):
    """The kandura mesh: the explicit picker if set, else the mesh being edited,
    else the active mesh object."""
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


def region_registered(ob, region):
    """True if ALL vertex groups of the region exist and are non-empty."""
    if ob is None:
        return False
    for vg_name in REGIONS[region]:
        vg = ob.vertex_groups.get(vg_name)
        if vg is None:
            return False
        # non-empty check: any vertex referencing this group index
        gi = vg.index
        if not any(g.group == gi for v in ob.data.vertices for g in v.groups):
            return False
    return True


def _assign(ob, vg_name, idxs):
    vg = ob.vertex_groups.get(vg_name)
    if vg is None:
        vg = ob.vertex_groups.new(name=vg_name)
    else:
        vg.remove(range(len(ob.data.vertices)))
    if idxs:
        vg.add(idxs, 1.0, 'REPLACE')
    return len(idxs)


def button_objects():
    """Separate button meshes tagged via 'Register Button Objects'."""
    return [o for o in bpy.data.objects
            if o.type == 'MESH' and o.get("sr_kan_button")]


def buttons_registered(ob):
    """Buttons count as registered if EITHER the vertex group is filled
    (buttons merged into the kandura mesh) OR button objects are tagged."""
    return region_registered(ob, 'BUTTONS') or bool(button_objects())


def enabled_regions(props):
    """The region list adapts to what THIS kandura actually has."""
    regions = ['WAIST', 'SLEEVES']
    if getattr(props, "kandura_has_neck", True):
        regions.append('NECK')
    if getattr(props, "kandura_has_placket", False):
        regions.append('PLACKET')
    if getattr(props, "kandura_has_buttons", False):
        regions.append('BUTTONS')
    if getattr(props, "kandura_has_pocket", False):
        regions.append('POCKET')
    if getattr(props, "kandura_has_side_pockets", False):
        regions.append('SIDE_POCKETS')
    return regions


def all_registered(ob, props=None):
    if props is None:
        props = bpy.context.scene.smartrig
    for r in enabled_regions(props):
        if r == 'BUTTONS':
            if not buttons_registered(ob):
                return False
        elif not region_registered(ob, r):
            return False
    return True


def _goto_metarig_edit(context):
    """After registration: jump straight into the metarig in Edit Mode so the
    user continues rigging without any manual mode juggling."""
    try:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    try:
        bpy.ops.smartrig.back_to_metarig()
        return True
    except Exception:
        pass
    meta = bpy.data.objects.get("SR_Metarig")
    if meta is None:
        return False
    try:
        meta.hide_set(False)
        bpy.ops.object.select_all(action='DESELECT')
        meta.select_set(True)
        context.view_layer.objects.active = meta
        bpy.ops.object.mode_set(mode='EDIT')
        return True
    except Exception:
        return False




# ====================================================================
# AUTO-DETECT: body bones are the reference — no loops, no topology.
# Works with partial/open geometry (placket-cut collars, slit waists,
# messy triangulated meshes) because every region is computed from the
# garment's distance to the BODY BONES, never from edge loops.
# ====================================================================

def _bone_seg(arm_ob, names):
    """World-space (head, tail) of the first existing bone in `names`."""
    for n in names:
        b = arm_ob.data.bones.get(n)
        if b is not None:
            mw = arm_ob.matrix_world
            return (mw @ b.head_local, mw @ b.tail_local)
    return None


def _ref_armature():
    """Reference skeleton: the metarig (clean names) or the generated rig."""
    ob = bpy.data.objects.get("SR_Metarig")
    if ob is not None:
        return ob, ""
    for o in bpy.data.objects:
        if o.type == 'ARMATURE' and o.name.startswith("RIG-"):
            return o, "DEF-"
    return None, ""


def _pt_seg_t_d(p, a, b):
    """Parameter t along segment ab (0..1) + distance from p."""
    ab = b - a
    L2 = ab.length_squared
    if L2 < 1e-12:
        return 0.0, (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / L2))
    return t, (p - (a + ab * t)).length


def auto_detect_regions(context):
    """Fill ALL the SR_KAN_* vertex groups automatically from the body rig.
    Returns a dict of region -> vert count (or an 'error' key)."""
    props = context.scene.smartrig
    ob = kandura_object(context)
    if ob is None:
        return {"error": "No kandura mesh"}
    arm, pfx = _ref_armature()
    if arm is None:
        return {"error": "No metarig / rig in the scene (build the body first)"}

    def seg(names):
        return _bone_seg(arm, [pfx + n for n in names] + list(names))

    arm_L = [seg(["upper_arm.L"]), seg(["forearm.L"])]
    arm_R = [seg(["upper_arm.R"]), seg(["forearm.R"])]
    thigh = seg(["thigh.L"])
    neck = seg(["spine.004", "neck", "spine.005"])
    head_b = seg(["spine.006", "head"])
    spine_segs = [s for s in (seg(["spine"]), seg(["spine.001"]),
                              seg(["spine.002"]), seg(["spine.003"])) if s]
    # legs join the BODY group so the low hem never classifies as 'sleeve'
    # (with hanging arms, hem verts can sit closer to the forearm TIP than
    # to the spine - the thigh/shin segments win that contest instead)
    leg_segs = [s for s in (seg(["thigh.L"]), seg(["thigh.R"]),
                            seg(["shin.L"]), seg(["shin.R"])) if s]
    body_segs = spine_segs + leg_segs
    if not spine_segs or thigh is None:
        return {"error": "Missing spine/thigh bones on the reference rig"}

    mw = ob.matrix_world
    cos = [mw @ v.co for v in ob.data.vertices]
    n_all = len(cos)
    zs = sorted(c.z for c in cos)
    gz0, gz1 = zs[0], zs[-1]
    gh = max(1e-6, gz1 - gz0)

    # ---- classify: nearest structure (torso vs arm.L vs arm.R) ----
    def d_group(p, segs):
        best_t, best_d, best_i = 0.0, 1e9, 0
        for i, s in enumerate(segs):
            if s is None:
                continue
            t, d = _pt_seg_t_d(p, s[0], s[1])
            if d < best_d:
                best_t, best_d, best_i = t, d, i
        return best_t, best_d, best_i

    sleeve_L, sleeve_R = [], []      # (chain-param, index)
    torso_idx = []
    for i, p in enumerate(cos):
        tl, dl, il = d_group(p, arm_L)
        tr, dr, ir = d_group(p, arm_R)
        _, dt, _ = d_group(p, body_segs)
        if dl < dt and dl <= dr:
            sleeve_L.append((il + tl, i))
        elif dr < dt and dr < dl:
            sleeve_R.append((ir + tr, i))
        else:
            torso_idx.append(i)

    out = {}
    # ---- sleeves: the cuff = the furthest ~ring along the arm chain ----
    for name, lst in ((VG_SLEEVE_L, sleeve_L), (VG_SLEEVE_R, sleeve_R)):
        if len(lst) < 8:
            out[name] = 0
            continue
        lst.sort(key=lambda x: -x[0])
        tmax = lst[0][0]
        cuff = [i for t, i in lst if t > tmax - 0.08]
        out[name] = _assign(ob, name, cuff)

    # ---- waist: torso cross-section band at the hip (thigh head) height ----
    z_hip = thigh[0].z
    band = 0.018 * gh
    waist = [i for i in torso_idx if abs(cos[i].z - z_hip) < band]
    if len(waist) < 8:  # garment may sit differently; widen once
        band *= 2.0
        waist = [i for i in torso_idx if abs(cos[i].z - z_hip) < band]
    out[VG_WAIST] = _assign(ob, VG_WAIST, waist)

    # ---- neck / collar: cylinder around the NECK BONE axis ----
    if neck is not None:
        a = neck[0]
        b = head_b[1] if head_b is not None else neck[1]
        cand = []
        for i, p in enumerate(cos):
            if p.z < a.z - 0.06 * gh:
                continue
            t, d = _pt_seg_t_d(p, a, b)
            cand.append((d, i))
        if len(cand) >= 8:
            cand.sort(key=lambda x: x[0])
            # adaptive radius: median distance of the closest half, * 1.9
            med = cand[max(0, len(cand) // 4)][0]
            R = med * 1.9
            collar = [i for d, i in cand if d <= R]
            if collar:
                # NECKLINE = the LOWEST height-band of the collar territory
                czs = sorted(cos[i].z for i in collar)
                z_lo = czs[0]
                z_band = max(0.012 * gh, (czs[-1] - z_lo) * 0.22)
                neckline = [i for i in collar if cos[i].z < z_lo + z_band]
                out[VG_NECK] = _assign(ob, VG_NECK, neckline)
    # ---- placket: through the registered/tagged BUTTON objects if any ----
    btns = button_objects()
    if btns and getattr(props, "kandura_has_placket", False):
        bx = sorted(o.matrix_world.translation.x for o in btns)
        bys = sorted(o.matrix_world.translation.y for o in btns)
        bzs = sorted(o.matrix_world.translation.z for o in btns)
        cx, cy = bx[len(bx)//2], bys[len(bys)//2]
        z_top = max(bzs) + 0.03 * gh
        z_bot = min(bzs) - 0.03 * gh
        rad = 0.02 * gh
        line = [i for i, p in enumerate(cos)
                if abs(p.x - cx) < rad and abs(p.y - cy) < rad * 3
                and z_bot <= p.z <= z_top]
        if line:
            out[VG_PLACKET] = _assign(ob, VG_PLACKET, line)
    return out


class SMARTRIG_OT_kandura_autodetect(bpy.types.Operator):
    bl_idname = "smartrig.kandura_autodetect"
    bl_label = "Auto-Detect Regions"
    bl_description = ("Detect waist / sleeves / neck (and placket via the button "
                      "objects) automatically using the BODY BONES as reference - "
                      "no loop selection needed, works with open collars and "
                      "messy topology. Review the results, then fix any region "
                      "manually if needed")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return kandura_object(context) is not None

    def execute(self, context):
        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        res = auto_detect_regions(context)
        if was_edit:
            try:
                bpy.ops.object.mode_set(mode='EDIT')
            except Exception:
                pass
        if "error" in res:
            self.report({'ERROR'}, res["error"])
            return {'CANCELLED'}
        parts = ", ".join("%s: %d" % (k.replace("SR_KAN_", ""), v)
                          for k, v in res.items())
        ob = kandura_object(context)
        context.scene.smartrig.kandura_object = ob
        self.report({'INFO'}, "Auto-detected - " + parts)
        return {'FINISHED'}




# ====================================================================
# BUILD KANDURA RIG — waist-down reuses the battle-tested SKIRT engine
# (same bone names + tagging  ->  ALL existing skirt automations work:
#  collision, jiggle, region masters, smart weights, follow, anti-pen).
# Sleeves = tentacle chains along the arm axis.  Neck / pockets = anchor
# bones.  Everything lands in the METARIG, so Generate integrates it.
# ====================================================================

def _vg_indices(ob, vg_name):
    vg = ob.vertex_groups.get(vg_name)
    if vg is None:
        return []
    gi = vg.index
    return [v.index for v in ob.data.vertices
            if any(g.group == gi for g in v.groups)]


def _vg_centroid_z(ob, vg_name, cos):
    idxs = _vg_indices(ob, vg_name)
    if not idxs:
        return None, None
    from mathutils import Vector
    c = Vector((0, 0, 0))
    for i in idxs:
        c += cos[i]
    c /= len(idxs)
    zs = sorted(cos[i].z for i in idxs)
    return c, zs[len(zs) // 2]


def build_kandura(context):
    """Build the full kandura rig into the metarig. Returns (ok, message)."""
    from . import skirt as _sk
    from . import utils as _ut
    from mathutils import Vector
    import numpy as np
    props = context.scene.smartrig
    ob = kandura_object(context)
    if ob is None:
        return False, "No kandura mesh"
    if not region_registered(ob, 'WAIST'):
        return False, "Register (or Auto-Detect) the Waist first"
    mo = bpy.data.objects.get("SR_Metarig")
    if mo is None:
        return False, "Build the body metarig first"
    arm, pfx = _ref_armature()

    def seg(names):
        return _bone_seg(arm, [pfx + n for n in names] + list(names))

    arm_chains = {"L": [seg(["upper_arm.L"]), seg(["forearm.L"])],
                  "R": [seg(["upper_arm.R"]), seg(["forearm.R"])]}
    spine_segs = [s for s in (seg(["spine"]), seg(["spine.001"]),
                              seg(["spine.002"]), seg(["spine.003"])) if s]
    leg_segs = [s for s in (seg(["thigh.L"]), seg(["thigh.R"]),
                            seg(["shin.L"]), seg(["shin.R"])) if s]
    body_segs = spine_segs + leg_segs

    mw = ob.matrix_world
    cos = [mw @ v.co for v in ob.data.vertices]

    def d_group(p, segs):
        best_t, best_d, best_i = 0.0, 1e9, 0
        for i, s in enumerate(segs):
            if s is None:
                continue
            t, d = _pt_seg_t_d(p, s[0], s[1])
            if d < best_d:
                best_t, best_d, best_i = t, d, i
        return best_t, best_d, best_i

    # classify each vert once: sleeve L / sleeve R / body
    sleeve = {"L": [], "R": []}
    body_idx = []
    for i, p in enumerate(cos):
        tl, dl, il = d_group(p, arm_chains["L"])
        tr, dr, ir = d_group(p, arm_chains["R"])
        _, dt, _ = d_group(p, body_segs)
        if dl < dt and dl <= dr:
            sleeve["L"].append((il + tl, i))
        elif dr < dt and dr < dl:
            sleeve["R"].append((ir + tr, i))
        else:
            body_idx.append(i)

    _, waist_z = _vg_centroid_z(ob, VG_WAIST, cos)
    if waist_z is None:
        return False, "Waist region is empty - register it again"

    report = []
    if mo.get("sr_kandura"):
        remove_kandura_bones(mo)

    # ---------- 1) WAIST-DOWN: the proven skirt engine ----------
    rest = _ut.read_rest_coords(ob)          # (N,3) stable rest coords
    sleeve_set = set(i for _t, i in sleeve["L"]) | set(i for _t, i in sleeve["R"])
    lower = np.array([rest[i] for i in body_idx
                      if i not in sleeve_set and cos[i].z <= waist_z + 1e-4])
    if len(lower) < 24:
        return False, "Too few vertices below the waist (%d)" % len(lower)
    cols = max(4, int(props.skirt_columns))
    rows = max(3, int(props.skirt_rows))     # a kandura is long: 3+ rows
    fa = _sk._FRONT_ANG.get(getattr(props, "skirt_front_axis", '-Y'),
                            _sk._FRONT_ANG['-Y'])
    grid = _sk._skirt_grid(lower, cols, rows,
                           front_ang=(fa if not props.skirt_symmetric else fa))
    if not grid:
        return False, "Could not build the waist-down grid"
    _sk._emit_chains(mo, grid, rows)
    mo["sr_skirt_kind"] = "TUBE"
    mo["sr_skirt_method"] = "kandura"
    mo["sr_skirt_cols_built"] = len(grid)
    # downstream skirt tools (collision/jiggle/weights) need these:
    props.skirt_source = 'SEPARATE'
    props.skirt_object = ob
    report.append("waist-down: %d cols x %d rows" % (len(grid), rows))

    # ---------- 2) SLEEVES: tentacle chain per arm ----------
    n_up = max(1, int(getattr(props, "kandura_sleeve_upper", 2)))
    n_lo = max(1, int(getattr(props, "kandura_sleeve_lower", 2)))
    made_sleeves = []
    if region_registered(ob, 'SLEEVES'):
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        mo.hide_set(False)
        bpy.context.view_layer.objects.active = mo
        bpy.ops.object.mode_set(mode='EDIT')
        eb = mo.data.edit_bones
        for b in [b for b in eb if b.name.startswith("kan_sleeve.")]:
            eb.remove(b)
        for side in ("L", "R"):
            pts_t = sleeve[side]
            if len(pts_t) < 8:
                continue
            ts = [t for t, _i in pts_t]
            t0, t1 = min(ts), max(ts)
            if t1 - t0 < 1e-3:
                continue
            # boundary params: n_up segments over the UPPER-ARM zone (t<=1)
            # + n_lo segments over the FOREARM zone (t>1) - the elbow (t=1)
            # is always a joint so the sleeve bends exactly with the arm
            bounds = []
            if t1 <= 1.0 + 1e-4:           # short sleeve: all upper-arm
                for kseg in range(n_up + n_lo + 1):
                    bounds.append(t0 + (t1 - t0) * kseg / (n_up + n_lo))
            else:
                u_end = min(1.0, t1)
                for kseg in range(n_up):
                    bounds.append(t0 + (u_end - t0) * kseg / n_up)
                for kseg in range(n_lo + 1):
                    bounds.append(u_end + (t1 - u_end) * kseg / n_lo)
            bw = max(1e-4, (t1 - t0) * 0.5 / max(1, len(bounds) - 1))
            chain = []
            for tb in bounds:
                band = [cos[i] for t, i in pts_t if abs(t - tb) < bw + 1e-4]
                if not band:
                    continue
                c = Vector((0, 0, 0))
                for p in band:
                    c += p
                chain.append(c / len(band))
            if len(chain) < 2:
                continue
            parent = eb.get("upper_arm." + side)
            prev = None
            for r in range(len(chain) - 1):
                name = "kan_sleeve.%s.%02d" % (side, r)
                b = eb.new(name)
                b.head = chain[r]; b.tail = chain[r + 1]
                if prev is None:
                    if parent is not None:
                        b.parent = parent
                        b.use_connect = False
                    made_sleeves.append(name)
                else:
                    b.parent = prev
                    b.use_connect = True
                prev = b
        bpy.ops.object.mode_set(mode='OBJECT')
        for name in made_sleeves:
            pb = mo.pose.bones.get(name)
            if pb is not None:
                try:
                    pb.rigify_type = "limbs.simple_tentacle"
                except Exception:
                    pass
        if made_sleeves:
            report.append("sleeves: %d chains (%d up + %d low)"
                          % (len(made_sleeves), n_up, n_lo))

    # ---------- 3) NECK + 4) POCKETS: anchor bones ----------
    anchors = []          # (bone_name, head_vg, parent_candidates)
    if region_registered(ob, 'NECK'):
        anchors.append(("kan_neck", VG_NECK, ["spine.004", "spine.003"]))
    if region_registered(ob, 'POCKET'):
        anchors.append(("kan_pocket", VG_POCKET, ["spine.002", "spine.001"]))
    if region_registered(ob, 'SIDE_POCKETS'):
        anchors.append(("kan_pocket_side.L", VG_SPOCKET_L, ["spine"]))
        anchors.append(("kan_pocket_side.R", VG_SPOCKET_R, ["spine"]))
    made_anchor = []
    if anchors:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = mo
        bpy.ops.object.mode_set(mode='EDIT')
        eb = mo.data.edit_bones
        for name, vg_name, parents in anchors:
            old = eb.get(name)
            if old is not None:
                eb.remove(old)
            c, _mz = _vg_centroid_z(ob, vg_name, cos)
            if c is None:
                continue
            b = eb.new(name)
            b.head = c
            b.tail = c + Vector((0.0, 0.0, 0.06))
            for pn in parents:
                pp = eb.get(pn)
                if pp is not None:
                    b.parent = pp
                    b.use_connect = False
                    break
            made_anchor.append(name)
        bpy.ops.object.mode_set(mode='OBJECT')
        for name in made_anchor:
            pb = mo.pose.bones.get(name)
            if pb is not None:
                try:
                    pb.rigify_type = "basic.super_copy"
                    pb.rigify_parameters.make_deform = True
                except Exception:
                    pass
        if made_anchor:
            report.append("anchors: " + ", ".join(made_anchor))

    mo["sr_kandura"] = True
    return True, " | ".join(report)




def remove_kandura_bones(mo):
    """Delete every bone the kandura build created from the metarig."""
    import bpy as _b
    if _b.context.object and _b.context.object.mode != 'OBJECT':
        _b.ops.object.mode_set(mode='OBJECT')
    try:
        mo.hide_set(False)
    except Exception:
        pass
    _b.ops.object.select_all(action='DESELECT')
    mo.select_set(True)
    _b.context.view_layer.objects.active = mo
    _b.ops.object.mode_set(mode='EDIT')
    eb = mo.data.edit_bones
    n = 0
    kandura_skirt = (mo.get("sr_skirt_method") == "kandura")
    for b in list(eb):
        nm = b.name
        if nm.startswith("kan_") or (kandura_skirt and nm.startswith("skirt.")):
            eb.remove(b)
            n += 1
    _b.ops.object.mode_set(mode='OBJECT')
    if kandura_skirt:
        for key in ("sr_skirt_kind", "sr_skirt_method", "sr_skirt_cols_built"):
            if key in mo:
                del mo[key]
    if "sr_kandura" in mo:
        del mo["sr_kandura"]
    return n


class SMARTRIG_OT_kandura_remove(bpy.types.Operator):
    bl_idname = "smartrig.kandura_remove"
    bl_label = "Remove Kandura Rig"
    bl_description = ("Delete all kandura bones (waist-down grid, sleeve chains, "
                      "neck/pocket anchors) from the metarig. Re-generate the rig "
                      "afterwards to update it")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        mo = bpy.data.objects.get("SR_Metarig")
        return mo is not None and bool(mo.get("sr_kandura"))

    def execute(self, context):
        mo = bpy.data.objects.get("SR_Metarig")
        n = remove_kandura_bones(mo)
        self.report({'INFO'}, "Removed %d kandura bones - Re-generate the rig" % n)
        return {'FINISHED'}


class SMARTRIG_OT_kandura_build(bpy.types.Operator):
    bl_idname = "smartrig.kandura_build"
    bl_label = "Build Kandura Rig"
    bl_description = ("Build the kandura bones from the registered regions: "
                      "waist-down skirt grid (inherits ALL skirt automations), "
                      "sleeve chains, neck and pocket anchors - into the metarig. "
                      "Review in Edit Mode, then Generate")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        ob = kandura_object(context)
        return (ob is not None
                and bpy.data.objects.get("SR_Metarig") is not None)

    def execute(self, context):
        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        ok, msg = build_kandura(context)
        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}
        _goto_metarig_edit(context)
        self.report({'INFO'}, "Kandura rig built - " + msg)
        return {'FINISHED'}


class SMARTRIG_OT_kandura_edit(bpy.types.Operator):
    bl_idname = "smartrig.kandura_edit"
    bl_label = "Start Registering (Edit Kandura)"
    bl_description = ("Open the kandura mesh in Edit Mode (edge select, nothing "
                      "selected) so you can Alt+Click the loops and Register them")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return kandura_object(context) is not None

    def execute(self, context):
        ob = kandura_object(context)
        context.scene.smartrig.kandura_object = ob
        try:
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        try:
            ob.hide_set(False)
        except Exception:
            pass
        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode='EDIT')
        context.tool_settings.mesh_select_mode = (False, True, False)  # edges
        try:
            bpy.ops.mesh.select_all(action='DESELECT')
        except Exception:
            pass
        self.report({'INFO'}, "Alt+Click a loop, then press its Register button")
        return {'FINISHED'}


class SMARTRIG_OT_kandura_done(bpy.types.Operator):
    bl_idname = "smartrig.kandura_done"
    bl_label = "Continue - Edit Metarig"
    bl_description = "Finish registering and jump into the metarig in Edit Mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if _goto_metarig_edit(context):
            self.report({'INFO'}, "Metarig in Edit Mode - continue rigging")
            return {'FINISHED'}
        self.report({'WARNING'}, "No metarig in the scene yet")
        return {'CANCELLED'}


class SMARTRIG_OT_kandura_buttons_objects(bpy.types.Operator):
    bl_idname = "smartrig.kandura_buttons_objects"
    bl_label = "Register Button Objects"
    bl_description = ("Tag the SELECTED objects as the kandura buttons (use this "
                      "when the buttons are separate small meshes, not part of "
                      "the kandura). Select them in Object Mode first")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        props = context.scene.smartrig
        kob = kandura_object(context)
        picked = [o for o in context.selected_objects
                  if o.type == 'MESH' and o != kob]
        if not picked:
            self.report({'ERROR'},
                        "Select the button objects first (not the kandura itself)")
            return {'CANCELLED'}
        for o in button_objects():
            del o["sr_kan_button"]
        for o in picked:
            o["sr_kan_button"] = True
        # buttons obviously exist -> flip the toggle on for the user
        props.kandura_has_buttons = True
        if all_registered(kob, props):
            if _goto_metarig_edit(context):
                self.report({'INFO'}, "%d buttons registered | ALL REGIONS DONE - metarig opened" % len(picked))
                return {'FINISHED'}
        self.report({'INFO'}, "%d button objects registered" % len(picked))
        return {'FINISHED'}


class SMARTRIG_OT_kandura_register(bpy.types.Operator):
    bl_idname = "smartrig.kandura_register"
    bl_label = "Register Kandura Region"
    bl_description = ("Store the edge loop(s) you selected in Edit Mode as this "
                      "kandura region (waist / sleeves / neck). The bones will be "
                      "placed on these loops automatically")
    bl_options = {'REGISTER', 'UNDO'}

    region: bpy.props.EnumProperty(
        items=[('WAIST', "Waist", "One horizontal loop at the waist"),
               ('SLEEVES', "Sleeves", "Both cuff loops (left+right at once)"),
               ('NECK', "Neck", "The neck-opening loop"),
               ('PLACKET', "Placket", "The vertical button-strip line down the "
                "chest center (almarad). Select its REAL extent: on a kandura "
                "it stops above the waist; on a full shirt it runs to the hem - "
                "the open-shirt automation will follow exactly what you select"),
               ('BUTTONS', "Buttons", "The button vertices on the placket "
                "(if the buttons are part of this mesh)"),
               ('POCKET', "Pocket", "The chest-pocket faces (select the pocket "
                "patch on the chest)"),
               ('SIDE_POCKETS', "Side Pockets", "BOTH waist/side pocket openings "
                "at hip level (left+right at once - split automatically)")])

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH'
                and context.edit_object is not None
                and context.edit_object.type == 'MESH')

    def execute(self, context):
        ob = context.edit_object
        bm = bmesh.from_edit_mesh(ob.data)
        sel = [v.index for v in bm.verts if v.select]
        if len(sel) < 4:
            self.report({'ERROR'}, "Select the edge loop first (Alt+Click), then Register")
            return {'CANCELLED'}
        # world-x per selected vert (for the L/R sleeve split)
        mw = ob.matrix_world
        xs = {v.index: (mw @ v.co).x for v in bm.verts if v.select}
        # leave Edit Mode to write vertex groups safely, then come back
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            if self.region == 'SLEEVES':
                left = [i for i in sel if xs[i] >= 0.0]
                right = [i for i in sel if xs[i] < 0.0]
                if not left or not right:
                    self.report({'ERROR'},
                                "Select BOTH cuff loops (left and right) then Register")
                    return {'CANCELLED'}
                nl = _assign(ob, VG_SLEEVE_L, left)
                nr = _assign(ob, VG_SLEEVE_R, right)
                msg = "Sleeves registered: L=%d, R=%d verts" % (nl, nr)
            elif self.region == 'WAIST':
                n = _assign(ob, VG_WAIST, sel)
                msg = "Waist registered: %d verts" % n
            elif self.region == 'PLACKET':
                # sanity: the placket is a VERTICAL line down the chest center
                cos = [mw @ ob.data.vertices[i].co for i in sel]
                zspan = max(c.z for c in cos) - min(c.z for c in cos)
                xspan = max(c.x for c in cos) - min(c.x for c in cos)
                if zspan < xspan:
                    self.report({'ERROR'},
                                "Placket must be a VERTICAL line down the chest "
                                "(select the center edge line, not a loop)")
                    return {'CANCELLED'}
                n = _assign(ob, VG_PLACKET, sel)
                msg = "Placket registered: %d verts" % n
            elif self.region == 'BUTTONS':
                n = _assign(ob, VG_BUTTONS, sel)
                msg = "Buttons registered: %d verts" % n
            elif self.region == 'POCKET':
                n = _assign(ob, VG_POCKET, sel)
                msg = "Pocket registered: %d verts" % n
            elif self.region == 'SIDE_POCKETS':
                left = [i for i in sel if xs[i] >= 0.0]
                right = [i for i in sel if xs[i] < 0.0]
                if not left or not right:
                    self.report({'ERROR'},
                                "Select BOTH side pockets (left and right) then Register")
                    return {'CANCELLED'}
                nl = _assign(ob, VG_SPOCKET_L, left)
                nr = _assign(ob, VG_SPOCKET_R, right)
                msg = "Side pockets registered: L=%d, R=%d verts" % (nl, nr)
            else:
                n = _assign(ob, VG_NECK, sel)
                msg = "Neck registered: %d verts" % n
        finally:
            bpy.ops.object.mode_set(mode='EDIT')
        # remember the kandura object on the scene props
        context.scene.smartrig.kandura_object = ob
        # all three regions done -> jump straight to the metarig in Edit Mode
        if all_registered(ob, context.scene.smartrig):
            if _goto_metarig_edit(context):
                self.report({'INFO'}, msg + "  |  ALL REGIONS DONE - metarig opened in Edit Mode")
                return {'FINISHED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class SMARTRIG_OT_kandura_clear(bpy.types.Operator):
    bl_idname = "smartrig.kandura_clear"
    bl_label = "Clear Kandura Regions"
    bl_description = "Remove all registered kandura regions (waist/sleeves/neck)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        ob = kandura_object(context)
        if ob is None:
            return {'CANCELLED'}
        for names in REGIONS.values():
            for vg_name in names:
                vg = ob.vertex_groups.get(vg_name)
                if vg is not None:
                    ob.vertex_groups.remove(vg)
        for bo in button_objects():
            del bo["sr_kan_button"]
        self.report({'INFO'}, "Kandura regions cleared")
        return {'FINISHED'}


classes = (SMARTRIG_OT_kandura_autodetect, SMARTRIG_OT_kandura_build,
           SMARTRIG_OT_kandura_remove,
           SMARTRIG_OT_kandura_edit, SMARTRIG_OT_kandura_done,
           SMARTRIG_OT_kandura_buttons_objects,
           SMARTRIG_OT_kandura_register, SMARTRIG_OT_kandura_clear)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
