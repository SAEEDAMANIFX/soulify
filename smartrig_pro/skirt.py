"""Short-skirt (ARP "Kilt"-style) cloth rig as a SmartRig sample.

The skirt geometry drives the bones: either a SEPARATE mesh (picked with the
eyedropper) or a region of the MERGED character mesh (selected in Edit Mode and
registered into the "SR_Skirt" vertex group). The addon analyses that geometry
and builds one FK ``limbs.simple_tentacle`` chain per column, running from the
top (waist) loop to the bottom (hem) loop, following the real shape. Bone roll
matches the thigh convention so the leg collision pushes the cloth cleanly."""
import bpy
import math
import re
import numpy as np
from mathutils import Vector, Matrix
from . import utils, fit
from .metarig import META_NAME

PREFIX = "skirt"
VGROUP = "SR_Skirt"


def skirt_verts_world(props):
    """World-space vertices of the skirt: separate object, or the registered
    vertex group on the merged character mesh. Returns Nx3 ndarray or None."""
    src = getattr(props, "skirt_source", 'MERGED')
    if src == 'SEPARATE':
        ob = getattr(props, "skirt_object", None)
        if ob is None or ob.type != 'MESH':
            return None
        return utils.read_world_coords(ob)
    # merged: read the SR_Skirt vertex group on the character mesh
    obj = props.target_mesh
    if obj is None or obj.type != 'MESH':
        return None
    vg = obj.vertex_groups.get(VGROUP)
    if vg is None:
        return None
    gi = vg.index
    mw = obj.matrix_world
    pts = []
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == gi and g.weight > 0.0:
                p = mw @ v.co
                pts.append((p.x, p.y, p.z))
                break
    if len(pts) < 6:
        return None
    return np.asarray(pts, dtype=float)


def _skirt_grid(co, cols, rows):
    """Build a [cols][rows+1] grid of world points from the skirt vertices,
    sliced by Z (top->bottom) and by angular sector around the center axis."""
    zmax = float(co[:, 2].max()); zmin = float(co[:, 2].min())
    if zmax - zmin < 1e-4:
        return None
    cx = float(np.median(co[:, 0]))
    cy = float(np.median(co[:, 1]))
    ang = (np.arctan2(co[:, 1] - cy, co[:, 0] - cx) + 2.0 * math.pi) % (2.0 * math.pi)
    # offset so column 0 is centered at the FRONT (-Y) of the body
    front = (math.atan2(-1.0, 0.0) + 2.0 * math.pi) % (2.0 * math.pi)
    sect = 2.0 * math.pi / cols
    rel = (ang - front + 0.5 * sect + 2.0 * math.pi) % (2.0 * math.pi)
    col_idx = (rel / sect).astype(int) % cols
    band = (zmax - zmin) / rows * 0.75
    grid = []
    for c in range(cols):
        cmask = col_idx == c
        colpts = []
        for l in range(rows + 1):
            z = zmax + (zmin - zmax) * (l / rows)
            sel = co[cmask & (np.abs(co[:, 2] - z) <= band)]
            if len(sel) == 0:
                sel = co[cmask]
                if len(sel) == 0:
                    colpts = None
                    break
                # nearest by z within the sector
                sel = sel[np.argsort(np.abs(sel[:, 2] - z))[:max(1, len(sel) // 8)]]
            # anchor Z to the exact slice height so the chain spans the FULL
            # skirt (top->hem); use the verts only for the horizontal position
            colpts.append(Vector((float(sel[:, 0].mean()),
                                  float(sel[:, 1].mean()),
                                  float(z))))
        if colpts is not None:
            grid.append((c, colpts))
    return grid if grid else None


def _ring_radii(co, cx, cy, z, h):
    """Torso half-width (X, arms excluded) and front/back Y at height z."""
    band = co[np.abs(co[:, 2] - z) < 0.025 * h]
    if len(band) < 10:
        return None
    xs = np.abs(band[:, 0] - cx)
    mx = float(xs.max())
    if mx < 1e-4:
        return None
    hist, edges = np.histogram(xs, bins=20, range=(0.0, mx))
    peak = hist[:6].max() if len(hist) >= 6 else hist.max()
    torso = mx
    for k in range(1, len(hist)):
        if hist[k] < max(1, 0.05 * peak) and edges[k] > 0.05:
            torso = float(edges[k]); break
    ys = band[:, 1] - cy
    return torso, float(ys.max()), float(ys.min())


def build_manual_skirt(props):
    """Starter ring fitted around the hips from the body cross-section. The user
    is then free to move / edit the bones by hand."""
    mo = bpy.data.objects.get(META_NAME)
    if mo is None:
        return None, "Build the Rigify metarig first, then add the skirt."
    mesh = props.target_mesh
    if mesh is None or mesh.type != 'MESH':
        return None, "Select your character mesh first."
    J, err, h = fit.compute_joints(props)
    if err:
        return None, err
    cx = 0.0
    cy = float(J["pelvis"][1]); pelz = float(J["pelvis"][2])
    knee_z = float(J["shin.L"][0][2])
    co = utils.read_world_coords(mesh)
    cols = max(4, int(props.skirt_columns)); rows = max(1, int(props.skirt_rows))
    top_z = pelz + 0.02 * h
    bot_z = top_z + float(props.skirt_length) * (knee_z - top_z)
    rt = _ring_radii(co, cx, cy, top_z, h) or (0.16 * (h / 1.6), 0.09, -0.09)
    rb = _ring_radii(co, cx, cy, bot_z, h) or (rt[0] * 1.12, rt[1], rt[2])
    rx_t, yf_t, yb_t = rt; rx_b, yf_b, yb_b = rb
    rx_t *= 1.06; rx_b = max(rx_b, rx_t) * 1.12
    ry_t = (yf_t - yb_t) * 0.5 * 1.06; ry_b = (yf_b - yb_b) * 0.5 * 1.12
    yc_t = cy + (yf_t + yb_t) * 0.5; yc_b = cy + (yf_b + yb_b) * 0.5
    grid = []
    for c in range(cols):
        th = 2.0 * math.pi * c / cols
        sn, csn = math.sin(th), math.cos(th)
        pts = []
        for l in range(rows + 1):
            f = l / rows
            rx = rx_t + (rx_b - rx_t) * f; ry = ry_t + (ry_b - ry_t) * f
            yc = yc_t + (yc_b - yc_t) * f; z = top_z + (bot_z - top_z) * f
            pts.append(Vector((cx + rx * sn, yc - ry * csn, z)))
        grid.append((c, pts))
    _emit_chains(mo, grid, rows)
    return mo, None


def live_rebuild(context):
    """Rebuild the skirt in place when Columns/Rows change - mesh-driven modes
    only, and only if a skirt already exists. Never touches manual edits."""
    p = context.scene.smartrig
    if getattr(p, "skirt_source", 'MERGED') == 'MANUAL':
        return
    if context.mode not in ('OBJECT', 'EDIT_ARMATURE', 'POSE'):
        return
    mo = bpy.data.objects.get(META_NAME)
    if mo is None:
        return
    if not any(b.name.startswith(PREFIX + ".") for b in mo.data.bones):
        return
    was_edit = (context.object is not None and context.object.mode == 'EDIT'
                and context.object == mo)
    build_skirt(p)
    # restore the user's edit mode on the metarig for a smooth live experience
    if was_edit:
        try:
            bpy.context.view_layer.objects.active = mo
            if mo.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass


def _emit_chains(mo, grid, rows):
    """Create one tentacle chain per column from the grid points, parent to
    the hips, roll like the thigh, and tag as Rigify simple_tentacle."""
    # ---- create the bones in edit mode ----
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = mo
    for o in bpy.context.selected_objects:
        o.select_set(False)
    mo.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = mo.data.edit_bones

    for b in [b for b in eb if b.name.startswith(PREFIX + ".")]:
        eb.remove(b)

    hips = eb.get("spine") or eb.get("spine.001")
    # PROFESSIONAL ROLL: each column's local Z points RADIALLY OUTWARD (and X is
    # tangent). Identical convention for every column at any count, so the
    # collision swings each panel purely radially -> panels never cross.
    allpts = [p for _c, pts in grid for p in pts]
    cx = sum(p.x for p in allpts) / len(allpts)
    cy = sum(p.y for p in allpts) / len(allpts)
    roots = []
    for c, pts in grid:
        prev = None
        for r in range(rows):
            head = pts[r]; tail = pts[r + 1]
            if (tail - head).length < 1e-5:
                tail = head + Vector((0.0, 0.0, -0.02))
            name = "%s.%02d.%02d" % (PREFIX, c, r)
            b = eb.new(name)
            b.head = head; b.tail = tail
            outward = Vector((head.x - cx, head.y - cy, 0.0))
            if outward.length < 1e-5:
                outward = Vector((0.0, -1.0, 0.0))
            outward.normalize()
            try:
                b.align_roll(outward)       # local Z = radial outward
            except Exception:
                pass
            if prev is None:
                if hips is not None:
                    b.parent = hips
                    b.use_connect = False
                roots.append(name)
            else:
                b.parent = prev
                b.use_connect = True
            prev = b

    bpy.ops.object.mode_set(mode='OBJECT')

    # ---- tag each column root as a Rigify simple_tentacle ----
    tagged = 0
    for name in roots:
        pb = mo.pose.bones.get(name)
        if pb is None:
            continue
        try:
            pb.rigify_type = "limbs.simple_tentacle"
            tagged += 1
            prm = pb.rigify_parameters
            for attr in ("tweak_layers_extra", "primary_layers_extra",
                         "secondary_layers_extra", "fk_layers_extra"):
                if hasattr(prm, attr) and isinstance(getattr(prm, attr), bool):
                    try:
                        setattr(prm, attr, False)
                    except Exception:
                        pass
        except Exception:
            pass
    return mo


def build_skirt(props):
    mo = bpy.data.objects.get(META_NAME)
    if mo is None:
        return None, "Build the Rigify metarig first, then add the skirt."
    co = skirt_verts_world(props)
    if co is None:
        if getattr(props, "skirt_source", 'MERGED') == 'SEPARATE':
            return None, "Pick the skirt mesh with the eyedropper first."
        return None, ("Select the skirt faces in Edit Mode, then press "
                      "'Register Skirt Selection'.")

    cols = max(4, int(props.skirt_columns))
    rows = max(1, int(props.skirt_rows))
    grid = _skirt_grid(co, cols, rows)
    if not grid:
        return None, "Could not analyse the skirt geometry. Check the selection."

    _emit_chains(mo, grid, rows)
    return mo, None


def _resolve_colliders(rig, names):
    """Map ANY chosen leg-bone name (control / org / deform) to the rig's DEFORM
    bones that move, e.g. 'thigh.L', 'thigh_fk.L' or 'DEF-thigh.L' all resolve to
    DEF-thigh.L + DEF-thigh.L.001. Returns a list of bone names."""
    targets = []
    for nm in names:
        if not nm:
            continue
        core = nm
        for pre in ("DEF-", "ORG-", "MCH-", "VIS_"):
            if core.startswith(pre):
                core = core[len(pre):]
        if "." in core:
            stem, side = core.rsplit(".", 1)
        else:
            stem, side = core, ""
        for suf in ("_fk", "_ik", "_tweak", "_parent"):
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
        base = stem + (("." + side) if side else "")
        found = [b.name for b in rig.data.bones
                 if b.use_deform and (b.name == "DEF-" + base
                                      or b.name.startswith("DEF-" + base + "."))]
        if not found and rig.data.bones.get(nm):
            found = [nm]
        targets.extend(found)
    seen = set(); out = []
    for t in targets:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def _add_driver(owner, path, expr, varspecs, index=-1):
    """varspecs: list of (name, id_obj, kind, data_path-or-bone, transform_type)."""
    try:
        d = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    except Exception:
        return None
    drv = d.driver
    drv.type = 'SCRIPTED'
    drv.expression = expr
    for v in list(drv.variables):
        drv.variables.remove(v)
    for spec in varspecs:
        name, kind = spec[0], spec[1]
        var = drv.variables.new()
        var.name = name
        var.type = kind
        if kind == 'SINGLE_PROP':
            _id, dpath = spec[2], spec[3]
            var.targets[0].id = _id
            var.targets[0].data_path = dpath
        elif kind == 'ROTATION_DIFF':
            rig_id, b1, b2 = spec[2], spec[3], spec[4]
            var.targets[0].id = rig_id; var.targets[0].bone_target = b1
            var.targets[1].id = rig_id; var.targets[1].bone_target = b2
    return drv


def live_kilt_tune(context):
    """Push the addon-panel collision sliders into the SKC_master custom props,
    which DRIVE the Floor constraints (so both the Item-tab bone sliders and the
    addon panel stay live, with no rebuild)."""
    from .metarig import META_NAME
    mo = bpy.data.objects.get(META_NAME)
    rig = None
    if mo is not None and getattr(mo.data, "rigify_target_rig", None):
        rig = mo.data.rigify_target_rig
    if rig is None:
        for o in bpy.data.objects:
            if o.type == 'ARMATURE' and o.name.startswith("RIG-") and o.get("sk_kilt"):
                rig = o; break
    if rig is None or not rig.get("sk_kilt"):
        return
    mpb = rig.pose.bones.get("SKC_master")
    if mpb is None:
        return
    p = context.scene.smartrig
    mpb["collide"] = 1.0 if getattr(p, "skirt_collide", True) else 0.0
    mpb["collide_dist"] = float(getattr(p, "skirt_collide_dist", 0.12))
    mpb["collide_dist_falloff"] = float(getattr(p, "skirt_collide_falloff", 0.4))
    mpb["collide_spread"] = float(getattr(p, "skirt_collide_spread", 1.0))
    rig.update_tag()


def kilt_rig(context):
    """Return the active generated rig that has the skirt collision OR jiggle."""
    def ok(o):
        return o is not None and o.type == 'ARMATURE' and (o.get("sk_kilt") or o.get("sk_jiggle") or o.get("sk_follow"))
    ob = context.active_object if context else None
    if ok(ob):
        return ob
    from .metarig import META_NAME
    mo = bpy.data.objects.get(META_NAME)
    if mo is not None and getattr(mo.data, "rigify_target_rig", None):
        r = mo.data.rigify_target_rig
        if ok(r):
            return r
    for o in bpy.data.objects:
        if ok(o):
            return o
    return None


def remove_skirt_collision(rig):
    """Remove all skirt collision constraints, helper bones and drivers, and
    RESTORE any skirt controls that were re-parented onto the SKC_dt bones."""
    if rig is None:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = rig
    n = 0
    restore = {}
    for pb in rig.pose.bones:
        for c in list(pb.constraints):
            if c.name.startswith(("SK_FLOOR", "SK_FOLLOW", "SK_LIMIT", "SK_DT", "SK_RIDE")):
                pb.constraints.remove(c); n += 1
        if "sk_origparent" in pb:
            restore[pb.name] = str(pb["sk_origparent"])
        for k in ("sk_base", "sk_sx", "sk_sz", "sk_axis", "sk_sgn",
                  "sk_oxn", "sk_oyn", "sk_origparent"):
            if k in pb:
                del pb[k]
        if rig.animation_data:
            dp = 'pose.bones["%s"].rotation_euler' % pb.name
            for dr in list(rig.animation_data.drivers):
                if dr.data_path == dp:
                    try:
                        rig.animation_data.drivers.remove(dr)
                    except Exception:
                        pass
    if "sk_kilt" in rig:
        del rig["sk_kilt"]
    ad = rig.animation_data
    if ad:
        for dr in list(ad.drivers):
            if "SKC_" in dr.data_path or "SK_FLOOR" in dr.data_path:
                try:
                    ad.drivers.remove(dr)
                except Exception:
                    pass
    bpy.ops.object.mode_set(mode='EDIT')
    ebs = rig.data.edit_bones
    # restore re-parented controls FIRST (before deleting the SKC_dt parents)
    for cname, pname in restore.items():
        cb = ebs.get(cname)
        if cb is None:
            continue
        cb.parent = ebs.get(pname) if pname else None
    for b in list(ebs):
        if b.name.startswith("SKC_"):
            ebs.remove(b)
    bpy.ops.object.mode_set(mode='OBJECT')
    return n


def live_tune(context):
    """Live-update the radial push strength on the generated rig by rewriting the
    driver expressions (no rebuild, no mode change)."""
    from .metarig import _generated_rig
    rig = _generated_rig()
    if rig is None or not rig.animation_data:
        return
    strength = float(getattr(context.scene.smartrig, "skirt_follow", 0.6))
    for pb in rig.pose.bones:
        if "sk_axis" not in pb:
            continue
        axis = int(pb["sk_axis"]); sgn = float(pb["sk_sgn"])
        oxn = float(pb["sk_oxn"]); oyn = float(pb["sk_oyn"])
        dp = 'pose.bones["%s"].rotation_euler' % pb.name
        for dr in rig.animation_data.drivers:
            if dr.data_path == dp and dr.array_index == axis:
                dr.driver.expression = ("%.5f*(max(0.0,%.5f*rx)+1.8*max(0.0,%.5f*rz)+%.5f*abs(rx))"
                                        % (sgn * strength, oyn, -oxn, 0.5 * abs(oxn)))


def _clear_skirt_drivers(rig, name):
    ad = rig.animation_data
    if not ad:
        return
    pref = 'pose.bones["%s"].rotation_euler' % name
    for dr in list(ad.drivers):
        if dr.data_path == pref:
            try:
                ad.drivers.remove(dr)
            except Exception:
                pass


def _skirt_columns(rig):
    """Return dict: col_index -> (root_control_name, hem_world, head_world)."""
    cols = {}
    rw = rig.matrix_world
    for b in rig.data.bones:
        m = re.match(r"^" + PREFIX + r"\.(\d+)\.(\d+)$", b.name)
        if m:
            ci = int(m.group(1)); ri = int(m.group(2))
            cols.setdefault(ci, {})[ri] = b.name
    out = {}
    for ci, rows in cols.items():
        root = rows[min(rows)]
        hemb = rig.data.bones[rows[max(rows)]]
        out[ci] = (root, rw @ hemb.tail_local, rw @ rig.data.bones[root].head_local)
    return out


def _ensure_master_widget():
    """Create (once) a distinctive double-ring + cross widget for SKC_master so the
    animator recognises it as the skirt settings control. Lives in a hidden
    widget collection."""
    name = "WGT-SKC_master"
    wgt = bpy.data.objects.get(name)
    if wgt is not None and wgt.type == 'MESH':
        return wgt
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    N = 28
    def ring(r):
        vs = [bm.verts.new((r * math.cos(2 * math.pi * i / N), 0.0,
                            r * math.sin(2 * math.pi * i / N))) for i in range(N)]
        for i in range(N):
            bm.edges.new((vs[i], vs[(i + 1) % N]))
        return vs
    ring(1.0); ring(0.62)
    # small cross in the middle
    a = bm.verts.new((-0.25, 0, 0)); b = bm.verts.new((0.25, 0, 0))
    c = bm.verts.new((0, 0, -0.25)); d = bm.verts.new((0, 0, 0.25))
    bm.edges.new((a, b)); bm.edges.new((c, d))
    bm.to_mesh(me); bm.free()
    wgt = bpy.data.objects.new(name, me)
    coll = bpy.data.collections.get("WGTS_SmartRig")
    if coll is None:
        coll = bpy.data.collections.new("WGTS_SmartRig")
        try:
            bpy.context.scene.collection.children.link(coll)
        except Exception:
            pass
        lc = bpy.context.view_layer.layer_collection.children.get("WGTS_SmartRig")
        if lc is not None:
            lc.exclude = True
    coll.objects.link(wgt)
    return wgt


# ============================ SKIRT FOLLOW BODY (sit/blend) ==================
def _hip_bone(rig):
    for n in ("ORG-spine", "DEF-spine", "ORG-pelvis.L", "spine_fk"):
        if rig.data.bones.get(n):
            return n
    return None


def _skirt_follow_objs(props):
    """The (skirt_object, body_mesh) for Surface-Deform follow. Returns (None,None)
    if not a SEPARATE skirt (Surface Deform needs a different target mesh)."""
    body = props.target_mesh
    sk = props.skirt_object if getattr(props, "skirt_source", 'MERGED') == 'SEPARATE' else None
    if sk is None or sk.type != 'MESH' or body is None or body.type != 'MESH':
        return None, None
    return sk, body


def add_skirt_follow_body(rig, props):
    """Blendable 'Follow Body' = the skirt CLINGS to the body surface (like a
    Surface Deform / weight transfer). A `Surface Deform` modifier on the skirt is
    bound to the body mesh; its strength is driven by the live `follow_body` slider
    (0 = skirt rig only, 1 = skirt follows the body surface -> drapes over the lap
    when seated). Needs a SEPARATE skirt mesh."""
    if rig is None:
        return 0
    sk, body = _skirt_follow_objs(props)
    if sk is None:
        return 0
    _ensure_drivers_trusted()
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    # rest pose so the bind captures the neutral shape
    if rig.mode != 'OBJECT':
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='POSE')
    for pbn in rig.pose.bones:
        pbn.matrix_basis.identity()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()
    remove_skirt_follow_body(rig)

    # Surface Deform modifier AFTER the armature (so it pulls the rigged skirt
    # onto the body surface).
    mod = sk.modifiers.get("SK_SurfaceFollow")
    if mod is None:
        mod = sk.modifiers.new("SK_SurfaceFollow", 'SURFACE_DEFORM')
    mod.target = body
    mod.strength = 0.0
    # bind (skirt active, object mode, body visible)
    bpy.ops.object.select_all(action='DESELECT')
    sk.select_set(True); bpy.context.view_layer.objects.active = sk
    # SMART ORDER: place Surface Deform right after the Armature and ABOVE any
    # Subdivision Surface, so the bind input is the rigged base cage. Subsurf BELOW
    # then just smooths the result and never invalidates the bind.
    arm_idx = next((i for i, mm in enumerate(sk.modifiers) if mm.type == 'ARMATURE'), -1)
    tgt_idx = arm_idx + 1 if arm_idx >= 0 else 0
    win = bpy.context.window
    area = next((a for a in win.screen.areas if a.type == 'VIEW_3D'), None) if win else None
    region = next((r for r in area.regions if r.type == 'WINDOW'), None) if area else None
    ov = {"object": sk, "active_object": sk}
    if win:
        ov["window"] = win
    if area:
        ov["area"] = area
    if region:
        ov["region"] = region
    try:
        with bpy.context.temp_override(**ov):
            if list(sk.modifiers).index(mod) != tgt_idx:
                bpy.ops.object.modifier_move_to_index(modifier="SK_SurfaceFollow", index=tgt_idx)
            # push every Subdivision Surface BELOW the Surface Deform (correct order:
            # Armature -> SurfaceDeform -> Subsurf), so subdivisions never invalidate
            # the bind and the smooth result still follows the body.
            for mm in list(sk.modifiers):
                if mm.type == 'SUBSURF':
                    last = len(sk.modifiers) - 1
                    if list(sk.modifiers).index(mm) < list(sk.modifiers).index(mod):
                        bpy.ops.object.modifier_move_to_index(modifier=mm.name, index=last)
    except Exception as e:
        print("SmartRig follow reorder:", e)
    try:
        with bpy.context.temp_override(**ov):
            bpy.ops.object.surfacedeform_bind(modifier="SK_SurfaceFollow")
    except Exception as e:
        print("SmartRig surface-deform bind:", e)

    # the modifier STRENGTH is the live "Follow Body" value (drawn directly in the
    # panels - keyframeable, immediate, no driver/trust dependency).
    mod.strength = float(getattr(props, "skirt_follow_body", 0.0))
    rig["sk_follow"] = 1
    bound = getattr(mod, "is_bound", True)
    return 1 if bound else 0


def remove_skirt_follow_body(rig):
    if rig is None:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    n = 0
    # remove the Surface Deform modifier (+ its driver) from any mesh that has it
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        for md in list(ob.modifiers):
            if md.name == "SK_SurfaceFollow":
                try:
                    ob.modifiers.remove(md); n += 1
                except Exception:
                    pass
        ad2 = ob.animation_data
        if ad2:
            for dr in list(ad2.drivers):
                if "SK_SurfaceFollow" in dr.data_path:
                    try: ad2.drivers.remove(dr)
                    except Exception: pass
    # remove the old bone-based SK_FOLLOW constraints (legacy) if present
    for pb in rig.pose.bones:
        for c in list(pb.constraints):
            if c.name == "SK_FOLLOW":
                pb.constraints.remove(c); n += 1
    ad = rig.animation_data
    if ad:
        for dr in list(ad.drivers):
            if 'SK_FOLLOW' in dr.data_path:
                try: ad.drivers.remove(dr)
                except Exception: pass
    for k in ("sk_follow", "follow_body"):
        if k in rig:
            del rig[k]
    return n


def live_follow_tune(context):
    try:
        md = follow_modifier(context)
        if md is not None:
            md.strength = float(context.scene.smartrig.skirt_follow_body)
    except Exception as e:
        print("SmartRig follow tune:", e)


def follow_modifier(context):
    """Return the skirt's SK_SurfaceFollow modifier (the Follow Body control), or None."""
    p = context.scene.smartrig
    sk = p.skirt_object if getattr(p, "skirt_source", 'MERGED') == 'SEPARATE' else None
    if sk is None:
        for o in bpy.data.objects:
            if o.type == 'MESH' and o.modifiers.get("SK_SurfaceFollow"):
                sk = o; break
    return sk.modifiers.get("SK_SurfaceFollow") if (sk and sk.type == 'MESH') else None


def follow_status(context):
    """Return ('none'|'ok'|'subsurf_above', modifier). 'subsurf_above' means a
    Subdivision Surface sits ABOVE SK_SurfaceFollow on the skirt -> the bind is
    invalid and the user should Re-bind (Apply Body Follow) to fix the order."""
    p = context.scene.smartrig
    ob = None
    cand = []
    sk = p.skirt_object if getattr(p, "skirt_source", 'MERGED') == 'SEPARATE' else None
    if sk is not None:
        cand.append(sk)
    cand += [o for o in bpy.data.objects if o.type == 'MESH']
    for o in cand:
        if o is not None and o.type == 'MESH' and o.modifiers.get("SK_SurfaceFollow") is not None:
            ob = o; break
    if ob is None:
        return 'none', None
    md = ob.modifiers.get("SK_SurfaceFollow")
    sd_idx = list(ob.modifiers).index(md)
    for i, mm in enumerate(ob.modifiers):
        if mm.type == 'SUBSURF' and i < sd_idx:
            return 'subsurf_above', md
    return 'ok', md


# ============================ SKIRT JIGGLE (live spring) =====================
_JIG_STATE = {}      # bone_name -> {"p":Vector,"v":Vector}
_JIG_LAST_FRAME = [None]


def _column_root_bone(rig, ci):
    """The bone at the TOP of column ci that the whole column hangs from:
    SKC_dt.CC.00 if collision exists, else the control skirt.CC.00."""
    return ("SKC_dt.%02d.00" % ci) if rig.data.bones.get("SKC_dt.%02d.00" % ci) else (PREFIX + ".%02d.00" % ci)


def add_skirt_jiggle(rig, props):
    """Insert one SKC_jig bone per column ABOVE the column root, pivoting at the
    waist and spanning to the hem. A frame-change spring handler swings it so the
    whole column sways (hem most, waist fixed) - live secondary motion."""
    if rig is None:
        return 0
    cols = {}
    for b in rig.data.bones:
        m = re.match(r"^" + PREFIX + r"\.(\d+)\.(\d+)$", b.name)
        if m:
            cols.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    if not cols:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    remove_skirt_jiggle(rig)
    rw = rig.matrix_world; rwi = rw.inverted()
    # waist (root head) + hem (last row tail) per column, in armature space
    geo = {}
    for ci, rows in cols.items():
        root = rig.data.bones.get(_column_root_bone(rig, ci))
        hem = rig.data.bones.get(PREFIX + ".%02d.%02d" % (ci, max(rows)))
        if root and hem:
            geo[ci] = (root.head_local.copy(), hem.tail_local.copy(), root.name)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    orig = {}
    for ci, (waist, hemtail, rootname) in geo.items():
        rc = eb.get(rootname)
        if rc is None:
            continue
        jig = eb.new("SKC_jig.%02d" % ci)
        jig.head = waist.copy(); jig.tail = hemtail.copy(); jig.use_deform = False
        op = rc.parent
        orig[rootname] = op.name if op else ""
        if op is not None:
            jig.parent = op
        rc.parent = jig
    bpy.ops.object.mode_set(mode='OBJECT')
    for rootname, pname in orig.items():
        pb = rig.pose.bones.get(rootname)
        if pb is not None:
            pb["sk_jigorig"] = pname
    rig["sk_jiggle"] = 1
    if "sk_jiggle_baked" in rig:
        del rig["sk_jiggle_baked"]
    # settings live on the RIG object (works with or without collision; keyframeable)
    spec = (("jiggle", 1.0, 0.0, 1.0, "Enable skirt jiggle (live secondary motion)"),
            ("jiggle_amount", float(getattr(props, "jiggle_amount", 1.0)), 0.0, 2.0, "How much the skirt sways"),
            ("jiggle_stiffness", float(getattr(props, "jiggle_stiffness", 0.40)), 0.02, 1.0, "Spring stiffness"),
            ("jiggle_damping", float(getattr(props, "jiggle_damping", 0.25)), 0.05, 0.99, "Damping (higher settles faster)"))
    for k, val, lo, hi, desc in spec:
        rig[k] = val
        try:
            ui = rig.id_properties_ui(k); ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=desc)
        except Exception:
            pass
    _organize_skirt_bones(rig)
    _JIG_STATE.clear(); _JIG_LAST_FRAME[0] = None
    register_jiggle_handler()
    return len(geo)


def remove_skirt_jiggle(rig):
    if rig is None:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    restore = {}
    for pb in rig.pose.bones:
        if "sk_jigorig" in pb:
            restore[pb.name] = str(pb["sk_jigorig"]); del pb["sk_jigorig"]
    for k in ("sk_jiggle", "sk_jiggle_baked", "jiggle", "jiggle_amount",
              "jiggle_stiffness", "jiggle_damping"):
        if k in rig:
            del rig[k]
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    for rootname, pname in restore.items():
        rc = eb.get(rootname)
        if rc is not None:
            rc.parent = eb.get(pname) if pname else None
    for b in list(eb):
        if b.name.startswith("SKC_jig"):
            eb.remove(b)
    bpy.ops.object.mode_set(mode='OBJECT')
    _JIG_STATE.clear(); _JIG_LAST_FRAME[0] = None
    if not any(o.type == 'ARMATURE' and o.get("sk_jiggle") for o in bpy.data.objects):
        unregister_jiggle_handler()
    return len(restore)


def _jiggle_rigs():
    return [o for o in bpy.data.objects if o.type == 'ARMATURE' and o.get("sk_jiggle")]


def skirt_jiggle_handler(scene, depsgraph=None):
    from mathutils import Vector, Matrix
    rigs = _jiggle_rigs()
    if not rigs:
        return
    frame = scene.frame_current
    last = _JIG_LAST_FRAME[0]
    reset = (last is None) or (frame <= last) or (frame - last > 1)
    _JIG_LAST_FRAME[0] = frame
    for rig in rigs:
        if rig.get("sk_jiggle_baked"):
            continue
        on = float(rig["jiggle"]) if "jiggle" in rig else 1.0
        amount = float(rig["jiggle_amount"]) if "jiggle_amount" in rig else 1.0
        stiff = float(rig["jiggle_stiffness"]) if "jiggle_stiffness" in rig else 0.40
        damp = float(rig["jiggle_damping"]) if "jiggle_damping" in rig else 0.25
        rw = rig.matrix_world
        for pb in rig.pose.bones:
            if not pb.name.startswith("SKC_jig"):
                continue
            par = pb.parent
            if par is not None:
                M = par.matrix @ par.bone.matrix_local.inverted() @ pb.bone.matrix_local
            else:
                M = rw @ pb.bone.matrix_local
            head = M.translation.copy()
            L = pb.bone.length
            rest_dir = (M.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
            goal = head + rest_dir * L
            st = _JIG_STATE.get(pb.name)
            if reset or st is None or on < 0.5:
                p = goal.copy(); v = Vector((0, 0, 0))
            else:
                p = st["p"]; v = st["v"]
                v += (goal - p) * stiff      # spring pull toward the animated goal
                v *= (1.0 - damp)            # damping (low -> bouncy, high -> settles)
                p = p + v
                d = p - head
                ln = d.length or 1e-6
                p = head + d * (L / ln)
            _JIG_STATE[pb.name] = {"p": p.copy(), "v": v.copy()}
            cur = rest_dir
            new = (p - head).normalized()
            if amount < 1.0:
                new = cur.lerp(new, max(0.0, min(1.0, amount))).normalized()
            elif amount > 1.0:
                # exaggerate beyond the simulated swing
                ang = cur.angle(new) * (amount - 1.0)
                if ang > 1e-5:
                    axis = cur.cross(new)
                    if axis.length > 1e-6:
                        new = (Matrix.Rotation(cur.angle(new) * amount, 4, axis.normalized()).to_3x3() @ cur).normalized()
            q = cur.rotation_difference(new)
            try:
                pb.matrix = Matrix.Translation(head) @ (q @ M.to_quaternion()).to_matrix().to_4x4()
            except Exception:
                pass


def register_jiggle_handler():
    unregister_jiggle_handler()
    bpy.app.handlers.frame_change_post.append(skirt_jiggle_handler)


def unregister_jiggle_handler():
    for h in list(bpy.app.handlers.frame_change_post):
        if getattr(h, "__name__", "") == "skirt_jiggle_handler":
            try:
                bpy.app.handlers.frame_change_post.remove(h)
            except Exception:
                pass


def _organize_skirt_bones(rig):
    """Tidy the skirt bones into bone collections with professional colours:
      - "Skirt"        (visible, pink)   = the FK controls the animator poses + master (gold)
      - "Skirt (Tweak)"(visible, purple) = the secondary tweak controls
      - "Skirt (MCH)"  (HIDDEN)          = SKC_dt driven helpers the animator must NOT touch
    Re-applied every time collision is built, so it survives re-generation."""
    arm = rig.data

    def get_coll(name, visible):
        c = next((x for x in arm.collections_all if x.name == name), None)
        if c is None:
            c = arm.collections.new(name)
        try:
            c.is_visible = visible
        except Exception:
            pass
        return c

    main = get_coll("Skirt", True)
    tweak = get_coll("Skirt (Tweak)", True)
    master_c = get_coll("Skirt (Master)", True)
    mch = get_coll("Skirt (MCH)", False)
    # show "Skirt" / "Skirt (Tweak)" as toggle buttons in the Rigify "Rig Layers"
    # panel (it reads rigify_ui_row); MCH stays out of it and hidden.
    try:
        main.rigify_ui_row = 20
        tweak.rigify_ui_row = 21
        master_c.rigify_ui_row = 22
        mch.rigify_ui_row = 0
    except Exception:
        pass

    def col(b, normal, select, active):
        bc = b.color
        bc.palette = 'CUSTOM'
        bc.custom.normal = normal
        bc.custom.select = select
        bc.custom.active = active

    PINK = ((0.78, 0.18, 0.45), (1.0, 0.55, 0.8), (1.0, 0.85, 0.95))
    PURP = ((0.45, 0.28, 0.62), (0.78, 0.6, 0.95), (0.95, 0.85, 1.0))
    GOLD = ((0.95, 0.72, 0.1), (1.0, 0.9, 0.4), (1.0, 1.0, 0.75))
    def reassign(b, coll):
        for c in list(b.collections):
            try:
                c.unassign(b)
            except Exception:
                pass
        coll.assign(b)
    for b in arm.bones:
        n = b.name
        if n.startswith("SKC_dt") or n.startswith("SKC_jig"):
            reassign(b, mch)
        elif n == "SKC_master":
            reassign(b, master_c); col(b, *GOLD)   # selectable settings control
        elif re.match(r"^" + PREFIX + r"\.\d+\.\d+$", n):
            reassign(b, main); col(b, *PINK)
        elif PREFIX in n and "tweak" in n:
            reassign(b, tweak); col(b, *PURP)
    mpb = rig.pose.bones.get("SKC_master")
    if mpb is not None:
        try:
            mpb.custom_shape = _ensure_master_widget()
            mpb.use_custom_shape_bone_size = False
            mpb.custom_shape_scale_xyz = (0.16, 0.16, 0.16)
        except Exception:
            pass


def _ensure_drivers_trusted():
    """Our collision uses Python-expression drivers that read other bones. Blender
    DISABLES such drivers when a .blend is opened with 'Auto Run Python Scripts'
    OFF. Turn the preference ON (persists for future opens) so the collision keeps
    working. NOTE: for the CURRENT file you must reload it once after enabling."""
    try:
        bpy.context.preferences.filepaths.use_scripts_auto_execute = True
        bpy.ops.wm.save_userpref()
    except Exception:
        pass


def add_skirt_collision(rig, props, h=None):
    """ARP Kilt-style TRUE collision: per-leg Floor plane (follows the leg) +
    per-column target (Floor-collided = real clearance) + dt (Damped Track). The
    column control is re-parented onto dt so it RIDES the collision while FK still
    works on top. Proximity push => no crossing; correct drape in any direction."""
    if rig is None:
        return 0
    _ensure_drivers_trusted()
    cols = _skirt_columns(rig)
    if not cols:
        return 0
    thL = _resolve_colliders(rig, [props.skirt_collider_l]) or ["DEF-thigh.L"]
    thR = _resolve_colliders(rig, [props.skirt_collider_r]) or ["DEF-thigh.R"]
    thigh_L = thL[0] if rig.data.bones.get(thL[0]) else None
    thigh_R = thR[0] if rig.data.bones.get(thR[0]) else None
    if not (thigh_L and thigh_R):
        return 0
    dist = float(getattr(props, "skirt_collide_dist", 0.12))
    spread = float(getattr(props, "skirt_collide_spread", 1.0))
    falloff = float(getattr(props, "skirt_collide_falloff", 0.4))

    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    remove_skirt_collision(rig)
    cols = _skirt_columns(rig)

    rwi = rig.matrix_world.inverted()
    maxx = max(1e-4, max(abs(v[2].x) for v in cols.values()))
    orig_parents = {}

    # full per-column row map (so the dt is SPLIT into one segment per row -> the
    # column bends progressively toward the hem like cloth, instead of a rigid swing)
    colrows = {}
    for b in rig.data.bones:
        mm = re.match(r"^" + PREFIX + r"\.(\d+)\.(\d+)$", b.name)
        if mm:
            colrows.setdefault(int(mm.group(1)), []).append((int(mm.group(2)), b.name))
    for ci in colrows:
        colrows[ci].sort()

    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    for ci, rws in colrows.items():
        prev = None
        for rr, bn in rws:
            rcb = eb.get(bn)
            if rcb is None:
                continue
            op = rcb.parent
            orig_parents[bn] = op.name if op else ""
            seg = eb.new("SKC_dt.%02d.%02d" % (ci, rr))
            seg.head = rcb.head.copy(); seg.tail = rcb.tail.copy(); seg.use_deform = False
            if prev is not None:
                seg.parent = prev
            elif op is not None:
                seg.parent = op
            rcb.parent = seg        # each row rides its own dt segment
            prev = seg
    # master control bone holding the 4 live collision settings (ARP c_kilt_master)
    cen = Vector((0.0, 0.0, 0.0))
    for _r, (_ro, _hm, _hd) in cols.items():
        cen = cen + _hd
    cen = rwi @ (cen / max(1, len(cols)))
    mb = eb.new("SKC_master")
    mb.head = cen; mb.tail = cen + Vector((0.0, 0.0, 0.16)); mb.use_deform = False
    _anyop = None
    for _pn in orig_parents.values():
        if _pn and eb.get(_pn):
            _anyop = eb.get(_pn); break
    if _anyop is not None:
        mb.parent = _anyop
    bpy.ops.object.mode_set(mode='OBJECT')

    for root, pname in orig_parents.items():
        pb = rig.pose.bones.get(root)
        if pb is not None:
            pb["sk_origparent"] = pname

    rig["sk_kilt"] = 1
    # ---- master control: 4 live, keyframeable settings (ARP c_kilt_master) ----
    mpb = rig.pose.bones.get("SKC_master")
    if mpb is not None:
        mpb.rotation_mode = 'XYZ'
        spec = (("collide", 1.0 if getattr(props, "skirt_collide", True) else 0.0, 0.0, 1.0,
                 "Enable leg collision (0 = off, 1 = on)"),
                ("collide_dist", dist, 0.0, 0.6, "Clearance kept between the skirt and the legs"),
                ("collide_dist_falloff", falloff, 0.0, 1.0, "Base clearance kept even at rest"),
                ("collide_spread", spread, 0.0, 2.0, "How many columns around each leg are pushed"))
        for key, val, lo, hi, desc in spec:
            mpb[key] = float(val)
            try:
                ui = mpb.id_properties_ui(key)
                ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=desc)
            except Exception:
                pass

    def _mvar(drv, nm, key):
        v = drv.variables.new(); v.name = nm; v.type = 'SINGLE_PROP'
        t = v.targets[0]; t.id_type = 'OBJECT'; t.id = rig
        t.data_path = 'pose.bones["SKC_master"]["%s"]' % key

    # COMPASS model (like Auto-Rig Pro): each column RIDES its SKC_dt bone. We
    # drive dt to rotate the column OUTWARD by how much the nearest leg's KNEE
    # swings toward that column. The knee-hip horizontal displacement is the
    # compass needle (points the way the leg kicks - forward/back/in/out, FK or
    # IK); each column only reacts to the component along ITS outward direction.
    # So a side kick moves only the side columns, a forward kick only the front,
    # etc. It only ever swings outward (no crossing); FK layers on top.
    rw = rig.matrix_world
    cx = sum(v[2].x for v in cols.values()) / len(cols)
    cy = sum(v[2].y for v in cols.values()) / len(cols)
    pL = rw @ rig.data.bones[thigh_L].head_local
    pR = rw @ rig.data.bones[thigh_R].head_local
    AMP = 5.5

    def _knee_bone(thn):
        for cand in (thn.replace("thigh", "shin"), thn.replace("thigh", "calf")):
            if rig.pose.bones.get(cand):
                return cand
        b = rig.data.bones.get(thn); last = thn
        while b and b.children:
            b = b.children[0]; last = b.name
        return last
    knee = {"L": _knee_bone(thigh_L), "R": _knee_bone(thigh_R)}
    hipb = {"L": thigh_L, "R": thigh_R}
    rdxy = {}
    for sd in ("L", "R"):
        kw = rw @ rig.data.bones[knee[sd]].head_local
        hw = rw @ rig.data.bones[hipb[sd]].head_local
        rdxy[sd] = (kw.x - hw.x, kw.y - hw.y)

    def _locvar(drv, nm, bone, axis):
        v = drv.variables.new(); v.name = nm; v.type = 'TRANSFORMS'
        t = v.targets[0]; t.id = rig; t.bone_target = bone
        t.transform_type = axis; t.transform_space = 'WORLD_SPACE'

    n = 0
    for ci, rws in colrows.items():
        nseg = max(1, len(rws))
        # column azimuth/outward + leg blend weights, from the ROOT row head
        rb = rig.data.bones.get(rws[0][1])
        if rb is None:
            continue
        rh = rw @ rb.head_local
        ox = rh.x - cx; oy = rh.y - cy
        ol = math.hypot(ox, oy) or 1.0
        oxn = ox / ol; oyn = oy / ol
        dL = math.hypot(rh.x - pL.x, rh.y - pL.y)
        dR = math.hypot(rh.x - pR.x, rh.y - pR.y)
        wL = dR / (dL + dR + 1e-5); wR = dL / (dL + dR + 1e-5)
        rdxL, rdyL = rdxy["L"]; rdxR, rdyR = rdxy["R"]
        compassL = "((kxL-hxL-(%.4f))*%.4f+(kyL-hyL-(%.4f))*%.4f)" % (rdxL, oxn, rdyL, oyn)
        compassR = "((kxR-hxR-(%.4f))*%.4f+(kyR-hyR-(%.4f))*%.4f)" % (rdxR, oxn, rdyR, oyn)
        # the total swing (AMP) is SPLIT across the row segments and accumulates
        # down the chain -> a smooth progressive bend toward the hem (cloth-like).
        for rr, bn in rws:
            seg = rig.pose.bones.get("SKC_dt.%02d.%02d" % (ci, rr))
            if seg is None:
                continue
            seg.rotation_mode = 'XYZ'
            M3 = (rw @ seg.bone.matrix_local).to_3x3()
            Xl = M3.col[0]; Zl = M3.col[2]
            dotZ = Zl.x * oxn + Zl.y * oyn
            dotX = Xl.x * oxn + Xl.y * oyn
            if abs(dotZ) >= abs(dotX):
                idx = 0; sgn = 1.0 if dotZ > 0 else -1.0
            else:
                idx = 2; sgn = -1.0 if dotX > 0 else 1.0
            drv = seg.driver_add("rotation_euler", idx).driver
            drv.type = 'SCRIPTED'
            _locvar(drv, "kxL", knee["L"], 'LOC_X'); _locvar(drv, "kyL", knee["L"], 'LOC_Y')
            _locvar(drv, "hxL", hipb["L"], 'LOC_X'); _locvar(drv, "hyL", hipb["L"], 'LOC_Y')
            _locvar(drv, "kxR", knee["R"], 'LOC_X'); _locvar(drv, "kyR", knee["R"], 'LOC_Y')
            _locvar(drv, "hxR", hipb["R"], 'LOC_X'); _locvar(drv, "hyR", hipb["R"], 'LOC_Y')
            _mvar(drv, "spread", "collide_spread"); _mvar(drv, "col", "collide")
            _mvar(drv, "dd", "collide_dist")
            drv.expression = (
                "%.4f*min(1.2,max(0.0,%.4f*%s+%.4f*%s))*(dd/0.12)*min(1.5,spread)*col"
                % (sgn * AMP / nseg, wL, compassL, wR, compassR))
        n += 1
    _organize_skirt_bones(rig)
    return n


class SMARTRIG_OT_skirt_collision(bpy.types.Operator):
    bl_idname = "smartrig.skirt_collision"
    bl_label = "Apply Skirt Collision"
    bl_description = ("Add / refresh the constrained collision between the skirt and "
                      "the chosen leg bones on the generated rig.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None:
            self.report({'ERROR'}, "Generate the rig first.")
            return {'CANCELLED'}
        import smartrig_pro.fit as _fit
        h = None
        try:
            _, _e, h = _fit.compute_joints(context.scene.smartrig)
        except Exception:
            h = None
        p = context.scene.smartrig
        if not p.skirt_collide:
            r = remove_skirt_collision(rig)
            self.report({'INFO'}, "Skirt collision removed (%d constraints)." % r)
            return {'FINISHED'}
        n = add_skirt_collision(rig, p, h)
        if not n:
            self.report({'WARNING'}, "No skirt bones or no collider bones found.")
            return {'CANCELLED'}
        self.report({'INFO'}, "Skirt collision applied (%d constraints)." % n)
        return {'FINISHED'}


class SMARTRIG_OT_register_skirt(bpy.types.Operator):
    bl_idname = "smartrig.register_skirt"
    bl_label = "Register Skirt Selection"
    bl_description = ("Record the currently selected skirt faces/vertices (in Edit "
                      "Mode on the character) into the 'SR_Skirt' vertex group, used "
                      "to build the skirt bones.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        obj = props.target_mesh
        if obj is None and context.active_object and context.active_object.type == 'MESH':
            obj = context.active_object
            props.target_mesh = obj
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select the character mesh first.")
            return {'CANCELLED'}
        was_edit = (context.object is not None and context.object.mode == 'EDIT')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        sel = [v.index for v in obj.data.vertices if v.select]
        if not sel:
            self.report({'ERROR'},
                        "No vertices selected. Enter Edit Mode, select the skirt, then Register.")
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            return {'CANCELLED'}
        vg = obj.vertex_groups.get(VGROUP) or obj.vertex_groups.new(name=VGROUP)
        # clear then add
        vg.remove([v.index for v in obj.data.vertices])
        vg.add(sel, 1.0, 'REPLACE')
        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, "Registered %d skirt vertices." % len(sel))
        return {'FINISHED'}


class SMARTRIG_OT_add_skirt(bpy.types.Operator):
    bl_idname = "smartrig.add_skirt"
    bl_label = "Add Short Skirt"
    bl_description = ("Analyse the skirt mesh and build a ring of FK tentacle "
                      "chains from waist to hem, fitted to the real shape. "
                      "Adjust Columns/Rows, then Generate the rig.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(META_NAME) is not None

    def execute(self, context):
        props = context.scene.smartrig
        if getattr(props, "skirt_source", 'MERGED') == 'MANUAL':
            mo, err = build_manual_skirt(props)
        else:
            mo, err = build_skirt(props)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        self.report({'INFO'}, "Short skirt built (%d columns). Generate the rig next."
                    % max(4, int(props.skirt_columns)))
        return {'FINISHED'}


def _seg_dist(p, a, b):
    ab = b - a
    L2 = ab.dot(ab)
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, (p - a).dot(ab) / L2))
    return (p - (a + ab * t)).length


def _skirt_vids(props, mesh):
    vg = mesh.vertex_groups.get(VGROUP)
    if vg is None:
        return set()
    gi = vg.index
    out = set()
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == gi and g.weight > 0.0:
                out.add(v.index); break
    return out


def _weight_to_skirt(obj, segs, vids=None):
    for n, _a, _b in segs:
        if obj.vertex_groups.get(n) is None:
            obj.vertex_groups.new(name=n)
    mw = obj.matrix_world
    idxs = range(len(obj.data.vertices)) if vids is None else vids
    for vi in idxs:
        p = mw @ obj.data.vertices[vi].co
        d = sorted(((_seg_dist(p, a, b), n) for n, a, b in segs))[:2]
        ws = [(n, 1.0 / (dist + 1e-4)) for dist, n in d]
        tot = sum(w for _, w in ws) or 1.0
        for n, w in ws:
            obj.vertex_groups[n].add([vi], w / tot, 'REPLACE')


def _smart_skirt_weights(obj, rig, vids=None):
    """Structure-aware skirt skinning. Uses the known skirt grid: weight each vertex
    to the 2 nearest COLUMNS by azimuth (angular blend -> no cross-column bleed) and,
    within each column, to the nearest 1-2 row SEGMENTS (inverse distance). Beats a
    generic heat map on thin cloth. Returns True if it ran."""
    grid = {}
    for b in rig.data.bones:
        m = re.match(r"^DEF-" + PREFIX + r"\.(\d+)\.(\d+)$", b.name)
        if m:
            grid.setdefault(int(m.group(1)), {})[int(m.group(2))] = b.name
    if not grid:
        return False
    rw = rig.matrix_world
    cols = sorted(grid)
    tops = {ci: rw @ rig.data.bones[grid[ci][min(grid[ci])]].head_local for ci in cols}
    cx = sum(tops[ci].x for ci in cols) / len(cols)
    cy = sum(tops[ci].y for ci in cols) / len(cols)
    az = {ci: math.atan2(tops[ci].y - cy, tops[ci].x - cx) for ci in cols}
    seg = {}
    for ci in cols:
        seg[ci] = [(grid[ci][rr],
                    rw @ rig.data.bones[grid[ci][rr]].head_local,
                    rw @ rig.data.bones[grid[ci][rr]].tail_local) for rr in sorted(grid[ci])]
    allbones = [bn for ci in cols for bn, _, _ in seg[ci]]
    for bn in allbones:
        if obj.vertex_groups.get(bn) is None:
            obj.vertex_groups.new(name=bn)
    idxs = list(range(len(obj.data.vertices))) if vids is None else list(vids)
    for bn in allbones:
        try:
            obj.vertex_groups[bn].remove(idxs)
        except Exception:
            pass
    mw = obj.matrix_world

    def adist(a, ci):
        return abs(((a - az[ci] + math.pi) % (2.0 * math.pi)) - math.pi)

    for vi in idxs:
        p = mw @ obj.data.vertices[vi].co
        a = math.atan2(p.y - cy, p.x - cx)
        nb = sorted(cols, key=lambda ci: adist(a, ci))[:2]
        c0, c1 = nb[0], nb[1]
        d0 = adist(a, c0); d1 = adist(a, c1)
        wA = {c0: d1 / (d0 + d1 + 1e-6), c1: d0 / (d0 + d1 + 1e-6)}
        for ci, wcol in wA.items():
            ds = sorted(((_seg_dist(p, h, t), bn) for bn, h, t in seg[ci]))[:2]
            inv = [(1.0 / (d + 1e-4), bn) for d, bn in ds]
            tot = sum(w for w, _ in inv) or 1.0
            for w, bn in inv:
                obj.vertex_groups[bn].add([vi], wcol * w / tot, 'ADD')
    return True


def bind_mesh(props, context):
    """Bind the body to the rig. Skirt bones are EXCLUDED from the body solve so
    the body never gets skirt weights; the skirt is weighted only to its own
    bones. Existing armature modifiers / deform groups are removed first to avoid
    a double bind (which corrupts the body shape)."""
    from .metarig import _generated_rig
    rig = _generated_rig()
    if rig is None:
        return None, "Generate the rig first, then bind."
    mesh = props.target_mesh
    if mesh is None or mesh.type != 'MESH':
        return None, "Select the character mesh first."

    skirt_bones = [b.name for b in rig.data.bones if b.name.startswith("DEF-" + PREFIX + ".")]
    has_skirt = bool(skirt_bones)
    rw = rig.matrix_world
    segs = [(n, rw @ rig.data.bones[n].head_local, rw @ rig.data.bones[n].tail_local)
            for n in skirt_bones]
    sep = props.skirt_object if props.skirt_source == 'SEPARATE' else None
    skirt_vids = set() if sep is not None else (_skirt_vids(props, mesh) if has_skirt else set())
    split = bool(props.skin_split_parts) and has_skirt
    if split and sep is None and not skirt_vids:
        return None, ("Tell the addon where the skirt is: select the skirt faces in "
                      "Edit Mode and press 'Register Skirt Selection'.")

    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    ptype = 'ARMATURE_ENVELOPE' if props.skin_engine == 'ENVELOPE' else 'ARMATURE_AUTO'

    def _clean(ob):
        for m in list(ob.modifiers):
            if m.type == 'ARMATURE':
                ob.modifiers.remove(m)
        if ob.parent is not None and ob.parent.type == 'ARMATURE':
            mw = ob.matrix_world.copy(); ob.parent = None; ob.matrix_world = mw
        for vg in list(ob.vertex_groups):
            if vg.name.startswith("DEF-"):
                ob.vertex_groups.remove(vg)

    def _parent_auto(ob):
        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True); rig.select_set(True)
        context.view_layer.objects.active = rig
        _vis = []
        try:
            for coll in rig.data.collections_all:
                _vis.append((coll, coll.is_visible)); coll.is_visible = True
        except Exception:
            try:
                for coll in rig.data.collections:
                    _vis.append((coll, coll.is_visible)); coll.is_visible = True
            except Exception:
                pass
        _win = context.window
        _area = next((a for a in _win.screen.areas if a.type == 'VIEW_3D'), None) if _win else None
        _region = next((r for r in _area.regions if r.type == 'WINDOW'), None) if _area else None
        _ov = dict(active_object=rig, object=rig,
                   selected_objects=[ob, rig], selected_editable_objects=[ob, rig])
        if _win:
            _ov["window"] = _win
        if _area:
            _ov["area"] = _area
        if _region:
            _ov["region"] = _region
        try:
            with context.temp_override(**_ov):
                bpy.ops.object.parent_set(type=ptype)
        except Exception:
            bpy.ops.object.parent_set(type=ptype)
        for coll, vis in _vis:
            try:
                coll.is_visible = vis
            except Exception:
                pass

    _clean(mesh)
    saved = {}
    if split:
        for n in skirt_bones:
            bd = rig.data.bones.get(n)
            if bd is not None:
                saved[n] = bd.use_deform; bd.use_deform = False
    _parent_auto(mesh)
    for n, v in saved.items():
        bd = rig.data.bones.get(n)
        if bd is not None:
            bd.use_deform = v
    for m in mesh.modifiers:
        if m.type == 'ARMATURE':
            m.use_deform_preserve_volume = bool(props.skin_preserve_volume)

    if split:
        body_groups = [vg for vg in mesh.vertex_groups if vg.name.startswith("DEF-")]
        smart = bool(getattr(props, "skin_smart_skirt", True))
        if sep is None:
            for vi in skirt_vids:
                for g in body_groups:
                    try:
                        g.remove([vi])
                    except Exception:
                        pass
            if not (smart and _smart_skirt_weights(mesh, rig, skirt_vids)):
                _weight_to_skirt(mesh, segs, skirt_vids)
        else:
            _clean(sep)
            done = False
            if smart:
                done = _smart_skirt_weights(sep, rig, None)
            if not done:
                # heat-bind to ONLY the skirt bones (disable non-skirt deform bones)
                _saved2 = {}
                for b in rig.data.bones:
                    if b.use_deform and not b.name.startswith("DEF-" + PREFIX + "."):
                        _saved2[b.name] = b.use_deform; b.use_deform = False
                _parent_auto(sep)
                for n2, v2 in _saved2.items():
                    bd = rig.data.bones.get(n2)
                    if bd is not None:
                        bd.use_deform = v2
                if not any(vg.name.startswith("DEF-" + PREFIX + ".") for vg in sep.vertex_groups):
                    _weight_to_skirt(sep, segs, None)
            # ensure the separate skirt is parented + has an armature modifier
            if sep.parent != rig:
                sep.parent = rig
                sep.matrix_parent_inverse = rig.matrix_world.inverted()
            if not any(m.type == 'ARMATURE' for m in sep.modifiers):
                sep.modifiers.new("Armature", 'ARMATURE').object = rig
            for m in sep.modifiers:
                if m.type == 'ARMATURE':
                    m.use_deform_preserve_volume = bool(props.skin_preserve_volume)

    try:
        bpy.ops.object.select_all(action='DESELECT')
        mesh.select_set(True); context.view_layer.objects.active = mesh
        bpy.ops.object.vertex_group_normalize_all(group_select_mode='BONE_DEFORM', lock_active=False)
    except Exception:
        pass

    if split:
        _sk = "smart-grid skirt weights" if bool(getattr(props, "skin_smart_skirt", True)) else props.skin_engine.title()
        return ("Bound. Body=%s; skirt=%s (own bones only)."
                % (props.skin_engine.title(), _sk)), None
    return "Bound the body to the rig (%s)." % props.skin_engine.title(), None


def unbind_mesh(props, context):
    mesh = props.target_mesh
    objs = [o for o in (mesh, (props.skirt_object if props.skirt_source == 'SEPARATE' else None)) if o]
    if not objs:
        return None, "Select the character mesh first."
    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    n = 0
    for ob in objs:
        for m in list(ob.modifiers):
            if m.type == 'ARMATURE':
                ob.modifiers.remove(m); n += 1
        if ob.parent is not None and ob.parent.type == 'ARMATURE':
            mw = ob.matrix_world.copy(); ob.parent = None; ob.matrix_world = mw
        for vg in list(ob.vertex_groups):
            if vg.name.startswith("DEF-"):
                ob.vertex_groups.remove(vg)
    return "Unbound (removed %d armature modifier(s) + deform groups)." % n, None


class SMARTRIG_OT_bind(bpy.types.Operator):
    bl_idname = "smartrig.bind"
    bl_label = "Bind"
    bl_description = ("Bind the mesh to the rig. With Split Parts on, the body ignores "
                      "skirt bones and the skirt follows only its own bones.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        msg, err = bind_mesh(context.scene.smartrig, context)
        if err:
            self.report({'ERROR'}, err); return {'CANCELLED'}
        self.report({'INFO'}, msg); return {'FINISHED'}


class SMARTRIG_OT_unbind(bpy.types.Operator):
    bl_idname = "smartrig.unbind"
    bl_label = "Unbind"
    bl_description = "Remove the bind (armature modifiers, parenting and deform vertex groups)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        msg, err = unbind_mesh(context.scene.smartrig, context)
        if err:
            self.report({'ERROR'}, err); return {'CANCELLED'}
        self.report({'INFO'}, msg); return {'FINISHED'}


class SMARTRIG_OT_skirt_jiggle(bpy.types.Operator):
    bl_idname = "smartrig.skirt_jiggle"
    bl_label = "Apply Skirt Jiggle"
    bl_description = ("Add live spring jiggle to the skirt (secondary motion). "
                     "Play the timeline to see it sway.")
    bl_options = {'REGISTER', 'UNDO'}
    remove: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None:
            self.report({'ERROR'}, "Generate the rig first."); return {'CANCELLED'}
        if self.remove:
            remove_skirt_jiggle(rig)
            self.report({'INFO'}, "Skirt jiggle removed."); return {'FINISHED'}
        n = add_skirt_jiggle(rig, context.scene.smartrig)
        if not n:
            self.report({'WARNING'}, "No skirt bones found."); return {'CANCELLED'}
        self.report({'INFO'}, "Skirt jiggle applied (%d columns). Play the timeline." % n)
        return {'FINISHED'}


class SMARTRIG_OT_bake_jiggle(bpy.types.Operator):
    bl_idname = "smartrig.bake_jiggle"
    bl_label = "Bake Jiggle to Keyframes"
    bl_description = ("Bake the live jiggle of the current frame range onto keyframes "
                     "(the live solver then stops; re-Apply Jiggle to go live again).")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None or not rig.get("sk_jiggle"):
            self.report({'ERROR'}, "Apply jiggle first."); return {'CANCELLED'}
        sc = context.scene
        if rig.mode != 'POSE':
            context.view_layer.objects.active = rig; bpy.ops.object.mode_set(mode='POSE')
        jigs = [pb for pb in rig.pose.bones if pb.name.startswith("SKC_jig")]
        for pb in jigs:
            pb.rotation_mode = 'QUATERNION'
        if "sk_jiggle_baked" in rig:
            del rig["sk_jiggle_baked"]
        for f in range(sc.frame_start, sc.frame_end + 1):
            sc.frame_set(f)   # spring handler runs and poses the jig bones
            for pb in jigs:
                pb.keyframe_insert("rotation_quaternion", frame=f)
        rig["sk_jiggle_baked"] = 1   # handler now skips this rig; keyframes play it back
        self.report({'INFO'}, "Baked jiggle %d-%d." % (sc.frame_start, sc.frame_end))
        return {'FINISHED'}


class SMARTRIG_OT_skirt_follow(bpy.types.Operator):
    bl_idname = "smartrig.skirt_follow"
    bl_label = "Apply Body Follow"
    bl_description = ("Add a blendable 'Follow Body' to the skirt (great for sitting): "
                     "the Follow Body slider blends from the skirt rig to following the legs/hips.")
    bl_options = {'REGISTER', 'UNDO'}
    remove: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None:
            self.report({'ERROR'}, "Generate the rig first."); return {'CANCELLED'}
        _ensure_drivers_trusted()
        if self.remove:
            remove_skirt_follow_body(rig)
            self.report({'INFO'}, "Body follow removed."); return {'FINISHED'}
        n = add_skirt_follow_body(rig, context.scene.smartrig)
        if not n:
            self.report({'WARNING'}, "No skirt bones found."); return {'CANCELLED'}
        self.report({'INFO'}, "Body follow applied (%d columns). Use the Follow Body slider." % n)
        return {'FINISHED'}


classes = (SMARTRIG_OT_register_skirt, SMARTRIG_OT_add_skirt,
           SMARTRIG_OT_bind, SMARTRIG_OT_unbind, SMARTRIG_OT_skirt_collision,
           SMARTRIG_OT_skirt_jiggle, SMARTRIG_OT_bake_jiggle,
           SMARTRIG_OT_skirt_follow)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    # re-arm the live jiggle handler if a jiggle rig is present (e.g. after reopen)
    try:
        if any(o.type == 'ARMATURE' and o.get("sk_jiggle") and not o.get("sk_jiggle_baked")
               for o in bpy.data.objects):
            register_jiggle_handler()
    except Exception:
        pass


def unregister():
    unregister_jiggle_handler()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
