"""Viewport overlay: glowing markers (always) + reference image (bottom-left)."""
import bpy
import os
import math
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from . import markers

LABELS = {
    "spine_root": "Spine Root", "neck": "Neck", "head_top": "Head",
    "shoulder.L": "Shoulder L", "shoulder.R": "Shoulder R",
    "elbow.L": "Elbow L", "elbow.R": "Elbow R",
    "hip.L": "Hip L", "hip.R": "Hip R",
    "knee.L": "Knee L", "knee.R": "Knee R",
    "wrist.L": "Wrist L", "wrist.R": "Wrist R",
    "ankle.L": "Ankle L", "ankle.R": "Ankle R",
}

_OVL = {"handle": None, "texture": None, "image": None}

GLOW = {"center": (0.18, 0.82, 1.0), "left": (1.0, 0.75, 0.2), "right": (0.6, 0.5, 0.25)}


def _role(name):
    return "right" if name.endswith(".R") else "left" if name.endswith(".L") else "center"


def _circle(cx, cy, r, seg=28):
    pts = [(cx, cy)]
    for i in range(seg + 1):
        a = 2 * math.pi * i / seg
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _ring(cx, cy, r, seg=28):
    return [(cx + r * math.cos(2 * math.pi * i / seg),
             cy + r * math.sin(2 * math.pi * i / seg)) for i in range(seg + 1)]


SEL_COLOR = (1.0, 0.95, 0.2)   # bright yellow selection highlight


def _selected(o):
    try:
        return o.select_get()
    except Exception:
        return False


LINE_GROUPS = [
    ((0.18, 0.82, 1.0), [("spine_root", "neck"), ("neck", "head_top")]),
    ((1.0, 0.75, 0.2), [("neck", "shoulder.L"), ("shoulder.L", "elbow.L"), ("elbow.L", "wrist.L"),
                        ("spine_root", "hip.L"), ("hip.L", "knee.L"), ("knee.L", "ankle.L")]),
    ((0.55, 0.45, 0.22), [("neck", "shoulder.R"), ("shoulder.R", "elbow.R"), ("elbow.R", "wrist.R"),
                          ("spine_root", "hip.R"), ("hip.R", "knee.R"), ("knee.R", "ankle.R")]),
]


def _p2(region, rv3d, name):
    o = bpy.data.objects.get(name)
    if not o or o.hide_get():
        return None
    return view3d_utils.location_3d_to_region_2d(region, rv3d, o.matrix_world.translation)


def _draw_lines(region, rv3d):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.5)
    groups = list(LINE_GROUPS)
    for fn in markers.FINGER_NAMES:
        groups.append(((0.3, 1.0, 0.45), [("wrist.L", "ftip_%s.L" % fn)]))
        groups.append(((0.2, 0.5, 0.3), [("wrist.R", "ftip_%s.R" % fn)]))
    # foot markers connect to the ankle (ankle -> ball -> toe tip), like the rest
    groups.append(((1.0, 0.75, 0.2), [("ankle.L", "ball.L"), ("ball.L", "foottip.L")]))
    groups.append(((0.55, 0.45, 0.22), [("ankle.R", "ball.R"), ("ball.R", "foottip.R")]))
    for col, pairs in groups:
        verts = []
        for a, b in pairs:
            pa = _p2(region, rv3d, a); pb = _p2(region, rv3d, b)
            if pa and pb:
                verts += [(pa.x, pa.y), (pb.x, pb.y)]
        if verts:
            batch = batch_for_shader(shader, 'LINES', {"pos": verts})
            shader.bind(); shader.uniform_float("color", (col[0], col[1], col[2], 0.9))
            batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _marker_mult():
    try:
        return float(bpy.context.scene.smartrig.marker_size)
    except Exception:
        return 1.0


def _face_marker_items():
    """(object, role) for the Face markers - same glow system."""
    items = []
    try:
        fcol = bpy.data.collections.get("SR_FaceMarkers")
        if fcol is not None and not fcol.hide_viewport:
            for o in fcol.objects:
                if not o.name.startswith("face_"):
                    continue
                try:
                    if o.hide_get():
                        continue
                except Exception:
                    pass
                items.append((o, _role(o.name)))
    except Exception:
        pass
    return items


def _draw_glow(region, rv3d):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    m = _marker_mult()
    items = [(bpy.data.objects.get(nm), None) for nm in markers.all_marker_names()]
    items += _face_marker_items()
    for o, role in items:
        if not o or o.hide_get():
            continue
        name = o.name
        p = view3d_utils.location_3d_to_region_2d(region, rv3d, o.matrix_world.translation)
        if not p:
            continue
        col = GLOW[role if role is not None else _role(name)]
        sel = _selected(o)
        # fit markers never shrink below full size (the rig marker_size
        # preference made them vanish)
        mm = m if role is None else max(m, 1.0)
        if name.startswith("face_"):
            mm *= 0.55                 # face markers sit close together
        s = (1.5 if sel else 1.0) * mm
        for rr, aa in ((27 * s, 0.12), (17 * s, 0.24), (10 * s, 0.60)):
            b = batch_for_shader(shader, 'TRI_FAN', {"pos": _circle(p.x, p.y, rr)})
            shader.bind(); shader.uniform_float("color", (col[0], col[1], col[2], aa)); b.draw(shader)
        core = SEL_COLOR if sel else (1, 1, 1)
        b = batch_for_shader(shader, 'TRI_FAN', {"pos": _circle(p.x, p.y, 4.5 * s)})
        shader.bind(); shader.uniform_float("color", (core[0], core[1], core[2], 0.98)); b.draw(shader)
        if sel:
            gpu.state.line_width_set(2.5)
            rb = batch_for_shader(shader, 'LINE_STRIP', {"pos": _ring(p.x, p.y, 24 * s)})
            shader.bind(); shader.uniform_float("color", (SEL_COLOR[0], SEL_COLOR[1], SEL_COLOR[2], 1.0)); rb.draw(shader)
            gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_labels(region, rv3d):
    fid = 0
    blf.enable(fid, blf.SHADOW)
    blf.shadow(fid, 5, 0, 0, 0, 1)
    blf.size(fid, 17)
    for name, label in LABELS.items():
        o = bpy.data.objects.get(name)
        if not o:
            continue
        p = view3d_utils.location_3d_to_region_2d(region, rv3d, o.matrix_world.translation)
        if not p:
            continue
        col = GLOW[_role(name)]
        blf.color(fid, min(col[0] + 0.3, 1), min(col[1] + 0.3, 1), min(col[2] + 0.3, 1), 1)
        blf.position(fid, p.x + 12, p.y + 8, 0)
        blf.draw(fid, label)


def _draw_face_labels(region, rv3d):
    fid = 0
    blf.enable(fid, blf.SHADOW)
    blf.shadow(fid, 5, 0, 0, 0, 1)
    blf.size(fid, 14)
    for o, role in _face_marker_items():
        if not _selected(o):
            continue
        p = view3d_utils.location_3d_to_region_2d(region, rv3d,
                                                  o.matrix_world.translation)
        if not p:
            continue
        col = GLOW[role]
        blf.color(fid, min(col[0] + 0.3, 1), min(col[1] + 0.3, 1),
                  min(col[2] + 0.3, 1), 1)
        blf.position(fid, p.x + 10, p.y + 7, 0)
        key = o.name[len("face_"):]
        blf.draw(fid, key.replace("_", " ").replace(".L", " L")
                 .replace(".R", " R").title())


def _draw_fingers(region, rv3d):
    """Glow + connecting lines for the manual finger/palm markers (fm.*)."""
    from . import fingers_manual as _fm
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    m = max(_marker_mult(), 1.0)
    for part in ("palm", "hand", "foot"):
        for side in ("L", "R"):
            try:
                chains = _fm.list_fingers(part, side)
            except Exception:
                continue
            for _fn, chain in chains.items():
                # each digit gets its OWN colour (thumb red, index orange,
                # middle yellow, ring green, pinky blue - same as the empties)
                col = _fm.color_for(_fn)
                pts2 = []
                for o in chain:
                    try:
                        if o.hide_get():
                            continue
                    except Exception:
                        pass
                    p = view3d_utils.location_3d_to_region_2d(
                        region, rv3d, o.matrix_world.translation)
                    if not p:
                        continue
                    pts2.append(p)
                    sel = _selected(o)
                    s2 = (1.4 if sel else 0.8) * m
                    for rr, aa in ((14 * s2, 0.18), (8 * s2, 0.45)):
                        b = batch_for_shader(shader, 'TRI_FAN',
                                             {"pos": _circle(p.x, p.y, rr)})
                        shader.bind()
                        shader.uniform_float("color", (col[0], col[1], col[2], aa))
                        b.draw(shader)
                    core = SEL_COLOR if sel else (1, 1, 1)
                    b = batch_for_shader(shader, 'TRI_FAN',
                                         {"pos": _circle(p.x, p.y, 3.2 * s2)})
                    shader.bind()
                    shader.uniform_float("color", (core[0], core[1], core[2], 0.98))
                    b.draw(shader)
                if len(pts2) >= 2:
                    gpu.state.line_width_set(1.6)
                    b = batch_for_shader(shader, 'LINE_STRIP',
                                         {"pos": [(p.x, p.y) for p in pts2]})
                    shader.bind()
                    shader.uniform_float("color", (col[0], col[1], col[2], 0.85))
                    b.draw(shader)
                    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_face_grid(region, rv3d):
    """FaceIt-style net for SR_FaceGrid: BLACK dots for the structure, ORANGE
    for the eyelid + lip rings, and BIG RED dots on the eye corners and the
    mouth corners so the user knows exactly where the key points belong."""
    ob = bpy.data.objects.get("SR_FaceGrid")
    if ob is None or not ob.visible_get():
        return
    from . import face as _fc
    from mathutils import Vector
    mw = ob.matrix_world

    RING = {"lip_T", "lip_B", "lip_T.L", "lip_B.L",
            "lid_T_in.L", "lid_T.L", "lid_T_out.L",
            "lid_B_in.L", "lid_B.L", "lid_B_out.L"}
    CORNER = {"eye_in.L", "eye_out.L", "mouth_corner.L"}

    # 2D positions per template name (draw the .R mirror ourselves - the
    # mirror-modifier vert order is not name-addressable)
    p2 = {}
    try:
        # use EDIT-MODE live coords when the user is editing the net
        if ob.mode == 'EDIT':
            import bmesh
            bm = bmesh.from_edit_mesh(ob.data)
            cos = [v.co.copy() for v in bm.verts]
        else:
            cos = [v.co for v in ob.data.vertices]
    except Exception:
        cos = [v.co for v in ob.data.vertices]
    for name, i in _fc.GRID_IDX.items():
        if i >= len(cos):
            continue
        w = mw @ cos[i]
        p2[name] = view3d_utils.location_3d_to_region_2d(region, rv3d, w)
        if name.endswith(".L"):
            wm = Vector((-w.x, w.y, w.z))
            p2[name[:-2] + ".R"] = view3d_utils.location_3d_to_region_2d(
                region, rv3d, wm)

    def _sides(nm):
        if nm.endswith(".L"):
            return [nm, nm[:-2] + ".R"]
        return [nm]

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    # ---- edges: ring edges orange, the rest dark (like FaceIt) ----
    for col, want_ring in (((0.07, 0.07, 0.07, 0.9), False),
                           ((1.0, 0.55, 0.1, 0.95), True)):
        lines = []
        for a, b in _fc._TE:
            ring_edge = (a in RING or a in CORNER) and (b in RING or b in CORNER)
            if ring_edge != want_ring:
                continue
            for sa, sb in zip(_sides(a) if a.endswith(".L") else _sides(a) * 2,
                              _sides(b) if b.endswith(".L") else _sides(b) * 2):
                pa, pb = p2.get(sa), p2.get(sb)
                if pa and pb:
                    lines.append((pa.x, pa.y)); lines.append((pb.x, pb.y))
        if lines:
            gpu.state.line_width_set(1.6 if want_ring else 1.2)
            b = batch_for_shader(shader, 'LINES', {"pos": lines})
            shader.bind(); shader.uniform_float("color", col); b.draw(shader)
    gpu.state.line_width_set(1.0)

    # ---- dots ----
    def dot(p, r, col):
        b = batch_for_shader(shader, 'TRI_FAN', {"pos": _circle(p.x, p.y, r)})
        shader.bind(); shader.uniform_float("color", col); b.draw(shader)

    for name in list(_fc.GRID_IDX):
        for nm in _sides(name):
            p = p2.get(nm)
            if not p:
                continue
            base = name if not nm.endswith(".R") else name
            if name in CORNER:
                # KEY corner: big red dot with a white halo - the user must
                # put these exactly on the eye / mouth corners
                dot(p, 7.5, (1.0, 1.0, 1.0, 0.95))
                dot(p, 5.2, (0.95, 0.12, 0.1, 1.0))
            elif name in RING:
                dot(p, 4.6, (0.04, 0.04, 0.04, 0.95))
                dot(p, 3.2, (1.0, 0.55, 0.1, 1.0))
            else:
                dot(p, 4.4, (0.04, 0.04, 0.04, 0.95))
                dot(p, 2.0, (0.25, 0.25, 0.25, 1.0))
    gpu.state.blend_set('NONE')


def _draw_cb():
    try:
        region = bpy.context.region
        rv3d = bpy.context.region_data
        if region is None or rv3d is None:
            return
        from . import fingers_manual
        has_body = any(bpy.data.objects.get(n) for n in markers.all_marker_names())
        has_fing = fingers_manual.has_manual(side="L") or fingers_manual.has_manual(side="R")
        has_face = bool(_face_marker_items())
        if not has_body and not has_fing and not has_face:
            return
        if has_body:
            _draw_lines(region, rv3d)
        if has_body or has_face:
            _draw_glow(region, rv3d)
        if has_face:
            _draw_face_labels(region, rv3d)
        _draw_face_grid(region, rv3d)
        if has_fing:
            _draw_fingers(region, rv3d)
    except Exception:
        pass


def register():
    if _OVL["handle"] is None:
        _OVL["handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw_cb, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    if _OVL["handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_OVL["handle"], 'WINDOW')
        except Exception:
            pass
        _OVL["handle"] = None
    _OVL["texture"] = None
    _OVL["image"] = None
