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


def _fit_marker_items():
    """(object, role) for the Fit Wizard markers - same glow system."""
    items = []
    try:
        from . import fit_wizard as _fw
        fcol = bpy.data.collections.get(_fw.MARKER_COL)
        if fcol is not None and not fcol.hide_viewport:
            for o in fcol.objects:
                if o.name.startswith(_fw.MARKER_PREFIX):
                    key = o.name[len(_fw.MARKER_PREFIX):].split(".")[0]
                    items.append((o, _fw._role(key)))
    except Exception:
        pass
    return items


def _draw_fit_chains(shader, region, rv3d):
    """Connecting lines between the Fit Wizard markers (skeleton chains +
    the chest/waist WIDTH spans) - same look as the character wizard."""
    try:
        from . import fit_wizard as _fw
        fcol = bpy.data.collections.get(_fw.MARKER_COL)
        if fcol is None or fcol.hide_viewport:
            return
        pts = []
        for a, b in _fw.CHAINS:
            oa = bpy.data.objects.get(_fw.MARKER_PREFIX + a)
            ob = bpy.data.objects.get(_fw.MARKER_PREFIX + b)
            if oa is None or ob is None:
                continue
            pa = view3d_utils.location_3d_to_region_2d(
                region, rv3d, oa.matrix_world.translation)
            pb = view3d_utils.location_3d_to_region_2d(
                region, rv3d, ob.matrix_world.translation)
            if pa and pb:
                pts += [(pa.x, pa.y), (pb.x, pb.y)]
        if not pts:
            return
        gpu.state.line_width_set(2.0)
        batch = batch_for_shader(shader, 'LINES', {"pos": pts})
        shader.bind()
        shader.uniform_float("color", (0.9, 0.95, 1.0, 0.45))
        batch.draw(shader)
        gpu.state.line_width_set(1.0)
    except Exception:
        pass


def _draw_glow(region, rv3d):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    m = _marker_mult()
    _draw_fit_chains(shader, region, rv3d)
    items = [(bpy.data.objects.get(nm), None) for nm in markers.all_marker_names()]
    items += _fit_marker_items()
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
    # Fit Wizard marker labels (same coloured system)
    for o, role in _fit_marker_items():
        p = view3d_utils.location_3d_to_region_2d(region, rv3d,
                                                  o.matrix_world.translation)
        if not p:
            continue
        col = GLOW[role]
        blf.color(fid, min(col[0] + 0.3, 1), min(col[1] + 0.3, 1),
                  min(col[2] + 0.3, 1), 1)
        blf.position(fid, p.x + 12, p.y + 8, 0)
        key = o.name.split("SRFM_", 1)[-1]
        blf.draw(fid, key.replace("_l", " L").replace("_r", " R")
                 .replace("_", " ").title())
    # fingertip labels (smaller, green)
    blf.size(fid, 13)
    blf.color(fid, 0.5, 1.0, 0.6, 1)
    for fn in markers.FINGER_NAMES:
        for s in (".L", ".R"):
            o = bpy.data.objects.get("ftip_%s%s" % (fn, s))
            if not o:
                continue
            p = view3d_utils.location_3d_to_region_2d(region, rv3d, o.matrix_world.translation)
            if not p:
                continue
            blf.position(fid, p.x + 9, p.y + 5, 0)
            blf.draw(fid, fn.capitalize() + " " + s[-1])
    blf.disable(fid, blf.SHADOW)


def _texture():
    if _OVL["texture"] is not None:
        return _OVL["texture"]
    fp = os.path.join(os.path.dirname(__file__), "assets", "guide_all.png")
    if not os.path.exists(fp):
        return None
    try:
        img = bpy.data.images.load(fp, check_existing=True)
        _OVL["image"] = img
        _OVL["texture"] = gpu.texture.from_image(img)
    except Exception:
        _OVL["texture"] = None
    return _OVL["texture"]


def _draw_guide(region):
    tex = _texture()
    if tex is None:
        return
    iw = 300
    ih = int(iw * 600.0 / 460.0)
    x = 84                      # clear of the left toolbar
    y = 24                      # just above the bottom
    try:
        shader = gpu.shader.from_builtin('IMAGE')
        pos = [(x, y), (x + iw, y), (x + iw, y + ih), (x, y + ih)]
        uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
        b = batch_for_shader(shader, 'TRI_FAN', {"pos": pos, "texCoord": uv})
        gpu.state.blend_set('ALPHA')
        shader.bind(); shader.uniform_sampler("image", tex); b.draw(shader)
        gpu.state.blend_set('NONE')
    except Exception:
        pass


PROMPT = {"text": None, "sub": None}


def _draw_prompt(region):
    if not PROMPT.get("text"):
        return
    fid = 0
    blf.enable(fid, blf.SHADOW)
    blf.shadow(fid, 5, 0, 0, 0, 1)
    blf.size(fid, 30)
    blf.color(fid, 0.2, 0.9, 1.0, 1.0)
    txt = PROMPT["text"]
    w, _hh = blf.dimensions(fid, txt)
    blf.position(fid, max(10, (region.width - w) / 2), region.height - 72, 0)
    blf.draw(fid, txt)
    sub = PROMPT.get("sub")
    if sub:
        blf.size(fid, 15)
        blf.color(fid, 0.85, 0.85, 0.85, 1.0)
        w2, _ = blf.dimensions(fid, sub)
        blf.position(fid, max(10, (region.width - w2) / 2), region.height - 98, 0)
        blf.draw(fid, sub)
    blf.disable(fid, blf.SHADOW)


# ---- guided reference panel (ARP-style square thumbnail + clickable hints) ---
GUIDE = {"active": False, "img": None, "tex": None, "uvs": [], "labels": [],
         "step": 0, "btn_cancel": None, "btn_back": None}


def _guide_tex():
    if GUIDE["tex"] is not None:
        return GUIDE["tex"]
    img = bpy.data.images.get(GUIDE.get("img") or "")
    if img is None:
        return None
    try:
        GUIDE["tex"] = gpu.texture.from_image(img)
    except Exception:
        GUIDE["tex"] = None
    return GUIDE["tex"]


def _icon_x(cx, cy, col):
    sh = gpu.shader.from_builtin('UNIFORM_COLOR'); gpu.state.blend_set('ALPHA'); gpu.state.line_width_set(2.0)
    s = 5
    b = batch_for_shader(sh, 'LINES', {"pos": [(cx-s, cy-s), (cx+s, cy+s), (cx-s, cy+s), (cx+s, cy-s)]})
    sh.bind(); sh.uniform_float("color", col); b.draw(sh)
    gpu.state.line_width_set(1.0); gpu.state.blend_set('NONE')


def _icon_back(cx, cy, col):
    sh = gpu.shader.from_builtin('UNIFORM_COLOR'); gpu.state.blend_set('ALPHA'); gpu.state.line_width_set(2.0)
    s = 6
    b = batch_for_shader(sh, 'LINES', {"pos": [(cx-s, cy), (cx+s, cy), (cx-s, cy), (cx-s+5, cy+5), (cx-s, cy), (cx-s+5, cy-5)]})
    sh.bind(); sh.uniform_float("color", col); b.draw(sh)
    gpu.state.line_width_set(1.0); gpu.state.blend_set('NONE')


def _draw_guide_panel(region):
    if not GUIDE.get("active"):
        return
    step = GUIDE["step"]
    if step >= len(GUIDE["labels"]):
        return
    fid = 0
    sz = 200
    cx = region.width // 2
    x = cx - sz // 2
    y = 52
    # label  (e.g. "ADD HEAD   1/9")
    blf.enable(fid, blf.SHADOW); blf.shadow(fid, 5, 0, 0, 0, 1)
    blf.size(fid, 24); blf.color(fid, 0.2, 0.9, 1.0, 1.0)
    lbl = GUIDE["labels"][step]
    w, _ = blf.dimensions(fid, lbl)
    blf.position(fid, cx - w / 2, y + sz + 16, 0); blf.draw(fid, lbl)
    # square thumbnail, UV-cropped to this joint (zoom into head / neck / ...)
    tex = _guide_tex()
    dot = None
    if tex is not None and step < len(GUIDE["uvs"]):
        uc, vc = GUIDE["uvs"][step]
        r = 0.20
        u0 = min(max(uc - r, 0.0), 1.0 - 2 * r); v0 = min(max(vc - r, 0.0), 1.0 - 2 * r)
        u1, v1 = u0 + 2 * r, v0 + 2 * r
        try:
            sh2 = gpu.shader.from_builtin('UNIFORM_COLOR'); gpu.state.blend_set('ALPHA')
            fr = batch_for_shader(sh2, 'TRI_FAN', {"pos": [(x-3, y-3), (x+sz+3, y-3), (x+sz+3, y+sz+3), (x-3, y+sz+3)]})
            sh2.bind(); sh2.uniform_float("color", (0.12, 0.45, 0.55, 0.92)); fr.draw(sh2)
            shader = gpu.shader.from_builtin('IMAGE')
            pos = [(x, y), (x + sz, y), (x + sz, y + sz), (x, y + sz)]
            uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            b = batch_for_shader(shader, 'TRI_FAN', {"pos": pos, "texCoord": uv})
            shader.bind(); shader.uniform_sampler("image", tex); b.draw(shader)
            gpu.state.blend_set('NONE')
            dot = (x + (uc - u0) / (u1 - u0) * sz, y + (vc - v0) / (v1 - v0) * sz)
        except Exception:
            pass
    # the BEAUTIFUL marker dot, exactly on the joint (like a real marker)
    if dot:
        dx, dy = dot
        sh2 = gpu.shader.from_builtin('UNIFORM_COLOR'); gpu.state.blend_set('ALPHA')
        for rr, aa in ((16, 0.16), (11, 0.28), (7, 0.55)):
            cb = batch_for_shader(sh2, 'TRI_FAN', {"pos": _circle(dx, dy, rr)})
            sh2.bind(); sh2.uniform_float("color", (0.2, 0.9, 1.0, aa)); cb.draw(sh2)
        cb = batch_for_shader(sh2, 'TRI_FAN', {"pos": _circle(dx, dy, 5.0)})
        sh2.bind(); sh2.uniform_float("color", (1, 1, 1, 0.98)); cb.draw(sh2)
        gpu.state.blend_set('NONE')
    # clickable hint buttons (icon + label) below the image
    by = y - 30
    _icon_x(x + 8, by + 7, (1.0, 0.5, 0.42, 1.0))
    blf.size(fid, 15); blf.color(fid, 1.0, 0.55, 0.45, 1.0)
    blf.position(fid, x + 22, by, 0); blf.draw(fid, "Cancel")
    cw, _ = blf.dimensions(fid, "Cancel")
    GUIDE["btn_cancel"] = (x - 2, by - 6, x + 22 + cw + 4, by + 18)
    x2 = x + 22 + cw + 28
    _icon_back(x2 + 2, by + 7, (0.8, 0.85, 0.92, 1.0))
    blf.color(fid, 0.8, 0.85, 0.92, 1.0)
    blf.position(fid, x2 + 20, by, 0); blf.draw(fid, "Back")
    bw, _ = blf.dimensions(fid, "Back")
    GUIDE["btn_back"] = (x2 - 4, by - 6, x2 + 20 + bw + 4, by + 18)
    blf.disable(fid, blf.SHADOW)


# clickable Cancel / Back buttons drawn IN the viewport during click-placement
# (panel buttons can't receive clicks while a modal grabs the input)
MODAL_BTNS = False


def _draw_modal_buttons(region):
    if not MODAL_BTNS:
        GUIDE["btn_cancel"] = None
        GUIDE["btn_back"] = None
        return
    fid = 0
    blf.enable(fid, blf.SHADOW); blf.shadow(fid, 5, 0, 0, 0, 1); blf.size(fid, 17)
    cancel_txt, back_txt = "Cancel", "Back"
    cw, _ = blf.dimensions(fid, cancel_txt)
    bw, _ = blf.dimensions(fid, back_txt)
    pad, gap = 26, 34
    by = region.height - 118
    total = (pad + bw) + gap + (pad + cw)
    bx = region.width // 2 - total // 2
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    # Back button
    rb = (bx - 8, by - 7, bx + pad + bw + 8, by + 21)
    gpu.state.blend_set('ALPHA')
    rect = batch_for_shader(sh, 'TRI_FAN', {"pos": [(rb[0], rb[1]), (rb[2], rb[1]), (rb[2], rb[3]), (rb[0], rb[3])]})
    sh.bind(); sh.uniform_float("color", (0.15, 0.17, 0.20, 0.88)); rect.draw(sh)
    gpu.state.blend_set('NONE')
    _icon_back(bx + 9, by + 8, (0.82, 0.87, 0.94, 1.0))
    blf.color(fid, 0.85, 0.9, 0.96, 1.0); blf.position(fid, bx + pad, by, 0); blf.draw(fid, back_txt)
    GUIDE["btn_back"] = rb
    # Cancel button
    cxx = bx + pad + bw + gap
    rc = (cxx - 8, by - 7, cxx + pad + cw + 8, by + 21)
    gpu.state.blend_set('ALPHA')
    rect = batch_for_shader(sh, 'TRI_FAN', {"pos": [(rc[0], rc[1]), (rc[2], rc[1]), (rc[2], rc[3]), (rc[0], rc[3])]})
    sh.bind(); sh.uniform_float("color", (0.30, 0.12, 0.12, 0.90)); rect.draw(sh)
    gpu.state.blend_set('NONE')
    _icon_x(cxx + 9, by + 8, (1.0, 0.5, 0.42, 1.0))
    blf.color(fid, 1.0, 0.6, 0.5, 1.0); blf.position(fid, cxx + pad, by, 0); blf.draw(fid, cancel_txt)
    GUIDE["btn_cancel"] = rc
    blf.disable(fid, blf.SHADOW)


def guide_hit(mx, my):
    """Return 'cancel' / 'back' / None for a click at region coords (mx,my)."""
    for key, name in (("btn_cancel", "cancel"), ("btn_back", "back")):
        r = GUIDE.get(key)
        if r and r[0] <= mx <= r[2] and r[1] <= my <= r[3]:
            return name
    return None


def guide_clear():
    GUIDE["active"] = False
    GUIDE["tex"] = None
    img = bpy.data.images.get(GUIDE.get("img") or "")
    if img:
        try:
            bpy.data.images.remove(img)
        except Exception:
            pass
    GUIDE["img"] = None
    GUIDE["uvs"] = []; GUIDE["labels"] = []; GUIDE["step"] = 0


def _draw_fingers(region, rv3d):
    """Green glowing finger markers + chain lines (same look as the body markers)."""
    from . import fingers_manual
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    pairs = [(part, side) for part in ("hand", "foot", "palm") for side in ("L", "R")]
    for part, side in pairs:
        for fn, chain in fingers_manual.list_fingers(part, side).items():
            if chain and chain[0].hide_get():
                continue                                  # this digit is hidden
            col = fingers_manual.color_for(fn)            # distinct colour per digit
            pts = []
            selpts = []
            for o in chain:
                p = view3d_utils.location_3d_to_region_2d(region, rv3d, o.matrix_world.translation)
                if p:
                    pts.append((p.x, p.y))
                    selpts.append((p.x, p.y, _selected(o)))
            if len(pts) >= 2:
                gpu.state.line_width_set(2.5)
                verts = []
                for i in range(len(pts) - 1):
                    verts += [pts[i], pts[i + 1]]
                b = batch_for_shader(shader, 'LINES', {"pos": verts})
                shader.bind(); shader.uniform_float("color", (col[0], col[1], col[2], 0.9)); b.draw(shader)
                gpu.state.line_width_set(1.0)
            m = _marker_mult()
            for (px, py, sel) in selpts:
                s = 1.4 if sel else 1.0
                for rr, aa in ((20 * m * s, 0.12), (13 * m * s, 0.26), (8 * m * s, 0.60)):
                    b = batch_for_shader(shader, 'TRI_FAN', {"pos": _circle(px, py, rr)})
                    shader.bind(); shader.uniform_float("color", (col[0], col[1], col[2], aa)); b.draw(shader)
                core = SEL_COLOR if sel else (1, 1, 1)
                b = batch_for_shader(shader, 'TRI_FAN', {"pos": _circle(px, py, 3.5 * m * s)})
                shader.bind(); shader.uniform_float("color", (core[0], core[1], core[2], 0.98)); b.draw(shader)
                if sel:
                    gpu.state.line_width_set(2.5)
                    rb = batch_for_shader(shader, 'LINE_STRIP', {"pos": _ring(px, py, 17 * m * s)})
                    shader.bind(); shader.uniform_float("color", (SEL_COLOR[0], SEL_COLOR[1], SEL_COLOR[2], 1.0)); rb.draw(shader)
                    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _draw_fit_labels(region, rv3d):
    """Names next to every Fit Wizard marker (same coloured style)."""
    fid = 0
    blf.enable(fid, blf.SHADOW)
    blf.shadow(fid, 5, 0, 0, 0, 1)
    blf.size(fid, 11)
    for o, role in _fit_marker_items():
        p = view3d_utils.location_3d_to_region_2d(
            region, rv3d, o.matrix_world.translation)
        if not p:
            continue
        col = GLOW[role]
        blf.color(fid, min(col[0] + 0.3, 1), min(col[1] + 0.3, 1),
                  min(col[2] + 0.3, 1), 1)
        blf.position(fid, p.x + 12, p.y + 8, 0)
        key = o.name.split("SRFM_", 1)[-1]
        blf.draw(fid, key.replace("_w_", " width ").replace("_d_", " depth ")
                 .replace("_l", " L").replace("_r", " R")
                 .replace("_f", " front").replace("_b", " back")
                 .replace("_", " ").title())


def _draw_cb():
    try:
        region = bpy.context.region
        rv3d = bpy.context.region_data
        if region is None or rv3d is None:
            return
        from . import fingers_manual
        has_body = any(bpy.data.objects.get(n) for n in markers.all_marker_names())
        has_fing = fingers_manual.has_manual(side="L") or fingers_manual.has_manual(side="R")
        has_fit = bool(_fit_marker_items())
        if not has_body and not has_fing and not has_fit:
            return
        if has_body:
            _draw_lines(region, rv3d)
        if has_body or has_fit:
            _draw_glow(region, rv3d)
        if has_fit:
            _draw_fit_labels(region, rv3d)
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
