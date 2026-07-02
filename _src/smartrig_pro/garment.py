"""Let's Fit — automatic garment fitting (v1.20.0).

Take ANY 3D clothing mesh (skirt, dress, shirt, pants, shalwar, thobe/kandura...)
and fit it onto the character automatically:

  1. ANALYSE   - the garment's opening rings (waist / neck / hem) via the same
                 world-space topology tools as the skirt module (_rim_rings), and
                 the body's per-height cross-section profile (radius + centre).
  2. PLACE     - grid-search the (anchor height, uniform scale) that best matches
                 the garment rings to the body profile; move/scale the OBJECT
                 transform only (fully reversible).
  3. CONFORM   - live modifier stack SRF_Wrap (Shrinkwrap OUTSIDE) +
                 SRF_Smooth (Corrective Smooth) + SRF_Touchup (Shrinkwrap) removes
                 body penetrations while keeping the garment's own detail.
  4. TUNE/APPLY- Ease / Smoothing / Scale / Height sliders are live; 'Apply Fit'
                 bakes the modifiers; 'Remove Fit' restores the original object.

All geometry is analysed in WORLD space (imported clothes are often rotated /
scaled at object level - see LESSONS.md).
"""
import bpy
import math
import numpy as np
from mathutils import Vector, Matrix

from . import utils

# modifier / custom-property names (all removable - full_cleanup discipline)
MOD_SNUG = "SRF_Snug"
MOD_WRAP = "SRF_Wrap"
MOD_SMOOTH = "SRF_Smooth"
MOD_TOUCH = "SRF_Touchup"
VG_SNUG = "SRF_Snug"
K_ORIG = "srf_orig_matrix"      # matrix_world before Let's Fit (flat 16)
K_BASE = "srf_base_matrix"      # matrix_world right after auto-place (flat 16)
K_ANCHOR = "srf_anchor"         # world anchor point (top-ring centre after place)
K_BODY = "srf_body"             # body object name
K_BODYH = "srf_body_h"          # body height (world) for relative sliders
K_INFO = "srf_info"             # human-readable fit summary for the UI
K_KEYS = "srf_added_keys"       # True when Let's Fit created the shape keys
SK_FIT = "SRF_Fit"              # preserve-shape conform shape key


# ----------------------------------------------------------------- analysis --

def _ring_stats(pts):
    """(centre Vector, median horizontal radius) of a ring point list (world)."""
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    c = Vector(((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5,
                (min(zs) + max(zs)) * 0.5))
    rr = sorted(math.hypot(p.x - c.x, p.y - c.y) for p in pts)
    return c, rr[len(rr) // 2]


def analyze_garment(ob, nslice=12):
    """Opening rings + full vertical RADIUS PROFILE of the garment (world).
    The profile (not just two rings) is what disambiguates WHERE the garment
    belongs on the body: a skirt anchored at the neck would be far too tight
    around the shoulders, so profile matching rejects it. Returns None on
    failure."""
    from .skirt import _rim_rings          # world-space, slit-proof (v1.19.146)
    top, bot = _rim_rings(ob)
    if not top:
        return None
    co = utils.read_rest_coords(ob)
    tc, tr = _ring_stats(top)
    bc, br = (None, None)
    if bot:
        bc, br = _ring_stats(bot)
    # layered / open-back garments (wedding dress): the highest WRAPPING ring
    # can be an inner layer far below the real top (open-back necklines never
    # wrap). If so, anchor on the actual TOP Z-BAND of the garment instead.
    zmx = float(co[:, 2].max()); zmn = float(co[:, 2].min())
    top_band = False
    if tc.z < zmx - 0.15 * max(zmx - zmn, 1e-9):
        band = co[co[:, 2] >= zmx - 0.04 * (zmx - zmn)]
        if len(band) >= 8:
            cx = (band[:, 0].min() + band[:, 0].max()) * 0.5
            cy = (band[:, 1].min() + band[:, 1].max()) * 0.5
            rr = np.hypot(band[:, 0] - cx, band[:, 1] - cy)
            tc = Vector((cx, cy, float(np.median(band[:, 2]))))
            tr = float(np.median(rr))
            top_band = True
    # radius profile from the TOP RING down to the garment bottom
    zlo = float(co[:, 2].min())
    span = max(tc.z - zlo, 1e-6)
    prof = []                              # [(dz_below_top_ring, radius), ...]
    for i in range(nslice):
        z_a = tc.z - span * i / nslice
        z_b = tc.z - span * (i + 1) / nslice
        sel = co[(co[:, 2] <= z_a) & (co[:, 2] >= z_b)]
        if len(sel) < 4:
            continue
        cx = (sel[:, 0].min() + sel[:, 0].max()) * 0.5
        cy = (sel[:, 1].min() + sel[:, 1].max()) * 0.5
        r = float(np.median(np.hypot(sel[:, 0] - cx, sel[:, 1] - cy)))
        prof.append((tc.z - (z_a + z_b) * 0.5, r))
    return {
        "top_c": tc, "top_r": tr, "bot_c": bc, "bot_r": br,
        "zmin": zlo, "zmax": float(co[:, 2].max()), "profile": prof,
        "top_band": top_band,       # True = no real collar (strapless/open back)
    }


def body_profile(body, nslice=64):
    """Per-height cross-section profile of the character (world, REST coords):
    returns (z0, z1, R(z) callable, C(z) callable) where R = median torso radius
    and C = horizontal slice centre (bbox midpoint - density-proof)."""
    co = utils.read_rest_coords(body)
    z0, z1 = float(co[:, 2].min()), float(co[:, 2].max())
    h = max(z1 - z0, 1e-6)
    edges = np.linspace(z0, z1, nslice + 1)
    R = np.zeros(nslice); CX = np.zeros(nslice); CY = np.zeros(nslice)
    for i in range(nslice):
        sel = co[(co[:, 2] >= edges[i]) & (co[:, 2] <= edges[i + 1])]
        if len(sel) < 4:
            R[i] = -1.0
            continue
        cx = (sel[:, 0].min() + sel[:, 0].max()) * 0.5
        cy = (sel[:, 1].min() + sel[:, 1].max()) * 0.5
        CX[i], CY[i] = cx, cy
        # DENSITY-PROOF torso radius: median radius per angular wedge, then the
        # median across wedges. A-pose hands/arms hang beside the hips and their
        # dense finger topology dominates any plain percentile - but they only
        # occupy ~2 of 8 wedges, so the wedge median ignores them.
        ang = np.arctan2(sel[:, 1] - cy, sel[:, 0] - cx)
        rad = np.hypot(sel[:, 0] - cx, sel[:, 1] - cy)
        wr = []
        for k in range(8):
            a0 = -math.pi + k * math.pi / 4.0
            m = (ang >= a0) & (ang < a0 + math.pi / 4.0)
            if m.sum() >= 2:
                wr.append(float(np.median(rad[m])))
        R[i] = float(np.median(wr)) if wr else -1.0
    # fill empty slices from neighbours
    for i in range(nslice):
        if R[i] < 0:
            j = next((k for k in range(1, nslice)
                      for s in (1, -1) if 0 <= i + s * k < nslice
                      and R[i + s * k] >= 0), None)
            R[i] = R[i + j] if j is not None and 0 <= i + j < nslice and R[i + j] >= 0 else 0.0

    def _idx(z):
        return int(min(max((z - z0) / h * nslice, 0), nslice - 1))

    return z0, z1, (lambda z: float(R[_idx(z)])), \
        (lambda z: Vector((float(CX[_idx(z)]), float(CY[_idx(z)]), z)))


def _all_rings(ob):
    """ALL wrapping boundary rings of the garment (world): [(centre, radius, n)].
    The opening SIGNATURE is how the addon recognizes the clothing type:
    pants = waist + 2 leg openings, shirt = neck + 2 lateral cuffs + hem,
    skirt/dress = 1-2 stacked rings. Slits/holes are excluded (_ring_wraps)."""
    import bmesh
    from .skirt import _boundary_loops, _ring_wraps
    me = getattr(ob, "data", None)
    if me is None or not me.polygons:
        return []
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.transform(ob.matrix_world)          # WORLD space (LESSONS)
        lv = [v for v in bm.verts if v.link_edges]
        if not lv:
            return []
        xs = [v.co.x for v in lv]; ys = [v.co.y for v in lv]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        out = []
        for L in _boundary_loops(bm):
            pts = [v.co.copy() for v in L]
            # a leg/sleeve opening wraps its OWN centre, not the garment axis
            lcx = (min(p.x for p in pts) + max(p.x for p in pts)) * 0.5
            lcy = (min(p.y for p in pts) + max(p.y for p in pts)) * 0.5
            if _ring_wraps(pts, lcx, lcy):
                c, r = _ring_stats(pts)
                P = np.array([[p.x, p.y, p.z] for p in pts])
                _, _, V = np.linalg.svd(P - P.mean(axis=0))
                out.append((c, r, len(pts), Vector(V[2])))  # V[2]=plane normal
        return out
    except Exception:
        return []
    finally:
        bm.free()


def classify_garment(g, rings):
    """Name the clothing from its opening signature + top-ring ratio.
    Returns (label, is_bottom)."""
    tr = g["top_r"]
    max_r = max([r for (_, r) in g["profile"]] + [tr])
    ratio = tr / max(max_r, 1e-9)
    span = max(g["zmax"] - g["zmin"], 1e-9)
    top_z = g["top_c"].z
    ax, ay = g["top_c"].x, g["top_c"].y
    lower_small = [(c, r) for (c, r, _, _n) in rings
                   if r < 0.55 * max_r and c.z < top_z - 0.15 * span]
    # legs/sleeves are OFFSET from the garment axis; a layered dress/tiered
    # skirt has many CONCENTRIC rings (centre ~on the axis) - never openings
    def _offset(c, r):
        return math.hypot(c.x - ax, c.y - ay) > 0.6 * max(r, 1e-9)
    legs = [x for x in lower_small
            if x[0].z < g["zmin"] + 0.45 * span and _offset(*x)]
    if ratio >= 0.6:
        # WIDE top opening = worn at waist/hips (bottoms). Offset low rings
        # here really are leg openings.
        if len(legs) >= 2:
            return "pants/shalwar", True
        return "skirt", True
    # NARROW top opening (collar-like) = worn from the neck/chest - it can
    # NEVER be pants (pants have a wide waist). Offset rings are SLEEVE CUFFS
    # at ANY height: in A-pose the sleeves hang DOWN beside the torso, exactly
    # where pants legs would be (the Mens_Shirt_4 bug: cuffs at the bottom 45%
    # were read as legs -> 'pants' -> hip anchor -> disaster).
    sleeves = [x for x in lower_small if _offset(*x)]
    if len(sleeves) >= 2:
        return "shirt", False
    return "dress/thobe", False


def _ai_landmarks(body):
    """CHARACTER RECOGNITION: run the addon's own trained pose model
    (models/smartrig_pose.onnx via detect.py) on the body render and return
    the world heights of the fitting landmarks. Falls back to None (geometric
    landmarks) when onnxruntime / the model is unavailable or unsure.
    Other meshes are hidden from the render so clothes don't confuse the net.
    Cached on the body object (key srf_ai) - detection renders the scene."""
    cached = body.get("srf_ai")
    if cached is not None and len(cached) == 5:
        return {"hips": cached[0] or None, "neck": cached[1] or None,
                "chest": cached[2] or None, "shoulders": cached[3] or None,
                "conf": cached[4]}
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

        def gz(*names):
            vs = [pts[n].z for n in names if n in pts and kc.get(n, 0.0) >= 0.3]
            return sum(vs) / len(vs) if vs else None

        lm = {"hips": gz("hip_l", "hip_r", "pelvis"), "neck": gz("neck"),
              "chest": gz("chest"), "shoulders": gz("shoulder_l", "shoulder_r"),
              "conf": float(res["conf"])}
        body["srf_ai"] = [lm["hips"] or 0.0, lm["neck"] or 0.0,
                          lm["chest"] or 0.0, lm["shoulders"] or 0.0,
                          lm["conf"]]
        return lm
    except Exception as e:
        print("SmartRig fit AI landmarks:", e)
        return None


# ------------------------------------------------------------- auto placing --

def auto_place(g_ob, body):
    """Find the (height, uniform scale) that best fits the garment onto the body
    and apply it to the garment's OBJECT transform. Returns an info string."""
    orig_mat = [v for row in g_ob.matrix_world for v in row]
    # ---- AUTO-ORIENTATION: garments are often imported lying down or upside-
    # down. The garment AXIS = the shared plane-normal of its wrapping rings
    # (waist / neck / bodice tube / hem are parallel slices of the tube).
    # 1) rotate so that axis is vertical; 2) the SMALLEST ring is the wearing
    # opening (waist/neck/chest tube) and belongs at the TOP - if it ends up
    # in the lower half, turn the garment over.
    flipped = False

    def _bbox_ctr():
        c = utils.read_rest_coords(g_ob)
        return c, Vector((((c[:, 0].min() + c[:, 0].max()) * 0.5),
                          ((c[:, 1].min() + c[:, 1].max()) * 0.5),
                          ((c[:, 2].min() + c[:, 2].max()) * 0.5)))

    # only SIGNIFICANT rings may define the orientation: a side slit / small
    # hole (Skirt_5's 22-vert slit) must never flip a garment that stands fine
    _co0, _ctr0 = _bbox_ctr()
    _gr = max(float(_co0[:, 0].max() - _co0[:, 0].min()),
              float(_co0[:, 1].max() - _co0[:, 1].min()),
              float(_co0[:, 2].max() - _co0[:, 2].min())) * 0.5
    rings0 = [x for x in _all_rings(g_ob)
              if x[2] >= 40 and x[1] >= 0.15 * max(_gr, 1e-9)]
    if rings0:
        biggest = max(rings0, key=lambda x: x[2])
        axis = Vector(biggest[3])
        acc = Vector((0.0, 0.0, 0.0))
        for (_, _, nv, nn) in rings0:              # sign-aligned weighted mean
            v = Vector(nn)
            if v.dot(axis) < 0.0:
                v = -v
            acc += v * nv
        if acc.length > 1e-6:
            axis = acc.normalized()
        # only a GENUINELY lying garment (axis >45 deg off vertical) gets
        # re-oriented: shirt collars tilt back ~25-30 deg BY DESIGN and their
        # normal was rotating upright shirts (Mens_Shirt_4 bug). The wedding
        # dress that lay flat had axis.z ~= 0 - still caught.
        if abs(axis.z) < 0.7:                      # lying down -> stand it up
            co0, ctr = _bbox_ctr()
            R = axis.rotation_difference(Vector((0.0, 0.0, 1.0))) \
                .to_matrix().to_4x4()
            g_ob.matrix_world = (Matrix.Translation(ctr) @ R
                                 @ Matrix.Translation(-ctr)) @ g_ob.matrix_world
            flipped = True
        rings1 = _all_rings(g_ob) if flipped else rings0
        if rings1:
            small = min(rings1, key=lambda x: x[1])
            co0, ctr = _bbox_ctr()
            if small[0].z < ctr.z:                 # opening below mid -> flip
                F = (Matrix.Translation(ctr)
                     @ Matrix.Rotation(math.pi, 4, 'X')
                     @ Matrix.Translation(-ctr))
                g_ob.matrix_world = F @ g_ob.matrix_world
                flipped = True
    g = analyze_garment(g_ob)
    if g is None:
        return None, "Could not find the garment's opening rings."
    z0, z1, R, C = body_profile(body)
    bh = max(z1 - z0, 1e-6)
    tc, tr = g["top_c"], max(g["top_r"], 1e-9)

    prof = g["profile"]
    if not prof:
        return None, "Could not read the garment's radius profile."

    def _cost(za, s):
        """Match the garment's FULL radius profile against the body profile.
        Asymmetric: tighter than the body is (nearly) impossible without
        stretching -> heavy penalty; looser is allowed (A-line, thobe) -> light
        penalty so snug placements still win over floating ones."""
        c = 0.0
        for (dz, rr) in prof:
            zi = za - s * dz
            ri = s * rr
            if zi < z0 - 0.02 * bh:                # hangs below the feet: free
                continue
            d = (ri - R(min(zi, z1))) / bh         # + = loose, - = too tight
            c += 6.0 * d * d if d < 0.0 else 0.3 * d * d
        gz_lo = za - s * (tc.z - g["zmin"])        # garment bottom after placing
        gz_hi = za + s * (g["zmax"] - tc.z)
        if gz_lo < z0 - 0.03 * bh:                 # sinks under the ground
            c += ((z0 - gz_lo) / bh) ** 2 * 4.0
        if gz_hi > z1 + 0.03 * bh:                 # pokes above the head
            c += ((gz_hi - z1) / bh) ** 2 * 4.0
        return c / max(len(prof), 1)

    # ---- RECOGNIZE the garment (opening signature) and the character ----
    rings = _all_rings(g_ob)
    label, is_bottom = classify_garment(g, rings)
    # character landmarks: the addon's trained pose net first (adapts to the
    # character's true proportions), geometric profile as the fallback
    lm = _ai_landmarks(body)
    zs = np.linspace(z0 + 0.35 * bh, z0 + 0.65 * bh, 40)
    z_hips = float(zs[int(np.argmax([R(z) for z in zs]))])
    zs2 = np.linspace(z0 + 0.75 * bh, z0 + 0.93 * bh, 40)
    z_neck = float(zs2[int(np.argmin([R(z) for z in zs2]))])
    z_chest = z0 + 0.72 * bh
    src = "geometric"
    if lm is not None:
        if lm.get("hips"):
            z_hips = lm["hips"]; src = "AI"
        # NOTE: the pose net's 'neck' is the neck BONE JOINT between the
        # shoulders (rigging semantics, ~71% height, R~0.14) - NOT where a
        # collar rests. A collar sits on the NARROWEST neck cross-section
        # (geometric min-R, ~86%, R~0.06). Using the AI joint scaled the
        # Mens_Shirt collar x1.8 to fit around the chest. Keep z_neck
        # geometric; AI is right for hips/chest/shoulders (joint semantics).
        if lm.get("chest"):
            z_chest = lm["chest"]
    # dresses are FITTED AT THE WAIST: if the garment has a clear waist ring
    # (smallest concentric ring in its middle region - a belt / waist seam),
    # anchor THAT on the body waist instead of hanging the top from the neck
    # (reference: wedding dress = lace bodice -> belt -> tiered skirt)
    waist_ring = None
    if not is_bottom:
        span_g = max(g["zmax"] - g["zmin"], 1e-9)
        conc = [x for x in rings
                if x[2] >= 40
                and math.hypot(x[0].x - tc.x, x[0].y - tc.y)
                <= 0.6 * max(x[1], 1e-9)
                and g["zmin"] + 0.30 * span_g <= x[0].z
                <= g["zmin"] + 0.80 * span_g]
        if conc:
            waist_ring = min(conc, key=lambda x: x[1])
    zs3 = np.linspace(z0 + 0.55 * bh, z0 + 0.72 * bh, 30)
    z_waist = float(zs3[int(np.argmin([R(z) for z in zs3]))])

    if is_bottom:
        kind, za0 = "%s (hips, %s)" % (label, src), z_hips + 0.02 * bh
    elif waist_ring is not None:
        kind, za0 = "%s (waist, %s)" % (label, src), None   # handled below
    elif g.get("top_band"):
        # no real collar (strapless / open-back bodice): the top edge sits at
        # the CHEST line, not the neck
        kind, za0 = "%s (chest, %s)" % (label, src), z_chest
    else:
        kind, za0 = "%s (neck, %s)" % (label, src), z_neck

    # ---- refine (anchor height, scale) locally with the profile cost ----
    best = None                                    # (cost, anchor_z, scale)
    if waist_ring is not None and not is_bottom:
        wc, wr = waist_ring[0], max(waist_ring[1], 1e-9)
        for dza in np.linspace(-0.04, 0.04, 7):
            zw = min(max(z_waist + dza * bh, z0 + 0.1 * bh), z1)
            rb = R(zw)
            if rb <= 1e-9:
                continue
            s0 = rb / wr                           # waist ring hugs the waist
            if not (1e-4 < s0 < 1e4):
                continue
            for m in (1.0, 1.04, 1.08):
                s = s0 * m
                za = zw + s * (tc.z - wc.z)        # implied top-ring height
                cost = _cost(za, s)
                if best is None or cost < best[0]:
                    best = (cost, za, s)
    else:
        for dza in np.linspace(-0.05, 0.05, 9):
            za = min(max(za0 + dza * bh, z0 + 0.1 * bh), z1)
            rb = R(za)
            if rb <= 1e-9:
                continue
            s0 = rb / tr                           # top ring hugs the body there
            if not (1e-4 < s0 < 1e4):
                continue
            for m in (1.0, 1.03, 1.08, 1.15):      # snug -> slightly loose
                s = s0 * m
                cost = _cost(za, s)
                if best is None or cost < best[0]:
                    best = (cost, za, s)
    if best is None:
        return None, "Could not match the garment to the body profile."
    _, za, s = best

    tgt = C(za)                                    # body slice centre at anchor
    D = (Matrix.Translation(Vector((tgt.x, tgt.y, za)))
         @ Matrix.Scale(s, 4)
         @ Matrix.Translation(-tc))
    g_ob[K_ORIG] = orig_mat                        # pre-flip user transform
    g_ob.matrix_world = D @ g_ob.matrix_world
    g_ob[K_BASE] = [v for row in g_ob.matrix_world for v in row]
    g_ob[K_ANCHOR] = list(Vector((tgt.x, tgt.y, za)))
    g_ob[K_BODY] = body.name
    g_ob[K_BODYH] = bh
    frac = (za - z0) / bh * 100.0
    info = "%s, scale x%.3g, at %.0f%% of body height" % (kind, s, frac)
    if flipped:
        info += ", auto-oriented"
    g_ob[K_INFO] = info
    return (za, s), info


def _mat_from_key(ob, key):
    v = ob.get(key)
    if v is None or len(v) != 16:
        return None
    return Matrix([list(v[0:4]), list(v[4:8]), list(v[8:12]), list(v[12:16])])


def apply_nudges(g_ob, props):
    """Live Scale / Height sliders: re-derive matrix_world from the stored base
    placement (never accumulates error)."""
    base = _mat_from_key(g_ob, K_BASE)
    anchor = g_ob.get(K_ANCHOR)
    if base is None or anchor is None:
        return
    a = Vector(anchor)
    s = max(props.garment_scale, 1e-3)
    dz = props.garment_height * g_ob.get(K_BODYH, 1.0)
    N = (Matrix.Translation(a + Vector((0, 0, dz)))
         @ Matrix.Scale(s, 4)
         @ Matrix.Translation(-a))
    g_ob.matrix_world = N @ base


# ------------------------------------------------- preserve-shape conform --

def conform_shape(g_ob, body, props, bands=20, wedges=10):
    """PRESERVE-SHAPE conform (default): instead of per-vertex shrinkwrap (which
    flattens pleats / folds / double walls), build a SMOOTH offset field on a
    (height-band x angular-wedge) grid:

      cell penetrating (innermost vert closer than ease) -> whole cell moves
        radially OUT together by (ease - min_gap): folds keep their shape;
      cell floating in the ANCHOR BAND (waistband/collar) -> whole cell pulls
        IN by (min_gap - ease) * snug weight: the band hugs, thickness intact;
      elsewhere -> untouched (the hem hangs exactly as designed).

    The field is blurred (angle wraps) and sampled bilinearly, then written to
    the SRF_Fit shape key - fully reversible, zero destruction of the design."""
    ease = _ease_abs(g_ob, props)
    me = g_ob.data
    n = len(me.vertices)
    if n == 0:
        return
    mw = g_ob.matrix_world
    base = [v.co.copy() for v in me.vertices]     # design shape (Basis)
    wco = [mw @ c for c in base]
    z0b, z1b, Rf, Cf = body_profile(body)
    zs = [p.z for p in wco]
    zg0, zg1 = min(zs), max(zs)
    span = max(zg1 - zg0, 1e-9)
    binv = body.matrix_world.inverted()
    bmw = body.matrix_world

    rdir = [None] * n                              # horizontal outward direction
    bf = np.zeros(n); wf = np.zeros(n)             # fractional grid coords
    for i, p in enumerate(wco):
        c = Cf(min(max(p.z, z0b), z1b))
        h = Vector((p.x - c.x, p.y - c.y, 0.0))
        rdir[i] = h.normalized() if h.length > 1e-9 else None
        bf[i] = min(max((zg1 - p.z) / span * bands - 0.5, 0.0), bands - 1.0)
        wf[i] = ((math.atan2(p.y - c.y, p.x - c.x) + math.pi)
                 / (2.0 * math.pi)) * wedges

    # anchor-band weight: full snug for the top 25%, fading out at 40%
    wsnug = np.array([max(0.0, min(1.0, (0.40 - (zg1 - p.z) / span) / 0.15))
                      for p in wco])
    # chest line (AI first): used to exempt the shoulder/arm-junction cells
    lm_ch = None
    try:
        lm = _ai_landmarks(body)
        lm_ch = lm.get("chest") if lm else None
    except Exception:
        lm_ch = None
    z_chest = lm_ch if lm_ch else z0b + 0.70 * (z1b - z0b)
    cell_b = np.minimum((bf + 0.5).astype(int), bands - 1)
    cell_w = wf.astype(int) % wedges
    passes = 1 + int(props.garment_smooth) // 8

    # ---- BODY ANGULAR ENVELOPE on the same grid -----------------------------
    # Clothes hang OVER the body's convex silhouette; the true nearest-surface
    # gap pulls fabric INTO the channel between the legs and tears wide gowns
    # apart. Renv[band][wedge] = the body's max horizontal radius (around the
    # slice centre) - the garment must stay at radius >= Renv + floor there.
    bco = utils.read_rest_coords(body)
    cell_r = {}
    for k in range(len(bco)):
        zk = bco[k, 2]
        if zk < zg0 - 0.02 * span or zk > zg1 + 0.02 * span:
            continue
        ck = Cf(min(max(zk, z0b), z1b))
        bi = min(max(int((zg1 - zk) / span * bands), 0), bands - 1)
        wi = int(((math.atan2(bco[k, 1] - ck.y, bco[k, 0] - ck.x) + math.pi)
                  / (2.0 * math.pi)) * wedges) % wedges
        rk = math.hypot(bco[k, 0] - ck.x, bco[k, 1] - ck.y)
        cell_r.setdefault((bi, wi), []).append(rk)
    Renv = np.full((bands, wedges), np.nan)
    Rfull = np.full((bands, wedges), np.nan)       # full cluster incl. the arm
    exempt = np.zeros((bands, wedges), dtype=bool)
    gap_thr = 0.04 * max(z1b - z0b, 1e-6)          # arm-to-torso air gap
    for (bi, wi), rs in cell_r.items():
        # A-pose ARMS must not join the envelope: keep only the radial cluster
        # CONNECTED to the innermost surface; a detached cluster further out
        # (arm/hand beside the torso or hips) starts after an air gap
        rs.sort()
        env = rs[0]
        for rk in rs[1:]:
            if rk - env > gap_thr:
                break
            env = rk
        Renv[bi, wi] = env
        Rfull[bi, wi] = rs[-1]
    # SHOULDER/ARM-JUNCTION exemption, per cell: above the chest line, a cell
    # whose envelope bulges far beyond its band's median is the shoulder-arm
    # mass (no air gap separates it) - necklines/cap sleeves rest ON that slope
    # vertically, so radial conform there only builds a fake shelf. Free it.
    for bi in range(bands):
        zb = zg1 - (bi + 0.5) / bands * span
        if zb <= z_chest:
            continue
        row = Renv[bi]
        med = np.nanmedian(row)
        if np.isnan(med):
            continue
        for wi in range(wedges):
            # LATERAL cells only (arms are always at the sides +-X): the BUST
            # bulges the same way at the FRONT and must keep its push, or the
            # bodice fabric stays behind the breasts (user's reference photo)
            aw = -math.pi + (wi + 0.5) * 2.0 * math.pi / wedges
            lateral = min(abs(aw), abs(aw - math.pi),
                          abs(aw + math.pi)) < math.radians(50)
            if lateral and row[wi] > 1.35 * med:
                Renv[bi, wi] = np.nan              # no cell-field push (shelf)
                exempt[bi, wi] = True              # but per-vert arm clamp on

    rvert = np.zeros(n)                            # vert radius about the axis
    for i, p in enumerate(wco):
        c = Cf(min(max(p.z, z0b), z1b))
        rvert[i] = math.hypot(p.x - c.x, p.y - c.y)

    def _sample(off, bff, wff):                     # bilinear, angle wraps
        b0 = int(bff); b1 = min(b0 + 1, bands - 1); tb = bff - b0
        w0 = int(wff) % wedges; w1 = (w0 + 1) % wedges; tw = wff - int(wff)
        return ((off[b0, w0] * (1 - tw) + off[b0, w1] * tw) * (1 - tb)
                + (off[b1, w0] * (1 - tw) + off[b1, w1] * tw) * tb)

    # ITERATE field -> apply -> re-measure: one pass under-corrects (the blur
    # dilutes band cells, and a cell's min-gap vert understates the rest), so
    # residual gaps like the front/back of an elliptical body stay. 3 rounds
    # converge to gap ~= ease all around while staying perfectly smooth.
    t = np.zeros(n)                                # accumulated radial offset
    wrel = np.ones(n)                              # 1 = designed AGAINST the
    rel = np.zeros(n)                              # designed offset above the
    for _it in range(3):                           # innermost vert of the cell
        d = np.full(n, 1e9)                        # gap to the ENVELOPE (-=in)
        for i in range(n):
            if rdir[i] is None:
                continue
            e = Renv[cell_b[i], cell_w[i]]
            if not np.isnan(e):
                d[i] = (rvert[i] + t[i]) - e
        gmin = np.full((bands, wedges), 1e9)
        wmax = np.zeros((bands, wedges))
        for i in range(n):
            bi, wi = cell_b[i], cell_w[i]
            if d[i] < gmin[bi, wi]:
                gmin[bi, wi] = d[i]
            if wsnug[i] > wmax[bi, wi]:
                wmax[bi, wi] = wsnug[i]
        if _it == 0:
            # DESIGN relative structure of the band (wall thickness, rim slant):
            # each vert's gap above its cell's innermost vert, on the placed
            # design before any conform. Used for the per-vertex band target.
            for i in range(n):
                g0 = gmin[cell_b[i], cell_w[i]]
                if d[i] < 1e8 and g0 < 1e8:
                    rel[i] = max(0.0, d[i] - g0)
            # cap the band's rel at ~the wall thickness (its median): keeps the
            # double wall, but a rim that FLARES outward by design gets pulled
            # down onto the body like a real elastic waistband - the user's
            # circled gap was exactly this designed flare standing off the hip
            sel = rel[wsnug > 0.5]
            if len(sel):
                cap = float(np.percentile(sel, 60))
                for i in range(n):
                    if wsnug[i] > 0.3:
                        # snug only what was DESIGNED against the body (bodice
                        # walls, waistband, its rolled rim). Free decoration at
                        # the same height (ruffles, flare skirts of a gown) has
                        # rel >> cap and must NOT be sucked onto the torso -
                        # that flattened the wedding dress into a disc.
                        if cap > 1e-9:
                            wrel[i] = max(0.0, min(1.0,
                                          (4.0 * cap - rel[i]) / (2.0 * cap)))
                        rel[i] = min(rel[i], cap)
            # the TOP EDGE itself presses onto the body like a real elastic
            # waistband: rel fades to 0 over the top 6% of the garment, so the
            # rim touches and the wall thickness returns gradually below it
            for i in range(n):
                f = ((zg1 - wco[i].z) / span) / 0.06
                if f < 1.0:
                    rel[i] *= max(0.0, f)
        off = np.zeros((bands, wedges))
        for bi in range(bands):
            for wi in range(wedges):
                g = gmin[bi, wi]
                if g > 1e8:
                    continue                        # empty cell -> 0
                if g < ease:
                    off[bi, wi] = ease - g          # push the whole cell OUT
                else:
                    off[bi, wi] = -wmax[bi, wi] * (g - ease)  # band pulls IN
        for _ in range(passes):                     # blur (angle wraps)
            o2 = off.copy()
            for bi in range(bands):
                for wi in range(wedges):
                    s = off[bi, wi] * 2.0; cnt = 2.0
                    if bi > 0:
                        s += off[bi - 1, wi]; cnt += 1
                    if bi < bands - 1:
                        s += off[bi + 1, wi]; cnt += 1
                    s += off[bi, (wi - 1) % wedges] + off[bi, (wi + 1) % wedges]
                    cnt += 2
                    o2[bi, wi] = s / cnt
            off = o2
        for i in range(n):
            if rdir[i] is not None:
                t[i] += _sample(off, bf[i], wf[i])

    # FINAL positions:
    #  1. BAND, per vertex: target = body surface + floor + the vert's DESIGNED
    #     offset above its cell's innermost vert (rel). The innermost wall lands
    #     in contact, walls keep their thickness, the rim keeps its slant - and
    #     the band hugs at ANY user Scale (a half-size skirt = stretched waist),
    #     including the mixed cells (rim outside + hip inside) that the cell
    #     field alone could only push further out.
    #  2. STRICT no-penetration clamp for everything else.
    floor = max(ease, 0.002 * g_ob.get(K_BODYH, 1.0))   # >= 2mm per m of body
    # ---------------- UPPER-BODY SURFACE CONFORM (v1.20.23) ------------------
    # Above the body waistline the garment is FITTED over vertical terrain
    # (bust, shoulder slope, armpits, sleeves) - radial push cannot model that
    # professionally. Per-vertex nearest-SURFACE conform along the surface
    # normal, structure-preserving (rel), then Laplacian-smoothed over the
    # mesh edges. Below the waist the radial envelope stays (legs channel).
    bh_b = max(z1b - z0b, 1e-6)
    zs_w = [z0b + (0.55 + 0.17 * k / 29.0) * bh_b for k in range(30)]
    z_up = min(zs_w, key=lambda z: Rf(z))
    blend0 = z_up - 0.02 * bh_b
    blend1 = z_up + 0.04 * bh_b
    offs = np.zeros(n * 3).reshape(n, 3)
    up_mask = np.zeros(n, dtype=bool)
    # pass 1: SURFACE-basis design measurements, taken from the REFERENCE
    # placement (K_BASE = the auto-fit at Scale 1.0). With user Scale nudges
    # the garment can sink PAST the body's medial axis - the nearest surface
    # then flips to the WRONG side (back fabric pushed out the front = the
    # collapsed-back / floating-shard artifact). The reference placement has
    # unambiguous correspondence; it stays valid for any nudge.
    mwb = _mat_from_key(g_ob, K_BASE)
    wref = wco if mwb is None else [mwb @ c for c in base]
    dS = np.full(n, 1e9)                           # signed design surface dist
    LW = np.zeros((n, 3)); NW = np.zeros((n, 3))
    for i in range(n):
        pr = wref[i]
        if pr.z <= blend0 or rdir[i] is None:
            continue
        pl = binv @ pr
        ok, loc, nrm, _ = body.closest_point_on_mesh(pl)
        if not ok:
            continue
        lw = bmw @ loc
        nw = (bmw.to_3x3() @ nrm).normalized()
        sgn = 1.0 if (pl - loc).dot(nrm) >= 0.0 else -1.0
        dS[i] = sgn * (pr - lw).length
        LW[i] = lw; NW[i] = nw
        up_mask[i] = True
    # per-cell min surface distance -> rel_s = designed height above the
    # innermost fabric layer, IN SURFACE TERMS (preserves layer separation)
    cmin = {}
    for i in range(n):
        if up_mask[i]:
            k = (cell_b[i], cell_w[i])
            if dS[i] < cmin.get(k, 1e9):
                cmin[k] = dS[i]
    rel_s = np.zeros(n)
    for i in range(n):
        if up_mask[i]:
            rel_s[i] = max(0.0, dS[i] - cmin[(cell_b[i], cell_w[i])])
    sel_s = rel_s[up_mask]
    cap_s = float(np.percentile(sel_s, 60)) if len(sel_s) else 0.0
    # THE ASSET IS ALREADY PERFECTLY DRAPED (store render on its mannequin) -
    # the user wants it ON the character looking exactly as imported. So the
    # upper zone does NOT pull/hug at all: only PUSH where the character is
    # bigger than the implied mannequin, spread as a smooth low-frequency
    # INFLATION so the pristine lace never wrinkles.
    for i in range(n):
        if not up_mask[i]:
            continue
        p = wco[i]
        lw = Vector(LW[i]); nw = Vector(NW[i])
        d_cur = (p - lw).dot(nw)                   # CURRENT depth along the
        if d_cur >= floor:                         # stored reference normal
            continue
        tgt = lw + nw * (floor + min(rel_s[i], 0.10 * span))
        ub = min(1.0, (wref[i].z - blend0) / max(blend1 - blend0, 1e-9))
        offs[i] = np.array(tgt - p) * ub
    if up_mask.any():
        ev = np.empty(2 * len(me.edges), dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        em = up_mask[ev[:, 0]] & up_mask[ev[:, 1]]
        E = ev[em]

        def _lap(np_passes):
            nonlocal offs
            for _ in range(np_passes):
                acc = np.zeros((n, 3)); cnt = np.zeros(n)
                np.add.at(acc, E[:, 0], offs[E[:, 1]])
                np.add.at(cnt, E[:, 0], 1.0)
                np.add.at(acc, E[:, 1], offs[E[:, 0]])
                np.add.at(cnt, E[:, 1], 1.0)
                nz = cnt > 0
                offs[nz] = 0.5 * offs[nz] + 0.5 * (acc[nz] / cnt[nz, None])

        _lap(10 + 2 * passes)                      # low-frequency inflation
        # residual pass: smoothing dilutes the push where it was needed most -
        # re-add what is still inside (measured along the STORED reference
        # normal - no re-query, no wrong-side flips), then feather it again
        for i in range(n):
            if not up_mask[i]:
                continue
            lw = Vector(LW[i]); nw = Vector(NW[i])
            p = wco[i] + Vector(offs[i])
            need = (floor + min(rel_s[i], 0.10 * span)) - (p - lw).dot(nw)
            if need > 0.0:
                offs[i] = offs[i] + np.array(nw * need)
        _lap(3)

    # combined constraint grid: normal cells = torso envelope, exempt shoulder
    # cells = arm-inclusive envelope. Sampled BILINEARLY (NaN-aware) - reading
    # it as per-cell constants carved the bodice into 36-deg staircase blocks.
    Cgrid = np.where(~np.isnan(Renv), Renv,
                     np.where(exempt, Rfull, np.nan))

    def _samp_nan(bff, wff):
        b0 = int(bff); b1 = min(b0 + 1, bands - 1); tb = bff - b0
        w0 = int(wff) % wedges; w1 = (w0 + 1) % wedges; tw = wff - int(wff)
        acc = 0.0; wsum = 0.0
        for (bb, ww, wt) in ((b0, w0, (1 - tb) * (1 - tw)),
                             (b0, w1, (1 - tb) * tw),
                             (b1, w0, tb * (1 - tw)),
                             (b1, w1, tb * tw)):
            v = Cgrid[bb, ww]
            if not np.isnan(v):
                acc += v * wt; wsum += wt
        return (acc / wsum) if wsum > 1e-9 else float('nan')

    final = []
    for i in range(n):
        if rdir[i] is None:
            final.append(wco[i]); continue
        p = wco[i]
        if up_mask[i] and wref[i].z >= blend1:
            # pure upper region: surface conform + final anti-penetration
            # (stored reference correspondence - stable under Scale nudges)
            pn = p + Vector(offs[i])
            lw = Vector(LW[i]); nw = Vector(NW[i])
            if (pn - lw).dot(nw) < floor * 0.7:
                pn = lw + nw * (floor + min(rel_s[i], 0.10 * span))
            final.append(pn)
            continue
        e = _samp_nan(bf[i], wf[i])
        r_now = rvert[i] + t[i]
        if not math.isnan(e):
            if wsnug[i] > 0.0:                     # band: designed offset above
                r_tgt = e + floor + rel[i]         # the envelope, per vertex
                r_now = r_now + (r_tgt - r_now) * min(1.0, float(wsnug[i])) \
                    * float(wrel[i])
            if r_now < e + floor:
                # strict no-penetration that PRESERVES the design structure:
                # each vert lands at its own designed offset above the
                # envelope - collapsing everything to one shell crushed the
                # layered lace bodice into blocks
                r_now = e + floor + min(rel[i], 0.06 * span)
        p_low = p + rdir[i] * (r_now - rvert[i])
        if up_mask[i] and wref[i].z > blend0:      # transition band: blend
            ub = (wref[i].z - blend0) / max(blend1 - blend0, 1e-9)
            final.append(p_low.lerp(p + Vector(offs[i]), min(1.0, ub)))
        else:
            final.append(p_low)

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
        sk.data[i].co = inv @ final[i]


def remove_fit_shape(g_ob):
    me = g_ob.data
    if me.shape_keys is None:
        return
    kb = me.shape_keys.key_blocks
    sk = kb.get(SK_FIT)
    if sk is not None:
        g_ob.shape_key_remove(sk)
    # if Let's Fit created the whole key set and only Basis is left, clear it
    if g_ob.get(K_KEYS) and me.shape_keys is not None and len(kb) == 1:
        g_ob.shape_key_clear()
    if K_KEYS in g_ob and me.shape_keys is None:
        del g_ob[K_KEYS]


# ---------------------------------------------------------------- modifiers --

def _ease_abs(g_ob, props):
    return props.garment_ease * 0.01 * g_ob.get(K_BODYH, 1.0)


def _snug_group(g_ob, falloff=0.30):
    """Vertex group that makes the garment's ANCHOR BAND (waistband / collar)
    HUG the body: weight 1 at the top ring, fading to 0 at `falloff` of the
    garment height. Without this, a garment whose opening is wider than the
    body simply FLOATS - Shrinkwrap OUTSIDE never pulls anything inward
    (measured 6.8 cm waistband gap on the test skirt)."""
    mw = g_ob.matrix_world
    zs = [(mw @ v.co).z for v in g_ob.data.vertices]
    z_top, z_lo = max(zs), min(zs)
    span = max(z_top - z_lo, 1e-9)
    vg = g_ob.vertex_groups.get(VG_SNUG)
    if vg is None:
        vg = g_ob.vertex_groups.new(name=VG_SNUG)
    idx = [v.index for v in g_ob.data.vertices]
    for i, z in zip(idx, zs):
        t = (z_top - z) / span
        w = max(0.0, 1.0 - t / falloff)
        vg.add([i], w, 'REPLACE')
    return vg


def add_fit_mods(g_ob, body, props):
    remove_fit_mods(g_ob)
    ease = _ease_abs(g_ob, props)
    _snug_group(g_ob)
    sn = g_ob.modifiers.new(MOD_SNUG, 'SHRINKWRAP')
    sn.target = body
    sn.wrap_mode = 'ON_SURFACE'            # pulls BOTH ways -> the band hugs
    sn.wrap_method = 'NEAREST_SURFACEPOINT'
    sn.offset = ease
    sn.vertex_group = VG_SNUG
    w = g_ob.modifiers.new(MOD_WRAP, 'SHRINKWRAP')
    w.target = body
    w.wrap_mode = 'OUTSIDE'
    w.wrap_method = 'NEAREST_SURFACEPOINT'
    w.offset = ease
    cs = g_ob.modifiers.new(MOD_SMOOTH, 'CORRECTIVE_SMOOTH')
    cs.factor = 0.5
    cs.iterations = int(props.garment_smooth)
    cs.smooth_type = 'LENGTH_WEIGHTED'
    t = g_ob.modifiers.new(MOD_TOUCH, 'SHRINKWRAP')
    t.target = body
    t.wrap_mode = 'OUTSIDE'
    t.wrap_method = 'NEAREST_SURFACEPOINT'
    t.offset = ease * 0.9                          # re-fix what smoothing pushed in


def remove_fit_mods(g_ob):
    for n in (MOD_SNUG, MOD_WRAP, MOD_SMOOTH, MOD_TOUCH):
        m = g_ob.modifiers.get(n)
        if m:
            g_ob.modifiers.remove(m)
    for vgn in (VG_SNUG, "SRF_Pin"):
        vg = g_ob.vertex_groups.get(vgn)
        if vg:
            g_ob.vertex_groups.remove(vg)


_TUNE_TOKEN = [0]                                  # debounce for heavy conform


def live_fit_tune(context):
    """update= callback for all fit sliders (works on a paused frame - LESSONS #3).
    The nudge (scale/height matrix) is applied INSTANTLY; the heavy
    conform_shape (seconds on a 150k-vert gown) is DEBOUNCED via a timer so it
    runs once ~0.4s after the user releases the slider - dragging stays fluid."""
    props = context.scene.smartrig
    g_ob = props.garment_object
    if g_ob is None or g_ob.get(K_BASE) is None:
        return
    apply_nudges(g_ob, props)
    ease = _ease_abs(g_ob, props)
    ks = g_ob.data.shape_keys
    if ks is not None and ks.key_blocks.get(SK_FIT) is not None:
        _TUNE_TOKEN[0] += 1
        tok = _TUNE_TOKEN[0]
        gname = g_ob.name

        def _deferred():
            if tok != _TUNE_TOKEN[0]:              # superseded by a newer tick
                return None
            g = bpy.data.objects.get(gname)
            if g is None:
                return None
            b = bpy.data.objects.get(g.get(K_BODY, ""))
            if b is not None:
                try:
                    conform_shape(g, b, bpy.context.scene.smartrig)
                except Exception as e:
                    print("SmartRig fit deferred conform:", e)
            return None

        bpy.app.timers.register(_deferred, first_interval=0.4)
        return
    m = g_ob.modifiers.get(MOD_SNUG)
    if m:
        m.offset = ease
    m = g_ob.modifiers.get(MOD_WRAP)
    if m:
        m.offset = ease
    m = g_ob.modifiers.get(MOD_TOUCH)
    if m:
        m.offset = ease * 0.9
    m = g_ob.modifiers.get(MOD_SMOOTH)
    if m:
        m.iterations = int(props.garment_smooth)


def _pick_body(props, g_ob):
    b = props.fit_body_object or props.target_mesh
    if b is not None and b is not g_ob:
        return b
    # fall back: the biggest other visible mesh in the scene
    best, vol = None, -1.0
    for o in bpy.context.scene.objects:
        if o.type != 'MESH' or o is g_ob or not o.visible_get():
            continue
        d = o.dimensions
        v = d.x * d.y * d.z
        if v > vol:
            best, vol = o, v
    return best


# ---------------------------------------------------------------- operators --

class SMARTRIG_OT_lets_fit(bpy.types.Operator):
    """Automatically fit the garment onto the character: analyse its openings,
    scale + place it on the matching body region, then conform it live"""
    bl_idname = "smartrig.lets_fit"
    bl_label = "Fit Garment"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        g_ob = props.garment_object
        if g_ob is None or g_ob.type != 'MESH':
            self.report({'ERROR'}, "Pick the garment mesh with the eyedropper first.")
            return {'CANCELLED'}
        body = _pick_body(props, g_ob)
        if body is None:
            self.report({'ERROR'}, "No character mesh found - pick the body too.")
            return {'CANCELLED'}
        props.fit_body_object = body
        # refit: restore the original transform first so Fit is repeatable
        orig = _mat_from_key(g_ob, K_ORIG)
        if orig is not None:
            g_ob.matrix_world = orig
        remove_fit_mods(g_ob)
        remove_fit_shape(g_ob)
        placed, info = auto_place(g_ob, body)
        if placed is None:
            self.report({'ERROR'}, info)
            return {'CANCELLED'}
        # reset nudges so sliders start neutral
        props.garment_scale = 1.0
        props.garment_height = 0.0
        if props.garment_preserve:
            conform_shape(g_ob, body, props)       # keeps pleats/volume intact
            info += ", shape preserved"
            g_ob[K_INFO] = info
        else:
            add_fit_mods(g_ob, body, props)
        self.report({'INFO'}, "Fitted: " + info)
        return {'FINISHED'}


class SMARTRIG_OT_fit_drape(bpy.types.Operator):
    """Professional finish: run a short pinned CLOTH SIMULATION so the fitted
    garment drapes naturally over the character (works with any body and any
    clothing - physics, not training). The anchor band stays pinned; the
    result is written back into the SRF_Fit shape key (fully reversible)"""
    bl_idname = "smartrig.fit_drape"
    bl_label = "Drape (Cloth)"
    bl_options = {'REGISTER', 'UNDO'}

    frames: bpy.props.IntProperty(
        name="Settle Frames", default=25, min=5, max=120,
        description="How many frames the cloth settles for")

    def execute(self, context):
        props = context.scene.smartrig
        g_ob = props.garment_object
        if g_ob is None or g_ob.get(K_BASE) is None:
            self.report({'ERROR'}, "Fit the garment first, then drape.")
            return {'CANCELLED'}
        body = bpy.data.objects.get(g_ob.get(K_BODY, "")) or props.fit_body_object
        if body is None:
            self.report({'ERROR'}, "Body not found - refit first.")
            return {'CANCELLED'}
        scene = context.scene
        me = g_ob.data
        n = len(me.vertices)
        bh = g_ob.get(K_BODYH, 1.0)

        # pin group: the anchor band (same falloff as the snug zone) so the
        # garment hangs from its waistband/collar instead of falling off
        mw = g_ob.matrix_world
        zs = [(mw @ v.co).z for v in me.vertices]
        z_top, z_lo = max(zs), min(zs)
        span = max(z_top - z_lo, 1e-9)
        vg = g_ob.vertex_groups.get("SRF_Pin")
        if vg is None:
            vg = g_ob.vertex_groups.new(name="SRF_Pin")
        for i, z in enumerate(zs):
            t = (z_top - z) / span
            vg.add([i], max(0.0, 1.0 - t / 0.20), 'REPLACE')
        # SMALL LOOSE PARTS (buttons, detached cuffs, brooches...) are rigid
        # accessories, not cloth - unpinned they just FALL through the sim
        # (Mens_Shirt cuffs landed at the knees). Pin them fully.
        import bmesh
        bm = bmesh.new(); bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        seen = set()
        for v0 in bm.verts:
            if v0.index in seen:
                continue
            stack = [v0]; comp = []
            seen.add(v0.index)
            while stack:
                cur = stack.pop(); comp.append(cur.index)
                for e in cur.link_edges:
                    o = e.other_vert(cur)
                    if o.index not in seen:
                        seen.add(o.index); stack.append(o)
            if len(comp) < 0.05 * n:               # accessory -> fully pinned
                vg.add(comp, 1.0, 'REPLACE')
        bm.free()

        # temporary physics setup
        col_added = False
        if not any(m.type == 'COLLISION' for m in body.modifiers):
            body.modifiers.new("SRF_Col", 'COLLISION')
            col_added = True
        body.collision.thickness_outer = 0.004 * bh
        cl = g_ob.modifiers.new("SRF_Cloth", 'CLOTH')
        cs = cl.settings
        cs.vertex_group_mass = "SRF_Pin"
        cs.quality = 6
        cs.mass = 0.2
        cs.tension_stiffness = 30.0                # fabric must not stretch long
        cs.compression_stiffness = 30.0
        cs.shear_stiffness = 8.0
        cs.bending_stiffness = 0.4
        cl.collision_settings.collision_quality = 3
        cl.collision_settings.distance_min = 0.003 * bh
        cl.collision_settings.use_self_collision = False
        cl.point_cache.frame_start = scene.frame_current
        cl.point_cache.frame_end = scene.frame_current + int(self.frames)

        f0 = scene.frame_current
        try:
            for f in range(f0, f0 + int(self.frames) + 1):
                scene.frame_set(f)
            # capture the settled cloth
            dg = context.evaluated_depsgraph_get()
            ev = g_ob.evaluated_get(dg).to_mesh()
            settled = [ev.vertices[i].co.copy() for i in range(n)]
            g_ob.evaluated_get(dg).to_mesh_clear()
        finally:
            g_ob.modifiers.remove(cl)
            if col_added:
                m = body.modifiers.get("SRF_Col")
                if m:
                    body.modifiers.remove(m)
            scene.frame_set(f0)

        # write into the SRF_Fit shape key (create the key set if needed)
        if me.shape_keys is None:
            g_ob.shape_key_add(name="Basis", from_mix=False)
            g_ob[K_KEYS] = True
        sk = me.shape_keys.key_blocks.get(SK_FIT)
        if sk is None:
            sk = g_ob.shape_key_add(name=SK_FIT, from_mix=False)
        sk.slider_min = 0.0
        sk.value = 1.0
        for i in range(n):
            sk.data[i].co = settled[i]             # evaluated = local coords
        self.report({'INFO'}, "Draped: cloth settled over %d frames." % self.frames)
        return {'FINISHED'}


class SMARTRIG_OT_fit_apply(bpy.types.Operator):
    """Bake the fit: apply the SRF_* conform modifiers to the garment mesh"""
    bl_idname = "smartrig.fit_apply"
    bl_label = "Apply Fit"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        g_ob = props.garment_object
        if g_ob is None:
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = g_ob
        # preserve-shape path: bake the SRF_Fit shape key into the mesh
        ks = g_ob.data.shape_keys
        if ks is not None and ks.key_blocks.get(SK_FIT) is not None:
            kb = ks.key_blocks[SK_FIT]
            coords = [kb.data[i].co.copy() for i in range(len(g_ob.data.vertices))]
            if g_ob.get(K_KEYS):                   # keys were ours -> clear all
                g_ob.shape_key_clear()
                for i, v in enumerate(g_ob.data.vertices):
                    v.co = coords[i]
                if K_KEYS in g_ob:
                    del g_ob[K_KEYS]
            else:                                  # user had own keys -> keep them
                self.report({'WARNING'},
                            "Mesh has its own shape keys - SRF_Fit left in place.")
        for n in (MOD_SNUG, MOD_WRAP, MOD_SMOOTH, MOD_TOUCH):
            if g_ob.modifiers.get(n):
                try:
                    with context.temp_override(object=g_ob, active_object=g_ob):
                        bpy.ops.object.modifier_apply(modifier=n)
                except Exception as e:
                    self.report({'WARNING'}, "Could not apply %s: %s" % (n, e))
        vg = g_ob.vertex_groups.get(VG_SNUG)
        if vg:
            g_ob.vertex_groups.remove(vg)
        for k in (K_BASE, K_ANCHOR, K_INFO):
            if k in g_ob:
                del g_ob[k]
        self.report({'INFO'}, "Fit applied - the garment mesh is final.")
        return {'FINISHED'}


class SMARTRIG_OT_fit_remove(bpy.types.Operator):
    """Undo Let's Fit: remove the conform modifiers and restore the garment's
    original position / scale"""
    bl_idname = "smartrig.fit_remove"
    bl_label = "Remove Fit"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        g_ob = props.garment_object
        if g_ob is None:
            return {'CANCELLED'}
        remove_fit_mods(g_ob)
        remove_fit_shape(g_ob)
        orig = _mat_from_key(g_ob, K_ORIG)
        if orig is not None:
            g_ob.matrix_world = orig
        for k in (K_ORIG, K_BASE, K_ANCHOR, K_BODY, K_BODYH, K_INFO):
            if k in g_ob:
                del g_ob[k]
        self.report({'INFO'}, "Fit removed - garment restored.")
        return {'FINISHED'}


_classes = (SMARTRIG_OT_lets_fit, SMARTRIG_OT_fit_drape,
            SMARTRIG_OT_fit_apply, SMARTRIG_OT_fit_remove)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
