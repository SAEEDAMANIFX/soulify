"""Soulify - FULL Storm face rig replication (v2.0).

Data-driven: reads storm_face_spec.json + storm_face_helpers.json (extracted
from the CC-BY Blender-Studio *Storm* rig) and rebuilds the COMPLETE face
system on the user's character:

  * all 538 face bones (masters, CTL/P/MCH/DEF/DSP/PRP/HK layers, ribbons'
    handles, teeth, tongue, jawline, sockets, micro lips, blink props...)
  * every constraint (incl. mesh-vertex-group targets = the ribbon trick)
  * the 94 face drivers (auto-blink, eyelid-follow, zipper...)
  * the helper RIBBON meshes (eyelids / lips / blink) with vgroups,
    shape keys and their drivers
  * the face LATTICES (cheeks, cheek-puff, eyes, face-mask, teeth) with
    hook modifiers, plus the lattice+corrective-smooth stack on the body
  * Storm bone collections, widgets, palettes, locks, custom props

Positions are retargeted with an RBF (linear kernel + affine term) anchored
on OUR landmarks: the registered eyelid loops, the registered mouth loop,
brows / cheeks / nose grid points, eye centers and the registered teeth /
tongue meshes.  Weights for the surface DEF bones are generated
procedurally (chain-partition along ribbons, radial blobs elsewhere) using
the region statistics measured from Storm's hand-painted weights.
"""

import json
import os
import math

import bpy
import numpy as np
from mathutils import Vector, Matrix

from . import utils
from . import face_widgets as _fw

# body-rig bones a Storm parent/subtarget may map to
_HEAD_ALIASES = ("STR-Head", "DEF-Head", "ROOT-Head", "Head", "FK-Head",
                 "STR-Head1", "STR-P-Head", "STR-P-Head1", "MCH-Neck",
                 "DEF-Neck", "Neck", "STR-Neck")

_SPEC_CACHE = {}


def _load(name):
    if name not in _SPEC_CACHE:
        p = os.path.join(os.path.dirname(__file__), name)
        with open(p, "r") as f:
            _SPEC_CACHE[name] = json.load(f)
    return _SPEC_CACHE[name]


def spec_available():
    try:
        _load("storm_face_spec.json")
        _load("storm_face_helpers.json")
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ RBF
class _RBF:
    """Linear-kernel RBF with affine polynomial term: exact at anchors,
    globally affine far away - no oscillation."""

    def __init__(self, src, dst, reg=1e-6):
        src = np.asarray(src, float)
        dst = np.asarray(dst, float)
        n = len(src)
        d = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
        A = np.zeros((n + 4, n + 4))
        A[:n, :n] = d + reg * np.eye(n)
        P = np.hstack([np.ones((n, 1)), src])
        A[:n, n:] = P
        A[n:, :n] = P.T
        B = np.zeros((n + 4, 3))
        B[:n] = dst
        self.src = src
        self.W = np.linalg.solve(A, B)

    def __call__(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        d = np.linalg.norm(pts[:, None, :] - self.src[None, :, :], axis=2)
        P = np.hstack([np.ones((len(pts), 1)), pts])
        out = d @ self.W[:-4] + P @ self.W[-4:]
        return out

    def p(self, pt):
        return self(np.asarray(pt, float))[0]

    def dir(self, at, v, eps=1e-3):
        a = np.asarray(at, float)
        v = np.asarray(v, float)
        m = self(np.vstack([a, a + eps * v]))
        d = m[1] - m[0]
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else v

    def jac_scale(self, at, eps=1e-3):
        a = np.asarray(at, float)
        pts = [a, a + (eps, 0, 0), a + (0, eps, 0), a + (0, 0, eps)]
        m = self(np.vstack(pts))
        s = [np.linalg.norm(m[i + 1] - m[0]) / eps for i in range(3)]
        return float(np.mean(s))

    def jac(self, at, eps=1e-3):
        a = np.asarray(at, float)
        pts = [a, a + (eps, 0, 0), a + (0, eps, 0), a + (0, 0, eps)]
        m = self(np.vstack(pts))
        J = np.stack([(m[i + 1] - m[0]) / eps for i in range(3)], axis=1)
        return J


# ------------------------------------------------------------- anchors
def _ring_resample(our_ring, our_center, storm_pts, storm_center):
    """For each storm ring point, sample OUR ring polyline at the same
    angle (front XZ projection). Returns list of our-side points."""
    oc = np.asarray(our_center, float)
    sc = np.asarray(storm_center, float)
    ring = [np.asarray(p, float) for p in our_ring]

    def ang(p, c):
        return math.atan2(p[2] - c[2], p[0] - c[0])

    n = len(ring)
    angs = [ang(p, oc) for p in ring]
    out = []
    for sp in storm_pts:
        a = ang(np.asarray(sp, float), sc)
        best, bi = 1e9, 0
        for i in range(n):
            j = (i + 1) % n
            a0, a1 = angs[i], angs[j]
            d0 = (a - a0 + math.pi) % (2 * math.pi) - math.pi
            d1 = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            if abs(d1) < 1e-9:
                continue
            t = d0 / d1
            if 0.0 <= t <= 1.0:
                cand = ring[i] + (ring[j] - ring[i]) * t
                err = 0.0
            else:
                t = min(1.0, max(0.0, t))
                cand = ring[i] + (ring[j] - ring[i]) * t
                err = min(abs(d0), abs(d0 - d1))
            if err < best:
                best, bi = err, cand
        out.append(bi)
    return out


def _vg_verts(host, vg_name, thr=0.15):
    """Vertices belonging to a registration vertex group.

    Registration groups (SR_brows, SR_eye_l, SR_teeth_up, ...) are BINARY
    membership - a vertex is either inside the region or not; the weight is
    meaningless. Blender's Auto-Normalize during binding CRUSHES their
    weights toward 0 (measured: SR_brows -> 0.0006, SR_lashes -> 0.0), so a
    weight>0.15 filter finds NOTHING and the brows/eyes/lashes silently fall
    back to synthetic placement. Read them by MEMBERSHIP instead, which
    survives normalization. Real weighted groups can still pass an explicit
    thr, but SR_ registration groups always use membership."""
    g = host.vertex_groups.get(vg_name) if host else None
    if g is None:
        return None
    gi = g.index
    binary = vg_name.startswith("SR_")
    out = []
    for v in host.data.vertices:
        for gr in v.groups:
            if gr.group == gi and (binary or gr.weight > thr):
                out.append(v.index)
                break
    return out


def _obj_or_vg_pts(props, body, obj_attr, vg_name):
    """World-space point cloud of a registered face part. Priority:
    SR_* vgroup ON the slot object (combined teeth_and_tongue case) ->
    the whole slot object -> SR_* vgroup on the body.
    Returns (points, host_object_or_None, vert_indices_or_None)."""
    ob = getattr(props, obj_attr, None)
    if ob is not None:
        idx = _vg_verts(ob, vg_name)
        try:
            c = utils.read_rest_coords(ob)
        except Exception:
            c = None
        if c is not None and len(c):
            if idx:
                return c[idx], ob, idx
            return c, ob, None
    idx = _vg_verts(body, vg_name) if body else None
    if idx:
        c = utils.read_rest_coords(body)
        return c[idx], body, idx
    # in-mesh registration on ANY other mesh (e.g. the user registered the
    # teeth vertices INSIDE a combined teeth_and_tongue object whose slot
    # is a different part - Saeed's Edit-Mode register flow)
    for other in bpy.data.objects:
        if other is ob or other is body or other.type != 'MESH':
            continue
        if other.name.startswith(("WGT", "SR_", "HLP-", "GEO-")):
            continue
        idx2 = _vg_verts(other, vg_name)
        if idx2:
            try:
                c = utils.read_rest_coords(other)
            except Exception:
                continue
            return c[idx2], other, idx2
    return None, None, None


def _real_face_anchors(face, props, body, gp, ours, ipd):
    """Saeed: استفد من الماركرات المسجلة. The flat net can be badly fitted
    outside the REGISTERED rings, so re-derive the outer anchors from real
    data: SR_brows geometry for the brows, the mesh itself for the nose,
    cheeks and ears. Registered eye rings / mouth loop stay as they are."""
    co = utils.read_rest_coords(body)
    eyeL = np.asarray(ours["eye.L"], float)
    eyeR = np.asarray(ours["eye.R"], float)
    eyeC = (eyeL + eyeR) / 2.0
    mouth = np.asarray(ours["mouth"], float)
    # head slab
    head = co[(co[:, 2] > mouth[2] - 2.2 * ipd) &
              (co[:, 2] < eyeC[2] + 3.5 * ipd)]
    if not len(head):
        return ours
    y_front = float(head[:, 1].min())

    # ---- brows from the registered brow geometry ----
    bp, _, _ = _obj_or_vg_pts(props, body, "skin_brows", "SR_brows")
    if bp is not None and len(bp) > 20:
        for sgn, s in ((1.0, ".L"), (-1.0, ".R")):
            sel = bp[bp[:, 0] * sgn > 0.02 * ipd]
            if len(sel) < 8:
                continue
            xs = np.abs(sel[:, 0])
            lo, hi = float(xs.min()), float(xs.max())
            for part, t in (("in", 0.15), ("mid", 0.5), ("out", 0.85)):
                xt = lo + (hi - lo) * t
                band = sel[np.abs(xs - xt) < max(0.12 * (hi - lo), 1e-4)]
                if len(band):
                    ours["brow_%s%s" % (part, s)] = band.mean(axis=0)
            ours["brow_all" + s] = ours["brow_mid" + s]

    # ---- nose tip / base from the mesh front line ----
    slab = head[(np.abs(head[:, 0]) < 0.25 * ipd)]
    zn_lo = mouth[2] + 0.25 * (eyeC[2] - mouth[2])
    zn_hi = mouth[2] + 0.85 * (eyeC[2] - mouth[2])
    nz = slab[(slab[:, 2] > zn_lo) & (slab[:, 2] < zn_hi)]
    if len(nz):
        tip = nz[int(nz[:, 1].argmin())]
        ours["nose_tip"] = tip
        base = nz[(nz[:, 2] > mouth[2] + 0.1 * (eyeC[2] - mouth[2])) &
                  (nz[:, 2] < tip[2])]
        if len(base):
            ours["nose_base"] = np.array([0.0,
                                          float(base[:, 1].min()),
                                          float(mouth[2] + 0.45 *
                                                (tip[2] - mouth[2]))])

    # ---- ears / face width from the actual head extremes ----
    ez = head[(head[:, 2] > eyeC[2] - 0.8 * ipd) &
              (head[:, 2] < eyeC[2] + 0.8 * ipd)]
    if len(ez):
        for s, sgn in ((".L", 1.0), (".R", -1.0)):
            side = ez[ez[:, 0] * sgn > 0]
            if len(side):
                ext = side[int((side[:, 0] * sgn).argmax())]
                ours["ear" + s] = ext

    # ---- cheeks: on the SKIN between eye and mouth corner ----
    for s, sgn in ((".L", 1.0), (".R", -1.0)):
        corn = np.asarray(ours["corner" + s], float)
        zc = 0.5 * (eyeC[2] + mouth[2])
        xc = corn[0] * 1.35
        band = head[(np.abs(head[:, 2] - zc) < 0.35 * ipd) &
                    (np.abs(head[:, 0] - xc) < 0.35 * ipd)]
        if len(band):
            front = band[int(band[:, 1].argmin())]
            ours["cheek_all" + s] = front
            inb = head[(np.abs(head[:, 2] - zc) < 0.3 * ipd) &
                       (np.abs(head[:, 0] - 0.55 * xc) < 0.25 * ipd)]
            outb = head[(np.abs(head[:, 2] - zc) < 0.3 * ipd) &
                        (np.abs(head[:, 0] - 1.45 * xc) < 0.3 * ipd)]
            if len(inb):
                ours["cheek_in" + s] = inb[int(inb[:, 1].argmin())]
            ours["cheek_mid" + s] = front
            if len(outb):
                ours["cheek_out" + s] = outb[int(outb[:, 1].argmin())]

    # ---- masters follow the corrected features ----
    browC_z = 0.5 * (ours["brow_mid.L"][2] + ours["brow_mid.R"][2])
    ours["face_upp"] = np.array([0.0, float(ours["face_upp"][1]), browC_z])
    return ours


def _build_anchor_pairs(face, props, body, gp, rig):
    """(storm_pts, our_pts) matched anchor arrays, our side in rig space."""
    A = _load("storm_face_spec.json")["anchors"]
    inv = np.array(rig.matrix_world.inverted())

    def to_rig(p):
        p = np.asarray(p, float)
        return inv[:3, :3] @ p + inv[:3, 3]

    eyeL = face._lm("face_eye.L")
    eyeR = face._lm("face_eye.R")
    ipd = float(np.linalg.norm(eyeL - eyeR))
    eyeC = (eyeL + eyeR) / 2.0

    # ---- REAL landmarks from the geometry detector (front-profile scan:
    # lips, nose, chin measured on the mesh - never trust an unfitted net)
    L = {}
    try:
        L, _ipd2, _sure = face.detect_landmarks(props, body)
        L = {k: np.asarray(v, float) for k, v in L.items()}
    except Exception:
        pass

    def pick(name, fallback):
        # the user's placed/derived MARKER is the ground truth in the
        # FaceIt-style flow (same as the eyes, which already use _lm).
        # The geometry detector is only a fallback - it mis-reads the lip
        # line low on stylised meshes and used to drag mouth/cheek/jaw down.
        try:
            m = face._lm(name)
            if m is not None:
                return np.asarray(m, float)
        except Exception:
            pass
        v = L.get(name)
        return v if v is not None else np.asarray(fallback, float)

    try:
        jaw_fb = face._lm("face_jaw.L")
    except Exception:
        jaw_fb = np.array([1.2 * ipd, float(eyeC[1]) + ipd,
                           float(eyeC[2]) - 0.8 * ipd])
    jawL = pick("face_jaw.L", jaw_fb)
    P_jaw = np.array([0.0, float(jawL[1]), float(jawL[2])])

    lip_T = pick("face_lip_up", gp["lip_T"])
    lip_B = pick("face_lip_low", gp["lip_B"])
    mouth_mid = (lip_T + lip_B) / 2.0
    # the grid mouth ring is only trusted when it actually sits AT the
    # detected mouth (i.e. the user registered a real lip loop)
    grid_mouth = (np.asarray(gp["lip_T"], float) +
                  np.asarray(gp["lip_B"], float)) / 2.0
    mouth_ok = float(np.linalg.norm(grid_mouth - mouth_mid)) < 0.35 * ipd
    if mouth_ok:
        lip_T, lip_B = gp["lip_T"], gp["lip_B"]
        mouth_mid = (np.asarray(lip_T) + np.asarray(lip_B)) / 2.0
    cornL = gp["mouth_corner.L"] if mouth_ok else \
        pick("face_mouth_corner.L", gp["mouth_corner.L"])
    nose_tip = pick("face_nose", gp["nose_tip"])
    browL = pick("face_brow.L", gp["brow_mid.L"])
    earL = pick("face_ear.L", gp["ear_low.L"])

    ours = {
        "eye.L": eyeL, "eye.R": eyeR,
        "eye_target": np.array([0.0, float(eyeC[1]) - 6.0 * ipd,
                                float(eyeC[2])]),
        "face_upp": np.array([0.0, float(P_jaw[1]), float(browL[2])]),
        "face_low": np.array([0.0, float(P_jaw[1]), float(mouth_mid[2])]),
        "mouth": mouth_mid,
        "nose_tip": nose_tip,
        "nose_base": np.array([0.0, float(nose_tip[1]) + 0.10 * ipd,
                               float(mouth_mid[2] + 0.45 *
                                     (nose_tip[2] - mouth_mid[2]))]),
        "lip_T": lip_T, "lip_B": lip_B,
    }
    for s, sgn in ((".L", 1.0), (".R", -1.0)):
        mir = np.array([sgn, 1.0, 1.0])
        ours["ear" + s] = earL * mir
        ours["corner" + s] = np.asarray(cornL, float) * mir
        bm = browL * mir
        ours["brow_mid" + s] = bm
        ours["brow_in" + s] = np.array([sgn * 0.35 * abs(bm[0]),
                                        bm[1], bm[2] + 0.04 * ipd])
        ours["brow_out" + s] = np.array([sgn * 1.55 * abs(bm[0]),
                                         bm[1] + 0.10 * ipd,
                                         bm[2] - 0.05 * ipd])
        ours["brow_all" + s] = bm
        cz = 0.5 * (float(eyeC[2]) + float(mouth_mid[2]))
        cx = sgn * abs(float(cornL[0])) * 1.35
        ours["cheek_all" + s] = np.array([cx, float(mouth_mid[1]) +
                                          0.15 * ipd, cz])
        ours["cheek_in" + s] = np.array([0.55 * cx, float(mouth_mid[1]) +
                                         0.10 * ipd, cz])
        ours["cheek_mid" + s] = ours["cheek_all" + s]
        ours["cheek_out" + s] = np.array([1.45 * cx, float(mouth_mid[1]) +
                                          0.35 * ipd, cz])

    # refine brows/nose/cheeks/ears against the REGISTERED geometry + skin
    try:
        ours = _real_face_anchors(face, props, body, gp, ours, ipd)
    except Exception:
        import traceback
        traceback.print_exc()

    src, dst = [], []

    def add(sp, op):
        if sp is None or op is None:
            return
        src.append([float(v) for v in sp])
        dst.append([float(v) for v in to_rig(op)])

    for k, op in ours.items():
        add(A.get(k), op)

    # ---- eyelid rings: the grid ring is used ONLY if it truly circles
    # the eye (registered); otherwise a clean ellipse around the eye ----
    ring_keys = ("eye_in", "lid_T_in", "lid_T", "lid_T_out",
                 "eye_out", "lid_B_out", "lid_B", "lid_B_in")
    try:
        eye_r = {}
        for ob_e in face._eye_meshes(props, body):
            c_e = utils.read_rest_coords(ob_e)
            ec_e = c_e.mean(axis=0)
            s_e = ".L" if ec_e[0] >= 0 else ".R"
            eye_r[s_e] = float(np.linalg.norm(c_e - ec_e, axis=1).max())
    except Exception:
        eye_r = {}
    for s in (".L", ".R"):
        ec = np.asarray(ours["eye" + s], float)
        r = eye_r.get(s, 0.5 * ipd)
        grid_ring = [np.asarray(gp[k + s], float) for k in ring_keys]
        gc = np.mean(grid_ring, axis=0)
        ring_ok = float(np.linalg.norm(gc[[0, 2]] - ec[[0, 2]])) < 0.5 * r \
            and float(np.linalg.norm(gc - ec)) < 2.5 * r
        if ring_ok:
            our_ring = grid_ring
        else:
            yf = float(ec[1]) - 0.9 * r
            our_ring = []
            for k in range(8):
                a = math.pi * 0.25 * k
                our_ring.append(np.array([
                    float(ec[0]) + 0.78 * r * math.cos(a), yf,
                    float(ec[2]) + 0.62 * r * math.sin(a)]))
        for part in ("upp", "low"):
            spts = A.get("lid_ring_%s%s" % (part, s))
            if not spts:
                continue
            sc = A["eye" + s]
            opts = _ring_resample(our_ring, ec, spts, sc)
            for sp, op in zip(spts, opts):
                add(sp, op)

    # ---- lip ring: registered grid loop when valid, else a clean
    # ellipse through the DETECTED corners + lips ----
    if mouth_ok:
        lip_ring = [gp["lip_T"], gp["lip_T.L"], gp["mouth_corner.L"],
                    gp["lip_B.L"], gp["lip_B"], gp["lip_B.R"],
                    gp["mouth_corner.R"], gp["lip_T.R"]]
    else:
        cl = np.asarray(ours["corner.L"], float)
        cr = np.asarray(ours["corner.R"], float)
        lt = np.asarray(lip_T, float)
        lb = np.asarray(lip_B, float)
        lip_ring = [lt, (lt + cl) / 2 + np.array([0, 0, 0.2]) * 0,
                    cl, (lb + cl) / 2, lb, (lb + cr) / 2, cr,
                    (lt + cr) / 2]
    s_mouth = A.get("mouth")
    for s in (".L", ".R"):
        for part in ("upp", "low"):
            spts = A.get("lip_ring_%s%s" % (part, s))
            if not spts:
                continue
            opts = _ring_resample(lip_ring, mouth_mid, spts, s_mouth)
            for sp, op in zip(spts, opts):
                add(sp, op)

    # teeth / tongue from the registered meshes
    up_pts, _, _ = _obj_or_vg_pts(props, body, "skin_teeth_up", "SR_teeth_up")
    lo_pts, _, _ = _obj_or_vg_pts(props, body, "skin_teeth_low",
                                  "SR_teeth_low")
    if up_pts is not None:
        add(A.get("teeth_upp"), up_pts.mean(axis=0))
    if lo_pts is not None:
        add(A.get("teeth_low"), lo_pts.mean(axis=0))
    tg_pts, _, _ = _obj_or_vg_pts(props, body, "skin_tongue", "SR_tongue")
    st = A.get("tongue")
    if tg_pts is not None and st and all(p is not None for p in st):
        ys = tg_pts[:, 1]
        y0, y1 = float(ys.max()), float(ys.min())   # back(+y) -> front(-y)
        sy = [p[1] for p in st]
        back_first = sy[0] > sy[-1]
        order = st if back_first else list(reversed(st))
        for i, sp in enumerate(order):
            t = i / (len(order) - 1.0)
            yb = y0 + (y1 - y0) * t
            band = tg_pts[np.abs(ys - yb) < max(0.12 * abs(y1 - y0), 1e-5)]
            if not len(band):
                continue
            add(sp, band.mean(axis=0))

    # ---- SYMMETRIZE: Storm is X-symmetric and the ribbon helpers use a
    # MIRROR modifier, so the mapping MUST be X-symmetric too, or the
    # mirrored ribbon halves land off the R-side bones (rest offsets).
    seen = {}
    for s, d in zip(src, dst):
        for sm, dm in ((s, d), ([-s[0], s[1], s[2]], [-d[0], d[1], d[2]])):
            key = (round(sm[0], 4), round(sm[1], 4), round(sm[2], 4))
            if key not in seen:
                seen[key] = (list(sm), list(dm))
    src = [v[0] for v in seen.values()]
    dst = [v[1] for v in seen.values()]
    return np.array(src), np.array(dst), ipd


# ------------------------------------------------------------- helpers
def _resolve_obj(name, rig, made_objs):
    if name is None:
        return None
    if name == "RIG-storm":
        return rig
    if name == "GEO-storm-head":
        return made_objs.get("__body__")
    if name.startswith("HLP-storm-"):
        return made_objs.get("HLP-SR-" + name[len("HLP-storm-"):])
    return bpy.data.objects.get(name)


def _map_sub(name, spec_bones, head_parent):
    if not name:
        return name
    if name in spec_bones:
        return name
    if name in _HEAD_ALIASES:
        return head_parent
    return name


def _helper_coll():
    c = bpy.data.collections.get("SR_FaceHelpers")
    if c is None:
        c = bpy.data.collections.new("SR_FaceHelpers")
        bpy.context.scene.collection.children.link(c)
    return c


def _clean_previous(face, props, context, rig, body, spec):
    """Remove bones/objects/modifiers from any previous face build
    (simple v1.99 layout OR a previous Storm build)."""
    old = ["DEF-jaw", "CTL-jaw", "CTL-Jaw", "master-mouth", "MSTR-Mouth",
           "CTL-eyes", "MSTR-Eye_target", "MSTR-Face_upp", "MSTR-Face_low",
           "MSTR-Nose", "CTL-Lips_main_upp", "CTL-Lips_main_low",
           "DEF-Lips_main_upp", "DEF-Lips_main_low", "DEF-Nose"]
    for s in (".L", ".R"):
        old += ["master-eye" + s, "MSTR-Eye" + s, "DEF-eye" + s,
                "CTL-eye" + s, "TGT-Eye" + s, "DEF-ear" + s,
                "MCH-Lips_corn" + s, "CTL-Lips_corn" + s, "DEF-Lips_corn" + s,
                "CTL-Brow_all" + s, "CTL-Cheek_all" + s,
                "CTL-Lid_upp" + s, "CTL-Lid_low" + s,
                "DEF-Lid_upp" + s, "DEF-Lid_low" + s]
        for p in ("in", "mid", "out"):
            old += ["CTL-Brow_%s%s" % (p, s), "DEF-Brow_%s%s" % (p, s),
                    "CTL-Cheek_%s%s" % (p, s), "DEF-Cheek_%s%s" % (p, s)]
    prev = list(rig.data.get("sr_storm_face_bones", []))
    names = set(old) | set(prev) | set(spec["bones"].keys())
    names |= {b.name for b in rig.data.bones
              if b.name.startswith(("CTL-Lips_local", "DEF-Lips_local"))}

    # vgroup renames that must SURVIVE (analytic carves we keep)
    if body is not None:
        vg = body.vertex_groups
        if vg.get("DEF-jaw") is not None and vg.get("DEF-Jaw") is None:
            vg["DEF-jaw"].name = "DEF-Jaw"
        for s in (".L", ".R"):
            g = vg.get("DEF-ear" + s)
            if g is not None and vg.get("DEF-Ear_base" + s) is None:
                g.name = "DEF-Ear_base" + s

    # strip other stale face vgroups of deleted bones (renormalize base)
    strip = [n for n in names if body is not None
             and body.vertex_groups.get(n) is not None
             and n not in ("DEF-Jaw",)]
    if strip:
        _strip_groups(body, strip)

    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    n = 0
    for nm in names:
        b = eb.get(nm)
        if b is not None:
            eb.remove(b)
            n += 1
    bpy.ops.object.mode_set(mode='OBJECT')

    # previous helper objects + lattice modifiers on the body
    for ob in list(bpy.data.objects):
        if ob.name.startswith(("HLP-SR-",)):
            bpy.data.objects.remove(ob, do_unlink=True)
    if body is not None:
        for m in list(body.modifiers):
            if m.name.startswith(("FACE-LTC", "FACE-CorrSmooth")):
                body.modifiers.remove(m)
    return n


def _strip_groups(body, names, deform_names=None):
    """Remove the given vgroups, giving their weight back to the remaining
    DEFORM groups proportionally (never to MSK/utility groups)."""
    me = body.data
    lidx = {g.index for g in body.vertex_groups if g.name in set(names)}
    if not lidx:
        return
    gmap = {g.index: g for g in body.vertex_groups}
    if deform_names is None:
        keep = {g.index for g in body.vertex_groups
                if not g.name.startswith(("MSK", "SR_", "WGT"))}
    else:
        keep = {g.index for g in body.vertex_groups
                if g.name in deform_names}
    for v in me.vertices:
        tot_l = sum(g.weight for g in v.groups if g.group in lidx)
        if tot_l <= 1e-6:
            continue
        rest = [(g.group, g.weight) for g in v.groups
                if g.group not in lidx and g.group in keep]
        tot_r = sum(x[1] for x in rest)
        if tot_r > 1e-6:
            f = (tot_r + tot_l) / tot_r
            for gi, gw in rest:
                gmap[gi].add([v.index], gw * f, 'REPLACE')
    for gi in sorted(lidx, reverse=True):
        body.vertex_groups.remove(gmap[gi])


# ------------------------------------------------------------- bones
def _build_bones(rig, spec, rbf, head_parent):
    bones = spec["bones"]
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    order = []
    seen = set()

    def visit(n):
        if n in seen or n not in bones:
            return
        seen.add(n)
        p = bones[n]["parent"]
        if p in bones:
            visit(p)
        order.append(n)
    for n in bones:
        visit(n)

    min_len = 1e-4
    for n in order:
        e = bones[n]
        b = eb.get(n)
        if b is None:
            b = eb.new(n)
        h = rbf.p(e["head"])
        t = rbf.p(e["tail"])
        if np.linalg.norm(t - h) < min_len:
            t = h + np.array([0.0, 0.0, min_len])
        b.head = Vector([float(v) for v in h])
        b.tail = Vector([float(v) for v in t])
        try:
            b.align_roll(Vector([float(v) for v in
                                 rbf.dir(e["head"], e["zaxis"])]))
        except Exception:
            pass
        b.use_deform = e["deform"]
        b.inherit_scale = e.get("inherit_scale", 'FULL')
        b.use_inherit_rotation = e.get("inherit_rot", True)
        b.use_local_location = e.get("local_loc", True)
    # parenting second pass (storm world-roots STAY roots - their follow
    # behaviour comes from their constraints, copied verbatim)
    for n in order:
        e = bones[n]
        b = eb[n]
        p = e["parent"]
        if p is None:
            if n == "Face_Root":        # storm hangs this under the head
                if head_parent and head_parent in eb:
                    b.parent = eb[head_parent]
            continue
        pn = p if (p in bones) else _map_sub(p, bones, head_parent)
        if pn and pn in eb:
            b.parent = eb[pn]
            b.use_connect = bool(e.get("connect")) and \
                (b.head - eb[pn].tail).length < 1e-5
    # bbones third pass (handles need all bones present)
    for n in order:
        bb = bones[n].get("bbone")
        if not bb:
            continue
        b = eb[n]
        for k, v in bb.items():
            if k.startswith("bbone_custom_handle"):
                b_h = eb.get(v) if v else None
                setattr(b, k, b_h)
            else:
                try:
                    setattr(b, k, v)
                except Exception:
                    pass
    # collections (visibility copied from Storm via spec meta)
    data = rig.data
    coll_vis = spec.get("coll_vis", {})
    bpy.ops.object.mode_set(mode='OBJECT')
    # IMPORTANT: look up existing collections in collections_ALL (nested
    # too). data.collections.get() only searches ROOTS - after organize()
    # nests "Face MCH" under "Rigging", a rebuild would not find it and
    # create "Face MCH.001".002... one per bone = hundreds of duplicate
    # visible root collections (the "scattered rig" bug).
    def _coll(cn):
        c = data.collections_all.get(cn)
        if c is None:
            c = data.collections.new(cn)
        return c
    for n in order:
        e = bones[n]
        db = data.bones.get(n)
        if db is None:
            continue
        for cn in e.get("colls", []):
            _coll(cn).assign(db)
    # visibility: helper/mechanism layers OFF for a clean animator view
    hide_colls = {"Face MCH", "Face Display", "Upper Master", "Lower Master",
                  "Mouth micro", "Eyes_micro", "Lattices", "Tweak"}
    for cn, vis in coll_vis.items():
        c = data.collections_all.get(cn)
        if c is not None:
            c.is_visible = bool(vis) and cn not in hide_colls
    for cn in hide_colls:
        c = data.collections_all.get(cn)
        if c is not None:
            c.is_visible = False
    # DSP display dots stay visible like Storm
    c = data.collections_all.get("Face Display")
    if c is not None:
        c.is_visible = True
    return order


def _apply_pose(rig, spec, made_objs, head_parent):
    bones = spec["bones"]
    for n, e in bones.items():
        pb = rig.pose.bones.get(n)
        if pb is None:
            continue
        try:
            pb.rotation_mode = e.get("rot_mode", 'QUATERNION')
        except Exception:
            pass
        try:
            rig.data.bones[n].hide = bool(e.get("hide", False))
        except Exception:
            pass
        for k, attr in (("lock_loc", "lock_location"),
                        ("lock_rot", "lock_rotation"),
                        ("lock_scale", "lock_scale")):
            try:
                setattr(pb, attr, e.get(k, [False] * 3))
            except Exception:
                pass
        try:
            if e.get("palette") and e["palette"] != 'DEFAULT':
                pb.color.palette = e["palette"]
            if e.get("bpalette") and e["bpalette"] != 'DEFAULT':
                rig.data.bones[n].color.palette = e["bpalette"]
        except Exception:
            pass
        for k, pd in (e.get("props") or {}).items():
            try:
                pb[k] = pd["value"]
                ui = pb.id_properties_ui(k)
                kw = {}
                for f in ("min", "max", "soft_min", "soft_max", "default"):
                    if pd.get(f) is not None:
                        kw[f] = pd[f]
                if kw:
                    ui.update(**kw)
            except Exception:
                pass
        if e.get("wgt"):
            try:
                _fw.assign(rig, n, e["wgt"], tuple(e.get("wgt_scale", (1, 1, 1))),
                           None)
                pb.custom_shape_rotation_euler = e.get("wgt_rot", (0, 0, 0))
                pb.custom_shape_translation = e.get("wgt_tr", (0, 0, 0))
                pb.use_custom_shape_bone_size = e.get("wgt_bone_size", True)
            except Exception:
                pass
    # any face bone WITHOUT a widget is machinery, not a control: hide it
    # (Storm keeps them nearly invisible via its display setup; on our rig
    # they would draw as 538 black octahedrons - the "horror movie")
    for n, e in bones.items():
        if not e.get("wgt"):
            db = rig.data.bones.get(n)
            if db is not None:
                db.hide = True
    # custom_shape_transform second pass
    for n, e in bones.items():
        xf = e.get("wgt_xform")
        if not xf:
            continue
        pb = rig.pose.bones.get(n)
        tb = rig.pose.bones.get(xf)
        if pb is not None and tb is not None:
            try:
                pb.custom_shape_transform = tb
            except Exception:
                pass


_LEN_PROPS = ("distance", "rest_length", "falloff_radius")
_LOC_RANGE = tuple("%s_%s%s" % (a, mm, ax) for a in ("from", "to")
                   for mm in ("min", "max") for ax in ("x", "y", "z"))

# Prefix priority when a DSP display bone must borrow a follow-source from
# our own rig (deform first, then the local control chain).
_DSP_SRC_PREF = ("DEF-", "P-", "STR-", "MSTR-", "CTL-", "MCH-", "ORG-")


def _dsp_source(rig, owner):
    """A DSP display bone in Storm is pinned to a head-mesh VERTEX GROUP
    (COPY_LOCATION, subtarget = vgroup name).  Our character mesh carries
    no such groups, so that constraint copies the mesh origin and the
    whole 'Face Display' layer collapses to (0,0,0) in pose.  Return the
    bone in OUR rig whose head sits where this display bone belongs -
    preferably the matching deform bone DEF-<stem>, else the nearest
    placed non-DSP bone - so the display follows the same skin point."""
    dbones = rig.data.bones
    d = dbones.get(owner)
    if d is None:
        return None
    mw = rig.matrix_world

    def placed(b):
        return (mw @ Vector(b.head_local)).length > 0.05

    stem = owner[4:]
    exact = dbones.get("DEF-" + stem)
    if exact is not None and placed(exact):
        return exact.name
    h = mw @ Vector(d.head_local)
    cands = []
    for nb in dbones:
        if nb.name.startswith("DSP-"):
            continue
        if not placed(nb):
            continue
        cands.append(((mw @ Vector(nb.head_local) - h).length, nb.name))
    if not cands:
        return None
    cands.sort()
    within = [nm for dd, nm in cands if dd < 0.015]
    pool = within if within else [cands[0][1]]
    for pref in _DSP_SRC_PREF:
        for nm in pool:
            if nm.startswith(pref):
                return nm
    return pool[0]


def _apply_constraints(rig, spec, made_objs, head_parent, scale=1.0):
    bones = spec["bones"]
    skip = {"type", "name", "targets"}
    n_con = 0
    for n, e in bones.items():
        pb = rig.pose.bones.get(n)
        if pb is None or not e.get("cons"):
            continue
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        for cd in e["cons"]:
            # Storm ties Face_Root & co. to ITS head geometry with
            # STRETCH/COPY constraints; on our rig those geometric
            # assumptions do not hold and they drag the whole face.
            # Parenting already provides the follow - drop them (ARP
            # style: bones stay exactly where the build put them).
            sub0 = cd.get("subtarget")
            if sub0 in _HEAD_ALIASES and cd["type"] in (
                    'STRETCH_TO', 'COPY_LOCATION', 'COPY_TRANSFORMS',
                    'DAMPED_TRACK', 'IK', 'COPY_SCALE'):
                continue
            # Storm 'Face Display' bones (DSP-*) are pinned to head-mesh
            # vertex groups our character lacks -> without this they copy
            # the mesh origin and the whole display layer collapses to
            # (0,0,0).  Re-anchor them to the matching bone in our rig so
            # the display follows the same skin point (see _dsp_source).
            tgt0 = cd.get("target")
            if (n.startswith("DSP-")
                    and cd["type"] in ('COPY_LOCATION', 'COPY_TRANSFORMS')
                    and isinstance(tgt0, dict)
                    and tgt0.get("__obj__") == "GEO-storm-head"):
                src = _dsp_source(rig, n)
                if src:
                    con = pb.constraints.new('COPY_LOCATION')
                    con.name = cd.get("name", "Copy Location")
                    con.target = rig
                    con.subtarget = src
                    try:
                        con.influence = float(cd.get("influence", 1.0))
                    except Exception:
                        pass
                    n_con += 1
                    continue
            try:
                con = pb.constraints.new(cd["type"])
            except Exception:
                continue
            con.name = cd.get("name", cd["type"])
            map_from = cd.get("map_from")
            map_to = cd.get("map_to")
            for k, v in cd.items():
                if k in skip:
                    continue
                try:
                    # absolute LENGTHS were measured on Storm: rescale
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if k in _LEN_PROPS:
                            v = v * scale
                        elif cd["type"] == 'LIMIT_LOCATION' and \
                                k.startswith(("min_", "max_")):
                            v = v * scale
                        elif cd["type"] == 'TRANSFORM' and k in _LOC_RANGE:
                            which = ("LOCATION" if map_from is None else
                                     map_from) if k.startswith("from") else \
                                    ("LOCATION" if map_to is None else map_to)
                            if which == 'LOCATION':
                                v = v * scale
                    if isinstance(v, dict) and "__obj__" in v:
                        setattr(con, k, _resolve_obj(v["__obj__"], rig,
                                                     made_objs))
                    elif k == "subtarget":
                        setattr(con, k, _map_sub(v, bones, head_parent))
                    elif isinstance(v, list):
                        setattr(con, k, v)
                    else:
                        setattr(con, k, v)
                except Exception:
                    pass
            if cd["type"] == 'ARMATURE':
                for td in cd.get("targets", []):
                    try:
                        t = con.targets.new()
                        t.target = _resolve_obj(td.get("target"), rig,
                                                made_objs)
                        t.subtarget = _map_sub(td.get("subtarget"), bones,
                                               head_parent)
                        t.weight = td.get("weight", 1.0)
                    except Exception:
                        pass
            n_con += 1
    return n_con


def _fix_stretch_rest(context, rig, names):
    """STRETCH_TO must be length-neutral at rest: rest_length = distance
    from the owner's head to the target point IN OUR rig (Storm's values
    are Storm-sized). Handles bone subtargets AND object targets (the
    vertex-parented lip hook empties)."""
    context.view_layer.update()
    dg = context.evaluated_depsgraph_get()
    n = 0
    for nm in names:
        pb = rig.pose.bones.get(nm)
        if pb is None:
            continue
        ob_b = rig.data.bones.get(nm)
        head_w = rig.matrix_world @ Vector(ob_b.head_local)
        for c in pb.constraints:
            if c.type != 'STRETCH_TO':
                continue
            if c.subtarget and c.target == rig:
                tb = rig.data.bones.get(c.subtarget)
                if tb is None:
                    continue
                ht = getattr(c, "head_tail", 0.0)
                pt = Vector(tb.head_local).lerp(Vector(tb.tail_local), ht)
                pt_w = rig.matrix_world @ pt
            elif c.target is not None:
                tgt_ev = c.target.evaluated_get(dg)
                pt_w = tgt_ev.matrix_world.translation
            else:
                continue
            c.rest_length = (pt_w - head_w).length
            n += 1
    return n


def _rest_reconcile(context, rig, names, iters=8, tol=5e-5, max_snap=0.06):
    """Make the CONSTRAINED rest state the actual rest: evaluate every
    face bone with its constraints applied and snap its edit-bone to the
    evaluated matrix, iterating until stable. This is what guarantees the
    Storm regression standard: evaluated rest mesh == base (0.00000)."""
    import numpy as _np
    total = 0
    for _ in range(iters):
        # STRETCH_TO must stay length-neutral as the rests move, or the
        # lip micro bones oscillate and never converge
        _fix_stretch_rest(context, rig, names)
        context.view_layer.update()
        dg = context.evaluated_depsgraph_get()
        rig_ev = rig.evaluated_get(dg)
        target = {}
        for n in names:
            pb = rig_ev.pose.bones.get(n)
            db = rig.data.bones.get(n)
            if pb is None or db is None:
                continue
            M = _np.array(pb.matrix)
            R = _np.array(db.matrix_local)
            if _np.abs(M - R).max() > tol:
                # SAFETY: a constraint that pulls a bone far away is a
                # broken assumption, not a rest state - never chase it
                if _np.linalg.norm(M[:3, 3] - R[:3, 3]) > max_snap:
                    continue
                target[n] = M
        if not target:
            break
        bpy.ops.object.mode_set(mode='EDIT')
        eb = rig.data.edit_bones
        for n, M in target.items():
            b = eb.get(n)
            if b is None:
                continue
            L = max(b.length, 1e-4)
            head = Vector((float(M[0][3]), float(M[1][3]), float(M[2][3])))
            ydir = Vector((float(M[0][1]), float(M[1][1]), float(M[2][1])))
            zdir = Vector((float(M[0][2]), float(M[1][2]), float(M[2][2])))
            b.head = head
            b.tail = head + ydir.normalized() * L
            try:
                b.align_roll(zdir)
            except Exception:
                pass
        bpy.ops.object.mode_set(mode='OBJECT')
        total = len(target)
    return total


# ------------------------------------------------------- helper objects
def _make_ribbons(rig, H, rbf, made_objs):
    coll = _helper_coll()
    for snm in ("HLP-storm-geometry_ribbon_eyelids",
                "HLP-storm-geometry_ribbon_lips",
                "HLP-storm-geometry_ribbon_blink.L",
                "HLP-storm-geometry_ribbon_blink.R"):
        R = H.get(snm)
        if not R:
            continue
        nm = "HLP-SR-" + snm[len("HLP-storm-"):]
        old = bpy.data.objects.get(nm)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        me = bpy.data.meshes.new(nm)
        verts = rbf(np.array(R["verts"], float))
        me.from_pydata([Vector([float(x) for x in v]) for v in verts],
                       [tuple(e) for e in R["edges"]],
                       [tuple(f) for f in R["faces"]])
        me.update()
        ob = bpy.data.objects.new(nm, me)
        coll.objects.link(ob)
        ob.parent = rig
        ob.matrix_world = rig.matrix_world.copy()
        for gname, lst in R["vgroups"].items():
            g = ob.vertex_groups.new(name=gname)
            for vi, w in lst:
                g.add([int(vi)], float(w), 'REPLACE')
        # shape keys
        for sk in R.get("shape_keys", []):
            kb = ob.shape_key_add(name=sk["name"], from_mix=False)
            sv = rbf(np.array(sk["v"], float))
            for i in range(len(me.vertices)):
                kb.data[i].co = Vector([float(x) for x in sv[i]])
            kb.value = sk.get("value", 0.0)
            kb.mute = sk.get("mute", False)
        if me.shape_keys:
            for sk in R.get("shape_keys", []):
                kb = me.shape_keys.key_blocks.get(sk["name"])
                rel = sk.get("rel")
                if kb and rel and me.shape_keys.key_blocks.get(rel):
                    kb.relative_key = me.shape_keys.key_blocks[rel]
        # modifiers in storm ORDER
        for md in R["mods"]:
            if md["type"] == 'MIRROR':
                m = ob.modifiers.new(md["name"], 'MIRROR')
                try:
                    m.use_axis = md.get("use_axis", [True, False, False])
                    m.use_mirror_vertex_groups = md.get(
                        "use_mirror_vertex_groups", True)
                except Exception:
                    pass
            elif md["type"] == 'MASK':
                m = ob.modifiers.new(md["name"], 'MASK')
                m.vertex_group = md.get("vertex_group", "")
                try:
                    m.invert_vertex_group = md.get("invert_vertex_group",
                                                   False)
                except Exception:
                    pass
            elif md["type"] == 'ARMATURE':
                m = ob.modifiers.new(md["name"], 'ARMATURE')
                m.object = rig
                m.use_vertex_groups = True
        ob.hide_set(True)
        ob.hide_render = True
        made_objs[nm] = ob
    return made_objs


def _make_empties(rig, H, made_objs):
    """Vertex-parented empties riding the ribbon meshes (Storm's lips_hook
    family): the lip MCH bones STRETCH_TO them, so ribbon motion drives
    the micro lip bones."""
    coll = _helper_coll()
    n = 0
    for snm, E in H.get("empties", {}).items():
        if E.get("type") != 'EMPTY':
            continue
        nm = "HLP-SR-" + snm[len("HLP-storm-"):]
        old = bpy.data.objects.get(nm)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        ob = bpy.data.objects.new(nm, None)
        ob.empty_display_size = E.get("empty_display_size", 0.005)
        coll.objects.link(ob)
        par = E.get("parent")
        pob = None
        if par and par.startswith("HLP-storm-"):
            pob = made_objs.get("HLP-SR-" + par[len("HLP-storm-"):])
        if pob is not None and E.get("parent_type") in ('VERTEX', 'VERTEX_3'):
            ob.parent = pob
            ob.parent_type = E["parent_type"]
            try:
                ob.parent_vertices = E.get("parent_vertices", [0, 0, 0])
            except Exception:
                pass
            # land exactly ON the parent vertex
            ob.matrix_parent_inverse.identity()
            ob.location = (0.0, 0.0, 0.0)
        ob.hide_set(True)
        ob.hide_render = True
        made_objs[nm] = ob
        n += 1
    return n


def _sk_drivers(rig, H, made_objs, spec_bones, head_parent):
    n = 0
    for snm in ("HLP-storm-geometry_ribbon_blink.L",
                "HLP-storm-geometry_ribbon_blink.R",
                "HLP-storm-geometry_ribbon_lips"):
        R = H.get(snm)
        if not R or not R.get("sk_drivers"):
            continue
        ob = made_objs.get("HLP-SR-" + snm[len("HLP-storm-"):])
        if ob is None or ob.data.shape_keys is None:
            continue
        key = ob.data.shape_keys
        for dd in R["sk_drivers"]:
            try:
                fc = _driver_add(key, dd["data_path"], dd["array_index"])
                d = fc.driver
                d.type = dd["type"]
                d.expression = dd["expression"]
                for vd in dd["vars"]:
                    v = d.variables.new()
                    v.name = vd["name"]
                    v.type = vd["type"]
                    for t, td in zip(v.targets, vd["targets"]):
                        t.id = _resolve_obj(td.get("id"), rig, made_objs)
                        if td.get("bone_target"):
                            t.bone_target = _map_sub(td["bone_target"],
                                                     spec_bones, head_parent)
                        if td.get("data_path"):
                            t.data_path = td["data_path"]
                        if td.get("transform_type"):
                            t.transform_type = td["transform_type"]
                        if td.get("transform_space"):
                            t.transform_space = td["transform_space"]
                n += 1
            except Exception:
                pass
    return n


def _make_lattices(rig, body, H, rbf, made_objs):
    coll = _helper_coll()
    lat = H.get("lattices", {})
    order = []
    for snm, L in lat.items():
        if "Watch" in snm or "watch" in snm:
            continue
        nm = "HLP-SR-" + snm[len("HLP-storm-"):]
        old = bpy.data.objects.get(nm)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        lt = bpy.data.lattices.new(nm)
        lt.points_u, lt.points_v, lt.points_w = L["pu"], L["pv"], L["pw"]
        lt.interpolation_type_u = L["iu"]
        lt.interpolation_type_v = L["iv"]
        lt.interpolation_type_w = L["iw"]
        lt.use_outside = L.get("use_outside", False)
        ob = bpy.data.objects.new(nm, lt)
        coll.objects.link(ob)
        M = np.array(L["matrix_arm"], float)
        c = M[:3, 3]
        J = rbf.jac(c)
        R3 = J @ M[:3, :3]
        newM = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                newM[i][j] = float(R3[i, j])
        nc = rbf.p(c)
        for i in range(3):
            newM[i][3] = float(nc[i])
        ob.matrix_world = rig.matrix_world @ newM
        # base points stay default; copy vgroups
        for gname, lst in L.get("vgroups", {}).items():
            g = ob.vertex_groups.new(name=gname)
            for vi, w in lst:
                g.add([int(vi)], float(w), 'REPLACE')
        ob.hide_set(True)
        ob.hide_render = True
        made_objs[nm] = ob
        order.append((snm, nm))
    # hooks after depsgraph knows the bones
    bpy.context.view_layer.update()
    for snm, nm in order:
        L = lat[snm]
        ob = made_objs[nm]
        for hk in L.get("hooks", []):
            sub = hk.get("subtarget")
            pb = rig.pose.bones.get(sub)
            if pb is None:
                continue
            m = ob.modifiers.new(hk.get("name", "Hook"), 'HOOK')
            m.object = rig
            m.subtarget = sub
            if hk.get("vertex_group"):
                m.vertex_group = hk["vertex_group"]
            try:
                m.strength = hk.get("strength", 1.0)
                m.falloff_type = hk.get("falloff_type", 'NONE')
                m.falloff_radius = hk.get("falloff_radius", 0.0)
            except Exception:
                pass
            # bind so the CURRENT (rest) state produces zero deformation.
            # Blender hook math: T = ob_world^-1 @ target_world@bone.matrix
            # @ matrix_inverse  ->  identity when:
            try:
                m.matrix_inverse = \
                    (rig.matrix_world @ pb.matrix).inverted() @ \
                    ob.matrix_world
            except Exception:
                pass
    return order


def _region_group(body, name, center_w, radius, feather=0.6):
    """(Re)create a radial mask vgroup on the body in WORLD space."""
    g = body.vertex_groups.get(name)
    if g is not None:
        body.vertex_groups.remove(g)
    g = body.vertex_groups.new(name=name)
    co = utils.read_rest_coords(body)
    d = np.linalg.norm(co - np.asarray(center_w, float), axis=1)
    t = np.clip((d - radius * feather) / max(radius * (1 - feather), 1e-6),
                0.0, 1.0)
    w = 1.0 - (t * t * (3 - 2 * t))
    idx = np.where(w > 1e-3)[0]
    for i in idx:
        g.add([int(i)], float(w[i]), 'REPLACE')
    return g


def _body_lattice_stack(rig, body, H, rbf, made_objs):
    """Recreate Storm's head modifier stack on our body: lattices (with
    generated region masks) + corrective smooth. Inserted after the
    armature modifier."""
    if body is None:
        return 0
    inv = np.array(bpy.data.objects[body.name].matrix_world.inverted())
    Mw = np.array(rig.matrix_world)

    def rig_to_world(p):
        return Mw[:3, :3] @ np.asarray(p, float) + Mw[:3, 3]

    masks = H.get("head_mask_stats", {})
    n = 0
    mods = []
    for md in H.get("head_mods", []):
        if md["type"] == 'LATTICE':
            snm = md.get("object")
            if not snm:
                continue
            nm = "HLP-SR-" + snm[len("HLP-storm-"):]
            ob = made_objs.get(nm)
            if ob is None:
                continue
            m = body.modifiers.new("FACE-LTC-" + nm[7:], 'LATTICE')
            m.object = ob
            vg = md.get("vertex_group")
            if vg and vg in masks:
                st = masks[vg]
                cen = rig_to_world(rbf.p(st["centroid"]))
                rad = st["max_r"] * rbf.jac_scale(st["centroid"])
                _region_group(body, vg, cen, float(rad))
                m.vertex_group = vg
            try:
                m.strength = md.get("strength", 1.0)
            except Exception:
                pass
            mods.append(m)
            n += 1
        elif md["type"] == 'CORRECTIVE_SMOOTH':
            m = body.modifiers.new("FACE-CorrSmooth", 'CORRECTIVE_SMOOTH')
            for k in ("factor", "iterations", "scale", "smooth_type",
                      "use_only_smooth", "use_pin_boundary", "rest_source"):
                try:
                    setattr(m, k, md[k])
                except Exception:
                    pass
            vg = md.get("vertex_group")
            if vg and vg in masks:
                st = masks[vg]
                cen = rig_to_world(rbf.p(st["centroid"]))
                rad = st["max_r"] * rbf.jac_scale(st["centroid"])
                _region_group(body, vg, cen, float(rad))
                try:
                    m.vertex_group = vg
                except Exception:
                    pass
            mods.append(m)
            n += 1
    # order: right after the armature modifier, preserving list order
    try:
        names = [m.name for m in body.modifiers]
        arm_i = next((i for i, m in enumerate(body.modifiers)
                      if m.type == 'ARMATURE'), 0)
        for k, m in enumerate(mods):
            cur = names.index(m.name) if m.name in names else None
            with bpy.context.temp_override(object=body):
                while body.modifiers.find(m.name) > arm_i + 1 + k:
                    bpy.ops.object.modifier_move_up(modifier=m.name)
    except Exception:
        pass
    return n


# ------------------------------------------------------------- drivers
def _driver_add(idblock, path, idx):
    """driver_add that copes with SCALAR properties (constraint influence,
    custom props): those must be added WITHOUT an array index."""
    try:
        idblock.driver_remove(path, idx)
    except Exception:
        try:
            idblock.driver_remove(path)
        except Exception:
            pass
    if idx is not None and idx >= 0:
        try:
            return idblock.driver_add(path, idx)
        except Exception:
            return idblock.driver_add(path)
    return idblock.driver_add(path)


def _apply_drivers(rig, spec, made_objs, head_parent):
    bones = spec["bones"]
    n = 0
    for dd in spec.get("drivers", []):
        try:
            fc = _driver_add(rig, dd["data_path"], dd["array_index"])
            d = fc.driver
            d.type = dd["type"]
            d.expression = dd["expression"]
            d.use_self = dd.get("use_self", False)
            for vd in dd["vars"]:
                v = d.variables.new()
                v.name = vd["name"]
                v.type = vd["type"]
                for t, td in zip(v.targets, vd["targets"]):
                    t.id = _resolve_obj(td.get("id"), rig, made_objs)
                    if td.get("bone_target"):
                        t.bone_target = _map_sub(td["bone_target"], bones,
                                                 head_parent)
                    if td.get("data_path"):
                        t.data_path = td["data_path"]
                    if td.get("transform_type"):
                        t.transform_type = td["transform_type"]
                    if td.get("transform_space"):
                        t.transform_space = td["transform_space"]
            n += 1
        except Exception:
            pass
    return n


# ------------------------------------------------------------- weights
_CHAINS = [
    ("DEF-Eyelid_upp%d.L", 1, 15), ("DEF-Eyelid_upp%d.R", 1, 15),
    ("DEF-Eyelid_low%d.L", 1, 15), ("DEF-Eyelid_low%d.R", 1, 15),
    ("DEF-Brow_local%d.L", 1, 7), ("DEF-Brow_local%d.R", 1, 7),
    ("DEF-jawline%d.L", 1, 3), ("DEF-jawline%d.R", 1, 3),
]


def _lip_chain(side_fmt="DEF-Lips_%s_micro"):
    up = ["DEF-Lips_upp_micro6.L", "DEF-Lips_upp_micro5.L",
          "DEF-Lips_upp_micro4.L", "DEF-Lips_upp_micro3.L",
          "DEF-Lips_upp_micro2.L", "DEF-Lips_upp_micro1.L",
          "DEF-Lips_upp_micro",
          "DEF-Lips_upp_micro1.R", "DEF-Lips_upp_micro2.R",
          "DEF-Lips_upp_micro3.R", "DEF-Lips_upp_micro4.R",
          "DEF-Lips_upp_micro5.R", "DEF-Lips_upp_micro6.R"]
    low = [n.replace("upp", "low") for n in up]
    return up, low


def face_weights(rig, body, spec, H, rbf, cap=0.8):
    """Procedural face weights: chain-partition along ribbons/strips,
    radial blobs elsewhere, sized from Storm's measured regions; carved
    once, proportionally, from the existing deform weights."""
    if body is None:
        return 0
    stats = H.get("head_group_stats", {})
    have = {b.name for b in rig.data.bones if b.use_deform}
    co = utils.read_rest_coords(body)
    inv = np.array(rig.matrix_world.inverted())
    co_r = co @ inv[:3, :3].T + inv[:3, 3]

    heads = {}
    for b in rig.data.bones:
        heads[b.name] = np.array(b.head_local)

    fields = {}

    def blob(name, cen, rad):
        d = np.linalg.norm(co_r - cen, axis=1)
        t = np.clip(d / max(rad, 1e-6), 0.0, 1.0)
        w = 1.0 - (t * t * (3 - 2 * t))
        if w.max() > 1e-3:
            fields[name] = np.maximum(fields.get(name, 0.0), w)

    chain_members = set()
    chains = []
    for fmt, a, bmax in _CHAINS:
        names = [fmt % i for i in range(a, bmax + 1)]
        names = [n for n in names if n in have]
        if len(names) >= 2:
            chains.append(names)
            chain_members.update(names)
    up, low = _lip_chain()
    for ch in (up, low):
        names = [n for n in ch if n in have]
        if len(names) >= 2:
            chains.append(names)
            chain_members.update(names)

    for names in chains:
        # polyline ON the mesh: storm's painted-weight centroids mapped
        # (the lip/lid bones themselves float in front of the surface)
        pts = np.array([rbf.p(stats[n]["centroid"]) if n in stats
                        else heads[n] for n in names])
        # band radius from storm stats (mapped), fallback to spacing
        rads = []
        for n in names:
            st = stats.get(n)
            if st:
                rads.append(st["mean_r"] * rbf.jac_scale(st["centroid"]) * 2.2)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).mean()
        band = float(np.median(rads)) if rads else seg * 1.4
        band = max(band, seg * 0.9)
        # project verts on polyline
        best_d = np.full(len(co_r), 1e9)
        best_seg = np.zeros(len(co_r), int)
        best_t = np.zeros(len(co_r))
        for i in range(len(pts) - 1):
            a, bb = pts[i], pts[i + 1]
            ab = bb - a
            L2 = float(ab @ ab)
            t = np.clip(((co_r - a) @ ab) / max(L2, 1e-12), 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(co_r - proj, axis=1)
            m = d < best_d
            best_d[m] = d[m]
            best_seg[m] = i
            best_t[m] = t[m]
        tt = np.clip(best_d / band, 0.0, 1.0)
        fall = 1.0 - (tt * tt * (3 - 2 * tt))
        sel = fall > 1e-3
        for i in range(len(pts) - 1):
            m = sel & (best_seg == i)
            if not m.any():
                continue
            wa = fall[m] * (1.0 - best_t[m])
            wb = fall[m] * best_t[m]
            for nm, wv in ((names[i], wa), (names[i + 1], wb)):
                arr = fields.get(nm)
                if arr is None:
                    arr = np.zeros(len(co_r))
                    fields[nm] = arr
                idx = np.where(m)[0]
                arr[idx] = np.maximum(arr[idx], wv)

    # blob groups for everything else Storm painted on the head
    for n, st in stats.items():
        if n in chain_members or n not in have:
            continue
        if n in ("DEF-Jaw", "DEF-Head", "DEF-Neck"):
            continue        # analytic carves / body bones stay
        # structural masters cover the WHOLE face region in Storm's paint:
        # as blobs they eat the per-vertex cap and starve the micro bones.
        # Their function survives through parenting + the lattices.
        if n.startswith(("STR-", "TGT-", "DEF-Facemask")):
            continue
        cen = rbf.p(st["centroid"])
        rad = st["mean_r"] * rbf.jac_scale(st["centroid"]) * 2.0
        blob(n, cen, float(rad))

    # single proportional carve
    me = body.data
    face_names = list(fields.keys())
    W = np.stack([fields[n] for n in face_names], axis=1)
    tot = W.sum(axis=1)
    scale = np.where(tot > cap, cap / np.maximum(tot, 1e-9), 1.0)
    W = W * scale[:, None]
    tot = W.sum(axis=1)

    for n in face_names:
        g = body.vertex_groups.get(n)
        if g is not None:
            body.vertex_groups.remove(g)
    groups = {n: body.vertex_groups.new(name=n) for n in face_names}
    gmap = {g.index: g for g in body.vertex_groups}
    fidx = {groups[n].index for n in face_names}
    touched = np.where(tot > 1e-3)[0]
    for i in touched:
        v = me.vertices[int(i)]
        base = [(g.group, g.weight) for g in v.groups
                if g.group not in fidx and g.weight > 0.0]
        tb = sum(x[1] for x in base)
        if tb <= 1e-6:
            continue
        f = float(tot[i])
        for gi, gw in base:
            gmap[gi].add([int(i)], gw * (1.0 - f), 'REPLACE')
        for k, n in enumerate(face_names):
            wv = float(W[i, k])
            if wv > 1e-4:
                groups[n].add([int(i)], wv * tb, 'REPLACE')
    return len(touched)


# --------------------------------------------------- eyes / teeth / tongue
def _bind_rigid(ob, rig, group):
    for m in list(ob.modifiers):
        if m.type == 'ARMATURE':
            ob.modifiers.remove(m)
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    g = ob.vertex_groups.new(name=group)
    g.add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
    if ob.parent != rig:
        mw = ob.matrix_world.copy()
        ob.parent = rig
        ob.matrix_world = mw
    m = ob.modifiers.new("Armature", 'ARMATURE')
    m.object = rig


def bind_parts(face, props, context, rig, body, rbf):
    rep = []
    # ---- eyes -> FK-Eye (storm's eye deformer) ----
    try:
        eye_meshes = face._eye_meshes(props, body)
    except Exception:
        eye_meshes = []
    ipd = 1.0
    for ob in eye_meshes:
        c = utils.read_rest_coords(ob)
        spans = (float(c[:, 0].max()) > 0.0 > float(c[:, 0].min())
                 and len(eye_meshes) == 1)
        for m in list(ob.modifiers):
            if m.type == 'ARMATURE':
                ob.modifiers.remove(m)
        for g in list(ob.vertex_groups):
            if g.name.startswith(("DEF-eye", "FK-Eye")):
                ob.vertex_groups.remove(g)
        if spans:
            xm = float(c[:, 0].mean())
            li = [i for i in range(len(c)) if c[i, 0] > xm]
            ri = [i for i in range(len(c)) if c[i, 0] <= xm]
            ob.vertex_groups.new(name="FK-Eye.L").add(li, 1.0, 'REPLACE')
            ob.vertex_groups.new(name="FK-Eye.R").add(ri, 1.0, 'REPLACE')
        else:
            s = ".L" if c.mean(axis=0)[0] >= 0 else ".R"
            g = ob.vertex_groups.new(name="FK-Eye" + s)
            g.add(list(range(len(c))), 1.0, 'REPLACE')
        if ob.parent != rig:
            mw = ob.matrix_world.copy()
            ob.parent = rig
            ob.matrix_world = mw
        ob.modifiers.new("Armature", 'ARMATURE').object = rig
    if eye_meshes:
        rep.append("eyes->FK-Eye x%d" % len(eye_meshes))

    # ---- eye AIM: without this FK-Eye never rotates, so the eyeball is
    # frozen when the animator moves MSTR-Eye_target. Storm's FK-Eye points
    # DOWN, so pick the local axis that actually aims at TGT-Eye instead of
    # assuming +Y, then DAMPED_TRACK it. ----
    _AX = (('TRACK_X', Vector((1, 0, 0))), ('TRACK_Y', Vector((0, 1, 0))),
           ('TRACK_Z', Vector((0, 0, 1))),
           ('TRACK_NEGATIVE_X', Vector((-1, 0, 0))),
           ('TRACK_NEGATIVE_Y', Vector((0, -1, 0))),
           ('TRACK_NEGATIVE_Z', Vector((0, 0, -1))))
    for s in (".L", ".R"):
        fk = rig.pose.bones.get("FK-Eye" + s)
        tg = rig.pose.bones.get("TGT-Eye" + s)
        if fk is None or tg is None:
            continue
        M = rig.matrix_world @ fk.matrix
        want = ((rig.matrix_world @ tg.head) - M.translation).normalized()
        ax = max(_AX, key=lambda a: (M.to_3x3() @ a[1]).normalized().dot(want))
        for c in list(fk.constraints):
            if c.name == "SR Eye Aim":
                fk.constraints.remove(c)
        con = fk.constraints.new('DAMPED_TRACK')
        con.name = "SR Eye Aim"
        con.target = rig
        con.subtarget = "TGT-Eye" + s
        con.track_axis = ax[0]
    if eye_meshes:
        rep.append("eye-aim")

    # ---- teeth: rigid to the master teeth bones ----
    tongue_ob = getattr(props, "skin_tongue", None)
    up_ob = getattr(props, "skin_teeth_up", None)
    lo_ob = getattr(props, "skin_teeth_low", None)
    combined = (up_ob is not None and (up_ob is lo_ob or up_ob is tongue_ob)) \
        or (lo_ob is not None and lo_ob is tongue_ob)
    if combined:
        host = up_ob or lo_ob or tongue_ob
        # no SR_* split registered -> split by mesh ISLANDS automatically
        if not any(_vg_verts(host, g) for g in
                   ("SR_teeth_up", "SR_teeth_low", "SR_tongue")):
            r2 = _split_combined(host, rig)
            rep.extend(r2)
            combined = 'done' if r2 else combined
    bound_hosts = set()
    if combined == 'done':
        return rep
    for attr, vgn, bone in (("skin_teeth_up", "SR_teeth_up", "MSTR-Teeth_upp"),
                            ("skin_teeth_low", "SR_teeth_low",
                             "MSTR-Teeth_low")):
        if bone not in rig.data.bones:
            continue
        pts, host, idx = _obj_or_vg_pts(props, body, attr, vgn)
        if host is None:
            continue
        if idx is None and host is not body and not combined:
            _bind_rigid(host, rig, bone)      # whole separate teeth mesh
            bound_hosts.add(host.name)
            rep.append("%s->%s" % (host.name, bone))
        elif idx:
            _strip_body_verts_to(host, idx, bone)
            if host is not body and host.name not in bound_hosts:
                _ensure_armature(host, rig)
                bound_hosts.add(host.name)
            rep.append("%s(%s)->%s" % (host.name, vgn, bone))

    # ---- tongue: distribute along DEF-Tongue1..5 ----
    t_names = ["DEF-Tongue%d" % i for i in range(1, 6)
               if "DEF-Tongue%d" % i in rig.data.bones]
    pts, host, idx = _obj_or_vg_pts(props, body, "skin_tongue", "SR_tongue")
    if pts is not None and idx is None and host is not None:
        idx = list(range(len(pts)))
        if host is not body and not combined:
            for m in list(host.modifiers):
                if m.type == 'ARMATURE':
                    host.modifiers.remove(m)
            for g in list(host.vertex_groups):
                host.vertex_groups.remove(g)
    if pts is not None and host is not None and len(t_names) >= 2 and len(pts):
        heads = np.array([rig.data.bones[n].head_local for n in t_names])
        Mw = np.array(rig.matrix_world)
        heads_w = heads @ Mw[:3, :3].T + Mw[:3, 3]
        param = heads_w[:, 1]                      # along Y (back->front)
        order = np.argsort(param)
        t_sorted = [t_names[i] for i in order]
        p_sorted = param[order]
        ys = np.asarray(pts)[:, 1]
        groups = {n: host.vertex_groups.get(n) or
                  host.vertex_groups.new(name=n) for n in t_sorted}
        for k, vi in enumerate(idx):
            y = ys[k]
            j = int(np.searchsorted(p_sorted, y))
            if j <= 0:
                ws = {t_sorted[0]: 1.0}
            elif j >= len(t_sorted):
                ws = {t_sorted[-1]: 1.0}
            else:
                a, b = p_sorted[j - 1], p_sorted[j]
                t = (y - a) / max(b - a, 1e-9)
                ws = {t_sorted[j - 1]: 1.0 - t, t_sorted[j]: t}
            for n, w in ws.items():
                if w > 1e-3:
                    groups[n].add([int(vi)], float(w), 'REPLACE')
        if host is not body:
            _ensure_armature(host, rig)
        rep.append("tongue->%d bones" % len(t_names))
    return rep


def _split_combined(host, rig):
    """Combined teeth_and_tongue mesh: split by connected ISLANDS.
    The island with the largest Y-extent = tongue (it reaches back into
    the mouth); the rest are teeth, upper/lower by height."""
    me = host.data
    n = len(me.vertices)
    if not n:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for e in me.edges:
        union(e.vertices[0], e.vertices[1])
    from collections import defaultdict
    isl = defaultdict(list)
    for i in range(n):
        isl[find(i)].append(i)
    islands = list(isl.values())
    if len(islands) < 2:
        return []
    co = utils.read_rest_coords(host)
    infos = []
    for idx in islands:
        pts = co[idx]
        infos.append({"idx": idx,
                      "yext": float(pts[:, 1].max() - pts[:, 1].min()),
                      "zc": float(pts[:, 2].mean()),
                      "n": len(idx)})
    tongue = max(infos, key=lambda d: d["yext"] * d["n"])
    teeth = [d for d in infos if d is not tongue]
    z_split = sum(d["zc"] * d["n"] for d in teeth) / \
        max(sum(d["n"] for d in teeth), 1)
    upp, low = [], []
    for d in teeth:
        (upp if d["zc"] >= z_split else low).extend(d["idx"])

    for m in list(host.modifiers):
        if m.type == 'ARMATURE':
            host.modifiers.remove(m)
    for g in list(host.vertex_groups):
        if not g.name.startswith(("SR_", "MSK")):
            host.vertex_groups.remove(g)
    rep = []
    for bone, idx in (("MSTR-Teeth_upp", upp), ("MSTR-Teeth_low", low)):
        if idx and bone in rig.data.bones:
            g = host.vertex_groups.new(name=bone)
            g.add([int(i) for i in idx], 1.0, 'REPLACE')
            rep.append("island->%s x%d" % (bone, len(idx)))
    t_names = ["DEF-Tongue%d" % i for i in range(1, 6)
               if "DEF-Tongue%d" % i in rig.data.bones]
    tidx = tongue["idx"]
    if tidx and len(t_names) >= 2:
        heads = np.array([rig.data.bones[nm].head_local for nm in t_names])
        Mw = np.array(rig.matrix_world)
        heads_w = heads @ Mw[:3, :3].T + Mw[:3, 3]
        param = heads_w[:, 1]
        order = np.argsort(param)
        t_sorted = [t_names[i] for i in order]
        p_sorted = param[order]
        groups = {nm: host.vertex_groups.new(name=nm) for nm in t_sorted}
        ys = co[tidx][:, 1]
        for k, vi in enumerate(tidx):
            y = ys[k]
            j = int(np.searchsorted(p_sorted, y))
            if j <= 0:
                ws = {t_sorted[0]: 1.0}
            elif j >= len(t_sorted):
                ws = {t_sorted[-1]: 1.0}
            else:
                a, b = p_sorted[j - 1], p_sorted[j]
                t = (y - a) / max(b - a, 1e-9)
                ws = {t_sorted[j - 1]: 1.0 - t, t_sorted[j]: t}
            for nm, w in ws.items():
                if w > 1e-3:
                    groups[nm].add([int(vi)], float(w), 'REPLACE')
        rep.append("island->tongue x%d" % len(tidx))
    _ensure_armature(host, rig)
    return rep


def _ensure_armature(ob, rig):
    if not any(m.type == 'ARMATURE' and m.object == rig
               for m in ob.modifiers):
        if ob.parent != rig:
            mw = ob.matrix_world.copy()
            ob.parent = rig
            ob.matrix_world = mw
        ob.modifiers.new("Armature", 'ARMATURE').object = rig


def _strip_body_verts_to(body, idx, bone):
    """Give the listed body verts 100% to `bone` (rigid parts like teeth
    embedded in the body mesh)."""
    g = body.vertex_groups.get(bone)
    if g is None:
        g = body.vertex_groups.new(name=bone)
    gmap = {gg.index: gg for gg in body.vertex_groups}
    gi = g.index
    for i in idx:
        v = body.data.vertices[int(i)]
        for gr in list(v.groups):
            if gr.group != gi and gr.weight > 0.0:
                gmap[gr.group].remove([int(i)])
        g.add([int(i)], 1.0, 'REPLACE')


def _brow_fallback(rig):
    """Storm animates brows through head-mesh SHAPE KEYS (topology-bound,
    not transferable). Bone fallback: every DEF-Brow_local follows its two
    nearest brow CTLs (delta, local space) so the painted brow weights
    respond to CTL-Brow_in/mid/out + Brow_all."""
    ctls = []
    for s in (".L", ".R"):
        for p in ("in", "mid", "out"):
            nm = "CTL-Brow_%s%s" % (p, s)
            b = rig.data.bones.get(nm)
            if b is not None:
                ctls.append((nm, np.array(b.head_local)))
    n = 0
    for pb in rig.pose.bones:
        if not pb.name.startswith("DEF-Brow_local"):
            continue
        side = pb.name[-2:]
        cands = [(nm, h) for nm, h in ctls if nm.endswith(side)]
        if not cands:
            continue
        h = np.array(rig.data.bones[pb.name].head_local)
        cands.sort(key=lambda c: np.linalg.norm(c[1] - h))
        (n1, h1), (n2, h2) = cands[0], cands[1] if len(cands) > 1 else cands[0]
        d1 = float(np.linalg.norm(h1 - h))
        d2 = float(np.linalg.norm(h2 - h))
        w1 = d2 / max(d1 + d2, 1e-9) if n1 != n2 else 1.0
        for c in list(pb.constraints):
            if c.name.startswith("SR Brow Follow"):
                pb.constraints.remove(c)
        for nm2, inf in ((n1, w1), (n2, 1.0 - w1)):
            if inf < 0.05:
                continue
            con = pb.constraints.new('COPY_LOCATION')
            con.name = "SR Brow Follow " + nm2
            con.target = rig
            con.subtarget = nm2
            con.use_offset = True
            con.target_space = 'LOCAL_OWNER_ORIENT'
            con.owner_space = 'LOCAL'
            con.influence = inf
        n += 1
    return n


def _widget_proportions(rig, spec, gp, ipd):
    """Saeed: widget sizes must fit the character. Harold's features sit
    close together in a WIDE face, so ipd-scaled widgets look lost. Scale
    the OUTER controls by the face-width ratio (ours vs Storm's)."""
    A = spec["anchors"]
    try:
        storm_w = abs(A["ear.L"][0] - A["ear.R"][0])
        storm_ipd = spec["meta"]["ipd"]
        our_w = abs(float(gp["ear_low.L"][0]) - float(gp["ear_low.R"][0]))
        f = (our_w / max(ipd, 1e-9)) / (storm_w / storm_ipd)
    except Exception:
        return 0
    f = max(0.8, min(2.5, f))
    outer = ("CTL-Brow_all", "CTL-Cheek_all", "MSTR-Face_upp",
             "MSTR-Face_low", "MSTR-Mouth", "CTL-Jaw", "DEF-Jaw",
             "MSTR-Teeth", "MSTR-Tongue")
    n = 0
    for pb in rig.pose.bones:
        if pb.custom_shape is None:
            continue
        if pb.name.startswith(outer):
            s = pb.custom_shape_scale_xyz
            pb.custom_shape_scale_xyz = (s[0] * f, s[1] * f, s[2] * f)
            n += 1
    return n


# ------------------------------------------------------------------ main
def _add_jaw_ctrl(rig, context):
    """Storm's spec ships DEF-Jaw but NO animator control above it, so the
    mouth cannot be opened. Add CTL-Jaw (coincident with DEF-Jaw) that
    DEF-Jaw copies in LOCAL space, give it the jaw widget and expose it."""
    if "DEF-Jaw" not in rig.data.bones:
        return
    if context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    dj = eb["DEF-Jaw"]
    if "CTL-Jaw" in eb:
        eb.remove(eb["CTL-Jaw"])
    cj = eb.new("CTL-Jaw")
    cj.head = dj.head.copy()
    cj.tail = dj.tail.copy()
    cj.roll = dj.roll
    cj.parent = dj.parent
    cj.use_deform = False
    bpy.ops.object.mode_set(mode='POSE')
    djp = rig.pose.bones["DEF-Jaw"]
    for c in list(djp.constraints):
        if c.name == "SR Jaw Ctrl":
            djp.constraints.remove(c)
    con = djp.constraints.new('COPY_ROTATION')
    con.name = "SR Jaw Ctrl"
    con.target = rig
    con.subtarget = "CTL-Jaw"
    con.target_space = 'LOCAL'
    con.owner_space = 'LOCAL'
    try:
        _fw.assign(rig, "CTL-Jaw", "WGT-Jaw", 1.5, "THEME04")
    except Exception:
        pass
    coll = (rig.data.collections_all.get("Mouth global")
            or rig.data.collections_all.get("Jawline"))
    cb = rig.data.bones.get("CTL-Jaw")
    if coll and cb:
        coll.assign(cb)


def _fix_region_rolls(rig, prefixes, updir):
    """Re-align the roll of every bone whose name starts with one of
    `prefixes` so its local Z best matches world `updir`. Keeps the bone
    HEAD/TAIL (position + direction) exactly; only the roll spins. Done in
    Edit Mode before constraints so the rest state is consistent."""
    import bpy as _bpy
    if _bpy.context.mode != 'OBJECT':
        _bpy.ops.object.mode_set(mode='OBJECT')
    _bpy.context.view_layer.objects.active = rig
    _bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    n = 0
    for b in eb:
        if b.name.startswith(tuple(prefixes)):
            try:
                b.align_roll(updir)
                n += 1
            except Exception:
                pass
    _bpy.ops.object.mode_set(mode='OBJECT')
    return n


def _tidy_brow_bones(rig):
    """Make the brow DEF bones read as one clean professional row.

    The DEF-Brow_local bones are driven purely by COPY_LOCATION (translation
    only - no rotation), so their DIRECTION and ROLL affect NOTHING about the
    deformation; they are cosmetic. The RBF leaves some pointing vertically
    and some horizontally = the 'unprofessional' scatter Saeed saw. Point
    every brow DEF bone horizontally along its side (+X on .L, -X on .R),
    keep its head + length, and roll Z up. Deformation is unchanged."""
    import bpy as _bpy
    from mathutils import Vector as _V
    if _bpy.context.mode != 'OBJECT':
        _bpy.ops.object.mode_set(mode='OBJECT')
    _bpy.context.view_layer.objects.active = rig
    _bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones
    n = 0
    for b in eb:
        if not b.name.startswith("DEF-Brow_local"):
            continue
        side = 1.0 if b.name.endswith(".L") else -1.0
        L = max(b.length, 1e-3)
        head = b.head.copy()
        b.tail = head + _V((side * L, 0.0, 0.0))   # horizontal, outward
        try:
            b.align_roll(_V((0.0, 0.0, 1.0)))        # Z up
        except Exception:
            pass
        n += 1
    _bpy.ops.object.mode_set(mode='OBJECT')
    return n


def build_full(face, props, context):
    """The whole Storm face on our character. `face` = the face module
    (avoids a circular import)."""
    spec = _load("storm_face_spec.json")
    H = _load("storm_face_helpers.json")

    body = getattr(props, "target_mesh", None) or context.active_object
    rig = face._target_rig()
    if rig is None:
        raise RuntimeError("Generate the body rig first (Match to Rig)")
    gp = face.grid_points()
    if not gp:
        raise RuntimeError("No face grid - run the face wizard first")
    head_parent = face._head_parent_name(rig) or "head"

    # widgets from the spec that we don't ship yet
    for wname, wd in spec.get("widgets", {}).items():
        if wname not in _fw.WIDGETS:
            _fw.WIDGETS[wname] = (wd["v"], wd["e"])

    src, dst, ipd = _build_anchor_pairs(face, props, body, gp, rig)
    rbf = _RBF(src, dst)

    prev_active = context.view_layer.objects.active
    prev_mode = context.mode
    if prev_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    context.view_layer.objects.active = rig
    rig.hide_set(False)
    rig.hide_viewport = False

    n_rm = _clean_previous(face, props, context, rig, body, spec)
    order = _build_bones(rig, spec, rbf, head_parent)
    rig.data["sr_storm_face_bones"] = list(order)
    # consistent roll for the brow chain: the RBF's directional derivative
    # distorts Storm's z-axis unevenly, leaving some brow bones rolled 90
    # (Z sideways/down) - the "unprofessional roll" Saeed saw. Re-align the
    # whole brow region to world +Z so they read as one clean row.
    _fix_region_rolls(rig, ("DEF-Brow_local", "CTL-Brow_", "P-Brow",
                            "STR-Brow", "DEF-Furrow", "CTL-brow_normal"),
                      Vector((0.0, 0.0, 1.0)))

    made_objs = {"__body__": body}
    _make_ribbons(rig, H, rbf, made_objs)
    _make_empties(rig, H, made_objs)
    _apply_pose(rig, spec, made_objs, head_parent)
    scale = rbf.jac_scale(spec["anchors"]["eye.L"])
    n_con = _apply_constraints(rig, spec, made_objs, head_parent, scale)
    # drivers BEFORE the rest-reconcile: they are part of the rest state
    n_drv = _apply_drivers(rig, spec, made_objs, head_parent)
    n_skd = _sk_drivers(rig, H, made_objs, spec["bones"], head_parent)
    _fix_stretch_rest(context, rig, order)
    n_rec = _rest_reconcile(context, rig, order, max_snap=0.7 * ipd)
    # lattices + their hooks bind AFTER the final rest is settled
    _make_lattices(rig, body, H, rbf, made_objs)
    n_lat = _body_lattice_stack(rig, body, H, rbf, made_objs)
    n_w = face_weights(rig, body, spec, H, rbf)
    rep = bind_parts(face, props, context, rig, body, rbf)
    _brow_fallback(rig)
    _widget_proportions(rig, spec, gp, ipd)
    _rest_reconcile(context, rig, order, iters=2, max_snap=0.7 * ipd)
    _add_jaw_ctrl(rig, context)
    # brow DEF bones point cleanly along the brow (cosmetic only - they are
    # COPY_LOCATION so deformation is unchanged). Done LAST so the rest
    # reconcile above cannot revert it.
    _tidy_brow_bones(rig)
    # professional nested bone-collection layout (Blender-Studio style):
    # animator sees Body + Face controls; ALL mechanism under hidden Rigging
    try:
        from . import organize as _org
        _org.organize(rig)
    except Exception:
        pass

    context.view_layer.update()
    if prev_active is not None:
        context.view_layer.objects.active = prev_active
    return {"bones": len(order), "removed": n_rm, "constraints": n_con,
            "drivers": n_drv, "sk_drivers": n_skd, "lattice_mods": n_lat,
            "weight_verts": n_w, "parts": rep, "anchors": len(src),
            "reconciled": n_rec, "scale": round(float(scale), 4)}
