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


def _edit_rig(rig):
    """Robustly enter Edit Mode on `rig` and return True on success.

    Blender refuses ``mode_set`` when the active object is HIDDEN:
    ``context.active_object`` becomes None even after assigning
    ``view_layer.objects.active`` -> "Context missing active object".
    So: leave the current mode, UNHIDE the rig (eye + monitor), select it,
    make it active, then enter Edit Mode. Never raises."""
    if rig is None:
        return False
    try:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    try:
        rig.hide_set(False)          # eye icon (view-layer hide)
    except Exception:
        pass                          # rig not in the current view layer
    rig.hide_viewport = False         # monitor icon (global disable)
    try:
        rig.select_set(True)
    except Exception:
        pass
    try:
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
    except Exception:
        return False
    return rig.mode == 'EDIT'


_NO_ACCESS = ("Rig is not accessible - unhide it (eye icon) "
              "or enable its collection in the View Layer.")



def skirt_verts_world(props):
    """World-space vertices of the skirt: separate object, or the registered
    vertex group on the merged character mesh. Returns Nx3 ndarray or None."""
    src = getattr(props, "skirt_source", 'MERGED')
    if src == 'SEPARATE':
        ob = getattr(props, "skirt_object", None)
        if ob is None or ob.type != 'MESH':
            return None
        # REST coords, not evaluated: after Generate the skirt is deformed by the
        # rig (jiggle/blow-up/pose), and reading the evaluated mesh would fit the
        # bones to that momentary shape -> bones jump/break on rebuild. Rest is stable.
        return utils.read_rest_coords(ob)
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


_FRONT_ANG = {
    '-Y': (math.atan2(-1.0, 0.0) + 2.0 * math.pi) % (2.0 * math.pi),
    '+Y': (math.atan2(1.0, 0.0) + 2.0 * math.pi) % (2.0 * math.pi),
    '+X': (math.atan2(0.0, 1.0) + 2.0 * math.pi) % (2.0 * math.pi),
    '-X': (math.atan2(0.0, -1.0) + 2.0 * math.pi) % (2.0 * math.pi),
}


def _boundary_loops(bm):
    """Group a bmesh's boundary edges into ordered vertex loops (each a closed or
    open border ring). Used to understand the skirt's shape (waist / hem / slit)."""
    from collections import defaultdict
    adj = defaultdict(list)
    for e in bm.edges:
        if e.is_boundary:
            a, b = e.verts
            adj[a].append(b); adj[b].append(a)
    loops = []
    visited = set()
    for start in list(adj.keys()):
        if start in visited:
            continue
        loop = []; cur = start; prev = None
        while cur is not None and cur not in visited:
            visited.add(cur); loop.append(cur)
            nxts = [n for n in adj[cur] if n is not prev and n not in visited]
            prev = cur
            cur = nxts[0] if nxts else None
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _ring_wraps(pts, cx, cy, sectors=12, min_cover=0.66):
    """True if the ring of points actually ENCIRCLES the vertical axis at (cx, cy):
    its verts must occupy most angular sectors around the centre. A side slit /
    vertical seam opening / small hole clusters in one narrow angle and must NOT be
    treated as a waist or hem rim (v1.19.145 - the 'Skirt_5' closed-waist bug:
    a slit boundary was taken as the waist, so every column started at the slit's
    median Z in mid-air and the top of the skirt got no bones)."""
    if not pts:
        return False
    occ = set()
    for p in pts:
        a = (math.atan2(p.y - cy, p.x - cx) + 2.0 * math.pi) % (2.0 * math.pi)
        occ.add(int(a / (2.0 * math.pi) * sectors) % sectors)
    return len(occ) >= max(3, int(min_cover * sectors))


def analyze_skirt(ob):
    """Heuristic 'skirt understanding': inspect the mesh topology and classify the
    skirt so the builder can pick the right bone-placement strategy. Returns a dict
    with 'kind' in {TUBE, OPEN, LAYERED, CLOSED, MERGED, MESSY, NONE} plus diagnostics.

      TUBE    - clean quad tube, open at waist + hem  -> edge-flow (best)
      OPEN    - single WRAPPING border (open-front / wrap / one drape) -> edge-flow best-effort
      LAYERED - >2 wrapping borders (tiers / ruffles / lining) -> edge-flow on the primary layer
      CLOSED  - borders exist but none wraps the axis (closed waist/hem, only a
                slit / seam / hole) -> rim-span on Z-bands (edge-flow can't start)
      MERGED  - no borders (skirt welded into the body) -> angular sampling on region
      MESSY   - too few quads (triangulated / n-gons)   -> angular sampling

    Only loops that pass _ring_wraps count toward the kind - a slit is not a rim.
    """
    import bmesh
    info = {"kind": "NONE", "quad_ratio": 0.0, "n_boundary_loops": 0}
    me = getattr(ob, "data", None)
    if me is None or not me.polygons:
        return info
    bm = bmesh.new()
    try:
        bm.from_mesh(me); bm.verts.ensure_lookup_table()
        # analyse in WORLD space: imported meshes (FBX etc.) often carry an
        # object-level rotation (e.g. X=90), so local Z is NOT the world up axis
        # (v1.19.145 - Skirt_5 was rotated 90 deg and every Z test was sideways)
        bm.transform(ob.matrix_world)
        nf = len(bm.faces)
        if nf == 0:
            return info
        quads = sum(1 for f in bm.faces if len(f.verts) == 4)
        info["quad_ratio"] = round(quads / nf, 2)
        loops = _boundary_loops(bm)
        info["n_boundary_loops"] = len(loops)
        # centre for the wrap test: bbox midpoint (density-proof, local space -
        # same space as the loop verts)
        _lv = [v for v in bm.verts if v.link_edges]
        if _lv:
            _xs = [v.co.x for v in _lv]; _ys = [v.co.y for v in _lv]
            _cx = (min(_xs) + max(_xs)) * 0.5
            _cy = (min(_ys) + max(_ys)) * 0.5
        else:
            _cx = _cy = 0.0
        # only loops that actually encircle the axis are rims; slits/holes are not
        wrapping = [L for L in loops
                    if _ring_wraps([v.co for v in L], _cx, _cy)]
        info["n_wrapping_loops"] = len(wrapping)
        if wrapping:
            lz = [sum(v.co.z for v in L) / len(L) for L in wrapping]
            order = sorted(range(len(wrapping)), key=lambda i: -lz[i])
            info["loop_sizes"] = [len(wrapping[i]) for i in order]
            info["loop_z_cm"] = [round(lz[i] * 100, 1) for i in order]
            info["waist_verts"] = len(wrapping[order[0]])
        if info["quad_ratio"] < 0.6:
            info["kind"] = "MESSY"
        elif len(loops) == 0:
            info["kind"] = "MERGED"
        elif len(wrapping) == 0:
            info["kind"] = "CLOSED"        # only slits/seams/holes - no real rims
        elif len(wrapping) == 2:
            info["kind"] = "TUBE"
        elif len(wrapping) == 1:
            info["kind"] = "OPEN"
        else:
            info["kind"] = "LAYERED"
        return info
    except Exception:
        return info
    finally:
        bm.free()


def _rim_rings(ob):
    """(waist_pts, hem_pts) in world space - the topological START (waist rim) and END
    (hem rim) of the skirt. Uses boundary loops when present; for a thick/folded hem
    with no bottom boundary it uses the lowest-Z band. Either may be None."""
    import bmesh
    me = getattr(ob, "data", None)
    if me is None or not me.polygons:
        return None, None
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        # WORLD space: imported meshes often carry an object rotation (FBX X=90),
        # so local Z is not up. After this, v.co IS the world position.
        bm.transform(ob.matrix_world)
        loops = _boundary_loops(bm)
        zc = [v.co.z for v in bm.verts if v.link_edges]
        if not zc:
            return None, None
        zmn, zmx = min(zc), max(zc); h = max(zmx - zmn, 1e-6)
        # a rim must ENCIRCLE the axis - drop slits / seam openings / small holes
        _lv = [v for v in bm.verts if v.link_edges]
        _xs = [v.co.x for v in _lv]; _ys = [v.co.y for v in _lv]
        _cx = (min(_xs) + max(_xs)) * 0.5
        _cy = (min(_ys) + max(_ys)) * 0.5
        loops = [L for L in loops if _ring_wraps([v.co for v in L], _cx, _cy)]

        def band(hi):   # verts in the top (hi=True) or bottom band
            return [v.co.copy() for v in bm.verts if v.link_edges and
                    ((v.co.z > zmx - 0.05 * h) if hi else (v.co.z < zmn + 0.05 * h))]
        if len(loops) >= 2:
            lz = [sum(v.co.z for v in L) / len(L) for L in loops]
            order = sorted(range(len(loops)), key=lambda i: -lz[i])
            waist = [v.co.copy() for v in loops[order[0]]]
            hem = [v.co.copy() for v in loops[order[-1]]]
        elif len(loops) == 1:
            lz = sum(v.co.z for v in loops[0]) / len(loops[0])
            top_loop = [v.co.copy() for v in loops[0]]
            # the single boundary is the waist if it's up high, else the hem
            if lz > (zmn + 0.5 * h):
                waist, hem = top_loop, band(False)
            else:
                waist, hem = band(True), top_loop
        else:
            waist, hem = band(True), band(False)
        waist = waist if waist and len(waist) >= 4 else None
        hem = hem if hem and len(hem) >= 4 else None
        return waist, hem
    except Exception:
        return None, None
    finally:
        bm.free()


def _anchor_ends(grid, waist, hem, cx, cy):
    """Pull each column's START (row 0) to the WAIST rim and END (last row) to the HEM
    rim, keeping the column's own ANGLE fixed. Only the radius + Z are taken from the
    rim (median of the rim verts in the column's angular wedge), so the top/bottom sit
    on the real skirt edges without skewing the bone sideways."""
    n = max(1, len(grid))
    half = math.pi / n

    def rim_pt(ang, tx, ty, ring):
        inw = [v for v in ring
               if abs(((math.atan2(v.y - cy, v.x - cx) - ang + math.pi) % (2.0 * math.pi)) - math.pi) <= half]
        if not inw:
            inw = ring
        rs = sorted(math.hypot(v.x - cx, v.y - cy) for v in inw)
        zs = sorted(v.z for v in inw)
        r = rs[len(rs) // 2]; z = zs[len(zs) // 2]
        return Vector((cx + r * tx, cy + r * ty, z))

    for c, pts in grid:
        if len(pts) < 2:
            continue
        d = pts[0]
        L = math.hypot(d.x - cx, d.y - cy) or 1.0
        ang = math.atan2(d.y - cy, d.x - cx)
        tx, ty = (d.x - cx) / L, (d.y - cy) / L
        if waist:
            pts[0] = rim_pt(ang, tx, ty, waist)
        if hem:
            pts[-1] = rim_pt(ang, tx, ty, hem)
    return grid


def _rim_rz(ring, target, half, cx, cy):
    """Median (radius, z) of the rim vertices inside the angular wedge around
    `target`; falls back to the whole rim if the wedge is empty."""
    inw = [v for v in ring
           if abs(((math.atan2(v.y - cy, v.x - cx) - target + math.pi) % (2.0 * math.pi)) - math.pi) <= half]
    if not inw:
        inw = ring
    rs = sorted(math.hypot(v.x - cx, v.y - cy) for v in inw)
    zs = sorted(v.z for v in inw)
    return rs[len(rs) // 2], zs[len(zs) // 2]


def _skirt_grid_between(co, cols, rows, waist, hem, cx, cy, front):
    """Build columns that run EXACTLY from the waist rim (start) to the hem rim (end),
    per angle - ignoring any geometry above the waist opening (folded waistbands) or
    below the hem. Row 0 sits on the waist rim, the last row on the hem rim, and the
    middle rows follow the mesh radius at the interpolated heights. This is the correct
    'where the skirt starts / ends' behaviour for thick or capped meshes."""
    co = np.asarray(co, dtype=float)
    ang = (np.arctan2(co[:, 1] - cy, co[:, 0] - cx))
    rad = np.hypot(co[:, 0] - cx, co[:, 1] - cy)
    half = math.pi / cols
    grid = []
    for c in range(cols):
        target = (front + 2.0 * math.pi * c / cols)
        tx, ty = math.cos(target), math.sin(target)
        wr, wz = _rim_rz(waist, target, half, cx, cy)
        hr, hz = _rim_rz(hem, target, half, cx, cy)
        if wz < hz:                       # ensure waist is the TOP end
            wr, wz, hr, hz = hr, hz, wr, wz
        band = max(abs(wz - hz) / max(1, rows) * 0.75, 1e-4)
        dang = np.abs(((ang - target + math.pi) % (2.0 * math.pi)) - math.pi)
        amask = dang <= half
        colpts = []
        for l in range(rows + 1):
            t = l / rows
            z = wz + (hz - wz) * t
            if l == 0:
                r = wr
            elif l == rows:
                r = hr
            else:
                sel = rad[amask & (np.abs(co[:, 2] - z) <= band)]
                r = float(np.median(sel)) if len(sel) else (wr + (hr - wr) * t)
            colpts.append(Vector((cx + r * tx, cy + r * ty, float(z))))
        grid.append((c, colpts))
    return grid if grid else None


def _skirt_grid(co, cols, rows, front_ang=None):
    """Build a [cols][rows+1] grid of world points from the skirt vertices,
    sliced by Z (top->bottom) and by angular sector around the center axis.

    ROBUST to any skirt shape: it rejects stray/outlier vertices (e.g. a rogue
    vert left near the arm), works off a per-Z-slice center so an off-axis or
    asymmetric skirt still reads correctly, and uses the MEDIAN (not the mean)
    of each sector so a single bad vertex can never drag a bone out of place."""
    co = np.asarray(co, dtype=float)
    if len(co) < 6:
        return None
    # --- reject stray vertices: anything whose radius from the axis is a gross
    # outlier relative to its own Z-slice. A skirt widens smoothly, so a vert far
    # beyond its neighbours at the same height is not skirt (modeling leftover,
    # arm bleed, etc.) and must not define a column.
    cx0 = float(np.median(co[:, 0])); cy0 = float(np.median(co[:, 1]))
    zmn = float(co[:, 2].min()); zmx = float(co[:, 2].max())
    if zmx - zmn < 1e-4:
        return None
    rr = np.hypot(co[:, 0] - cx0, co[:, 1] - cy0)
    keep = np.ones(len(co), dtype=bool)
    nsl = max(4, rows * 2)
    for s in range(nsl):
        z0 = zmx + (zmn - zmx) * (s / nsl); z1 = zmx + (zmn - zmx) * ((s + 1) / nsl)
        m = (co[:, 2] <= max(z0, z1)) & (co[:, 2] >= min(z0, z1))
        if m.sum() < 4:
            continue
        thr = np.percentile(rr[m], 90) * 1.8 + 1e-4   # generous: keeps real flare
        keep &= ~(m & (rr > thr))
    if keep.sum() >= 6:
        co = co[keep]
    zmax = float(co[:, 2].max()); zmin = float(co[:, 2].min())
    # CENTRE = bounding-box midpoint, NOT the median: the median is pulled off-axis by
    # dense vertex clusters (thick waistbands, detailed regions), which skews every
    # column angle and leaves the true front/centre empty. bbox mid is density-proof
    # (strays were already rejected above).
    cx = float((co[:, 0].min() + co[:, 0].max()) * 0.5)
    cy = float((co[:, 1].min() + co[:, 1].max()) * 0.5)
    ang = (np.arctan2(co[:, 1] - cy, co[:, 0] - cx) + 2.0 * math.pi) % (2.0 * math.pi)
    rad = np.hypot(co[:, 0] - cx, co[:, 1] - cy)
    front = _FRONT_ANG['-Y'] if front_ang is None else front_ang
    band = (zmax - zmin) / rows * 0.75
    half = math.pi / cols                       # half-sector angular window
    grid = []
    for c in range(cols):
        # Place the column at the EXACT target angle, spaced evenly FROM THE FRONT:
        # c=0 -> front-centre, c=cols/2 -> back-centre, c <-> cols-c are mirrors
        # (equal left/right). We then read only the mesh RADIUS in a wedge around that
        # angle - so dense vertex clusters can't drag the column off its angle, and the
        # true shape (radius per height) is preserved.
        target = (front + 2.0 * math.pi * c / cols) % (2.0 * math.pi)
        tx, ty = math.cos(target), math.sin(target)
        dang = np.abs(((ang - target + math.pi) % (2.0 * math.pi)) - math.pi)
        amask = dang <= half
        colpts = []
        for l in range(rows + 1):
            z = zmax + (zmin - zmax) * (l / rows)
            sel = rad[amask & (np.abs(co[:, 2] - z) <= band)]
            if len(sel) == 0:
                sel = rad[amask]                             # any height, this wedge
                if len(sel) == 0:
                    sel = rad[np.abs(co[:, 2] - z) <= band]  # any angle, this height
                    if len(sel) == 0:
                        sel = rad
            r = float(np.median(sel))
            colpts.append(Vector((cx + r * tx, cy + r * ty, float(z))))
        grid.append((c, colpts))
    return grid if grid else None


def _skirt_grid_topo(ob, cols, rows, symmetric=True, front_ang=None):
    """EDGE-FLOW grid: follow the mesh's REAL vertical edge loops so the bones run
    exactly along the topology (best for a clean quad skirt). Returns the same
    [(col, [Vector]*(rows+1))] format as _skirt_grid, or None if the mesh is not a
    clean open quad tube (the caller then falls back to angular sampling).

    symmetric=True places the columns at even angles measured FROM THE FRONT, so there
    is always a front-centre and a back-centre column and the left/right counts match
    (i <-> cols-i are mirror positions). Each column still samples the REAL loop nearest
    its target angle, so an intentionally asymmetric skirt keeps its true shape."""
    import bmesh
    me = getattr(ob, "data", None)
    if me is None or not me.polygons:
        return None
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        # WORLD space (imported meshes are often rotated at object level)
        bm.transform(ob.matrix_world)
        if not bm.faces:
            return None
        quads = sum(1 for f in bm.faces if len(f.verts) == 4)
        if quads < 0.8 * len(bm.faces):          # not a clean quad grid
            return None
        bnd = set()
        for e in bm.edges:
            if e.is_boundary:
                bnd.add(e.verts[0]); bnd.add(e.verts[1])
        if len(bnd) < 6:
            return None                          # closed/no borders -> can't find waist
        zs = [v.co.z for v in bnd]
        zmid = (max(zs) + min(zs)) * 0.5
        topv = [v for v in bnd if v.co.z > zmid]  # waist loop
        if len(topv) < 4:
            return None
        # bbox centre (density-proof), not median which dense clusters pull off-axis
        _xs = [v.co.x for v in bm.verts if v.link_edges]
        _ys = [v.co.y for v in bm.verts if v.link_edges]
        cx = (min(_xs) + max(_xs)) * 0.5
        cy = (min(_ys) + max(_ys)) * 0.5
        front = _FRONT_ANG['-Y'] if front_ang is None else front_ang

        def rel_ang(v):
            a = (math.atan2(v.co.y - cy, v.co.x - cx) + 2.0 * math.pi) % (2.0 * math.pi)
            return (a - front + 2.0 * math.pi) % (2.0 * math.pi)

        topv.sort(key=rel_ang)                   # column 0 at FRONT, going around
        ntop = len(topv)

        def walk(v0):
            """Walk the vertical edge loop from a waist vert down to the hem."""
            chain = [v0]
            e_in = min(v0.link_edges, key=lambda e: e.other_vert(v0).co.z)
            cur = e_in.other_vert(v0); chain.append(cur)
            for _ in range(500):
                fin = set(e_in.link_faces)
                cont = [e for e in cur.link_edges
                        if e is not e_in and not (set(e.link_faces) & fin)
                        and e.other_vert(cur).co.z < cur.co.z + 1e-4]
                if not cont:
                    break
                e_in = min(cont, key=lambda e: e.other_vert(cur).co.z)
                cur = e_in.other_vert(cur); chain.append(cur)
                if cur in bnd and cur.co.z < zmid:
                    break
            return chain

        ncols = max(4, int(cols))
        picks, seen = [], set()
        if symmetric:
            # target angles evenly from the FRONT: i=0 front-centre, i=ncols/2 back-centre,
            # i <-> ncols-i are mirror positions -> equal left/right. Pick the real loop
            # nearest each target (keeps edge-flow + the true, possibly asymmetric, shape).
            rels = [rel_ang(v) for v in topv]
            for i in range(ncols):
                target = (2.0 * math.pi * i / ncols) % (2.0 * math.pi)
                best, bd = -1, 1e18
                for j, a in enumerate(rels):
                    if j in seen:
                        continue
                    d = abs((a - target + math.pi) % (2.0 * math.pi) - math.pi)
                    if d < bd:
                        bd, best = d, j
                if best >= 0:
                    seen.add(best); picks.append(best)
        else:
            for i in range(ncols):
                ti = int(round(i * ntop / ncols)) % ntop
                if ti not in seen:               # can't exceed the real loop count
                    seen.add(ti); picks.append(ti)
        # full mesh Z height (bm already in world space) - a good loop-walk must
        # descend most of it
        zc = [v.co.z for v in bm.verts if v.link_edges]
        mesh_h = (max(zc) - min(zc)) if zc else 0.0
        grid = []
        good = 0
        for ci, ti in enumerate(picks):
            chain = walk(topv[ti])
            if len(chain) < 2:
                continue
            wpts = [v.co.copy() for v in chain]   # top -> hem, world space
            span = wpts[0].z - wpts[-1].z
            if mesh_h > 1e-6 and span >= 0.5 * mesh_h:
                good += 1                          # this loop really reached the hem
            m = len(wpts)
            out = []
            for l in range(rows + 1):             # resample evenly to rows+1 points
                x = (l / rows) * (m - 1)
                i0 = int(math.floor(x)); i1 = min(i0 + 1, m - 1)
                out.append(wpts[i0].lerp(wpts[i1], x - i0))
            grid.append((ci, out))
        # SELF-CHECK: if the edge-loop walk didn't cleanly descend waist->hem for most
        # columns (e.g. a solidified / double-sided / non-tube mesh), bail out so the
        # caller falls back to the robust angular sampler instead of shipping bad bones.
        if len(grid) < 3 or good < 0.6 * len(grid):
            return None
        return grid
    except Exception:
        return None
    finally:
        bm.free()


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


def _symmetrize_grid(grid, cx):
    """Force the columns to be perfectly mirror-symmetric about the X=cx plane
    (character's sagittal plane): each column becomes the average of itself and the
    MIRROR of its opposite-side partner. Columns that sit on the centre line (front /
    back) are pulled exactly onto x=cx. Guarantees left == right for clean X-mirror
    posing, symmetric weights and balanced deformation - even if the mesh was modelled
    slightly off-centre."""
    n = len(grid)
    if n < 2:
        return grid

    def mir(v):
        return Vector((2.0 * cx - v.x, v.y, v.z))

    tops = [pts[0] for _c, pts in grid]
    # pair each column with the one whose top is nearest to its mirrored top
    partner = []
    for i in range(n):
        mi = mir(tops[i])
        best, bd = i, 1e18
        for j in range(n):
            d = (tops[j] - mi).length
            if d < bd:
                bd, best = d, j
        partner.append(best)
    out = []
    for i, (c, pts) in enumerate(grid):
        ppts = grid[partner[i]][1]
        m = min(len(pts), len(ppts))
        out.append((c, [(pts[k] + mir(ppts[k])) * 0.5 for k in range(m)]))
    return out


def _emit_chains(mo, grid, rows):
    """Create one tentacle chain per column from the grid points, parent to
    the hips, roll like the thigh, and tag as Rigify simple_tentacle."""
    # ---- create the bones in edit mode ----
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    # the metarig may be HIDDEN (Generate hides it) - a hidden object can't be
    # made active for Edit Mode -> "Context missing active object". Unhide first.
    try:
        mo.hide_set(False)
    except Exception:
        pass
    mo.hide_viewport = False
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = mo
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
    # ---- understand the skirt, then choose the placement strategy ----
    kind = "MESSY"; method = "angular"
    ob = getattr(props, "skirt_object", None) if getattr(props, "skirt_source", 'MERGED') == 'SEPARATE' else None
    if ob is not None and ob.type == 'MESH':
        kind = analyze_skirt(ob).get("kind", "MESSY")
    _sym = bool(getattr(props, "skirt_symmetric", True))
    _fa = _FRONT_ANG.get(getattr(props, "skirt_front_axis", '-Y'), _FRONT_ANG['-Y'])
    # detect the skirt's real START (waist rim) and END (hem rim) from topology
    waist_r, hem_r = (_rim_rings(ob) if ob is not None else (None, None))
    _cx = float((co[:, 0].min() + co[:, 0].max()) * 0.5)
    _cy = float((co[:, 1].min() + co[:, 1].max()) * 0.5)
    grid = None
    # 1) clean topology -> edge-flow (walks the real waist->hem loops)
    if ob is not None and kind in ("TUBE", "OPEN", "LAYERED"):
        grid = _skirt_grid_topo(ob, cols, rows, symmetric=_sym, front_ang=_fa)
        if grid:
            method = "edge-flow"
    # 2) else build the columns BETWEEN the detected rims (correct start/end, ignores
    #    folded waistbands above the waist and geometry below the hem)
    if not grid and waist_r and hem_r:
        grid = _skirt_grid_between(co, cols, rows, waist_r, hem_r, _cx, _cy, _fa)
        if grid:
            method = "rim-span"
    # 3) last resort: plain angular over the full Z extent
    if not grid:
        grid = _skirt_grid(co, cols, rows, front_ang=_fa)
        method = "angular"
    if not grid:
        return None, "Could not analyse the skirt geometry. Check the selection."

    _emit_chains(mo, grid, rows)
    # record what we detected so the UI / operator can tell the user
    try:
        mo["sr_skirt_kind"] = kind
        mo["sr_skirt_method"] = method
        mo["sr_skirt_cols_built"] = len(grid)
    except Exception:
        pass
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
        return o is not None and o.type == 'ARMATURE' and (o.get("sk_kilt") or o.get("sk_jiggle") or o.get("sk_follow") or o.get("sk_antipen") or o.get("sk_chest_jiggle"))
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
    if not _edit_rig(rig):
        return -1
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


def _skirt_surface_world(props):
    """World-space points of the ACTUAL skirt surface, EVALUATED with its
    modifiers (Solidify / Subsurf / etc.), so masters snap to the real cloth.
    Separate skirt -> evaluated mesh; merged -> the SR_Skirt vertex-group verts."""
    sep = getattr(props, "skirt_source", 'MERGED') == 'SEPARATE'
    src = props.skirt_object if sep else props.target_mesh
    if src is None or src.type != 'MESH':
        return []
    mw = src.matrix_world
    if not sep:
        vg = src.vertex_groups.get(VGROUP)
        if vg is None:
            return []
        gi = vg.index
        return [mw @ v.co.copy() for v in src.data.vertices
                if any(g.group == gi and g.weight > 0 for g in v.groups)]
    # separate: evaluate modifiers so Solidify/Subsurf thickness is included
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = src.evaluated_get(dg)
        me = ev.to_mesh()
        pts = [mw @ v.co.copy() for v in me.vertices]
        ev.to_mesh_clear()
        return pts
    except Exception:
        return [mw @ v.co.copy() for v in src.data.vertices]


def remove_skirt_masters(rig):
    """Remove the global + sector master controls and restore each column's top
    bone to its original parent."""
    if rig is None:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    restore = {}
    for pb in rig.pose.bones:
        if "sk_master_origparent" in pb:
            restore[pb.name] = str(pb["sk_master_origparent"])
            del pb["sk_master_origparent"]
    if not _edit_rig(rig):
        return -1
    eb = rig.data.edit_bones
    for nm, pn in restore.items():
        b = eb.get(nm)
        if b is not None:
            b.parent = eb.get(pn) if (pn and eb.get(pn)) else None
            b.use_connect = False
    n = 0
    for b in [bb for bb in eb if bb.name.startswith("skirt_master")]:
        eb.remove(b); n += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    if "sk_masters" in rig:
        del rig["sk_masters"]
    return n


def add_skirt_masters(rig, props):
    """Build a GLOBAL master + N SECTOR masters (front / sides / back ...) around
    the waist. Each column's top bone is re-parented to its nearest sector master
    so the animator can pose whole regions at once. Sits ABOVE the collision
    (re-parents the per-column SKC_dt root, whose drivers are LOCAL rotations), so
    collision keeps working. N = props.skirt_masters (user can increase)."""
    import math
    from mathutils import Vector
    from collections import Counter
    if rig is None or not getattr(props, "skirt_use_masters", True):
        return 0
    cols = _skirt_columns(rig)
    if not cols:
        return 0
    N = max(2, int(getattr(props, "skirt_masters", 4)))
    remove_skirt_masters(rig)
    cols = _skirt_columns(rig)
    rw = rig.matrix_world; rwi = rw.inverted()
    heads = [h for (_r, _hem, h) in cols.values()]
    cx = sum(p.x for p in heads) / len(heads)
    cy = sum(p.y for p in heads) / len(heads)
    cz = sum(p.z for p in heads) / len(heads)
    rad = max(0.02, sum(math.hypot(p.x - cx, p.y - cy) for p in heads) / len(heads))

    def col_top(ci, root):
        dt = "SKC_dt.%02d.00" % ci
        return dt if rig.data.bones.get(dt) else root

    def sector_of(p):
        ang = math.atan2(p.x - cx, -(p.y - cy)) % (2 * math.pi)   # 0 = FRONT (-Y)
        return int(round(ang / (2 * math.pi / N))) % N

    tops = {}
    for ci, (root, _hem, head) in cols.items():
        tb = col_top(ci, root)
        b = rig.data.bones.get(tb)
        tops[ci] = (tb, (b.parent.name if b and b.parent else None), head)
    pc = Counter(par for (_t, par, _h) in tops.values() if par)
    common_parent = pc.most_common(1)[0][0] if pc else None

    if not _edit_rig(rig):
        return -1
    eb = rig.data.edit_bones

    def Lp(v):
        return rwi @ v
    up = 0.12 * rad
    gm = eb.new("skirt_master")
    gm.use_deform = False              # control bone - must NEVER take weights
    gh = Lp(Vector((cx, cy, cz)))
    gm.head = gh; gm.tail = gh + Vector((0, 0, up))
    if common_parent and eb.get(common_parent):
        gm.parent = eb[common_parent]
    gm.use_connect = False
    # place each sector master ON the real skirt SURFACE (front/side/back), at
    # mid height, the bone pointing radially OUTWARD so its ring widget lies flat
    # against the cloth ("stuck on the skirt"). Uses the evaluated mesh surface
    # (with Solidify/Subsurf) when available, else the bone mids as a fallback.
    sec_pts = {s: [] for s in range(N)}
    for ci, (root, hem, head) in cols.items():
        sec_pts[sector_of(head)].append((head + hem) * 0.5)
    allz = ([h.z for (_r, _hem, h) in cols.values()] +
            [hm.z for (_r, hm, _h) in cols.values()])
    zspan = (max(allz) - min(allz)) or rad
    band = 0.22 * zspan
    surf = _skirt_surface_world(props)

    def _az(q):
        return math.atan2(q.x - cx, -(q.y - cy)) % (2 * math.pi)

    secnames = []
    for s in range(N):
        nm = "skirt_master.%02d" % s
        sb = eb.new(nm); secnames.append(nm)
        sb.use_deform = False          # control bone - must NEVER take weights
        sec_ang = s * (2 * math.pi / N)
        # 1) RELIABLE anchor from the skirt BONES (always on the skirt, no strays)
        if sec_pts[s]:
            anchor = sum(sec_pts[s], Vector((0.0, 0.0, 0.0))) / len(sec_pts[s])
        else:
            anchor = Vector((cx + math.sin(sec_ang) * rad,
                             cy - math.cos(sec_ang) * rad, cz))
        # 2) snap to the NEAREST evaluated-surface vertex (reads Solidify/Subsurf
        #    thickness) so the master sits exactly on the real cloth.
        pos = anchor
        bestd = 1e18
        for q in surf:
            d = (q - anchor).length_squared
            if d < bestd:
                bestd = d; pos = q
        radial = Vector((pos.x - cx, pos.y - cy, 0.0))
        if radial.length < 1e-4:
            radial = Vector((math.sin(sec_ang), -math.cos(sec_ang), 0.0))
        radial.normalize()
        sb.head = Lp(pos)
        sb.tail = Lp(pos + radial * (0.20 * rad))   # outward -> widget tangent to cloth
        sb.parent = eb["skirt_master"]; sb.use_connect = False
    for ci, (tb, par, head) in tops.items():
        cb = eb.get(tb)
        if cb is None:
            continue
        cb.parent = eb["skirt_master.%02d" % sector_of(head)]
        cb.use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    for ci, (tb, par, head) in tops.items():
        pbn = rig.pose.bones.get(tb)
        if pbn is not None:
            pbn["sk_master_origparent"] = par or ""
    rig["sk_masters"] = N
    # widgets
    wgt = _ensure_master_widget()
    gpb = rig.pose.bones.get("skirt_master")
    if gpb is not None:
        gpb.custom_shape = wgt
        gpb.use_custom_shape_bone_size = False
        gpb.custom_shape_scale_xyz = (rad * 2.4, rad * 2.4, rad * 2.4)
    swgt = _ensure_sector_widget()
    for nm in secnames:
        spb = rig.pose.bones.get(nm)
        if spb is not None:
            spb.custom_shape = swgt
            spb.use_custom_shape_bone_size = False
            _ssc = min(rad * 0.8, 1.7 * math.pi * rad / N)   # fit the sector, no overlap
            spb.custom_shape_scale_xyz = (_ssc, _ssc, _ssc)
    _organize_skirt_bones(rig)
    return N


def _wgt_collection():
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
    return coll


def _make_wgt(name, edges_fn):
    """(Re)build a widget mesh object in the hidden WGTS collection."""
    import bmesh
    old = bpy.data.objects.get(name)
    if old is not None:
        try:
            bpy.data.objects.remove(old, do_unlink=True)
        except Exception:
            pass
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    edges_fn(bm)
    bm.to_mesh(me); bm.free()
    wgt = bpy.data.objects.new(name, me)
    _wgt_collection().objects.link(wgt)
    return wgt


def _ensure_master_widget():
    """GLOBAL skirt master = an elegant compass dial (double ring + 4 outward
    ticks + centre diamond). Drawn in the XZ plane."""
    name = "WGT-SK_MasterAll"
    w = bpy.data.objects.get(name)
    if w is not None and w.type == 'MESH':
        return w

    def build(bm):
        def ring(r, N=48):
            vs = [bm.verts.new((r * math.cos(2 * math.pi * i / N), 0.0,
                                r * math.sin(2 * math.pi * i / N))) for i in range(N)]
            for i in range(N):
                bm.edges.new((vs[i], vs[(i + 1) % N]))
        ring(1.0); ring(0.84)
        # 4 outward ticks at the cardinal points
        for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            p0 = bm.verts.new((math.cos(a), 0, math.sin(a)))
            p1 = bm.verts.new((1.16 * math.cos(a), 0, 1.16 * math.sin(a)))
            bm.edges.new((p0, p1))
        # small centre diamond
        d = [bm.verts.new((0.16 * math.cos(a), 0, 0.16 * math.sin(a)))
             for a in (0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
        for i in range(4):
            bm.edges.new((d[i], d[(i + 1) % 4]))
    return _make_wgt(name, build)


def _ensure_sector_widget():
    """SECTOR master = a clean rounded-square (squircle) handle that reads nicely
    lying flat on the cloth. Drawn in the XZ plane."""
    name = "WGT-SK_MasterSector"
    w = bpy.data.objects.get(name)
    if w is not None and w.type == 'MESH':
        return w

    def build(bm):
        N = 48; n = 4.0   # superellipse exponent -> rounded square
        vs = []
        for i in range(N):
            t = 2 * math.pi * i / N
            ct = math.cos(t); st = math.sin(t)
            x = math.copysign(abs(ct) ** (2.0 / n), ct)
            z = math.copysign(abs(st) ** (2.0 / n), st)
            vs.append(bm.verts.new((x, 0.0, z)))
        for i in range(N):
            bm.edges.new((vs[i], vs[(i + 1) % N]))
        # tiny centre dot (a small diamond) for a clear pivot
        d = [bm.verts.new((0.12 * math.cos(a), 0, 0.12 * math.sin(a)))
             for a in (0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
        for i in range(4):
            bm.edges.new((d[i], d[(i + 1) % 4]))
    return _make_wgt(name, build)


# ============================ SKIRT ANTI-PENETRATION ========================
def add_skirt_antipen(rig, props):
    """Stop the skirt poking INTO the body: a Shrinkwrap (Outside Surface) on the
    SEPARATE skirt pushes ONLY penetrating verts back to the body surface (+offset),
    masked to the lower skirt (waistband excluded). Topology-safe -> never breaks
    the Surface Deform bind; sits below Follow, above generative modifiers."""
    sk, body = _skirt_follow_objs(props)
    if sk is None:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    remove_skirt_antipen(rig)
    # vertex-group mask: 0 at the waistband (top ~15%), ramping to 1 toward the hem
    vg = sk.vertex_groups.get("SR_AntiPen") or sk.vertex_groups.new(name="SR_AntiPen")
    mw = sk.matrix_world
    zs = [(mw @ v.co).z for v in sk.data.vertices]
    if not zs:
        return 0
    zmin, zmax = min(zs), max(zs)
    span = (zmax - zmin) or 1.0
    for v in sk.data.vertices:
        f = (zmax - (mw @ v.co).z) / span          # 0 waist -> 1 hem
        w = max(0.0, min(1.0, (f - 0.15) / 0.85))
        vg.add([v.index], w, 'REPLACE')
    mod = sk.modifiers.get("SK_AntiPen") or sk.modifiers.new("SK_AntiPen", 'SHRINKWRAP')
    mod.target = body
    mod.wrap_method = 'NEAREST_SURFACEPOINT'
    # 'OUTSIDE' = push out ONLY verts that are INSIDE the body; leave verts that
    # are already outside exactly where the rig put them. (Do NOT use
    # 'OUTSIDE_SURFACE' - that SNAPS every masked vert onto the nearest surface,
    # so the skirt clings to whatever is closest, e.g. a nearby hand/arm.)
    mod.wrap_mode = 'OUTSIDE'
    mod.offset = float(getattr(props, "skirt_antipen_offset", 0.01))
    mod.vertex_group = "SR_AntiPen"
    # order: after Armature & Surface Deform, ABOVE any generative modifier
    bpy.ops.object.select_all(action='DESELECT')
    sk.select_set(True); bpy.context.view_layer.objects.active = sk
    after = None
    for mm in sk.modifiers:
        if mm.type in ('ARMATURE', 'SURFACE_DEFORM'):
            after = mm.name
    win = bpy.context.window
    area = next((a for a in win.screen.areas if a.type == 'VIEW_3D'), None) if win else None
    region = next((r for r in area.regions if r.type == 'WINDOW'), None) if area else None
    ov = {"object": sk, "active_object": sk}
    if win: ov["window"] = win
    if area: ov["area"] = area
    if region: ov["region"] = region
    try:
        with bpy.context.temp_override(**ov):
            if after is not None:
                tgt = list(m.name for m in sk.modifiers).index(after) + 1
                if list(m.name for m in sk.modifiers).index("SK_AntiPen") != tgt:
                    bpy.ops.object.modifier_move_to_index(modifier="SK_AntiPen", index=tgt)
            # keep generative modifiers BELOW anti-pen
            for mm in list(sk.modifiers):
                if mm.type in _GENERATIVE_MODS:
                    last = len(sk.modifiers) - 1
                    if list(sk.modifiers).index(mm) < list(sk.modifiers).index(sk.modifiers["SK_AntiPen"]):
                        bpy.ops.object.modifier_move_to_index(modifier=mm.name, index=last)
    except Exception as e:
        print("SmartRig anti-pen reorder:", e)
    rig["sk_antipen"] = 1
    return 1


def remove_skirt_antipen(rig):
    n = 0
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        md = ob.modifiers.get("SK_AntiPen")
        if md is not None:
            try:
                ob.modifiers.remove(md); n += 1
            except Exception:
                pass
    if rig is not None and "sk_antipen" in rig:
        del rig["sk_antipen"]
    return n


def antipen_modifier(context):
    p = context.scene.smartrig
    sk = p.skirt_object if getattr(p, "skirt_source", 'MERGED') == 'SEPARATE' else None
    if sk is None:
        for o in bpy.data.objects:
            if o.type == 'MESH' and o.modifiers.get("SK_AntiPen"):
                sk = o; break
    return sk.modifiers.get("SK_AntiPen") if (sk and sk.type == 'MESH') else None


def live_antipen_tune(context):
    try:
        md = antipen_modifier(context)
        if md is not None:
            md.offset = float(context.scene.smartrig.skirt_antipen_offset)
    except Exception as e:
        print("SmartRig anti-pen tune:", e)


# ============================ SKIRT FOLLOW BODY (sit/blend) ==================
# Modifier types that CHANGE topology / vertex count. If any of these sits ABOVE
# SK_SurfaceFollow, the Surface Deform bind input changes -> the bind breaks. We
# keep SK_SurfaceFollow right after the Armature and force ALL of these BELOW it,
# so the bind input is always the stable rigged base cage (never invalidated).
_GENERATIVE_MODS = {
    'SUBSURF', 'MULTIRES', 'SOLIDIFY', 'MIRROR', 'ARRAY', 'BEVEL', 'DECIMATE',
    'REMESH', 'SCREW', 'SKIN', 'WELD', 'EDGE_SPLIT', 'TRIANGULATE', 'WIREFRAME',
    'MASK', 'BUILD', 'BOOLEAN', 'OCEAN', 'EXPLODE', 'NODES',
}


def _order_skirt_deformers(sk):
    """Enforce the professional skirt modifier order so Anti-Penetration is the
    LAST deformer (its no-penetration push wins), and generative mods sit at the
    bottom (so the Surface-Deform bind never breaks):
        Armature -> SK_SurfaceFollow -> SK_Smooth -> SK_AntiPen -> generative.
    """
    arm = [m.name for m in sk.modifiers if m.type == 'ARMATURE']
    surf = [m.name for m in sk.modifiers if m.name == 'SK_SurfaceFollow']
    smooth = [m.name for m in sk.modifiers if m.name == 'SK_Smooth']
    antipen = [m.name for m in sk.modifiers if m.name == 'SK_AntiPen']
    gen = [m.name for m in sk.modifiers if m.type in _GENERATIVE_MODS]
    ranked = arm + surf + smooth + antipen + gen
    others = [m.name for m in sk.modifiers if m.name not in set(ranked)]
    desired = arm + others + surf + smooth + antipen + gen
    try:
        with bpy.context.temp_override(object=sk, active_object=sk):
            for i, nm in enumerate(desired):
                if sk.modifiers.get(nm):
                    bpy.ops.object.modifier_move_to_index(modifier=nm, index=i)
    except Exception as e:
        print("SmartRig order skirt mods:", e)


def skirt_mods_order_ok(props):
    """True if the skirt's modifier stack is in the safe order. Returns
    (ok, message). The required relative order of OUR modifiers is:
    Armature < SK_SurfaceFollow < SK_Smooth < SK_AntiPen < generative."""
    sep = getattr(props, "skirt_source", 'MERGED') == 'SEPARATE'
    sk = props.skirt_object if sep else None
    if sk is None:
        return True, ""
    idx = {m.name: i for i, m in enumerate(sk.modifiers)}
    seq = ["Armature", "SK_SurfaceFollow", "SK_Smooth", "SK_AntiPen"]
    present = [n for n in seq if n in idx]
    last = -1
    for n in present:
        if idx[n] < last:
            return False, "Skirt modifier order is wrong (will break follow / let it clip into the body)."
        last = idx[n]
    # generative must be BELOW SK_AntiPen / SK_SurfaceFollow
    anchor = idx.get("SK_AntiPen", idx.get("SK_SurfaceFollow", -1))
    for m in sk.modifiers:
        if m.type in _GENERATIVE_MODS and anchor >= 0 and idx[m.name] < anchor:
            return False, "A generative modifier (Subsurf/Solidify) is above the skirt deformers."
    return True, ""


class SMARTRIG_OT_skirt_fix_order(bpy.types.Operator):
    bl_idname = "smartrig.skirt_fix_order"
    bl_label = "Fix Skirt Modifier Order"
    bl_description = ("Restore the safe skirt modifier order: Armature -> Follow -> "
                      "Smooth -> Anti-Penetration -> Subsurf/Solidify. Re-binds Follow "
                      "if needed so the skirt deforms correctly and stays out of the body.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.smartrig
        sk = p.skirt_object if getattr(p, "skirt_source", 'MERGED') == 'SEPARATE' else None
        if sk is None:
            self.report({'WARNING'}, "Needs a SEPARATE skirt mesh.")
            return {'CANCELLED'}
        _order_skirt_deformers(sk)
        # a generative mod that had moved above Surface Deform invalidates its bind
        sf = sk.modifiers.get("SK_SurfaceFollow")
        from .metarig import _generated_rig
        rig = _generated_rig()
        if sf is not None and not sf.is_bound and rig is not None:
            try:
                add_skirt_follow_body(rig, p)   # rebuild the bind in the right order
            except Exception:
                pass
        self.report({'INFO'}, "Skirt modifier order fixed.")
        return {'FINISHED'}


def remove_skirt_smooth(props):
    sk = props.skirt_object if getattr(props, "skirt_source", 'MERGED') == 'SEPARATE' else None
    if sk is None:
        return 0
    md = sk.modifiers.get("SK_Smooth")
    if md is not None:
        sk.modifiers.remove(md)
        return 1
    return 0


def add_skirt_smooth(props):
    """Add a Corrective Smooth on the SEPARATE skirt to relax the pinching/
    collapsing that Follow Body (Surface Deform) and Anti-Pen can introduce.
    Placed BEFORE Anti-Pen so the shrinkwrap still has the final no-penetration
    say. ORCO rest source so it smooths toward the undeformed shape."""
    sep = getattr(props, "skirt_source", 'MERGED') == 'SEPARATE'
    sk = props.skirt_object if sep else None
    if sk is None:
        return 0
    md = sk.modifiers.get("SK_Smooth") or sk.modifiers.new("SK_Smooth", 'CORRECTIVE_SMOOTH')
    md.smooth_type = 'LENGTH_WEIGHTED'      # preserves volume better than SIMPLE
    md.rest_source = 'ORCO'
    md.factor = float(getattr(props, "skirt_smooth_factor", 0.5))
    md.iterations = int(getattr(props, "skirt_smooth_iter", 5))
    md.use_only_smooth = False
    md.use_pin_boundary = True              # keep the waistband edge anchored
    # smooth the lower skirt only (reuse the anti-pen mask if present)
    if sk.vertex_groups.get("SR_AntiPen") is not None:
        md.vertex_group = "SR_AntiPen"
    _order_skirt_deformers(sk)
    return 1
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
        try:
            bpy.context.view_layer.objects.active = rig
            bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass
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
            # push EVERY topology-changing modifier BELOW the Surface Deform
            # (order: Armature -> SurfaceDeform -> Subsurf/Solidify/Mirror/...),
            # so they never invalidate the bind and still process the clung result.
            for mm in list(sk.modifiers):
                if mm.type in _GENERATIVE_MODS:
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
        if mm.type in _GENERATIVE_MODS and i < sd_idx:
            return 'subsurf_above', md   # a topology modifier is above -> re-bind
    return 'ok', md


# ============================ SKIRT JIGGLE (live spring) =====================
_JIG_STATE = {}      # bone_name -> {"p":Vector,"v":Vector}
_JIG_LAST_FRAME = [None]


def _column_root_bone(rig, ci):
    """The bone at the TOP of column ci that the whole column hangs from:
    SKC_dt.CC.00 if collision exists, else the control skirt.CC.00."""
    return ("SKC_dt.%02d.00" % ci) if rig.data.bones.get("SKC_dt.%02d.00" % ci) else (PREFIX + ".%02d.00" % ci)


def set_skirt_bbone_segments(rig, n):
    """Set B-bone segments on every DEF-skirt bone. With their AUTO handles the
    deform bones then CURVE smoothly along the springing chain -> a smoother,
    more professional cloth wave (no re-weighting, no extra bones). n=1 = off."""
    if rig is None:
        return 0
    n = max(1, int(n))
    c = 0
    for b in rig.data.bones:
        if re.match(r"^DEF-" + PREFIX + r"\.\d+\.\d+$", b.name):
            try:
                b.bbone_segments = n
                c += 1
            except Exception:
                pass
    return c


def add_skirt_jiggle(rig, props):
    """PROGRESSIVE cloth jiggle: insert a SKC_jig spring bone ABOVE *every row*
    control of each column (not just the root). The spring handler makes each
    row lag its parent, so the motion cascades down the column as a follow-through
    WAVE (waist barely moves, hem waves most) - the real cloth feel."""
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
    if not _edit_rig(rig):
        return -1
    eb = rig.data.edit_bones
    orig = {}
    n_cols = 0
    for ci, rows in cols.items():
        rr = sorted(rows)
        made_any = False
        # insert a jig bone as the PARENT of each row control, top -> hem, so the
        # jig bones interleave with the FK controls and form a springy chain.
        for ridx in rr:
            cname = "%s.%02d.%02d" % (PREFIX, ci, ridx)
            rc = eb.get(cname)
            if rc is None:
                continue
            jn = "SKC_jig.%02d.%02d" % (ci, ridx)
            jig = eb.new(jn)
            jig.head = rc.head.copy(); jig.tail = rc.tail.copy()
            jig.use_deform = False
            op = rc.parent
            orig[cname] = op.name if op else ""
            if op is not None:
                jig.parent = op
            rc.parent = jig
            made_any = True
        if made_any:
            n_cols += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    for cname, pname in orig.items():
        pb = rig.pose.bones.get(cname)
        if pb is not None:
            pb["sk_jigorig"] = pname
    rig["sk_jiggle"] = 1
    if "sk_jiggle_baked" in rig:
        del rig["sk_jiggle_baked"]
    # settings live on the RIG object (works with or without collision; keyframeable)
    spec = (("jiggle", 1.0, 0.0, 1.0, "Enable skirt jiggle (live secondary motion)"),
            ("jiggle_amount", float(getattr(props, "jiggle_amount", 2.5)), 0.0, 5.0, "How much the skirt sways (higher = stronger)"),
            ("jiggle_stiffness", float(getattr(props, "jiggle_stiffness", 0.40)), 0.02, 1.0, "Spring stiffness"),
            ("jiggle_damping", float(getattr(props, "jiggle_damping", 0.25)), 0.05, 0.99, "Damping (higher settles faster)"))
    for k, val, lo, hi, desc in spec:
        rig[k] = val
        try:
            ui = rig.id_properties_ui(k); ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=desc)
        except Exception:
            pass
    set_skirt_bbone_segments(rig, getattr(props, "skirt_jiggle_segments", 3))
    _organize_skirt_bones(rig)
    _JIG_STATE.clear(); _JIG_LAST_FRAME[0] = None
    register_jiggle_handler()
    return n_cols


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
    if not _edit_rig(rig):
        return -1
    eb = rig.data.edit_bones
    for rootname, pname in restore.items():
        rc = eb.get(rootname)
        if rc is not None:
            rc.parent = eb.get(pname) if pname else None
    for b in list(eb):
        if b.name.startswith("SKC_jig") and not b.name.startswith("SKC_jigB"):
            eb.remove(b)
    bpy.ops.object.mode_set(mode='OBJECT')
    _JIG_STATE.clear(); _JIG_LAST_FRAME[0] = None
    if not any(o.type == 'ARMATURE' and (o.get("sk_jiggle") or o.get("sk_chest_jiggle"))
               for o in bpy.data.objects):
        unregister_jiggle_handler()
    return len(restore)


def remove_chest_jiggle(rig):
    """Take the breast bones out of jiggle and restore their parents."""
    if rig is None:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    restore = {}
    for pb in rig.pose.bones:
        if "sk_jigBorig" in pb:
            restore[pb.name] = str(pb["sk_jigBorig"]); del pb["sk_jigBorig"]
    for k in ("sk_chest_jiggle", "sk_chest_jiggle_baked", "chest_jiggle",
              "chest_jiggle_amount", "chest_jiggle_stiffness", "chest_jiggle_damping"):
        if k in rig:
            del rig[k]
    if not _edit_rig(rig):
        return -1
    eb = rig.data.edit_bones
    for rootname, pname in restore.items():       # legacy single-bone version
        rc = eb.get(rootname)
        if rc is not None:
            rc.parent = eb.get(pname) if pname else None
    for b in list(eb):
        if b.name.startswith("SKC_jigB"):
            eb.remove(b)
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    # restore any DEF-breast B-bone we turned on back to a plain bone
    for db in rig.data.bones:
        if "sk_jigB_bbone" in db.keys():
            try:
                db.bbone_segments = 1
                db.bbone_handle_type_end = 'AUTO'
                db.bbone_custom_handle_end = None
            except Exception:
                pass
            del db["sk_jigB_bbone"]
    _JIG_STATE.clear(); _JIG_LAST_FRAME[0] = None
    if not any(o.type == 'ARMATURE' and (o.get("sk_jiggle") or o.get("sk_chest_jiggle"))
               for o in bpy.data.objects):
        unregister_jiggle_handler()
    return len(restore)


def add_chest_jiggle(rig, props):
    """Jiggle the breasts with SOFT, progressive motion. Each DEF-breast bone is
    turned into a B-BONE (multi-segment) whose END HANDLE is a spring bone at the
    tip; the spring lags/bounces when the torso or shoulder moves, so the breast
    curves smoothly from base (still) to tip (bounces most) - a 'jelly' wobble,
    NOT a rigid swing. No body re-weighting needed (the B-bone bends the existing
    weights). Falls back to a simple single-bone spring if there's no DEF-breast."""
    if rig is None:
        return 0
    pairs = [(c, "DEF-" + c) for c in ("breast.L", "breast.R")
             if rig.data.bones.get(c)]
    if not pairs:
        return 0
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    # rig may be hidden (after Back to Metarig) - unhide + activate first
    try:
        rig.hide_set(False)
    except Exception:
        pass
    rig.hide_viewport = False
    try:
        bpy.ops.object.select_all(action='DESELECT')
    except Exception:
        pass
    try:
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
    except Exception:
        pass
    remove_chest_jiggle(rig)
    segs = max(2, int(getattr(props, "chest_jiggle_segments", 3)) + 1)
    geo = {}
    for ctrl, defb in pairs:
        b = rig.data.bones[ctrl]
        geo[ctrl] = (b.tail_local.copy(), (b.tail_local - b.head_local).copy(),
                     defb if rig.data.bones.get(defb) else None)
    if not _edit_rig(rig):
        return -1
    eb = rig.data.edit_bones
    made = []
    for ctrl, (tip, dvec, defb) in geo.items():
        cb = eb.get(ctrl)
        if cb is None:
            continue
        side = ctrl.split(".")[-1]
        nm = "SKC_jigB." + side
        L = dvec.length or 0.05
        jb = eb.new(nm)
        jb.head = tip.copy()
        jb.tail = tip + (dvec.normalized() * (L * 0.5))
        jb.use_deform = False
        jb.parent = cb                      # rides the breast control, then springs
        made.append((ctrl, nm, defb))
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    # turn each DEF-breast into a B-bone whose END handle is the spring tip
    for ctrl, nm, defb in made:
        if defb is None:
            continue
        db = rig.data.bones.get(defb)
        hb = rig.data.bones.get(nm)
        if db is None or hb is None:
            continue
        db["sk_jigB_bbone"] = 1            # remember we changed it (for clean removal)
        db.bbone_segments = segs
        db.bbone_handle_type_end = 'TANGENT'
        db.bbone_custom_handle_end = hb
    rig["sk_chest_jiggle"] = 1
    if "sk_chest_jiggle_baked" in rig:
        del rig["sk_chest_jiggle_baked"]
    spec = (("chest_jiggle", 1.0, 0.0, 1.0, "Enable chest jiggle (live secondary motion)"),
            ("chest_jiggle_amount", float(getattr(props, "chest_jiggle_amount", 2.0)), 0.0, 5.0, "How much the chest bounces"),
            ("chest_jiggle_stiffness", float(getattr(props, "chest_jiggle_stiffness", 0.45)), 0.02, 1.0, "Spring stiffness"),
            ("chest_jiggle_damping", float(getattr(props, "chest_jiggle_damping", 0.30)), 0.05, 0.99, "Damping (higher settles faster)"))
    for k, val, lo, hi, desc in spec:
        rig[k] = val
        try:
            ui = rig.id_properties_ui(k); ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=desc)
        except Exception:
            pass
    # hide the jiggle helpers in the MCH layer
    mch = next((c for c in rig.data.collections_all if c.name == "MCH"), None)
    for n in ("SKC_jigB.L", "SKC_jigB.R"):
        b = rig.data.bones.get(n)
        if b is not None and mch is not None:
            for c in list(b.collections):
                c.unassign(b)
            mch.assign(b)
    _JIG_STATE.clear(); _JIG_LAST_FRAME[0] = None
    register_jiggle_handler()
    return len(made)


def _jiggle_rigs():
    return [o for o in bpy.data.objects if o.type == 'ARMATURE'
            and (o.get("sk_jiggle") or o.get("sk_chest_jiggle"))]


@bpy.app.handlers.persistent
def skirt_jiggle_handler(scene, depsgraph=None):
    from mathutils import Vector, Matrix, noise
    import math
    rigs = _jiggle_rigs()
    if not rigs:
        return
    frame = scene.frame_current
    last = _JIG_LAST_FRAME[0]
    reset = (last is None) or (frame <= last) or (frame - last > 1)
    _JIG_LAST_FRAME[0] = frame
    # SEPARATE force params for the skirt vs the chest (skirt settings never touch
    # the chest and vice-versa). g=gravity, w=wind, d=dir(rad), tb=gust,
    # sp=speed, bl=billow.
    _sp = getattr(scene, "smartrig", None)

    def _fp(pre, has_billow):
        g = float(getattr(_sp, pre + "gravity", 0.0)) if _sp else 0.0
        w = float(getattr(_sp, pre + "wind", 0.0)) if _sp else 0.0
        d = math.radians(float(getattr(_sp, pre + "wind_dir", 0.0))) if _sp else 0.0
        tb = float(getattr(_sp, pre + "wind_turb", 0.3)) if _sp else 0.3
        spd = float(getattr(_sp, pre + "wind_speed", 1.0)) if _sp else 1.0
        bl = float(getattr(_sp, "jiggle_wind_billow", 1.2)) if (has_billow and _sp) else 0.0
        lf = float(getattr(_sp, "jiggle_wind_lift", 0.0)) if (has_billow and _sp) else 0.0
        return (g, w, d, tb, spd, bl, math.sin(d), -math.cos(d), lf)
    _skirt_f = _fp("jiggle_", True)     # jiggle_gravity, jiggle_wind, ...
    _chest_f = _fp("chest_", False)     # chest_gravity, chest_wind, ...

    def _force_for(bname, ncols, chest):
        g, w, d, tb, spd, bl, wx, wy, lf = _chest_f if chest else _skirt_f
        idx = 0; row = 5
        mm = re.match(r"SKC_jig\.(\d+)\.(\d+)$", bname)
        if mm:
            idx = int(mm.group(1)); row = int(mm.group(2))
        else:
            mbb = re.search(r"SKC_jigB?\.(\d+)", bname)
            if mbb:
                idx = int(mbb.group(1))
        off = idx * (7.0 / max(1, ncols))
        t = frame * 0.06 * spd
        nmag = noise.noise(Vector((t, off, 0.0)))
        nside = noise.noise(Vector((t * 0.8, off, 5.3)))
        billow = 1.0 + bl * max(0.0, 1.0 - row / 5.0)
        gust = 1.0 + tb * nmag
        wf = w * 0.035 * gust * billow
        px, py = -wy, wx
        side = w * 0.020 * tb * nside * billow
        # NB: "Lift (blow up)" is applied as a direct outward/up bone rotation in the
        # solver (like posing finger bones), NOT as a wind force here.
        return Vector((wx * wf + px * side, wy * wf + py * side, 0.0)) + \
            Vector((0.0, 0.0, -g * 0.035))
    for rig in rigs:
        rw = rig.matrix_world
        skirt_ok = bool(rig.get("sk_jiggle")) and not rig.get("sk_jiggle_baked")
        chest_ok = bool(rig.get("sk_chest_jiggle")) and not rig.get("sk_chest_jiggle_baked")
        if not (skirt_ok or chest_ok):
            continue
        # column count for a smooth single flutter-wave around the ring
        _ncols = 1
        for b in rig.data.bones:
            mmc = re.match(r"^SKC_jig\.(\d+)\.\d+$", b.name)
            if mmc:
                _ncols = max(_ncols, int(mmc.group(1)) + 1)
        s_par = (float(rig.get("jiggle", 1.0)), float(rig.get("jiggle_amount", 1.0)),
                 float(rig.get("jiggle_stiffness", 0.40)), float(rig.get("jiggle_damping", 0.25)))
        c_par = (float(rig.get("chest_jiggle", 1.0)), float(rig.get("chest_jiggle_amount", 1.0)),
                 float(rig.get("chest_jiggle_stiffness", 0.45)), float(rig.get("chest_jiggle_damping", 0.30)))
        # ---- Blow Up (flip the WHOLE skirt): translate the tweak (shape) bones up
        # and outward. The deform follows tweak POSITIONS (ORG STRETCH_TO between
        # consecutive tweaks -> DEF), so this lifts EVERY row - unlike rotating the
        # jig bones, which only moved the hem. Progressive (hem/tip move most) so the
        # skirt opens like an umbrella and, at high Blow Up, flips up over the top.
        blow = _skirt_f[8] if skirt_ok else 0.0
        for tb in rig.pose.bones:
            if not tb.name.startswith("tweak_skirt."):
                continue
            if blow <= 1e-4:
                if tb.location.length > 1e-7:
                    tb.location = Vector((0.0, 0.0, 0.0))
                continue
            mt = re.search(r"tweak_skirt\.(\d+)\.0*(\d+)$", tb.name)
            row = int(mt.group(2)) if mt else 3
            # NOT capped at 1.0: the tip tweak (row 6, the STRETCH_TO target of the hem
            # DEF) must rise MORE than the hem, otherwise the last segment ends up flat
            # or pointing DOWN. Uncapped -> tip lifts ~20% past the hem and the curl
            # keeps going up.
            rf = row / 5.0                              # 0 waist .. 1 hem .. 1.2 tip
            head = rw @ tb.bone.head_local
            rad = Vector((head.x - rw.translation.x, head.y - rw.translation.y, 0.0))
            if rad.length > 1e-4:
                rad.normalize()
            world_off = Vector((0.0, 0.0, blow * 0.022 * rf)) + rad * (blow * 0.014 * rf)
            try:
                tb.location = tb.matrix.to_3x3().inverted() @ world_off
            except Exception:
                pass
        for pb in rig.pose.bones:
            nm = pb.name
            if nm.startswith("SKC_jigB"):           # breast jiggle (separate params)
                if not chest_ok:
                    continue
                on, amount, stiff, damp = c_par; is_chest = True
            elif nm.startswith("SKC_jig"):           # skirt jiggle
                if not skirt_ok:
                    continue
                on, amount, stiff, damp = s_par; is_chest = False
            else:
                continue
            # Enable = 0 -> do NOTHING: leave the bone to follow its parent
            # naturally (clear any leftover jiggle). Avoids the 1-frame lag/curve
            # from posing it every frame when the jiggle is OFF.
            if on < 0.5:
                try:
                    pb.matrix_basis = Matrix.Identity(4)
                except Exception:
                    pass
                _JIG_STATE.pop(pb.name, None)
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
                _ff = _chest_f if is_chest else _skirt_f
                if _ff[1] or _ff[0] or _ff[8]:   # wind, gravity, or lift
                    v += _force_for(nm, _ncols, is_chest)
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
            # NB: "Blow Up (flip skirt)" is NOT done here. Rotating the jig bones only
            # moved the hem (the skirt deforms from tweak POSITIONS via ORG STRETCH_TO,
            # not from jig rotation). Blow Up is applied by TRANSLATING the tweak bones
            # in the dedicated loop below.
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

    # professional top-down order: Master (coarse) -> FK (per column) -> Tweak
    # (fine, HIDDEN by default) -> MCH/Dynamics (always hidden, never touched).
    master_c = get_coll("Skirt (Master)", True)
    main = get_coll("Skirt (FK)", True)
    tweak = get_coll("Skirt (Tweak)", False)     # hidden by default; toggle in Rig Layers
    mch = get_coll("Skirt (MCH)", False)
    # migrate any old "Skirt" collection name to "Skirt (FK)"
    old = next((x for x in arm.collections_all if x.name == "Skirt"), None)
    if old is not None and old is not main:
        for b in list(getattr(old, "bones", [])):
            main.assign(b)
        try:
            arm.collections.remove(old)
        except Exception:
            pass
    # Rigify "Rig Layers" panel reads rigify_ui_row (top -> bottom). Master first.
    try:
        master_c.rigify_ui_row = 20
        main.rigify_ui_row = 21
        tweak.rigify_ui_row = 22
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
        if n.startswith("SKC_dt") or n.startswith("SKC_jig") or n == "SKC_master":
            reassign(b, mch)                         # SKC_master = settings holder -> hidden
        elif n.startswith("skirt_master"):
            reassign(b, master_c); col(b, *GOLD)     # real movement masters
        elif re.match(r"^" + PREFIX + r"\.\d+\.\d+$", n):
            reassign(b, main); col(b, *PINK)
        elif PREFIX in n and "tweak" in n:
            reassign(b, tweak); col(b, *PURP)
    # SKC_master is a SETTINGS container (its custom props drive the collision; its
    # values are edited from the N-panel Item sliders), NOT a transform control -
    # so it lives in the hidden MCH layer and needs no widget. Drop the orphan one.
    mpb = rig.pose.bones.get("SKC_master")
    if mpb is not None:
        mpb.custom_shape = None
    w = bpy.data.objects.get("WGT-SKC_master")
    if w is not None:
        try:
            bpy.data.objects.remove(w, do_unlink=True)
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

    if not _edit_rig(rig):
        return -1
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
            _mvar(drv, "dd", "collide_dist"); _mvar(drv, "ddf", "collide_dist_falloff")
            # base clearance (ddf) is a CONSTANT outward push applied even at rest,
            # so the panels always sit slightly off the legs (stops static
            # penetration); the second term is the leg-movement swing on top.
            drv.expression = (
                "%.4f*(0.18*ddf + min(1.2,max(0.0,%.4f*%s+%.4f*%s)))*(dd/0.12)*min(1.5,spread)*col"
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
            if r < 0:
                self.report({'ERROR'}, _NO_ACCESS)
                return {'CANCELLED'}
            self.report({'INFO'}, "Skirt collision removed (%d constraints)." % r)
            return {'FINISHED'}
        n = add_skirt_collision(rig, p, h)
        if n < 0:
            self.report({'ERROR'}, _NO_ACCESS)
            return {'CANCELLED'}
        if not n:
            self.report({'WARNING'}, "No skirt bones or no collider bones found.")
            return {'CANCELLED'}
        self.report({'INFO'}, "Skirt collision applied (%d constraints)." % n)
        return {'FINISHED'}


class SMARTRIG_OT_skirt_masters(bpy.types.Operator):
    bl_idname = "smartrig.skirt_masters"
    bl_label = "Skirt Region Masters"
    bl_description = ("Build a global + per-region (front/sides/back) master controls "
                      "so you can pose whole regions of the skirt at once. Set the "
                      "number of sectors first. Use 'Remove' to take them off.")
    bl_options = {'REGISTER', 'UNDO'}
    remove: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None:
            self.report({'ERROR'}, "Generate the rig first.")
            return {'CANCELLED'}
        if self.remove:
            r = remove_skirt_masters(rig)
            if r < 0:
                self.report({'ERROR'}, _NO_ACCESS)
                return {'CANCELLED'}
            self.report({'INFO'}, "Removed %d master controls." % r)
            return {'FINISHED'}
        n = add_skirt_masters(rig, context.scene.smartrig)
        if n < 0:
            self.report({'ERROR'}, _NO_ACCESS)
            return {'CANCELLED'}
        if not n:
            self.report({'WARNING'}, "No skirt bones found (build + generate the skirt first).")
            return {'CANCELLED'}
        self.report({'INFO'}, "Built 1 global + %d region masters." % n)
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
        kind = mo.get("sr_skirt_kind", "?") if mo else "?"
        method = mo.get("sr_skirt_method", "?") if mo else "?"
        nb = int(mo.get("sr_skirt_cols_built", 0)) if mo else 0
        _label = {"TUBE": "clean tube", "OPEN": "open-front", "LAYERED": "layered",
                  "CLOSED": "closed tube (slit)", "MERGED": "merged-in-body",
                  "MESSY": "irregular"}.get(kind, kind)
        self.report({'INFO'}, "Skirt detected: %s -> %s placement (%d columns). Generate the rig next."
                    % (_label, method, nb))
        return {'FINISHED'}


def remove_skirt(context):
    """Remove EVERYTHING the Short Skirt added: all skirt bones (metarig AND the
    generated rig), the collision / jiggle / follow / anti-penetration extras,
    the skirt bone collections and widgets. The skirt MESH is left untouched."""
    if context.object and context.object.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    meta = bpy.data.objects.get(META_NAME)
    rig = bpy.data.objects.get("RIG-" + META_NAME)
    # 1) strip the live extras from the generated rig (constraints/mods/drivers)
    if rig is not None:
        for fn in (remove_skirt_masters, remove_skirt_collision, remove_skirt_jiggle,
                   remove_skirt_follow_body, remove_skirt_antipen):
            try:
                fn(rig)
            except Exception:
                pass

    def _is_skirt(n):
        # NOT the chest-jiggle helpers (SKC_jigB) - that's a separate feature
        if n.startswith("SKC_jigB"):
            return False
        return ("skirt" in n.lower()) or n.startswith("SKC_")

    n_removed = 0
    for arm in (meta, rig):
        if arm is None:
            continue
        try:
            arm.hide_set(False)
        except Exception:
            pass
        arm.hide_viewport = False
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            pass
        bpy.context.view_layer.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        eb = arm.data.edit_bones
        for b in [bb for bb in eb if _is_skirt(bb.name)]:
            try:
                eb.remove(b)
                n_removed += 1
            except Exception:
                pass
        bpy.ops.object.mode_set(mode='OBJECT')
        # drop the now-empty skirt bone collections
        for cn in ("Skirt", "Skirt (FK)", "Skirt (Tweak)", "Skirt (Master)", "Skirt (MCH)"):
            bc = arm.data.collections.get(cn)
            if bc is not None:
                try:
                    arm.data.collections.remove(bc)
                except Exception:
                    pass
    # 2) skirt widgets
    for o in list(bpy.data.objects):
        if o.name.startswith("WGT-") and ("skirt" in o.name.lower() or "SKC" in o.name):
            md = o.data
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
            if md and getattr(md, "users", 1) == 0:
                try:
                    bpy.data.meshes.remove(md)
                except Exception:
                    pass
    # 3) clear the skirt marker flags
    for r in (meta, rig):
        if r is None:
            continue
        for k in ("sk_kilt", "sk_jiggle", "sk_jiggle_baked", "sk_follow", "sk_antipen"):
            if k in r.keys():
                try:
                    del r[k]
                except Exception:
                    pass
    return n_removed


def check_skirt_integrity(arm):
    """Return a list of (column_index, reason) problems with the skirt chains on
    `arm`. Each column must be a connected chain of >=2 bones with rows
    contiguous from .00. Empty list = OK (or no skirt)."""
    import re as _re
    if arm is None:
        return []
    rx = _re.compile(r"^skirt\.(\d+)\.(\d+)$")
    cols = {}
    for b in arm.data.bones:
        m = rx.match(b.name)
        if m:
            cols.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    problems = []
    for c in sorted(cols):
        rows = cols[c]
        if 0 not in rows:
            problems.append((c, "missing its root bone (row .00)"))
            continue
        k = 0
        while k in rows:
            k += 1
        chain_len = k                       # contiguous rows 0..k-1
        if chain_len < 2:
            problems.append((c, "has only %d bone - a column needs 2+" % chain_len))
        elif chain_len < len(rows):
            problems.append((c, "has a missing middle bone (broken chain)"))
    return problems


def skirt_integrity_message(problems):
    """One clear, actionable sentence describing how to fix broken skirt columns."""
    if not problems:
        return ""
    cols = ", ".join(str(c) for c, _ in problems)
    return ("Skirt column(s) %s %s. A skirt column must be a full connected chain "
            "of 2+ bones. Fix: delete the WHOLE column, OR press 'Remove Skirt' and "
            "rebuild, OR use the Rows/Columns sliders. Do NOT delete individual "
            "skirt bones." % (cols, problems[0][1]))


class SMARTRIG_OT_remove_skirt(bpy.types.Operator):
    bl_idname = "smartrig.remove_skirt"
    bl_label = "Remove Skirt"
    bl_description = ("Delete ALL skirt bones and extras (collision, jiggle, follow, "
                      "anti-penetration) from the metarig and the generated rig. "
                      "Your skirt MESH is NOT touched.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        meta = bpy.data.objects.get(META_NAME)
        rig = bpy.data.objects.get("RIG-" + META_NAME)
        for arm in (meta, rig):
            if arm and any(b.name.startswith("skirt.") or b.name.startswith("SKC_")
                           or "skirt" in b.name.lower() for b in arm.data.bones):
                return True
        return False

    def execute(self, context):
        n = remove_skirt(context)
        self.report({'INFO'}, "Removed %d skirt bones + extras. (Mesh untouched.)" % n)
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
        try:
            context.view_layer.objects.active = rig
        except Exception:
            return
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
        _skirtish = ("DEF-" + PREFIX + ".", PREFIX + "_master", PREFIX + ".",
                     "tweak_" + PREFIX + ".", "SKC_")
        for b in rig.data.bones:
            if b.use_deform and b.name.startswith(_skirtish):
                saved[b.name] = b.use_deform; b.use_deform = False
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
            r = remove_skirt_jiggle(rig)
            if r < 0:
                self.report({'ERROR'}, _NO_ACCESS)
                return {'CANCELLED'}
            self.report({'INFO'}, "Skirt jiggle removed."); return {'FINISHED'}
        n = add_skirt_jiggle(rig, context.scene.smartrig)
        if n < 0:
            self.report({'ERROR'}, _NO_ACCESS)
            return {'CANCELLED'}
        if not n:
            self.report({'WARNING'}, "No skirt bones found."); return {'CANCELLED'}
        self.report({'INFO'}, "Skirt jiggle applied (%d columns). Play the timeline." % n)
        return {'FINISHED'}


class SMARTRIG_OT_bake_jiggle(bpy.types.Operator):
    bl_idname = "smartrig.bake_jiggle"
    bl_label = "Bake Skirt Jiggle"
    bl_description = ("Bake the live skirt jiggle of the frame range onto keyframes "
                     "(the live solver then stops). Use Clear Bake to go live again.")
    bl_options = {'REGISTER', 'UNDO'}
    remove: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None or not rig.get("sk_jiggle"):
            self.report({'ERROR'}, "Apply skirt jiggle first."); return {'CANCELLED'}
        sc = context.scene
        if context.object is not rig or rig.mode != 'POSE':
            try:
                context.view_layer.objects.active = rig; rig.hide_set(False)
            except Exception:
                pass
            try:
                bpy.ops.object.mode_set(mode='POSE')
            except Exception:
                pass
        jigs = [pb for pb in rig.pose.bones
                if pb.name.startswith("SKC_jig") and not pb.name.startswith("SKC_jigB")]
        if self.remove:
            _remove_jig_fcurves(rig, False)
            for pb in jigs:
                try:
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = (1, 0, 0, 0)
                    pb.matrix_basis = pb.matrix_basis.Identity(4)
                except Exception:
                    pass
            if "sk_jiggle_baked" in rig:
                del rig["sk_jiggle_baked"]
            _JIG_STATE.clear(); _JIG_LAST_FRAME[0] = None
            register_jiggle_handler()
            self.report({'INFO'}, "Skirt bake cleared - live jiggle again.")
            return {'FINISHED'}
        for pb in jigs:
            pb.rotation_mode = 'QUATERNION'
        if "sk_jiggle_baked" in rig:
            del rig["sk_jiggle_baked"]
        for f in range(sc.frame_start, sc.frame_end + 1):
            sc.frame_set(f)   # spring handler runs and poses the jig bones
            for pb in jigs:
                pb.keyframe_insert("rotation_quaternion", frame=f)
        rig["sk_jiggle_baked"] = 1   # handler now skips this rig; keyframes play it back
        self.report({'INFO'}, "Baked skirt jiggle %d-%d." % (sc.frame_start, sc.frame_end))
        return {'FINISHED'}


def _jig_dp_match(dp, chest):
    """True if the fcurve data_path belongs to the chest (SKC_jigB) or the skirt
    (SKC_jig but NOT SKC_jigB) jiggle bones."""
    is_b = "SKC_jigB" in dp
    return is_b if chest else ("SKC_jig" in dp and not is_b)


def _remove_jig_fcurves(rig, chest):
    """Remove ALL keyframes on the skirt (chest=False) or chest (chest=True) jiggle
    bones. Version-agnostic (legacy action.fcurves OR 4.4+/5.x channelbags)."""
    ad = rig.animation_data
    act = ad.action if ad else None
    if act is None:
        return 0
    n = 0
    if hasattr(act, "fcurves"):                 # legacy
        try:
            for fc in list(act.fcurves):
                if _jig_dp_match(fc.data_path, chest):
                    act.fcurves.remove(fc); n += 1
            return n
        except Exception:
            pass
    for layer in getattr(act, "layers", []):    # slotted (4.4+/5.x)
        for strip in getattr(layer, "strips", []):
            for cb in (getattr(strip, "channelbags", None) or []):
                for fc in list(cb.fcurves):
                    if _jig_dp_match(fc.data_path, chest):
                        try:
                            cb.fcurves.remove(fc); n += 1
                        except Exception:
                            pass
    return n


def _jig_has_keys(rig, chest):
    ad = rig.animation_data if rig else None
    act = ad.action if ad else None
    if act is None:
        return False
    if hasattr(act, "fcurves"):
        try:
            return any(_jig_dp_match(fc.data_path, chest) for fc in act.fcurves)
        except Exception:
            pass
    for layer in getattr(act, "layers", []):
        for strip in getattr(layer, "strips", []):
            for cb in (getattr(strip, "channelbags", None) or []):
                if any(_jig_dp_match(fc.data_path, chest) for fc in cb.fcurves):
                    return True
    return False


def _remove_jigB_fcurves(rig):
    return _remove_jig_fcurves(rig, True)


def chest_jiggle_has_keys(rig):
    return _jig_has_keys(rig, True)


def skirt_jiggle_has_keys(rig):
    return _jig_has_keys(rig, False)


class SMARTRIG_OT_chest_bake(bpy.types.Operator):
    bl_idname = "smartrig.chest_bake"
    bl_label = "Bake Chest Jiggle"
    bl_description = ("Bake the live chest jiggle of the frame range onto keyframes "
                      "(the live solver then stops). Use Clear Bake to go live again.")
    bl_options = {'REGISTER', 'UNDO'}
    remove: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None or not rig.get("sk_chest_jiggle"):
            self.report({'ERROR'}, "Apply chest jiggle first.")
            return {'CANCELLED'}
        jigs = [pb for pb in rig.pose.bones if pb.name.startswith("SKC_jigB")]
        if context.object is not rig or rig.mode != 'POSE':
            try:
                context.view_layer.objects.active = rig
                rig.hide_set(False)
            except Exception:
                pass
            try:
                bpy.ops.object.mode_set(mode='POSE')
            except Exception:
                pass
        sc = context.scene
        if self.remove:
            # remove ALL baked keyframes on the SKC_jigB bones (robust, any range)
            _remove_jigB_fcurves(rig)
            for pb in jigs:
                try:
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = (1, 0, 0, 0)
                    pb.matrix_basis = pb.matrix_basis.Identity(4)
                except Exception:
                    pass
            for k in ("sk_chest_jiggle_baked", "sk_chest_bake_s", "sk_chest_bake_e"):
                if k in rig:
                    del rig[k]
            _JIG_STATE.clear(); _JIG_LAST_FRAME[0] = None
            register_jiggle_handler()
            self.report({'INFO'}, "Chest bake cleared - live jiggle again.")
            return {'FINISHED'}
        for pb in jigs:
            pb.rotation_mode = 'QUATERNION'
        if "sk_chest_jiggle_baked" in rig:
            del rig["sk_chest_jiggle_baked"]
        for f in range(sc.frame_start, sc.frame_end + 1):
            sc.frame_set(f)              # spring handler poses the SKC_jigB bones
            for pb in jigs:
                pb.keyframe_insert("rotation_quaternion", frame=f)
        rig["sk_chest_jiggle_baked"] = 1   # handler now skips chest; keyframes play it
        rig["sk_chest_bake_s"] = sc.frame_start
        rig["sk_chest_bake_e"] = sc.frame_end
        self.report({'INFO'}, "Baked chest jiggle %d-%d." % (sc.frame_start, sc.frame_end))
        return {'FINISHED'}


class SMARTRIG_OT_skirt_antipen(bpy.types.Operator):
    bl_idname = "smartrig.skirt_antipen"
    bl_label = "Apply Anti-Penetration"
    bl_description = ("Stop the skirt poking into the body: a Shrinkwrap (Outside Surface) "
                     "pushes only penetrating verts back out. Needs a SEPARATE skirt mesh.")
    bl_options = {'REGISTER', 'UNDO'}
    remove: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None:
            self.report({'ERROR'}, "Generate the rig first."); return {'CANCELLED'}
        if self.remove:
            remove_skirt_antipen(rig)
            self.report({'INFO'}, "Anti-penetration removed."); return {'FINISHED'}
        p = context.scene.smartrig
        if not (p.skirt_source == 'SEPARATE' and p.skirt_object is not None):
            self.report({'WARNING'}, "Anti-Penetration needs a SEPARATE skirt mesh.")
            return {'CANCELLED'}
        n = add_skirt_antipen(rig, p)
        if not n:
            self.report({'WARNING'}, "Failed - is the skirt mesh valid?"); return {'CANCELLED'}
        self.report({'INFO'}, "Anti-penetration added (Shrinkwrap Outside). Tune Offset.")
        return {'FINISHED'}


class SMARTRIG_OT_skirt_smooth(bpy.types.Operator):
    bl_idname = "smartrig.skirt_smooth"
    bl_label = "Corrective Smooth"
    bl_description = ("Add/refresh a Corrective Smooth on the skirt to relax pinching "
                      "from Follow Body / Anti-Pen. Placed BEFORE Anti-Penetration so "
                      "the skirt still can't enter the body. Needs a SEPARATE skirt.")
    bl_options = {'REGISTER', 'UNDO'}
    remove: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        p = context.scene.smartrig
        if not (p.skirt_source == 'SEPARATE' and p.skirt_object is not None):
            self.report({'WARNING'}, "Corrective Smooth needs a SEPARATE skirt mesh.")
            return {'CANCELLED'}
        if self.remove:
            remove_skirt_smooth(p)
            self.report({'INFO'}, "Corrective Smooth removed.")
            return {'FINISHED'}
        add_skirt_smooth(p)
        self.report({'INFO'}, "Corrective Smooth added (before Anti-Pen, so no body penetration).")
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
        p = context.scene.smartrig
        if not (p.skirt_source == 'SEPARATE' and p.skirt_object is not None):
            self.report({'WARNING'},
                        "Follow Body needs a SEPARATE skirt mesh. Collision + Jiggle still "
                        "work; a merged skirt loses the surface-cling. For a merged skirt, "
                        "select its faces and use 'Register Skirt Selection' for skinning.")
            return {'CANCELLED'}
        n = add_skirt_follow_body(rig, p)
        if not n:
            self.report({'WARNING'}, "Bind failed - is the skirt mesh valid?"); return {'CANCELLED'}
        self.report({'INFO'}, "Body follow applied. Raise 'Follow Body' (it clings when seated).")
        return {'FINISHED'}


class SMARTRIG_OT_chest_jiggle(bpy.types.Operator):
    bl_idname = "smartrig.chest_jiggle"
    bl_label = "Jiggle Chest"
    bl_description = ("Turn the breast bones into live jiggle (spring secondary "
                      "motion). Toggle and tune its strength below. Needs a "
                      "generated rig with breast bones.")
    bl_options = {'REGISTER', 'UNDO'}
    remove: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None:
            self.report({'ERROR'}, "Generate the rig first.")
            return {'CANCELLED'}
        if self.remove:
            r = remove_chest_jiggle(rig)
            if r < 0:
                self.report({'ERROR'}, _NO_ACCESS)
                return {'CANCELLED'}
            self.report({'INFO'}, "Chest jiggle removed.")
            return {'FINISHED'}
        n = add_chest_jiggle(rig, context.scene.smartrig)
        if n < 0:
            self.report({'ERROR'}, _NO_ACCESS)
            return {'CANCELLED'}
        if not n:
            self.report({'WARNING'}, "No breast bones found on the rig.")
            return {'CANCELLED'}
        self.report({'INFO'}, "Chest jiggle added. Play the timeline to see it.")
        return {'FINISHED'}


class SMARTRIG_OT_rig_skirt_standalone(bpy.types.Operator):
    bl_idname = "smartrig.rig_skirt_standalone"
    bl_label = "Build Skirt Metarig"
    bl_description = ("Build a STANDALONE skirt METARIG - no body/markers. Creates a "
                      "root bone at the waist + editable skirt chains (edge-flow when "
                      "the topology is clean, else robust angular). Does NOT generate - "
                      "tweak it, then press Generate Rig.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import numpy as np
        from .metarig import META_NAME, _ensure_rigify
        props = context.scene.smartrig
        ob = getattr(props, "skirt_object", None)
        if getattr(props, "skirt_source", 'MERGED') != 'SEPARATE' or ob is None or ob.type != 'MESH':
            self.report({'ERROR'}, "Pick a separate skirt mesh in the Mesh field first.")
            return {'CANCELLED'}
        if not _ensure_rigify():
            self.report({'ERROR'}, "Rigify add-on is not available / enabled.")
            return {'CANCELLED'}
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # root centre = bounding-box midpoint of the WHOLE skirt (density-proof, and
        # the SAME vertical axis the columns are placed around), at the top. The median
        # would be pulled off-centre by dense vertex clusters.
        co = utils.read_rest_coords(ob)
        zmax = float(co[:, 2].max()); zmin = float(co[:, 2].min()); h = max(zmax - zmin, 1e-4)
        cx = float((co[:, 0].min() + co[:, 0].max()) * 0.5)
        cy = float((co[:, 1].min() + co[:, 1].max()) * 0.5)
        # fresh metarig (inherits Rigify's armature config), stripped to one root bone
        old = bpy.data.objects.get(META_NAME)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        before = set(o.name for o in bpy.data.objects)
        bpy.ops.object.armature_human_metarig_add()
        new = [o for o in bpy.data.objects if o.name not in before and o.type == 'ARMATURE']
        if not new:
            self.report({'ERROR'}, "Could not create a metarig (Rigify).")
            return {'CANCELLED'}
        mo = new[0]; mo.name = META_NAME; mo.data.name = META_NAME; mo.show_in_front = True
        mo.location = (0.0, 0.0, 0.0); mo.scale = (1.0, 1.0, 1.0)
        for o in context.selected_objects:
            o.select_set(False)
        context.view_layer.objects.active = mo; mo.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.mode_set(mode='EDIT')
        eb = mo.data.edit_bones
        for b in list(eb):
            eb.remove(b)
        root = eb.new("spine")
        root.head = (cx, cy, zmax)
        root.tail = (cx, cy, zmax + max(0.05 * h, 0.02))
        bpy.ops.object.mode_set(mode='OBJECT')
        pbroot = mo.pose.bones.get("spine")
        if pbroot is not None:
            try:
                pbroot.rigify_type = "basic.super_copy"
            except Exception:
                pass
        # build the skirt chains (they parent to 'spine') - METARIG ONLY, no generate
        _mo, err = build_skirt(props)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        # leave the metarig VISIBLE + active so the user can tweak Columns/Rows or edit
        # bones, then press "Generate Rig" (same flow as Character mode - no auto-gen).
        for o in context.selected_objects:
            o.select_set(False)
        try:
            mo.hide_set(False)
        except Exception:
            pass
        mo.hide_viewport = False
        mo.select_set(True); context.view_layer.objects.active = mo
        kind = mo.get("sr_skirt_kind", "?"); method = mo.get("sr_skirt_method", "?")
        self.report({'INFO'}, "Skirt metarig built (%s -> %s). Tweak Columns/Rows or edit "
                    "bones, then press Generate Rig." % (kind, method))
        return {'FINISHED'}


classes = (SMARTRIG_OT_register_skirt, SMARTRIG_OT_add_skirt,
           SMARTRIG_OT_remove_skirt, SMARTRIG_OT_skirt_masters,
           SMARTRIG_OT_bind, SMARTRIG_OT_unbind, SMARTRIG_OT_skirt_collision,
           SMARTRIG_OT_skirt_jiggle, SMARTRIG_OT_bake_jiggle,
           SMARTRIG_OT_skirt_follow, SMARTRIG_OT_skirt_antipen,
           SMARTRIG_OT_skirt_smooth, SMARTRIG_OT_skirt_fix_order,
           SMARTRIG_OT_chest_jiggle, SMARTRIG_OT_chest_bake,
           SMARTRIG_OT_rig_skirt_standalone)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    # Always arm the live jiggle handler (it no-ops when there are no jiggle rigs).
    # The handler is @persistent so it also survives file loads. This is more
    # reliable than conditionally re-arming, which could miss after a reload.
    try:
        register_jiggle_handler()
    except Exception:
        pass


def unregister():
    unregister_jiggle_handler()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
