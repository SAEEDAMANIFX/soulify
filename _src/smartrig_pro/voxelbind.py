# Soulify VoxelBind - part-aware, layer-aware voxel weights for garments.
#
# WHY (measured, v1.35.1 session): plain voxel heat (vhd.exe, mesh-online)
# is PART-BLIND - heat crosses the armpit seam exactly like Euclidean
# distance, leaving 35% of torso fabric dominated by ARM bones (our old
# inverse-distance weights: 40%). More diffusion cannot fix that; DOMAINS
# do: sleeve fabric may only receive arm-bone heat, the collar only the
# upper spine, everything else only the spine (+ legs for pants).
# Layers come free: heat travels through FABRIC voxels only, so an air
# gap >= 1 voxel splits stacked shells (jacket / shirt / body) - weights
# never jump between layers, and the BODY's voxels are hard obstacles so
# heat can never shortcut through the character's chest.
#
# PIPELINE (numpy only, no external binary):
#   1. occupancy grid of the garment surface (verts + edge midpoints +
#      poly centres, 1-voxel dilation seals pinholes)
#   2. body voxels become obstacles (fabric cells stay traversable)
#   3. SEMANTIC DOMAINS from the CHARACTER's own skin (MetaTailor's
#      insight): each garment vertex takes the part class of its nearest
#      BODY vertex, and the body knows its parts from its rig weights
#      (DEF-upper_arm.L fabric = the left sleeve, spine fabric = torso).
#      Measured: pure geodesic Voronoi CANNOT work - the armpit SEAM is
#      sewn fabric, a perfect heat bridge (ours 40%, vhd.exe 35% torso
#      arm-bleed). Semantics must come from outside the fabric graph.
#   4. per domain x allowed bone: occlusion-tested seeding + GEODESIC
#      distance through the domain fabric; 3-voxel feather at borders;
#      weights 1/(d+r0)^2, top-4. WIZARD-registered parts stay hard
#      overrides; no body rig -> auto part groups -> radius fallback.

import numpy as np

VOX_LONG = 128          # cells along the longest garment axis
FEATHER = 2.0           # seam blend band, in voxels
RELAX_IT = 72           # geodesic relaxation sweeps (dist beyond ~cap unused)
INF = 1e9


# ------------------------------------------------------------------ grid --
def _grid(wco, pad=3):
    lo = wco.min(axis=0)
    hi = wco.max(axis=0)
    vs = float((hi - lo).max()) / float(VOX_LONG - 1)
    vs = max(vs, 1e-5)
    lo = lo - pad * vs
    dims = np.ceil((hi - lo) / vs).astype(int) + pad + 1
    dims = np.minimum(dims, 170)
    return lo, vs, tuple(int(d) for d in dims)


def _cells(pts, lo, vs, dims):
    c = np.floor((pts - lo) / vs).astype(int)
    for a in range(3):
        np.clip(c[:, a], 0, dims[a] - 1, out=c[:, a])
    return c


def _occupancy(g_ob, wco, lo, vs, dims):
    """Garment surface occupancy: verts + edge midpoints + poly centres."""
    occ = np.zeros(dims, dtype=bool)
    pts = [wco]
    me = g_ob.data
    if len(me.edges):
        ev = np.empty(2 * len(me.edges), dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        pts.append(0.5 * (wco[ev[:, 0]] + wco[ev[:, 1]]))
    if len(me.polygons):
        n = len(me.vertices)
        # poly centres via loop scatter
        li = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", li)
        tot = np.empty(len(me.polygons), dtype=np.int64)
        me.polygons.foreach_get("loop_total", tot)
        pid = np.repeat(np.arange(len(me.polygons)), tot)
        ctr = np.zeros((len(me.polygons), 3))
        np.add.at(ctr, pid, wco[li])
        ctr /= np.maximum(tot[:, None], 1)
        pts.append(ctr)
    allp = np.vstack(pts)
    c = _cells(allp, lo, vs, dims)
    occ[c[:, 0], c[:, 1], c[:, 2]] = True
    return occ


def _dilate(m):
    out = m.copy()
    out[1:, :, :] |= m[:-1, :, :]
    out[:-1, :, :] |= m[1:, :, :]
    out[:, 1:, :] |= m[:, :-1, :]
    out[:, :-1, :] |= m[:, 1:, :]
    out[:, :, 1:] |= m[:, :, :-1]
    out[:, :, :-1] |= m[:, :, 1:]
    return out


# ------------------------------------------------- geodesic distance field --
def _relax(d, walk, iters=RELAX_IT):
    """Obstacle-aware distance: iterative 6-neighbour relaxation of the
    seeded field d (INF where unseeded) across walkable cells. Vectorised;
    distances in voxel units."""
    d = np.where(walk, d, INF)
    for _ in range(iters):
        nd = d
        nd = np.minimum(nd, np.pad(d, ((1, 0), (0, 0), (0, 0)),
                                   constant_values=INF)[:-1] + 1.0)
        nd = np.minimum(nd, np.pad(d, ((0, 1), (0, 0), (0, 0)),
                                   constant_values=INF)[1:] + 1.0)
        nd = np.minimum(nd, np.pad(d, ((0, 0), (1, 0), (0, 0)),
                                   constant_values=INF)[:, :-1] + 1.0)
        nd = np.minimum(nd, np.pad(d, ((0, 0), (0, 1), (0, 0)),
                                   constant_values=INF)[:, 1:] + 1.0)
        nd = np.minimum(nd, np.pad(d, ((0, 0), (0, 0), (1, 0)),
                                   constant_values=INF)[:, :, :-1] + 1.0)
        nd = np.minimum(nd, np.pad(d, ((0, 0), (0, 0), (0, 1)),
                                   constant_values=INF)[:, :, 1:] + 1.0)
        nd = np.where(walk, nd, INF)
        if np.array_equal(nd, d):
            break
        d = nd
    return d


def _bone_seed(a0, b0, walk_idx, walk_pts, vs, occ, lo, dims):
    """Seed a bone segment onto the fabric it actually REACHES: candidate
    cells within (min distance + 2 voxels) of the segment, then an
    OCCLUSION test - the straight ray from the bone to the cell must not
    pass through OTHER fabric first (the sleeve wall occludes the chest;
    body voxels do not occlude, the bone legitimately sits inside the
    body). Euclidean-only seeding leaked arm heat straight onto the side
    chest = the 39% regression."""
    ab = b0 - a0
    L2 = float(ab @ ab) + 1e-12
    t = np.clip(((walk_pts - a0) @ ab) / L2, 0.0, 1.0)
    cl = a0 + t[:, None] * ab
    d = np.linalg.norm(walk_pts - cl, axis=1) / vs
    dmin = float(d.min()) if len(d) else 0.0
    seed = d <= dmin + 2.0
    ci = np.nonzero(seed)[0]
    if len(ci) == 0:
        return seed, d
    P0 = cl[ci]                      # entry point on the bone
    P1 = walk_pts[ci]                # candidate cell centre
    tgt = walk_idx[ci]
    occl = np.zeros(len(ci), dtype=bool)
    for tt in np.linspace(0.15, 0.85, 18):
        S = P0 + tt * (P1 - P0)
        c = np.floor((S - lo) / vs).astype(int)
        ok = np.all((c >= 0) & (c < np.array(dims)), axis=1)
        hit = np.zeros(len(ci), dtype=bool)
        cc = c[ok]
        hit[ok] = occ[cc[:, 0], cc[:, 1], cc[:, 2]]
        # a hit in the candidate's own cell (or its 26-hood) is the
        # candidate's own wall, not an occluder
        own = np.max(np.abs(c - tgt), axis=1) <= 1
        occl |= hit & ~own
    keep = ~occl
    if not keep.any():               # everything occluded: nearest ring
        keep = d[ci] <= dmin + 1.0
    out = np.zeros_like(seed)
    out[ci[keep]] = True
    return out, d


def _bone_field(a0, b0, walk, lo, vs, occ, iters=RELAX_IT):
    idx = np.argwhere(walk)
    if len(idx) == 0:
        return None
    pts = (idx + 0.5) * vs + lo
    seed, d0 = _bone_seed(a0, b0, idx, pts, vs, occ, lo, walk.shape)
    if not seed.any():
        return None
    f = np.full(walk.shape, INF)
    f[idx[seed, 0], idx[seed, 1], idx[seed, 2]] = d0[seed]
    return _relax(f, walk, iters)




# ------------------------------------------------- body-driven part classes --
def _body_classes(body):
    """Per-body-vertex part class from the body's OWN deform weights:
    0 torso/spine, 1 arm_l, 2 arm_r, 3 leg_l, 4 leg_r. Returns (verts_world,
    classes) or None when the body carries no usable groups."""
    import re
    n = len(body.data.vertices)
    if n == 0 or not body.vertex_groups:
        return None
    cls_of_group = {}
    for vg in body.vertex_groups:
        nm = vg.name.lower()
        side = "r" if re.search(r"[._-]r(?:$|[.\d])", nm) else \
               "l" if re.search(r"[._-]l(?:$|[.\d])", nm) else ""
        c = None
        if any(k in nm for k in ("upper_arm", "forearm", "hand",
                                 "shoulder", "clavicle")):
            c = 1 if side != "r" else 2
        elif any(k in nm for k in ("thigh", "shin", "foot", "toe", "leg")):
            c = 3 if side != "r" else 4
        elif any(k in nm for k in ("spine", "chest", "hips", "pelvis",
                                   "neck", "head", "breast", "torso")):
            c = 0
        if c is not None:
            cls_of_group[vg.index] = c
    if not cls_of_group:
        return None
    cls = np.full(n, -1, dtype=int)
    best = np.zeros(n)
    for v in body.data.vertices:
        for g in v.groups:
            c = cls_of_group.get(g.group)
            if c is not None and g.weight > best[v.index]:
                best[v.index] = g.weight
                cls[v.index] = c
    if (cls >= 0).sum() < 0.5 * n:
        return None
    from . import utils
    bw = utils.read_rest_coords(body)
    ok = cls >= 0
    return bw[ok], cls[ok]


def _worn_normals(g_ob, wco):
    """Vertex normals recomputed on the WORN positions (design normals are
    stale after the warp): area-weighted face-fan cross products."""
    me = g_ob.data
    n = len(wco)
    N = np.zeros((n, 3))
    if not len(me.polygons):
        return None
    li = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", li)
    tot = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get("loop_total", tot)
    start = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get("loop_start", start)
    # fan triangles (v0, vi, vi+1) per polygon
    for off in range(1, int(tot.max()) - 1):
        sel = tot > off + 1
        a = li[start[sel]]
        b = li[start[sel] + off]
        c = li[start[sel] + off + 1]
        fn = np.cross(wco[b] - wco[a], wco[c] - wco[a])
        np.add.at(N, a, fn)
        np.add.at(N, b, fn)
        np.add.at(N, c, fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(ln, 1e-12)


def _garment_classes(g_ob, wco, body):
    """Nearest-body-vertex class per garment vertex - NORMAL-FILTERED:
    fabric FACES the part it dresses, so the body point must lie on the
    fabric's inner side. Without this, an A-pose arm's skin is often
    euclidean-closer to the side-chest fabric than the torso's own skin
    and the arm class swallows the chest."""
    bc = _body_classes(body)
    if bc is None:
        return None
    bw, cls = bc
    nrm = _worn_normals(g_ob, wco)
    from mathutils import kdtree
    kd = kdtree.KDTree(len(bw))
    for i in range(len(bw)):
        kd.insert(bw[i], i)
    kd.balance()
    out = np.empty(len(wco), dtype=int)
    K = 10
    for i in range(len(wco)):
        best_j, best_s = -1, 1e18
        for (_co, j, d) in kd.find_n(wco[i], K):
            if nrm is not None:
                v = wco[i] - np.array(_co[:])
                lv = float(np.linalg.norm(v))
                inner = (float(v @ nrm[i]) / max(lv, 1e-9)) if lv > 1e-9 \
                    else 1.0
                # body point behind the fabric (inner>0) is trusted;
                # a point on the OUTER side is penalised heavily
                sc = d / max(inner, 0.15) if inner > 0.0 else d * 8.0
            else:
                sc = d
            if sc < best_s:
                best_s, best_j = sc, j
        out[i] = cls[best_j]
    return out


# ---------------------------------------------------------------- weights --
def _group_mask(g_ob, n, name, thresh=0.5):
    vg = g_ob.vertex_groups.get(name)
    if vg is None:
        return None
    m = np.zeros(n, dtype=bool)
    gi = vg.index
    for v in g_ob.data.vertices:
        for g in v.groups:
            if g.group == gi and g.weight > thresh:
                m[v.index] = True
                break
    return m if m.any() else None


def weights(g_ob, wco, jt, names, segs, loose=False):
    """(n, len(names)) part-aware voxel weights, or None on failure."""
    n = len(wco)
    if n == 0 or not names:
        return None
    ni = {nm: k for k, nm in enumerate(names)}
    lo, vs, dims = _grid(wco)
    occ = _occupancy(g_ob, wco, lo, vs, dims)

    # body: obstacle grid + semantic classes
    body = None
    try:
        import bpy
        bname = g_ob.get("srf_body")
        body = bpy.data.objects.get(bname) if bname else None
    except Exception:
        body = None
    block = np.zeros(dims, dtype=bool)
    gcls = None
    if body is not None:
        try:
            from . import utils
            bco = utils.read_rest_coords(body)
            keep = np.all((bco >= lo) & (bco < lo + vs * np.array(dims)),
                          axis=1)
            bc = _cells(bco[keep], lo, vs, dims)
            bocc = np.zeros(dims, dtype=bool)
            bocc[bc[:, 0], bc[:, 1], bc[:, 2]] = True
            block = bocc & ~occ
        except Exception:
            pass
        # EXPERIMENTAL (scene["srf_vb_bodyclass"]=True): body-weight
        # driven classes. Measured on the A-pose shirt: nearest-body is
        # ambiguous at the side chest even normal-filtered (arm classes
        # swallowed 58% incl. chest slabs) - needs votes + component
        # validation before it can be the default.
        gcls = None
        try:
            import bpy as _b
            if _b.context.scene.get("srf_vb_bodyclass"):
                gcls = _garment_classes(g_ob, wco, body)
        except Exception as e:
            print("Soulify VoxelBind classes:", e)
            gcls = None

    walk_all = _dilate(occ) & ~block
    vcell = _cells(wco, lo, vs, dims)

    # ---- domains -----------------------------------------------------------
    bones_of = {
        0: [b for b in ("spine1", "spine2") if b in ni],
        1: [b for b in ("arm_l", "fore_l") if b in ni],
        2: [b for b in ("arm_r", "fore_r") if b in ni],
        3: [b for b in ("leg_l", "shin_l") if b in ni],
        4: [b for b in ("leg_r", "shin_r") if b in ni],
    }
    domains = []
    if gcls is not None:
        for c, bones in bones_of.items():
            m = gcls == c
            if not m.any():
                continue
            if not bones:                # e.g. legs on a LOOSE column
                bones = bones_of[0] or list(names)
            domains.append((m, bones))
    else:
        # WORN-STATE domains (v1.36.0 final): in the WORN shape, with the
        # CHARACTER's joints (jt here = dst), the parts are unambiguous -
        # sleeve fabric WRAPS the arm (d_arm small), side panels HUG the
        # torso (rho small). The pre-warp radius mask both under-covered
        # the upper sleeve (kinks) and over-covered flared side panels
        # (dragged wings). Bias 0.55 keeps ambiguous fabric on the torso.
        def _segd(a, b):
            a = np.array(a[:]); b = np.array(b[:])
            ab = b - a
            l2 = float(ab @ ab) + 1e-12
            t_ = np.clip(((wco - a) @ ab) / l2, 0.0, 1.0)
            return np.linalg.norm(wco - (a + t_[:, None] * ab), axis=1)

        used = np.zeros(n, dtype=bool)
        if "pelvis" in jt and "neck" in jt:
            rho = _segd(jt["pelvis"], jt["neck"])
            pz = float(jt["pelvis"].z)
            for side in ("l", "r"):
                bones = [b for b in ("arm_" + side, "fore_" + side)
                         if b in ni]
                if not bones:
                    continue
                d_arm = None
                for a, b in (("shoulder_" + side, "elbow_" + side),
                             ("elbow_" + side, "wrist_" + side)):
                    if a in jt and b in jt:
                        d = _segd(jt[a], jt[b])
                        d_arm = d if d_arm is None else np.minimum(d_arm, d)
                if d_arm is None:
                    continue
                m = (d_arm < 0.55 * rho) & (wco[:, 2] > pz) & ~used
                if m.any():
                    domains.append((m, bones))
                    used |= m
            for side in ("l", "r"):
                bones = [b for b in ("leg_" + side, "shin_" + side)
                         if b in ni]
                if not bones or loose:
                    continue
                d_leg = None
                for a, b in (("hip_" + side, "knee_" + side),
                             ("knee_" + side, "ankle_" + side)):
                    if a in jt and b in jt:
                        d = _segd(jt[a], jt[b])
                        d_leg = d if d_leg is None else np.minimum(d_leg, d)
                if d_leg is None:
                    continue
                m = (d_leg < 0.55 * rho) & (wco[:, 2] < pz) & ~used
                if m.any():
                    domains.append((m, bones))
                    used |= m
        rest = ~used
        core = [b for b in ("spine1", "spine2") if b in ni] or list(names)
        domains.append((rest, core))
    if loose:
        # loose columns: leg-classed fabric still binds to the spine
        domains = [(m, [b for b in bones if not b.startswith(("leg",
                                                              "shin"))]
                    or (bones_of[0] or list(names)))
                   for (m, bones) in domains]

    # WIZARD-registered parts override the class of their verts
    try:
        auto = bool(g_ob.get("srf_auto_parts"))
    except Exception:
        auto = True
    if not auto:
        for nm, bones in (("SRF_Sleeve", None), ("SRF_Collar",
                          [b for b in ("spine2",) if b in ni]),
                          ("SRF_Lower",
                           [b for b in ("spine1",) if b in ni])):
            m = _group_mask(g_ob, n, nm)
            if m is None:
                continue
            if nm == "SRF_Sleeve":
                cx = float(jt["pelvis"].x) if "pelvis" in jt \
                    else float(wco[:, 0].mean())
                for side, mm in (("l", m & (wco[:, 0] < cx)),
                                 ("r", m & (wco[:, 0] >= cx))):
                    bb = [b for b in ("arm_" + side, "fore_" + side)
                          if b in ni]
                    if bb and mm.any():
                        domains = [(dm & ~mm, db) for dm, db in domains]
                        domains.append((mm, bb))
            elif bones:
                domains = [(dm & ~m, db) for dm, db in domains]
                domains.append((m, bones))
    domains = [(m, b) for m, b in domains if m.any() and b]
    if not domains:
        return None

    # ---- per-domain geodesic fields + feathered masks ----------------------
    W = np.zeros((n, len(names)))
    Msum = np.zeros(n)
    for vm, bones in domains:
        dom = np.zeros(dims, dtype=bool)
        c = vcell[vm]
        dom[c[:, 0], c[:, 1], c[:, 2]] = True
        dom = _dilate(dom) & walk_all
        reach = dom.copy()
        for _ in range(int(FEATHER)):
            reach = _dilate(reach) & walk_all
        f = np.full(dims, INF)
        f[dom] = 0.0
        f = _relax(f, reach, iters=int(FEATHER) + 2)
        dv = f[vcell[:, 0], vcell[:, 1], vcell[:, 2]]
        mk = np.clip(1.0 - dv / FEATHER, 0.0, 1.0) ** 2   # sharp fade
        if not (mk > 0).any():
            continue
        Wd = np.zeros((n, len(names)))
        for b in bones:
            k = ni[b]
            a0, b0 = segs[k]
            fld = _bone_field(np.asarray(a0, float),
                              np.asarray(b0, float), reach, lo, vs, occ)
            if fld is None:
                continue
            dgeo = fld[vcell[:, 0], vcell[:, 1], vcell[:, 2]]
            ok = dgeo < INF * 0.5
            Wd[ok, k] += (1.0 / (dgeo[ok] + 1.5)) ** 2
        sd = Wd.sum(axis=1, keepdims=True)
        good = sd[:, 0] > 1e-12
        Wd[good] /= sd[good]
        mk = np.where(good, mk, 0.0)
        W += mk[:, None] * Wd
        Msum += mk

    lost = Msum < 1e-6
    if lost.any():
        best_d = np.full(n, np.inf)
        best_k = np.zeros(n, dtype=int)
        for k, (a0, b0) in enumerate(segs):
            a0 = np.asarray(a0, float)
            b0 = np.asarray(b0, float)
            ab = b0 - a0
            L2 = float(ab @ ab) + 1e-12
            t = np.clip(((wco - a0) @ ab) / L2, 0.0, 1.0)
            d = np.linalg.norm(wco - (a0 + t[:, None] * ab), axis=1)
            upd = d < best_d
            best_d[upd] = d[upd]
            best_k[upd] = k
        W[lost] = 0.0
        W[lost, best_k[lost]] = 1.0

    # MESH-GRAPH SMOOTHING (the vhd 'diffuse loops' equivalent): voxel
    # cells quantise the weights - posed, that shows as rigid SLABS.
    # A few Laplacian passes over the vertex adjacency dissolve them.
    try:
        me = g_ob.data
        if len(me.edges):
            ev = np.empty(2 * len(me.edges), dtype=np.int64)
            me.edges.foreach_get("vertices", ev)
            ev = ev.reshape(-1, 2)
            deg = np.zeros(n)
            np.add.at(deg, ev[:, 0], 1.0)
            np.add.at(deg, ev[:, 1], 1.0)
            deg = np.maximum(deg, 1.0)[:, None]
            for _ in range(3):
                acc = np.zeros_like(W)
                np.add.at(acc, ev[:, 0], W[ev[:, 1]])
                np.add.at(acc, ev[:, 1], W[ev[:, 0]])
                W = 0.5 * W + 0.5 * (acc / deg)
    except Exception as e:
        print("Soulify VoxelBind smooth:", e)

    if W.shape[1] > 4:
        idx = np.argsort(W, axis=1)[:, :-4]
        np.put_along_axis(W, idx, 0.0, axis=1)
    ssum = W.sum(axis=1, keepdims=True)
    W /= np.maximum(ssum, 1e-12)
    return W
