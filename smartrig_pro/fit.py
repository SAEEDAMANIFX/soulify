import bpy
import numpy as np
from mathutils import Vector
from . import utils, markers


# --------------------------------------------------------------- geometry helpers
def _volume_center_y(co, xc, zc, h):
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    tol = 0.04 * h
    m = (np.abs(x - xc) < tol) & (np.abs(z - zc) < tol)
    if m.sum() < 3:
        return float(y.mean())
    ys = y[m]
    return 0.5 * (float(ys.min()) + float(ys.max()))


def _torso_center(co, zc, h, fallback):
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    m = (np.abs(z - zc) < 0.04 * h) & (np.abs(x) < 0.20)
    if m.sum() > 10:
        ys = y[m]
        ymid = 0.5 * (float(np.percentile(ys, 2)) + float(np.percentile(ys, 98)))
        return Vector((float(np.median(x[m])), ymid, zc))
    return fallback


def _user_moved_y(name, cur):
    """True ONLY when the user DRAGGED this marker in Y after it was placed.
    Placement (click / auto-detect) records its chosen Y in `sr_place_y`;
    markers without the tag (old scenes) count as un-moved, so the fit falls
    back to the auto volume/torso centre - never the raw surface click."""
    o = markers.get_marker(name)
    if o is None:
        return False
    py = o.get("sr_place_y", None)
    if py is None:
        return False
    return abs(float(cur.y) - float(py)) > 1e-4


def _leg_depth_y(co, zc, h, sign, fallback):
    """Depth (Y) centre of ONE leg's cross-section at height zc."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    m = (np.abs(z - zc) < 0.02 * h) & (x * sign > 0.015 * h) & (x * sign < 0.15 * h)
    if m.sum() > 5:
        return float(y[m].mean())
    return float(fallback)


def _limb_center(co, p0, p1, f, radius=0.10):
    p0 = np.array(p0, dtype=np.float64)
    p1 = np.array(p1, dtype=np.float64)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-6:
        return None
    an = axis / L
    cpt = p0 + f * axis
    d = co - cpt
    al = d @ an
    perp = np.linalg.norm(d - np.outer(al, an), axis=1)
    m = (np.abs(al) < 0.05 * L) & (perp < radius)
    if m.sum() > 5:
        return Vector([float(v) for v in co[m].mean(0)])
    return None


def _leg_centerline(co, ground, h, sign):
    """Fit a line through one leg's cross-section centers; return (hip, knee)."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    side = (x * sign) > 0.02 * h
    zs, xs, ys = [], [], []
    for frac in np.linspace(0.38, 0.15, 6):
        zc = ground + frac * h
        m = side & (np.abs(z - zc) < 0.03 * h)
        if m.sum() > 3:
            zs.append(zc); xs.append(float(x[m].mean())); ys.append(float(y[m].mean()))
    hip_z = ground + 0.53 * h
    knee_z = ground + 0.285 * h
    if len(zs) >= 2:
        px = np.polyfit(zs, xs, 1); py = np.polyfit(zs, ys, 1)
        hip = Vector((float(np.polyval(px, hip_z)), float(np.polyval(py, hip_z)), hip_z))
        knee = Vector((float(np.polyval(px, knee_z)), float(np.polyval(py, knee_z)), knee_z))
    else:
        lx = sign * 0.12 * h; ly = float(y.mean())
        hip = Vector((lx, ly, hip_z)); knee = Vector((lx, ly, knee_z))
    return hip, knee


def _foot_points(co, ground, h, ankle):
    """DETECT the real foot direction from the mesh and return (ball, toe_tip).
    The toe tip is the farthest foot vertex FORWARD of the ankle in the ground
    plane (so the bones follow the foot even when the toes splay outward), and the
    ball sits ~60% along the ankle->toe line. Heel (behind the ankle) is ignored."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    leg_x = float(ankle.x)
    ank_y = float(ankle.y)
    # all foot verts low to the ground near this leg's X
    m = (z < ground + 0.16 * h) & (np.abs(x - leg_x) < 0.14 * h)
    sel = co[m]

    def med_z(px, py, r=0.05 * h):
        """height just above the SOLE at (px,py), kept inside the foot. Using the local
        sole (column min) + a small rise avoids the tall instep pulling the ball up and
        out the top - the foot bone then points down-forward into the foot."""
        if len(sel):
            nb = sel[(np.abs(sel[:, 0] - px) < r) & (np.abs(sel[:, 1] - py) < r)]
            if len(nb) >= 3:
                zlo = float(nb[:, 2].min()); zhi = float(nb[:, 2].max())
                return zlo + min(0.02 * h, 0.5 * (zhi - zlo))
        return ground + 0.025 * h

    if sel.shape[0] >= 5:
        fwd = sel[sel[:, 1] < ank_y + 0.02 * h]      # forward of the ankle (toward -Y)
        if fwd.shape[0] < 3:
            fwd = sel
        a2 = np.array([leg_x, ank_y])
        d2 = np.linalg.norm(fwd[:, :2] - a2, axis=1)
        thr = np.percentile(d2, 94)
        tip = fwd[d2 >= thr].mean(0)
        # pull the tip back 12% toward the ankle so the bone END stays INSIDE the toe
        tx = leg_x + 0.88 * (float(tip[0]) - leg_x)
        ty = ank_y + 0.88 * (float(tip[1]) - ank_y)
        toe = Vector((tx, ty, med_z(tx, ty)))
    else:
        toe = Vector((leg_x, float(y.min()) + 0.02 * h, ground + 0.04 * h))
    # ball: ~60% along the ankle->toe line, at the foot's medial height
    bx = leg_x + 0.60 * (toe.x - leg_x)
    by = ank_y + 0.60 * (toe.y - ank_y)
    ball = Vector((bx, by, med_z(bx, by)))
    return ball, toe


def _hand_tip(co, h, wrist, hand_dir):
    """Find the real end of the hand mesh (past the wrist along hand_dir),
    centred. Used as the hand bone tail when there are no finger markers."""
    w = np.array(wrist, dtype=np.float64)
    a = np.array(hand_dir, dtype=np.float64)
    d = co - w
    t = d @ a                                       # distance along hand dir from wrist
    perp = np.linalg.norm(d - np.outer(t, a), axis=1)
    m = (t > 0.0) & (t < 0.32 * h) & (perp < 0.10 * h)
    if m.sum() < 5:
        return Vector(wrist) + Vector(hand_dir) * (0.10 * h)
    tt = t[m]
    far = co[m][tt >= np.percentile(tt, 92)]        # the farthest cluster = fingertips
    return Vector([float(v) for v in far.mean(0)])


# --------------------------------------------------------------- skeleton compute
def compute_joints(props):
    mesh = props.target_mesh
    co = utils.read_world_coords(mesh)
    z = co[:, 2]
    ground, top = float(z.min()), float(z.max())
    h = top - ground

    def mk(name):
        o = markers.get_marker(name)
        return Vector(o.matrix_world.translation) if o else None

    m_root = mk("spine_root"); m_neck = mk("neck"); m_head = mk("head_top")
    m_sh = mk("shoulder.L"); m_wr = mk("wrist.L"); m_ank = mk("ankle.L")
    m_hip = mk("hip.L")
    m_elbow = mk("elbow.L"); m_knee = mk("knee.L")     # optional precise joints
    missing = [n for n, v in [("spine_root", m_root), ("neck", m_neck),
               ("head_top", m_head), ("shoulder.L", m_sh), ("wrist.L", m_wr),
               ("ankle.L", m_ank), ("hip.L", m_hip)] if v is None]
    if missing:
        return None, "Missing markers: " + ", ".join(missing), h

    J = {}
    # --- axial chain
    # Honor an axial marker's Y ONLY when the user really DRAGGED it after
    # placement (sr_place_y provenance tag, see _user_moved_y) - a plain click
    # always re-centres in depth. Clicked markers land near the surface, so
    # trusting every different Y (old way) glued the spine to the belly.
    _auto_pelvis_y = _volume_center_y(co, 0.0, m_root.z, h)
    pelvis_y = m_root.y if _user_moved_y("spine_root", m_root) else _auto_pelvis_y
    pelvis = Vector((0.0, pelvis_y, m_root.z))
    _auto_neck_y = _torso_center(co, m_neck.z, h, Vector((0.0, pelvis.y, m_neck.z))).y
    neck_y = m_neck.y if _user_moved_y("neck", m_neck) else _auto_neck_y
    neck_base = Vector((0.0, neck_y, m_neck.z))
    head_base = Vector((0.0, neck_base.y, neck_base.z + 0.4 * (m_head.z - neck_base.z)))
    J["pelvis"] = pelvis
    J["neck_base"] = neck_base
    J["head_base"] = head_base
    # head_top honors the marker's Y (tilt); default marker Y == head_base.y
    head_top_y = m_head.y if _user_moved_y("head_top", m_head) else head_base.y
    J["head_top"] = Vector((0.0, head_top_y, m_head.z))
    J["root"] = Vector((0.0, pelvis.y, ground))

    # ---- breast bones: auto-detect each breast APEX from the mesh ----
    # In the upper-chest Z band, on each side, the most forward-protruding vertex
    # is the breast tip -> bone tail; the head sits behind+inboard on the chest.
    # Falls back (leaves the stock bone) for flat/male chests where no apex stands
    # out, so it never makes a non-breasted character worse.
    ymed_co = float(np.median(co[:, 1]))
    blo = pelvis.z + 0.60 * (neck_base.z - pelvis.z)
    bhi = pelvis.z + 0.92 * (neck_base.z - pelvis.z)

    def _breast_bone(side_sign):
        if side_sign > 0:
            xlo, xhi = 0.03 * h, 0.45 * h
        else:
            xlo, xhi = -0.45 * h, -0.03 * h
        msk = ((co[:, 2] > blo) & (co[:, 2] < bhi) &
               (co[:, 0] > xlo) & (co[:, 0] < xhi) & (co[:, 1] < ymed_co))
        sub = co[msk]
        if len(sub) < 8:
            return None
        apex = sub[np.argmin(sub[:, 1])]                 # most forward (min Y)
        tail = Vector((float(apex[0]), float(apex[1]), float(apex[2])))
        # require a real protrusion in front of the chest centre, else skip
        if (ymed_co - tail.y) < 0.02 * h:
            return None
        # STRAIGHT horizontal bone like stock Rigify: head sits directly behind
        # the apex (SAME X, SAME Z) so the bone points straight forward (-Y) with
        # no downward or sideways tilt.
        head = Vector((tail.x, ymed_co + 0.02 * h, tail.z))
        return (head, tail)
    _bl = _breast_bone(1)
    _br = _breast_bone(-1)
    if _bl:
        J["breast.L"] = _bl
    if _br:
        J["breast.R"] = _br

    n = props.spine_count
    spine_pts = []
    for i in range(n + 1):
        t = i / n
        p = pelvis.lerp(neck_base, t)
        p = _torso_center(co, p.z, h, p)
        spine_pts.append(p)
    J["spine_pts"] = spine_pts

    # --- left arm : respect the user's precise markers (they are already
    # centred inside the mesh by the click raycast). Don't shift them.
    sh_joint = m_sh.copy()
    elbow = m_elbow.copy() if m_elbow is not None else (_limb_center(co, m_sh, m_wr, 0.50) or m_sh.lerp(m_wr, 0.5))
    wrist = m_wr.copy()
    arm_dir = (wrist - elbow); arm_dir = arm_dir.normalized() if arm_dir.length > 1e-6 else Vector((1, 0, 0))
    hand_tip = _hand_tip(co, h, wrist, arm_dir)         # real end of the hand mesh
    # clavicle: from the sternum (centre, just under the neck) out to the shoulder
    clav_head = Vector((0.02 * h, neck_base.y, sh_joint.z + 0.02 * h))
    J["clavicle.L"] = (clav_head, sh_joint)
    J["upper_arm.L"] = (sh_joint, elbow)
    J["forearm.L"] = (elbow, wrist)
    J["hand.L"] = (wrist, hand_tip)

    # --- left leg (hip from the user marker; knee from the leg centerline fit)
    hip_fit, knee = _leg_centerline(co, ground, h, +1)
    hip = m_hip.copy()
    if not _user_moved_y("hip.L", m_hip):
        hip.y = _leg_depth_y(co, hip.z, h, +1, hip.y)   # depth-centre in the leg
    if abs(knee.z - hip.z) < 0.05 * h or knee.z > hip.z:
        knee = hip.lerp(m_ank, 0.5)
    if m_knee is not None:                      # user-placed knee wins
        knee = m_knee.copy()
    ankle = m_ank.copy()
    # foot: if the user placed the 2 TOP-view foot markers, use them (precise);
    # else auto-detect ball + toe tip from the mesh.
    m_ball = mk("ball.L"); m_ftip = mk("foottip.L")
    if m_ball is not None and m_ftip is not None:
        ball, toe_tip = m_ball.copy(), m_ftip.copy()
    else:
        ball, toe_tip = _foot_points(co, ground, h, ankle)
    # real heel extent from the mesh: rear + width of the foot at ground level
    # (heel.02 was placed by a fixed ank.y+0.06h formula that floats BEHIND
    # short/stubby feet and spans a hardcoded width)
    _fm = ((co[:, 2] < ground + 0.04 * h) &
           (np.abs(co[:, 0] - ankle.x) < 0.10 * h) &
           (co[:, 0] > 0.015 * h))          # LEFT foot only - never the other foot

    if _fm.sum() > 5:
        _fs = co[_fm]
        J["heel_back_y.L"] = float(np.percentile(_fs[:, 1], 99))
        J["heel_x.L"] = (float(np.percentile(_fs[:, 0], 3)),
                         float(np.percentile(_fs[:, 0], 97)))
    J["thigh.L"] = (hip, knee)
    J["shin.L"] = (knee, ankle)
    J["foot.L"] = (ankle, ball)         # ankle -> ball of the foot
    J["toe.L"] = (ball, toe_tip)        # ball -> real toe tip (on the mesh)

    # --- fingers (left): MANUAL markers win (reliable on any hand); else auto-detect
    fingers = {}
    from . import fingers_manual
    hand_m = fingers_manual.manual_chains_world("hand", "L")
    foot_m = fingers_manual.manual_chains_world("foot", "L")
    palm_m = fingers_manual.manual_chains_world("palm", "L")
    if hand_m or foot_m or palm_m:
        if hand_m:
            J["fingers_manual"] = hand_m
        if foot_m:
            J["toes_manual"] = foot_m
        if palm_m:
            J["palm_manual"] = palm_m
        nfg = 0                                          # manual wins, skip auto
    elif getattr(props, "auto_fingers", False):
        nfg = int(getattr(props, "finger_count", 5))     # auto only if explicitly enabled
    else:
        nfg = 0                                          # default: NO auto fingers
    if nfg > 0:
        side_z = 1.0 if wrist.x >= 0 else -1.0
        tmul = float(getattr(props, "finger_thickness", 1.0))

        def _score(cand):
            # number of finger bones that came out a sensible length
            return sum(1 for js in cand.values() if (js[0] - js[3]).length > 0.02 * h)

        # AUTO-TUNE the voxel parameters (what ARP/Mixamo do internally): try a few
        # thickness/precision pairs from coarse->fine and keep the cleanest result.
        best = {}; best_s = -1
        for th, pr in ((1.0, 6), (0.8, 7), (0.65, 8), (0.5, 9), (0.4, 10)):
            try:
                cand = _voxel_fingers(mesh, h, wrist, elbow, nfg, th * tmul, pr, side_z)
            except Exception as e:
                print("SmartRig: voxel try failed:", e); cand = {}
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
            if s >= nfg:                      # found them all - stop searching
                break
        fingers = best
        # RENDER-based neural candidate (models/finger_kp.onnx): the robust ARP-style
        # path - canonical palm-normal render -> 2D keypoints -> back-project. Handles
        # down-hands / stylized meshes where voxel fails. Highest priority when ready.
        try:
            from . import finger_render_ai
            cand = finger_render_ai.detect(mesh, co, h, wrist, elbow, side_z)
            if _score(cand) >= best_s:
                best_s, fingers = _score(cand), cand
                print("SmartRig: render-based finger model used (score %d)" % best_s)
        except Exception as e:
            print("SmartRig: render finger model failed:", e)
        # point-cloud neural candidate (models/finger.onnx): secondary calibrator.
        try:
            from . import finger_ai
            cand = finger_ai.ai_fingers(mesh, co, h, wrist, elbow, side_z)
            if _score(cand) > best_s:
                best_s, fingers = _score(cand), cand
                print("SmartRig: point-cloud finger model used (score %d)" % best_s)
        except Exception as e:
            print("SmartRig: AI fingers failed:", e)
        # fallbacks: surface topology, then point-cloud (if voxel under-performed)
        if best_s < max(2, nfg - 1):
            ftips = _detect_fingertips_auto(co, h, sh_joint, wrist)
            if ftips:
                for fn in (lambda: _topo_fingers(mesh, h, wrist, elbow, ftips, side_z),
                           lambda: _compute_fingers(co, h, elbow, wrist, ftips)):
                    try:
                        cand = fn()
                    except Exception:
                        cand = {}
                    if _score(cand) > best_s:
                        best_s, fingers = _score(cand), cand
    J["fingers"] = fingers
    # --- toes (left): only auto-detect if explicitly enabled (default OFF)
    J["toes"] = (_detect_toes_auto(co, ground, h, ankle, ball, toe_tip)
                 if getattr(props, "auto_fingers", False) else {})
    return J, None, h


def _finger_trace(P, C, co, h, steps=14):
    """Trace from fingertip P toward palm centre C, snapping to the finger's
    medial line. Returns [tip, j1, j2, base] (4 joints) and the traced length."""
    r = 0.022 * h
    P = np.array(P, dtype=np.float64); C = np.array(C, dtype=np.float64)
    pts = [P.copy()]; cur = P.copy()
    for _ in range(steps):
        to = C - cur; dist = np.linalg.norm(to)
        if dist < 0.012:
            break
        cur = cur + to / dist * min(0.012, dist * 0.5)
        near = np.linalg.norm(co - cur, axis=1) < r
        if near.sum() >= 4:
            cur = 0.5 * cur + 0.5 * co[near].mean(0)
        pts.append(cur.copy())
    pts = np.array(pts)
    base = pts[-1].copy()                          # where the trace met the palm
    axis = base - P
    axL = float(np.linalg.norm(axis))
    if axL < 1e-5:
        return [Vector(P)] * 4, 0.0
    axn = axis / axL
    # MONOTONIC + SMOOTH: sort every traced point by how far it is ALONG the
    # tip->base axis, so a finger that curls (or a wobbly medial snap) can never
    # produce a bone that reverses direction (the old zig-zag that tore fists).
    proj = (pts - P) @ axn
    order = np.argsort(proj)
    spts = pts[order]; sproj = proj[order]
    # resample 4 joints at even ALONG-AXIS fractions, taking the medial point
    # nearest each target depth (keeps the gentle real curve, no fold-back).
    js = []
    for fr in (0.0, 1 / 3, 2 / 3, 1.0):
        target = fr * axL
        k = int(np.searchsorted(sproj, target))
        k = max(0, min(k, len(spts) - 1))
        js.append(Vector([float(v) for v in spts[k]]))
    # gentle smoothing of the two middle joints toward the straight line so a
    # noisy medial can not kink the chain
    for i in (1, 2):
        straight = P + axn * (i / 3.0 * axL)
        j = np.array(js[i])
        js[i] = Vector([float(v) for v in (0.6 * j + 0.4 * straight)])
    return js, axL        # [tip, j1, j2, base]


def _voxel_fingers(obj, h, wrist, elbow, n_fingers, thick_mult, precision, side_z):
    """VOXEL finger detection (the Auto-Rig-Pro principle): fill the hand VOLUME
    into a 3D voxel grid (inside/outside via ray-cast), find `n_fingers` tips as the
    most distal volume points, then watershed the volume to each tip (separates even
    touching fingers down the middle) and ride each finger's centre line. Tunable by
    finger thickness + voxel precision, exactly like ARP. Returns {name:[tip,j1,j2,base]}.
    """
    try:
        from mathutils.bvhtree import BVHTree
        import bmesh
        import heapq
    except Exception:
        return {}
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    bm = bmesh.new(); bm.from_mesh(ev.to_mesh()); bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    cw = np.array([list(v.co) for v in bm.verts], dtype=np.float64)
    bm.free(); ev.to_mesh_clear()
    if cw.shape[0] < 20:
        return {}
    wn = np.array(wrist); dn = np.array((wrist - elbow).normalized())
    rel = cw - wn; proj = rel @ dn; dist = np.linalg.norm(rel, axis=1)
    hm = (proj > 0.0) & (dist < 0.20 * h) & (cw[:, 0] * side_z > 0)
    H = cw[hm]
    if H.shape[0] < 20:
        return {}
    thickness = 0.013 * h * float(thick_mult)
    cell = thickness / (float(precision) * 0.8 + 1.0)
    lo = H.min(0) - thickness * 0.6; hi = H.max(0) + thickness * 0.6
    dims = ((hi - lo) / cell).astype(int) + 1
    if int(np.prod(dims)) > 500000:
        return {}

    def inside(px, py, pz):
        o = Vector((px, py, pz)); d = Vector((1.0, 0.0, 0.0)); c = 0
        for _ in range(24):
            hit = bvh.ray_cast(o, d)
            if hit[0] is None:
                break
            c += 1; o = hit[0] + d * 1e-5
        return c % 2 == 1

    occ = np.zeros(tuple(dims), bool)
    for ix in range(dims[0]):
        px = lo[0] + (ix + 0.5) * cell
        for iy in range(dims[1]):
            py = lo[1] + (iy + 0.5) * cell
            for iz in range(dims[2]):
                if inside(px, py, lo[2] + (iz + 0.5) * cell):
                    occ[ix, iy, iz] = True
    filled = np.argwhere(occ); N = len(filled)
    if N < 20:
        return {}
    centers = lo + (filled + 0.5) * cell
    vidx = {tuple(f): i for i, f in enumerate(filled)}
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if any((a, b, c))]
    adj = [[] for _ in range(N)]
    for i in range(N):
        f = filled[i]
        for o in offs:
            j = vidx.get((f[0] + o[0], f[1] + o[1], f[2] + o[2]))
            if j is not None:
                adj[i].append((j, float(np.linalg.norm(o)) * cell))
    dw = np.linalg.norm(centers - wn, axis=1)
    order = np.argsort(-dw); tips = []; supp = thickness * 1.7
    for idx in order:
        if all(np.linalg.norm(centers[idx] - centers[t]) > supp for t in tips):
            tips.append(int(idx))
        if len(tips) >= n_fingers:
            break
    if len(tips) < 2:
        return {}
    INF = 1e18
    own = np.full(N, -1); dd = np.full(N, INF); pq = []
    for k, t in enumerate(tips):
        dd[t] = 0.0; own[t] = k; heapq.heappush(pq, (0.0, t))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dd[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dd[v]:
                dd[v] = nd; own[v] = own[u]; heapq.heappush(pq, (nd, v))
    wv = int(np.argmin(dw)); gd = np.full(N, INF); gd[wv] = 0.0; pq = [(0.0, wv)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > gd[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < gd[v]:
                gd[v] = nd; heapq.heappush(pq, (nd, v))
    tip_gd = np.array([gd[t] if gd[t] < INF else dw[t] for t in tips])
    valid = tip_gd[tip_gd < INF]
    med = float(np.median(valid)) if valid.size else 0.10 * h
    flen = 0.42 * med
    T = centers[tips]
    if len(T) >= 3:
        nn = np.array([min(np.linalg.norm(T[i] - T[j]) for j in range(len(T)) if j != i)
                       for i in range(len(T))])
        thumb_i = int(np.argmax(nn)) if nn.max() > max(0.045 * h, 1.5 * float(np.median(nn))) else -1
    else:
        thumb_i = -1
    spread = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ dn) * dn
    ns = np.linalg.norm(spread); spread = spread / ns if ns > 1e-6 else np.array([1.0, 0.0, 0.0])
    fingers_order = [i for i in range(len(tips)) if i != thumb_i]
    fingers_order.sort(key=lambda i: (centers[tips[i]] - wn) @ spread)
    names = (["thumb"] if thumb_i >= 0 else []) + ["index", "middle", "ring", "pinky"][:len(fingers_order)]
    klist = ([thumb_i] if thumb_i >= 0 else []) + fingers_order
    dnv = Vector([float(v) for v in dn])
    out = {}
    for nm, k in zip(names, klist):
        tip = Vector([float(v) for v in centers[tips[k]]])
        base_gd = tip_gd[k] - flen
        members = np.where((own == k) & (gd >= base_gd - cell) & (gd < INF))[0]
        if tip_gd[k] >= INF or len(members) < 4:
            base = tip - dnv * flen
            out[nm] = [tip, tip.lerp(base, 0.27), tip.lerp(base, 0.55), base]
            continue
        g = gd[members]; pts = centers[members]
        line = []
        for fr in (0.0, 0.45, 0.73):
            gc = base_gd + fr * (tip_gd[k] - base_gd)
            m = np.abs(g - gc) < max(cell * 2, (tip_gd[k] - base_gd) * 0.25)
            line.append(Vector([float(v) for v in (pts[m].mean(0) if m.sum() >= 1 else centers[tips[k]])]))
        base, j2, j1 = line
        out[nm] = [tip, j1, j2, base]
    return out


def _topo_fingers(obj, h, wrist, elbow, fingertip_dict, side_z):
    """TOPOLOGY-based finger detection (the ARP principle): segment the fingers by
    GEODESIC distance over the mesh surface, not by distance in space. Touching
    fingers stay separate because their surfaces aren't welded, so the on-surface
    distance between them is large. Returns {name: [tip, j1, j2, base]} or {}.
    """
    import heapq
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    nv = len(me.vertices)
    if nv < 8 or len(me.edges) < 8:
        ev.to_mesh_clear()
        return {}
    co = np.empty(nv * 3, dtype=np.float64); me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3)
    ed = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    ev.to_mesh_clear()
    mw = np.array(obj.matrix_world)
    cow = (np.column_stack([co, np.ones(nv)]) @ mw.T)[:, :3]

    wn = np.array(wrist); dn = np.array((wrist - elbow).normalized())
    rel = cow - wn
    proj = rel @ dn
    dist = np.linalg.norm(rel, axis=1)
    handmask = (proj > -0.05 * h) & (dist < 0.24 * h) & (cow[:, 0] * side_z > 0)
    idx = np.where(handmask)[0]
    if len(idx) < 12:
        return {}
    g2l = {int(g): i for i, g in enumerate(idx)}
    Lco = cow[idx]
    adj = [[] for _ in range(len(idx))]
    for a, b in ed:
        a = int(a); b = int(b)
        if a in g2l and b in g2l:
            la, lb = g2l[a], g2l[b]
            w = float(np.linalg.norm(cow[a] - cow[b]))
            adj[la].append((lb, w)); adj[lb].append((la, w))

    def dijkstra(seeds):
        INF = 1e18
        d = [INF] * len(idx); own = [-1] * len(idx); pq = []
        for oi, s in seeds:
            if d[s] > 0:
                d[s] = 0.0; own[s] = oi; heapq.heappush(pq, (0.0, s))
        while pq:
            du, u = heapq.heappop(pq)
            if du > d[u]:
                continue
            for v, w in adj[u]:
                nd = du + w
                if nd < d[v]:
                    d[v] = nd; own[v] = own[u]; heapq.heappush(pq, (nd, v))
        return d, own

    names = list(fingertip_dict.keys())
    tip_local = [int(np.argmin(np.linalg.norm(Lco - np.array(t), axis=1))) for t in fingertip_dict.values()]
    # owner = which fingertip each vertex belongs to (geodesically nearest)
    _d, owner = dijkstra(list(enumerate(tip_local)))
    # geodesic distance from the wrist seed (for ordering base->tip within a finger)
    wseed = int(np.argmin(np.linalg.norm(Lco - wn, axis=1)))
    gd, _o = dijkstra([(0, wseed)])

    out = {}
    INF = 1e18
    for ti, fn in enumerate(names):
        sel = [i for i in range(len(idx)) if owner[i] == ti and gd[i] < INF]
        if len(sel) < 4:
            continue
        sel = np.array(sel)
        g = np.array([gd[i] for i in sel])
        pts = Lco[sel]
        gmin, gmax = float(g.min()), float(g.max())
        if gmax - gmin < 0.01 * h:
            continue
        joints = []
        ok = True
        # anatomical phalange proportions from the knuckle: proximal ~45%,
        # middle ~28%, distal ~27% (CGDive "Finger Placement" - joints sit at the
        # real knuckles, not at equal thirds)
        for fr in (0.0, 0.45, 0.73):                    # base(knuckle), PIP, DIP
            gc = gmin + fr * (gmax - gmin)
            m = np.abs(g - gc) < max(0.018 * h, (gmax - gmin) * 0.18)
            if m.sum() >= 2:
                joints.append(Vector([float(v) for v in pts[m].mean(0)]))
            else:
                ok = False; break
        if not ok:
            continue
        base, j2, j1 = joints
        tip = Vector([float(v) for v in fingertip_dict[fn]])
        out[fn] = [tip, j1, j2, base]
    return out


def _trace_finger_cylinder(tip, fco, down, knuckle_proj, wrist, h):
    """Follow ONE finger as a cylinder: march from the tip toward the knuckle, and
    at each step recentre on the mean of nearby vertices within the finger radius.
    Stays inside the finger tube even when fingers are touching. Returns
    [tip, j1, j2, base] or None."""
    dn = np.array(down, dtype=np.float64); wn = np.array(wrist, dtype=np.float64)
    cur = np.array(tip, dtype=np.float64)
    r = 0.028 * h
    pts = [cur.copy()]
    for _ in range(18):
        if float((cur - wn) @ dn) <= knuckle_proj:
            break
        step = cur - dn * (0.016 * h)                  # toward the knuckle
        near = fco[np.linalg.norm(fco - step, axis=1) < r]
        cur = (0.35 * step + 0.65 * near.mean(0)) if len(near) >= 2 else step
        pts.append(cur.copy())
    pts = np.array(pts)
    if len(pts) < 2:
        return None
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = float(seg.sum())
    if L < 1e-5:
        return None
    cum = np.concatenate([[0], np.cumsum(seg)])
    js = []
    for fr in (0.0, 0.27, 0.55, 1.0):                  # tip, DIP, PIP, base (anatomical)
        idx = min(int(np.searchsorted(cum, fr * L)), len(pts) - 1)
        js.append(Vector([float(v) for v in pts[idx]]))
    return [js[0], js[1], js[2], js[3]]                # [tip, j1, j2, base]


def _compute_fingers(co, h, elbow, wrist, fingers_markers):
    """Build clean, STRAIGHT, PARALLEL finger chains. The 4 fingers each run back
    from their tip along the hand axis to their own knuckle (so they don't fan out
    to a single palm point). The thumb aims at its wrist-side base. Returns
    {name: [tip, j1, j2, base]}."""
    a = (wrist - elbow)
    if a.length < 1e-6:
        return {}
    down = a.normalized()                              # hand axis: wrist -> fingertips
    side_z = 1.0 if wrist.x >= 0 else -1.0
    d = co - np.array(wrist)
    t = d @ np.array(down)
    perp = np.linalg.norm(d - np.outer(t, np.array(down)), axis=1)
    hand = (t > -0.02 * h) & (t < 0.24 * h) & (perp < 0.11 * h) & (co[:, 0] * side_z > 0)
    if hand.sum() < 10:
        palm = wrist
    else:
        hco = co[hand]; ht = t[hand]
        palm = Vector([float(v) for v in hco[ht < np.percentile(ht, 30)].mean(0)])
    dn = np.array(down); wn = np.array(wrist)
    rel = co - wn
    proj = rel @ dn
    perp = rel - np.outer(proj, dn)                     # lateral component vectors
    nonthumb = [(fn, np.array(t, dtype=np.float64)) for fn, t in fingers_markers.items() if fn != "thumb"]
    tip_projs = [float((t - wn) @ dn) for _fn, t in nonthumb]
    med_tip = float(np.median(tip_projs)) if tip_projs else 0.12 * h
    knuckle = 0.5 * med_tip                             # where the fingers meet the palm
    finger_len = 0.34 * med_tip
    thumb_base = wrist.lerp(palm, 0.4)
    out = {}

    # THUMB: a clean straight chain aimed at its wrist-side base
    if "thumb" in fingers_markers:
        tip = fingers_markers["thumb"]
        direction = (thumb_base - tip)
        direction = direction.normalized() if direction.length > 1e-6 else (down * -1.0)
        base = tip + direction * (0.7 * med_tip)
        out["thumb"] = [tip, tip.lerp(base, 1.0 / 3), tip.lerp(base, 2.0 / 3), base]

    # FINGERS: segment the distal hand and follow each finger's own centre line.
    # Every distal vertex is assigned to its nearest fingertip (laterally); each
    # finger's bones then ride the medial line of just that finger's vertices, so
    # the bones stay INSIDE the correct finger even on bunched hands.
    dmask = (proj > 0.8 * knuckle) & (proj < med_tip + 0.07 * h) & \
            (np.linalg.norm(perp, axis=1) < 0.13 * h) & (co[:, 0] * side_z > 0)
    D = co[dmask]; Dp = proj[dmask]; Dpe = perp[dmask]
    seg_ok = nonthumb and D.shape[0] >= 8
    if seg_ok:
        tip_lat = np.array([(t - wn) - (((t - wn) @ dn) * dn) for _fn, t in nonthumb])
        assign = np.argmin(np.linalg.norm(Dpe[:, None, :] - tip_lat[None, :, :], axis=2), axis=1)
    for fi, (fn, tnp) in enumerate(nonthumb):
        tip = Vector([float(v) for v in tnp])
        chain = None
        if seg_ok:
            sel = assign == fi
            if sel.sum() >= 4:
                chain = _trace_finger_cylinder(tnp, D[sel], down, knuckle, wrist, h)
        if chain is None:                                # fallback: straight back
            base = tip - down * finger_len
            chain = [tip, tip.lerp(base, 1.0 / 3), tip.lerp(base, 2.0 / 3), base]
        out[fn] = chain
    return out


def _detect_fingertips_auto(co, h, shoulder, wrist):
    """Detect fingertips straight from the hand mesh (no markers). Returns
    {finger_name: Vector tip} ordered thumb->pinky, auto count (2..5). Returns {}
    on a closed fist / unresolvable hand so fingers are simply skipped."""
    wr = np.array(wrist, dtype=np.float64)
    sh = np.array(shoulder, dtype=np.float64)
    down = wr - sh
    nd = np.linalg.norm(down)
    down = down / nd if nd > 1e-6 else np.array([0.3, 0.0, -1.0])
    # HAND ball: below/around the wrist (the hand hangs past the wrist), NOT the
    # forearm (which is above the wrist). This keeps the thumb (which branches
    # sideways, not straight down) inside the region.
    rel = co - wr
    proj = rel @ down
    dist = np.linalg.norm(rel, axis=1)
    side_z = 1.0 if wr[0] >= 0 else -1.0
    # hand ball: distal of the wrist (proj>0 drops the wrist-back bulge & forearm)
    hand = (dist < 0.20 * h) & (proj > 0.01 * h) & (co[:, 0] * side_z > 0.0)
    if hand.sum() < 10:
        return {}
    H = co[hand]
    Hd = np.linalg.norm(H - wr, axis=1)
    spread = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ down) * down
    ns = np.linalg.norm(spread)
    spread = spread / ns if ns > 1e-6 else np.array([1.0, 0.0, 0.0])

    # all digit-tip protrusions: NMS on distance-from-wrist (well separated)
    tips = []
    for idx in np.argsort(-Hd):
        pt = H[idx]
        if all(np.linalg.norm(pt - t) > 0.026 * h for t in tips):
            tips.append(pt)
        if len(tips) >= 8:
            break
    if len(tips) < 2:
        return {}

    # THUMB = the most ISOLATED digit (largest nearest-neighbour gap in 3D);
    # orientation-independent, so it works whichever way the thumb branches.
    thumb = None
    if len(tips) >= 3:
        T = np.array(tips)
        nn = np.empty(len(T))
        for i in range(len(T)):
            dd = np.linalg.norm(T - T[i], axis=1); dd[i] = 1e9
            nn[i] = dd.min()
        i = int(np.argmax(nn))
        if nn[i] > max(0.045 * h, 1.6 * float(np.median(nn))):
            thumb = tips.pop(i)

    # the 4 fingers = the most-distal of what remains, ordered along the spread
    fingers = sorted(tips, key=lambda t: -((t - wr) @ down))[:4]
    fingers = sorted(fingers, key=lambda t: (t - wr) @ spread)
    ordered = ([thumb] + fingers) if thumb is not None else fingers
    names = (["thumb"] + ["index", "middle", "ring", "pinky"][:len(fingers)]) if thumb is not None \
        else markers.FINGER_NAMES[:len(ordered)]
    return {nm: Vector([float(v) for v in t]) for nm, t in zip(names, ordered)}


def _detect_toes_auto(co, ground, h, ankle, ball, toe_tip):
    """Detect individual toe tips across the front of the foot. Returns
    {toe_name: (base, tip)} — each toe runs from the ball line to its own tip."""
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    leg_x = float(ankle.x)
    m = (z < ground + 0.06 * h) & (np.abs(x - leg_x) < 0.16 * h) & (y < float(ball.y) + 0.01 * h)
    F = co[m]
    if F.shape[0] < 6:
        return {}
    order = np.argsort(F[:, 1])                         # most-forward (-Y) first
    tips = []
    suppx = 0.016 * h
    for idx in order:
        pt = F[idx]
        if all(abs(pt[0] - t[0]) > suppx for t in tips):
            tips.append(pt)
        if len(tips) >= 5:
            break
    if len(tips) < 2:
        return {}
    tips = sorted(tips, key=lambda t: t[0])             # lateral order
    out = {}
    for i, t in enumerate(tips):
        tip = Vector((float(t[0]), float(t[1]), ground + 0.02 * h))
        base = Vector((float(t[0]), float(ball.y), ground + 0.03 * h))
        out["toe_%d" % (i + 1)] = (base, tip)
    return out


def _palm_normal(co, h, wrist, hand_dir):
    """PCA of the hand point cloud -> the thin axis = palm normal (back of hand).
    Used to roll the hand + finger bones so their Z faces the back of the hand."""
    w = np.array(wrist, dtype=np.float64)
    a = np.array(hand_dir, dtype=np.float64)
    d = co - w
    t = d @ a
    perp = np.linalg.norm(d - np.outer(t, a), axis=1)
    m = (t > 0.02 * h) & (t < 0.32 * h) & (perp < 0.13 * h)
    pts = co[m]
    if pts.shape[0] < 8:
        return Vector((0.0, -1.0, 0.0))
    c = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]                                   # smallest-variance axis
    nv = Vector((float(n[0]), float(n[1]), float(n[2])))
    if nv.z < 0:                                # consistent: point up-ish
        nv = -nv
    return nv.normalized()


def _bend_normal(p0, p1, p2, prefer):
    """Plane normal of a 3-joint chain (the hinge plane). `prefer` orients the
    sign so IK poles are consistent. Falls back to `prefer` when the limb is
    straight (no usable bend)."""
    n = (p1 - p0).cross(p2 - p1)
    if n.length < 1e-6:
        return prefer.normalized()
    n = n.normalized()
    if n.dot(prefer) < 0:
        n = -n
    return n


def _orient_fingers_pro(eb, only_selected=False, arm=None):
    """PROFESSIONAL finger/thumb bone roll = Rigify's OWN algorithm.

    Uses rigify.utils.bones.align_chain_x_axis, which aligns every bone in a
    finger chain to the SAME axis (perpendicular to the chain's own bend plane).
    This is exactly what Rigify does for 'automatic' fingers, so:
      * within each finger the roll is perfectly consistent (no candy-wrapper /
        no torn tips even for fingers that are slightly curled at rest), and
      * Rigify then emits its native scale-curl drivers on generate.
    The user never has to think about bone roll - it is arranged for them.

    `arm` is the armature OBJECT (needed by the Rigify helper). If not given we
    fall back to a robust per-chain bend-plane roll (same idea, no Rigify import).
    Palm/metacarpal bones roll toward the back of the hand."""
    import re
    if arm is None:
        arm = getattr(eb, "id_data", None)      # edit_bones -> Armature datablock
        try:
            import bpy as _bpy
            arm = next((o for o in _bpy.data.objects
                        if o.type == 'ARMATURE' and o.data == arm), None) or arm
        except Exception:
            pass
    fingers = {}
    palms = []
    for nm in list(eb.keys()):
        m = re.match(r"f_([a-z0-9]+)\.(\d+)(\.[LR])$", nm)
        if m:
            fingers.setdefault((m.group(3), m.group(1)), []).append((int(m.group(2)), nm)); continue
        m = re.match(r"thumb\.(\d+)(\.[LR])$", nm)
        if m:
            fingers.setdefault((m.group(2), "thumb"), []).append((int(m.group(1)), nm)); continue
        if nm.startswith("palm"):
            palms.append(nm)

    _acx = None
    _abx = None
    try:
        from rigify.utils.bones import align_chain_x_axis as _acx
        from rigify.utils.bones import align_bone_x_axis as _abx
    except Exception:
        _acx = None; _abx = None

    def _finger_normal(bones):
        """Bend-plane normal of one finger chain."""
        P = [b.head for b in bones] + [bones[-1].tail]
        for i in range(len(P) - 2):
            c = (P[i + 1] - P[i]).cross(P[i + 2] - P[i + 1])
            if c.length > 1e-7:
                return c.normalized()
        return None

    # ONE SHARED bend axis for all four long fingers per side = the average of
    # their bend-plane normals. Every finger bone's X is aligned to it, so all
    # fingers curl in the SAME plane (consistent - rotating/scaling any finger
    # behaves identically; no more "index goes sideways"). The clean monotonic
    # placement makes this tear-free (within-finger stays ~1.0). Thumb keeps its
    # own plane.
    _shared = {}
    for _side in (".L", ".R"):
        acc = Vector((0.0, 0.0, 0.0)); ref = None; nn = 0
        for _f in ("index", "middle", "ring", "pinky"):
            ch = [eb.get("f_%s.%s%s" % (_f, j, _side)) for j in ("01", "02", "03")]
            ch = [b for b in ch if b]
            if len(ch) < 2:
                continue
            N = _finger_normal(ch)
            if N is None:
                continue
            if ref is None:
                ref = N
            if N.dot(ref) < 0:
                N = -N
            acc += N; nn += 1
        _shared[_side] = (acc / nn).normalized() if nn else None

    def _bendplane_roll(bones):
        """Fallback: one shared bend-plane normal for the whole chain."""
        pts = [bones[0].head] + [b.tail for b in bones]
        N = None
        for i in range(len(pts) - 2):
            c = (pts[i + 1] - pts[i]).cross(pts[i + 2] - pts[i + 1])
            if c.length > 1e-7:
                N = c.normalized(); break
        if N is None:
            N = Vector((0.0, 1.0, 0.0))
        for b in bones:
            d = (b.tail - b.head)
            if d.length > 1e-6:
                z = N.cross(d.normalized())
                if z.length > 1e-6:
                    try:
                        b.align_roll(z)
                    except Exception:
                        pass

    for (side, fn), lst in fingers.items():
        lst.sort()
        bones = [eb[n] for _, n in lst]
        if only_selected and not any(getattr(b, 'select', False) for b in bones):
            continue
        # IDEMPOTENT: align_bone_x_axis ADDS to the current roll, so zero it
        # first -> identical result on every Back-to-Metarig / Re-generate.
        for b in bones:
            b.roll = 0.0
        SN = _shared.get(side) if fn != "thumb" else None
        if SN is not None and _abx is not None and arm is not None:
            try:
                for b in bones:                      # all fingers -> ONE shared axis
                    _abx(arm, b.name, SN)
                continue
            except Exception:
                pass
        if fn == "thumb" and _acx is not None and arm is not None and len(bones) >= 2:
            try:
                _acx(arm, [b.name for b in bones])   # thumb keeps its own plane
                continue
            except Exception:
                pass
        _bendplane_roll(bones)                        # robust fallback

    for nm in palms:                                        # metacarpals -> back of hand
        b = eb.get(nm)
        if b is None:
            continue
        if only_selected and not b.select:
            continue
        d = (b.tail - b.head)
        if d.length > 1e-6:
            z = Vector((0.0, 1.0, 0.0)).cross(d.normalized())
            if z.length > 1e-6:
                try:
                    b.align_roll(z)
                except Exception:
                    pass


def orient_bones_pro(arm, J, co, h):
    """Bone roll matching RIGIFY's convention exactly (measured from the Rigify
    human metarig and verified to reproduce its rolls to ~0.01 rad):
      axial (spine/neck/head) + arms (clavicle/upper_arm/forearm/hand): Z -> front (-Y)
      legs (thigh/shin) + foot:  Z -> back (+Y)
      toe:                       Z -> up (+Z)
      fingers (f_*/palm_*):      align_roll(globalY x bone_dir)  -> X faces back of hand
    """
    eb = arm.data.edit_bones
    FRONT = Vector((0.0, -1.0, 0.0))
    BACK = Vector((0.0, 1.0, 0.0))
    UP = Vector((0.0, 0.0, 1.0))
    Y = Vector((0.0, 1.0, 0.0))

    def setz(name, z):
        b = eb.get(name)
        if b is not None and z is not None and z.length > 1e-6:
            try:
                b.align_roll(z)
            except Exception:
                pass

    for nm in list(eb.keys()):
        if nm == "root" or nm == "neck_01" or nm == "head" or nm.startswith("spine_"):
            setz(nm, FRONT)
        elif (nm.startswith("clavicle") or nm.startswith("upper_arm")
              or nm.startswith("forearm") or nm.startswith("hand")):
            setz(nm, FRONT)
        elif nm.startswith("thigh") or nm.startswith("shin"):
            setz(nm, BACK)
        elif nm.startswith("foot") or nm.startswith("toe"):
            setz(nm, UP)        # foot points forward -> Z up keeps X lateral (clean roll)
        # fingers / palm handled below (bend-plane aware, pro)
    _orient_fingers_pro(eb)


def _mirror(v):
    return Vector((-v.x, v.y, v.z))


def build_reference(props):
    J, err, h = compute_joints(props)
    if err:
        return None, err

    # fresh armature
    old = bpy.data.objects.get(utils.REF_NAME)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    arm_data = bpy.data.armatures.new(utils.REF_NAME)
    arm = bpy.data.objects.new(utils.REF_NAME, arm_data)
    bpy.context.scene.collection.objects.link(arm)
    arm.show_in_front = True

    bpy.context.view_layer.objects.active = arm
    for o in bpy.context.selected_objects:
        o.select_set(False)
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    def add(name, head, tail, parent=None, conn=False):
        b = eb.new(name)
        b.head = Vector(head); b.tail = Vector(tail)
        if parent:
            b.parent = eb[parent]; b.use_connect = conn
        return b

    # root + spine
    sp = J["spine_pts"]
    add("root", J["root"], sp[0])
    prev = "root"
    for i in range(len(sp) - 1):
        nm = "spine_%02d" % (i + 1)
        add(nm, sp[i], sp[i + 1], parent=prev, conn=(prev != "root"))
        prev = nm
    spine_top = prev
    # neck + head
    add("neck_01", J["neck_base"], J["head_base"], parent=spine_top)
    add("head", J["head_base"], J["head_top"], parent="neck_01", conn=True)

    # limbs (left, then mirror)
    def add_side(suffix, mirror):
        f = _mirror if mirror else (lambda v: v)
        ch, sh = J["clavicle.L"]; ua0, ua1 = J["upper_arm.L"]
        fa0, fa1 = J["forearm.L"]; ha0, ha1 = J["hand.L"]
        if props.use_clavicles:
            add("clavicle" + suffix, f(ch), f(sh), parent=spine_top)
            arm_parent = "clavicle" + suffix
        else:
            arm_parent = spine_top
        add("upper_arm" + suffix, f(ua0), f(ua1), parent=arm_parent)
        add("forearm" + suffix, f(fa0), f(fa1), parent="upper_arm" + suffix, conn=True)
        add("hand" + suffix, f(ha0), f(ha1), parent="forearm" + suffix, conn=True)
        th0, th1 = J["thigh.L"]; sh0, sh1 = J["shin.L"]
        ft0, ft1 = J["foot.L"]; to0, to1 = J["toe.L"]
        add("thigh" + suffix, f(th0), f(th1), parent="root")
        add("shin" + suffix, f(sh0), f(sh1), parent="thigh" + suffix, conn=True)
        add("foot" + suffix, f(ft0), f(ft1), parent="shin" + suffix, conn=True)
        add("toe" + suffix, f(to0), f(to1), parent="foot" + suffix, conn=True)
        # fingers: PALM (metacarpal) bone from the wrist out to each knuckle,
        # then the 3 phalanges parented to it.
        hand_root = J["hand.L"][0]                      # = wrist
        # MANUAL hand fingers + foot toes: build the exact chains the user placed,
        # with professional Rigify-style names (thumb.0k / palm.0N / f_<name>.0k).
        if J.get("fingers_manual") or J.get("toes_manual") or J.get("palm_manual"):
            hand_root = J["hand.L"][0]            # wrist
            # palm bones are named by the finger's ANATOMICAL index
            # (index=palm.01, middle=02, ring=03, pinky=04) and paired to the NEAREST
            # user-placed palm chain - so numbering is anatomical, not placement order.
            ORDER_NUM = {"index": 1, "middle": 2, "ring": 3, "pinky": 4}
            palm_chains = [ch for ch in J.get("palm_manual", {}).values() if len(ch) >= 2]
            used_palms = set()
            extra_num = 5
            for fn, chain in J.get("fingers_manual", {}).items():
                if len(chain) < 2:
                    continue
                base = chain[0]
                if fn == "thumb":                 # thumb: bones straight off the hand
                    prev = "hand" + suffix; conn = False
                    for k in range(len(chain) - 1):
                        nm = "thumb.%02d%s" % (k + 1, suffix)
                        add(nm, f(chain[k]), f(chain[k + 1]), parent=prev, conn=conn)
                        finger_bone_names.append(nm); prev = nm; conn = True
                    continue
                num = ORDER_NUM.get(fn)
                if num is None:
                    num = extra_num; extra_num += 1
                if palm_chains:                   # nearest unused manual palm
                    k = min((kk for kk in range(len(palm_chains)) if kk not in used_palms),
                            key=lambda kk: (base - palm_chains[kk][-1]).length, default=-1)
                    if k >= 0:
                        used_palms.add(k); pch = palm_chains[k]
                        pnm = "palm.%02d%s" % (num, suffix)
                        add(pnm, f(pch[0]), f(pch[-1]), parent="hand" + suffix)
                        finger_bone_names.append(pnm)
                        parent = pnm
                        conn = (base - pch[-1]).length < 0.02 * h
                    else:
                        parent = "hand" + suffix; conn = False
                elif getattr(props, "palm_bones", True):   # AUTO palm (no manual palms)
                    pnm = "palm.%02d%s" % (num, suffix)
                    add(pnm, f(hand_root), f(base), parent="hand" + suffix)
                    finger_bone_names.append(pnm)
                    parent = pnm; conn = True
                else:
                    parent = "hand" + suffix; conn = False
                prev = parent
                for k in range(len(chain) - 1):
                    nm = "f_%s.%02d%s" % (fn, k + 1, suffix)
                    add(nm, f(chain[k]), f(chain[k + 1]), parent=prev, conn=conn)
                    finger_bone_names.append(nm); prev = nm; conn = True
            for tn, chain in J.get("toes_manual", {}).items():
                if len(chain) < 2:
                    continue
                prev = "foot" + suffix; conn = False
                for k in range(len(chain) - 1):
                    nm = "toe_%s.%02d%s" % (tn, k + 1, suffix)
                    add(nm, f(chain[k]), f(chain[k + 1]), parent=prev, conn=conn)
                    toe_bone_names.append(nm)
                    prev = nm; conn = True
            return
        for fn, js in J.get("fingers", {}).items():
            tip, j1, j2, base = js
            n1 = "f_%s.01%s" % (fn, suffix)
            n2 = "f_%s.02%s" % (fn, suffix)
            n3 = "f_%s.03%s" % (fn, suffix)
            if fn == "thumb":
                # CloudRig: the THUMB has NO separate carpal - 3 bones straight off the hand
                add(n1, f(base), f(j2), parent="hand" + suffix)
                finger_bone_names.append(n1)
            else:
                # 4 fingers: a CARPAL (metacarpal/palm) bone, then 3 phalanges
                pm = "palm_%s%s" % (fn, suffix)
                pa0 = hand_root.lerp(base, 0.45)        # carpal starts mid-palm (CloudRig ~46%)
                add(pm, f(pa0), f(base), parent="hand" + suffix)
                add(n1, f(base), f(j2), parent=pm, conn=True)
                finger_bone_names.append(pm); finger_bone_names.append(n1)
            add(n2, f(j2), f(j1), parent=n1, conn=True)
            add(n3, f(j1), f(tip), parent=n2, conn=True)
            finger_bone_names.extend([n2, n3])
        # individual toes (detail bones on the foot)
        for tn, (tb0, tb1) in J.get("toes", {}).items():
            nm = "%s%s" % (tn, suffix)          # toe_1.L ...
            add(nm, f(tb0), f(tb1), parent="foot" + suffix)
            toe_bone_names.append(nm)

    finger_bone_names = []
    toe_bone_names = []
    add_side(".L", False)
    add_side(".R", True)

    # professional, ARP-style reference-bone roll (bend-plane aware)
    co_roll = utils.read_world_coords(props.target_mesh)
    orient_bones_pro(arm, J, co_roll, h)
    axial = ["root", "neck_01", "head"] + ["spine_%02d" % (i + 1) for i in range(len(sp) - 1)]
    legs = []
    arms = []
    for s in (".L", ".R"):
        legs += ["thigh" + s, "shin" + s, "foot" + s, "toe" + s]
        arms += ["clavicle" + s, "upper_arm" + s, "forearm" + s, "hand" + s]
    arms += finger_bone_names
    legs += toe_bone_names

    # bone collections
    def_col = utils.bone_collection(arm_data, "DEF")
    for b in arm_data.edit_bones:
        def_col.assign(b)

    bpy.ops.object.mode_set(mode='OBJECT')

    # colour the reference bones by region (visible in edit/pose)
    def _col(names, theme):
        for nm in names:
            b = arm_data.bones.get(nm)
            if b:
                try:
                    b.color.palette = theme
                except Exception:
                    pass
    _col(axial, 'THEME03')          # axial -> green
    _col(arms, 'THEME09')           # arms  -> yellow/gold
    _col(legs, 'THEME01')           # legs  -> red
    # hide the root bone (it's a parent helper - not for posing/display)
    rb = arm_data.bones.get("root")
    if rb is not None:
        try:
            rb.hide = True
        except Exception:
            pass
    return arm, None


class SMARTRIG_OT_go(bpy.types.Operator):
    bl_idname = "smartrig.go"
    bl_label = "Go!  (build reference skeleton)"
    bl_description = "Derive the skeleton from the markers + mesh geometry into SR_Reference"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.smartrig.target_mesh is not None

    def execute(self, context):
        props = context.scene.smartrig
        arm, err = build_reference(props)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        # hide the markers once the skeleton is built (Show Markers brings them back)
        try:
            markers.set_markers_hidden(True)
            props.markers_hidden = True
        except Exception:
            pass
        # jump straight into Edit Mode on the coloured reference for tweaking
        try:
            for o in context.selected_objects:
                o.select_set(False)
            context.view_layer.objects.active = arm
            arm.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass
        self.report({'INFO'}, "Skeleton built. You're in Edit Mode - tweak bones, then Match to Rig.")
        return {'FINISHED'}


def _ensure_arm_edit(context):
    obj = context.active_object
    if obj is None or obj.type != 'ARMATURE':
        return None
    if obj.mode != 'EDIT':
        try:
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            return None
    return obj


class SMARTRIG_OT_roll_recalc(bpy.types.Operator):
    bl_idname = "smartrig.roll_recalc"
    bl_label = "Recalculate Roll"
    bl_description = "Recalculate the roll of the SELECTED bones toward the chosen axis"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, c):
        return c.active_object is not None and c.active_object.type == 'ARMATURE'

    def execute(self, context):
        obj = _ensure_arm_edit(context)
        if obj is None:
            self.report({'ERROR'}, "Select the skeleton (armature) first.")
            return {'CANCELLED'}
        sel = [b for b in obj.data.edit_bones if b.select]
        if not sel:
            self.report({'WARNING'}, "Select bone(s) first (a group or one).")
            return {'CANCELLED'}
        try:
            bpy.ops.armature.calculate_roll(type=context.scene.smartrig.roll_axis)
        except Exception as e:
            self.report({'WARNING'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Roll recalculated for %d bone(s)." % len(sel))
        return {'FINISHED'}


class SMARTRIG_OT_roll_nudge(bpy.types.Operator):
    bl_idname = "smartrig.roll_nudge"
    bl_label = "Nudge Roll"
    bl_description = "Rotate the roll of the selected bones by a small step"
    bl_options = {'REGISTER', 'UNDO'}
    amount: bpy.props.FloatProperty(default=0.0872665)

    def execute(self, context):
        obj = _ensure_arm_edit(context)
        if obj is None:
            return {'CANCELLED'}
        n = 0
        for b in obj.data.edit_bones:
            if b.select:
                b.roll += self.amount; n += 1
        self.report({'INFO'}, "Nudged %d bone(s)." % n)
        return {'FINISHED'}


class SMARTRIG_OT_roll_clear(bpy.types.Operator):
    bl_idname = "smartrig.roll_clear"
    bl_label = "Clear Roll"
    bl_description = "Set the roll of the selected bones to 0"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _ensure_arm_edit(context)
        if obj is None:
            return {'CANCELLED'}
        for b in obj.data.edit_bones:
            if b.select:
                b.roll = 0.0
        return {'FINISHED'}


class SMARTRIG_OT_roll_fingers_pro(bpy.types.Operator):
    bl_idname = "smartrig.roll_fingers_pro"
    bl_label = "Pro Finger Roll"
    bl_description = "Re-apply the professional bend-plane roll to the SELECTED finger/thumb bones"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _ensure_arm_edit(context)
        if obj is None:
            self.report({'ERROR'}, "Select the skeleton (armature) first.")
            return {'CANCELLED'}
        _orient_fingers_pro(obj.data.edit_bones, only_selected=True)
        self.report({'INFO'}, "Pro finger roll applied to selected finger bones.")
        return {'FINISHED'}


classes = (SMARTRIG_OT_go, SMARTRIG_OT_roll_recalc, SMARTRIG_OT_roll_nudge,
           SMARTRIG_OT_roll_clear, SMARTRIG_OT_roll_fingers_pro)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
