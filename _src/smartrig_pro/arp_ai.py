"""arp_ai.py - Soulify bridge to Auto-Rig Pro's official AI inference tools
(the AI files were provided to the user by the ARP author for his use).

Pipeline (mirrors ARP's guess_fingers): ortho camera on the hand -> N OpenGL
screenshots rotating around the hand axis -> inference_fingers exe -> 20
keypoints per view (thumb1..pinky4, pixel coords in 256x256, None = miss) ->
multi-view triangulation (median of pairwise closest-points) -> finger joint
chain (MCP, PIP, DIP, tip) per finger -> written into the SR_Metarig.
"""
import bpy, os, json, math, random, subprocess, platform
from itertools import combinations
from mathutils import Vector, Matrix

FIN = ["thumb", "index", "middle", "ring", "pinky"]


# ------------------------------------------------------------------ paths
def ai_root():
    """Folder holding info.dat + inference/ (per-OS package from the ARP
    author: AI_win / AI_mac / AI_macIntel / AI_linux). Search order: scene
    prop, AI folder next to the .blend, AI folder inside the addon, ~/Soulify_AI."""
    cands = []
    p = getattr(bpy.context.scene.smartrig, "ai_tools_path", "")
    if p:
        cands.append(bpy.path.abspath(p))
    if bpy.data.filepath:
        cands.append(os.path.join(os.path.dirname(bpy.data.filepath), "AI"))
    cands.append(os.path.join(os.path.dirname(__file__), "AI"))
    cands.append(os.path.join(os.path.expanduser("~"), "Soulify_AI"))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "info.dat")):
            return c
    return None


def inference_dir():
    r = ai_root()
    return os.path.join(r, "inference") if r else None


def fingers_available():
    d = inference_dir()
    if d is None:
        return False
    exe = "inference_fingers.exe" if platform.system() == "Windows" else "inference_fingers"
    return (os.path.exists(os.path.join(d, exe))
            and os.path.exists(os.path.join(d, "fingers_model.pth")))


# ------------------------------------------------------------------ maths
def _lookat_up(matrix, target, up_axis):
    eye = matrix.to_translation()
    forward = Vector(eye - target).normalized()
    up = up_axis.normalized()
    right = up.cross(forward).normalized()
    up = forward.cross(right).normalized()
    return Matrix(([right[0], up[0], forward[0], 0],
                   [right[1], up[1], forward[1], 0],
                   [right[2], up[2], forward[2], 0],
                   [0, 0, 0, 1]))


def _closest_between_lines(p1, d1, p2, d2, tol=1e-8):
    w0 = p2 - p1
    a = d1.dot(d1); b = d1.dot(d2); c = d2.dot(d2)
    e = w0.dot(d1); f = w0.dot(d2)
    denom = a * c - b * b
    if abs(denom) < tol:
        return (p1 + p2) * 0.5
    s = (e * c - b * f) / denom
    t = (e * b - a * f) / denom
    return ((p1 + d1 * s) + (p2 + d2 * t)) * 0.5


def _median_v(points):
    out = Vector((0, 0, 0))
    for ax in range(3):
        vals = sorted(p[ax] for p in points)
        n = len(vals)
        out[ax] = vals[n // 2] if n % 2 else 0.5 * (vals[n//2 - 1] + vals[n//2])
    return out


# ------------------------------------------------------------------ render
def _view3d_override():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return {"window": win, "area": area, "region": region,
                                "space_data": area.spaces.active}
    return None


def _screenshot_hand(mesh, wrist, tip, view_dir, samples, inf_dir, side_tag):
    """ARP's screenshot protocol. Returns (cam_matrices, ortho_scale, count)."""
    scn = bpy.context.scene
    ov = _view3d_override()
    if ov is None:
        raise RuntimeError("Need an open 3D viewport")
    # render the REST shape: posed/curled fingers ruin the detection
    rest_set = []
    for md in mesh.modifiers:
        if md.type == 'ARMATURE' and md.object is not None:
            if md.object.data.pose_position != 'REST':
                md.object.data.pose_position = 'REST'
                rest_set.append(md.object.name)
    space = ov["space_data"]
    r3d = space.region_3d

    # ---- store state
    st = {"res": (scn.render.resolution_x, scn.render.resolution_y,
                  scn.render.resolution_percentage),
          "film": scn.render.film_transparent,
          "vt": scn.view_settings.view_transform,
          "shad": space.shading.type,
          "light": space.shading.light,
          "ctype": space.shading.color_type,
          "scol": tuple(space.shading.single_color),
          "slight": space.shading.studio_light,
          "bgt": space.shading.background_type,
          "bgc": tuple(space.shading.background_color),
          "over": space.overlay.show_overlays,
          "cam": scn.camera,
          "persp": r3d.view_perspective,
          "vmat": r3d.view_matrix.copy(),
          "fmt": scn.render.image_settings.file_format,
          "cm": scn.render.image_settings.color_mode}
    hidden = []
    for ob in bpy.context.view_layer.objects:
        if ob is not mesh and not ob.hide_get():
            ob.hide_set(True)
            hidden.append(ob.name)
    try:
        mesh.hide_set(False)
    except Exception:
        pass

    scn.render.resolution_x = scn.render.resolution_y = 256
    scn.render.resolution_percentage = 100
    scn.render.film_transparent = False
    scn.render.image_settings.file_format = 'JPEG'
    scn.render.image_settings.color_mode = 'RGB'
    try:
        scn.render.image_settings.quality = 98
    except Exception:
        pass
    try:
        scn.view_settings.view_transform = 'Standard'
    except Exception:
        pass
    space.shading.type = 'SOLID'
    try:
        # EXACT ARP shading recipe (the model is trained on these renders)
        space.shading.color_type = 'SINGLE'
        space.shading.single_color = (0.8, 0.8, 0.8)
        space.shading.light = 'STUDIO'
        space.shading.studio_light = 'Default'
        space.shading.use_world_space_lighting = False
        space.shading.show_cavity = False
        space.shading.background_type = 'VIEWPORT'
        space.shading.background_color = (0.040914, 0.0409144, 0.0409144)
        space.shading.show_xray = False
    except Exception:
        pass
    space.overlay.show_overlays = False

    # subsurf for low-poly shading, like ARP
    subs = None
    if len(mesh.data.polygons) < 6000:
        subs = mesh.modifiers.new('arp_ai_subsurf', 'SUBSURF')

    margin = 1.6
    rot_field = 65 + (samples - 5) * 4
    hand_dir = tip - wrist
    hand_mid = (tip + wrist) * 0.5

    cam_data = bpy.data.cameras.new('sr_ai_cam')
    cam_obj = bpy.data.objects.new('sr_ai_cam', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scn.camera = cam_obj
    # camera along the PALM NORMAL (view_dir), fingers pointing image-right.
    # ARP's fixed formula assumes a T-pose arm; with lowered arms it films the
    # hand edge-on and the model sees nothing.
    fwd = view_dir.normalized()
    cam_pos = hand_mid + fwd * hand_dir.magnitude
    right = hand_dir.normalized()
    up2 = fwd.cross(right).normalized()
    right = up2.cross(fwd).normalized()
    rot = Matrix(([right[0], up2[0], fwd[0], 0],
                  [right[1], up2[1], fwd[1], 0],
                  [right[2], up2[2], fwd[2], 0],
                  [0, 0, 0, 1]))
    cam_obj.matrix_world = Matrix.Translation(cam_pos) @ rot
    bpy.context.view_layer.update()
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = hand_dir.magnitude * margin
    ortho = cam_data.ortho_scale
    cam_data.clip_start = hand_dir.magnitude / 100
    cam_data.clip_end = hand_dir.magnitude * 100

    r3d.view_perspective = 'CAMERA'

    cams = []
    idx = 0

    def rotate_cam(ang):
        ax = Vector((cam_obj.matrix_world[0][0], cam_obj.matrix_world[1][0],
                     cam_obj.matrix_world[2][0]))
        rr = Matrix.Rotation(ang, 4, ax)
        loc = Matrix.Translation(hand_mid)
        cam_obj.matrix_world = loc @ rr @ loc.inverted() @ cam_obj.matrix_world
        bpy.context.view_layer.update()

    def snap():
        nonlocal idx
        idx += 1
        path = os.path.join(inf_dir, 'hand%d%s.jpg' % (idx, side_tag))
        with bpy.context.temp_override(**{k: ov[k] for k in ("window", "area", "region")}):
            bpy.ops.render.opengl(write_still=False)
        bpy.data.images['Render Result'].save_render(filepath=path)
        cams.append(cam_obj.matrix_world.copy())

    snap()
    step = math.radians(rot_field / samples)
    for _ in range(samples):
        rotate_cam(step)
        snap()
    rotate_cam(math.radians(-rot_field))
    for _ in range(samples):
        rotate_cam(-step)
        snap()

    # ---- restore
    for nm in rest_set:
        ob = bpy.data.objects.get(nm)
        if ob is not None:
            ob.data.pose_position = 'POSE'
    bpy.data.objects.remove(cam_obj)
    if subs is not None:
        mesh.modifiers.remove(subs)
    for nm in hidden:
        ob = bpy.data.objects.get(nm)
        if ob is not None:
            try:
                ob.hide_set(False)
            except Exception:
                pass
    scn.render.resolution_x, scn.render.resolution_y, \
        scn.render.resolution_percentage = st["res"]
    scn.render.film_transparent = st["film"]
    scn.render.image_settings.file_format = st["fmt"]
    scn.render.image_settings.color_mode = st["cm"]
    try:
        scn.view_settings.view_transform = st["vt"]
    except Exception:
        pass
    space.shading.type = st["shad"]
    try:
        space.shading.light = st["light"]
        space.shading.color_type = st["ctype"]
        space.shading.single_color = st["scol"]
        space.shading.studio_light = st["slight"]
        space.shading.background_type = st["bgt"]
        space.shading.background_color = st["bgc"]
    except Exception:
        pass
    space.overlay.show_overlays = st["over"]
    scn.camera = st["cam"]
    r3d.view_perspective = st["persp"]
    r3d.view_matrix = st["vmat"]
    return cams, ortho, idx


# ------------------------------------------------------------------ engine
def _palm_normal(mesh, wrist, tip):
    """Palm normal = smallest PCA axis of the hand vert cloud."""
    import numpy as np
    mw = mesh.matrix_world
    mid = (wrist + tip) * 0.5
    L = (tip - wrist).length
    H = np.array([tuple(mw @ v.co) for v in mesh.data.vertices
                  if (mw @ v.co - mid).length < 0.75 * L])
    if len(H) < 30:
        return Vector((0, -1, 0))
    ctr = H.mean(0)
    _u, _s, vt = np.linalg.svd(H - ctr)
    return Vector(vt[2]).normalized()


def _run_exe(inf, imgs, thresh, nfingers):
    exe = os.path.join(inf, 'inference_fingers.exe'
                       if platform.system() == 'Windows' else 'inference_fingers')
    if platform.system() != 'Windows':
        # zip extraction loses the exec bit on mac/linux
        try:
            os.chmod(exe, 0o755)
        except Exception:
            pass
    # ARP's run_process appends FINGERS then THRESHOLD (misleading names)
    return subprocess.run([exe, imgs + ',', str(nfingers), str(thresh)],
                          capture_output=True, text=True, cwd=inf)


def _parse_kp(inf, side_tag, count):
    dicts = []
    nonnull = 0
    for i in range(1, count + 1):
        d = {}
        kp = os.path.join(inf, 'hand%d%s_kp.py' % (i, side_tag))
        if os.path.exists(kp):
            with open(kp) as f:
                d = json.loads(f.readline())
        nonnull += sum(1 for v in d.values() if v is not None)
        bad = {k[:-1] for k, v in d.items() if v is None}
        for b in bad:
            for j in range(1, 5):
                d.pop(b + str(j), None)
        dicts.append(d)
    return dicts, nonnull


def _cleanup(inf, side_tag, count):
    for i in range(1, count + 1):
        for suff in ('.jpg', '_kp.py'):
            try:
                os.remove(os.path.join(inf, 'hand%d%s%s' % (i, side_tag, suff)))
            except Exception:
                pass


def detect_fingers(mesh, wrist, tip, side_tag,
                   samples=8, thresh=0.5, nfingers=5):
    """Returns {finger: [Vector x4 (MCP..tip)]} or {} on failure. Tries BOTH
    palm-normal directions (back of hand vs palm) and keeps the better one."""
    inf = inference_dir()
    pn = _palm_normal(mesh, wrist, tip)
    best = None
    for tag_suffix, sign in (("a", 1.0), ("b", -1.0)):
        tag = side_tag + tag_suffix
        cams, ortho, count = _screenshot_hand(mesh, wrist, tip, pn * sign,
                                              samples, inf, tag)
        imgs = ','.join('hand%d%s.jpg' % (i, tag) for i in range(1, count + 1))
        res = _run_exe(inf, imgs, thresh, nfingers)
        dicts, nonnull = _parse_kp(inf, tag, count)
        print('AI fingers %s rc=%s nonnull=%d' % (tag, res.returncode, nonnull))
        _cleanup(inf, tag, count)
        if best is None or nonnull > best[0]:
            best = (nonnull, dicts, cams, ortho)
    if best is None or best[0] == 0:
        return {}
    _n, dicts, cams, ortho = best
    out = {}
    for f in FIN:
        chain = []
        for j in range(1, 5):
            key = f + str(j)
            data = [(ci, d[key]) for ci, d in enumerate(dicts) if key in d]
            if len(data) < 2:
                chain = None
                break
            pairs = list(combinations(data, 2))
            random.shuffle(pairs)
            pairs = pairs[:max(1, len(data) - 1)]
            pts = []
            for (c1, p1), (c2, p2) in pairs:
                n1 = Vector((((p1[0] - 128) / 128), (-(p1[1] - 128) / 128), 0.0))
                n2 = Vector((((p2[0] - 128) / 128), (-(p2[1] - 128) / 128), 0.0))
                m1, m2 = cams[c1], cams[c2]
                w1 = m1 @ (n1 * (ortho / 2))
                w2 = m2 @ (n2 * (ortho / 2))
                d1 = Vector((m1[0][2], m1[1][2], m1[2][2]))
                d2 = Vector((m2[0][2], m2[1][2], m2[2][2]))
                pts.append(_closest_between_lines(w1, d1, w2, d2))
            chain.append(_median_v(pts))
        if chain:
            out[f] = chain
    return out


def apply_to_metarig(meta, res, side):
    """Write AI joint chains into the metarig fingers + palm tails."""
    inv = meta.matrix_world.inverted()
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    meta.hide_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    meta.select_set(True)
    bpy.context.view_layer.objects.active = meta
    bpy.ops.object.mode_set(mode='EDIT')
    eb = meta.data.edit_bones
    mirr = meta.data.use_mirror_x
    meta.data.use_mirror_x = False
    n = 0
    for f, chain in res.items():
        pre = "thumb." if f == "thumb" else "f_%s." % f
        b1 = eb.get(pre + "01" + side)
        b2 = eb.get(pre + "02" + side)
        b3 = eb.get(pre + "03" + side)
        if not (b1 and b2 and b3):
            continue
        J0, J1, J2, tip = [Vector(c) for c in chain]
        if f != "thumb":
            b1.head = inv @ J0
        b1.tail = inv @ J1
        b2.head = inv @ J1
        b2.tail = inv @ J2
        b3.head = inv @ J2
        b3.tail = inv @ tip
        pi = {"index": "01", "middle": "02", "ring": "03",
              "pinky": "04"}.get(f)
        if pi:
            pb = eb.get("palm.%s%s" % (pi, side))
            if pb is not None:
                pb.tail = inv @ J0
        n += 1
    meta.data.use_mirror_x = mirr
    bpy.ops.object.mode_set(mode='OBJECT')
    return n


class SMARTRIG_OT_ai_fingers(bpy.types.Operator):
    bl_idname = "smartrig.ai_fingers"
    bl_label = "AI Fingers"
    bl_description = ("Detect the finger joints with Auto-Rig Pro's official "
                      "AI tools (multi-view renders + keypoint model) and fit "
                      "the metarig fingers to them. Press Generate Rig after")
    bl_options = {'REGISTER', 'UNDO'}
    samples: bpy.props.IntProperty(default=8, min=2, max=16)
    thresh: bpy.props.FloatProperty(default=0.5, min=0.1, max=0.7)

    def execute(self, context):
        meta = bpy.data.objects.get("SR_Metarig")
        mesh = context.scene.smartrig.target_mesh
        if meta is None or mesh is None:
            self.report({'ERROR'}, "Need the metarig and the character mesh.")
            return {'CANCELLED'}
        if not fingers_available():
            self.report({'ERROR'}, "AI tools not found - set the AI folder "
                        "path (must contain info.dat + inference/).")
            return {'CANCELLED'}
        mm = meta.matrix_world
        done = 0
        for side, tag in ((".L", "_l"), (".R", "_r")):
            hb = meta.data.bones.get("hand" + side)
            tb = meta.data.bones.get("f_middle.03" + side)
            if hb is None or tb is None:
                continue
            wrist = mm @ hb.head_local
            tip = mm @ tb.tail_local
            try:
                res = detect_fingers(mesh, wrist, tip, tag,
                                     samples=self.samples, thresh=self.thresh)
            except Exception as e:
                self.report({'ERROR'}, "AI detection failed: %s" % e)
                return {'CANCELLED'}
            if res:
                done += apply_to_metarig(meta, res, side)
        if not done:
            self.report({'WARNING'}, "AI could not detect the fingers - try "
                        "more samples or a lower threshold.")
            return {'CANCELLED'}
        self.report({'INFO'}, "AI fitted %d finger chains. Generate Rig to "
                    "rebuild." % done)
        return {'FINISHED'}


classes = (SMARTRIG_OT_ai_fingers,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
