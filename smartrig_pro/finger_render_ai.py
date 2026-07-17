"""
finger_render_ai.py — RENDER-based finger detector (Auto-Rig-Pro's technique,
our own implementation). Robust to arm pose & stylized hands, where the 3D
voxel / point-cloud heuristics fail.

Pipeline (validated): orient an ORTHO camera on the PALM NORMAL (PCA of the hand
verts, sign from the arm direction -> identical framing at train & inference) ->
render the hand to a 128 grayscale image -> models/finger_kp.onnx predicts 20
keypoints (5 fingers x [root,j1,j2,tip]) in image space -> back-project each
keypoint by gathering the hand verts under it and taking the front/back medial
midpoint (validated ~8.5 mm on VRoid). Returns the voxel-format finger dict
{name: [tip, near, mid, base]}.

Needs onnxruntime + models/finger_kp.onnx (train with train_finger_kp.py on Mac).
Returns {} (degrades to the voxel detector) if either is missing.
"""
import os
import numpy as np
from mathutils import Vector, Matrix

FIN = ["thumb", "index", "middle", "ring", "pinky"]
RES = 128
MODEL_REL = os.path.join("models", "finger_kp.onnx")
_SESSION = None


def model_path():
    return os.path.join(os.path.dirname(__file__), MODEL_REL)


def available():
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return os.path.exists(model_path())


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3
        _SESSION = ort.InferenceSession(model_path(), so, providers=["CPUExecutionProvider"])
    return _SESSION


def palm_frame(H, wrist, elbow):
    """Canonical hand frame from the hand verts only (PCA); sign from the arm.
    Returns (center, xc, yc, zc, ortho_scale). zc = palm normal, yc = finger dir.
    Identical at training and inference -> consistent renders."""
    ctr = H.mean(0)
    u, s, vt = np.linalg.svd(H - ctr)
    fdir = vt[0]                                   # longest axis ~ finger direction
    arm = np.array((wrist - elbow).normalized())
    if fdir @ arm < 0:
        fdir = -fdir                               # point away from the arm
    pn = vt[2]                                     # shortest axis ~ palm normal
    yc = Vector(fdir).normalized()
    zc = Vector(pn).normalized()
    xc = yc.cross(zc)
    xc = xc.normalized() if xc.length > 1e-5 else Vector((1, 0, 0))
    zc = xc.cross(yc).normalized()
    reach = float(np.max(np.linalg.norm(H - ctr, axis=1)))
    return Vector(ctr), xc, yc, zc, reach * 2.2


def _render_gray(mesh, cen, xc, yc, zc, ortho):
    import bpy
    sc = bpy.context.scene
    save = (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y,
            sc.camera, sc.render.filepath, sc.render.film_transparent)
    cd = bpy.data.cameras.new("SR_HC"); cd.type = 'ORTHO'; cd.ortho_scale = ortho
    cd.clip_start = 0.001; cd.clip_end = 50
    cam = bpy.data.objects.new("SR_HC", cd); sc.collection.objects.link(cam)
    cam.matrix_world = Matrix.Translation(cen + zc * max(ortho, 0.5)) @ \
        Matrix((xc, yc, zc)).transposed().to_4x4()
    hide = [o for o in bpy.data.objects if o.type == 'MESH' and o is not mesh]
    prev = [(o, o.hide_render) for o in hide]
    for o in hide:
        o.hide_render = True
    for m in mesh.modifiers:
        if m.type == 'PARTICLE_SYSTEM':
            m.show_render = False
    sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = RES; sc.render.resolution_y = RES
    sc.render.film_transparent = False; sc.camera = cam
    import tempfile
    fp = os.path.join(tempfile.gettempdir(), "sr_hand.png"); sc.render.filepath = fp
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(fp)
    W, Hh = img.size
    px = np.array(img.pixels[:]).reshape(Hh, W, 4)[..., :3].mean(-1).astype(np.float32)
    bpy.data.images.remove(img)
    bpy.data.objects.remove(cam, do_unlink=True)
    for o, h in prev:
        o.hide_render = h
    (sc.render.engine, sc.render.resolution_x, sc.render.resolution_y,
     sc.camera, sc.render.filepath, sc.render.film_transparent) = save
    return px                                      # (RES,RES), row0=bottom (matches uv y)


def detect(mesh, co, h, wrist, elbow, side_z):
    """co: world verts (N,3). Returns {name:[tip,near,mid,base]} or {}."""
    if not available():
        return {}
    try:
        wn = np.array(wrist, dtype=np.float64)
        d = (wrist - elbow)
        dn = np.array(d.normalized() if d.length > 1e-6 else Vector((1, 0, 0)))
        rel = co - wn; proj = rel @ dn; dist = np.linalg.norm(rel, axis=1)
        H = co[(proj > -0.03 * h) & (dist < 0.25 * h) & (co[:, 0] * side_z > 0)]
        if H.shape[0] < 50:
            return {}
        cen, xc, yc, zc, ortho = palm_frame(H, wrist, elbow)
        gray = _render_gray(mesh, cen, xc, yc, zc, ortho)
        inp = gray[None, None].astype(np.float32)
        kp = _session().run(None, {"image": inp})[0].reshape(20, 2)   # (u,v) in 0..1
        # project hand verts to the same image plane (ortho) for back-projection
        R = np.array([[xc.x, xc.y, xc.z], [yc.x, yc.y, yc.z], [zc.x, zc.y, zc.z]])
        loc = (H - np.array(cen)) @ R.T               # x=right, y=up, z=depth
        u = loc[:, 0] / ortho + 0.5; v = loc[:, 1] / ortho + 0.5
        uvH = np.stack([u, v], 1); depth = loc[:, 2]
        out = {}
        for fi, fn in enumerate(FIN):
            J = []
            for ji in range(4):
                tgt = kp[fi * 4 + ji]
                r = 0.04
                for _ in range(4):
                    m = np.linalg.norm(uvH - tgt, axis=1) < r
                    if m.sum() >= 3:
                        break
                    r *= 1.6
                if m.sum() < 3:
                    J = []
                    break
                sub = H[m]; dp = depth[m]
                p3 = (sub[dp.argmin()] + sub[dp.argmax()]) * 0.5
                J.append(Vector([float(c) for c in p3]))
            if len(J) == 4 and (J[3] - J[0]).length > 0.02 * h:
                out[fn] = [J[3], J[2], J[1], J[0]]    # [tip, near, mid, base]
        return out
    except Exception as e:
        print("SmartRig finger_render_ai:", e)
        return {}
