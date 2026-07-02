"""
SmartRig Pro - Stone #3: NEURAL JOINT DETECTION (ONNX)
======================================================
Renders the character from the front, runs the trained YOLO-pose ONNX model
(models/smartrig_pose.onnx) to find 16 body joints in 2D, then back-projects
each joint through the camera onto the body's mid-depth plane to recover a
world-space point.

The model was trained on 25 fully-rigged characters (Blender Studio CC-BY +
the user's own "to animate" rigs) rendered from many viewpoints - it is built
entirely from the user's data, with ZERO dependency on any commercial tool.

HOW IT IS USED (the honest, reliable design):
  The net's strongest, most reliable signal across stylized meshes is the
  *height* of each joint (which adapts to each character's true proportions -
  long torso, big head, short legs ...). Lateral left/right detail is weaker on
  cartoon meshes, so the verified GEOMETRIC engine (markers._guess_positions)
  still locks the exact lateral (X) and depth (Y) of every joint. detect() here
  supplies the adaptive heights; geometry does the rest.

Requires onnxruntime in Blender's Python:
    <blender_python> -m pip install onnxruntime
"""

import bpy
import os
import tempfile
import numpy as np
from mathutils import Vector

INP = 384                      # model input size (square)
MODEL_REL = os.path.join("models", "smartrig_pose.onnx")

# YOLO-pose output keypoint order. MUST match make_yolo_dataset.KEYPOINTS order.
# 16 body joints + 10 fingertips (5 per hand). Older 16-kpt models still work
# (the extra names are simply absent from the model output and skipped).
JOINT_NAMES = [
    "head", "neck", "chest", "pelvis",
    "shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
    "wrist_l", "wrist_r", "hip_l", "hip_r",
    "knee_l", "knee_r", "ankle_l", "ankle_r",
    "thumb_tip_l", "thumb_tip_r", "index_tip_l", "index_tip_r",
    "middle_tip_l", "middle_tip_r", "ring_tip_l", "ring_tip_r",
    "pinky_tip_l", "pinky_tip_r",
]

# map addon finger names -> detected fingertip joint (left side; .R mirrors)
FINGER_TIP_JOINT = {
    "thumb": "thumb_tip_l", "index": "index_tip_l", "middle": "middle_tip_l",
    "ring": "ring_tip_l", "pinky": "pinky_tip_l",
}

_SESSION = None


# --------------------------------------------------------------- availability
def model_path():
    return os.path.join(os.path.dirname(__file__), MODEL_REL)


def has_model():
    return os.path.exists(model_path())


def has_runtime():
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        pass
    # Blender disables the user site-packages (site.ENABLE_USER_SITE = False),
    # but `pip install onnxruntime` on a Program-Files Blender lands exactly
    # there. Add it to sys.path so the install is actually found.
    try:
        import site
        import sys
        us = site.getusersitepackages()
        if us and os.path.isdir(us) and us not in sys.path:
            sys.path.append(us)
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def available():
    return has_model() and has_runtime()


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # hardware acceleration wherever available: CUDA/TensorRT on NVIDIA,
        # CoreML on Apple Silicon, DirectML on Windows GPUs - CPU always last.
        want = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                "CoreMLExecutionProvider", "DmlExecutionProvider",
                "CPUExecutionProvider"]
        avail = set(ort.get_available_providers())
        providers = [p for p in want if p in avail] or ["CPUExecutionProvider"]
        _SESSION = ort.InferenceSession(model_path(), so, providers=providers)
        try:
            print("Soulify detect: ONNX providers =", _SESSION.get_providers())
        except Exception:
            pass
    return _SESSION


# --------------------------------------------------------------- rendering
def _mesh_bounds(obj):
    lo = Vector((1e9, 1e9, 1e9)); hi = -lo
    for c in obj.bound_box:
        w = obj.matrix_world @ Vector(c)
        lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
    return lo, hi


def _setup_camera(scene, obj):
    lo, hi = _mesh_bounds(obj)
    center = (lo + hi) * 0.5
    # SAME framing as training (smartrig_dataset_gen): mesh bounds + margin so
    # the whole character incl. fingertips stays in frame.
    maxdim = max((hi - lo).x, (hi - lo).y, (hi - lo).z)
    radius = max(maxdim * 1.9, 0.1)
    cam = bpy.data.objects.get("SR_INFER_CAM")
    if cam is None:
        cd = bpy.data.cameras.new("SR_INFER_CAM")
        cam = bpy.data.objects.new("SR_INFER_CAM", cd)
        scene.collection.objects.link(cam)
    cam.data.type = 'PERSP'
    cam.data.lens = 50.0
    cam.data.sensor_width = 36.0
    # front view: az=0, el=0  ->  looking down +Y
    cam.location = center + Vector((0.0, -radius, 0.0))
    cam.rotation_euler = (cam.location - center).to_track_quat('Z', 'Y').to_euler()
    return cam, center


def _render_front(scene, cam):
    prev = dict(
        engine=scene.render.engine,
        rx=scene.render.resolution_x, ry=scene.render.resolution_y,
        pct=scene.render.resolution_percentage,
        fp=scene.render.filepath,
        ff=scene.render.image_settings.file_format,
        ft=scene.render.film_transparent,
        cam=scene.camera,
    )
    tmp_light = None
    try:
        scene.camera = cam
        scene.render.resolution_x = INP
        scene.render.resolution_y = INP
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.film_transparent = False
        # a flat solid render is fast and matches what the model needs
        try:
            scene.render.engine = 'BLENDER_WORKBENCH'
        except Exception:
            pass
        if not any(o.type == 'LIGHT' for o in scene.objects):
            ld = bpy.data.lights.new("SR_INFER_SUN", 'SUN'); ld.energy = 3.0
            tmp_light = bpy.data.objects.new("SR_INFER_SUN", ld)
            scene.collection.objects.link(tmp_light)
            tmp_light.rotation_euler = (0.6, 0.1, 0.3)
        fp = os.path.join(tempfile.gettempdir(), "sr_infer.png")
        scene.render.filepath = fp
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)
    finally:
        if tmp_light is not None:
            bpy.data.objects.remove(tmp_light, do_unlink=True)
        scene.render.engine = prev["engine"]
        scene.render.resolution_x = prev["rx"]
        scene.render.resolution_y = prev["ry"]
        scene.render.resolution_percentage = prev["pct"]
        scene.render.filepath = prev["fp"]
        scene.render.image_settings.file_format = prev["ff"]
        scene.render.film_transparent = prev["ft"]
        scene.camera = prev["cam"]
    return fp


def _load_rgb(fp):
    img = bpy.data.images.load(fp, check_existing=False)
    img.reload()
    a = np.array(img.pixels[:], dtype=np.float32).reshape(INP, INP, 4)
    a = a[::-1, :, :3]                       # bottom-up -> top-down, drop alpha
    bpy.data.images.remove(img)
    return np.ascontiguousarray(a)


# --------------------------------------------------------------- inference
def _infer(rgb):
    inp = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
    sess = _session()
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]   # 1,53,N
    out = out[0].T                                               # N,53
    cls = out[:, 4]
    best = int(np.argmax(cls))
    conf = float(cls[best])
    kps = out[best, 5:].reshape(-1, 3)                           # 16,3 (px,py,c)
    return conf, kps


# --------------------------------------------------------------- back-project
def _pixel_ray(cam, kx, ky):
    scene = bpy.context.scene
    tr, br, bl, tl = cam.data.view_frame(scene=scene)
    M3 = cam.matrix_world.to_3x3()
    origin = cam.matrix_world.translation
    u = kx / INP
    v = 1.0 - (ky / INP)
    corner = bl.lerp(br, u).lerp(tl.lerp(tr, u), v)
    d = (M3 @ corner).normalized()
    return origin, d


def _backproject(cam, center, kx, ky, obj=None):
    """Recover a 3D world point from a 2D keypoint. PREFERRED: cast the camera
    ray at the mesh and take the VOLUME CENTRE (midpoint of the first & last
    surface hits) - this gives the true depth, so fingertips/hands that sit in
    FRONT of the body land correctly (a single mid-depth plane cannot). Falls
    back to the mid-depth plane when the ray misses the mesh."""
    origin, d = _pixel_ray(cam, kx, ky)
    if obj is not None:
        try:
            dg = bpy.context.evaluated_depsgraph_get()
            oe = obj.evaluated_get(dg)
            mi = obj.matrix_world.inverted()
            o_l = mi @ origin
            d_l = (mi.to_3x3() @ d).normalized()
            hits = []
            cur = o_l.copy()
            for _ in range(24):
                res, loc, nrm, idx = oe.ray_cast(cur, d_l)
                if not res:
                    break
                hits.append(loc.copy())
                cur = loc + d_l * 1e-4
            if hits:
                mid = (hits[0] + hits[-1]) * 0.5
                return obj.matrix_world @ mid
        except Exception:
            pass
    if abs(d.y) < 1e-6:
        return None
    t = (center.y - origin.y) / d.y
    return origin + d * t


# --------------------------------------------------------------- public API
def detect(obj, min_conf=0.25):
    """Run the model on `obj`. Returns dict or None:
        {
          'conf': float,                       # detection confidence
          'points': {joint_name: Vector},      # world points (mid-depth plane)
          'kconf':  {joint_name: float},       # per-joint confidence
          'ground': float, 'top': float, 'h': float,
        }
    Cleans up its temporary camera. Returns None if no model / low confidence.
    """
    if not available():
        return None
    scene = bpy.context.scene
    cam, center = _setup_camera(scene, obj)
    try:
        fp = _render_front(scene, cam)
        rgb = _load_rgb(fp)
        conf, kps = _infer(rgb)
        if conf < min_conf:
            return None
        pts, kc = {}, {}
        for i, name in enumerate(JOINT_NAMES):
            if i >= len(kps):
                break
            kx, ky, c = (float(kps[i][0]), float(kps[i][1]), float(kps[i][2]))
            p = _backproject(cam, center, kx, ky, obj)
            if p is not None:
                pts[name] = p
                kc[name] = c
        lo, hi = _mesh_bounds(obj)
        return {
            "conf": conf, "points": pts, "kconf": kc,
            "ground": float(lo.z), "top": float(hi.z), "h": float(hi.z - lo.z),
        }
    finally:
        c = bpy.data.objects.get("SR_INFER_CAM")
        if c is not None:
            bpy.data.objects.remove(c, do_unlink=True)


def detect_height_fractions(obj, min_conf=0.25, min_kconf=0.30):
    """Convenience: adaptive joint HEIGHTS (z as a fraction of mesh height) that
    feed markers._guess_positions. Only joints detected confidently are
    returned; the geometric guesser keeps its default for the rest.

    Keys returned (when confident): spine_root, neck, hip, ankle.

    NOTE - joint-definition matching (important, learned the hard way):
      Only joints whose TRAINING label matches the addon's MARKER meaning are
      used. pelvis->spine_root, neck->neck, thigh-head->hip, foot-head->ankle all
      match. SHOULDER does NOT: the net's 'shoulder_l' was labelled from the
      clavicle/shoulder bone HEAD (near the sternum - low & central), while the
      addon's shoulder marker is the ARM-START joint (higher & lateral). Using
      the net height there pulls the marker down onto the arm. So shoulder stays
      fully geometric. head_top also stays = mesh top (net 'head' = head centre).
    """
    res = detect(obj, min_conf=min_conf)
    if not res or res["h"] <= 1e-6:
        return None, (res["conf"] if res else 0.0)
    g, h, kc, pts = res["ground"], res["h"], res["kconf"], res["points"]

    def frac(joint):
        if joint in pts and kc.get(joint, 0.0) >= min_kconf:
            return (pts[joint].z - g) / h
        return None

    out = {}
    for marker, joint in (("spine_root", "pelvis"), ("neck", "neck"),
                          ("hip", "hip_l"), ("ankle", "ankle_l")):
        f = frac(joint)
        if f is not None:
            out[marker] = max(0.0, min(1.0, f))
    return (out if out else None), res["conf"]


def detect_fingertips(obj, min_conf=0.25, min_kconf=0.25, res=None):
    """Return {finger_name: world Vector} for LEFT-hand fingertips the model
    detected confidently (thumb/index/middle/ring/pinky). Empty if the model
    has no finger keypoints (old 16-kpt model) or detection is weak. `res` may
    be a precomputed detect() result to avoid re-rendering."""
    if res is None:
        res = detect(obj, min_conf=min_conf)
    if not res:
        return {}
    pts, kc = res["points"], res["kconf"]
    out = {}
    for finger, joint in FINGER_TIP_JOINT.items():
        if joint in pts and kc.get(joint, 0.0) >= min_kconf:
            out[finger] = pts[joint]
    return out
