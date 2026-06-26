"""
finger_ai.py — neural finger-joint predictor (models/finger.onnx).

A small PointNet trained on legally-sourced rigged hands (VRoid + HumGen) with
per-finger procedural augmentation. Given the hand's mesh point cloud it predicts
5 fingers x 4 joints in the wrist frame. Runs on CPU via onnxruntime (no GPU).

Used by fit.compute_joints as a CANDIDATE that competes with / calibrates the
voxel detector: whichever yields the cleaner finger set wins. Degrades silently
to the voxel engine if the model or onnxruntime is missing.

Output format matches the voxel detector: {name: [tip, near, mid, base]}.
"""
import os
import numpy as np
from mathutils import Vector

NPTS = 256
FIN = ["thumb", "index", "middle", "ring", "pinky"]
MODEL_REL = os.path.join("models", "finger.onnx")
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
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _SESSION = ort.InferenceSession(model_path(), so,
                                        providers=["CPUExecutionProvider"])
    return _SESSION


def _hand_points(co, h, wrist, elbow, side_z):
    """hand-region verts (same region the voxel detector uses)."""
    wn = np.array(wrist, dtype=np.float64)
    d = (wrist - elbow)
    dn = np.array(d.normalized() if d.length > 1e-6 else Vector((1, 0, 0)))
    rel = co - wn
    proj = rel @ dn
    dist = np.linalg.norm(rel, axis=1)
    sel = (proj > -0.02 * h) & (dist < 0.22 * h) & (co[:, 0] * side_z > 0)
    return co[sel]


def ai_fingers(mesh, co, h, wrist, elbow, side_z):
    """Predict finger joints from the hand mesh. {} if unavailable / too few verts."""
    if not available():
        return {}
    try:
        H = _hand_points(co, h, wrist, elbow, side_z)
        if H.shape[0] < 30:
            return {}
        wn = np.array(wrist, dtype=np.float64)
        hs = float(np.max(np.linalg.norm(H - wn, axis=1)))
        if hs < 1e-6:
            return {}
        P = ((H - wn) / hs).astype(np.float32)
        ii = np.random.RandomState(0).choice(P.shape[0], NPTS, replace=P.shape[0] < NPTS)
        out = _session().run(None, {"points": P[ii][None]})[0].reshape(5, 4, 3)
        res = {}
        for i, nm in enumerate(FIN):
            j = out[i] * hs + wn                 # (4,3) world: [root, j1, j2, tip]
            root, j1, j2, tip = [Vector([float(c) for c in p]) for p in j]
            if (tip - root).length < 0.02 * h:
                continue
            res[nm] = [tip, j2, j1, root]        # voxel format: [tip, near, mid, base]
        return res
    except Exception as e:
        print("SmartRig finger_ai:", e)
        return {}
