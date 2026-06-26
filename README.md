# SmartRig Pro

Blender addon that auto-rigs a character mesh from a few user-placed markers plus
mesh geometry — the Auto-Rig Pro "Smart" workflow, rebuilt from scratch. The user
clicks a small number of markers, hits **Go!**, and every joint is derived from
the mesh and a clean reference skeleton + IK/FK control rig is built.

**Version:** 0.76.0 · **Blender:** 4.0+ (tested on 5.1, Python 3.13) · **Author:** Saeed

## Recognition engine

- **Geometric core** — robust mesh-based joint placement (depth/volume centering,
  torso & limb centerlines, anatomical ratios). Locks the lateral (X) and depth (Y)
  of every joint.
- **Neural detection (ONNX)** — optional, runs on CPU via `onnxruntime`:
  - `models/smartrig_pose.onnx` — YOLO-pose, 16+ body keypoints. Supplies adaptive
    joint *heights* that adapt to each character's proportions.
  - `models/finger.onnx` — PointNet finger-joint predictor, competes with the voxel
    finger detector.

## Module map

| File | Responsibility |
|------|----------------|
| `__init__.py` | `bl_info`, register/unregister |
| `properties.py` | `SmartRigProps` (settings: spine/neck/finger counts, voxel precision, clavicles, mirror, overlays) |
| `markers.py` | Interactive click-to-place marker wizard |
| `wizard.py` | ARP-style guided panel placement + GPU viewport overlays |
| `fit.py` | The engine — `compute_joints`, fingers (voxel/topo/AI), `build_reference`, `SMARTRIG_OT_go` |
| `fingers_manual.py` | Manual per-joint finger/foot placement (reliable default) |
| `detect.py` | Neural body-joint detection (ONNX YOLO-pose) |
| `finger_ai.py` / `finger_render_ai.py` | Neural finger detection helpers |
| `generate.py` | `Match to Rig` — reference → IK/FK control rig |
| `skinning.py` | Skinning + IK/FK switch |
| `ui.py` | Sidebar panels |
| `utils.py` | Bone collections, edit-bone/driver/constraint/widget helpers |

## Pipeline

```
markers → Go! (fit.py → SR_Reference) → tweak in Edit Mode → Match to Rig (generate.py → SR_Rig) → skin
```

## Requirements

Neural detection needs `onnxruntime` in Blender's Python:

```
<blender_python> -m pip install onnxruntime
```

The geometric engine works without it.
