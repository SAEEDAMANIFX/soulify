bl_info = {
    "name": "SmartRig Pro",
    "author": "Saeed",
    "version": (1, 19, 15),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > SmartRig",
    "description": "Automatic body rig from markers + mesh geometry, with neural (ONNX) joint-proportion detection.",
    "category": "Rigging",
}

import importlib
from . import properties, utils, icons, detect, finger_ai, finger_render_ai, markers, fingers_manual, fit, generate, skinning, metarig, skirt, wizard, ui

_modules = [properties, utils, icons, detect, finger_ai, finger_render_ai, markers, fingers_manual, fit, generate, skinning, metarig, skirt, wizard, ui]


def register():
    for m in _modules:
        importlib.reload(m)
    for m in _modules:
        if hasattr(m, "register"):
            m.register()


def unregister():
    for m in reversed(_modules):
        if hasattr(m, "unregister"):
            m.unregister()
