bl_info = {
    "name": "Soulify",
    "author": "Saeed",
    "version": (1, 99, 20),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Soulify",
    "description": "Give it a soul. Fit + Rig + Animate: automatic body/garment rigging from markers + mesh geometry, garment fitting, and animation tools.",
    "category": "Rigging",
}

import importlib
from . import properties, utils, icons, detect, finger_ai, finger_render_ai, markers, fingers_manual, fit, generate, skinning, metarig, skirt, voxelbind, kandura, face_widgets, face, wizard, arp_ai, ui

_modules = [properties, utils, icons, detect, finger_ai, finger_render_ai, markers, fingers_manual, fit, generate, skinning, metarig, skirt, voxelbind, kandura, face_widgets, face, wizard, arp_ai, ui]


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
