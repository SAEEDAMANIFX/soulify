bl_info = {
    "name": "Soulify",
    "author": "Saeed",
    "version": (2, 9, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Soulify",
    "description": "Give it a soul. Fit + Rig + Animate: automatic body/garment rigging from markers + mesh geometry, garment fitting, and animation tools.",
    "category": "Rigging",
}

import importlib
# Face v1 (face, storm_face, face_widgets, expressions) was REMOVED from the
# addon on 2026-07-19 - rebuilding the face part-by-part. Those .py files stay
# on disk (parked, unregistered) only so the wipe_face cleanup + lazy imports
# keep working; the new face lives in eye_sample (Part 1 = Eye).
from . import properties, utils, icons, detect, finger_ai, finger_render_ai, markers, fingers_manual, fit, generate, skinning, metarig, skirt, voxelbind, kandura, eye_sample, organize, character, wizard, arp_ai, ui

_modules = [properties, utils, icons, detect, finger_ai, finger_render_ai, markers, fingers_manual, fit, generate, skinning, metarig, skirt, voxelbind, kandura, eye_sample, organize, character, wizard, arp_ai, ui]


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
