"""Custom SmartRig panel icons, loaded from assets/icons/*.png into a preview
collection. Use icons.get('hand') -> icon_id for layout.*(icon_value=...)."""
import os
import bpy
from bpy.utils import previews

_pcoll = None
ICON_NAMES = ["body", "foot", "hand", "palm", "finger", "marker", "bone", "rig", "skirt", "move"]


def get(name):
    """Return the icon_id for a bundled icon, or 0 if unavailable (safe fallback)."""
    if _pcoll is None:
        return 0
    it = _pcoll.get(name)
    return it.icon_id if it is not None else 0


def register():
    global _pcoll
    try:
        _pcoll = previews.new()
        d = os.path.join(os.path.dirname(__file__), "assets", "icons")
        for n in ICON_NAMES:
            p = os.path.join(d, n + ".png")
            if os.path.exists(p):
                _pcoll.load(n, p, 'IMAGE')
        for fn in ("foot_guide", "foot_ball", "foot_tip"):
            fp = os.path.join(os.path.dirname(__file__), "assets", fn + ".png")
            if os.path.exists(fp):
                _pcoll.load(fn, fp, 'IMAGE')
        for hn in ("hand_palm", "hand_thumb", "hand_index", "hand_middle",
                   "hand_ring", "hand_pinky"):
            hp = os.path.join(os.path.dirname(__file__), "assets", hn + ".png")
            if os.path.exists(hp):
                _pcoll.load(hn, hp, 'IMAGE')
    except Exception as e:
        print("SmartRig: icon load failed:", e)
        _pcoll = None


def unregister():
    global _pcoll
    if _pcoll is not None:
        try:
            previews.remove(_pcoll)
        except Exception:
            pass
        _pcoll = None
