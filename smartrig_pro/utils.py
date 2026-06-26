import bpy
import numpy as np
from mathutils import Vector

REF_NAME = "SR_Reference"
RIG_NAME = "SR_Rig"
MARKERS_COLL = "SR_Markers"

# Marker sequence. .L markers are mirrored to .R when props.mirror is on.
MARKER_SEQUENCE = [
    "spine_root", "neck", "head_top",
    "shoulder.L", "wrist.L",
    "shoulder.R", "wrist.R",
    "ankle.L", "ankle.R",
]
LEFT_MARKERS = ["shoulder.L", "wrist.L", "ankle.L"]


# ---------------------------------------------------------------- collections
def ensure_collection(name, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if name not in [c.name for c in parent.children]:
        try:
            parent.children.link(coll)
        except RuntimeError:
            pass
    return coll


def bone_collection(arm_data, name):
    col = arm_data.collections.get(name)
    if col is None:
        col = arm_data.collections.new(name)
    return col


# ---------------------------------------------------------------- mesh reading
def read_world_coords(obj):
    """Return (N,3) numpy array of evaluated world-space vertex coords."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    n = len(me.vertices)
    c = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", c)
    c = c.reshape(n, 3)
    ev.to_mesh_clear()
    mw = np.array(obj.matrix_world)
    co = (np.column_stack([c, np.ones(n)]) @ mw.T)[:, :3]
    return co


# ---------------------------------------------------------------- edit bones
def new_bone(eb, name, head, tail, parent=None, connected=False):
    b = eb.new(name)
    b.head = Vector(head)
    b.tail = Vector(tail)
    if parent is not None:
        b.parent = eb[parent] if isinstance(parent, str) else parent
        b.use_connect = connected
    return b


def set_roll(arm_obj, names, roll_type):
    """Select the given bones (head+tail+bone) and apply calculate_roll."""
    eb = arm_obj.data.edit_bones
    for e in eb:
        e.select = e.select_head = e.select_tail = False
    sel = [eb[n] for n in names if n in eb]
    if not sel:
        return
    for e in sel:
        e.select = e.select_head = e.select_tail = True
    eb.active = sel[0]
    try:
        bpy.ops.armature.calculate_roll(type=roll_type)
    except RuntimeError:
        pass


def coerce(v):
    """Coerce numpy scalars/Vectors to plain python for safe serialization."""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, Vector):
        return [float(x) for x in v]
    return v
