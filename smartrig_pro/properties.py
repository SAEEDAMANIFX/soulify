import bpy
from bpy.props import (PointerProperty, IntProperty, BoolProperty,
                       FloatProperty, StringProperty)
from bpy.types import PropertyGroup


def _mesh_poll(self, obj):
    return obj.type == 'MESH'


def _wire_update(self, context):
    """drive the viewport WIREFRAME overlay + its opacity from the addon."""
    scr = getattr(context, "screen", None) or bpy.context.screen
    if scr is None:
        return
    for area in scr.areas:
        if area.type == 'VIEW_3D':
            try:
                ov = area.spaces.active.overlay
                ov.show_wireframes = self.show_wireframe
                ov.wireframe_opacity = self.wireframe_opacity
            except Exception:
                pass


class SmartRigProps(PropertyGroup):
    target_mesh: PointerProperty(
        name="Character Mesh",
        type=bpy.types.Object,
        poll=_mesh_poll,
        description="The body mesh to rig (in A/T-pose, facing -Y)",
    )
    spine_count: IntProperty(name="Spine bones", default=3, min=1, max=8)
    neck_count: IntProperty(name="Neck bones", default=1, min=1, max=4)
    finger_count: IntProperty(
        name="Fingers", default=5, min=0, max=5,
        description="Number of fingers per hand to detect (0 = none). The voxel detector "
                    "traces exactly this many finger tubes from the hand volume.",
    )
    finger_thickness: FloatProperty(
        name="Finger Thickness", default=1.0, min=0.3, max=3.0,
        description="Voxel finger thickness (like Auto-Rig Pro). Increase for thick "
                    "fingers, decrease to separate thin/close fingers.",
    )
    voxel_precision: IntProperty(
        name="Voxel Precision", default=6, min=3, max=10,
        description="Voxel grid resolution for finger detection. Higher resolves gaps "
                    "between close fingers better (slower).",
    )
    auto_fingers: BoolProperty(
        name="Auto-detect fingers & toes", default=False,
        description="If ON, Build tries to detect fingers/toes from the mesh automatically. "
                    "OFF (default): fingers come ONLY from the manual finger markers you add - "
                    "reliable on any hand. Leave OFF unless you want automatic guessing.",
    )
    marker_size: FloatProperty(
        name="Marker Size", default=1.3, min=0.3, max=4.0,
        description="Size of the coloured marker glow in the viewport",
    )
    show_wireframe: BoolProperty(
        name="Wireframe", default=False, update=_wire_update,
        description="Show the mesh wireframe in the viewport while rigging",
    )
    wireframe_opacity: FloatProperty(
        name="Opacity", default=0.31, min=0.0, max=1.0, update=_wire_update,
        description="Wireframe overlay opacity",
    )
    palm_bones: BoolProperty(
        name="Palm bones (metacarpals)", default=True,
        description="Auto-build a palm/metacarpal bone (palm.0N) from the wrist to each "
                    "finger base (except the thumb), like Rigify. Reliable geometry.",
    )
    use_clavicles: BoolProperty(name="Clavicles", default=True)
    mirror: BoolProperty(
        name="Mirror L -> R", default=True,
        description="Place only left-side markers; right side is mirrored across X",
    )
    show_guide: BoolProperty(
        name="Show Guide", default=True,
        description="Show the reference image overlay in the viewport",
    )
    # status / bookkeeping
    active_marker_index: IntProperty(default=0)
    wizard_running: BoolProperty(default=False)
    # ---- guided placement (panel-driven reference, ARP-style) ----
    guide_active: BoolProperty(default=False)
    guide_step: IntProperty(default=0)
    guide_total: IntProperty(default=0)
    guide_label: StringProperty(default="")
    guide_request: StringProperty(default="")   # '', 'cancel', 'back'
    placing: BoolProperty(default=False)         # True while the click modal is active
    # ---- manual finger placement (continuous click per joint) ----
    finger_placing: BoolProperty(default=False)  # True while clicking finger joints
    finger_current: StringProperty(default="")   # name of the finger being placed
    finger_part: StringProperty(default="hand")  # 'hand' or 'foot'


def register():
    bpy.utils.register_class(SmartRigProps)
    bpy.types.Scene.smartrig = PointerProperty(type=SmartRigProps)


def unregister():
    del bpy.types.Scene.smartrig
    bpy.utils.unregister_class(SmartRigProps)
