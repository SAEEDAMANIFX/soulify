import bpy
from . import utils


class SMARTRIG_OT_skin(bpy.types.Operator):
    bl_idname = "smartrig.skin"
    bl_label = "Skin Mesh (automatic weights)"
    bl_description = "Bind the character mesh to SR_Rig with automatic weights"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (bpy.data.objects.get(utils.RIG_NAME) is not None
                and context.scene.smartrig.target_mesh is not None)

    def execute(self, context):
        rig = bpy.data.objects.get(utils.RIG_NAME)
        mesh = context.scene.smartrig.target_mesh
        if rig is None:
            self.report({'ERROR'}, "Run Match to Rig first."); return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='OBJECT')
        for o in context.selected_objects:
            o.select_set(False)
        mesh.select_set(True)
        rig.select_set(True)
        context.view_layer.objects.active = rig
        try:
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        except RuntimeError as e:
            self.report({'ERROR'}, "Auto-weight failed: %s" % str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Mesh skinned to SR_Rig.")
        return {'FINISHED'}


classes = (SMARTRIG_OT_skin,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
