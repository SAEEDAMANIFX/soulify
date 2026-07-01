import bpy
from math import radians
from mathutils import Vector
from . import utils


def _add_ik_controls(rig):
    """In edit mode: add hand/foot IK targets and elbow/knee poles."""
    eb = rig.data.edit_bones
    created = []
    for s in (".L", ".R"):
        if ("forearm" + s) in eb and ("hand" + s) in eb:
            wrist = eb["forearm" + s].tail.copy()
            hand_v = eb["hand" + s]
            length = (hand_v.tail - hand_v.head).length or 0.1
            ik = eb.new("hand_ik" + s)
            ik.head = wrist
            ik.tail = wrist + Vector((0, 0, length))
            elbow = eb["forearm" + s].head.copy()
            pole = eb.new("elbow_pole" + s)
            pole.head = elbow + Vector((0, 0.2, 0))
            pole.tail = elbow + Vector((0, 0.3, 0))
            created += ["hand_ik" + s, "elbow_pole" + s]
        if ("shin" + s) in eb and ("foot" + s) in eb:
            ankle = eb["shin" + s].tail.copy()
            ik = eb.new("foot_ik" + s)
            ik.head = ankle
            ik.tail = eb["foot" + s].tail.copy()
            knee = eb["shin" + s].head.copy()
            pole = eb.new("knee_pole" + s)
            pole.head = knee + Vector((0, -0.2, 0))
            pole.tail = knee + Vector((0, -0.3, 0))
            created += ["foot_ik" + s, "knee_pole" + s]
    # IK controls must not deform
    for nm in created:
        eb[nm].use_deform = False
    return created


def _add_ik_constraints(rig):
    pb = rig.pose.bones
    for s in (".L", ".R"):
        if ("forearm" + s) in pb and ("hand_ik" + s) in pb:
            c = pb["forearm" + s].constraints.new('IK')
            c.target = rig; c.subtarget = "hand_ik" + s
            c.pole_target = rig; c.pole_subtarget = "elbow_pole" + s
            c.pole_angle = radians(-90)
            c.chain_count = 2
        if ("shin" + s) in pb and ("foot_ik" + s) in pb:
            c = pb["shin" + s].constraints.new('IK')
            c.target = rig; c.subtarget = "foot_ik" + s
            c.pole_target = rig; c.pole_subtarget = "knee_pole" + s
            c.pole_angle = radians(90)
            c.chain_count = 2


class SMARTRIG_OT_match_to_rig(bpy.types.Operator):
    bl_idname = "smartrig.match_to_rig"
    bl_label = "Match to Rig (IK/FK)"
    bl_description = "Turn SR_Reference into a posable rig (SR_Rig) with IK on arms and legs"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(utils.REF_NAME) is not None

    def execute(self, context):
        ref = bpy.data.objects.get(utils.REF_NAME)
        if ref is None:
            self.report({'ERROR'}, "Run Go! first."); return {'CANCELLED'}
        old = bpy.data.objects.get(utils.RIG_NAME)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)

        rig = ref.copy()
        rig.data = ref.data.copy()
        rig.name = utils.RIG_NAME
        rig.data.name = utils.RIG_NAME
        context.scene.collection.objects.link(rig)

        for o in context.selected_objects:
            o.select_set(False)
        context.view_layer.objects.active = rig
        rig.select_set(True)

        bpy.ops.object.mode_set(mode='EDIT')
        created = _add_ik_controls(rig)
        ctl = utils.bone_collection(rig.data, "CTL")
        for nm in created:
            if nm in rig.data.edit_bones:
                ctl.assign(rig.data.edit_bones[nm])
        bpy.ops.object.mode_set(mode='POSE')
        _add_ik_constraints(rig)
        bpy.ops.object.mode_set(mode='OBJECT')

        self.report({'INFO'}, "SR_Rig built with IK. Pole angles may need tuning.")
        return {'FINISHED'}


classes = (SMARTRIG_OT_match_to_rig,)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
