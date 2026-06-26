import bpy
from . import utils
from . import detect
from . import markers


class SMARTRIG_PT_panel(bpy.types.Panel):
    bl_label = "SmartRig Pro"
    bl_idname = "SMARTRIG_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SmartRig"

    def draw(self, context):
        layout = self.layout
        props = context.scene.smartrig
        has_markers = bpy.data.objects.get("spine_root") is not None
        has_ref = bpy.data.objects.get(utils.REF_NAME) is not None

        # ===== GUIDED click-placement: everything lives HERE in the panel =====
        if props.guide_active:
            box = layout.box()
            r = box.row(); r.alignment = 'CENTER'
            r.label(text=props.guide_label, icon='RESTRICT_SELECT_OFF')
            icon = markers.guide_icon(props.guide_step)
            if icon:
                ir = box.row(); ir.alignment = 'CENTER'
                ir.template_icon(icon_value=icon, scale=11.0)
            if props.placing:
                box.label(text="Click each joint on the body", icon='MOUSE_LMB')
                box.label(text="move here to pause / go Back")
            else:
                place = box.row(); place.scale_y = 1.7
                place.operator("smartrig.guide_place", text="Start clicking", icon='RESTRICT_SELECT_OFF')
                box.label(text="then click each joint on the body", icon='MOUSE_LMB')
            nav = box.row(align=True); nav.scale_y = 1.4
            nav.operator("smartrig.guide_back", text="Back", icon='LOOP_BACK')
            nav.operator("smartrig.guide_cancel", text="Cancel", icon='X')
            # Once the ankle (last body joint) is placed: offer fingers / toes / build
            if bpy.data.objects.get("ankle.L") is not None:
                dn = layout.box()
                hr = dn.row(); hr.label(text="Body markers — Done", icon='COLORSET_03_VEC')
                dn.separator()
                dn.label(text="Step 2 — Feet", icon='PLAY')
                dn.operator("smartrig.place_foot", text="Place Foot (Top View)", icon='EMPTY_DATA')
                dn.separator()
                dn.label(text="Step 3 — Hands", icon='PLAY')
                cc = dn.column(align=True); cc.scale_y = 1.25
                op = cc.operator("smartrig.finger_place", text="1) Palm Bones", icon='ADD')
                op.part = "palm"; op.fname = ""
                op = cc.operator("smartrig.finger_place", text="2) Fingers", icon='ADD')
                op.part = "hand"; op.fname = ""
                op = dn.operator("smartrig.finger_place", text="Foot toes (optional)", icon='ADD')
                op.part = "foot"; op.fname = ""
                dn.separator()
                b = dn.row(); b.scale_y = 1.9
                b.operator("smartrig.guide_done", text="Step 4 — Build Reference Bones", icon='BONE_DATA')
            else:
                d = box.row(); d.scale_y = 1.5
                d.operator("smartrig.guide_done", text="Done & Build Skeleton", icon='BONE_DATA')
            return

        # Start: ONLY "Let's Rig" - it rigs whatever character you have selected
        ao = context.active_object
        has_sel_mesh = (ao is not None and ao.type == 'MESH') or \
            any(o.type == 'MESH' for o in context.selected_objects)
        big = layout.row(); big.scale_y = 2.0
        big.operator("smartrig.place_guided", text="Let's Rig", icon='OUTLINER_OB_ARMATURE')
        if not has_sel_mesh:
            layout.label(text="Select your character first", icon='INFO')

        # After the markers are placed: Fingers/Toes -> Build -> (Match/Skin) -> Reset
        if has_markers:
            # ---- MANUAL palm / fingers / toes (after the ankle is placed) ----
            if bpy.data.objects.get("ankle.L") is not None:
                from . import fingers_manual
                col = layout.column()
                arm = bpy.data.objects.get("SR_Reference")
                if arm is not None:
                    hidden = arm.hide_get()
                    col.operator("smartrig.toggle_skeleton",
                                 text=("Show Skeleton" if hidden else "Hide Skeleton"),
                                 icon=('HIDE_ON' if hidden else 'HIDE_OFF'))
                col.operator("smartrig.align_selected",
                             text="Align Selected (one level)", icon='ALIGN_FLUSH')
                col.prop(props, "marker_size", slider=True)
                wr = col.row(align=True)
                wr.prop(props, "show_wireframe", toggle=True, icon='MOD_WIREFRAME')
                sub = wr.row(align=True); sub.active = props.show_wireframe
                sub.prop(props, "wireframe_opacity", text="Opacity", slider=True)

                def digit_rows(box, part):
                    placed = fingers_manual.list_fingers(part, "L")
                    for fn, chain in placed.items():
                        hid = chain[0].hide_get() if chain else False
                        r = box.row(align=True)
                        r.label(text="%s (%d)" % (fn, len(chain)), icon='DOT')
                        op = r.operator("smartrig.finger_hide", text="",
                                        icon=('HIDE_ON' if hid else 'HIDE_OFF'))
                        op.fname = fn; op.part = part
                        op = r.operator("smartrig.finger_straighten", text="", icon='IPO_LINEAR')
                        op.fname = fn; op.part = part
                        op = r.operator("smartrig.finger_remove", text="", icon='TRASH')
                        op.fname = fn; op.part = part
                    if len(placed) >= 2:
                        op = box.operator("smartrig.finger_align_bases", text="Align Bases",
                                          icon='ALIGN_FLUSH')
                        op.part = part

                if props.finger_placing:
                    cur = props.finger_current
                    n = len(fingers_manual.list_fingers(props.finger_part or "hand", "L").get(cur, []))
                    pb = col.box()
                    pb.label(text="Placing %s: %s  (%d joints)" % (props.finger_part, cur, n), icon='REC')
                    pb.label(text="Click joints · Enter = next · Esc = finish", icon='MOUSE_LMB')
                elif props.finger_current:
                    pb = col.box()
                    pb.label(text="Paused: %s" % props.finger_current, icon='PAUSE')
                    r = pb.row(align=True); r.scale_y = 1.3
                    r.operator("smartrig.finger_resume", text="Resume", icon='PLAY')
                    op = r.operator("smartrig.finger_next", text="Next", icon='FRAME_NEXT')
                    op.part = props.finger_part or "hand"
                    r.operator("smartrig.finger_finish", text="Finish", icon='CHECKMARK')
                else:
                    # ===== FEET =====
                    fbox = col.box()
                    fbox.label(text="Feet", icon='VIEW_PERSPECTIVE')
                    has_foot = bpy.data.objects.get("ball.L") is not None
                    fbox.operator("smartrig.place_foot",
                                  text=("Re-place Foot (Top View)" if has_foot else "Place Foot (Top View)"),
                                  icon='EMPTY_DATA')
                    op = fbox.operator("smartrig.finger_place", text="Add Foot Toes", icon='ADD')
                    op.part = "foot"; op.fname = ""
                    digit_rows(fbox, "foot")
                    # ===== HANDS =====
                    hbox = col.box()
                    hbox.label(text="Hands", icon='VIEW_PERSPECTIVE')
                    hbox.prop(props, "palm_bones")
                    rr = hbox.row(align=True); rr.scale_y = 1.3
                    op = rr.operator("smartrig.finger_place", text="1) Palm Bones", icon='ADD')
                    op.part = "palm"; op.fname = ""
                    op = rr.operator("smartrig.finger_place", text="2) Fingers", icon='ADD')
                    op.part = "hand"; op.fname = ""
                    digit_rows(hbox, "palm")
                    digit_rows(hbox, "hand")
                    if (fingers_manual.list_fingers("palm", "L")
                            and fingers_manual.list_fingers("hand", "L")):
                        hbox.operator("smartrig.snap_palm",
                                      text="Snap Fingers ↔ Palm", icon='SNAP_ON')

            # ---- Build the reference skeleton (last step) ----
            b = layout.row(); b.scale_y = 1.8
            b.operator("smartrig.go", text="Build Reference Bones", icon='BONE_DATA')
            if has_ref:
                layout.operator("smartrig.match_to_rig", icon='CONSTRAINT_BONE')
                layout.operator("smartrig.skin", icon='MOD_ARMATURE')
            layout.separator()
            r = layout.row(); r.scale_y = 1.1
            r.operator("smartrig.reset", text="Cancel / Start Over", icon='X')


class SMARTRIG_PT_settings(bpy.types.Panel):
    bl_label = "Settings"
    bl_idname = "SMARTRIG_PT_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SmartRig"
    bl_parent_id = "SMARTRIG_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        # only show Settings AFTER the markers are placed (not on the start screen,
        # not while still clicking joints)
        props = context.scene.smartrig
        return (not props.guide_active) and bpy.data.objects.get("spine_root") is not None

    def draw(self, context):
        props = context.scene.smartrig
        col = self.layout.column(align=True)
        col.prop(props, "spine_count")
        col.prop(props, "neck_count")
        col.prop(props, "use_clavicles")
        col.prop(props, "mirror")
        box = self.layout.box()
        box.label(text="Fingers / Toes", icon='HAND')
        box.prop(props, "auto_fingers")
        if props.auto_fingers:
            fc = box.column(align=True)
            fc.prop(props, "finger_count")
            fc.prop(props, "finger_thickness")
            fc.prop(props, "voxel_precision")
            box.label(text="If fingers merge: lower Thickness,", icon='INFO')
            box.label(text="raise Precision, then Build again.")
        else:
            box.label(text="Add fingers manually in the main panel.", icon='INFO')


classes = (SMARTRIG_PT_panel, SMARTRIG_PT_settings)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
