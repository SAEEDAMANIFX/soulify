import bpy
from . import utils
from . import detect
from . import markers
from . import icons


def _step(layout, num, label, icon, state, icon_value=0):
    """Numbered step header box. state: 'done'|'active'|'todo'. Pass either a
    native Blender `icon` string or a custom `icon_value` (preview icon id)."""
    box = layout.box()
    head = box.row(align=True)

    def lab(txt):
        if icon_value:
            head.label(text=txt, icon_value=icon_value)
        elif icon:
            head.label(text=txt, icon=icon)
        else:
            head.label(text=txt)

    if state == 'done':
        lab("%d. %s" % (num, label))
        head.label(text="", icon='CHECKMARK')
        head.active = False
    elif state == 'active':
        head.alert = True
        lab("> %d. %s" % (num, label))
    else:
        head.active = False
        lab("%d. %s" % (num, label))
    return box


class SMARTRIG_PT_panel(bpy.types.Panel):
    bl_label = "SmartRig Pro"
    bl_idname = "SMARTRIG_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SmartRig"

    def draw(self, context):
        layout = self.layout
        props = context.scene.smartrig
        # ===== top tabs (ARP-style: Rig / Skin / Misc) =====
        trow = layout.row(align=True)
        trow.scale_y = 1.3
        trow.prop(props, "ui_tab", expand=True)
        if props.ui_tab == 'SKIN':
            self._draw_skin(layout, context)
            return
        if props.ui_tab == 'MISC':
            self._draw_misc(layout, context)
            return
        # ===== RIG tab =====
        foot_iv = icons.get('foot')
        hand_iv = icons.get('hand')
        has_markers = bpy.data.objects.get("spine_root") is not None
        has_ref = bpy.data.objects.get(utils.REF_NAME) is not None

        # ===== GUIDED click-placement (Step 1: body markers) =====
        if props.guide_active:
            self._marker_tools(layout, context)
            self._align_tools(layout, context)
            body_done = bpy.data.objects.get("ankle.L") is not None
            if not body_done:
                # ---- still placing body markers: show ONLY the body guide ----
                box = layout.box()
                r = box.row(); r.alignment = 'CENTER'
                r.alert = True
                r.label(text=props.guide_label, icon='EMPTY_DATA')
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
                return
            # ---- body markers done: sequential cards (foot -> ask -> hands -> build) ----
            ball_done = bpy.data.objects.get("ball.L") is not None
            tip_done = bpy.data.objects.get("foottip.L") is not None
            foot_done = ball_done and tip_done
            if not foot_done:
                if not ball_done:
                    _flbl = "FEET - click the BALL"; _fimg = icons.get('foot_ball')
                else:
                    _flbl = "FEET - now click the TOE TIP"; _fimg = icons.get('foot_tip')
                lr = layout.row(); lr.alignment = 'CENTER'; lr.alert = True
                lr.label(text=_flbl, icon_value=foot_iv)
                if _fimg:
                    _ir = layout.row(); _ir.alignment = 'CENTER'
                    _ir.template_icon(icon_value=_fimg, scale=13.0)
                pr = layout.row(); pr.scale_y = 1.6
                pr.operator("smartrig.place_foot",
                            text=("Place Foot (Top View)" if not ball_done else "Re-place Foot"),
                            icon_value=foot_iv)
                layout.separator()
                layout.operator("smartrig.guide_cancel", text="Cancel / Start Over", icon='X')
                return
            if not props.hands_decided:
                qb = layout.box()
                hr = qb.row(); hr.alignment = 'CENTER'
                hr.label(text="Foot done!", icon='CHECKMARK')
                qb.label(text="Do you want to rig the hands?")
                qr = qb.row(); qr.scale_y = 1.5
                op = qr.operator("smartrig.choose_hands", text="Yes, rig hands", icon_value=hand_iv)
                op.do_hands = True
                op = qr.operator("smartrig.choose_hands", text="No, skip", icon='X')
                op.do_hands = False
                return
            if props.want_hands:
                _cur = props.finger_current
                if _cur in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                    _hc = icons.get('hand_' + _cur); _hlbl = _cur.upper()
                else:
                    _hc = icons.get('hand_palm'); _hlbl = 'PALM'
                lr = layout.row(); lr.alignment = 'CENTER'; lr.alert = True
                lr.label(text=_hlbl, icon_value=hand_iv)
                if _hc:
                    _hr = layout.row(); _hr.alignment = 'CENTER'
                    _hr.template_icon(icon_value=_hc, scale=13.0)
                cc = layout.column(align=True); cc.scale_y = 1.4
                op = cc.operator("smartrig.finger_place", text="1) Palm Bones (2 clicks)", icon='GROUP_BONE')
                op.part = "palm"; op.fname = ""
                op = cc.operator("smartrig.finger_place", text="2) Fingers (4 clicks)", icon_value=hand_iv)
                op.part = "hand"; op.fname = ""
                self._finger_picker(layout, context)
                from . import fingers_manual as _fm
                self._digit_rows(layout, "palm")
                self._digit_rows(layout, "hand")
                if (_fm.list_fingers("palm", "L") and _fm.list_fingers("hand", "L")):
                    layout.operator("smartrig.snap_palm", text="Snap Fingers <-> Palm", icon='SNAP_ON')
                _ob = layout.row(align=True)
                _ob.label(text="Align:")
                _ob.prop(props, "align_orient", expand=True)
                _al = layout.row(align=True)
                _al.label(text="Sel:")
                for _ax in ('X', 'Y', 'Z'):
                    _op = _al.operator("smartrig.align_selected", text=_ax)
                    _op.axis = _ax; _op.orient = props.align_orient
                layout.separator()
            br = layout.row(); br.scale_y = 1.8
            br.operator("smartrig.build_metarig", text="Build Rigify Metarig", icon='OUTLINER_OB_ARMATURE')
            layout.separator()
            layout.operator("smartrig.guide_cancel", text="Cancel / Start Over", icon='X')
            return

        # ===== START =====
        if not has_markers:
            ao = context.active_object
            has_sel_mesh = (ao is not None and ao.type == 'MESH') or \
                any(o.type == 'MESH' for o in context.selected_objects)
            big = layout.row(); big.scale_y = 2.0
            big.operator("smartrig.place_guided", text="Let's Rig", icon='OUTLINER_OB_ARMATURE')
            if not has_sel_mesh:
                layout.label(text="Select your character first", icon='INFO')
            return

        # ===== AFTER MARKERS: numbered, colored steps =====
        from . import fingers_manual
        has_hands = bool(fingers_manual.list_fingers("palm", "L")
                         or fingers_manual.list_fingers("hand", "L"))

        self._marker_tools(layout, context)
        self._display_tools(layout, context)
        self._align_tools(layout, context)

        # ---- BONE ROLL box (metarig or reference) ----
        if has_ref or bpy.data.objects.get('SR_Metarig') is not None:
            rbx = layout.box()
            _rh = rbx.row(align=True)
            _rh.prop(props, "show_roll", text="Bone Roll", emboss=False,
                     icon=('TRIA_DOWN' if props.show_roll else 'TRIA_RIGHT'))
            _rh.label(text="", icon='CON_ROTLIKE')
            if props.show_roll:
                rbx.label(text="Edit skeleton, select bone(s):", icon='INFO')
                _arm = context.active_object if (context.active_object and context.active_object.type == 'ARMATURE') else (bpy.data.objects.get('SR_Metarig') or bpy.data.objects.get(utils.REF_NAME))
                if _arm is not None:
                    rbx.prop(_arm.data, "show_axes", text="Show Bone Axes", toggle=True, icon='EMPTY_AXIS')
                rbx.prop(props, "roll_axis", text="To")
                rbx.operator("smartrig.roll_recalc", text="Recalculate Roll", icon='CON_ROTLIKE')
                _rn = rbx.row(align=True)
                _o = _rn.operator("smartrig.roll_nudge", text="-5\u00b0"); _o.amount = -0.0872665
                _o = _rn.operator("smartrig.roll_nudge", text="+5\u00b0"); _o.amount = 0.0872665
                rbx.operator("smartrig.roll_clear", text="Clear Roll (0)", icon='X')
                rbx.operator("smartrig.roll_fingers_pro", text="Pro Finger Roll (selected)",
                             icon_value=hand_iv)

        if bpy.data.objects.get('SR_Metarig') is not None:
            self._rigify_samples(layout, context)
            layout.separator()
            _rr = layout.row(); _rr.scale_y = 1.1
            _rr.operator("smartrig.reset", text="Cancel / Start Over", icon='X')
            return

        layout.separator()

        # ---- Step 1: body markers ----
        _step(layout, 1, "Body Markers", 'OUTLINER_OB_MESH', 'done')

        # ---- Step 2: Feet ----
        feet_done = bpy.data.objects.get("ball.L") is not None
        fb = _step(layout, 2, "Feet", None, 'done' if feet_done else 'active', icon_value=foot_iv)
        if not feet_done:
            _gi = icons.get('foot_guide')
            if _gi:
                _ir = fb.row(); _ir.alignment = 'CENTER'
                _ir.template_icon(icon_value=_gi, scale=9.0)
        fb.operator("smartrig.place_foot",
                    text=("Re-place Foot (Top View)" if feet_done else "Place Foot (Top View)"),
                    icon_value=foot_iv)
        op = fb.operator("smartrig.finger_place", text="Add Foot Toes", icon_value=foot_iv)
        op.part = "foot"; op.fname = ""
        self._digit_rows(fb, "foot")

        # ---- Step 3: Hands ----
        hb = _step(layout, 3, "Hands", None,
                   'done' if has_hands else ('active' if feet_done else 'todo'),
                   icon_value=hand_iv)
        _cur = props.finger_current
        if props.finger_part == 'palm' or (_cur and _cur.startswith('palm')):
            _hc = icons.get('hand_palm')
        elif _cur in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            _hc = icons.get('hand_' + _cur)
        else:
            _hc = icons.get('hand_palm')
        if _hc and feet_done and not has_hands:
            _hr = hb.row(); _hr.alignment = 'CENTER'
            _hr.template_icon(icon_value=_hc, scale=9.0)
        hb.prop(props, "palm_bones")
        rr = hb.row(align=True); rr.scale_y = 1.3
        op = rr.operator("smartrig.finger_place", text="1) Palm (2)", icon='GROUP_BONE')
        op.part = "palm"; op.fname = ""
        op = rr.operator("smartrig.finger_place", text="2) Fingers (4)", icon_value=hand_iv)
        op.part = "hand"; op.fname = ""
        self._finger_picker(hb, context)
        if props.finger_placing:
            cur = props.finger_current
            n = len(fingers_manual.list_fingers(props.finger_part or "hand", "L").get(cur, []))
            pb = hb.box(); pb.alert = True
            pb.label(text="Placing %s: %s  (%d joints)" % (props.finger_part, cur, n), icon='REC')
            pb.label(text="Click joints - auto-next - Esc = finish", icon='MOUSE_LMB')
        elif props.finger_current:
            pb = hb.box()
            pb.label(text="Paused: %s" % props.finger_current, icon='PAUSE')
            r = pb.row(align=True); r.scale_y = 1.3
            r.operator("smartrig.finger_resume", text="Resume", icon='PLAY')
            op = r.operator("smartrig.finger_next", text="Next", icon='FRAME_NEXT')
            op.part = props.finger_part or "hand"
            r.operator("smartrig.finger_finish", text="Finish", icon='CHECKMARK')
        self._digit_rows(hb, "palm")
        self._digit_rows(hb, "hand")
        if (fingers_manual.list_fingers("palm", "L")
                and fingers_manual.list_fingers("hand", "L")):
            hb.operator("smartrig.snap_palm", text="Snap Fingers <-> Palm", icon='SNAP_ON')

        # ---- Step 4: Build the Rigify metarig (Rigify only) ----
        bb = _step(layout, 4, "Build Rigify Metarig", 'OUTLINER_OB_ARMATURE', 'active')
        b = bb.row(); b.scale_y = 1.8
        b.operator("smartrig.build_metarig", text="Build Rigify Metarig", icon='OUTLINER_OB_ARMATURE')

        layout.separator()
        r = layout.row(); r.scale_y = 1.1
        r.operator("smartrig.reset", text="Cancel / Start Over", icon='X')

    def _finger_picker(self, layout, context):
        """Pick ANY digit to place / continue, in any order (keeps the names/order)."""
        from . import fingers_manual as _fm
        box = layout.box()
        box.label(text="Pick a digit to place / continue:", icon_value=icons.get('hand'))
        ph = _fm.list_fingers("palm", "L")
        pr = box.row(align=True)
        for nm in ("palm1", "palm2", "palm3", "palm4"):
            n = len(ph.get(nm, []))
            op = pr.operator("smartrig.finger_place",
                             text=(nm + (" OK" if n >= 2 else "")), depress=(n >= 2))
            op.part = "palm"; op.fname = nm
        hh = _fm.list_fingers("hand", "L")
        fc = box.column(align=True)
        for nm in ("thumb", "index", "middle", "ring", "pinky"):
            n = len(hh.get(nm, []))
            op = fc.operator("smartrig.finger_place",
                             text="%s  (%d)" % (nm.capitalize(), n),
                             icon='DOT', depress=(n >= 2))
            op.part = "hand"; op.fname = nm

    # ---- helper: list placed digits with select / hide / straighten / remove ----
    def _digit_rows(self, box, part):
        from . import fingers_manual
        placed = fingers_manual.list_fingers(part, "L")
        for fn, chain in placed.items():
            hid = chain[0].hide_get() if chain else False
            r = box.row(align=True)
            r.label(text="%s (%d)" % (fn, len(chain)), icon='DOT')
            op = r.operator("smartrig.finger_select", text="", icon='RESTRICT_SELECT_OFF')
            op.fname = fn; op.part = part
            op = r.operator("smartrig.finger_hide", text="",
                            icon=('HIDE_ON' if hid else 'HIDE_OFF'))
            op.fname = fn; op.part = part
            op = r.operator("smartrig.finger_straighten", text="", icon='IPO_LINEAR')
            op.fname = fn; op.part = part
            op = r.operator("smartrig.finger_remove", text="", icon='TRASH')
            op.fname = fn; op.part = part
        if len(placed) >= 2:
            op = box.operator("smartrig.finger_align_bases", text="Align Bases", icon='ALIGN_FLUSH')
            op.part = part


    def _rigify_samples(self, layout, context):
        from . import metarig as _mr
        props = context.scene.smartrig
        rig = _mr._generated_rig()
        box = layout.box()
        box.label(text="Rigify", icon='OUTLINER_OB_ARMATURE')

        # If a rig is already generated, lead with the edit round-trip controls.
        if rig is not None:
            st = box.box()
            sr = st.row(); sr.alignment = 'CENTER'
            sr.label(text="Rig generated: %s" % rig.name, icon='CHECKMARK')
            br = st.row(); br.scale_y = 1.5
            br.operator("smartrig.back_to_metarig",
                        text="Back to Metarig (edit more)", icon='LOOP_BACK')

        nav = box.row(align=True)
        nav.operator("smartrig.back_to_markers", text="Back to Markers", icon='LOOP_BACK')
        nav.operator("smartrig.refit_metarig", text="Re-fit to Markers", icon='FILE_REFRESH')
        sh = box.row(align=True)
        sh.prop(props, "show_rigify", text="Add Samples", emboss=False,
                icon=('TRIA_DOWN' if props.show_rigify else 'TRIA_RIGHT'))
        if props.show_rigify:
            box.label(text="Click a sample, position it, then Generate.", icon='INFO')
            expanded = set(g for g in props.samples_expanded.split(",") if g)
            for gname, items in _mr.SAMPLE_GROUPS:
                gb = box.box()
                is_open = gname in expanded
                hd = gb.row(align=True)
                op = hd.operator("smartrig.toggle_sample_group", text=gname, emboss=False,
                                 icon=('TRIA_DOWN' if is_open else 'TRIA_RIGHT'))
                op.group = gname
                if is_open:
                    col = gb.column(align=True)
                    for mtype, label, icon in items:
                        op = col.operator("smartrig.add_sample", text=label, icon=icon)
                        op.metarig_type = mtype

        # ---- Short Skirt (cloth) sample ----
        skirt_iv = icons.get('skirt')
        skb = box.box()
        sk = skb.row(align=True)
        sk.prop(props, "show_skirt", text="Short Skirt (cloth)", emboss=False,
                icon=('TRIA_DOWN' if props.show_skirt else 'TRIA_RIGHT'))
        if skirt_iv:
            sk.label(text="", icon_value=skirt_iv)
        else:
            sk.label(text="", icon='MOD_CLOTH')
        if props.show_skirt:
            src = skb.row(align=True)
            src.prop(props, "skirt_source", expand=True)
            if props.skirt_source == 'SEPARATE':
                skb.prop(props, "skirt_object", text="Skirt")
                skb.label(text="Bones span the mesh top -> hem.", icon='INFO')
            elif props.skirt_source == 'MERGED':
                reg = skb.row(); reg.scale_y = 1.2
                reg.operator("smartrig.register_skirt",
                             text="Register Skirt Selection", icon='GROUP_VERTEX')
                has_vg = (props.target_mesh is not None
                          and props.target_mesh.vertex_groups.get("SR_Skirt") is not None)
                skb.label(text=("Registered - bones span top -> hem." if has_vg
                                else "Edit Mode -> select skirt faces -> Register."),
                          icon=('CHECKMARK' if has_vg else 'INFO'))
            else:  # MANUAL
                skb.label(text="Starter ring; edit bones freely after.", icon='INFO')
            cc = skb.column(align=True)
            cc.prop(props, "skirt_columns")
            cc.prop(props, "skirt_rows")
            if props.skirt_source == 'MANUAL':
                cc.prop(props, "skirt_length", slider=True)
            cc.prop(props, "skirt_collide")
            ar = skb.row(); ar.scale_y = 1.4
            _txt = "Add Starter Ring" if props.skirt_source == 'MANUAL' else "Add Short Skirt"
            if skirt_iv:
                ar.operator("smartrig.add_skirt", text=_txt, icon_value=skirt_iv)
            else:
                ar.operator("smartrig.add_skirt", text=_txt, icon='MOD_CLOTH')
            if props.skirt_source != 'MANUAL':
                skb.label(text="Columns/Rows update live.", icon='FILE_REFRESH')
        box.separator()
        gen = box.row(); gen.scale_y = 1.6
        gen.operator("smartrig.generate",
                     text=("Re-generate Rig" if rig is not None else "Generate Rig"),
                     icon='POSE_HLT')

    def _rig_armature(self, context):
        """The armature the display controls act on: active armature, else
        the metarig, else the reference skeleton."""
        ao = context.active_object
        if ao is not None and ao.type == 'ARMATURE':
            return ao
        return (bpy.data.objects.get('SR_Metarig')
                or bpy.data.objects.get(utils.REF_NAME))

    def _display_tools(self, layout, context):
        """Professional viewport-display controls for the rig (In Front, Axes,
        Names, Display As, Wireframe) - so the user can see bones clearly."""
        props = context.scene.smartrig
        arm = self._rig_armature(context)
        if arm is None:
            return
        box = layout.box()
        h = box.row(align=True)
        h.prop(props, "show_display", text="Display", emboss=False,
               icon=('TRIA_DOWN' if props.show_display else 'TRIA_RIGHT'))
        h.label(text=arm.name, icon='OUTLINER_OB_ARMATURE')
        if not props.show_display:
            return
        # row 1: In Front + Axes
        r1 = box.row(align=True)
        r1.prop(arm, "show_in_front", text="In Front", toggle=True, icon='MOD_OPACITY')
        r1.prop(arm.data, "show_axes", text="Axes", toggle=True, icon='EMPTY_AXIS')
        # row 2: Names + Wireframe (object wire overlay)
        r2 = box.row(align=True)
        r2.prop(arm.data, "show_names", text="Names", toggle=True, icon='SORTALPHA')
        r2.prop(arm, "show_wire", text="Wireframe", toggle=True, icon='MOD_WIREFRAME')
        # Display As (octahedral / stick / wire / b-bone / envelope)
        box.prop(arm.data, "display_type", text="Display As")

    def _align_tools(self, layout, context):
        """Standalone Align + Wireframe section - usable for markers AND while
        adding/positioning Rigify samples (kept separate from Marker Tools)."""
        props = context.scene.smartrig
        box = layout.box()
        h = box.row(align=True)
        h.prop(props, "show_align", text="Align & Wireframe", emboss=False,
               icon=('TRIA_DOWN' if props.show_align else 'TRIA_RIGHT'))
        h.label(text="", icon='SNAP_MIDPOINT')
        if not props.show_align:
            return
        box.label(text="Align selected (markers):", icon='SNAP_MIDPOINT')
        ao = box.row(align=True)
        ao.label(text="Space:")
        ao.prop(props, "align_orient", expand=True)
        al = box.row(align=True); al.scale_y = 1.2
        al.label(text="Flatten:")
        for ax in ('X', 'Y', 'Z'):
            op = al.operator("smartrig.align_selected", text=ax)
            op.axis = ax; op.orient = props.align_orient
        box.separator()
        wr = box.row(align=True)
        wr.prop(props, "show_wireframe", text="Character Wireframe",
                toggle=True, icon='MOD_WIREFRAME')
        sub = wr.row(align=True); sub.active = props.show_wireframe
        sub.prop(props, "wireframe_opacity", text="Opacity", slider=True)

    def _marker_tools(self, layout, context):
        props = context.scene.smartrig
        has_ref = bpy.data.objects.get(utils.REF_NAME) is not None
        tb = layout.box()
        _th = tb.row(align=True)
        _th.prop(props, "show_tools", text="Marker Tools", emboss=False,
                 icon=('TRIA_DOWN' if props.show_tools else 'TRIA_RIGHT'))
        _th.label(text="", icon='EMPTY_DATA')
        if not props.show_tools:
            return
        tb.prop(props, "lock_mesh", toggle=True,
                text=("Mesh Locked (markers only)" if props.lock_mesh else "Lock Mesh Selection"),
                icon=('LOCKED' if props.lock_mesh else 'UNLOCKED'))
        vis = tb.row(align=True)
        vis.operator("smartrig.toggle_markers",
                     text=("Show Markers" if props.markers_hidden else "Hide Markers"),
                     icon=('HIDE_ON' if props.markers_hidden else 'HIDE_OFF'))
        if has_ref:
            arm = bpy.data.objects.get(utils.REF_NAME)
            hidden = arm.hide_get() if arm else False
            vis.operator("smartrig.toggle_skeleton",
                         text=("Show Skeleton" if hidden else "Hide Skeleton"),
                         icon=('HIDE_ON' if hidden else 'HIDE_OFF'))
        er = tb.row(align=True)
        er.operator("smartrig.edit_markers", text="Edit Markers", icon='GREASEPENCIL')
        er.operator("smartrig.reset_markers", text="Reset to Default", icon='LOOP_BACK')
        # ---- re-fit the metarig after editing markers ----
        if bpy.data.objects.get('SR_Metarig') is not None:
            rf = tb.row(); rf.scale_y = 1.3
            rf.operator("smartrig.refit_metarig",
                        text="Re-fit Metarig to Markers", icon='FILE_REFRESH')
        tb.prop(props, "marker_size", slider=True)
        ll = tb.row()
        ll.operator("smartrig.live_link",
                    text=("Live Link: ON" if props.live_link else "Live Link: OFF"),
                    icon=('LINKED' if props.live_link else 'UNLINKED'),
                    depress=props.live_link)
        if has_ref:
            tb.operator("smartrig.sync_from_skeleton",
                        text="Sync Markers from Skeleton", icon='UV_SYNC_SELECT')


    def _draw_misc(self, layout, context):

        props = context.scene.smartrig
        col = layout.column(align=True)
        col.prop(props, "spine_count")
        col.prop(props, "neck_count")
        col.prop(props, "use_clavicles")
        col.prop(props, "mirror")
        box = layout.box()
        box.label(text="Fingers / Toes", icon_value=icons.get('hand'))
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

    def _draw_skin(self, layout, context):

        from . import metarig as _mr
        props = context.scene.smartrig
        rig = _mr._generated_rig()
        mesh = props.target_mesh

        layout.prop(props, "target_mesh", text="Mesh")
        if rig is None:
            layout.label(text="Generate the rig first.", icon='INFO')
            return

        eng = layout.box()
        eng.label(text="Bind Engine", icon='MOD_VERTEX_WEIGHT')
        eng.prop(props, "skin_engine", text="Engine")
        eng.prop(props, "skin_split_parts")
        eng.prop(props, "skin_preserve_volume")

        # skirt-aware: if Split Parts on and a skirt exists but isn't located, ask
        has_skirt = any(b.name.startswith("DEF-skirt.") for b in rig.pose.bones)
        knows_skirt = (props.skirt_source == 'SEPARATE' and props.skirt_object is not None) \
            or (props.skirt_source != 'SEPARATE' and mesh is not None
                and mesh.vertex_groups.get("SR_Skirt") is not None)
        ready = True
        if has_skirt and props.skin_split_parts and not knows_skirt:
            sk = layout.box()
            sk.label(text="Where is the skirt?", icon='INFO')
            if props.skirt_source == 'SEPARATE':
                sk.prop(props, "skirt_object", text="Skirt")
            else:
                sk.operator("smartrig.register_skirt",
                            text="Register Skirt Selection", icon='GROUP_VERTEX')
            ready = knows_skirt

        bnd = layout.box()
        bnd.label(text="Bind", icon='POSE_HLT')
        row = bnd.row(align=True); row.scale_y = 1.6
        sub = row.row(align=True); sub.enabled = ready
        sub.operator("smartrig.bind", text="Bind", icon='MOD_VERTEX_WEIGHT')
        row.operator("smartrig.unbind", text="Unbind", icon='X')
        if has_skirt and props.skin_split_parts:
            bnd.label(text="Body ignores skirt bones; skirt follows its own.", icon='INFO')

        # ---- Skirt leg collision (ARP Kilt-style) ----
        if has_skirt:
            cb = layout.box()
            cb.label(text="Skirt Leg Collision", icon='MOD_PHYSICS')
            cb.prop(props, "skirt_collide", text="Collide with Legs")
            if props.skirt_collide:
                cr = cb.row(); cr.scale_y = 1.5
                cr.operator("smartrig.skirt_collision",
                            text="Apply Leg Collision", icon='MOD_PHYSICS')
                col = cb.column(align=True)
                col.prop(props, "skirt_collide_dist", text="Swing", slider=True)
                col.prop(props, "skirt_collide_spread", text="Strength", slider=True)
                cb.label(text="Animate live: N-panel → Item → Short Skirt.",
                         icon='INFO')
                # advanced: which leg bones to collide with (defaults are correct)
                adv = cb.box()
                ah = adv.row(align=True)
                ah.prop(props, "show_skirt_adv", text="Advanced", emboss=False,
                        icon=('TRIA_DOWN' if props.show_skirt_adv else 'TRIA_RIGHT'))
                if props.show_skirt_adv:
                    adv.prop_search(props, "skirt_collider_l", rig.pose, "bones", text="Left Leg")
                    adv.prop_search(props, "skirt_collider_r", rig.pose, "bones", text="Right Leg")


class SMARTRIG_PT_skirt_item(bpy.types.Panel):
    """Short-skirt leg-collision settings, live in the N-panel Item tab so the
    animator can tweak (and keyframe) them while posing."""
    bl_label = "Short Skirt"
    bl_idname = "SMARTRIG_PT_skirt_item"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Item"

    @classmethod
    def poll(cls, context):
        from . import skirt
        return skirt.kilt_rig(context) is not None

    def draw(self, context):
        from . import skirt
        layout = self.layout
        rig = skirt.kilt_rig(context)
        if rig is None:
            return
        mpb = rig.pose.bones.get("SKC_master")
        col = layout.column(align=True)
        col.label(text="Leg Collision", icon='MOD_CLOTH')
        if mpb is None:
            col.label(text="Apply Leg Collision first.", icon='INFO')
            return
        try:
            col.prop(mpb, '["collide"]', text="Collide", slider=True)
            col.prop(mpb, '["collide_dist"]', text="Swing", slider=True)
            col.prop(mpb, '["collide_spread"]', text="Strength", slider=True)
        except Exception:
            col.label(text="Re-apply Leg Collision.", icon='ERROR')
        col.separator()
        col.label(text="Keyframe these to animate the cloth.", icon='KEYTYPE_KEYFRAME_VEC')


classes = (SMARTRIG_PT_panel, SMARTRIG_PT_skirt_item)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
