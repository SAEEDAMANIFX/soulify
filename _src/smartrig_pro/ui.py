import bpy
from . import utils
from . import detect
from . import markers
from . import icons


def _wf_draw_folder(container, props, rig, mesh, sel, skf, fi, fol):
    """Draw one weight folder recursively (nesting subfolders inside)."""
    bx = container.box()
    hd = bx.row(align=True)
    hd.prop(fol, "expanded", text="",
            icon=('TRIA_DOWN' if fol.expanded else 'TRIA_RIGHT'), emboss=False)
    active = (props.weight_folders_index == fi)
    hd.operator("smartrig.wf_select", text="",
                icon=('RADIOBUT_ON' if active else 'RADIOBUT_OFF'),
                emboss=False, depress=active).index = fi
    hd.prop(fol, "name", text="")
    hd.label(text="%d" % len(skf._wf_descendant_bones(props, fol)))
    _ln, _tn = skf._wf_lock_stats(props, mesh, fol)
    if _ln:
        _ind = hd.row(align=True)
        _ind.alert = (_ln == _tn)
        _ind.label(text=str(_ln), icon='LOCKED')
    hd.operator("smartrig.wf_isolate", text="",
                icon=('RESTRICT_VIEW_OFF'
                      if props.weight_isolated_folder == fol.uid else 'VIEWZOOM'),
                depress=(props.weight_isolated_folder == fol.uid)).index = fi
    hd.operator("smartrig.wf_select_verts", text="", icon='VERTEXSEL').index = fi
    _allk = (_tn > 0 and _ln == _tn)
    _lk = hd.operator("smartrig.wf_lock", text="",
                      icon=('LOCKED' if _allk else 'UNLOCKED'), depress=_allk)
    _lk.index = fi; _lk.lock = (not _allk)
    hd.operator("smartrig.wf_assign", text="", icon='ADD').index = fi
    hd.operator("smartrig.wf_new_sub", text="", icon='NEWFOLDER').parent = fol.uid
    hd.operator("smartrig.wf_delete", text="", icon='X').index = fi
    if fol.expanded:
        mem = [m for m in fol.members.split(",") if m]
        has_child = (any(f.parent == fol.uid for f in props.weight_folders)
                     if fol.uid else False)
        if not mem and not has_child:
            bx.label(text="Empty - pick a bone, press +", icon='DOT')
        for bn in mem:
            if rig.data.bones.get(bn) is None:
                continue
            r = bx.row(align=True)
            r.separator(factor=1.4)
            op = r.operator("smartrig.wf_pick", text=skf._wt_pretty(bn),
                            icon='BONE_DATA', depress=(bn == sel))
            op.bone = bn
            vg = mesh.vertex_groups.get(bn)
            if vg is not None:
                r.prop(vg, "lock_weight", text="", emboss=False,
                       icon=('LOCKED' if vg.lock_weight else 'UNLOCKED'))
            rm = r.operator("smartrig.wf_remove_bone", text="", icon='PANEL_CLOSE')
            rm.index = fi; rm.bone = bn
        if fol.uid:
            for cj, cf in enumerate(props.weight_folders):
                if cf.parent == fol.uid:
                    _wf_draw_folder(bx, props, rig, mesh, sel, skf, cj, cf)


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


class SMARTRIG_PT_sleeve_item(bpy.types.Panel):
    """ARP Cloth-Kilt style live settings in the N-panel > Item tab:
    appears on the generated rig, one block per sleeve master."""
    bl_label = "Soulify Sleeves"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Item'

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob is not None and ob.type == 'ARMATURE'
                and (ob.pose.bones.get("kan_rollup.L") is not None
                     or ob.pose.bones.get("kan_rollup.R") is not None))

    def draw(self, context):
        lay = self.layout
        ob = context.active_object
        props = context.scene.smartrig
        for side, label in (("L", "Left Sleeve"), ("R", "Right Sleeve")):
            pb = ob.pose.bones.get("kan_rollup." + side)
            if pb is None:
                continue
            box = lay.box()
            box.label(text=label, icon='MOD_CLOTH')
            r = box.row(align=True)
            if "roll_up" in pb.keys():
                r.prop(pb, '["roll_up"]', text="Gather Sleeve", slider=True)
            col = box.column(align=True)
            if "cuff_collide" in pb.keys():
                col.prop(pb, '["cuff_collide"]', text="Hand Collide Strength",
                         slider=True)
            if "cuff_dist" in pb.keys():
                col.prop(pb, '["cuff_dist"]', text="Hand Collide Distance",
                         slider=True)
            if "hand_clear" in pb.keys():
                col.prop(pb, '["hand_clear"]', text="Hand Clearance (retreat)",
                         slider=True)
            col2 = box.column(align=True)
            if "bulge" in pb.keys():
                col2.prop(pb, '["bulge"]', text="Gather Thickness", slider=True)
            if "inflate" in pb.keys():
                col2.prop(pb, '["inflate"]', text="Push Out (inflate)",
                          slider=True)
            if "hand_follow" in pb.keys():
                col2.prop(pb, '["hand_follow"]', text="Soft Hand Follow",
                          slider=True)
        try:
            from . import kandura as _kn
            k_ob = _kn.kandura_object(context)
            if k_ob is not None and k_ob.modifiers.get("KAN_AntiPen"):
                bx = lay.box()
                bx.label(text="Kandura", icon='OUTLINER_OB_FORCE_FIELD')
                bx.prop(props, "kandura_antipen_offset",
                        text="Body Clearance", slider=True)
                if k_ob.modifiers.get("KAN_Smooth"):
                    bx.prop(props, "kandura_smooth",
                            text="Cloth Smoothing", slider=True)
        except Exception:
            pass


class SMARTRIG_PT_panel(bpy.types.Panel):
    bl_label = "Soulify"
    bl_idname = "SMARTRIG_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Soulify"

    def draw(self, context):
        layout = self.layout
        props = context.scene.smartrig
        # ===== header: the FIT / RIG / ANIMATE pipeline + Simple|Pro level =====
        tabs = layout.row(align=True)
        tabs.scale_y = 1.5
        tabs.prop(props, "ui_tab", expand=True)
        lv = layout.row(align=True)
        lv.scale_y = 0.9
        lv.prop(props, "ui_level", expand=True)
        layout.separator(factor=0.4)
        if props.ui_tab == 'FIT':
            self._draw_fit(layout, context)
            return
        if props.ui_tab == 'ANIM':
            self._draw_animate(layout, context)
            return
        # ===== RIG phase =====
        _rig_obj = None
        try:
            from .metarig import _generated_rig
            _rig_obj = _generated_rig()
        except Exception:
            _rig_obj = None
        _has_mk = bpy.data.objects.get("spine_root") is not None
        _started = (props.rig_started or _has_mk or _rig_obj is not None
                    or props.guide_active or bpy.data.objects.get("SR_Metarig") is not None)
        if not _started:
            ao = context.active_object
            has_sel_mesh = (ao is not None and ao.type == 'MESH') or \
                any(o.type == 'MESH' for o in context.selected_objects)
            big = layout.row(); big.scale_y = 2.0
            big.prop(props, "rig_started", text="Let's Rig", toggle=True,
                     icon='OUTLINER_OB_ARMATURE')
            if not has_sel_mesh:
                layout.label(text="Select your character first", icon='INFO')
            return
        # ===== after Let's Rig, before choosing: show ONLY the question =====
        _chosen = (props.mode_chosen or _has_mk or _rig_obj is not None
                   or props.guide_active or bpy.data.objects.get("SR_Metarig") is not None)
        if not _chosen:
            qb = layout.box()
            qb.label(text="What are you rigging?", icon='QUESTION')
            cr = qb.column(align=True)
            c1 = cr.row(); c1.scale_y = 1.6
            c1.operator("smartrig.pick_mode", text="Character (humanoid)",
                        icon='OUTLINER_OB_ARMATURE').mode = 'CHARACTER'
            c2 = cr.row(); c2.scale_y = 1.6
            c2.operator("smartrig.pick_mode", text="Parts (skirt / cloth / ...)",
                        icon='MOD_CLOTH').mode = 'PARTS'
            return
        # ===== mode chosen: Build | Skin sections, then the tools =====
        srow = layout.row(align=True)
        srow.scale_y = 1.2
        srow.prop(props, "rig_sub", expand=True)
        if props.rig_sub == 'SKIN':
            self._draw_skin(layout, context)
            return
        mrow = layout.row(align=True)
        mrow.prop(props, "rig_mode", expand=True)
        if props.rig_mode == 'PARTS':
            self._draw_parts(layout, context)
            return
        # ===== RIG tab (Character) =====
        foot_iv = icons.get('foot')
        hand_iv = icons.get('hand')
        has_markers = bpy.data.objects.get("spine_root") is not None
        has_ref = bpy.data.objects.get(utils.REF_NAME) is not None

        # ===== GUIDED click-placement (Step 1: body markers) =====
        if props.guide_active:
            self._marker_tools(layout, context)
            if props.ui_level == 'PRO':
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
            big.operator("smartrig.place_guided", text="Place Body Markers", icon='OUTLINER_OB_ARMATURE')
            if not has_sel_mesh:
                layout.label(text="Select your character first", icon='INFO')
            return

        # ===== AFTER MARKERS: numbered, colored steps =====
        from . import fingers_manual
        has_hands = bool(fingers_manual.list_fingers("palm", "L")
                         or fingers_manual.list_fingers("hand", "L"))

        self._marker_tools(layout, context)
        if props.ui_level == 'PRO':
            self._display_tools(layout, context)
            self._align_tools(layout, context)

        # ---- BONE ROLL box (metarig or reference; Pro only) ----
        if props.ui_level == 'PRO' and (
                has_ref or bpy.data.objects.get('SR_Metarig') is not None):
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

        # ---- Options (under Bone Roll); metarig built + Pro only ----
        if props.ui_level == 'PRO' and bpy.data.objects.get('SR_Metarig') is not None:
            _ob = layout.box()
            _oh = _ob.row(align=True)
            _oh.prop(props, "show_options", text="Options", emboss=False,
                     icon=('TRIA_DOWN' if props.show_options else 'TRIA_RIGHT'))
            _oh.label(text="", icon='PREFERENCES')
            if props.show_options:
                self._draw_misc(_ob, context)

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
            self._draw_skirt_settings(skb, context, props, standalone=False)
            # Remove all skirt bones + extras (shown only when a skirt exists)
            rr = skb.row()
            rr.enabled = True
            rr.operator("smartrig.remove_skirt", text="Remove Skirt", icon='TRASH')
            # PROACTIVE warning if the user deleted skirt bones and broke a column
            try:
                from . import skirt as _sk
                _meta = bpy.data.objects.get('SR_Metarig')
                _probs = _sk.check_skirt_integrity(_meta)
                if _probs:
                    wb = skb.box(); wb.alert = True
                    wb.label(text="Broken skirt column(s): %s"
                             % ", ".join(str(c) for c, _ in _probs), icon='ERROR')
                    wb.label(text="A column needs a connected chain of 2+ bones.")
                    wb.label(text="Fix: delete the WHOLE column, or Remove Skirt &")
                    wb.label(text="rebuild, or use Rows/Columns. Not single bones.")
            except Exception:
                pass
            # ---- Leg Collision (RIGGING: build the compass collision rig) ----
            if rig is not None and any(b.name.startswith("DEF-skirt.")
                                       for b in rig.pose.bones):
                lc = skb.box()
                lc.label(text="Leg Collision", icon='MOD_PHYSICS')
                lc.prop(props, "skirt_collide", text="Collide with Legs")
                if props.skirt_collide:
                    cr = lc.row(); cr.scale_y = 1.4
                    cr.operator("smartrig.skirt_collision",
                                text="Apply Leg Collision", icon='MOD_PHYSICS')
                    lc.label(text="Tune/animate live: N-panel → Item → Short Skirt.",
                             icon='INFO')
                    adv = lc.box()
                    ah = adv.row(align=True)
                    ah.prop(props, "show_skirt_adv", text="Advanced (leg bones)",
                            emboss=False,
                            icon=('TRIA_DOWN' if props.show_skirt_adv else 'TRIA_RIGHT'))
                    if props.show_skirt_adv:
                        adv.prop_search(props, "skirt_collider_l", rig.pose, "bones",
                                        text="Left Leg")
                        adv.prop_search(props, "skirt_collider_r", rig.pose, "bones",
                                        text="Right Leg")
                # ---- Region Masters (front/sides/back; user can increase) ----
                mbx = skb.box()
                mbx.label(text="Region Masters", icon='GROUP_BONE')
                mbx.prop(props, "skirt_use_masters", text="Use Region Masters")
                if props.skirt_use_masters:
                    mbx.prop(props, "skirt_masters", text="Sectors")
                    has_m = rig.get("sk_masters")
                    mr = mbx.row(align=True); mr.scale_y = 1.3
                    mr.operator("smartrig.skirt_masters",
                                text=("Rebuild Masters" if has_m else "Build Masters"),
                                icon='GROUP_BONE')
                    if has_m:
                        op = mr.operator("smartrig.skirt_masters", text="", icon='X')
                        op.remove = True
                        mbx.label(text="Pose whole regions; controls in 'Skirt (Master)'.",
                                  icon='INFO')
                    else:
                        mbx.label(text="1 global + N region controls at the waist.",
                                  icon='INFO')
                # ---- Skirt Jiggle (live secondary motion / spring) ----
                jb = skb.box()
                jb.label(text="Jiggle (secondary motion)", icon='PHYSICS')
                if not rig.get("sk_jiggle"):
                    jr = jb.row(); jr.scale_y = 1.4
                    jr.operator("smartrig.skirt_jiggle", text="Apply Jiggle", icon='PHYSICS')
                else:
                    jc = jb.column(align=True)
                    jc.prop(props, "jiggle_amount", slider=True)
                    jc.prop(props, "jiggle_stiffness", slider=True)
                    jc.prop(props, "jiggle_damping", slider=True)
                    jb.label(text="Play timeline to see it. Live tune in Item.", icon='INFO')
                    br = jb.row(align=True)
                    br.operator("smartrig.bake_jiggle", text="Bake to Keyframes", icon='ACTION')
                    op = br.operator("smartrig.skirt_jiggle", text="Remove", icon='X')
                    op.remove = True
                # ---- Follow Body + Anti-Penetration: SEPARATE skirt ONLY ----
                # (they're Surface-Deform / Shrinkwrap modifiers that need a
                # separate target mesh; hidden entirely for a merged skirt so the
                # panel only shows what actually works.)
                from . import skirt as _sk
                _is_sep = (props.skirt_source == 'SEPARATE' and props.skirt_object is not None)
                if _is_sep:
                    fbx = skb.box()
                    fbx.label(text="Follow Body (sitting)", icon='MOD_MESHDEFORM')
                    _fmod = _sk.follow_modifier(context)
                    if not rig.get("sk_follow") or _fmod is None:
                        fr = fbx.row(); fr.scale_y = 1.3
                        fr.operator("smartrig.skirt_follow", text="Apply Body Follow",
                                    icon='MOD_MESHDEFORM')
                    else:
                        fbx.prop(props, "skirt_follow_body", text="Follow Body", slider=True)
                        if _sk.follow_status(context)[0] == 'subsurf_above':
                            w = fbx.box(); w.alert = True
                            w.label(text="A modifier is above Follow — Re-bind:", icon='ERROR')
                            w.operator("smartrig.skirt_follow", text="Re-bind (fix order)", icon='FILE_REFRESH')
                        else:
                            fbx.label(text="0 = skirt rig, 1 = clings to body (sitting).", icon='INFO')
                        rb = fbx.row(align=True)
                        rb.operator("smartrig.skirt_follow", text="Re-bind", icon='FILE_REFRESH')
                        op = rb.operator("smartrig.skirt_follow", text="Remove", icon='X')
                        op.remove = True
                    # ---- Anti-Penetration (Shrinkwrap Outside) ----
                    apx = skb.box()
                    apx.label(text="Anti-Penetration", icon='MOD_SHRINKWRAP')
                    if not rig.get("sk_antipen"):
                        ar = apx.row(); ar.scale_y = 1.3
                        ar.operator("smartrig.skirt_antipen", text="Apply Anti-Penetration",
                                    icon='MOD_SHRINKWRAP')
                    else:
                        apx.prop(props, "skirt_antipen_offset", text="Offset", slider=True)
                        apx.label(text="Pushes only penetrating verts out.", icon='INFO')
                        op = apx.operator("smartrig.skirt_antipen", text="Remove", icon='X')
                        op.remove = True
                    # ---- Corrective Smooth (relaxes Follow/Anti-Pen pinching) ----
                    smx = skb.box()
                    smx.label(text="Corrective Smooth", icon='MOD_SMOOTH')
                    _smod = props.skirt_object.modifiers.get("SK_Smooth") if props.skirt_object else None
                    if _smod is None:
                        sr = smx.row(); sr.scale_y = 1.2
                        sr.operator("smartrig.skirt_smooth", text="Add Corrective Smooth",
                                    icon='MOD_SMOOTH')
                        smx.label(text="Great with Follow Body; stays before Anti-Pen.",
                                  icon='INFO')
                    else:
                        smx.prop(props, "skirt_smooth_factor", text="Smooth", slider=True)
                        smx.prop(props, "skirt_smooth_iter", text="Iterations")
                        op = smx.operator("smartrig.skirt_smooth", text="Remove", icon='X')
                        op.remove = True
                    # ---- modifier-ORDER guard: warn + one-click fix ----
                    _ok, _omsg = _sk.skirt_mods_order_ok(props)
                    if not _ok:
                        ob = skb.box(); ob.alert = True
                        ob.label(text="Skirt modifier order changed!", icon='ERROR')
                        ob.label(text=_omsg)
                        ofr = ob.row(); ofr.scale_y = 1.3
                        ofr.operator("smartrig.skirt_fix_order",
                                     text="Fix Modifier Order", icon='SORTSIZE')
        # ---- Kandura (Emirati thobe) ----
        knb = box.box()
        kh = knb.row(align=True)
        kh.prop(props, "show_kandura", text="Kandura (thobe)", emboss=False,
                icon=('TRIA_DOWN' if props.show_kandura else 'TRIA_RIGHT'))
        kh.label(text="", icon='MOD_CLOTH')
        if props.show_kandura:
            from . import kandura as _kn
            k_ob = _kn.kandura_object(context)
            knb.prop(props, "kandura_object", text="Mesh")
            if k_ob is None:
                knb.label(text="Pick / select the kandura mesh first.", icon='INFO')
            else:
                # ---- Align to Surface + Mirror: on/off systems ----
                al = knb.row(align=True); al.scale_y = 1.3
                al.prop(props, "kandura_align_surface",
                        text=("Align to Surface: ON"
                              if props.kandura_align_surface
                              else "Align to Surface: OFF"),
                        toggle=True,
                        icon=('SNAP_ON' if props.kandura_align_surface
                              else 'SNAP_OFF'))
                mi = knb.row(align=True); mi.scale_y = 1.3
                mi.prop(props, "kandura_mirror",
                        text=("Mirror: ON" if props.kandura_mirror
                              else "Mirror: OFF"),
                        toggle=True,
                        icon=('MOD_MIRROR' if props.kandura_mirror
                              else 'X'))
                fo = knb.row(align=True); fo.scale_y = 1.3
                fo.prop(props, "kandura_focus",
                        text=("Body Bones: HIDDEN" if props.kandura_focus
                              else "Body Bones: SHOWN"),
                        toggle=True,
                        icon=('HIDE_ON' if props.kandura_focus
                              else 'HIDE_OFF'))
                if context.mode == 'EDIT_ARMATURE':
                    an = knb.column(align=True); an.scale_y = 1.2
                    an.operator("smartrig.kandura_align_now",
                                text="Align Selected to Surface",
                                icon='MOD_SHRINKWRAP')
                    an.operator("smartrig.kandura_mirror_now",
                                text="Mirror Selected to Other Side",
                                icon='MOD_MIRROR')
                knb.separator()
                # ---- Waist-down ----
                st = knb.column(align=True)
                st.label(text="Waist-down:", icon='MOD_CLOTH')
                sr_ = st.row(align=True)
                sr_.prop(props, "kandura_columns", text="Columns")
                sr_.prop(props, "kandura_rows", text="Rows x2")
                bw = st.row(align=True); bw.scale_y = 1.5
                bw.operator("smartrig.kandura_add_waist",
                            text="Add Waist-Down Bones", icon='ADD')
                _ow = bw.operator("smartrig.kandura_remove", text="",
                                  icon='TRASH')
                _ow.part = 'WAIST'
                rw_ = st.row(align=True); rw_.scale_y = 1.2
                rw_.operator("smartrig.kandura_waist_register",
                             text="Register from Loop (Edit Mode)",
                             icon='SNAP_EDGE')
                # ---- Sleeves ----
                st.separator()
                st.label(text="Sleeves:", icon='BONE_DATA')
                sl_ = st.row(align=True)
                sl_.prop(props, "kandura_sleeve_upper", text="Upper Arm")
                sl_.prop(props, "kandura_sleeve_lower", text="Lower Arm")
                bs = st.row(align=True); bs.scale_y = 1.5
                bs.operator("smartrig.kandura_add_sleeves",
                            text="Add Sleeve Bones", icon='ADD')
                _os = bs.operator("smartrig.kandura_remove", text="",
                                  icon='TRASH')
                _os.part = 'SLEEVES'
                # ---- Collar ----
                st.separator()
                st.label(text="Collar:", icon='MESH_CIRCLE')
                cl_ = st.row(align=True)
                cl_.prop(props, "kandura_collar_count", text="Bones")
                bc = st.row(align=True); bc.scale_y = 1.5
                bc.operator("smartrig.kandura_add_collar",
                            text="Add Collar Bones", icon='ADD')
                _oc = bc.operator("smartrig.kandura_remove", text="",
                                  icon='TRASH')
                _oc.part = 'COLLAR'
                # ---- Cuffs ----
                st.separator()
                st.label(text="Cuffs (wrist):", icon='MESH_CIRCLE')
                cf_ = st.row(align=True)
                cf_.prop(props, "kandura_cuff_count", text="Bones")
                cf_.prop(props, "kandura_cuff_rows", text="Rows")
                bf = st.row(align=True); bf.scale_y = 1.5
                bf.operator("smartrig.kandura_add_cuffs",
                            text="Add Cuff Bones", icon='ADD')
                rl = st.row(align=True); rl.scale_y = 1.2
                rl.operator("smartrig.kandura_cuffs_register",
                            text="Register from Loop (Edit Mode)",
                            icon='SNAP_EDGE')
                _of = bf.operator("smartrig.kandura_remove", text="",
                                  icon='TRASH')
                _of.part = 'CUFF'
                knb.label(text="Re-Add keeps your manual placement.",
                          icon='INFO')
                pw = knb.row(); pw.scale_y = 1.3
                pw.operator("smartrig.kandura_polish_weights",
                            text="Polish Sleeve Weights (bind fix)",
                            icon='MOD_VERTEX_WEIGHT')
                ap = knb.row()
                ap.prop(props, "kandura_antipen_offset",
                        text="Body Clearance", slider=True)
                rm = knb.row(); rm.scale_y = 1.2
                _oa = rm.operator("smartrig.kandura_remove",
                                  text="Remove ALL Kandura Bones",
                                  icon='TRASH')
                _oa.part = 'ALL'

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
        box.label(text="Align selected (markers or bones):", icon='SNAP_MIDPOINT')
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
        # ---- Lock Mesh Selection: ALWAYS available so you can freely edit the
        # markers without grabbing the character by mistake, before OR after
        # generating (Saeed's request). ----
        lb = layout.box()
        lb.alert = props.lock_mesh
        lr = lb.row(); lr.scale_y = 1.3
        lr.prop(props, "lock_mesh", toggle=True,
                text=("MESH LOCKED — markers only" if props.lock_mesh else "Lock Mesh Selection"),
                icon=('LOCKED' if props.lock_mesh else 'UNLOCKED'))
        if props.lock_mesh:
            lb.label(text="Character not selectable — unlock to select it.", icon='INFO')
        has_ref = bpy.data.objects.get(utils.REF_NAME) is not None
        tb = layout.box()
        _th = tb.row(align=True)
        _th.prop(props, "show_tools", text="Marker Tools", emboss=False,
                 icon=('TRIA_DOWN' if props.show_tools else 'TRIA_RIGHT'))
        _th.label(text="", icon='EMPTY_DATA')
        if not props.show_tools:
            return
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
        _mo = bpy.data.objects.get('SR_Metarig')
        if _mo is not None:
            _mh = False
            try:
                _mh = _mo.hide_get()
            except Exception:
                _mh = _mo.hide_viewport
            vis.operator("smartrig.toggle_metarig",
                         text=("Show Metarig" if _mh else "Hide Metarig"),
                         icon=('HIDE_ON' if _mh else 'HIDE_OFF'))
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
        eng.prop(props, "skin_smart_bones", icon='ARMATURE_DATA')
        eng.prop(props, "skin_split_parts")
        eng.prop(props, "skin_optimize_highres")
        _hr = eng.row()
        _hr.enabled = props.skin_optimize_highres
        _hr.prop(props, "skin_polycount_threshold")
        eng.prop(props, "skin_refine_head")
        eng.prop(props, "skin_smooth_twist")
        eng.prop(props, "skin_improve_hips")
        eng.prop(props, "skin_improve_heels")
        has_skirt_b = any(b.name.startswith("DEF-skirt.") for b in rig.pose.bones)
        if has_skirt_b:
            eng.prop(props, "skin_smart_skirt")

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

        fb = layout.box()
        fr = fb.row(align=True)
        fr.prop(props, "show_skin_facial", text="",
                icon=('TRIA_DOWN' if props.show_skin_facial else 'TRIA_RIGHT'),
                emboss=False)
        fr.label(text="Facial Features")
        fr.prop(props, "skin_facial", text="")
        if props.show_skin_facial:
            fb.operator("smartrig.skin_facial_detect",
                        text="Auto-Detect (by name)", icon='VIEWZOOM')
            fb.prop(props, "skin_eye_l")
            fb.prop(props, "skin_eye_r")
            fb.prop(props, "skin_teeth_up")
            fb.prop(props, "skin_teeth_low")
            fb.prop(props, "skin_tongue")
            fb.label(text="Eyes/teeth -> head-area bones, rigid.", icon='INFO')

        # ---- Weight Editing (per deform bone) - manual, professional ----
        gb = layout.box()
        gr = gb.row(align=True)
        gr.prop(props, "show_skin_fine", text="",
                icon=('TRIA_DOWN' if props.show_skin_fine else 'TRIA_RIGHT'),
                emboss=False)
        gr.label(text="Weight Editing (per bone)", icon='WPAINT_HLT')
        if props.show_skin_fine:
            from .metarig import _generated_rig
            _mesh = props.target_mesh
            _rig = _generated_rig()
            _inwp = False
            try:
                _inwp = (context.object is _mesh and _mesh is not None
                         and _mesh.mode == 'WEIGHT_PAINT')
            except Exception:
                _inwp = False
            gb.label(text="Fingers & body are skinned by AI. Touch up any bone by hand here.",
                     icon='INFO')
            wpr = gb.row(align=True); wpr.scale_y = 1.3
            wpr.operator("smartrig.weight_paint",
                         text=("Leave Weight Paint" if _inwp else "Enter Weight Paint"),
                         icon=('LOOP_BACK' if _inwp else 'WPAINT_HLT'),
                         depress=_inwp)
            from . import skirt as _skv
            _bshown = False
            try:
                _bshown = _skv._deform_bones_shown(_rig) if _rig is not None else False
            except Exception:
                _bshown = False
            vrow = gb.row(align=True)
            vrow.operator("smartrig.toggle_deform_bones",
                          text=("Hide Deform Bones" if _bshown else "Show Deform Bones"),
                          icon=('HIDE_OFF' if _bshown else 'HIDE_ON'),
                          depress=_bshown)
            _ined = (_mesh is not None and _mesh.mode == 'EDIT')
            if _mesh is not None and (_inwp or _ined):
                es = gb.row(align=True); es.scale_y = 1.2
                es.operator("smartrig.edit_select",
                            text=("Back to Weight Paint" if _ined
                                  else "Edit-Select Part of Body"),
                            icon=('BRUSH_DATA' if _ined else 'EDITMODE_HLT'),
                            depress=_ined)
                mk = gb.row(align=True)
                mk.label(text="Mask:")
                mk.prop(_mesh.data, "use_paint_mask_vertex", text="Verts",
                        toggle=True, icon='VERTEXSEL')
                mk.prop(_mesh.data, "use_paint_mask", text="Faces",
                        toggle=True, icon='FACESEL')
                if _ined or (_mesh.data.use_paint_mask
                             or _mesh.data.use_paint_mask_vertex):
                    iv = gb.row(align=True)
                    iv.operator("smartrig.invert_selection",
                                text="Invert Selection", icon='SELECT_SUBTRACT')
                if _inwp and (_mesh.data.use_paint_mask
                              or _mesh.data.use_paint_mask_vertex):
                    gb.label(text="Mask on: switch bones from the list below "
                                  "(viewport click = select, not bone).",
                             icon='INFO')
            if _mesh is None:
                gb.label(text="Pick the character mesh first.", icon='ERROR')
            elif _rig is None:
                gb.label(text="Generate the rig to list its deform bones.", icon='ERROR')
            else:
                _sel = ""
                try:
                    _sel = _rig.data.bones[props.weight_bone_index].name
                except Exception:
                    _sel = ""
                from . import skirt as _skf
                vm = gb.row(align=True)
                vm.prop(props, "weight_use_folders", text="Folders",
                        icon='OUTLINER', toggle=True)
                vm.prop(props, "weight_show_all_bones",
                        text=("All" if props.weight_show_all_bones else "Deform"),
                        icon='BONE_DATA', toggle=True)
                _nb = sum(1 for _b in _rig.data.bones
                          if props.weight_show_all_bones or _b.use_deform)
                vm.label(text="%d" % _nb)
                if props.weight_use_folders:
                    fbar = gb.row(align=True)
                    fbar.operator("smartrig.wf_autobuild", text="Auto-Build",
                                  icon='FILE_REFRESH')
                    fbar.operator("smartrig.wf_new", text="New Folder",
                                  icon='NEWFOLDER')
                    if len(props.weight_folders):
                        fbar.operator("smartrig.wf_clear", text="", icon='TRASH')
                    if not len(props.weight_folders):
                        gb.label(text="No folders - press Auto-Build or New Folder.",
                                 icon='INFO')
                    _assigned = set()
                    for _fl in props.weight_folders:
                        _assigned.update(m for m in _fl.members.split(",") if m)
                    if len(props.weight_folders):
                        _ai = props.weight_folders_index
                        _an = (props.weight_folders[_ai].name
                               if 0 <= _ai < len(props.weight_folders) else "-")
                        _rb = gb.row(align=True)
                        _rb.label(text="Reorder: %s" % _an, icon='RADIOBUT_ON')
                        _rb.operator("smartrig.wf_move_up", text="", icon='TRIA_UP')
                        _rb.operator("smartrig.wf_move_down", text="", icon='TRIA_DOWN')
                        _rb.operator("smartrig.wf_outdent", text="", icon='BACK')
                        _rb.operator("smartrig.wf_indent", text="", icon='FORWARD')
                    for _fi, _fl in enumerate(props.weight_folders):
                        if _fl.parent == "":
                            _wf_draw_folder(gb, props, _rig, _mesh, _sel, _skf,
                                            _fi, _fl)
                    _ung = [b.name for b in _rig.data.bones
                            if (props.weight_show_all_bones or b.use_deform)
                            and b.name not in _assigned]
                    if _ung:
                        _ub = gb.box()
                        _ub.label(text="Ungrouped (%d) - pick one, then + on a folder"
                                       % len(_ung), icon='DOT')
                        for _bn in _ung[:60]:
                            _op = _ub.operator("smartrig.wf_pick",
                                               text=_skf._wt_pretty(_bn),
                                               icon='BONE_DATA', depress=(_bn == _sel))
                            _op.bone = _bn
                else:
                    gb.template_list("SMARTRIG_UL_deform_bones", "",
                                     _rig.data, "bones",
                                     props, "weight_bone_index", rows=9)
                prow = gb.row(align=True); prow.scale_y = 1.3
                op = prow.operator("smartrig.weight_paint",
                                   text=(("Paint: " + _sel) if _sel
                                         else "Paint Selected Bone"),
                                   icon='BRUSH_DATA')
                op.group = _sel
                lrow = gb.row(align=True)
                _lk = lrow.operator("smartrig.lock_bones", text="Lock All",
                                    icon='LOCKED'); _lk.lock = True
                _ul = lrow.operator("smartrig.lock_bones", text="Unlock All",
                                    icon='UNLOCKED'); _ul.lock = False
                if (_rig.data.bones.get("DEF-head") is None
                        and _rig.data.bones.get("head") is not None):
                    fxr = gb.row(align=True)
                    fxr.operator("smartrig.fix_head_neck_names",
                                 text="Fix Head / Neck Names", icon='SORTALPHA')
                if _sel and _mesh.vertex_groups.get(_sel) is None:
                    gb.label(text="No weights on this bone yet - paint to add.",
                             icon='INFO')
                tb = gb.box()
                tb.label(text="Weight Tools (active group):", icon='MODIFIER')
                anr = tb.row(align=True)
                anr.prop(context.scene.tool_settings, "use_auto_normalize",
                         text="Auto Normalize (keep sum = 1 while painting)",
                         toggle=True, icon='MOD_VERTEX_WEIGHT')
                trow = tb.row(align=True)
                trow.operator("smartrig.weight_smooth", text="Smooth",
                              icon='MOD_SMOOTH')
                trow.operator("smartrig.weight_normalize", text="Normalize",
                              icon='MOD_VERTEX_WEIGHT')
                trow.operator("smartrig.weight_clean", text="Clean",
                              icon='BRUSH_DATA')
                mrow = tb.row(align=True)
                mrow.operator("smartrig.weight_mirror", text="Mirror L / R",
                              icon='MOD_MIRROR')
                mall = mrow.operator("smartrig.weight_mirror",
                                     text="Symmetrise All", icon='ARROW_LEFTRIGHT')
                mall.all_groups = True

        bnd = layout.box()
        bnd.label(text="Bind", icon='POSE_HLT')
        # SMART scene doctor: warn about old-rig clutter BEFORE the user binds
        try:
            from . import skirt as _skh
            _h = _skh.scene_health_scan(context)
            _nh = _skh.scene_health_total(_h)
        except Exception:
            _nh = 0
        if _nh:
            hb = bnd.box()
            hb.alert = True
            hb.label(text="%d old-rig leftover(s) clutter the scene" % _nh,
                     icon='GHOST_ENABLED')
            hb.label(text="They look like a broken rig. Clean up:", icon='INFO')
            rr = hb.row(align=True)
            rr.operator("smartrig.scene_fix", text="Hide Them",
                        icon='HIDE_ON').mode = 'HIDE'
            rr.operator("smartrig.scene_fix", text="Delete Them",
                        icon='TRASH').mode = 'DELETE'
        bnd.prop(props, "skin_selected_bones_only")
        if props.skin_selected_bones_only:
            from . import skirt as _skm
            if bpy.app.version >= (5, 0, 0):
                _nsel = sum(1 for _pb in rig.pose.bones
                            if _pb.select and _pb.bone.use_deform)
            else:
                _nsel = sum(1 for _pb in rig.pose.bones
                            if _pb.bone.select and _pb.bone.use_deform)
            pk = bnd.box()
            hdr = pk.row(align=True)
            hdr.prop(props, "show_skin_pick", text="",
                     icon=('TRIA_DOWN' if props.show_skin_pick
                           else 'TRIA_RIGHT'), emboss=False)
            hdr.label(text="Pick Bones  -  %d selected" % _nsel,
                      icon='RESTRICT_SELECT_OFF')
            if props.show_skin_pick:
                r = pk.row(align=True)
                r.operator("smartrig.selbones_pick", text="All").part = 'ALL'
                r.operator("smartrig.selbones_pick", text="None").part = 'NONE'
                r = pk.row(align=True)
                r.operator("smartrig.selbones_pick", text="Spine").part = 'SPINE'
                r.operator("smartrig.selbones_pick", text="Arms").part = 'ARMS'
                r.operator("smartrig.selbones_pick", text="Legs").part = 'LEGS'
                r = pk.row(align=True)
                r.operator("smartrig.selbones_pick", text="Feet").part = 'FEET'
                r.operator("smartrig.selbones_pick", text="Fingers").part = 'FINGERS'
                _hasb = lambda pfx: any(b.name.startswith(pfx)
                                        for b in rig.data.bones)
                r2 = pk.row(align=True)
                _any2 = False
                for _pfx, _lbl, _prt in (("DEF-skirt.", "Skirt", 'SKIRT'),
                                         ("DEF-kan_sleeve.", "Sleeves", 'SLEEVES'),
                                         ("DEF-kan_collar.", "Collar", 'COLLAR'),
                                         ("DEF-kan_cuff.", "Cuffs", 'CUFFS')):
                    if _hasb(_pfx):
                        r2.operator("smartrig.selbones_pick", text=_lbl).part = _prt
                        _any2 = True
                if not _any2:
                    r2.label(text="")
                # SMART SCAN: any OTHER deform bones (Rigify samples - tail,
                # wings, face...) get their own auto-detected buttons.
                try:
                    _tops, _rest = _skm.pick_extra_split(rig)
                except Exception:
                    _tops, _rest = [], []
                for _i in range(0, len(_tops), 3):
                    _rr = pk.row(align=True)
                    for _root, _names in _tops[_i:_i + 3]:
                        _o = _rr.operator(
                            "smartrig.selbones_pick",
                            text="%s (%d)" % (_root.title(), len(_names)))
                        _o.part = 'DYN:' + _root
                if _rest:
                    pk.operator("smartrig.selbones_pick",
                                text="Other (%d)" % len(_rest)).part = 'OTHER'
                pk.label(text="Twist bones follow their limb automatically.",
                         icon='INFO')
        bnd.prop(props, "skin_selected_verts_only")
        bnd.prop(props, "skin_apply_shapekeys")
        bnd.prop(props, "skin_preserve_volume")
        bnd.prop(props, "skin_scale_fix")
        row = bnd.row(align=True); row.scale_y = 1.6
        sub = row.row(align=True); sub.enabled = ready
        sub.operator("smartrig.bind", text="Bind", icon='MOD_VERTEX_WEIGHT')
        row.operator("smartrig.unbind", text="Unbind", icon='X')
        # finger scale-curl repair (only shown when the drivers are missing)
        try:
            from . import skirt as _skf
            if _skf.finger_curl_missing(rig):
                fx = bnd.box(); fx.alert = True
                fx.label(text="Fingers won't curl when scaled", icon='ERROR')
                fx.operator("smartrig.fix_finger_curl",
                            text="Fix Finger Curl", icon='HAND')
        except Exception:
            pass
        if has_skirt and props.skin_split_parts:
            bnd.label(text="Body ignores skirt bones; skirt follows its own.", icon='INFO')

        # Skirt leg collision lives in the RIG tab (rigging) and the N-panel Item
        # tab (animation) — NOT here. Skinning = binding only.

    # ----------------------------------------------------------------- ANIMATE
    def _draw_animate(self, layout, context):
        """ANIMATE phase: everything the animator needs AFTER the rig is built.
        Live cloth/secondary-motion controls (with bake), plus the planned
        animation systems shown honestly as 'planned'."""
        from . import metarig as _mr
        from . import skirt as _sk
        props = context.scene.smartrig
        rig = _mr._generated_rig()
        krig = _sk.kilt_rig(context)
        if rig is None and krig is None:
            box = layout.box()
            box.label(text="No rig in this scene yet.", icon='INFO')
            box.label(text="Build one in the Rig tab first.")
            return
        if rig is not None:
            hb = layout.box()
            hr = hb.row(); hr.alignment = 'CENTER'
            hr.label(text="Rig: %s" % rig.name, icon='CHECKMARK')
        cb = layout.box()
        cb.label(text="Cloth & Secondary Motion", icon='PHYSICS')
        _any_live = False
        if krig is not None:
            if krig.get("sk_jiggle"):
                _any_live = True
                jr = cb.row(align=True)
                jr.label(text="Skirt Jiggle", icon='MOD_CLOTH')
                _sbaked = (bool(krig.get("sk_jiggle_baked"))
                           or _sk.skirt_jiggle_has_keys(krig))
                jr.operator("smartrig.bake_jiggle",
                            text=("Baked" if _sbaked else "Bake"),
                            icon=('CHECKMARK' if _sbaked else 'ACTION'),
                            depress=_sbaked).remove = False
            if krig.get("sk_follow") or krig.get("sk_antipen"):
                _any_live = True
                cb.label(text="Follow Body / Anti-Penetration active",
                         icon='MOD_MESHDEFORM')
        if _any_live:
            cb.label(text="Live sliders: N-panel > Item.", icon='INFO')
        else:
            cb.label(text="Add Jiggle / Collision in the Rig tab.", icon='INFO')
        layout.separator(factor=0.5)
        layout.label(text="Animation Systems", icon='PLAY')
        for name, ic in (("Locomotion (drive bone)", 'ORIENTATION_GIMBAL'),
                         ("Action Packs (walk / run / fight)", 'ACTION'),
                         ("Animation Layers", 'NLA'),
                         ("Lipsync (audio to mouth)", 'SPEAKER'),
                         ("Pose Library (expressions)", 'ARMATURE_DATA'),
                         ("Ground Adaptation", 'SNAP_FACE_CENTER')):
            pb = layout.box()
            rr = pb.row()
            rr.label(text=name, icon=ic)
            rr.label(text="planned", icon='TIME')

    # ---------------------------------------------------------------- LET'S FIT
    def _draw_fit(self, layout, context):
        """The Fit phase, organized in USE ORDER (Rig the character first!):
        1) pick garment+body  2) Fit to Character (one click)  3) tune live
        4) hand control (garment rig / mannequin)  5) Drape  6) Apply/Remove."""
        props = context.scene.smartrig
        from .garment import (K_BASE, K_INFO, MOD_WRAP)

        # -- 1. WHAT --------------------------------------------------------
        box = layout.box()
        box.label(text="Fit Clothing to Character", icon='MOD_CLOTH')
        col = box.column(align=True)
        col.prop(props, "garment_object", text="Garment")
        col.prop(props, "fit_body_object", text="Body")
        g_ob = props.garment_object
        if g_ob is None:
            # ONE CLICK, ZERO SETUP (v1.35.0): empty pickers = auto-detect.
            # Select the garment in the viewport (or select nothing) and hit
            # the big button - garment + body are found automatically.
            box.label(text="Empty = auto-detect (or select the garment)",
                      icon='INFO')
            big = box.row()
            big.scale_y = 1.7
            big.operator("smartrig.mannequin_match",
                         text="Fit to Character", icon='ARMATURE_DATA')
            return
        # the recommended flow reminder
        has_rig = (bpy.data.objects.get("SR_Metarig") is not None)
        if not has_rig:
            box.label(text="Tip: Rig the character first (Rig tab) "
                           "for exact joints", icon='INFO')

        # -- 1.5 FIT WIZARD (step by step, like the rig marker wizard) -------
        step = props.fitwiz_step
        wiz = box.box()
        hr = wiz.row()
        hr.label(text="Fit Wizard (step by step)", icon='PRESET')
        if step == 0:
            hr.operator("smartrig.fitwiz_start", text="Start", icon='PLAY')
        else:
            hr.operator("smartrig.fitwiz_cancel", text="", icon='X')
            nav = wiz.row(align=True)
            b = nav.operator("smartrig.fitwiz_goto", text="",
                             icon='TRIA_LEFT')
            b.target = max(step - 1, 1)
            f = nav.operator("smartrig.fitwiz_goto", text="",
                             icon='TRIA_RIGHT')
            f.target = min(step + 1, 5)
            e = nav.operator("smartrig.fitwiz_goto", text="",
                             icon='FF')
            e.target = 5
            if step == 1:
                wiz.label(text="1/5  Place the garment over the character",
                          icon='ORIENTATION_GLOBAL')
                r = wiz.row(align=True)
                r.operator("smartrig.fitwiz_view", text="Front").axis = 'FRONT'
                r.operator("smartrig.fitwiz_view", text="Side").axis = 'LEFT'
                r.operator("smartrig.lets_fit", text="Auto Place")
                wiz.prop(props, "fitwiz_mirror", icon='MOD_MIRROR')
                wiz.prop(props, "fitwiz_xray", slider=True)
                wiz.prop(props, "fitwiz_ref_alpha", slider=True)
                wiz.label(text="Move / Rotate / Scale freely (G R S)")
                wiz.operator("smartrig.fitwiz_markers",
                             text="Next: Markers", icon='FORWARD')
            elif step == 2:
                wiz.label(text="2/5  JOINTS front: drag wrong markers",
                          icon='EMPTY_AXIS')
                wiz.prop(props, "fitwiz_mirror", icon='MOD_MIRROR')
                wiz.prop(props, "fitwiz_xray", slider=True)
                wiz.prop(props, "fitwiz_ref_alpha", slider=True)
                r = wiz.row(align=True)
                r.operator("smartrig.fitwiz_markers", text="Rebuild")
                r.operator("smartrig.fitwiz_side",
                           text="Next: Side", icon='FORWARD')
            elif step == 3:
                wiz.label(text="3/5  JOINTS side: forward / back",
                          icon='ORIENTATION_VIEW')
                wiz.prop(props, "fitwiz_mirror", icon='MOD_MIRROR')
                wiz.prop(props, "fitwiz_xray", slider=True)
                wiz.prop(props, "fitwiz_ref_alpha", slider=True)
                r = wiz.row(align=True)
                r.operator("smartrig.fitwiz_view", text="Front").axis = 'FRONT'
                r.operator("smartrig.fitwiz_size",
                           text="Next: Size", icon='FORWARD')
            elif step == 4:
                wiz.label(text="4/5  SIZE: chest & waist girth",
                          icon='FIXED_SIZE')
                wiz.label(text="Width from Front - depth from Side")
                wiz.prop(props, "fitwiz_mirror", icon='MOD_MIRROR')
                wiz.prop(props, "fitwiz_xray", slider=True)
                r = wiz.row(align=True)
                r.operator("smartrig.fitwiz_view", text="Front").axis = 'FRONT'
                r.operator("smartrig.fitwiz_view", text="Side").axis = 'LEFT'
                wiz.operator("smartrig.fitwiz_extras",
                             text="Next: Parts", icon='FORWARD')
            elif step == 5:
                wiz.label(text="5/5  Parts (pre-filled - correct if needed)",
                          icon='GROUP_VERTEX')
                wiz.label(text="See what was detected (highlights it):")
                sr = wiz.row(align=True)
                sr.operator("smartrig.fitwiz_show",
                            text="Sleeve", icon='HIDE_OFF').part = 'SLEEVE'
                sr.operator("smartrig.fitwiz_show",
                            text="Collar", icon='HIDE_OFF').part = 'COLLAR'
                sr2 = wiz.row(align=True)
                sr2.operator("smartrig.fitwiz_show",
                             text="Lower", icon='HIDE_OFF').part = 'LOWER'
                sr2.operator("smartrig.fitwiz_show",
                             text="Rigid", icon='HIDE_OFF').part = 'RIGID'
                hb = wiz.box()
                hb.label(text="Sleeve = arm tube, shoulder to cuff")
                hb.label(text="Collar = band around the neck opening")
                hb.label(text="Lower = skirt / hem below the waist")
                hb.label(text="Rigid = belt, buttons, pockets, pins")
                wiz.label(text="Fix: select verts, then register as:")
                pr = wiz.row(align=True)
                pr.operator("smartrig.fitwiz_register",
                            text="Sleeve").part = 'SLEEVE'
                pr.operator("smartrig.fitwiz_register",
                            text="Collar").part = 'COLLAR'
                pr2 = wiz.row(align=True)
                pr2.operator("smartrig.fitwiz_register",
                             text="Lower").part = 'LOWER'
                pr2.operator("smartrig.fitwiz_register",
                             text="Rigid").part = 'RIGID'
                gr = wiz.row(); gr.scale_y = 1.6
                gr.operator("smartrig.fitwiz_go",
                            text="FIT!", icon='ARMATURE_DATA')

        # -- 2. ONE CLICK ---------------------------------------------------
        fitted = g_ob.get(K_BASE) is not None \
            or g_ob.modifiers.get(MOD_WRAP) is not None
        big = box.row(); big.scale_y = 1.7
        big.enabled = step == 0
        big.operator("smartrig.mannequin_match",
                     text="Fit to Character", icon='ARMATURE_DATA')
        srow = box.row(align=True)
        srow.operator("smartrig.lets_fit",
                      text="Refit (place only)" if fitted else "Place Garment",
                      icon='PLAY')
        srow.prop(props, "garment_preserve", text="", icon='MOD_MESHDEFORM')
        info = g_ob.get(K_INFO)
        if info:
            box.label(text=str(info), icon='CHECKMARK')
        if not fitted:
            return

        # -- 3. TUNE (live) --------------------------------------------------
        # Tune drives the CONFORM engine (Place Garment). After a Mannequin
        # Match it would overwrite the match with a tent - so it locks.
        matched = bpy.data.objects.get("SRF_GarmentRig") is not None
        tune = box.box()
        tune.label(text="Tune (live)", icon='TOOL_SETTINGS')
        if matched:
            tune.label(text="Locked: garment is Matched (use Hand Control)",
                       icon='LOCKED')
        col = tune.column(align=True)
        col.enabled = not matched
        col.prop(props, "garment_ease")
        col.prop(props, "garment_smooth")
        col.prop(props, "garment_scale")
        col.prop(props, "garment_height")

        # -- 4. HAND CONTROL -------------------------------------------------
        hc = box.box()
        hc.label(text="Hand Control", icon='ARMATURE_DATA')
        if bpy.data.objects.get("SRF_GarmentRig") is not None:
            hc.label(text="Pose Mode on SRF_GarmentRig: grab any bone",
                     icon='CHECKMARK')
        mrow = hc.row(align=True)
        mrow.operator("smartrig.garment_mannequin",
                      text="Mannequin", icon='OUTLINER_OB_ARMATURE')
        if bpy.data.objects.get("SRF_Mannequin") is not None:
            col = hc.column(align=True)
            col.prop(props, "mann_arm_open")
            col.prop(props, "mann_elbow_bend")
            col.prop(props, "mann_neck_len")
            col.prop(props, "mann_torso_vol")
            col.prop(props, "mann_arm_vol")

        # -- 5. FINISH -------------------------------------------------------
        fin = box.box()
        fin.label(text="Finish", icon='CHECKMARK')
        dr = fin.row(); dr.scale_y = 1.3
        dr.operator("smartrig.fit_drape", text="Drape (Cloth)", icon='MOD_CLOTH')
        row = fin.row(align=True)
        row.operator("smartrig.fit_apply", text="Apply Fit", icon='CHECKMARK')
        row.operator("smartrig.fit_remove", text="Remove", icon='X')

    # ------------------------------------------------------------------ PARTS
    def _draw_parts(self, layout, context):
        """Standalone 'Parts & Accessories' mode: rig a skirt / cloth / appendage
        with NO body markers. Only Cloth (skirt) is functional today; the rest are
        shown as planned so the layout is complete and honest."""
        props = context.scene.smartrig
        layout.label(text="Standalone parts - no body markers needed.", icon='INFO')

        # ---------- CLOTH ----------
        cloth = layout.box()
        cloth.label(text="Cloth", icon='MOD_CLOTH')
        sb = cloth.box()
        sh = sb.row(align=True)
        sh.prop(props, "show_skirt", text="Skirt",
                emboss=False, icon=('TRIA_DOWN' if props.show_skirt else 'TRIA_RIGHT'))
        _iv = icons.get('skirt')
        if _iv:
            sh.label(text="", icon_value=_iv)
        if props.show_skirt:
            self._draw_skirt_settings(sb, context, props, standalone=True)
            # Remove just the skirt bones + extras (keeps the rig)
            rmr = sb.row()
            rmr.operator("smartrig.remove_skirt", text="Remove Skirt", icon='TRASH')

        # ---------- Generate / Back to Metarig (same as Character mode) ----------
        _mo = bpy.data.objects.get("SR_Metarig")
        if _mo is not None:
            gb = layout.box()
            _rig = None
            try:
                _rig = _mo.data.rigify_target_rig
            except Exception:
                _rig = None
            if _rig is not None:
                gb.label(text="Rig generated: %s" % _rig.name, icon='CHECKMARK')
                br = gb.row(); br.scale_y = 1.4
                br.operator("smartrig.back_to_metarig",
                            text="Back to Metarig (edit more)", icon='LOOP_BACK')
            gr = gb.row(); gr.scale_y = 1.6
            gr.operator("smartrig.generate",
                        text=("Re-generate Rig" if _rig is not None else "Generate Rig"),
                        icon='POSE_HLT')
            # delete EVERYTHING (metarig + generated rig) and start fresh
            dr = gb.row()
            dr.operator("smartrig.reset", text="Delete Rig / Start Over", icon='X')

        # ---------- planned categories (structure ready, generators to come) ----------
        for name, ic in (("Appendages  (tail / ears / wings)", 'BONE_DATA'),
                         ("Props  (rigid objects)", 'MESH_CUBE'),
                         ("Face-only", 'USER')):
            pb = layout.box()
            rr = pb.row()
            rr.label(text=name, icon=ic)
            rr.label(text="planned", icon='TIME')

    # ------------------------------------------------- shared skirt settings
    def _draw_skirt_settings(self, box, context, props, standalone=False):
        """Skirt SETTINGS shared by the Character panel and the standalone Parts
        panel: source (Separate/Merged/Manual), mesh, columns, rows, length, collide,
        and the detected-type label. `standalone` swaps the build button."""
        skirt_iv = icons.get('skirt')
        src = box.row(align=True)
        src.prop(props, "skirt_source", expand=True)
        if props.skirt_source == 'SEPARATE':
            box.prop(props, "skirt_object", text="Skirt")
            box.label(text="Bones span the mesh top -> hem.", icon='INFO')
            _mo = bpy.data.objects.get("SR_Metarig")
            if _mo is not None and "sr_skirt_kind" in _mo:
                _lbl = {"TUBE": "clean tube", "OPEN": "open-front",
                        "LAYERED": "layered", "CLOSED": "closed tube (slit)",
                        "MERGED": "merged-in-body",
                        "MESSY": "irregular"}.get(_mo["sr_skirt_kind"], str(_mo["sr_skirt_kind"]))
                _mth = _mo.get("sr_skirt_method", "?")
                _ic = 'CHECKMARK' if _mth == "edge-flow" else 'INFO'
                box.label(text="Detected: %s  ->  %s" % (_lbl, _mth), icon=_ic)
            sbx = box.box()
            shr = sbx.row()
            shr.label(text="", icon='COLOR_GREEN')
            shr.prop(props, "skirt_sep_help",
                     text="Separate skirt - what works (read me)", emboss=False,
                     icon=('TRIA_DOWN' if props.skirt_sep_help else 'TRIA_RIGHT'))
            if props.skirt_sep_help:
                sbx.label(text="Full feature set is available:", icon='CHECKMARK')
                sbx.label(text="Leg Collision + Jiggle (bone-based),")
                sbx.label(text="Follow Body (clings when sitting) AND")
                sbx.label(text="Anti-Penetration (no poking into body).")
                sbx.label(text="Recommended for the best result.")
        elif props.skirt_source == 'MERGED':
            reg = box.row(); reg.scale_y = 1.2
            reg.operator("smartrig.register_skirt",
                         text="Register Skirt Selection", icon='GROUP_VERTEX')
            has_vg = (props.target_mesh is not None
                      and props.target_mesh.vertex_groups.get("SR_Skirt") is not None)
            box.label(text=("Registered - bones span top -> hem." if has_vg
                            else "Edit Mode -> select skirt faces -> Register."),
                      icon=('CHECKMARK' if has_vg else 'INFO'))
            mb = box.box()
            mh = mb.row()
            mh.label(text="", icon='COLOR_GREEN')
            mh.prop(props, "skirt_merged_help",
                    text="Merged skirt - what works (read me)", emboss=False,
                    icon=('TRIA_DOWN' if props.skirt_merged_help else 'TRIA_RIGHT'))
            if props.skirt_merged_help:
                mb.label(text="Works: Leg Collision + Jiggle (bone-based).", icon='CHECKMARK')
                mb.label(text="Unavailable: Follow Body + Anti-Penetration", icon='CANCEL')
                mb.label(text="(they need a SEPARATE skirt mesh).")
        else:  # MANUAL
            box.label(text="Starter ring; edit bones freely after.", icon='INFO')
        cc = box.column(align=True)
        cc.prop(props, "skirt_columns")
        cc.prop(props, "skirt_rows")
        if props.skirt_source == 'MANUAL':
            cc.prop(props, "skirt_length", slider=True)
        fr = cc.row(align=True)
        fr.prop(props, "skirt_front_axis", text="Front")
        cc.prop(props, "skirt_symmetric")
        cc.prop(props, "skirt_collide")
        ar = box.row(); ar.scale_y = 1.5
        if standalone:
            if props.skirt_source == 'SEPARATE':
                if skirt_iv:
                    ar.operator("smartrig.rig_skirt_standalone",
                                text="Build Skirt Metarig", icon_value=skirt_iv)
                else:
                    ar.operator("smartrig.rig_skirt_standalone",
                                text="Build Skirt Metarig", icon='MOD_CLOTH')
                box.label(text="Builds an editable metarig. Tweak, then Generate Rig below.",
                          icon='INFO')
            else:
                box.label(text="Merged / Manual need a character body (use Character mode).",
                          icon='INFO')
        else:
            _txt = "Add Starter Ring" if props.skirt_source == 'MANUAL' else "Add Short Skirt"
            if skirt_iv:
                ar.operator("smartrig.add_skirt", text=_txt, icon_value=skirt_iv)
            else:
                ar.operator("smartrig.add_skirt", text=_txt, icon='MOD_CLOTH')
            if props.skirt_source != 'MANUAL':
                box.label(text="Columns/Rows update live.", icon='FILE_REFRESH')


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
        rig = skirt.kilt_rig(context)
        # show only for actual SKIRT features
        return rig is not None and any(rig.get(k) for k in
                                       ("sk_kilt", "sk_jiggle", "sk_follow", "sk_antipen"))

    def draw_header(self, context):
        iv = icons.get('skirt')
        if iv:
            self.layout.label(text="", icon_value=iv)

    def draw(self, context):
        from . import skirt
        layout = self.layout
        rig = skirt.kilt_rig(context)
        if rig is None:
            return
        skirt_iv = icons.get('skirt')
        mpb = rig.pose.bones.get("SKC_master")
        # ---- Leg Collision (driven by the SKC_master bone props) ----
        if mpb is not None and "collide" in mpb:
            col = layout.column(align=True)
            col.label(text="Leg Collision", icon='MOD_CLOTH')
            try:
                col.prop(mpb, '["collide"]', text="Collide", slider=True)
                col.prop(mpb, '["collide_dist"]', text="Swing", slider=True)
                col.prop(mpb, '["collide_spread"]', text="Strength", slider=True)
                if "leg_follow" in mpb:
                    col.prop(mpb, '["leg_follow"]', text="Leg Follow (sit)",
                             slider=True)
                    col.prop(mpb, '["shin_follow"]', text="Shin Follow",
                             slider=True)
                col.prop(context.scene.smartrig, "skirt_collide_falloff",
                         text="Push Out (flare)", slider=True)
            except Exception:
                col.label(text="Re-apply Leg Collision.", icon='ERROR')
        # ---- Jiggle (live spring; props on the rig object) ----
        if "jiggle" in rig:
            jc = layout.column(align=True)
            jc.label(text="Jiggle (secondary motion)", icon='PHYSICS')
            try:
                jc.prop(rig, '["jiggle"]', text="Jiggle", slider=True)
                jc.prop(rig, '["jiggle_amount"]', text="Amount", slider=True)
                jc.prop(rig, '["jiggle_stiffness"]', text="Stiffness", slider=True)
                jc.prop(rig, '["jiggle_damping"]', text="Damping", slider=True)
            except Exception:
                pass
            jc.prop(context.scene.smartrig, "skirt_jiggle_segments", text="Segments")
            fb = layout.column(align=True)
            fb.label(text="Wind & Gravity (skirt)", icon='FORCE_WIND')
            fb.prop(context.scene.smartrig, "jiggle_gravity", text="Gravity", slider=True)
            fb.prop(context.scene.smartrig, "jiggle_wind", text="Wind", slider=True)
            fb.prop(context.scene.smartrig, "jiggle_wind_speed", text="Wind Speed", slider=True)
            fb.prop(context.scene.smartrig, "jiggle_wind_billow", text="Billow", slider=True)
            wr = fb.row(align=True)
            wr.prop(context.scene.smartrig, "jiggle_wind_dir", text="Dir")
            wr.prop(context.scene.smartrig, "jiggle_wind_turb", text="Gust")
            # Blow Up = a SEPARATE control (direct pose flip), not a wind force
            lb = layout.column(align=True)
            lb.label(text="Blow Up (flip skirt up)", icon='TRIA_UP')
            lb.prop(context.scene.smartrig, "jiggle_wind_lift", text="Blow Up", slider=True)
            _sbaked = bool(rig.get("sk_jiggle_baked")) or skirt.skirt_jiggle_has_keys(rig)
            if _sbaked:
                bx = layout.box()
                bx.label(text="BAKED to keyframes - live solver OFF", icon='CHECKMARK')
            else:
                layout.label(text="Live (not baked).", icon='PLAY')
            jr = layout.row(align=True)
            jr.operator("smartrig.bake_jiggle",
                        text=("Baked ✓" if _sbaked else "Bake"),
                        icon=('CHECKMARK' if _sbaked else 'ACTION'),
                        depress=_sbaked).remove = False
            jr.operator("smartrig.bake_jiggle", text="Clear Bake", icon='TRASH').remove = True
        # ---- Follow Body (sitting blend) = Surface Deform strength ----
        fmod = skirt.follow_modifier(context)
        if fmod is None and (rig.get("sk_kilt") or rig.get("kan_floor")):
            sb = layout.column(align=True)
            sb.label(text="Sit (Follow Body)", icon='MOD_MESHDEFORM')
            sb.operator("smartrig.skirt_follow",
                        text="Enable Sit Follow (Surface Deform)",
                        icon='MOD_MESHDEFORM')
        if fmod is not None:
            fb = layout.column(align=True)
            fb.label(text="Follow Body (sit)", icon='MOD_MESHDEFORM')
            fb.prop(context.scene.smartrig, "skirt_follow_body", text="Follow Body", slider=True)
            if skirt.follow_status(context)[0] == 'subsurf_above':
                w = fb.box(); w.alert = True
                w.label(text="A modifier is above Follow!", icon='ERROR')
                w.operator("smartrig.skirt_follow", text="Re-bind (fix order)", icon='FILE_REFRESH')
        # ---- Floor (ground clamp) ----
        _kf = next((o for o in bpy.data.objects if o.type == 'MESH'
                    and o.modifiers.get("KAN_Floor")), None)
        if _kf is not None:
            fg = layout.column(align=True)
            fg.label(text="Floor (ground)", icon='AXIS_TOP')
            fg.prop(context.scene.smartrig, "kandura_floor_offset",
                    text="Floor Clearance", slider=True)
        # ---- Anti-Penetration (Shrinkwrap Outside) ----
        amod = skirt.antipen_modifier(context)
        if amod is not None:
            ab = layout.column(align=True)
            ab.label(text="Anti-Penetration", icon='MOD_SHRINKWRAP')
            ab.prop(context.scene.smartrig, "skirt_antipen_offset", text="Offset", slider=True)
        if (mpb is None or "collide" not in mpb) and "jiggle" not in rig \
                and fmod is None and amod is None:
            layout.label(text="Apply Collision / Jiggle / Follow first.", icon='INFO')
            return
        layout.separator()
        layout.label(text="Keyframe these to animate the cloth.", icon='KEYTYPE_KEYFRAME_VEC')


classes = (SMARTRIG_PT_panel, SMARTRIG_PT_skirt_item,
           SMARTRIG_PT_sleeve_item)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
