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
            # ---- Chest Jiggle (breast spring secondary motion) ----
            has_breast = any(b.name in ("breast.L", "breast.R") for b in rig.data.bones)
            if has_breast:
                cj = box.box()
                cj.label(text="Chest Jiggle", icon='PHYSICS')
                if not rig.get("sk_chest_jiggle"):
                    cj.prop(props, "chest_jiggle_segments", text="Segments")
                    cr = cj.row(); cr.scale_y = 1.3
                    cr.operator("smartrig.chest_jiggle", text="Jiggle Chest", icon='PHYSICS')
                    cj.label(text="1 = rigid bounce, 3+ = soft jelly wobble.", icon='INFO')
                else:
                    sr = cj.row(align=True)
                    sr.prop(props, "chest_jiggle_segments", text="Segments")
                    sr.operator("smartrig.chest_jiggle", text="Rebuild", icon='FILE_REFRESH')
                    cj.label(text="Live settings: N-panel > Item.", icon='INFO')
                    op = cj.operator("smartrig.chest_jiggle", text="Remove", icon='X')
                    op.remove = True

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
        # ---- Lock Mesh Selection: prominent, ABOVE Marker Tools, so the user
        # always sees why the character can't be selected when it's locked. ----
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

        bnd = layout.box()
        bnd.label(text="Bind", icon='POSE_HLT')
        row = bnd.row(align=True); row.scale_y = 1.6
        sub = row.row(align=True); sub.enabled = ready
        sub.operator("smartrig.bind", text="Bind", icon='MOD_VERTEX_WEIGHT')
        row.operator("smartrig.unbind", text="Unbind", icon='X')
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
            if krig.get("sk_chest_jiggle"):
                _any_live = True
                cr = cb.row(align=True)
                cr.label(text="Chest Jiggle", icon='PHYSICS')
                _cbaked = (bool(krig.get("sk_chest_jiggle_baked"))
                           or _sk.chest_jiggle_has_keys(krig))
                cr.operator("smartrig.chest_bake",
                            text=("Baked" if _cbaked else "Bake"),
                            icon=('CHECKMARK' if _cbaked else 'ACTION'),
                            depress=_cbaked).remove = False
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
        """Automatic garment fitting: pick a clothing mesh, press Fit Garment,
        tune live, then Apply. Works on any clothing type (skirt / shirt / pants /
        thobe...) - analysis is fully automatic."""
        props = context.scene.smartrig
        from .garment import (K_BASE, K_INFO, MOD_WRAP)
        box = layout.box()
        box.label(text="Fit Clothing to Character", icon='MOD_CLOTH')
        box.prop(props, "garment_object", text="Garment")
        box.prop(props, "fit_body_object", text="Body")
        g_ob = props.garment_object
        fitted = g_ob is not None and (g_ob.get(K_BASE) is not None
                                       or g_ob.modifiers.get(MOD_WRAP) is not None)
        box.prop(props, "garment_preserve")
        big = box.row(); big.scale_y = 1.6
        big.operator("smartrig.lets_fit",
                     text="Refit" if fitted else "Fit Garment", icon='PLAY')
        if g_ob is None:
            box.label(text="Pick the clothing mesh first", icon='INFO')
            return
        if fitted:
            info = g_ob.get(K_INFO)
            if info:
                box.label(text="Auto-fit: %s" % info, icon='CHECKMARK')
            tune = box.box()
            tune.label(text="Tune (live)", icon='TOOL_SETTINGS')
            col = tune.column(align=True)
            col.prop(props, "garment_ease")
            col.prop(props, "garment_smooth")
            col.prop(props, "garment_scale")
            col.prop(props, "garment_height")
            row = box.row(align=True)
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
        # show only for actual SKIRT features (chest jiggle has its own panel)
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
        if fmod is not None:
            fb = layout.column(align=True)
            fb.label(text="Follow Body (sit)", icon='MOD_MESHDEFORM')
            fb.prop(context.scene.smartrig, "skirt_follow_body", text="Follow Body", slider=True)
            if skirt.follow_status(context)[0] == 'subsurf_above':
                w = fb.box(); w.alert = True
                w.label(text="A modifier is above Follow!", icon='ERROR')
                w.operator("smartrig.skirt_follow", text="Re-bind (fix order)", icon='FILE_REFRESH')
        # ---- Anti-Penetration (Shrinkwrap Outside) ----
        amod = skirt.antipen_modifier(context)
        if amod is not None:
            ab = layout.column(align=True)
            ab.label(text="Anti-Penetration", icon='MOD_SHRINKWRAP')
            ab.prop(context.scene.smartrig, "skirt_antipen_offset", text="Offset", slider=True)
        if (mpb is None or "collide" not in mpb) and "jiggle" not in rig \
                and "chest_jiggle" not in rig and fmod is None and amod is None:
            layout.label(text="Apply Collision / Jiggle / Follow first.", icon='INFO')
            return
        layout.separator()
        layout.label(text="Keyframe these to animate the cloth.", icon='KEYTYPE_KEYFRAME_VEC')


class SMARTRIG_PT_chest_item(bpy.types.Panel):
    """Chest-jiggle live settings, in its OWN N-panel Item section (separate
    from the skirt) so the animator can tweak + keyframe the bounce."""
    bl_label = "Chest Jiggle"
    bl_idname = "SMARTRIG_PT_chest_item"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Item"

    @classmethod
    def poll(cls, context):
        from . import skirt
        rig = skirt.kilt_rig(context)
        return rig is not None and bool(rig.get("sk_chest_jiggle"))

    def draw_header(self, context):
        self.layout.label(text="", icon='PHYSICS')

    def draw(self, context):
        from . import skirt
        layout = self.layout
        rig = skirt.kilt_rig(context)
        if rig is None or "chest_jiggle" not in rig:
            return
        c = layout.column(align=True)
        c.prop(rig, '["chest_jiggle"]', text="Enable", slider=True)
        c.prop(rig, '["chest_jiggle_amount"]', text="Strength", slider=True)
        c.prop(rig, '["chest_jiggle_stiffness"]', text="Stiffness", slider=True)
        c.prop(rig, '["chest_jiggle_damping"]', text="Damping", slider=True)
        fb = layout.column(align=True)
        fb.label(text="Wind & Gravity (chest)", icon='FORCE_WIND')
        fb.prop(context.scene.smartrig, "chest_gravity", text="Gravity", slider=True)
        fb.prop(context.scene.smartrig, "chest_wind", text="Wind", slider=True)
        fb.prop(context.scene.smartrig, "chest_wind_speed", text="Wind Speed", slider=True)
        wr = fb.row(align=True)
        wr.prop(context.scene.smartrig, "chest_wind_dir", text="Dir")
        wr.prop(context.scene.smartrig, "chest_wind_turb", text="Gust")
        baked = bool(rig.get("sk_chest_jiggle_baked")) or skirt.chest_jiggle_has_keys(rig)
        if baked:
            bx = layout.box()
            bx.label(text="BAKED to keyframes - live solver OFF", icon='CHECKMARK')
        else:
            layout.label(text="Live (not baked).", icon='PLAY')
        br = layout.row(align=True)
        br.operator("smartrig.chest_bake",
                    text=("Baked ✓" if baked else "Bake"),
                    icon=('CHECKMARK' if baked else 'ACTION'),
                    depress=baked).remove = False
        br.operator("smartrig.chest_bake", text="Clear Bake", icon='TRASH').remove = True
        layout.label(text="Play to preview, or Bake for render/export.",
                     icon='KEYTYPE_KEYFRAME_VEC')


classes = (SMARTRIG_PT_panel, SMARTRIG_PT_skirt_item, SMARTRIG_PT_chest_item)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
