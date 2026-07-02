"""FIT WIZARD - step-by-step garment fitting (Saeed's design, v1.29.0).

Same philosophy that made the rig marker wizard reliable: AUTOMATION MAKES
THE FIRST GUESS, THE USER CORRECTS IT, the engine receives EXACT inputs.

  Step 1  PLACE   - position the garment over the ACTUAL rigged character
                    (the character is the anatomy reference: its joints are
                    known exactly); front/side view buttons + auto-place.
  Step 2  MARKERS - joint markers appear PRE-FILLED from the garment
                    analysis; the user drags only the wrong ones
                    (e.g. the wrist marker onto the true cuff).
  Step 3  EXTRAS  - register rigid extras (belt, pockets, buttons, flowers,
                    ornaments) into the SRF_Rigid vertex group so they move
                    as solid pieces; small loose parts are auto-rigid.
  Step 4  FIT     - one click: markers override the analysis and the match
                    engine (warp + design preservation) does everything.
"""
import bpy
from mathutils import Vector

MARKER_COL = "SRF_FitMarkers"
REF_COL = "SRF_FitRef"
MARKER_PREFIX = "SRFM_"
VG_RIGID = "SRF_Rigid"

# marker set shown per garment: only joints the analysis produced
_ORDERED = ("neck", "chest", "pelvis",
            "shoulder_l", "elbow_l", "wrist_l",
            "shoulder_r", "elbow_r", "wrist_r",
            "hip_l", "knee_l", "ankle_l",
            "hip_r", "knee_r", "ankle_r",
            "chest_w_l", "chest_w_r", "waist_w_l", "waist_w_r",
            "chest_d_f", "chest_d_b", "waist_d_f", "waist_d_b")

# chain lines drawn between markers (like the character wizard) + the two
# WIDTH spans (chest / waist girth control)
CHAINS = (("pelvis", "chest"), ("chest", "neck"),
          ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
          ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
          ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
          ("hip_r", "knee_r"), ("knee_r", "ankle_r"),
          ("chest_w_l", "chest_w_r"), ("waist_w_l", "waist_w_r"),
          ("chest_d_f", "chest_d_b"), ("waist_d_f", "waist_d_b"))


def _garment(context):
    return context.scene.smartrig.garment_object


# ------------------------- LIVE SYMMETRY (toggleable) -----------------------
# Mirrors around the GARMENT's own centre (not world x=0), works in BOTH
# directions: drag left OR right, the counterpart follows. fitwiz_mirror
# toggles it.
_MIRRORING = False


def _mirror_handler(scene, depsgraph):
    global _MIRRORING
    if _MIRRORING:
        return
    try:
        props = scene.smartrig
        if props.fitwiz_step not in (2, 3) or not props.fitwiz_mirror:
            return
        col = bpy.data.collections.get(MARKER_COL)
        if col is None or col.hide_viewport:
            return
        cx = float(col.get("srf_cx", 0.0))
        for u in depsgraph.updates:
            idb = getattr(u.id, "original", u.id)
            if not isinstance(idb, bpy.types.Object):
                continue
            nm = idb.name
            if not nm.startswith(MARKER_PREFIX) \
                    or not getattr(u, "is_updated_transform", False):
                continue
            key = nm[len(MARKER_PREFIX):]
            if key.endswith("_l"):
                other = MARKER_PREFIX + key[:-2] + "_r"
            elif key.endswith("_r"):
                other = MARKER_PREFIX + key[:-2] + "_l"
            else:
                continue
            src = bpy.data.objects.get(nm)
            dst = bpy.data.objects.get(other)
            if src is None or dst is None or not src.select_get():
                continue
            tgt = Vector((2.0 * cx - src.location.x,
                          src.location.y, src.location.z))
            if (tgt - dst.location).length > 1e-6:
                _MIRRORING = True
                try:
                    dst.location = tgt
                finally:
                    _MIRRORING = False
    except Exception:
        pass


def _role(key):
    """Same colour system as the character markers."""
    if key.endswith("_l"):
        return 'left'
    if key.endswith("_r"):
        return 'right'
    return 'center'


def _isolate(context, g):
    """Everything disappears - only the garment stays (Saeed's spec)."""
    hidden = list(context.scene.get("srf_wiz_hidden", []))
    for ob in context.view_layer.objects:
        # never hide the wizard's own markers / reference backdrops
        if ob.name.startswith(MARKER_PREFIX) \
                or ob.name.startswith("SRF_Ref_"):
            continue
        if ob is not g and not ob.hide_get():
            try:
                ob.hide_set(True)
                if ob.name not in hidden:
                    hidden.append(ob.name)
            except Exception:
                pass
    context.scene["srf_wiz_hidden"] = hidden


def _restore(context):
    for nm in list(context.scene.get("srf_wiz_hidden", [])):
        ob = bpy.data.objects.get(nm)
        if ob is not None:
            try:
                ob.hide_set(False)
            except Exception:
                pass
    if "srf_wiz_hidden" in context.scene:
        del context.scene["srf_wiz_hidden"]


def _marker_col(create=False):
    col = bpy.data.collections.get(MARKER_COL)
    if col is None and create:
        col = bpy.data.collections.new(MARKER_COL)
        bpy.context.scene.collection.children.link(col)
    return col


def _ref_col(create=False):
    col = bpy.data.collections.get(REF_COL)
    if col is None and create:
        col = bpy.data.collections.new(REF_COL)
        bpy.context.scene.collection.children.link(col)
    return col


def clear_markers():
    col = bpy.data.collections.get(MARKER_COL)
    if col is not None:
        for ob in list(col.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(col)


def clear_reference():
    col = bpy.data.collections.get(REF_COL)
    if col is not None:
        for ob in list(col.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(col)


def marker_joints():
    """{joint: world Vector} read from the wizard's marker empties. Empty
    dict when the wizard is not in use - the analysis stays in charge."""
    col = bpy.data.collections.get(MARKER_COL)
    if col is None or col.hide_viewport:
        return {}
    out = {}
    for ob in col.objects:
        if ob.name.startswith(MARKER_PREFIX):
            key = ob.name[len(MARKER_PREFIX):].split(".")[0]
            out[key] = ob.matrix_world.translation.copy()
    return out


def _make_reference(context, body):
    """THE REFERENCE IS NOT JUST A PICTURE (Saeed's spec): render the ACTUAL
    character front + back (ortho, transparent) and PAINT ITS MEASURED
    JOINTS on the image with the marker colours - so placing the garment
    and dragging markers is done against real, pre-measured anatomy."""
    import tempfile, os, math
    import numpy as np
    from . import mannequin as _mq
    scene = context.scene
    lo = [1e9] * 3
    hi = [-1e9] * 3
    for c in body.bound_box:
        w = body.matrix_world @ Vector(c)
        for i in range(3):
            lo[i] = min(lo[i], w[i])
            hi[i] = max(hi[i], w[i])
    center = Vector(((lo[0] + hi[0]) * .5, (lo[1] + hi[1]) * .5,
                     (lo[2] + hi[2]) * .5))
    span = max(hi[2] - lo[2], hi[0] - lo[0]) * 1.15
    jt = _mq.character_joints(body) or {}
    res = 1024
    cd = bpy.data.cameras.new("SRF_REF_CAM")
    cam = bpy.data.objects.new("SRF_REF_CAM", cd)
    scene.collection.objects.link(cam)
    cd.type = 'ORTHO'
    cd.ortho_scale = span
    r = scene.render
    prev = (r.engine, r.resolution_x, r.resolution_y,
            r.resolution_percentage, r.filepath,
            r.image_settings.file_format, r.film_transparent, scene.camera)
    hidden_r = []
    paths = {}
    try:
        for ob in scene.objects:
            if ob.type == 'MESH' and ob is not body and not ob.hide_render:
                ob.hide_render = True
                hidden_r.append(ob)
        try:
            r.engine = 'BLENDER_WORKBENCH'
        except Exception:
            pass
        r.resolution_x = r.resolution_y = res
        r.resolution_percentage = 100
        r.image_settings.file_format = 'PNG'
        r.film_transparent = True
        for tag, az in (("front", 0.0), ("side", -90.0), ("back", 180.0)):
            a = math.radians(az)
            cam.location = center + Vector((10 * math.sin(a),
                                            -10 * math.cos(a), 0.0))
            cam.rotation_euler = (cam.location - center).to_track_quat(
                'Z', 'Y').to_euler()
            scene.camera = cam
            fp = os.path.join(tempfile.gettempdir(), "srf_ref_%s.png" % tag)
            r.filepath = fp
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)
            paths[tag] = fp
    finally:
        for ob in hidden_r:
            ob.hide_render = False
        bpy.data.objects.remove(cam, do_unlink=True)
        (r.engine, r.resolution_x, r.resolution_y, r.resolution_percentage,
         r.filepath, r.image_settings.file_format, r.film_transparent,
         scene.camera) = prev
    col = _marker_col(create=True)
    for tag, fp in paths.items():
        img = bpy.data.images.load(fp, check_existing=False)
        px = np.empty(res * res * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        px = px.reshape(res, res, 4)          # row 0 = bottom
        # MEASUREMENT LINES (like the anatomy proportion sheet): a thin
        # horizontal line at every measured joint height
        zs = sorted({round(float(v.z), 3) for k, v in jt.items()
                     if isinstance(v, Vector)
                     and k in ("neck", "chest", "pelvis", "shoulder_l",
                               "elbow_l", "wrist_l", "knee_l", "ankle_l")})
        for z in zs:
            cy = int((0.5 + (z - center.z) / span) * res)
            if 1 <= cy < res - 1:
                px[cy - 1:cy + 1, :, :] = (0.35, 0.55, 0.9, 0.9)
        sgn = 1.0 if tag == "front" else -1.0
        for k, v in jt.items():
            if not isinstance(v, Vector):
                continue
            if tag == "side":
                u = 0.5 - (v.y - center.y) / span
            else:
                u = 0.5 + sgn * (v.x - center.x) / span
            w_ = 0.5 + (v.z - center.z) / span
            cx, cy = int(u * res), int(w_ * res)
            rr = 7
            y0, y1 = max(cy - rr, 0), min(cy + rr, res)
            x0, x1 = max(cx - rr, 0), min(cx + rr, res)
            if y0 >= y1 or x0 >= x1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr * rr
            cc = (0.2, 0.9, 1.0, 1.0)
            if k.endswith("_l"):
                cc = (1.0, 0.8, 0.1, 1.0)
            elif k.endswith("_r"):
                cc = (0.55, 0.45, 0.2, 1.0)
            sub = px[y0:y1, x0:x1]
            sub[m] = cc
            px[y0:y1, x0:x1] = sub
        img.pixels.foreach_set(px.ravel())
        img.pack()
        em = bpy.data.objects.new("SRF_Ref_%s" % tag, None)
        em.empty_display_type = 'IMAGE'
        em.data = img
        em.empty_display_size = span
        em.empty_image_depth = 'BACK'          # always drawn behind
        em.use_empty_image_alpha = True
        try:
            alpha = float(context.scene.smartrig.fitwiz_ref_alpha)
        except Exception:
            alpha = 0.85
        em.color = (1.0, 1.0, 1.0, alpha)
        rot_off = {"front": (0.0, Vector((0.0, 0.03, 0.0))),
                   "back": (math.pi, Vector((0.0, -0.03, 0.0))),
                   "side": (-math.pi / 2, Vector((0.03, 0.0, 0.0)))}[tag]
        em.rotation_euler = (math.pi / 2, 0.0, rot_off[0])
        em.location = center + rot_off[1]
        em.hide_select = True     # background only: G can never grab it
        _ref_col(create=True).objects.link(em)


class SMARTRIG_OT_fitwiz_start(bpy.types.Operator):
    """Start the step-by-step Fit Wizard (place the garment over the
    character, correct the markers, register extras, fit)"""
    bl_idname = "smartrig.fitwiz_start"
    bl_label = "Start Fit Wizard"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        if props.garment_object is None or props.fit_body_object is None:
            self.report({'ERROR'}, "Pick the garment and the body first.")
            return {'CANCELLED'}
        col = bpy.data.collections.get(MARKER_COL)
        if col is not None:
            col.hide_viewport = False
            for mo in col.objects:
                try:
                    mo.hide_set(False)   # un-hide markers eaten by isolate
                except Exception:
                    pass
        props.fitwiz_step = 1
        g = props.garment_object
        # SAEED'S SPEC: everything disappears - only the garment - and the
        # view snaps to FRONT (same entrance as the character marker wizard)
        _isolate(context, g)
        # the measured anatomy reference (front/side/back of the ACTUAL
        # character, joints + proportion lines painted on)
        body = props.fit_body_object
        if body is not None:
            try:
                _make_reference(context, body)
            except Exception as e:
                print("Soulify fit reference:", e)
        from . import markers as _mk
        _mk.set_front_view(context)
        for ob in context.selected_objects:
            ob.select_set(False)
        g.select_set(True)
        context.view_layer.objects.active = g
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_view(bpy.types.Operator):
    """Look at the character from the front or the side while placing"""
    bl_idname = "smartrig.fitwiz_view"
    bl_label = "Wizard View"
    bl_options = {'REGISTER'}

    axis: bpy.props.EnumProperty(items=[('FRONT', "Front", ""),
                                        ('LEFT', "Left", "")])

    def execute(self, context):
        try:
            bpy.ops.view3d.view_axis(type=self.axis)
        except Exception:
            pass
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_markers(bpy.types.Operator):
    """Build (or rebuild) the joint markers, PRE-FILLED by the automatic
    garment analysis - drag any marker that looks wrong"""
    bl_idname = "smartrig.fitwiz_markers"
    bl_label = "Show Markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import mannequin
        props = context.scene.smartrig
        g = props.garment_object
        if g is None:
            self.report({'ERROR'}, "Pick the garment first.")
            return {'CANCELLED'}
        jt = mannequin.garment_skeleton(g)
        if jt is None:
            self.report({'ERROR'}, "Could not analyze the garment.")
            return {'CANCELLED'}
        g["srf_wiz_label"] = str(jt.get("label", "garment"))
        # WIDTH markers (Saeed): chest + waist girth control, pre-filled
        # from the analyzed torso radii
        radii = jt.get("radii") or {}
        r_ch = radii.get("chest") or radii.get("torso")
        r_wa = radii.get("torso") or r_ch
        if "chest" in jt and r_ch:
            c = jt["chest"]
            jt["chest_w_l"] = c + Vector((-r_ch, 0.0, 0.0))
            jt["chest_w_r"] = c + Vector((r_ch, 0.0, 0.0))
            jt["chest_d_f"] = c + Vector((0.0, -r_ch, 0.0))
            jt["chest_d_b"] = c + Vector((0.0, r_ch, 0.0))
        if "pelvis" in jt and r_wa:
            p = jt["pelvis"]
            jt["waist_w_l"] = p + Vector((-r_wa, 0.0, 0.0))
            jt["waist_w_r"] = p + Vector((r_wa, 0.0, 0.0))
            jt["waist_d_f"] = p + Vector((0.0, -r_wa, 0.0))
            jt["waist_d_b"] = p + Vector((0.0, r_wa, 0.0))
        clear_markers()
        col = _marker_col(create=True)
        # size relative to the garment
        bb = [g.matrix_world @ Vector(c) for c in g.bound_box]
        h = max(max(p.z for p in bb) - min(p.z for p in bb), 1e-3)
        made = 0
        for key in _ORDERED:
            v = jt.get(key)
            if not isinstance(v, Vector):
                continue
            em = bpy.data.objects.new(MARKER_PREFIX + key, None)
            # SAME SYSTEM as the character markers: tiny core, the coloured
            # GPU glow (wizard overlay) is the visual; roles share colours
            role = _role(key)
            em.empty_display_type = 'PLAIN_AXES'
            em.empty_display_size = 0.012 * h
            em.location = v
            em.show_name = False
            em.show_in_front = True
            em.color = {'center': (0.2, 0.9, 1.0, 1.0),
                        'left': (1.0, 0.8, 0.1, 1.0),
                        'right': (0.55, 0.45, 0.2, 1.0)}[role]
            col.objects.link(em)
            made += 1
        # live symmetry pivot = the garment's own centre x
        bbx = [(g.matrix_world @ Vector(c)).x for c in g.bound_box]
        col["srf_cx"] = 0.5 * (min(bbx) + max(bbx))
        # LOCK the garment while dragging markers (Saeed: no accidental
        # garment selection in the markers step) - unlocked on exit
        g.select_set(False)
        g.hide_select = True
        props.fitwiz_step = 2
        self.report({'INFO'}, "%d markers - drag the wrong ones" % made)
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_side(bpy.types.Operator):
    """Side pass: snap to the LEFT view to push markers forward/back
    (depth) against the side reference"""
    bl_idname = "smartrig.fitwiz_side"
    bl_label = "Side Pass"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            bpy.ops.view3d.view_axis(type='LEFT')
        except Exception:
            pass
        context.scene.smartrig.fitwiz_step = 3
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_extras(bpy.types.Operator):
    """Register rigid extras: belt, pockets, buttons, flowers... select
    their vertices in Edit Mode then press 'Register Selected'"""
    bl_idname = "smartrig.fitwiz_extras"
    bl_label = "Extras Step"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        g = props.garment_object
        if g is None:
            return {'CANCELLED'}
        if g.vertex_groups.get(VG_RIGID) is None:
            g.vertex_groups.new(name=VG_RIGID)
        # AUTO PRE-FILL the part registration (suggestion only - the user
        # corrects): Sleeve = fabric outside the torso column above pelvis,
        # Collar = above the neck marker, Lower = below the pelvis marker.
        import numpy as np
        me = g.data
        nv = len(me.vertices)
        co = np.empty(nv * 3)
        me.vertices.foreach_get("co", co)
        R3 = np.array(g.matrix_world.to_3x3())
        w = co.reshape(-1, 3) @ R3.T + np.array(g.matrix_world.translation[:])

        def mark(nm):
            ob = bpy.data.objects.get(MARKER_PREFIX + nm)
            return np.array(ob.matrix_world.translation[:]) if ob else None

        pel, nk = mark("pelvis"), mark("neck")
        wl_, wr_ = mark("waist_w_l"), mark("waist_w_r")
        created = {}
        for nm in ("SRF_Sleeve", "SRF_Collar", "SRF_Lower"):
            vg = g.vertex_groups.get(nm)
            created[nm] = vg is None
            if vg is None:
                g.vertex_groups.new(name=nm)
        if pel is not None and nk is not None:
            ax = nk - pel
            L2 = float(ax @ ax) + 1e-12
            tt = np.clip(((w - pel) @ ax) / L2, 0.0, 1.0)
            rho = np.linalg.norm(w - (pel + tt[:, None] * ax), axis=1)
            t_r = 0.5 * float(np.linalg.norm(wl_ - wr_)) \
                if (wl_ is not None and wr_ is not None) \
                else 0.22 * float(np.sqrt(L2))
            fills = {
                "SRF_Sleeve": (rho > 1.45 * t_r) & (w[:, 2] > pel[2]),
                "SRF_Collar": w[:, 2] > nk[2],
                "SRF_Lower": w[:, 2] < pel[2],
            }
            for nm, m in fills.items():
                if created.get(nm) and m.any():
                    g.vertex_groups[nm].add(
                        [int(i) for i in np.nonzero(m)[0]], 1.0, 'REPLACE')
        g.hide_select = False          # extras step needs Edit Mode on it
        props.fitwiz_step = 4
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_register(bpy.types.Operator):
    """Add the selected vertices (Edit Mode) to the rigid extras - each
    connected piece will move as ONE solid object during the fit"""
    bl_idname = "smartrig.fitwiz_register"
    bl_label = "Register Selected Part"
    bl_options = {'REGISTER', 'UNDO'}

    part: bpy.props.EnumProperty(items=[
        ('SLEEVE', "Sleeve", "Binds to the arms only"),
        ('COLLAR', "Collar", "Follows the neck, stays crisp"),
        ('LOWER', "Lower", "Kandura/dress/skirt bottom: spine-bound"),
        ('RIGID', "Rigid", "Belt/buttons/ornaments: one solid piece")])

    _GROUPS = {'SLEEVE': "SRF_Sleeve", 'COLLAR': "SRF_Collar",
               'LOWER': "SRF_Lower", 'RIGID': VG_RIGID}

    def execute(self, context):
        g = _garment(context)
        if g is None or g.mode != 'EDIT':
            self.report({'ERROR'},
                        "Enter Edit Mode on the garment and select the "
                        "part first (sleeve / collar / belt...)")
            return {'CANCELLED'}
        name = self._GROUPS[self.part]
        # one part per vertex: remove from the other part groups first
        for other in self._GROUPS.values():
            if other == name:
                continue
            vg = g.vertex_groups.get(other)
            if vg is not None:
                g.vertex_groups.active_index = vg.index
                bpy.ops.object.vertex_group_remove_from()
        vg = g.vertex_groups.get(name)
        if vg is None:
            vg = g.vertex_groups.new(name=name)
        g.vertex_groups.active_index = vg.index
        bpy.ops.object.vertex_group_assign()
        self.report({'INFO'}, "Registered as %s" % self.part.title())
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_show(bpy.types.Operator):
    """SEE the part on YOUR garment: highlights the registered vertices in
    Edit Mode so you know exactly what the addon detected - and where"""
    bl_idname = "smartrig.fitwiz_show"
    bl_label = "Show Part"
    bl_options = {'REGISTER', 'UNDO'}

    part: bpy.props.EnumProperty(items=[
        ('SLEEVE', "Sleeve", "The arm tube: shoulder to cuff"),
        ('COLLAR', "Collar", "The band around the neck opening"),
        ('LOWER', "Lower", "Everything hanging below the waist"),
        ('RIGID', "Rigid", "Belt / buttons / pockets / ornaments")])

    def execute(self, context):
        g = _garment(context)
        if g is None:
            return {'CANCELLED'}
        name = SMARTRIG_OT_fitwiz_register._GROUPS[self.part]
        vg = g.vertex_groups.get(name)
        if vg is None:
            self.report({'WARNING'}, "Nothing registered as %s yet"
                        % self.part.title())
            return {'CANCELLED'}
        context.view_layer.objects.active = g
        g.hide_select = False
        g.select_set(True)
        if g.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        g.vertex_groups.active_index = vg.index
        bpy.ops.object.vertex_group_select()
        self.report({'INFO'}, "%s highlighted on the garment"
                    % self.part.title())
        return {'FINISHED'}


class SMARTRIG_OT_fitwiz_go(bpy.types.Operator):
    """FIT: the markers override the automatic analysis and the full match
    engine runs (warp + design preservation + live garment rig)"""
    bl_idname = "smartrig.fitwiz_go"
    bl_label = "Fit! (wizard)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.smartrig
        if props.garment_object is not None:
            props.garment_object.hide_select = False
            if props.garment_object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        _restore(context)          # the character comes back for the fit
        if props.fit_body_object is not None:
            try:
                props.fit_body_object.hide_set(False)   # ALWAYS visible
            except Exception:
                pass
        r = bpy.ops.smartrig.mannequin_match()
        if 'FINISHED' in r:
            col = bpy.data.collections.get(MARKER_COL)
            if col is not None:
                col.hide_viewport = True     # kept for a later refit
            clear_reference()
            props.fitwiz_xray = 1.0      # solid again
            props.fitwiz_step = 0
        return r


class SMARTRIG_OT_fitwiz_cancel(bpy.types.Operator):
    """Leave the wizard and remove its markers"""
    bl_idname = "smartrig.fitwiz_cancel"
    bl_label = "Cancel Wizard"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        g = _garment(context)
        if g is not None:
            g.hide_select = False
        clear_markers()
        clear_reference()
        _restore(context)
        context.scene.smartrig.fitwiz_xray = 1.0
        context.scene.smartrig.fitwiz_step = 0
        return {'FINISHED'}


_CLASSES = (SMARTRIG_OT_fitwiz_start, SMARTRIG_OT_fitwiz_view,
            SMARTRIG_OT_fitwiz_markers, SMARTRIG_OT_fitwiz_side,
            SMARTRIG_OT_fitwiz_extras, SMARTRIG_OT_fitwiz_register,
            SMARTRIG_OT_fitwiz_show, SMARTRIG_OT_fitwiz_go,
            SMARTRIG_OT_fitwiz_cancel)


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)
    if _mirror_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_mirror_handler)


def unregister():
    if _mirror_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_mirror_handler)
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
