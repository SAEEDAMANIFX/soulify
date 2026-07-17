"""Soulify - CHARACTER tab: name, organize, check & fix, link-ready.

Blender-Studio convention: ONE top collection per character that a project
file links as a single unit:

    CH-<name>                 (link THIS into shots)
        RIG-<name>            (the control rig object)
        GEO-<name>            (collection: every mesh bound to the rig)
        HLP-<name>            (collection, hidden: markers, grids, lattices,
                               ribbons, hook empties, the metarig)
        WGT-<name>            (collection, excluded: widget objects)

Also here: the full-face "Start Over" (cancel) that wipes EVERYTHING the
face wizard/builder created so the user can restart cleanly.
"""
import bpy


# ------------------------------------------------------------------ helpers
def _rig(props):
    from . import face as _face
    return _face._target_rig()


def _bound_meshes(rig, sc):
    out = []
    for ob in sc.objects:
        if ob.type != 'MESH' or ob.name.startswith(("WGT", "HLP-", "SR_")):
            continue
        if any(m.type == 'ARMATURE' and m.object is rig
               for m in ob.modifiers):
            out.append(ob)
    return out


def _ensure_coll(name, parent):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
    if c.name not in {ch.name for ch in parent.children}:
        try:
            parent.children.link(c)
        except Exception:
            pass
    return c


def _move_obj(ob, coll):
    for uc in list(ob.users_collection):
        if uc is not coll:
            try:
                uc.objects.unlink(ob)
            except Exception:
                pass
    if ob.name not in coll.objects:
        try:
            coll.objects.link(ob)
        except Exception:
            pass


def _move_coll(c, parent):
    if c is None or c is parent:
        return
    for p in list(bpy.data.collections) + [bpy.context.scene.collection]:
        if c.name in {ch.name for ch in p.children} and p is not parent:
            try:
                p.children.unlink(c)
            except Exception:
                pass
    if c.name not in {ch.name for ch in parent.children}:
        try:
            parent.children.link(c)
        except Exception:
            pass


def _layer_coll(lc, name):
    if lc.collection.name == name:
        return lc
    for ch in lc.children:
        r = _layer_coll(ch, name)
        if r is not None:
            return r
    return None


# ------------------------------------------------------------ organize op
class SMARTRIG_OT_char_organize(bpy.types.Operator):
    """Name the character and organize EVERYTHING into one link-ready
    CH-<name> collection (rig + GEO + hidden HLP + excluded WGT) -
    Blender-Studio style. Safe to re-run any time"""
    bl_idname = "smartrig.char_organize"
    bl_label = "Organize Character (Link-Ready)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        sc = context.scene
        props = sc.smartrig
        name = (props.char_name or "").strip()
        if not name:
            body = getattr(props, "target_mesh", None)
            name = (body.name.split(".")[0] if body else "character")
            props.char_name = name
        rig = _rig(props)
        if rig is None:
            self.report({'ERROR'}, "Generate the rig first")
            return {'CANCELLED'}

        # --- rename the rig (pointer-based lookups survive this) ---
        rig_name = "RIG-" + name
        if rig.name != rig_name:
            rig.name = rig_name
        if rig.data is not None:
            rig.data.name = rig_name

        # --- collections ---
        root = sc.collection
        ch = _ensure_coll("CH-" + name, root)
        try:
            ch.color_tag = 'COLOR_01'
        except Exception:
            pass
        geo = _ensure_coll("GEO-" + name, ch)
        hlp = _ensure_coll("HLP-" + name, ch)
        wgt = _ensure_coll("WGT-" + name, ch)

        _move_obj(rig, ch)
        n_geo = 0
        meshes = _bound_meshes(rig, sc)
        # registered part objects too (may be unbound yet)
        for attr in ("skin_eye_l", "skin_eye_r", "skin_teeth_up",
                     "skin_teeth_low", "skin_tongue", "skin_brows",
                     "skin_lashes", "skin_hair"):
            ob = getattr(props, attr, None)
            if ob is not None and ob not in meshes:
                meshes.append(ob)
        body = getattr(props, "target_mesh", None)
        if body is not None and body not in meshes:
            meshes.append(body)
        for ob in meshes:
            _move_obj(ob, geo)
            n_geo += 1

        # helpers: metarig, grids, markers, ribbons, lattices, hook empties
        n_hlp = 0
        for ob in list(sc.objects) + [o for o in bpy.data.objects
                                      if o.name.startswith("HLP-")]:
            nm = ob.name
            if (nm.startswith(("HLP-", "SR_", "face_", "fm."))
                    or nm in ("SR_Metarig", "META-" + name)
                    or ob.type == 'LATTICE'):
                if ob is rig:
                    continue
                _move_obj(ob, hlp)
                n_hlp += 1
        for cn in ("SR_Markers", "SR_FaceMarkers", "SR_FaceHelpers"):
            _move_coll(bpy.data.collections.get(cn), hlp)

        # widgets
        n_wgt = 0
        for ob in list(bpy.data.objects):
            if ob.name.startswith("WGT") and ob.type == 'MESH':
                _move_obj(ob, wgt)
                n_wgt += 1
        _move_coll(bpy.data.collections.get("WGTS_rig"), wgt)
        _move_coll(bpy.data.collections.get("WGTS_" + rig.name), wgt)

        # visibility: HLP hidden, WGT excluded
        hlp.hide_viewport = True
        hlp.hide_render = True
        lc = _layer_coll(context.view_layer.layer_collection, wgt.name)
        if lc is not None:
            lc.exclude = True
        lc = _layer_coll(context.view_layer.layer_collection, hlp.name)
        if lc is not None:
            lc.hide_viewport = True

        # keep the scene root clean: nothing of ours left at top level
        context.view_layer.update()
        self.report({'INFO'}, "CH-%s ready: %d geo, %d helpers, %d widgets - "
                    "link 'CH-%s' into your project" %
                    (name, n_geo, n_hlp, n_wgt, name))
        return {'FINISHED'}


# --------------------------------------------------------------- check op
class SMARTRIG_OT_char_check(bpy.types.Operator):
    """Check the character from every angle (transforms, binding, stray
    objects, helpers, structure) and auto-fix the safe issues.
    Results show below the button"""
    bl_idname = "smartrig.char_check"
    bl_label = "Character Check & Fix"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import json
        sc = context.scene
        props = sc.smartrig
        name = (props.char_name or "").strip() or "character"
        rig = _rig(props)
        lines = []

        def add(label, val, ok):
            lines.append([label, val, bool(ok)])

        if rig is None:
            self.report({'ERROR'}, "Generate the rig first")
            return {'CANCELLED'}

        # 1) transforms applied
        def _tf_ok(ob):
            return (max(abs(s - 1.0) for s in ob.scale) < 1e-4
                    and all(abs(r) < 1e-4 for r in ob.rotation_euler))
        bad_tf = [ob.name for ob in [rig] + _bound_meshes(rig, sc)
                  if not _tf_ok(ob)]
        add("Transforms applied (scale/rot)",
            "all clean" if not bad_tf else ", ".join(bad_tf[:3]), not bad_tf)

        # 2) binding: every GEO mesh has an armature modifier -> this rig
        meshes = _bound_meshes(rig, sc)
        add("Bound meshes", "%d" % len(meshes), len(meshes) > 0)
        no_vg = [ob.name for ob in meshes
                 if not any(g.name.startswith("DEF") for g in
                            ob.vertex_groups)]
        add("Deform weights present",
            "all" if not no_vg else "missing: " + ", ".join(no_vg[:3]),
            not no_vg)

        # 3) stray duplicates: meshes far from origin with no rig binding
        stray = []
        for ob in sc.objects:
            if ob.type != 'MESH' or ob in meshes or \
                    ob.name.startswith(("WGT", "HLP-", "SR_")):
                continue
            if abs(ob.location.y) > 50 or abs(ob.location.x) > 50:
                stray.append(ob.name)
        add("Stray far-away meshes (idle copies)",
            "none" if not stray else ", ".join(stray[:4]), not stray)

        # 4) visible helpers (FIX: hide them)
        n_fixed_h = 0
        for ob in sc.objects:
            if ob.name.startswith(("HLP-", "SR_", "face_", "fm.")) and \
                    ob.visible_get():
                ob.hide_viewport = True
                n_fixed_h += 1
        add("Helpers hidden", ("fixed %d" % n_fixed_h) if n_fixed_h
            else "all hidden", True)

        # 5) structure: everything under CH-<name>
        ch = bpy.data.collections.get("CH-" + name)
        if ch is None:
            add("CH-%s collection" % name, "missing - press Organize", False)
        else:
            inside = set()

            def walk(c):
                for o in c.objects:
                    inside.add(o.name)
                for cc in c.children:
                    walk(cc)
            walk(ch)
            outs = [ob.name for ob in [rig] + meshes
                    if ob.name not in inside]
            add("All character objects in CH-%s" % name,
                "yes" if not outs else "outside: " + ", ".join(outs[:3]),
                not outs)

        # 6) rig naming
        add("Rig named RIG-%s" % name, rig.name,
            rig.name == "RIG-" + name)

        # 7) orphan data (FIX: purge)
        try:
            n_orph = bpy.data.orphans_purge(do_recursive=True)
        except Exception:
            n_orph = 0
        add("Orphan data purged", "%d removed" % n_orph, True)

        sc["sr_char_check"] = json.dumps(lines)
        npass = sum(1 for l in lines if l[2])
        self.report({'INFO'} if npass == len(lines) else {'WARNING'},
                    "Character check: %d/%d passed" % (npass, len(lines)))
        return {'FINISHED'}


# ---------------------------------------------------- face START OVER op
class SMARTRIG_OT_face_start_over(bpy.types.Operator):
    """CANCEL the face rig completely: removes ALL face bones, ribbons,
    lattices, hook empties, markers, grid, expressions and face weights -
    the body rig stays intact. Then start the face wizard fresh"""
    bl_idname = "smartrig.face_start_over"
    bl_label = "Start Over (Face)"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from . import face as _face
        from . import storm_face as _sf
        sc = context.scene
        props = sc.smartrig
        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
        rig = _face._target_rig()
        body = getattr(props, "target_mesh", None)
        n_bones = 0
        if rig is not None:
            _face._ensure_selectable(context, rig)
            context.view_layer.objects.active = rig
            try:
                spec = _sf._load("storm_face_spec.json")
                n_bones = _sf._clean_previous(_face, props, context, rig,
                                              body, spec)
            except Exception as e:
                self.report({'WARNING'}, "bone cleanup: %s" % e)
            # face bone collections
            arm = rig.data
            face_colls = ["Face", "Face Upper", "Face Lower", "Brows",
                          "Brows global", "Brows local", "Eyes",
                          "Eyes global", "Eyes local", "Eyes_micro",
                          "Nose", "Cheeks", "Ears", "Upper Master",
                          "Lower Master", "Lattices", "Mouth",
                          "Mouth global", "Mouth local", "Mouth micro",
                          "Jawline", "Teeth", "Tongue", "Face MCH",
                          "Face Display", "Face (Primary)",
                          "Face (Secondary)", "Face (MCH)"]
            for cn in face_colls:
                c = arm.collections_all.get(cn)
                if c is not None:
                    try:
                        arm.collections.remove(c)
                    except Exception:
                        pass
            if "sr_storm_face_bones" in arm:
                del arm["sr_storm_face_bones"]

        # helper objects (ribbons, hooks, lattices) + markers + grid
        for ob in list(bpy.data.objects):
            if ob.name.startswith(("HLP-SR-", "HLP-storm-", "face_", "fm_face")):
                bpy.data.objects.remove(ob, do_unlink=True)
        g = bpy.data.objects.get(_face.GRID_NAME)
        if g is not None:
            bpy.data.objects.remove(g, do_unlink=True)
        for cn in ("SR_FaceMarkers", "SR_FaceHelpers"):
            c = bpy.data.collections.get(cn)
            if c is not None:
                for ob in list(c.objects):
                    bpy.data.objects.remove(ob, do_unlink=True)
                bpy.data.collections.remove(c)

        # expressions: action + list + correctives + check results
        act = bpy.data.actions.get("SR_Expressions")
        if act is not None:
            bpy.data.actions.remove(act)
        try:
            sc.sr_face_expressions.clear()
        except Exception:
            pass
        for ob in sc.objects:
            if ob.type != 'MESH' or ob.data.shape_keys is None:
                continue
            for kb in list(ob.data.shape_keys.key_blocks):
                if kb.name.startswith("CORR-"):
                    ob.shape_key_remove(kb)
            names = list(ob.data.get("sr_expr_baked", []))
            for nm in names:
                kb = ob.data.shape_keys.key_blocks.get(nm) \
                    if ob.data.shape_keys else None
                if kb is not None:
                    ob.shape_key_remove(kb)
            if "sr_expr_baked" in ob.data:
                del ob.data["sr_expr_baked"]
        for k in ("sr_rig_check",):
            if k in sc:
                del sc[k]

        # unlock the view, show the rig again
        try:
            from . import markers as _mk
            _mk.lock_front_view(context, False)
        except Exception:
            pass
        try:
            _face.set_rigs_hidden(False)
        except Exception:
            pass
        self.report({'INFO'}, "Face wiped clean (%d bones removed) - press "
                    "Face Markers to start over" % n_bones)
        return {'FINISHED'}


# --------------------------------------------------------------------- UI
def draw(layout, context):
    import json
    sc = context.scene
    props = sc.smartrig
    box = layout.box()
    box.label(text="Character", icon='OUTLINER_OB_ARMATURE')
    box.prop(props, "char_name", text="Name", icon='SORTALPHA')
    r = box.row(); r.scale_y = 1.5
    r.operator("smartrig.char_organize", icon='OUTLINER_COLLECTION')
    r = box.row(); r.scale_y = 1.3
    r.operator("smartrig.char_check", icon='CHECKMARK')
    data = sc.get("sr_char_check")
    if data:
        bx = box.box()
        try:
            for lbl, val, ok in json.loads(data):
                rr = bx.row()
                rr.label(text="%s - %s" % (lbl, val),
                         icon=('CHECKMARK' if ok else 'CANCEL'))
        except Exception:
            pass
    name = (props.char_name or "").strip()
    if name and bpy.data.collections.get("CH-" + name):
        box.label(text="Link 'CH-%s' into any project file" % name,
                  icon='LINKED')

    dz = layout.box()
    dz.label(text="Danger Zone", icon='ERROR')
    r = dz.row(); r.scale_y = 1.2
    r.operator("smartrig.face_start_over", icon='CANCEL')


_classes = (SMARTRIG_OT_char_organize, SMARTRIG_OT_char_check,
            SMARTRIG_OT_face_start_over)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
