import bpy
from bpy.props import (PointerProperty, IntProperty, BoolProperty,
                       FloatProperty, StringProperty, EnumProperty,
                       CollectionProperty)
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


def _lock_update(self, context):
    try:
        from . import markers
        markers.set_character_selectable(self, not self.lock_mesh)
    except Exception:
        pass


def _skirt_jiggle_seg_update(self, context):
    """Live-update the B-bone segments on the skirt deform bones (smoother wave)."""
    try:
        from . import skirt, metarig
        rig = metarig._generated_rig()
        if rig is not None:
            skirt.set_skirt_bbone_segments(rig, self.skirt_jiggle_segments)
    except Exception as e:
        print("SmartRig jiggle-seg update:", e)


def _spine_neck_update(self, context):
    """Real-time rebuild of the metarig spine with the chosen torso (spine_count)
    and neck (neck_count) counts, preserving positions. Only acts once the
    metarig exists (before Generate); silently no-ops otherwise."""
    try:
        if bpy.data.objects.get("SR_Metarig") is None:
            return
        from . import metarig
        metarig.set_spine_neck(self)
    except Exception as e:
        print("SmartRig spine/neck live update:", e)


def _skirt_update(self, context):
    """Real-time rebuild of the skirt bones when Columns/Rows change (only for
    the mesh-driven modes; manual edits are never overwritten)."""
    try:
        from . import skirt
        skirt.live_rebuild(context)
    except Exception as e:
        print("SmartRig skirt live update:", e)


def _skirt_live_update(self, context):
    """Live-tune skirt follow strength / max angle on the generated rig with no
    rebuild and no mode change (ARP-style interactive settings)."""
    try:
        from . import skirt
        skirt.live_tune(context)
    except Exception as e:
        print("SmartRig skirt live tune:", e)


def _skirt_collide_update(self, context):
    """Live-tune the ARP-style kilt collision (collide on/off, distance, spread,
    falloff) on the generated rig with no rebuild (ARP interactive settings)."""
    try:
        from . import skirt
        skirt.live_kilt_tune(context)
    except Exception as e:
        print("SmartRig skirt collide tune:", e)


def _skirt_follow_update(self, context):
    try:
        from . import skirt
        skirt.live_follow_tune(context)
    except Exception as e:
        print("SmartRig follow tune:", e)


def _antipen_update(self, context):
    try:
        from . import skirt
        skirt.live_antipen_tune(context)
    except Exception as e:
        print("SmartRig anti-pen tune:", e)


def _jiggle_update(self, context):
    """Push jiggle sliders into the SKC_master props that the spring handler reads."""
    try:
        from . import skirt
        for rig in skirt._jiggle_rigs():
            rig["jiggle"] = 1.0 if getattr(self, "skirt_jiggle", True) else 0.0
            rig["jiggle_amount"] = float(self.jiggle_amount)
            rig["jiggle_stiffness"] = float(self.jiggle_stiffness)
            rig["jiggle_damping"] = float(self.jiggle_damping)
    except Exception as e:
        print("SmartRig jiggle tune:", e)
    _jiggle_force_update(self, context)


def _jiggle_force_update(self, context):
    """Re-run the live spring solver right now so Wind / Gravity / Lift changes are
    visible while the timeline is PAUSED. The solver is a frame_change_post handler,
    so without this it would only recompute when the frame actually changes - which
    is why dragging Lift on a paused frame looked like 'nothing happens'."""
    try:
        from . import skirt
        sc = getattr(context, "scene", None) or bpy.context.scene
        skirt.skirt_jiggle_handler(sc)
    except Exception as e:
        print("SmartRig force tune:", e)


def _skin_selbones_update(self, context):
    """Selected Bones Only: show ONLY deform bones while picking; restore on off."""
    try:
        from . import skirt
        skirt.selected_bones_focus(context, self.skin_selected_bones_only)
    except Exception as e:
        print("Soulify selected-bones focus:", e)


def _kandura_grid_update(self, context):
    """REAL-TIME: Columns/Rows change -> rebuild the waist grid immediately,
    following the placed shape (subdivide-style)."""
    try:
        from . import kandura
        kandura.live_rebuild_waist(context)
    except Exception as e:
        print("SmartRig kandura live grid:", e)


def _kandura_sleeve_update(self, context):
    """REAL-TIME: sleeve bone counts -> rebuild the sleeve chains."""
    try:
        from . import kandura
        kandura.live_rebuild_sleeves(context)
    except Exception as e:
        print("SmartRig kandura live sleeves:", e)


def _kandura_collar_update(self, context):
    """REAL-TIME: collar count -> rebuild the collar ring."""
    try:
        from . import kandura
        kandura.live_rebuild_collar(context)
    except Exception as e:
        print("SmartRig kandura live collar:", e)


def _kandura_cuff_update(self, context):
    """REAL-TIME: cuff count -> rebuild the wrist rings."""
    try:
        from . import kandura
        kandura.live_rebuild_cuffs(context)
    except Exception as e:
        print("SmartRig kandura live cuff:", e)


def _kandura_smooth_update(self, context):
    """LIVE: fold-smoothing slider -> corrective smooth factor."""
    try:
        from . import kandura
        kandura.live_kandura_smooth(context)
    except Exception as e:
        print("SmartRig kandura smooth:", e)


def _kandura_antipen_update(self, context):
    """LIVE: body-clearance slider -> anti-pen modifier offset."""
    try:
        from . import kandura
        kandura.live_kandura_antipen(context)
    except Exception as e:
        print("SmartRig kandura anti-pen:", e)


def _kandura_floor_update(self, context):
    """LIVE: floor-clearance slider -> KAN_Floor modifier offset."""
    try:
        from . import kandura
        kandura.live_kandura_floor(context)
    except Exception as e:
        print("SmartRig kandura floor:", e)


def _kandura_focus_update(self, context):
    """Hide/show the body bones (focus on kandura bone placement)."""
    try:
        from . import kandura
        kandura.focus_apply(context, self.kandura_focus)
    except Exception as e:
        print("SmartRig kandura focus:", e)


def _kandura_mirror_update(self, context):
    """Start/stop X-axis mirroring while placing the kandura bones."""
    try:
        from . import kandura
        kandura.mirror_apply(context, self.kandura_mirror)
    except Exception as e:
        print("SmartRig kandura mirror:", e)


def _kandura_align_update(self, context):
    """Start/stop the Align-to-Surface system: FACE snapping so dragged bone
    points stick to the garment surface while placing them manually."""
    try:
        from . import kandura
        kandura.align_snap_apply(context, self.kandura_align_surface)
    except Exception as e:
        print("SmartRig kandura align:", e)


_SYNC_FROM_VIEWPORT = False   # set True by the WP->list sync timer to avoid re-select loops


def _weight_bone_update(self, context):
    if _SYNC_FROM_VIEWPORT:
        return
    """Selecting a bone in the Weight-Editing list lights it up on the character:
    unhide the rig, draw bones in front, reveal its bone collection, select +
    activate ONLY that bone so the user sees exactly what they are painting."""
    try:
        from .metarig import _generated_rig
        rig = _generated_rig()
        if rig is None:
            return
        arm = rig.data
        bones = arm.bones
        i = self.weight_bone_index
        if i < 0 or i >= len(bones):
            return
        b = bones[i]
        try:
            rig.hide_set(False); rig.hide_viewport = False; rig.show_in_front = True
        except Exception:
            pass
        try:
            for coll in b.collections:
                coll.is_visible = True
        except Exception:
            pass
        try:
            arm.bones.active = b          # active bone highlights in the viewport
        except Exception:
            pass
        # if we're weight painting, ALSO switch the painted group so clicking a
        # bone in the list instantly changes what you paint - no viewport Ctrl-click
        try:
            _msh = context.scene.smartrig.target_mesh
            if _msh is not None and _msh.mode == 'WEIGHT_PAINT':
                _vg = _msh.vertex_groups.get(b.name)
                if _vg is not None:
                    _msh.vertex_groups.active_index = _vg.index
        except Exception:
            pass
        # if the rig is in Pose Mode, also select just this pose bone (5.x path)
        try:
            pbones = rig.pose.bones
            for pb in pbones:
                pb.bone.select = False
        except Exception:
            pass
        try:
            rig.pose.bones[b.name].bone.select = True
        except Exception:
            pass
    except Exception:
        pass


class WeightFolder(PropertyGroup):
    """A user folder that groups deform bones for the Weight-Editing list.
    Membership is stored as comma-separated BONE NAMES so it survives Rigify
    regeneration (bones are re-created with the same names)."""
    name: StringProperty(name="Folder", default="Folder")
    members: StringProperty(default="")   # comma-separated bone names
    expanded: BoolProperty(default=True)
    uid: StringProperty(default="")       # stable unique id
    parent: StringProperty(default="")    # parent folder uid ("" = top level)


class SmartRigProps(PropertyGroup):
    ai_tools_path: StringProperty(
        name="AI Tools", subtype='DIR_PATH', default="",
        description="Folder with Auto-Rig Pro's AI files (info.dat + inference/)")
    target_mesh: PointerProperty(
        name="Character Mesh",
        type=bpy.types.Object,
        poll=_mesh_poll,
        description="The body mesh to rig (in A/T-pose, facing -Y)",
    )
    spine_count: IntProperty(name="Spine bones", default=4, min=2, max=8,
                             description="Torso (spine) segment count — rebuilds the "
                             "metarig spine live, preserving positions",
                             update=_spine_neck_update)
    neck_count: IntProperty(name="Neck bones", default=2, min=1, max=4,
                            description="Neck segment count — rebuilds the metarig "
                            "neck live, preserving positions",
                            update=_spine_neck_update)
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
    # ---- live link & marker visibility ----
    live_link: BoolProperty(default=False)
    markers_hidden: BoolProperty(default=False)
    lock_mesh: BoolProperty(
        name="Lock Mesh Selection", default=False, update=_lock_update,
        description="Make the character mesh un-clickable so box-select / clicks hit only the markers")
    show_tools: BoolProperty(name="Marker Tools", default=True,
        description="Collapse / expand the Marker Tools section")
    show_roll: BoolProperty(name="Bone Roll", default=False,
        description="Collapse / expand the Bone Roll editing section")
    show_face: BoolProperty(name="Face Rig", default=False,
        description="Collapse / expand the Face Rig (beta) tools")
    show_rigify: BoolProperty(name="Rigify", default=False,
        description="Collapse / expand the Rigify samples list")
    rig_generated: BoolProperty(name="Rig Generated", default=False,
        description="Whether a Rigify rig has been generated from the metarig")
    show_display: BoolProperty(name="Display", default=True,
        description="Collapse / expand the viewport display controls for the rig")
    show_align: BoolProperty(name="Align", default=True,
        description="Collapse / expand the Align & Wireframe section")
    # ---- short skirt (ARP Kilt-style) sample ----
    skirt_source: EnumProperty(
        name="Skirt Source", default='MERGED',
        items=[('MANUAL', "Manual", "Add a starter ring of bones, then place / edit them by hand"),
               ('SEPARATE', "Separate Mesh", "The skirt is its own object - pick it with the eyedropper"),
               ('MERGED', "Merged with Body", "The skirt is part of the character mesh - select it in Edit Mode and Register")])
    skirt_object: PointerProperty(
        name="Skirt Mesh", type=bpy.types.Object, poll=_mesh_poll,
        description="The separate skirt mesh object to analyse")
    skirt_columns: IntProperty(name="Columns", default=8, min=4, soft_max=32, max=256,
        update=_skirt_update,
        description="Number of skirt chains around the hips. Effectively open (type any "
        "value up to 256). Placement is purely mathematical (even angular sectors) so it "
        "never misplaces; but the real ceiling is the skirt MESH density - past ~1 column "
        "per vertical mesh loop, extra columns overlap or get skipped. Subdivide the skirt "
        "mesh for more clean columns")
    skirt_rows: IntProperty(name="Rows", default=2, min=1, soft_max=8, max=64,
        update=_skirt_update,
        description="Segments down each skirt chain (more = smoother bending). Effectively "
        "open (up to 64). Rows are even Z-slices, so any count places cleanly; B-bone "
        "Segments already smooth the bend, so 2-6 is usually plenty")
    skirt_length: FloatProperty(name="Length", default=0.6, min=0.15, max=1.0,
        description="Skirt length from waist toward the knee (0.15 short ... 1.0 to knee)")
    skirt_collide: BoolProperty(name="Collide with Legs", default=True, update=_skirt_collide_update,
        description="Add ARP-style constrained collision with the legs")
    skirt_front_axis: EnumProperty(
        name="Front", default='-Y', update=_skirt_update,
        description="Which world axis is the FRONT of the skirt (character convention: "
        "front = -Y). The front-centre and back-centre columns and the left/right split "
        "are measured from this. Change it only if the skirt was imported rotated",
        items=[('-Y', "-Y (front)", "Front faces -Y (Blender character default)"),
               ('+Y', "+Y", "Front faces +Y"),
               ('+X', "+X", "Front faces +X"),
               ('-X', "-X", "Front faces -X")])
    skirt_symmetric: BoolProperty(name="Symmetric (mirror L/R)", default=True,
        update=_skirt_update,
        description="Force the skirt columns to be perfectly mirror-symmetric about the "
        "centre (X=0): starts from the front-centre column and makes the left half an "
        "exact mirror of the right. Enables clean X-mirror posing, symmetric weights and "
        "balanced deformation. Turn off for an intentionally asymmetric skirt")
    jiggle_gravity: FloatProperty(name="Gravity", default=0.0, min=0.0, max=3.0,
        update=_jiggle_force_update,
        description="Downward pull on the jiggle (skirt + chest) - makes cloth hang/sag")
    jiggle_wind: FloatProperty(name="Wind", default=0.0, min=0.0, max=6.0,
        update=_jiggle_force_update,
        description="Wind force on the jiggle - blows the cloth in the wind direction")
    jiggle_wind_dir: FloatProperty(name="Wind Dir", default=0.0, min=0.0, max=360.0,
        subtype='ANGLE', update=_jiggle_force_update,
        description="Wind direction around the character (0 = front)")
    jiggle_wind_turb: FloatProperty(name="Gust", default=0.3, min=0.0, max=1.0,
        update=_jiggle_force_update,
        description="Wind gustiness / turbulence (0 = steady, 1 = very gusty)")
    jiggle_wind_speed: FloatProperty(name="Wind Speed", default=1.0, min=0.1, max=15.0,
        update=_jiggle_force_update,
        description="How fast the wind gusts change / move (higher = faster gusts)")
    jiggle_wind_billow: FloatProperty(name="Billow", default=1.2, min=0.0, max=12.0,
        update=_jiggle_force_update,
        description="How much the wind lifts the upper/middle of the skirt (not just "
        "the hem) - higher = the whole skirt billows")
    jiggle_wind_lift: FloatProperty(name="Lift", default=0.0, min=0.0, max=20.0,
        update=_jiggle_force_update,
        description="Blows the skirt UP by rotating each panel outward/up like posing "
        "finger bones. Low = gentle lift; high = full umbrella; very high = flips "
        "inside-out over the top")
    skirt_jiggle_segments: IntProperty(name="Jiggle Segments", default=3, min=1, max=8,
        description="B-bone segments on the skirt deform bones: higher = smoother, "
        "more professional cloth wave (live; no rebuild). 1 = faceted/off",
        update=_skirt_jiggle_seg_update)
    skirt_use_masters: BoolProperty(name="Region Masters", default=True,
        description="Build a global + per-region (front/sides/back) master controls "
        "so you can pose whole regions of the skirt at once")
    skirt_masters: IntProperty(name="Master sectors", default=4, min=2, max=12,
        description="How many region master controls around the waist "
        "(4 = front/right/back/left). Increase for finer regional control")
    skirt_smooth: BoolProperty(name="Corrective Smooth", default=False,
        description="Add a Corrective Smooth modifier to relax pinching from Follow "
        "Body / Anti-Penetration (placed before Anti-Pen so it stays out of the body)")
    skirt_smooth_factor: FloatProperty(name="Smooth", default=0.5, min=0.0, max=1.0,
        description="Corrective Smooth strength")
    skirt_smooth_iter: IntProperty(name="Smooth iterations", default=5, min=1, max=30)
    skirt_merged_help: BoolProperty(name="Merged skirt help", default=False,
        description="Show what a merged skirt supports and what needs a separate mesh")
    skirt_sep_help: BoolProperty(name="Separate skirt help", default=False,
        description="Show what a separate skirt mesh supports")
    skirt_collider_l: StringProperty(name="Left Collider", default="DEF-thigh.L",
        description="Left leg bone the skirt collides with (e.g. thigh.L)")
    skirt_collider_r: StringProperty(name="Right Collider", default="DEF-thigh.R",
        description="Right leg bone the skirt collides with (e.g. thigh.R)")
    skirt_collide_dist: FloatProperty(name="Collision Distance", default=0.12, min=0.01, max=0.6,
        update=_skirt_collide_update,
        description="Clearance kept between the skirt and the leg bones")
    skirt_collide_spread: FloatProperty(name="Spread", default=1.0, min=0.0, max=2.0,
        update=_skirt_collide_update,
        description="How many columns around each leg are affected by the collision")
    skirt_collide_falloff: FloatProperty(name="Base Clearance", default=0.0, min=0.0, max=1.0,
        update=_skirt_collide_update,
        description="Base clearance kept even at rest (ARP collide_dist_falloff)")
    skirt_follow: FloatProperty(name="Leg Follow", default=0.6, min=0.0, max=1.5, update=_skirt_live_update,
        description="How strongly the skirt rotates with the leg (Pierrick-style). 0 = none")
    skirt_limit_deg: FloatProperty(name="Max Angle", default=60.0, min=5.0, max=150.0, update=_skirt_live_update,
        description="Maximum outward swing of a skirt panel (lower = less chance skirt bones overlap)")
    kandura_object: PointerProperty(
        name="Kandura", type=bpy.types.Object, poll=_mesh_poll,
        description="The kandura (thobe) mesh")
    kandura_columns: IntProperty(
        name="Columns", default=8, min=4, max=32,
        update=_kandura_grid_update,
        description="Waist-down bone columns around the kandura tube. "
        "REAL-TIME: changing it re-divides the bones immediately, keeping "
        "the shape you placed (subdivide-style)")
    kandura_rows: IntProperty(
        name="Rows / Zone", default=1, min=1, max=6,
        update=_kandura_grid_update,
        description="Bone rows PER ZONE: 1 = one row for the THIGH (waist "
        "to knee) + one row for the SHIN (knee to hem); 2 = 2 up + 2 down, "
        "etc. The knee ring is always a boundary for the leg automation. "
        "REAL-TIME update, keeps the placed shape")
    kandura_collar_count: IntProperty(
        name="Collar Bones", default=6, min=3, max=16,
        update=_kandura_collar_update,
        description="Number of collar bones added as a ring around the "
        "neck. REAL-TIME: rebuilds the ring keeping the placed shape")
    kandura_cuff_rows: IntProperty(
        name="Cuff Rows", default=1, min=1, max=4,
        update=_kandura_cuff_update,
        description="LENGTHWISE subdivision of each cuff bone (a chain "
        "from the registered loop down to the sleeve end) - more rows = "
        "smoother cuff automation. REAL-TIME, rebuilds from the "
        "registered loop")
    kandura_cuff_count: IntProperty(
        name="Cuff Bones", default=6, min=3, max=16,
        update=_kandura_cuff_update,
        description="Number of cuff bones added as a ring around each "
        "sleeve END (wrist opening). REAL-TIME: rebuilds the rings "
        "keeping the placed shape")
    kandura_smooth: FloatProperty(
        name="Fold Smoothing", default=0.65, min=0.0, max=1.0,
        update=_kandura_smooth_update,
        description="Corrective-smooth strength on the SLEEVE fabric: "
        "evens the rolled-up folds into clean rounds (LIVE)")
    kandura_antipen_offset: FloatProperty(
        name="Body Clearance", default=0.005, min=0.0, max=0.05,
        update=_kandura_antipen_update,
        description="Anti-penetration: minimum gap kept between the body "
        "and the kandura cloth (LIVE)")
    kandura_floor_offset: FloatProperty(
        name="Floor Clearance", default=0.004, min=0.0, max=0.05,
        update=_kandura_floor_update,
        description="Ground clamp: the kandura cloth is kept this far "
        "ABOVE the auto-detected floor - deep sits pool the hem on the "
        "ground instead of sinking through it (LIVE)")
    kandura_focus: BoolProperty(
        name="Hide Body Bones", default=False,
        update=_kandura_focus_update,
        description="Hide the BODY bones of the metarig so only the "
        "kandura bones stay visible - focus on garment bone placement. "
        "Turn OFF to show the body bones again")
    kandura_mirror: BoolProperty(
        name="Mirror", default=True,
        update=_kandura_mirror_update,
        description="ON: X-axis mirror while placing bones — .L/.R bones "
        "(sleeves) mirror live while dragging; for the waist grid and "
        "collar use 'Mirror Selected to Other Side'. OFF: no mirroring")
    kandura_align_surface: BoolProperty(
        name="Align to Surface", default=True,
        update=_kandura_align_update,
        description="ON: dragged bone heads/tails snap to the garment "
        "surface (FACE snapping) while you place them manually. "
        "OFF: free movement")
    kandura_sleeve_upper: IntProperty(
        name="Upper Arm Bones", default=2, min=1, max=6,
        update=_kandura_sleeve_update,
        description="Sleeve chain bones along the UPPER ARM part. "
        "REAL-TIME: rebuilds the chains keeping the placed shape")
    kandura_sleeve_lower: IntProperty(
        name="Lower Arm Bones", default=2, min=1, max=6,
        update=_kandura_sleeve_update,
        description="Sleeve chain bones along the FOREARM part (down to "
        "the cuff). REAL-TIME: rebuilds keeping the placed shape")
    show_kandura: BoolProperty(name="Kandura", default=False,
        description="Collapse / expand the Kandura (thobe) section")
    show_skirt: BoolProperty(name="Skirt", default=False,
        description="Collapse / expand the Short Skirt section")
    show_skirt_adv: BoolProperty(name="Advanced", default=False,
        description="Show the leg-bone pickers for the skirt collision (defaults are correct)")
    skirt_follow_body: FloatProperty(name="Follow Body", default=0.0, min=0.0, max=1.0,
        update=_skirt_follow_update,
        description="Blend the skirt from its own rig (0) to following the legs/hips (1) - great for sitting")
    skirt_antipen_offset: FloatProperty(name="Offset", default=0.01, min=0.0, max=0.1,
        update=_antipen_update,
        description="Clearance kept outside the body surface by the anti-penetration Shrinkwrap")
    skirt_jiggle: BoolProperty(name="Jiggle", default=True, update=_jiggle_update,
        description="Live spring secondary motion: the skirt sways when the body moves")
    jiggle_amount: FloatProperty(name="Amount", default=2.0, min=0.0, max=5.0, update=_jiggle_update,
        description="How much the skirt sways (0 = none)")
    jiggle_stiffness: FloatProperty(name="Stiffness", default=0.16, min=0.02, max=1.0, update=_jiggle_update,
        description="Spring stiffness - LOWER = slower, smoother flowing waves; higher = snappier/sharper")
    jiggle_damping: FloatProperty(name="Damping", default=0.28, min=0.05, max=0.99, update=_jiggle_update,
        description="Damping - higher settles the wobble faster (smoother, less jitter)")
    samples_expanded: StringProperty(default="Limbs",
        description="Internal: comma-separated names of expanded sample groups")
    # ---- Skin / Binding panel (ARP-style) ----
    skin_engine: EnumProperty(
        name="Engine", default='HEAT',
        items=[('HEAT', "Heat Maps", "Automatic weights from the Blender heat-map solver"),
               ('ENVELOPE', "Envelope", "Weights from the bone envelopes (fast, rough)")])
    skin_split_parts: BoolProperty(name="Split Parts", default=True,
        description="Keep the body and the skirt independent: the body ignores skirt bones and the skirt follows only its own bones")
    skin_preserve_volume: BoolProperty(name="Preserve Volume", default=False,
        description="Use preserve-volume (dual quaternion) deformation on the armature modifier")
    skin_smart_skirt: BoolProperty(name="Smart Skirt Weights", default=True,
        description="Skin the skirt from its known grid (angular column blend x vertical "
                    "row blend) instead of a generic heat map - cleaner, no cross-column bleed")
    skin_smart_bones: BoolProperty(name="Smart Bone Filter", default=True,
        description="Recognize which bones the mesh actually covers and bind ONLY those: "
                    "a shirt gets no finger/head weights, a hat binds only to the head, "
                    "a full body keeps every bone")
    skin_selected_bones_only: BoolProperty(name="Selected Bones Only", default=False,
        update=_skin_selbones_update,
        description="Re-bind ONLY the bones you select: turning this ON shows "
                    "JUST the deform bones (everything else hides) and enters "
                    "Pose Mode - pick the bones, then Bind. Turning it OFF "
                    "restores the normal bone display (like Auto-Rig Pro)")
    skin_selected_verts_only: BoolProperty(name="Selected Vertices Only", default=False,
        description="Recompute weights ONLY for the vertices selected in Edit Mode; "
                    "the rest of the mesh keeps its existing weights (like Auto-Rig Pro)")
    skin_optimize_highres: BoolProperty(name="Optimize High Res", default=False,
        description="Above the polycount threshold, solve the heat weights on a "
                    "DECIMATED proxy and transfer them to the full mesh - much "
                    "faster and more robust on very dense meshes")
    skin_polycount_threshold: IntProperty(name="Polycount Threshold", default=70000,
        min=1000, max=2000000,
        description="Vertex count above which the high-res proxy optimization kicks in")
    skin_refine_head: BoolProperty(name="Refine Head Weights", default=True,
        description="Post-bind smoothing pass on the head/neck weight groups")
    skin_smooth_twist: BoolProperty(name="Smooth Twist Weights", default=True,
        description="Post-bind smoothing pass on the limb twist segments "
                    "(upper_arm/forearm/thigh/shin) for cleaner twisting")
    skin_improve_hips: BoolProperty(name="Improve Hips Weights", default=True,
        description="Post-bind smoothing pass on the pelvis/spine/thigh blend zone")
    skin_improve_heels: BoolProperty(name="Improve Heels Weights", default=True,
        description="Post-bind smoothing pass on the foot/toe/heel weight groups")
    skin_apply_shapekeys: BoolProperty(name="Apply Shape Keys", default=False,
        description="Bake the CURRENT shape-key mix into the base mesh before "
                    "binding, so the weights are computed on the real shape")
    skin_scale_fix: BoolProperty(name="Scale Fix", default=True,
        description="Apply the mesh object scale before binding - the heat "
                    "solver misbehaves on scaled objects")
    skin_facial: BoolProperty(name="Facial Features", default=True,
        description="Also bind the facial feature meshes (eyes, teeth, tongue) "
                    "when binding: eyes -> eye/head bone, upper teeth -> head, "
                    "lower teeth + tongue -> jaw (rigid weights, like Auto-Rig Pro)")
    show_skin_facial: BoolProperty(name="Facial Features", default=False,
        description="Collapse / expand the Facial Features slots")
    skin_fine_hands: BoolProperty(name="Fine Finger Skin", default=True,
        description="PRO per-finger skinning on the hand region: each finger is "
                    "weighted to its OWN bones only (no bleed between fingers) - "
                    "crisp finger deformation. Register or auto-detect the region")
    skin_fine_feet: BoolProperty(name="Fine Toe Skin", default=False,
        description="PRO per-toe skinning on the foot region (no bleed between toes)")
    show_skin_fine: BoolProperty(name="Fine Skinning", default=False,
        description="Collapse / expand the fine hand/foot skinning tools")
    weight_bone_index: IntProperty(name="Deform Bone", default=0,
        description="Active bone in the weight-editing list",
        update=_weight_bone_update)
    weight_show_all_bones: BoolProperty(name="All Bones", default=False,
        description="Show every bone, not only the deform (DEF-) bones")
    weight_use_folders: BoolProperty(name="Folders", default=True,
        description="Group the deform bones into user folders (fingers, head, "
                    "eyes...) for faster navigation, locking and masking")
    weight_folders: CollectionProperty(type=WeightFolder)
    weight_folders_index: IntProperty(default=0)
    weight_isolated_folder: StringProperty(default="")   # uid of isolated folder, if any
    weight_folder_uid_next: IntProperty(default=0)
    weight_move_uid: StringProperty(default="")   # folder uid picked up for moving
    show_skin_pick: BoolProperty(name="Pick Bones", default=False,
        description="Collapse / expand the bone family pick buttons")
    skin_eye_l: PointerProperty(name="Eye L", type=bpy.types.Object, poll=_mesh_poll,
        description="LEFT eye mesh (rigid to the left eye bone, else the head)")
    skin_eye_r: PointerProperty(name="Eye R", type=bpy.types.Object, poll=_mesh_poll,
        description="RIGHT eye mesh (rigid to the right eye bone, else the head)")
    skin_teeth_up: PointerProperty(name="Teeth Up", type=bpy.types.Object, poll=_mesh_poll,
        description="UPPER teeth mesh (rigid to the head bone)")
    skin_teeth_low: PointerProperty(name="Teeth Low", type=bpy.types.Object, poll=_mesh_poll,
        description="LOWER teeth mesh (rigid to the jaw bone, else the head)")
    skin_hair: PointerProperty(name="Hair", type=bpy.types.Object, poll=_mesh_poll,
        description="Hair mesh - rigid-bound to the head bone at bind")
    skin_tongue: PointerProperty(name="Tongue", type=bpy.types.Object, poll=_mesh_poll,
        description="Tongue mesh (rigid to the jaw bone, else the head)")
    face_storm_full: BoolProperty(
        name="Full Storm Face", default=True,
        description="Build the COMPLETE Storm face system (538 bones: "
                    "ribbon eyelids + auto-blink, micro lips + zipper, "
                    "brow/cheek strips, teeth, tongue, jawline, lattices, "
                    "drivers) retargeted onto this character. Off = the "
                    "simple 42-control layout")
    face_lip_ctls: IntProperty(name="Lip Controls", default=2, min=1, max=4,
                               description="Extra lip controls PER SIDE per lip "
                                           "(between the center and the corner). "
                                           "Rebuild Face Base to apply")
    skin_brows: PointerProperty(name="Brows", type=bpy.types.Object, poll=_mesh_poll,
                                description="Eyebrow mesh (registered like FaceIt; used by the "
                                            "brow module and facial bind)")
    skin_lashes: PointerProperty(name="Eyelashes", type=bpy.types.Object, poll=_mesh_poll,
                                 description="Eyelashes mesh (registered like FaceIt; follows "
                                             "the eyelids)")
    # ---- top-level phases in THE recommended order: Rig -> Fit -> Animate
    # (rig the character first = exact joints, then dress her, then animate)
    ui_tab: EnumProperty(
        name="Phase", default='RIG',
        items=[('RIG', "Rig", "Rig the character FIRST - markers, metarig, "
                "skinning and Rigify samples",
                'OUTLINER_OB_ARMATURE', 0),
               ('ANIM', "Animate", "Cloth dynamics, locomotion, poses and more",
                'PLAY', 2),
               ('CHAR', "Character", "Name, organize, check & fix - make the "
                "character link-ready for projects", 'USER', 3)])
    char_name: StringProperty(
        name="Character Name", default="",
        description="Used for the CH-/RIG-/GEO- names when organizing the "
                    "character for linking into projects")
    rig_sub: EnumProperty(
        name="Rig Section", default='BUILD',
        items=[('BUILD', "Build", "Markers, metarig and Rigify samples"),
               ('SKIN', "Skin", "Bind / skin the mesh to the rig")])
    ui_level: EnumProperty(
        name="Level", default='SIMPLE',
        items=[('SIMPLE', "Simple", "Show only the essential steps"),
               ('PRO', "Pro", "Show every tool: bone roll, align, display and "
                "advanced options")])
    rig_started: BoolProperty(
        name="Rig Started", default=False,
        description="Internal: becomes True after 'Let's Rig' so the Character/Parts "
        "choice appears")
    mode_chosen: BoolProperty(
        name="Mode Chosen", default=False,
        description="Internal: becomes True after the user picks Character or Parts, "
        "so the tabs and tools appear")
    rig_mode: EnumProperty(
        name="Rig Mode", default='CHARACTER',
        description="What are you rigging? Character = the humanoid marker/Smart "
        "workflow. Parts = rig a standalone accessory (skirt, cape, tail...) with no "
        "body markers",
        items=[('CHARACTER', "Character", "Full humanoid: place body markers, then generate"),
               ('PARTS', "Parts", "Standalone accessory (skirt / cloth / appendage) - no markers")])
    show_options: BoolProperty(name="Options", default=False,
        description="Collapse / expand rig options (spine/neck/clavicles/mirror/fingers)")
    roll_axis: EnumProperty(
        name="Roll To", default='GLOBAL_POS_Z',
        items=[('GLOBAL_POS_Z', "+Z up", ""), ('GLOBAL_NEG_Z', "-Z", ""),
               ('GLOBAL_POS_X', "+X", ""), ('GLOBAL_NEG_X', "-X", ""),
               ('GLOBAL_POS_Y', "+Y", ""), ('GLOBAL_NEG_Y', "-Y back", ""),
               ('VIEW', "View", ""), ('CURSOR', "Cursor", ""), ('ACTIVE', "Active Bone", "")])
    # ---- guided hands decision ----
    hands_decided: BoolProperty(default=False)
    want_hands: BoolProperty(default=False)
    # ---- alignment orientation for the X/Y/Z align buttons ----
    align_orient: EnumProperty(
        name="Align Orientation", default='NORMAL',
        items=[('GLOBAL', "World", "World X/Y/Z axes"),
               ('NORMAL', "Normal", "The finger's own direction (won't distort a tilted finger)"),
               ('BOX', "Box", "The selection's oriented bounding box")])


def register():
    bpy.utils.register_class(WeightFolder)
    bpy.utils.register_class(SmartRigProps)
    bpy.types.Scene.smartrig = PointerProperty(type=SmartRigProps)


def unregister():
    del bpy.types.Scene.smartrig
    bpy.utils.unregister_class(SmartRigProps)
    bpy.utils.unregister_class(WeightFolder)
