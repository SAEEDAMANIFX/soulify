import bpy
from bpy.props import (PointerProperty, IntProperty, BoolProperty,
                       FloatProperty, StringProperty, EnumProperty)
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


def _fit_live_update(self, context):
    """Live-tune the Let's Fit garment (ease / smoothing / scale / height) with
    no rebuild - mandatory update= callback (LESSONS: sliders without callbacks
    read as 'broken' on a paused frame)."""
    try:
        from . import garment
        garment.live_fit_tune(context)
    except Exception as e:
        print("SmartRig fit live tune:", e)


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


class SmartRigProps(PropertyGroup):
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
    # ---- CHEST forces: fully SEPARATE from the skirt ----
    chest_gravity: FloatProperty(name="Gravity", default=0.0, min=0.0, max=3.0,
        update=_jiggle_force_update,
        description="Downward pull on the chest jiggle only")
    chest_wind: FloatProperty(name="Wind", default=0.0, min=0.0, max=3.0,
        update=_jiggle_force_update,
        description="Wind force on the chest jiggle only")
    chest_wind_dir: FloatProperty(name="Wind Dir", default=0.0, min=0.0, max=360.0,
        subtype='ANGLE', update=_jiggle_force_update,
        description="Chest wind direction (0 = front)")
    chest_wind_turb: FloatProperty(name="Gust", default=0.3, min=0.0, max=1.0,
        update=_jiggle_force_update,
        description="Chest wind gustiness / turbulence")
    chest_wind_speed: FloatProperty(name="Wind Speed", default=1.0, min=0.1, max=15.0,
        update=_jiggle_force_update,
        description="How fast the chest wind gusts change")
    skirt_jiggle_segments: IntProperty(name="Jiggle Segments", default=3, min=1, max=8,
        description="B-bone segments on the skirt deform bones: higher = smoother, "
        "more professional cloth wave (live; no rebuild). 1 = faceted/off",
        update=_skirt_jiggle_seg_update)
    chest_jiggle_segments: IntProperty(name="Jiggle Segments", default=3, min=1, max=8,
        description="B-bone segments on each breast: 1 = rigid bounce, 3+ = soft "
        "progressive 'jelly' wobble (tip bends more than the base)")
    chest_jiggle_amount: FloatProperty(name="Chest Jiggle", default=2.0, min=0.0, max=5.0,
        description="How much the chest bounces (higher = stronger)")
    chest_jiggle_stiffness: FloatProperty(name="Stiffness", default=0.45, min=0.02, max=1.0)
    chest_jiggle_damping: FloatProperty(name="Damping", default=0.30, min=0.05, max=0.99)
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
    # ---- top-level phases: the Fit + Rig + Animate pipeline ----
    ui_tab: EnumProperty(
        name="Phase", default='RIG',
        items=[('FIT', "Fit", "Fit clothing onto the character automatically",
                'MOD_CLOTH', 0),
               ('RIG', "Rig", "Markers, metarig, skinning and Rigify samples",
                'OUTLINER_OB_ARMATURE', 1),
               ('ANIM', "Animate", "Cloth dynamics, locomotion, poses and more",
                'PLAY', 2)])
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
    # ---- Let's Fit (automatic garment fitting) ----
    fit_started: BoolProperty(
        name="Let's Fit", default=False,
        description="Fit any 3D clothing (skirt, shirt, pants, thobe...) onto the "
        "character automatically")
    garment_object: PointerProperty(
        name="Garment", type=bpy.types.Object, poll=_mesh_poll,
        description="The clothing mesh to fit onto the character")
    fit_body_object: PointerProperty(
        name="Body", type=bpy.types.Object, poll=_mesh_poll,
        description="The character mesh to fit the clothing onto (auto-detected "
        "if empty)")
    garment_preserve: BoolProperty(
        name="Preserve Shape", default=True,
        description="Keep the garment's designed shape (pleats, folds, volume, "
        "thickness): penetrations are resolved by moving whole regions smoothly "
        "instead of projecting each vertex onto the body. Disable for a "
        "skin-tight shrinkwrap conform")
    garment_ease: FloatProperty(
        name="Ease", default=0.5, min=0.0, max=5.0, subtype='PERCENTAGE',
        update=_fit_live_update,
        description="Gap between the body and the garment, as % of body height "
        "(0 = skin-tight)")
    garment_smooth: IntProperty(
        name="Smoothing", default=8, min=0, max=40, update=_fit_live_update,
        description="Corrective-smooth iterations that relax the conformed areas "
        "while keeping the garment's own detail")
    garment_scale: FloatProperty(
        name="Scale", default=1.0, min=0.5, max=1.5, update=_fit_live_update,
        description="Fine-tune the auto-detected garment scale (about its anchor "
        "ring)")
    garment_height: FloatProperty(
        name="Height", default=0.0, min=-0.25, max=0.25, update=_fit_live_update,
        description="Slide the garment up/down the body, as a fraction of body "
        "height")
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
    bpy.utils.register_class(SmartRigProps)
    bpy.types.Scene.smartrig = PointerProperty(type=SmartRigProps)


def unregister():
    del bpy.types.Scene.smartrig
    bpy.utils.unregister_class(SmartRigProps)
