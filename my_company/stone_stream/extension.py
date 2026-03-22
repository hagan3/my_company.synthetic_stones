# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni.ext
import omni.usd
import omni.ui as ui
import omni.timeline
import asyncio
import carb
import glob
import os
import random
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf, Usd, Sdf
import math

# Coordinate transformation base constants
USD_ORIGIN_X = 1747.89742
USD_ORIGIN_Y = 0.0
USD_ORIGIN_Z = 36.5

# Collision avoidance constants
MIN_STONE_DISTANCE = 30.0
MIN_STONE_DISTANCE_SQ = MIN_STONE_DISTANCE * MIN_STONE_DISTANCE
MAX_PLACEMENT_ATTEMPTS = 50
# Render product resolution (must match render_product creation in _setup_graph)
RENDER_WIDTH = 640
RENDER_HEIGHT = 640

# Performance: set True to enable per-stone/per-frame print() logging
DEBUG_LOGGING = False

# Prim paths for scene objects that need bounding box labels
HOG_LINE_PRIM_PATH = "/World/hd_rings/hog_line/fp_logo_plane"
HOUSE_RINGS_PRIM_PATH = "/World/hd_rings/gsoc_rings/fp_logo_plane"

# Class IDs (matching generate_synthetic_data.py)
CLASS_RED = 0
CLASS_YELLOW = 1
CLASS_HOG = 2
CLASS_HOUSE = 3

# Approximate world-space radius of a curling stone (for bbox estimation)
STONE_RADIUS = 14.0

class stoneUpdateExtension(omni.ext.IExt):
    def on_startup(self, _ext_id):
        """This is called every time the extension is activated."""
        carb.log_info("[my_company.stone_stream] Extension startup")

        self._window = None
        self._build_ui()

    def _build_ui(self):
        self._window = ui.Window("Synthetic Stones", width=300, height=150)
        with self._window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Synthetic Stone Generator", alignment=ui.Alignment.CENTER)
                with ui.HStack(height=22):
                    ui.Label("Number of Images:", width=120)
                    self._image_count_model = ui.SimpleIntModel(100)
                    ui.IntDrag(model=self._image_count_model, min=1, max=10000, step=1)
                ui.Button("Start Generation", clicked_fn=self._on_start_clicked)
                ui.Button("Stop Generation", clicked_fn=self._on_stop_clicked)

    def _on_stop_clicked(self):
        self._is_running = False
        carb.log_info("Stopping Synthetic Stone Generation...")
        print("Stopping Synthetic Stone Generation...")

    def _on_start_clicked(self):
        carb.log_info("synthetic stones started")
        print("synthetic stones started")

        if not omni.usd.get_context().get_stage():
            carb.log_warn("No USD stage found. Please open a stage first.")
            return

        # Calculate absolute output path
        self._output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "_output_stones"))
        self._images_dir = os.path.join(self._output_dir, "images")
        self._labels_dir = os.path.join(self._output_dir, "labels")
        print(f"Output Directory: {self._output_dir}")
        carb.log_info(f"Output Directory: {self._output_dir}")

        # Ensure subdirectories exist
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._labels_dir, exist_ok=True)

        # Clear previous output data
        for d in [self._images_dir, self._labels_dir]:
            if os.path.exists(d):
                for f in glob.glob(os.path.join(d, "*")):
                    os.remove(f)
        carb.log_info(f"Cleared previous output from {self._output_dir}")

        self._frame_counter = 0
        self._is_running = True

        asyncio.ensure_future(self._generate_async())

    def on_shutdown(self):
        """This is called every time the extension is deactivated."""
        carb.log_info("[my_company.stone_stream] Extension shutdown")
        if self._window:
            self._window.destroy()
            self._window = None

    async def _generate_async(self):
        # 1. Setup Graph (Camera, Writer)
        self._setup_graph()

        # 2. Warm-up: prime the render pipeline AND run one full
        #    randomize-then-render cycle so the camera/scene are fully
        #    initialised.  Images produced here are discarded (no labels).
        await rep.orchestrator.step_async()
        await omni.kit.app.get_app().next_update_async()

        self._randomize_stones_per_frame()
        await omni.kit.app.get_app().next_update_async()
        await rep.orchestrator.step_async()

        # Detect how many images the warm-up produced so we can
        # start our label numbering at the same offset.
        warmup_images = glob.glob(os.path.join(self._images_dir, "rgb_*.png"))
        label_start = len(warmup_images)
        carb.log_info(f"Warm-up produced {label_start} image(s)")

        # 3. Generation loop
        #
        # The Replicator render pipeline has a 1-frame delay: step_async()
        # at time T writes the image that was *rendered* during step T-1.
        # To keep labels in sync with images we capture each frame's label
        # data immediately, then write it one step later — when the
        # matching image is actually flushed to disk.
        num_images = self._image_count_model.get_value_as_int()
        carb.log_info(f"Generating {num_images} images...")
        print(f"Generating {num_images} images...")

        pending_label = None  # (visible_objects, cam_params) awaiting image flush

        for i in range(num_images):
            if not self._is_running:
                break

            self._randomize_stones_per_frame()
            await omni.kit.app.get_app().next_update_async()

            # Capture label data for this scene before stepping
            current_label = (
                list(self._frame_visible_objects),
                self._get_camera_params(),
            )

            await rep.orchestrator.step_async()

            # step just saved the PREVIOUS scene's render to disk.
            # Write the matching label.
            if pending_label is not None:
                frame_id = label_start + i
                prev_objects, prev_cam = pending_label
                self._write_yolo_labels(frame_id, prev_objects, prev_cam)
                if DEBUG_LOGGING:
                    print(f"Image Generated: rgb_{frame_id:04d}.png")
                carb.log_info(f"Generated rgb_{frame_id:04d}.png")

            pending_label = current_label

        # One final step to flush the last frame's render to disk
        if pending_label is not None and self._is_running:
            await rep.orchestrator.step_async()
            frame_id = label_start + num_images
            prev_objects, prev_cam = pending_label
            self._write_yolo_labels(frame_id, prev_objects, prev_cam)
            carb.log_info(f"Generated rgb_{frame_id:04d}.png")

        # Remove all images that have no matching label file
        labeled_bases = set()
        for lf in glob.glob(os.path.join(self._labels_dir, "rgb_*.txt")):
            labeled_bases.add(os.path.splitext(os.path.basename(lf))[0])
        for img in glob.glob(os.path.join(self._images_dir, "rgb_*.png")):
            if os.path.splitext(os.path.basename(img))[0] not in labeled_bases:
                try:
                    os.remove(img)
                    carb.log_info(f"Removed unlabeled image: {img}")
                except OSError:
                    pass

        carb.log_info("Generation Finished.")

    def _setup_graph(self):
        # Reset any existing Replicator graph by deleting the Prim
        stage = omni.usd.get_context().get_stage()
        if stage:
            replicator_prim = stage.GetPrimAtPath("/Replicator")
            if replicator_prim.IsValid():
                stage.RemovePrim("/Replicator")

        # 1. Camera Setup
        # Located 4 meters (400 units) above the USD origin
        # We create a camera using Replicator
        camera = None

        with rep.new_layer():
            camera = rep.create.camera(
                position=(USD_ORIGIN_X, USD_ORIGIN_Y, USD_ORIGIN_Z + 600),
                look_at=(USD_ORIGIN_X, USD_ORIGIN_Y, USD_ORIGIN_Z),
                name="StoneCamera"
            )

        if not camera:
            carb.log_error("Failed to create camera. Replicator layer might not have initialized.")
            return

        # 2. Lighting Setup for Domain Randomization
        # Remove and recreate lights to avoid stale/malformed prims from previous runs
        for light_path in ["/World/DomeLight", "/World/KeyLight"]:
            old_prim = stage.GetPrimAtPath(light_path)
            if old_prim.IsValid():
                stage.RemovePrim(light_path)
                carb.log_info(f"Removed stale light prim: {light_path}")

        # Create dome light for ambient illumination
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.GetIntensityAttr().Set(500.0)
        dome_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        carb.log_info("Created DomeLight at /World/DomeLight")

        # Create key light (sphere light) for directional lighting variation
        key_light = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
        key_light.GetIntensityAttr().Set(30000.0)
        key_light.GetRadiusAttr().Set(50.0)
        key_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        xform = UsdGeom.Xformable(key_light.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(USD_ORIGIN_X - 200, USD_ORIGIN_Y + 100, USD_ORIGIN_Z + 500))
        carb.log_info("Created KeyLight at /World/KeyLight")

        # 3. Writer (RGB only — labels are computed via manual projection)
        render_product = rep.create.render_product(camera, (RENDER_WIDTH, RENDER_HEIGHT))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=self._images_dir,
            rgb=True
        )
        writer.attach([render_product])

        # Cache camera prim references to avoid stage.Traverse() every frame
        self._camera_xform_prim = None
        self._camera_camera_prim = None
        for prim in stage.Traverse():
            if prim.GetName().startswith("StoneCamera"):
                if prim.IsA(UsdGeom.Camera):
                    self._camera_camera_prim = prim
                else:
                    self._camera_xform_prim = prim

        # 4. Pre-compute world-space bounding boxes for static objects
        self._static_objects = []  # list of (class_id, bbox_min, bbox_max)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        hog_prim = stage.GetPrimAtPath(HOG_LINE_PRIM_PATH)
        if hog_prim.IsValid():
            bbox = bbox_cache.ComputeWorldBound(hog_prim)
            rng = bbox.ComputeAlignedRange()
            if not rng.IsEmpty():
                self._static_objects.append((CLASS_HOG, rng.GetMin(), rng.GetMax()))
                carb.log_info(f"Hog line bbox: {rng.GetMin()} -> {rng.GetMax()}")
        else:
            carb.log_warn(f"Hog line prim not found: {HOG_LINE_PRIM_PATH}")

        house_prim = stage.GetPrimAtPath(HOUSE_RINGS_PRIM_PATH)
        if house_prim.IsValid():
            bbox = bbox_cache.ComputeWorldBound(house_prim)
            rng = bbox.ComputeAlignedRange()
            if not rng.IsEmpty():
                self._static_objects.append((CLASS_HOUSE, rng.GetMin(), rng.GetMax()))
                carb.log_info(f"House rings bbox: {rng.GetMin()} -> {rng.GetMax()}")
        else:
            carb.log_warn(f"House rings prim not found: {HOUSE_RINGS_PRIM_PATH}")

        # RTX real-time render settings for quality
        settings = carb.settings.get_settings()
        settings.set_int("/rtx/post/dlss/execMode", 2)  # Quality Mode
        settings.set_float("/rtx/post/resample/fractionalResolution", 1.0)
        settings.set_int("/rtx/post/aa/op", 2)  # TAA Enabled
        settings.set_int("/rtx/materialDb/anisotropy", 16)  # Sharp textures at angles
        settings.set_int("/rtx/raytracing/subsurface/maxSamplePerFrame", 3)

        # Subdivision & Tessellation — ensure Catmull-Clark subdivision
        # actually runs in the RTX render product (not just the viewport).
        # The viewport's smooth appearance comes from the Storm renderer
        # honouring refinementLevel, but RTX needs these explicit settings.
        settings.set_bool("/rtx/hydra/subdivision/enabled", True)
        settings.set_int("/rtx/hydra/subdivision/refinementLevel", 1)
        settings.set_bool("/rtx/hydra/Tessellation/enabled", True)
        settings.set_int("/rtx/hydra/Tessellation/maxSubdivisionLevel", 2)
        settings.set_bool("/rtx/hydra/Tessellation/adaptiveTessellation", True)
        settings.set_bool("/rtx/hydra/faceCulling/enabled", False)

        # Ensure RTX Real-Time render mode (matches viewport)
        settings.set_string("/rtx/rendermode", "RaytracedLighting")

        # Set Catmull-Clark subdivision on all stone meshes for smoother rendering
        self._apply_subdivision_to_stones(stage)

        # Cache stone translate ops to avoid searching xform ops every frame
        self._stone_translate_ops = {}

        # Per-frame visible objects list (populated by _randomize_stones_per_frame)
        self._frame_visible_objects = []

        # Validate which stones can actually be controlled (visibility + position).
        # Locked / instanced prims are excluded from randomization.
        self._controllable_stones = self._validate_stones(stage)

    def _apply_subdivision_to_stones(self, stage):
        """Set Catmull-Clark subdivision scheme and refinement level 3 on all stone meshes."""
        stones_prim = stage.GetPrimAtPath("/World/Stones")
        if not stones_prim.IsValid():
            carb.log_warn("No stones found at /World/Stones for subdivision setup")
            return

        count = 0
        for prim in Usd.PrimRange(stones_prim):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                mesh.GetSubdivisionSchemeAttr().Set("catmullClark")
                refinement_attr = prim.GetAttribute("refinementLevel")
                if not refinement_attr or not refinement_attr.IsValid():
                    refinement_attr = prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
                refinement_attr.Set(1)
                count += 1

        carb.log_info(f"Applied Catmull-Clark subdivision (refinement=1) to {count} stone meshes")
        print(f"Applied Catmull-Clark subdivision (refinement=1) to {count} stone meshes")

    def _validate_stones(self, stage):
        """Check which stones can be controlled (visibility + transform).
        Excludes instance proxies and prims whose visibility or translation
        cannot actually be changed.  Returns list of controllable prims."""
        stones_prim = stage.GetPrimAtPath("/World/Stones")
        if not stones_prim.IsValid():
            carb.log_warn("No stones found at /World/Stones for validation")
            return []

        controllable = []
        all_children = [p for p in stones_prim.GetChildren() if p.IsValid()]

        for prim in all_children:
            name = prim.GetName()

            # Instance proxies are read-only projections of a prototype —
            # we cannot change their visibility or transform.
            if prim.IsInstanceProxy() or prim.IsInstance():
                carb.log_warn(f"Stone '{name}' is an instance/proxy — excluding")
                continue

            # --- Visibility test ---
            imageable = UsdGeom.Imageable(prim)
            vis_attr = imageable.GetVisibilityAttr()
            if not vis_attr or not vis_attr.IsValid():
                carb.log_warn(f"Stone '{name}' has no visibility attr — excluding")
                continue

            original_vis = vis_attr.Get()
            imageable.MakeInvisible()
            readback_vis = vis_attr.Get()
            # Restore original
            if original_vis == UsdGeom.Tokens.invisible:
                imageable.MakeInvisible()
            else:
                imageable.MakeVisible()

            if readback_vis != UsdGeom.Tokens.invisible:
                carb.log_warn(
                    f"Stone '{name}' visibility cannot be changed "
                    f"(wrote 'invisible', read back '{readback_vis}') — excluding"
                )
                continue

            # --- Translation test ---
            xform = UsdGeom.Xformable(prim)
            test_pos = Gf.Vec3d(99999.0, 99999.0, 99999.0)
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            if translate_op:
                old_val = translate_op.Get()
                translate_op.Set(test_pos)
                readback_pos = translate_op.Get()
                # Restore
                if old_val is not None:
                    translate_op.Set(old_val)
                else:
                    translate_op.Set(Gf.Vec3d(0, 0, 0))

                if readback_pos != test_pos:
                    carb.log_warn(
                        f"Stone '{name}' translate cannot be changed — excluding"
                    )
                    continue
            else:
                # No existing translate op — try to create one
                try:
                    new_op = xform.AddTranslateOp()
                    new_op.Set(test_pos)
                    readback_pos = new_op.Get()
                    new_op.Set(Gf.Vec3d(0, 0, 0))
                    if readback_pos != test_pos:
                        carb.log_warn(
                            f"Stone '{name}' new translate op not writable — excluding"
                        )
                        continue
                except Exception as e:
                    carb.log_warn(f"Stone '{name}' cannot add translate op ({e}) — excluding")
                    continue

            controllable.append(prim)

        excluded = len(all_children) - len(controllable)
        carb.log_info(
            f"Stone validation: {len(controllable)} controllable, "
            f"{excluded} excluded out of {len(all_children)} total"
        )
        if excluded > 0:
            excluded_prims = [p for p in all_children if p not in controllable]
            excluded_names = [p.GetName() for p in excluded_prims]
            carb.log_warn(f"Excluded stones: {excluded_names}")
            print(f"Excluded stones from randomization: {excluded_names}")
            # Remove uncontrollable stones from the stage so they don't
            # appear as unlabeled objects in the rendered training images.
            for prim in excluded_prims:
                prim_path = str(prim.GetPath())
                stage.RemovePrim(prim_path)
                carb.log_info(f"Removed uncontrollable stone from stage: {prim_path}")

        return controllable

    def _randomize_stones_per_frame(self):
        """
        Custom function mapped to Replicator randomizer.
        Selects 0-16 stones, makes them visible, randomizes their positions,
        and hides the rest.
        """
        self._frame_counter += 1
        if DEBUG_LOGGING:
            current_file = os.path.join(self._images_dir, f"rgb_{self._frame_counter:04d}.png")
            print(f"Generating Frame {self._frame_counter}: {current_file}")
            carb.log_info(f"Generating Frame {self._frame_counter}: {current_file}")

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        # Use only stones that passed the controllability check at setup
        all_stones = self._controllable_stones
        if not all_stones:
            return

        if DEBUG_LOGGING:
            carb.log_info(f"Frame {self._frame_counter}: Randomizing {len(all_stones)} stones...")

        # 1. Random Count [0, 16] (or max available)
        max_limit = min(len(all_stones), 16)
        count = random.randint(0, max_limit)

        # 2. Select 'count' stones to be active
        selected_stones = set(random.sample(all_stones, count))

        # 3. Apply Visibility & Position with Collision Avoidance
        placed_positions = []  # Track placed stone positions for collision check

        # Reset per-frame visible objects (stones only; static objects added in _write_yolo_labels)
        self._frame_visible_objects = []

        for prim in all_stones:
            imageable = UsdGeom.Imageable(prim)

            if prim in selected_stones:
                if DEBUG_LOGGING:
                    print(f"Stone Identified: {prim.GetName()}")
                    carb.log_info(f"Stone Identified: {prim.GetName()}")

                # Find a non-colliding position
                position = self._find_non_colliding_position(placed_positions)

                if position:
                    rand_x, rand_y = position
                    placed_positions.append((rand_x, rand_y))

                    # Make Visible
                    imageable.MakeVisible()

                    if DEBUG_LOGGING:
                        carb.log_info(f"Stone {prim.GetName()} -> Pos: ({rand_x:.2f}, {rand_y:.2f})")

                    # Set Translation (use cached op when available)
                    pos = Gf.Vec3d(rand_x, rand_y, USD_ORIGIN_Z)
                    prim_path = prim.GetPath()
                    if prim_path in self._stone_translate_ops:
                        self._stone_translate_ops[prim_path].Set(pos)
                    else:
                        xform = UsdGeom.Xformable(prim)
                        for op in xform.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                self._stone_translate_ops[prim_path] = op
                                op.Set(pos)
                                break
                        else:
                            op = xform.AddTranslateOp()
                            op.Set(pos)
                            self._stone_translate_ops[prim_path] = op

                    # Determine class from prim name
                    name_lower = prim.GetName().lower()
                    if '_y' in name_lower or 'yellow' in name_lower:
                        class_id = CLASS_YELLOW
                    else:
                        class_id = CLASS_RED
                    self._frame_visible_objects.append((class_id, rand_x, rand_y))
                else:
                    # Could not place without collision, hide this stone
                    carb.log_warn(f"Could not place stone {prim.GetName()} after {MAX_PLACEMENT_ATTEMPTS} attempts, hiding.")
                    imageable.MakeInvisible()

            else:
                # Make Invisible
                imageable.MakeInvisible()

        # 4. Camera Pose: fixed position, pan only (look-at target varies)
        # Real camera is static overhead; pans primarily up/down the sheet (X),
        # with slight left/right (Y) variation
        camera_xform_prim = self._camera_xform_prim
        camera_camera_prim = self._camera_camera_prim

        if not camera_xform_prim:
            carb.log_error(f"Frame {self._frame_counter}: Could not find camera 'StoneCamera'!")

        if camera_xform_prim:
            # Camera stays at its fixed overhead position
            cam_pos = Gf.Vec3d(USD_ORIGIN_X, USD_ORIGIN_Y, USD_ORIGIN_Z + 600)

            # Pan: vary look-at target primarily along the sheet (X axis)
            target_x = random.uniform(USD_ORIGIN_X - 500, USD_ORIGIN_X + 182)
            target_y = random.uniform(USD_ORIGIN_Y - 30, USD_ORIGIN_Y + 30)
            target_z = USD_ORIGIN_Z
            target_pos = Gf.Vec3d(target_x, target_y, target_z)

            if DEBUG_LOGGING:
                msg = f"Camera pan -> target: ({target_x:.0f}, {target_y:.0f})"
                print(msg)
                carb.log_info(msg)

            # Use horizontal up vector (-X = "up" in image) to avoid
            # gimbal flipping on a near-vertical overhead camera
            up_vec = Gf.Vec3d(-1, 0, 0)
            view_mtx = Gf.Matrix4d()
            view_mtx.SetLookAt(cam_pos, target_pos, up_vec)
            xform_mtx = view_mtx.GetInverse()

            xform = UsdGeom.Xformable(camera_xform_prim)
            self._set_transform_matrix(xform, xform_mtx)

        # 5. Curler Obfuscation Object (appears in ~50% of frames)
        curler_prim = stage.GetPrimAtPath("/World/curler")
        if curler_prim.IsValid():
            curler_imageable = UsdGeom.Imageable(curler_prim)
            show_curler = random.random() < 0.5
            if show_curler:
                curler_imageable.MakeVisible()
                curler_x = random.uniform(-200, 100)
                curler_y = random.uniform(-100, 100)
                curler_pos = Gf.Vec3d(curler_x, curler_y, USD_ORIGIN_Z)
                curler_xform = UsdGeom.Xformable(curler_prim)
                self._set_translation(curler_xform, curler_pos)
                if DEBUG_LOGGING:
                    carb.log_info(f"Curler visible -> Pos: ({curler_x:.2f}, {curler_y:.2f})")
            else:
                curler_imageable.MakeInvisible()
        else:
            carb.log_warn("Curler prim not found at /World/curler")

        # 6. Lighting Domain Randomization
        self._randomize_lighting(stage)

        # 7. Camera Aperture / Focus Domain Randomization
        self._randomize_camera_intrinsics(camera_camera_prim)

    def _find_non_colliding_position(self, placed_positions):
        """Find a random position that doesn't collide with already-placed stones.
        Returns (x, y) tuple or None if no valid position found."""
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            rand_x = random.uniform(USD_ORIGIN_X - 640, USD_ORIGIN_X + 182)
            rand_y = random.uniform(USD_ORIGIN_Y - 200, USD_ORIGIN_Y + 200)

            # Check squared distance against all placed stones (avoids sqrt)
            is_valid = True
            for px, py in placed_positions:
                dist_sq = (rand_x - px) ** 2 + (rand_y - py) ** 2
                if dist_sq < MIN_STONE_DISTANCE_SQ:
                    is_valid = False
                    break

            if is_valid:
                return (rand_x, rand_y)

        return None

    def _project_to_screen(self, world_pt, cam_pos, cam_right, cam_up, cam_fwd,
                           focal_length, h_aperture):
        """Project a 3D world point to 2D screen pixel coordinates.
        Returns (px, py) or None if the point is behind the camera."""
        to_pt = world_pt - cam_pos
        depth = Gf.Dot(to_pt, cam_fwd)
        if depth <= 0:
            return None

        right = Gf.Dot(to_pt, cam_right)
        up = Gf.Dot(to_pt, cam_up)

        # The renderer uses horizontal-fit mode: horizontal FOV is set by
        # h_aperture, and vertical FOV is derived from the render aspect
        # ratio — NOT from the camera's v_aperture.  For a square render
        # (640x640) the effective vertical aperture equals h_aperture.
        effective_v_aperture = h_aperture * (float(RENDER_HEIGHT) / float(RENDER_WIDTH))

        # Pinhole projection: NDC in [-1, 1]
        ndc_x = (right * 2.0 * focal_length) / (depth * h_aperture)
        ndc_y = (up * 2.0 * focal_length) / (depth * effective_v_aperture)

        # Convert to pixel coordinates (Y flipped for image space)
        px = (ndc_x + 1.0) * 0.5 * RENDER_WIDTH
        py = (1.0 - ndc_y) * 0.5 * RENDER_HEIGHT
        return px, py

    def _get_camera_params(self):
        """Extract camera axes and lens parameters for projection."""
        xform_cache = UsdGeom.XformCache()
        cam_world = xform_cache.GetLocalToWorldTransform(self._camera_camera_prim)
        cam_pos = Gf.Vec3d(cam_world[3][0], cam_world[3][1], cam_world[3][2])

        # Camera axes from the world transform (USD camera looks along -Z)
        cam_right = Gf.Vec3d(cam_world[0][0], cam_world[0][1], cam_world[0][2]).GetNormalized()
        cam_up = Gf.Vec3d(cam_world[1][0], cam_world[1][1], cam_world[1][2]).GetNormalized()
        cam_fwd = -Gf.Vec3d(cam_world[2][0], cam_world[2][1], cam_world[2][2]).GetNormalized()

        usd_cam = UsdGeom.Camera(self._camera_camera_prim)
        focal_length = usd_cam.GetFocalLengthAttr().Get()
        h_aperture = usd_cam.GetHorizontalApertureAttr().Get()

        return cam_pos, cam_right, cam_up, cam_fwd, focal_length, h_aperture

    def _project_bbox_to_yolo(self, class_id, bbox_min_3d, bbox_max_3d, cam_params):
        """Project a 3D axis-aligned bounding box to YOLO format.
        Returns a YOLO line string or None if not visible."""
        cam_pos, cam_right, cam_up, cam_fwd, fl, ha = cam_params

        # Project the 8 corners of the 3D AABB
        corners = [
            Gf.Vec3d(bbox_min_3d[0], bbox_min_3d[1], bbox_min_3d[2]),
            Gf.Vec3d(bbox_max_3d[0], bbox_min_3d[1], bbox_min_3d[2]),
            Gf.Vec3d(bbox_min_3d[0], bbox_max_3d[1], bbox_min_3d[2]),
            Gf.Vec3d(bbox_max_3d[0], bbox_max_3d[1], bbox_min_3d[2]),
            Gf.Vec3d(bbox_min_3d[0], bbox_min_3d[1], bbox_max_3d[2]),
            Gf.Vec3d(bbox_max_3d[0], bbox_min_3d[1], bbox_max_3d[2]),
            Gf.Vec3d(bbox_min_3d[0], bbox_max_3d[1], bbox_max_3d[2]),
            Gf.Vec3d(bbox_max_3d[0], bbox_max_3d[1], bbox_max_3d[2]),
        ]

        screen_xs = []
        screen_ys = []
        for c in corners:
            result = self._project_to_screen(c, cam_pos, cam_right, cam_up, cam_fwd, fl, ha)
            if result is None:
                continue
            screen_xs.append(result[0])
            screen_ys.append(result[1])

        if len(screen_xs) < 2:
            return None

        # Screen-space AABB from projected corners
        x_min = max(0.0, min(screen_xs))
        y_min = max(0.0, min(screen_ys))
        x_max = min(float(RENDER_WIDTH), max(screen_xs))
        y_max = min(float(RENDER_HEIGHT), max(screen_ys))

        if x_max <= x_min or y_max <= y_min:
            return None

        # YOLO normalized format
        w_n = (x_max - x_min) / RENDER_WIDTH
        h_n = (y_max - y_min) / RENDER_HEIGHT

        # Skip tiny boxes (likely not meaningfully visible)
        if w_n < 0.005 or h_n < 0.005:
            return None

        cx_n = ((x_min + x_max) / 2.0) / RENDER_WIDTH
        cy_n = ((y_min + y_max) / 2.0) / RENDER_HEIGHT

        return f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"

    def _write_yolo_labels(self, frame_index, visible_objects, cam_params):
        """Write YOLO format labels by projecting known 3D object positions
        through the camera.
        Format: class_id center_x center_y width height (all normalized 0-1)"""
        label_path = os.path.join(self._labels_dir, f"rgb_{frame_index:04d}.txt")

        lines = []

        # 1. Stones (positions tracked during randomization)
        for class_id, wx, wy in visible_objects:
            bbox_min = Gf.Vec3d(wx - STONE_RADIUS, wy - STONE_RADIUS, USD_ORIGIN_Z)
            bbox_max = Gf.Vec3d(wx + STONE_RADIUS, wy + STONE_RADIUS, USD_ORIGIN_Z + STONE_RADIUS)
            line = self._project_bbox_to_yolo(class_id, bbox_min, bbox_max, cam_params)
            if line:
                lines.append(line)

        # 2. Static objects (hog line, house rings) — always in the scene
        for class_id, bbox_min, bbox_max in self._static_objects:
            line = self._project_bbox_to_yolo(class_id, bbox_min, bbox_max, cam_params)
            if line:
                lines.append(line)

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

        carb.log_info(f"YOLO Labels: {label_path} ({len(lines)} objects)")

    def _randomize_lighting(self, stage):
        """Randomize lighting properties each frame for domain randomization.
        Varies: dome light intensity, key light intensity/color temp/position."""

        # --- Dome Light: ambient intensity variation ---
        dome_prim = stage.GetPrimAtPath("/World/DomeLight")
        if dome_prim.IsValid():
            dome_light = UsdLux.DomeLight(dome_prim)
            # Intensity range: dim (200) to bright (1500)
            dome_intensity = random.uniform(200.0, 1500.0)
            dome_light.GetIntensityAttr().Set(dome_intensity)

            # Slight color temperature shift (warm to cool white)
            # Map color temp to approximate RGB: 4000K(warm) -> 8000K(cool)
            color_temp = random.uniform(4000, 8000)
            # Simplified blackbody approx: warm = more red, cool = more blue
            temp_norm = (color_temp - 4000) / 4000.0  # 0 = warm, 1 = cool
            r = 1.0 - temp_norm * 0.15  # Slight red reduction as temp increases
            g = 1.0 - abs(temp_norm - 0.5) * 0.05  # Green stays relatively constant
            b = 0.85 + temp_norm * 0.15  # Blue increases with temp
            dome_light.GetColorAttr().Set(Gf.Vec3f(r, g, b))

            if DEBUG_LOGGING:
                carb.log_info(f"DomeLight -> intensity: {dome_intensity:.0f}, colorTemp: {color_temp:.0f}K")

        # --- Key Light: intensity, position, color variation ---
        key_prim = stage.GetPrimAtPath("/World/KeyLight")
        if key_prim.IsValid():
            key_light = UsdLux.SphereLight(key_prim)

            # Intensity: low (10000) to high (80000)
            key_intensity = random.uniform(10000.0, 80000.0)
            key_light.GetIntensityAttr().Set(key_intensity)

            # Slight color variation for the key light
            key_r = random.uniform(0.9, 1.0)
            key_g = random.uniform(0.9, 1.0)
            key_b = random.uniform(0.85, 1.0)
            key_light.GetColorAttr().Set(Gf.Vec3f(key_r, key_g, key_b))

            # Randomize key light position (orbit around the scene)
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(200, 500)
            key_x = USD_ORIGIN_X - 300 + radius * math.cos(angle)
            key_y = USD_ORIGIN_Y + radius * math.sin(angle)
            key_z = USD_ORIGIN_Z + random.uniform(300, 700)

            key_pos = Gf.Vec3d(key_x, key_y, key_z)
            xform = UsdGeom.Xformable(key_prim)
            self._set_translation(xform, key_pos)

            if DEBUG_LOGGING:
                carb.log_info(f"KeyLight -> intensity: {key_intensity:.0f}, pos: ({key_x:.0f}, {key_y:.0f}, {key_z:.0f})")

    def _randomize_camera_intrinsics(self, camera_prim):
        """Randomize camera lens properties each frame for domain randomization.
        Varies: focal length, f-stop (aperture), focus distance."""
        if not camera_prim or not camera_prim.IsValid():
            return

        usd_camera = UsdGeom.Camera(camera_prim)

        # --- Focal Length ---
        # Range: 18mm (wide) to 50mm (normal) — controls field of view
        # Wider FOV captures more context, narrower mimics telephoto
        focal_length = random.uniform(18.0, 50.0)
        usd_camera.GetFocalLengthAttr().Set(focal_length)

        # --- F-Stop (Aperture) ---
        # Range: f/1.8 (shallow DOF, bokeh) to f/22 (deep DOF, everything sharp)
        # Lower f-stop = more background blur (realistic broadcast look)
        # Higher f-stop = sharper overall (overhead camera look)
        fstop = random.uniform(1.8, 22.0)
        usd_camera.GetFStopAttr().Set(fstop)

        # --- Focus Distance ---
        # Should roughly match camera-to-subject distance (300-800 range)
        # Vary to simulate auto-focus hunting and different focal planes
        focus_distance = random.uniform(250.0, 850.0)
        usd_camera.GetFocusDistanceAttr().Set(focus_distance)

        if DEBUG_LOGGING:
            carb.log_info(
                f"Camera Intrinsics -> focalLength: {focal_length:.1f}mm, "
                f"fStop: f/{fstop:.1f}, focusDist: {focus_distance:.0f}"
            )

    def _set_transform_matrix(self, xform, matrix):
        """Helper to set transform matrix on a USD prim"""
        # Clear existing ops to avoid conflict or accumulation if complex
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(matrix)

    def _set_translation(self, xform, pos):
        """Helper to set translation on a USD prim"""
        # Check for existing translate op
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(pos)
                return

        # If not found, add it
        xform.AddTranslateOp().Set(pos)
