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
import shutil
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf, Usd
import math

# Coordinate transformation base constants
USD_ORIGIN_X = 1747.89742
USD_ORIGIN_Y = 0.0
USD_ORIGIN_Z = 36.5

# Collision avoidance constants
MIN_STONE_DISTANCE = 30.0
MIN_STONE_DISTANCE_SQ = MIN_STONE_DISTANCE * MIN_STONE_DISTANCE
MAX_PLACEMENT_ATTEMPTS = 50
STONE_RADIUS = 15.0  # Fixed physical radius in cm (~29 cm diameter curling stone)

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
        # rep.orchestrator.stop() # No longer needed with manual loop control

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

        # First randomised frame — ensures camera adjustments, lighting,
        # and stone positions have all been applied at least once before
        # we start recording.  The writer may save this as an image, but
        # it won't have a matching label so we discard it below.
        self._randomize_stones_per_frame()
        await omni.kit.app.get_app().next_update_async()
        await rep.orchestrator.step_async()

        # Detect how many images the warm-up produced so we can
        # start our label numbering at the same offset.
        warmup_images = glob.glob(os.path.join(self._images_dir, "rgb_*.png"))
        label_start = len(warmup_images)
        carb.log_info(f"Warm-up produced {label_start} image(s); labels will start at index {label_start}")

        # 3. Generation loop
        num_images = self._image_count_model.get_value_as_int()
        carb.log_info(f"Generating {num_images} images...")
        print(f"Generating {num_images} images...")
        for i in range(num_images):
            if not self._is_running:
                break

            # Randomize (Synchronous Python Update)
            self._randomize_stones_per_frame()

            # Flush USD attribute changes to the renderer so the capture
            # reflects the scene we just randomised (without this, the
            # renderer may still see the *previous* frame's state, causing
            # an off-by-one between images and labels).
            await omni.kit.app.get_app().next_update_async()

            # Step Replicator (Render one frame)
            await rep.orchestrator.step_async()

            # Write YOLO labels with an index that matches the writer's
            # file counter (offset by any warm-up frames).
            frame_id = label_start + i
            self._write_yolo_labels(frame_id)

            if DEBUG_LOGGING:
                print(f"Image Generated: rgb_{frame_id:04d}.png")
            carb.log_info(f"Generated rgb_{frame_id:04d}.png")

        # Remove warm-up images that have no matching labels
        for f in warmup_images:
            try:
                os.remove(f)
                carb.log_info(f"Removed warm-up image: {f}")
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

        # 3. Writer
        # Initialize the BasicWriter to save RGB images
        render_product = rep.create.render_product(camera, (RENDER_WIDTH, RENDER_HEIGHT))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=self._images_dir,
            rgb=True
        )
        writer.attach([render_product])

        # 4. Semantic labels on prims (documents class mapping, useful for annotators)
        self._setup_semantic_labels(stage)

        # Cache camera prim references to avoid stage.Traverse() every frame
        self._camera_xform_prim = None
        self._camera_camera_prim = None
        for prim in stage.Traverse():
            if prim.GetName().startswith("StoneCamera"):
                if prim.IsA(UsdGeom.Camera):
                    self._camera_camera_prim = prim
                else:
                    self._camera_xform_prim = prim

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

    def _setup_semantic_labels(self, stage):
        """Add semantic class labels to prims so the bbox annotator can identify them."""
        from pxr import Sdf

        def _add_label(prim, label):
            prim.CreateAttribute(
                "semantics:Semantics:params:semanticType",
                Sdf.ValueTypeNames.String
            ).Set("class")
            prim.CreateAttribute(
                "semantics:Semantics:params:semanticData",
                Sdf.ValueTypeNames.String
            ).Set(label)

        # Stone prims — classify by name
        stones_prim = stage.GetPrimAtPath("/World/Stones")
        if stones_prim.IsValid():
            for prim in stones_prim.GetChildren():
                if not prim.IsValid():
                    continue
                name_lower = prim.GetName().lower()
                if '_r' in name_lower or 'red' in name_lower:
                    _add_label(prim, "red_stone")
                elif '_y' in name_lower or 'yellow' in name_lower:
                    _add_label(prim, "yellow_stone")
                else:
                    _add_label(prim, "red_stone")
            carb.log_info(f"Labelled {len(list(stones_prim.GetChildren()))} stone prims")

        # Hog line mesh
        hog_prim = stage.GetPrimAtPath(HOG_LINE_PRIM_PATH)
        if hog_prim.IsValid():
            _add_label(hog_prim, "hog_line")
            carb.log_info(f"Labelled hog line: {HOG_LINE_PRIM_PATH}")
        else:
            carb.log_warn(f"Hog line prim not found: {HOG_LINE_PRIM_PATH}")

        # House rings mesh
        house_prim = stage.GetPrimAtPath(HOUSE_RINGS_PRIM_PATH)
        if house_prim.IsValid():
            _add_label(house_prim, "house")
            carb.log_info(f"Labelled house rings: {HOUSE_RINGS_PRIM_PATH}")
        else:
            carb.log_warn(f"House rings prim not found: {HOUSE_RINGS_PRIM_PATH}")

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
                    from pxr import Sdf
                    refinement_attr = prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
                refinement_attr.Set(1)
                count += 1

        carb.log_info(f"Applied Catmull-Clark subdivision (refinement=1) to {count} stone meshes")
        print(f"Applied Catmull-Clark subdivision (refinement=1) to {count} stone meshes")

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

        stones_prim = stage.GetPrimAtPath("/World/Stones")
        if not stones_prim.IsValid():
            carb.log_warn("No stones found at /World/Stones")
            return

        # Get all valid stone prims
        all_stones = [prim for prim in stones_prim.GetChildren() if prim.IsValid()]
        if not all_stones:
            carb.log_error("No stones found!")
            return

        if DEBUG_LOGGING:
            carb.log_info(f"Frame {self._frame_counter}: Randomizing {len(all_stones)} stones...")

        # 1. Random Count [0, 16] (or max available)

        # 1. Random Count [0, 16] (or max available)
        max_limit = min(len(all_stones), 16)
        count = random.randint(0, max_limit)

        # 2. Select 'count' stones to be active
        selected_stones = set(random.sample(all_stones, count))

        # 3. Apply Visibility & Position with Collision Avoidance
        placed_positions = []  # Track placed stone positions for collision check

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

            # Store for label projection — this is the exact matrix used
            # to position the camera, so labels will match the render.
            self._view_matrix = view_mtx

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

    def _write_yolo_labels(self, frame_index):
        """Write YOLO format label file for the current frame.
        Uses UsdGeom.BBoxCache for actual prim geometry bounds and the stored
        view matrix from _randomize_stones_per_frame for camera projection.
        Format: class_id center_x center_y width height (all normalized 0-1)"""
        label_path = os.path.join(self._labels_dir, f"rgb_{frame_index:04d}.txt")

        stage = omni.usd.get_context().get_stage()
        if not stage:
            with open(label_path, 'w') as f:
                f.write('')
            return

        camera_prim = self._camera_camera_prim
        if not camera_prim or not camera_prim.IsValid():
            carb.log_error("Cannot write YOLO labels: camera not found")
            return

        time_code = Usd.TimeCode.Default()

        # --- View matrix: use the exact matrix we computed to position the camera ---
        # Reading back from USD via XformCache can return stale data if
        # step_async() hasn't flushed yet, or post-step Replicator graph
        # evaluation overwrites the transform.
        if not hasattr(self, '_view_matrix') or self._view_matrix is None:
            carb.log_error("Cannot write YOLO labels: no view matrix stored")
            return
        view_matrix = self._view_matrix

        # --- Camera intrinsics ---
        usd_camera = UsdGeom.Camera(camera_prim)
        focal_length = usd_camera.GetFocalLengthAttr().Get()
        h_aperture = usd_camera.GetHorizontalApertureAttr().Get()

        if not focal_length or not h_aperture:
            carb.log_error("Cannot write YOLO labels: missing camera intrinsics")
            return

        # Effective v_aperture: the renderer overrides the USD attribute to
        # match the render product's aspect ratio.
        v_aperture = h_aperture * (RENDER_HEIGHT / RENDER_WIDTH)
        half_h = h_aperture * 0.5
        half_v = v_aperture * 0.5

        # --- BBoxCache for actual mesh geometry ---
        bbox_cache = UsdGeom.BBoxCache(time_code, ['default', 'render'])

        lines = []

        def _project_prim_bbox(prim, class_id):
            """Compute world-space bbox for a prim, project 8 corners to screen
            space, and append a YOLO label line."""
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility(time_code) == UsdGeom.Tokens.invisible:
                return
            world_bbox = bbox_cache.ComputeWorldBound(prim)
            aligned = world_bbox.ComputeAlignedRange()
            if aligned.IsEmpty():
                return

            bmin = aligned.GetMin()
            bmax = aligned.GetMax()

            # Project all 8 corners of the 3D AABB to normalized screen coords
            sxs, sys_ = [], []
            for xi in (0, 1):
                for yi in (0, 1):
                    for zi in (0, 1):
                        wx = bmax[0] if xi else bmin[0]
                        wy = bmax[1] if yi else bmin[1]
                        wz = bmax[2] if zi else bmin[2]

                        cam_pt = view_matrix.Transform(Gf.Vec3d(wx, wy, wz))

                        if cam_pt[2] >= 0:  # behind camera (looks down -Z)
                            continue

                        depth = -cam_pt[2]
                        ndc_x = (focal_length * cam_pt[0]) / (half_h * depth)
                        ndc_y = (focal_length * cam_pt[1]) / (half_v * depth)

                        sxs.append((ndc_x + 1.0) * 0.5)
                        sys_.append((1.0 - ndc_y) * 0.5)

            if not sxs:
                return

            # Screen-space AABB, clipped to [0, 1]
            x_min = max(0.0, min(sxs))
            y_min = max(0.0, min(sys_))
            x_max = min(1.0, max(sxs))
            y_max = min(1.0, max(sys_))

            if x_max <= x_min or y_max <= y_min:
                return

            w_norm = x_max - x_min
            h_norm = y_max - y_min
            cx_norm = (x_min + x_max) / 2.0
            cy_norm = (y_min + y_max) / 2.0

            if w_norm < 0.005 or h_norm < 0.005:
                return

            lines.append(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        # --- Stones (fixed-radius projection for consistent bounding boxes) ---
        # All curling stones are the same physical size, so use a fixed radius
        # instead of BBoxCache which varies with mesh detail and rotation.
        def _project_stone(prim, class_id):
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility(time_code) == UsdGeom.Tokens.invisible:
                return

            # Read the stone's world position from its translate op
            xformable = UsdGeom.Xformable(prim)
            world_pos = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    world_pos = op.Get()
                    break
            if world_pos is None:
                return

            # Project 8 points around the stone's rim plus center.
            # This produces a tight bbox even when the camera is tilted,
            # where a circular stone projects as an ellipse.
            wx, wy, wz = world_pos[0], world_pos[1], world_pos[2]
            sxs, sys_ = [], []
            for k in range(8):
                angle = k * math.pi / 4.0
                rx = wx + STONE_RADIUS * math.cos(angle)
                ry = wy + STONE_RADIUS * math.sin(angle)
                cam_pt = view_matrix.Transform(Gf.Vec3d(rx, ry, wz))
                if cam_pt[2] >= 0:
                    continue
                d = -cam_pt[2]
                sxs.append(((focal_length * cam_pt[0]) / (half_h * d) + 1.0) * 0.5)
                sys_.append((1.0 - (focal_length * cam_pt[1]) / (half_v * d)) * 0.5)

            if len(sxs) < 2:
                return

            x_min = max(0.0, min(sxs))
            y_min = max(0.0, min(sys_))
            x_max = min(1.0, max(sxs))
            y_max = min(1.0, max(sys_))

            if x_max <= x_min or y_max <= y_min:
                return

            w_norm = x_max - x_min
            h_norm = y_max - y_min
            cx_norm = (x_min + x_max) / 2.0
            cy_norm = (y_min + y_max) / 2.0

            if w_norm < 0.005 or h_norm < 0.005:
                return

            lines.append(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        stones_prim = stage.GetPrimAtPath("/World/Stones")
        if stones_prim.IsValid():
            for prim in stones_prim.GetChildren():
                if not prim.IsValid():
                    continue
                name_lower = prim.GetName().lower()
                cls = CLASS_YELLOW if ('_y' in name_lower or 'yellow' in name_lower) else CLASS_RED
                _project_stone(prim, cls)

        # --- Hog Line ---
        hog_prim = stage.GetPrimAtPath(HOG_LINE_PRIM_PATH)
        if hog_prim.IsValid():
            _project_prim_bbox(hog_prim, CLASS_HOG)

        # --- House Rings ---
        house_prim = stage.GetPrimAtPath(HOUSE_RINGS_PRIM_PATH)
        if house_prim.IsValid():
            _project_prim_bbox(house_prim, CLASS_HOUSE)

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
