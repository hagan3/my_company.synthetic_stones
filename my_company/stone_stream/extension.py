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
STONE_RADIUS = 15.0  # Approximate stone radius in cm

# Performance: set True to enable per-stone/per-frame print() logging
DEBUG_LOGGING = False

# House rings and hog line world-space offsets (relative to USD_ORIGIN_X/Y)
HOUSE_CENTER_OFFSET_X = 0.0    # House rings centered at (0,0) offset
HOUSE_CENTER_OFFSET_Y = 0.0
HOUSE_ASSET_RADIUS = 183.0     # Full house asset extent (includes dark branding area)
HOUSE_RINGS_RADIUS = 91.5     # 12-ft scoring rings only (~50% of asset, excludes branding)

HOG_LINE_OFFSET_X = -640.0     # Hog line centered at (-640, 0) offset
HOG_LINE_OFFSET_Y = 0.0
HOG_LINE_HALF_WIDTH = 183.0    # Approximate half-width (same as house width)
HOG_LINE_HALF_HEIGHT = 5.0     # Hog line is a thin strip

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
        print(f"Output Directory: {self._output_dir}")
        carb.log_info(f"Output Directory: {self._output_dir}")

        # Clear previous output data
        if os.path.exists(self._output_dir):
            for f in glob.glob(os.path.join(self._output_dir, "*.png")) + glob.glob(os.path.join(self._output_dir, "*.txt")):
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

        # 2. Loop
        num_images = self._image_count_model.get_value_as_int()
        carb.log_info(f"Generating {num_images} images...")
        print(f"Generating {num_images} images...")
        for i in range(num_images):
            if not self._is_running:
                break

            # Randomize (Synchronous Python Update)
            self._randomize_stones_per_frame()

            # Step Replicator (Render one frame)
            # This pushes the current USD state to the writer(s)
            await rep.orchestrator.step_async()

            # Write YOLO ground truth labels for this frame
            self._write_yolo_labels(i)

            if DEBUG_LOGGING:
                print(f"Image Generated: rgb_{i:04d}.png")
            carb.log_info(f"Generated rgb_{i:04d}.png")

            # Yield to UI loop every 10 frames to keep interface responsive
            if i % 10 == 0:
                await omni.kit.app.get_app().next_update_async()

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
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=self._output_dir,
            rgb=True
        )
        writer.attach([rep.create.render_product(camera, (640, 640))])

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
        settings.set_int("/rtx/hydra/Tessellation/maxSubdivisionLevel", 4)  # GPU subdivides triangles up to 4 times
        settings.set_bool("/rtx/hydra/Tessellation/adaptiveTessellation", False)  # Force high detail everywhere

        # Set Catmull-Clark subdivision on all stone meshes for smoother rendering
        self._apply_subdivision_to_stones(stage)

        # Cache stone translate ops to avoid searching xform ops every frame
        self._stone_translate_ops = {}

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
                refinement_attr.Set(3)
                count += 1

        carb.log_info(f"Applied Catmull-Clark subdivision (refinement=3) to {count} stone meshes")
        print(f"Applied Catmull-Clark subdivision (refinement=3) to {count} stone meshes")

    def _randomize_stones_per_frame(self):
        """
        Custom function mapped to Replicator randomizer.
        Selects 0-16 stones, makes them visible, randomizes their positions,
        and hides the rest.
        """
        self._frame_counter += 1
        if DEBUG_LOGGING:
            current_file = os.path.join(self._output_dir, f"rgb_{self._frame_counter:04d}.png")
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
        self._current_frame_stones = []  # Track stone data for YOLO labels

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

                    # Store for YOLO label generation
                    self._current_frame_stones.append({
                        'name': prim.GetName(),
                        'x': rand_x,
                        'y': rand_y,
                        'z': USD_ORIGIN_Z
                    })

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
        Format: class_id center_x center_y width height (all normalized 0-1)
        Class 0 = red stone, Class 1 = yellow stone, Class 2 = hog line, Class 3 = house rings"""
        label_path = os.path.join(self._output_dir, f"rgb_{frame_index:04d}.txt")

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        camera_prim = self._camera_camera_prim

        if not camera_prim:
            carb.log_error("Cannot write YOLO labels: camera not found")
            return

        # Get camera frustum for 3D-to-2D projection
        usd_camera = UsdGeom.Camera(camera_prim)
        gf_camera = usd_camera.GetCamera(Usd.TimeCode.Default())
        frustum = gf_camera.frustum

        view_matrix = frustum.ComputeViewMatrix()
        proj_matrix = frustum.ComputeProjectionMatrix()

        lines = []

        # --- Helper: project a 3D point to normalized image coords ---
        def project_to_norm(world_pt):
            view_pt = view_matrix.Transform(world_pt)
            ndc_pt = proj_matrix.Transform(view_pt)
            cx = (ndc_pt[0] + 1.0) / 2.0
            cy = 1.0 - (ndc_pt[1] + 1.0) / 2.0
            return cx, cy

        # --- Helper: write a bbox label from center + half-extents ---
        # Projects all 4 corners for correct perspective (angled views make ellipses)
        # and clips the bbox to frame bounds when partially off-screen
        def write_bbox_label(class_id, cx_world, cy_world, cz_world, half_w, half_h):
            corners = [
                Gf.Vec3d(cx_world - half_w, cy_world - half_h, cz_world),
                Gf.Vec3d(cx_world + half_w, cy_world - half_h, cz_world),
                Gf.Vec3d(cx_world + half_w, cy_world + half_h, cz_world),
                Gf.Vec3d(cx_world - half_w, cy_world + half_h, cz_world),
            ]

            xs, ys = [], []
            for corner in corners:
                nx, ny = project_to_norm(corner)
                xs.append(nx)
                ys.append(ny)

            # Unclipped bbox extents
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Clip to frame bounds [0, 1]
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            x_max = min(1.0, x_max)
            y_max = min(1.0, y_max)

            # Skip if fully outside frame or degenerate
            if x_max <= x_min or y_max <= y_min:
                return

            # YOLO format: center_x, center_y, width, height (all normalized)
            w_norm = x_max - x_min
            h_norm = y_max - y_min
            cx_norm = (x_min + x_max) / 2.0
            cy_norm = (y_min + y_max) / 2.0

            lines.append(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        # --- House Rings ---
        house_x = USD_ORIGIN_X + HOUSE_CENTER_OFFSET_X
        house_y = USD_ORIGIN_Y + HOUSE_CENTER_OFFSET_Y
        write_bbox_label(CLASS_HOUSE, house_x, house_y, USD_ORIGIN_Z, HOUSE_RINGS_RADIUS, HOUSE_RINGS_RADIUS)

        # --- Hog Line ---
        hog_x = USD_ORIGIN_X + HOG_LINE_OFFSET_X
        hog_y = USD_ORIGIN_Y + HOG_LINE_OFFSET_Y
        write_bbox_label(CLASS_HOG, hog_x, hog_y, USD_ORIGIN_Z, HOG_LINE_HALF_WIDTH, HOG_LINE_HALF_HEIGHT)

        # --- Stones ---
        if hasattr(self, '_current_frame_stones') and self._current_frame_stones:
            for stone_data in self._current_frame_stones:
                name = stone_data['name']
                sx, sy, sz = stone_data['x'], stone_data['y'], stone_data['z']

                name_lower = name.lower()
                if '_r' in name_lower or 'red' in name_lower:
                    class_id = CLASS_RED
                elif '_y' in name_lower or 'yellow' in name_lower:
                    class_id = CLASS_YELLOW
                else:
                    class_id = CLASS_RED

                write_bbox_label(class_id, sx, sy, sz, STONE_RADIUS, STONE_RADIUS)

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
