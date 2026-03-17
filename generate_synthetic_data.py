import cv2
import numpy as np
import os
import random
import carb


# --- CONFIGURATION ---
OUTPUT_DIR = "datasets/curling_panning"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

# Assets
BG_PATH = "scripts/assets/empty_ice.jpg"     # Standard texture
RED_STONE_PATH = "scripts/assets/red stone.png"
YEL_STONE_PATH = "scripts/assets/yellow stone.png"
HOG_LINE_PATH = "scripts/assets/hog_line.png"
HOUSE_PATH = "scripts/assets/house.png"           # The rings

NUM_IMAGES = 1000
VIEWPORT_SIZE = (640, 640)  # What the camera sees
WORLD_HEIGHT_BUFFER = 100   # Extra space above hog line

# Physical Dimensions
HOUSE_DIAMETER_FT = 12.0
STONE_DIAMETER_FT = 0.9186  # 28 cm
SHEET_WIDTH_FT = 14.0
HOG_LINE_DIST_FT = 21.0     # Center of house to center of hog line (approx)

# Class IDs
CLASSES = {"red": 0, "yellow": 1, "hog": 2, "house": 3}

def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

def load_asset(path, target_width_px=None):
    if not os.path.exists(path):
        # Create Dummy if missing
        dummy = np.zeros((100, 100, 4), dtype=np.uint8)
        dummy[:] = [random.randint(0,255) for _ in range(4)]
        img = dummy
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    if target_width_px is not None:
        target_width_px = int(target_width_px)
        scale = target_width_px / img.shape[1]
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    return img

def overlay_transparent(background, overlay, x, y):
    """
    Overlays sprite onto background.
    Returns modified background and the bounding box in (x1, y1, x2, y2) format.
    """
    bg_h, bg_w = background.shape[:2]
    h, w = overlay.shape[:2]

    # Quick boundary check
    if x >= bg_w or y >= bg_h or x + w < 0 or y + h < 0:
        return background, None

    # Calculate clipping coordinates
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)

    # Dimensions of the overlap area
    w_crop = x2 - x1
    h_crop = y2 - y1

    if w_crop <= 0 or h_crop <= 0: return background, None

    # Source coordinates (overlay)
    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + w_crop
    oy2 = oy1 + h_crop

    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    bg_crop = background[y1:y2, x1:x2]

    alpha = overlay_crop[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(3):
        bg_crop[:, :, c] = (alpha * overlay_crop[:, :, c] +
                            alpha_inv * bg_crop[:, :, c])

    background[y1:y2, x1:x2] = bg_crop

    return background, (x1, y1, x2, y2)

def augment_sprite(img):
    # Random Rotation
    angle = random.randint(0, 360)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    # Compute new bounding dimensions
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - center[0]
    M[1, 2] += (nh / 2) - center[1]
    img = cv2.warpAffine(img, M, (nw, nh))

    # Random Brightness
    val = random.randint(-40, 40)
    img = img.astype(np.int16)
    img[:,:,:3] += val
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def yolo_norm(x, y, w, h, img_w, img_h):
    return (x + w/2)/img_w, (y + h/2)/img_h, w/img_w, h/img_h

def generate_panning_dataset():
    ensure_dirs()

    # --- 1. Load House to establish scale ---
    # We load the house first without scaling to get its native pixel size?
    # Actually, we should probably pick a reasonable resolution for the House first.
    # Let's say we want the House to be ~50-60% of the viewport width (640px).
    # 640 * 0.5 = 320px width for 12 ft. This gives decent resolution.

    # Let's load House and force it to be, say, 360 pixels wide.
    # Then PIXELS_PER_FOOT = 360 / 12 = 30 px/ft.
    TARGET_HOUSE_WIDTH_PX = 360

    house = load_asset(HOUSE_PATH, target_width_px=TARGET_HOUSE_WIDTH_PX)

    PIXELS_PER_FOOT = house.shape[1] / HOUSE_DIAMETER_FT
    print(f"Scale Established: {PIXELS_PER_FOOT:.2f} px/ft")

    # --- 2. Calculate Dimensions based on Scale ---
    STONE_WIDTH_PX = int(STONE_DIAMETER_FT * PIXELS_PER_FOOT)
    SHEET_WIDTH_PX = int(SHEET_WIDTH_FT * PIXELS_PER_FOOT)
    HOG_LINE_DIST_PX = int(HOG_LINE_DIST_FT * PIXELS_PER_FOOT)

    print(f"Sheet Width: {SHEET_WIDTH_PX} px")
    print(f"Stone Width: {STONE_WIDTH_PX} px")
    print(f"Hog Line Dist: {HOG_LINE_DIST_PX} px")

    # Calculate World Height to include House + Hog Line + some buffer
    # House is at the bottom, Hog is above it.
    # Let's place House Center at y = HOG_LINE_DIST_PX + WORLD_HEIGHT_BUFFER
    # Actually, usually House is at bottom. Let's define top-down.
    # y=0 is top of buffer.
    # y = WORLD_HEIGHT_BUFFER is Hog Line.
    # y = WORLD_HEIGHT_BUFFER + HOG_LINE_DIST_PX is House Center.
    # valid stone area is between them.

    # BUT house has a height. Center of house is (house.shape[0] // 2) from its top.

    # Let's define coordinate system relative to top of World:
    # 1. Hog Line Y (center of hog line strip)
    HOG_STRIP_Y = WORLD_HEIGHT_BUFFER

    # 2. House Center Y
    HOUSE_CENTER_Y = HOG_STRIP_Y + HOG_LINE_DIST_PX

    # 3. World Height
    # Needs to accommodate the house bottom.
    # House bottom Y = HOUSE_CENTER_Y + (house.shape[0] // 2)

    WORLD_HEIGHT = HOUSE_CENTER_Y + (house.shape[0] // 2) + 100 # +100 px buffer below house

    print(f"World Size: {SHEET_WIDTH_PX} x {WORLD_HEIGHT}")

    # --- 3. Load other assets with correct scale ---
    stone_r = load_asset(RED_STONE_PATH, target_width_px=STONE_WIDTH_PX)
    stone_y = load_asset(YEL_STONE_PATH, target_width_px=STONE_WIDTH_PX)

    # Hog line width should match sheet width usually, or at least be wide enough
    # We'll make it the sheet width
    hog_line = load_asset(HOG_LINE_PATH, target_width_px=SHEET_WIDTH_PX)

    # --- 4. Prepare World Canvas ---
    tile = cv2.imread(BG_PATH)
    if tile is None: tile = np.zeros((640, 640, 3), dtype=np.uint8)
    tile = cv2.resize(tile, (SHEET_WIDTH_PX, SHEET_WIDTH_PX)) # make it square relative to sheet width

    num_tiles = (WORLD_HEIGHT // tile.shape[0]) + 2
    world_base = np.vstack([tile for _ in range(num_tiles)])[:WORLD_HEIGHT, :]

    print(f"Generating {NUM_IMAGES} panning scenarios...")
    print("Updating real-time RTX render settings")
    settings = carb.settings.get_settings()
    # Restore resolution and sharpness
    settings.set_int("/rtx/post/dlss/execMode", 2)  # Quality Mode
    settings.set_float("/rtx/post/resample/fractionalResolution", 1.0)
    settings.set_int("/rtx/post/aa/op", 2)  # TAA Enabled
    settings.set_int("/rtx/materialDb/anisotropy", 16)  # Sharp textures at angles

    # Keep these for speed (they don't affect 'roundness' or 'lines')
    settings.set_int("/rtx/translucency/maxRefractionBounces", 1)
    settings.set_int("/rtx/raytracing/subsurface/maxSamplePerFrame", 0)

    for i in range(NUM_IMAGES):
        world = world_base.copy()

        # --- Place Static Objects ---

        # 1. House (Rings)
        # Centered horizontally
        house_x = (SHEET_WIDTH_PX - house.shape[1]) // 2
        house_y = HOUSE_CENTER_Y - (house.shape[0] // 2)
        world, house_bbox = overlay_transparent(world, house, house_x, house_y)

        # 2. Hog Line
        # Centered horizontally, at specific Y relative to house center
        hog_y = HOG_STRIP_Y
        hog_x = (SHEET_WIDTH_PX - hog_line.shape[1]) // 2
        world, hog_bbox = overlay_transparent(world, hog_line, hog_x, hog_y)

        final_objects = []
        if hog_bbox: final_objects.append((CLASSES['hog'], hog_bbox))
        if house_bbox: final_objects.append((CLASSES['house'], house_bbox))

        # --- Place Stones ---
        # We'll spawn them between Hog Line (y=hog_y) and slightly below House.

        valid_y_min = hog_y
        valid_y_max = house_y + house.shape[0] + 50

        num_red = random.randint(1, 8)
        num_yel = random.randint(1, 8)
        stones = [(CLASSES['red'], stone_r.copy()) for _ in range(num_red)] + \
                 [(CLASSES['yellow'], stone_y.copy()) for _ in range(num_yel)]
        random.shuffle(stones)

        for cls_id, sprite in stones:
            sprite = augment_sprite(sprite)
            # Ensure stone stays on sheet (x)
            r_x = random.randint(10, SHEET_WIDTH_PX - 10 - sprite.shape[1])
            r_y = random.randint(valid_y_min, valid_y_max)

            world, bbox = overlay_transparent(world, sprite, r_x, r_y)
            if bbox:
                final_objects.append((cls_id, bbox))

        # --- Simulate Camera Pan (Crop) ---
        # The sheet is now SHEET_WIDTH_PX wide.

        full_scene = np.zeros((WORLD_HEIGHT, VIEWPORT_SIZE[0], 3), dtype=np.uint8)
        # Center the sheet in the 640px width
        x_offset = (VIEWPORT_SIZE[0] - SHEET_WIDTH_PX) // 2

        # Paste world into full_scene
        full_scene[:, x_offset:x_offset+SHEET_WIDTH_PX] = world

        # Adjust object bbox coords by x_offset
        final_objects_shifted = []
        for cls_id, (x1, y1, x2, y2) in final_objects:
            final_objects_shifted.append((cls_id, (x1+x_offset, y1, x2+x_offset, y2)))

        # Panning/Cropping in Y only
        max_cam_y = WORLD_HEIGHT - VIEWPORT_SIZE[1]
        if max_cam_y < 0: max_cam_y = 0
        cam_y = random.randint(0, max_cam_y)

        view_img = full_scene[cam_y : cam_y + VIEWPORT_SIZE[1], 0 : VIEWPORT_SIZE[0]]

        # Labels
        view_labels = []
        for cls_id, (wx1, wy1, wx2, wy2) in final_objects_shifted:
            # Camera relative coords
            cx1, cy1 = wx1, wy1 - cam_y
            cx2, cy2 = wx2, wy2 - cam_y

            # Intersection with Viewport (0,0, 640,640)
            ix1 = max(0, cx1)
            iy1 = max(0, cy1)
            ix2 = min(VIEWPORT_SIZE[0], cx2)
            iy2 = min(VIEWPORT_SIZE[1], cy2)

            if ix2 > ix1 and iy2 > iy1:
                obj_area = (wx2-wx1)*(wy2-wy1)
                vis_area = (ix2-ix1)*(iy2-iy1)

                # Visibility threshold
                if obj_area > 0 and (vis_area / obj_area) > 0.15:
                    norm_label = yolo_norm(ix1, iy1, ix2-ix1, iy2-iy1,
                                           VIEWPORT_SIZE[0], VIEWPORT_SIZE[1])
                    view_labels.append((cls_id, *norm_label))

        # Save
        base_name = f"pan_{i:05d}"
        cv2.imwrite(os.path.join(IMAGES_DIR, f"{base_name}.jpg"), view_img)
        with open(os.path.join(LABELS_DIR, f"{base_name}.txt"), "w") as f:
            for lbl in view_labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

    print("Done generating panning dataset.")

if __name__ == "__main__":
    generate_panning_dataset()
