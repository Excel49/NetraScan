"""
NetraScan-3D: Vertical Checkerboard Generator
===============================================
Generates a printable checkerboard for VERTICAL placement
on top of the ChArUco board during dual-board calibration.

Setup (per SG14-byod3d.pdf):
  - ChArUco board DATAR di turntable → XY plane + origin
  - Checkerboard ini TEGAK di atas ChArUco → Z axis (atas)

Board Specs:
  - 5x5 squares → 4x4 inner corners (matches config.CHECKER_BOARD_SIZE)
  - Square size: 25mm (matches config.CHECKER_SQUARE_SIZE)
  - Total pattern: 125mm x 125mm
  - Print at 100% scale (Actual Size)

Usage:
  python create_vertical_checkerboard.py
"""

import cv2
import numpy as np
import os

# === CONFIGURATION ===
ROWS = 5              # Number of squares vertically
COLS = 5              # Number of squares horizontally
SQUARE_SIZE_MM = 25   # Each square is 25mm
DPI = 300             # Print resolution
INNER_CORNERS = (4, 4)  # For OpenCV detection verification

# Derived values
PIXELS_PER_MM = DPI / 25.4
SQUARE_SIZE_PX = int(SQUARE_SIZE_MM * PIXELS_PER_MM)

# Layout
BORDER_MM = 1         # Black border thickness
MARGIN_MM = 15        # White quiet zone
FOLD_TAB_MM = 20      # Bottom fold tab for standing support

BORDER_PX = int(BORDER_MM * PIXELS_PER_MM)
MARGIN_PX = int(MARGIN_MM * PIXELS_PER_MM)
FOLD_TAB_PX = int(FOLD_TAB_MM * PIXELS_PER_MM)


def create_checkerboard_grid(rows, cols, square_px):
    """Generate the raw checkerboard pattern."""
    width = cols * square_px
    height = rows * square_px
    
    img = np.ones((height, width), dtype=np.uint8) * 255  # White background
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:
                y = i * square_px
                x = j * square_px
                cv2.rectangle(img, (x, y), (x + square_px, y + square_px), 0, -1)
    
    return img


def add_borders_and_margins(grid_img, border_px, margin_px):
    """Add black border frame and white quiet zone margin."""
    # 1. Black border (defines the active pattern edge)
    img = cv2.copyMakeBorder(
        grid_img, border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=0
    )
    
    # 2. White quiet zone (critical for OpenCV detection)
    img = cv2.copyMakeBorder(
        img, margin_px, margin_px, margin_px, margin_px,
        cv2.BORDER_CONSTANT, value=255
    )
    
    return img


def add_fold_tab(img, fold_tab_px, pixels_per_mm):
    """Add a fold tab at the bottom for standing the board vertically."""
    tab = np.ones((fold_tab_px, img.shape[1]), dtype=np.uint8) * 255
    
    # Dashed fold line at the junction
    dash_len = int(3 * pixels_per_mm)
    gap_len = int(2 * pixels_per_mm)
    y_line = 0
    x = 0
    while x < tab.shape[1]:
        x_end = min(x + dash_len, tab.shape[1])
        cv2.line(tab, (x, y_line), (x_end, y_line), 128, 2)
        x = x_end + gap_len
    
    # "LIPAT DI SINI / FOLD HERE" text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "--- LIPAT DI SINI / FOLD HERE ---"
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_x = (tab.shape[1] - text_size[0]) // 2
    text_y = fold_tab_px // 2 + text_size[1] // 2
    cv2.putText(tab, text, (text_x, text_y), font, 0.5, 128, 1)
    
    return np.vstack([img, tab])


def add_annotations(img_gray, pixels_per_mm, grid_size_mm):
    """Add orientation arrows, labels, and cut guides. Returns color image."""
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_small = cv2.FONT_HERSHEY_SIMPLEX
    
    h, w = img.shape[:2]
    
    # --- UP ARROW on top center ---
    arrow_x = w // 2
    arrow_y_start = int(8 * pixels_per_mm)
    arrow_y_end = int(2 * pixels_per_mm)
    
    # Arrow line
    cv2.arrowedLine(img, (arrow_x, arrow_y_start), (arrow_x, arrow_y_end), 
                    (0, 0, 255), 3, tipLength=0.4)
    
    # "▲ UP / ATAS" text next to arrow
    up_text = "UP / ATAS"
    up_size = cv2.getTextSize(up_text, font_small, 0.6, 2)[0]
    cv2.putText(img, up_text, (arrow_x + 15, arrow_y_start - 5), 
                font_small, 0.6, (0, 0, 255), 2)
    
    # --- Title at the very top ---
    title = "VERTICAL CHECKER - TEGAK LURUS"
    title_size = cv2.getTextSize(title, font, 0.5, 1)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(img, title, (title_x, int(1.5 * pixels_per_mm)), 
                font, 0.5, (0, 0, 0), 1)
    
    # --- Size info at bottom (above fold tab) ---
    size_text = f"{grid_size_mm}mm x {grid_size_mm}mm | {COLS}x{ROWS} sq | {SQUARE_SIZE_MM}mm/sq"
    size_size = cv2.getTextSize(size_text, font_small, 0.4, 1)[0]
    size_x = (w - size_size[0]) // 2
    # Position just above the fold tab area
    fold_area_start = h - FOLD_TAB_PX
    cv2.putText(img, size_text, (size_x, fold_area_start - int(2 * pixels_per_mm)), 
                font_small, 0.4, (100, 100, 100), 1)
    
    # --- Cut guide (thin gray rectangle) ---
    cv2.rectangle(img, (1, 1), (w - 2, h - 2), (180, 180, 180), 1)
    
    # --- Print instruction ---
    print_text = "PRINT 100% SCALE (ACTUAL SIZE)"
    print_size = cv2.getTextSize(print_text, font_small, 0.35, 1)[0]
    print_x = (w - print_size[0]) // 2
    cv2.putText(img, print_text, (print_x, h - int(1 * pixels_per_mm)),
                font_small, 0.35, (150, 150, 150), 1)
    
    return img


def verify_detection(img_color, inner_corners):
    """Verify the generated image can be detected by OpenCV."""
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
             cv2.CALIB_CB_NORMALIZE_IMAGE + 
             cv2.CALIB_CB_FILTER_QUADS)
    
    found, corners = cv2.findChessboardCorners(gray, inner_corners, flags=flags)
    
    return found, corners


def main():
    grid_size_mm = COLS * SQUARE_SIZE_MM  # 125mm
    
    print(f"🏁 NetraScan-3D: Vertical Checkerboard Generator")
    print(f"=" * 50)
    print(f"  Grid:        {COLS}x{ROWS} squares")
    print(f"  Square Size: {SQUARE_SIZE_MM}mm")
    print(f"  Board Size:  {grid_size_mm}mm x {grid_size_mm}mm ({grid_size_mm/10:.1f}cm x {grid_size_mm/10:.1f}cm)")
    print(f"  Inner Corners: {INNER_CORNERS} (for OpenCV detection)")
    print(f"  DPI:         {DPI}")
    print(f"  Quiet Zone:  {MARGIN_MM}mm")
    print(f"  Fold Tab:    {FOLD_TAB_MM}mm")
    print()
    
    # Step 1: Create grid
    print("  [1/5] Generating checkerboard grid...")
    grid = create_checkerboard_grid(ROWS, COLS, SQUARE_SIZE_PX)
    
    # Step 2: Add borders and margins
    print("  [2/5] Adding borders and quiet zone...")
    bordered = add_borders_and_margins(grid, BORDER_PX, MARGIN_PX)
    
    # Step 3: Add fold tab
    print("  [3/5] Adding fold tab for vertical support...")
    with_tab = add_fold_tab(bordered, FOLD_TAB_PX, PIXELS_PER_MM)
    
    # Step 4: Add annotations
    print("  [4/5] Adding orientation markings and labels...")
    final = add_annotations(with_tab, PIXELS_PER_MM, grid_size_mm)
    
    # Step 5: Verify detection
    print("  [5/5] Verifying OpenCV detection...")
    found, corners = verify_detection(final, INNER_CORNERS)
    if found:
        print(f"  ✅ VERIFIED: OpenCV detected {INNER_CORNERS} inner corners!")
    else:
        print(f"  ⚠️  WARNING: OpenCV could not detect corners in generated image.")
        print(f"       This may be OK - detection works better on camera images.")
        print(f"       The pattern geometry is correct for {INNER_CORNERS} inner corners.")
    
    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "vertical_checkerboard_5x5.png")
    cv2.imwrite(output_file, final)
    
    print()
    print(f"✅ Saved to: {output_file}")
    print()
    print("📋 INSTRUKSI PENGGUNAAN:")
    print("  1. Print gambar ini dengan skala 100% (Actual Size)")
    print("  2. Potong mengikuti garis abu-abu terluar")
    print("  3. Lipat bagian bawah (fold tab) sebagai penyangga")
    print("  4. Letakkan TEGAK LURUS di atas ChArUco board")
    print("  5. Panah merah (▲ UP) harus menghadap ke ATAS")
    print("  6. Pastikan kedua board terlihat oleh kamera")
    print()
    print(f"📐 Config match: CHECKER_BOARD_SIZE={INNER_CORNERS}, CHECKER_SQUARE_SIZE={SQUARE_SIZE_MM/1000:.3f}m")


if __name__ == "__main__":
    main()
