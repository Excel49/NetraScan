import cv2
import numpy as np

def create_checkerboard(rows, cols, square_size_mm, dpi=300):
    """
    Generate a checkerboard image for printing with EXTRA WIDE MARGINS.
    """
    # Convert mm to pixels
    pixels_per_mm = dpi / 25.4
    square_size_px = int(square_size_mm * pixels_per_mm)
    
    # Grid size
    grid_w = cols * square_size_px
    grid_h = rows * square_size_px
    
    # 1. Create grid image
    img = np.zeros((grid_h, grid_w), dtype=np.uint8)
    img.fill(255) # White background
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:
                # Black square
                y = i * square_size_px
                x = j * square_size_px
                cv2.rectangle(img, (x, y), (x + square_size_px, y + square_size_px), 0, -1)
                
    # 2. Add a thin black border (1mm) to define the ACTIVE pattern
    border_px = int(1 * pixels_per_mm)
    img = cv2.copyMakeBorder(img, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=0)

    # 3. Add EXTRA WIDE white margin (30mm) - "QUIET ZONE"
    margin_px = int(30 * pixels_per_mm) 
    img_final = cv2.copyMakeBorder(img, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255)
    
    # 4. Add CUT LINES (Dotted line effect via simple drawing) outside that margin
    # We add another small border for the cut line itself
    cut_line_px = int(1 * pixels_per_mm)
    # Draw a thin gray rect around the whole thing to serve as cut line
    cv2.rectangle(img_final, (0, 0), (img_final.shape[1]-1, img_final.shape[0]-1), 128, cut_line_px)
    
    # 5. Add text instructions
    font_scale = 1.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Top text
    text_top = "POTONG DI GARIS BUJUR SANGKAR TERLUAR (GRAY LINE)"
    text_size_top = cv2.getTextSize(text_top, font, font_scale, 2)[0]
    text_x_top = (img_final.shape[1] - text_size_top[0]) // 2
    cv2.putText(img_final, text_top, (text_x_top, 40), font, font_scale, 0, 2)

    # Bottom text
    text_bot = "JANGAN POTONG AREA PUTIH DI DALAM!! (Keep White Area)"
    text_size_bot = cv2.getTextSize(text_bot, font, font_scale, 2)[0]
    text_x_bot = (img_final.shape[1] - text_size_bot[0]) // 2
    cv2.putText(img_final, text_bot, (text_x_bot, img_final.shape[0] - 20), font, font_scale, 0, 2)
    
    return img_final

if __name__ == "__main__":
    ROWS = 5 # Squares
    COLS = 5 # Squares
    SQUARE_SIZE = 25 # mm
    FILENAME = "checkerboard_optimized.png"
    
    print(f"Generating optimized {COLS}x{ROWS} checkerboard (25mm squares) with 30mm margin...")
    img = create_checkerboard(ROWS, COLS, SQUARE_SIZE)
    
    cv2.imwrite(FILENAME, img)
    print(f"✅ Saved to {FILENAME}")
    print("Silakan print gambar ini Scale 100% (Actual Size).")
