import cv2
import numpy as np

def create_checkerboard(rows, cols, square_size_mm, dpi=300):
    """
    Generate a checkerboard image for printing.
    """
    # Convert mm to pixels
    pixels_per_mm = dpi / 25.4
    square_size_px = int(square_size_mm * pixels_per_mm)
    
    width = cols * square_size_px
    height = rows * square_size_px
    
    # Create image
    img = np.zeros((height, width), dtype=np.uint8)
    img.fill(255) # White background
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:
                # Black square
                y = i * square_size_px
                x = j * square_size_px
                cv2.rectangle(img, (x, y), (x + square_size_px, y + square_size_px), 0, -1)
                
                
    # Add a thin black border (1mm) to define the edge
    border_px = int(1 * pixels_per_mm)
    img_with_black_border = cv2.copyMakeBorder(img, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=0)

    # Add a white margin (10mm)
    margin = int(10 * pixels_per_mm) 
    img_with_border = cv2.copyMakeBorder(img_with_black_border, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    
    return img_with_border

if __name__ == "__main__":
    ROWS = 5
    COLS = 5
    SQUARE_SIZE = 25 # mm
    FILENAME = "checkerboard_5x5.png"
    
    print(f"Generating {COLS}x{ROWS} checkerboard with {SQUARE_SIZE}mm squares...")
    img = create_checkerboard(ROWS, COLS, SQUARE_SIZE)
    
    cv2.imwrite(FILENAME, img)
    print(f"✅ Saved to {FILENAME}")
    print("Silakan print gambar ini dengan skala 100% (Actual Size).")
