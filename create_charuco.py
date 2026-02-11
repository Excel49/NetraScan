
import cv2
import cv2.aruco as aruco
import numpy as np
import os

# --- KONFIGURASI SESUAI REQUEST (12.5cm x 12.5cm) ---
# Dimensi Total: 12.5cm
# Grid: 5 x 5
# Ukuran Square: 12.5 / 5 = 2.5 cm -> 0.025 m
# Ukuran Marker: 1.8 cm -> 0.018 m (approx 72% dari square)

SQUARE_LENGTH = 0.025
MARKER_LENGTH = 0.018
BOARD_SIZE = (5, 5)
DICT_ID = aruco.DICT_6X6_250

def generate_board():
    print(f"Generating ChArUco Board {BOARD_SIZE}...")
    
    aruco_dict = aruco.getPredefinedDictionary(DICT_ID)
    
    # Create Board (OpenCV 4.x syntax)
    # Note: CharucoBoard constructor signature depends on version.
    # Trying common constructor: CharucoBoard(size, squareLength, markerLength, dictionary)
    try:
        board = aruco.CharucoBoard(
            BOARD_SIZE, 
            SQUARE_LENGTH, 
            MARKER_LENGTH, 
            aruco_dict
        )
    except AttributeError:
        # Fallback for older OpenCV
        board = aruco.CharucoBoard_create(
            BOARD_SIZE[0], BOARD_SIZE[1], 
            SQUARE_LENGTH, MARKER_LENGTH, 
            aruco_dict
        )

    # Calculate Pixel Size for 300 DPI
    # Total physical width = 5 * 2.5cm = 12.5cm = 125mm
    SIZE_MM = 125 
    DPI = 300
    SIZE_PX = int((SIZE_MM / 25.4) * DPI) # ~1476 px
    
    print(f"Target Size: {SIZE_MM}mm = {SIZE_PX}px")
    
    # Generate Image
    img = board.generateImage((SIZE_PX, SIZE_PX), marginSize=0, borderBits=1)
    
    # Create A4 Canvas to center it
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297
    A4_WIDTH_PX = int((A4_WIDTH_MM / 25.4) * DPI)
    A4_HEIGHT_PX = int((A4_HEIGHT_MM / 25.4) * DPI)
    
    canvas = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype=np.uint8) * 255
    
    x_off = (A4_WIDTH_PX - SIZE_PX) // 2
    y_off = (A4_HEIGHT_PX - SIZE_PX) // 2
    
    canvas[y_off:y_off+SIZE_PX, x_off:x_off+SIZE_PX] = img
    
    # Draw Red Cut Lines
    canvas_color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(canvas_color, (x_off, y_off), (x_off+SIZE_PX, y_off+SIZE_PX), (0,0,255), 4)
    
    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas_color, "12.5cm x 12.5cm ChArUco", (x_off, y_off - 50), font, 2, (0,0,0), 4)
    cv2.putText(canvas_color, "GRID: 5x5 | SQ: 2.5cm", (x_off, y_off + SIZE_PX + 100), font, 1.5, (0,0,0), 3)
    
    output_filename = "charuco_12.5x12.5cm.png"
    cv2.imwrite(output_filename, canvas_color)
    print(f"✅ Saved to {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    generate_board()
