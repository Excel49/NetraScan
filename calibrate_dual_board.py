"""
NetraScan-3D: Dual-Board Extrinsic Calibration (IMPROVED)
==========================================================
Kalibrasi posisi kamera menggunakan 2 board:
1. ChArUco board DATAR di turntable → XY plane + origin
2. Checkerboard biasa TEGAK di atas ChArUco → Z axis (atas)

IMPROVEMENTS:
- Multi-strategy checkerboard detection (CLAHE, blur, threshold, etc.)
- Real-time diagnostics overlay
- ROI hints for better detection
- Configurable detection parameters

INSTRUCTIONS:
1. Taruh ChArUco board datar di pusat turntable
2. Taruh checkerboard biasa berdiri tegak lurus di atas ChArUco (di tengah)
3. Pastikan kedua board terlihat oleh kamera
4. Press [SPACE] ketika kedua board terdeteksi (hijau)
5. Press [D] untuk toggle debug view
6. Press [Q] untuk keluar
"""
import cv2
import numpy as np
import json
import time
import os
import sys

# Add parent dir to path and change to script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from core.system import SystemController
from core import config


# ===== ENHANCED CHECKERBOARD DETECTION =====

def preprocess_for_detection(gray):
    """
    Generate multiple preprocessed versions of the image
    to maximize detection chance on the vertical checkerboard.
    """
    versions = []
    
    # 1. Original
    versions.append(("original", gray))
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    versions.append(("CLAHE", clahe_img))
    
    # 3. Gaussian blur + CLAHE (reduces noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_clahe = clahe.apply(blurred)
    versions.append(("blur+CLAHE", blurred_clahe))
    
    # 4. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 5
    )
    versions.append(("adaptive_thresh", adaptive))
    
    # 5. Bilateral filter (edge-preserving noise reduction)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    versions.append(("bilateral", bilateral))
    
    # 6. Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    versions.append(("sharpened", sharpened))
    
    return versions


def find_checkerboard_robust(gray, camera_matrix, dist_coeffs):
    """
    Robust checkerboard detection using multiple preprocessing strategies.
    Returns (rvec, tvec, corners, method_used) or (None, None, None, None).
    """
    board_size = config.CHECKER_BOARD_SIZE
    square_size = config.CHECKER_SQUARE_SIZE
    
    # Multiple flag combinations to try
    flag_sets = [
        ("ADAPTIVE+NORM", cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE),
        ("ADAPTIVE+NORM+FILTER", cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS),
        ("ADAPTIVE+FAST", cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK),
    ]
    
    # Get preprocessed versions
    versions = preprocess_for_detection(gray)
    
    # Try each combination
    for version_name, processed in versions:
        for flag_name, flags in flag_sets:
            found, corners = cv2.findChessboardCorners(processed, board_size, flags=flags)
            
            if found:
                # Refine corners on original gray image
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Create object points
                obj_pts = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
                for i in range(board_size[1]):
                    for j in range(board_size[0]):
                        obj_pts[i * board_size[0] + j] = [j * square_size, i * square_size, 0]
                
                # Solve PnP
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners_refined,
                    camera_matrix, dist_coeffs
                )
                
                if success:
                    method = f"{version_name}+{flag_name}"
                    return rvec, tvec, corners_refined, method
    
    return None, None, None, None


def find_charuco(gray, charuco_detector, charuco_board, camera_matrix, dist_coeffs):
    """Detect ChArUco board and return pose (rvec, tvec) with origin at board center."""
    corners, ids, _, _ = charuco_detector.detectBoard(gray)
    
    if corners is None or len(corners) < 6:
        return None, None, None, None
    
    all_board_corners = charuco_board.getChessboardCorners()
    obj_pts = []
    img_pts = []
    
    for j, cid in enumerate(ids):
        id_val = int(cid[0])
        if id_val < len(all_board_corners):
            obj_pts.append(all_board_corners[id_val])
            img_pts.append(corners[j])
    
    if len(obj_pts) < 6:
        return None, None, None, None
    
    obj_pts_arr = np.array(obj_pts, dtype=np.float32).reshape(-1, 3)
    img_pts_arr = np.array(img_pts, dtype=np.float32).reshape(-1, 2)
    
    # Center origin to board center
    center_x = (config.BOARD_SIZE[0] * config.SQUARE_LENGTH) / 2.0
    center_y = (config.BOARD_SIZE[1] * config.SQUARE_LENGTH) / 2.0
    center_offset = np.array([center_x, center_y, 0], dtype=np.float32)
    obj_pts_centered = obj_pts_arr - center_offset
    
    success, rvec, tvec = cv2.solvePnP(
        obj_pts_centered, img_pts_arr,
        camera_matrix, dist_coeffs
    )
    
    if success:
        return rvec, tvec, corners, ids
    return None, None, None, None


def compute_camera_position(charuco_rvec, charuco_tvec, checker_rvec, checker_tvec):
    """
    Compute camera position using:
    - ChArUco: defines XY plane origin
    - Checkerboard: defines Z axis (up direction from vertical board)
    
    Returns camera_position in turntable coordinates.
    """
    # --- ChArUco gives us the horizontal plane transform ---
    R_charuco, _ = cv2.Rodrigues(charuco_rvec)
    
    # Camera position relative to ChArUco center (= turntable center)
    camera_pos_charuco = -R_charuco.T @ charuco_tvec.flatten()
    
    # --- Checkerboard gives us the "up" direction ---
    R_checker, _ = cv2.Rodrigues(checker_rvec)
    
    # The checkerboard is vertical. Its Y axis (going "up" on the board) 
    # represents the world Z axis (up).
    checker_y_in_camera = R_checker[:, 1]  # Local Y of checkerboard
    
    # Transform this direction to ChArUco/turntable frame
    up_direction_world = R_charuco.T @ checker_y_in_camera
    
    # Normalize
    up_direction_world = up_direction_world / np.linalg.norm(up_direction_world)
    
    # Build a proper rotation matrix for the turntable frame
    z_axis = up_direction_world
    
    # ChArUco's X axis in camera frame
    charuco_x_in_camera = R_charuco[:, 0]
    charuco_x_world = R_charuco.T @ charuco_x_in_camera
    
    # Make X perpendicular to Z
    x_axis = charuco_x_world - np.dot(charuco_x_world, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Y = Z cross X
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Build world rotation: columns are the world axes in turntable coords
    R_world = np.column_stack([x_axis, y_axis, z_axis])
    
    # Transform camera position to this corrected frame
    camera_pos_corrected = R_world.T @ camera_pos_charuco
    
    return camera_pos_corrected, R_world, up_direction_world


# ===== DIAGNOSTIC OVERLAY =====

def draw_diagnostics(display, gray, charuco_ok, charuco_count, checker_ok, checker_count,
                     checker_method, capture_requested, show_debug):
    """Draw all status overlays and diagnostics."""
    h, w = display.shape[:2]
    
    # --- Status Bar (top) ---
    y_pos = 30
    
    # ChArUco status
    charuco_color = (0, 255, 0) if charuco_ok else (0, 0, 255)
    charuco_icon = "✓" if charuco_ok else "✗"
    charuco_text = f"ChArUco: {charuco_count} corners ({'OK' if charuco_ok else 'Too Few'})"
    cv2.putText(display, charuco_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, charuco_color, 2)
    
    # Checker status
    y_pos += 30
    checker_color = (0, 255, 0) if checker_ok else (0, 0, 255)
    checker_text = f"Checker: {checker_count} corners ({'OK' if checker_ok else 'Not Found'})"
    cv2.putText(display, checker_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, checker_color, 2)
    
    # Detection method (if found)
    if checker_ok and checker_method:
        y_pos += 25
        cv2.putText(display, f"  Method: {checker_method}", (20, y_pos),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    
    # Overall status
    y_pos += 35
    if capture_requested:
        if charuco_ok and checker_ok:
            cv2.putText(display, "CAPTURING NOW... DON'T MOVE!", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "WAITING FOR LOCK... (Keep steady)", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    elif charuco_ok and checker_ok:
        cv2.putText(display, "READY - Press SPACE to capture", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        cv2.putText(display, "Waiting for both boards...", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    
    # --- Tips when checker not found ---
    if not checker_ok:
        tips_y = y_pos + 35
        tips = [
            "TIPS jika checker tidak terdeteksi:",
            "  - Pastikan pencahayaan merata (tidak silau)",
            "  - Board harus tegak lurus (90 derajat)",
            "  - Jarak cukup dekat ke kamera",
            "  - Semua 5x5 kotak harus terlihat",
            "  - Coba geser/putar sedikit board-nya",
        ]
        for tip in tips:
            cv2.putText(display, tip, (20, tips_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
            tips_y += 20
    
    # --- Debug view ---
    if show_debug:
        # Show preprocessing thumbnails in corner
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        debug_img = clahe.apply(gray)
        thumb_h = h // 4
        thumb_w = w // 4
        thumb = cv2.resize(debug_img, (thumb_w, thumb_h))
        thumb_color = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.putText(thumb_color, "CLAHE Preview", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        display[h - thumb_h:h, w - thumb_w:w] = thumb_color
    
    # Controls
    cv2.putText(display, "[SPACE] Calibrate | [D] Debug | [Q] Quit", (20, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return display


# ===== MAIN =====

def main():
    print("🎯 NetraScan-3D: DUAL-BOARD EXTRINSIC CALIBRATION (IMPROVED)")
    print("=" * 60)
    print("Setup:")
    print("  1. ChArUco board DATAR di turntable (XY plane)")
    print("  2. Checkerboard biasa TEGAK di atas ChArUco (Z axis)")
    print()
    print("Controls:")
    print("  [SPACE] = Calibrate (when both boards green)")
    print("  [D]     = Toggle debug view")
    print("  [Q]     = Quit")
    print()
    
    system = SystemController.get_instance()
    
    # Ensure camera is ready
    if not system.camera.cap or not system.camera.cap.isOpened():
        print("📷 Connecting to camera...")
        system.camera.connect()
        for _ in range(20):
            system.camera.get_frame()
    
    # Check camera calibration (intrinsics needed)
    if not system.triangulation.calibrated:
        print("❌ ERROR: Camera not calibrated! Run intrinsic calibration first.")
        return
    
    camera_matrix = system.triangulation.camera_matrix
    dist_coeffs = system.triangulation.dist_coeffs
    
    print("✅ Camera intrinsics loaded.")
    print(f"📋 ChArUco: {config.BOARD_SIZE}, square={config.SQUARE_LENGTH*100:.1f}cm")
    print(f"📋 Checker: {config.CHECKER_BOARD_SIZE}, square={config.CHECKER_SQUARE_SIZE*100:.1f}cm")
    print()
    
    # Improves detection robustness
    params = system.auto_calib.detector_params
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.01
    
    detector = system.auto_calib.charuco_detector
    detector.setDetectorParameters(params)
    
    print("🔴 LIVE FEED STARTED")
    print("   Multi-strategy detection ENABLED (6 preprocessing × 3 flag combos)")
    print()
    
    capture_requested = False
    show_debug = False
    frame_count = 0
    
    # Detection history for stability
    checker_found_count = 0
    
    while True:
        frame = system.camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Detect ChArUco ---
        charuco_corners, charuco_ids, marker_corners, marker_ids = system.auto_calib.charuco_detector.detectBoard(gray)
        
        # Draw ArUco markers
        if marker_corners:
            cv2.aruco.drawDetectedMarkers(display, list(marker_corners), marker_ids)
        
        # Check ChArUco corners
        charuco_ok = False
        charuco_count = 0
        charuco_rvec = None
        charuco_tvec = None
        
        if charuco_corners is not None and len(charuco_corners) >= 6:
            charuco_count = len(charuco_corners)
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))
            
            # Manual Pose Estimation
            all_board_corners = system.auto_calib.charuco_board.getChessboardCorners()
            obj_pts = []
            img_pts = []
            
            for j, cid in enumerate(charuco_ids):
                id_val = int(cid[0])
                if id_val < len(all_board_corners):
                    obj_pts.append(all_board_corners[id_val])
                    img_pts.append(charuco_corners[j])
            
            if len(obj_pts) >= 6:
                obj_pts_arr = np.array(obj_pts, dtype=np.float32).reshape(-1, 3)
                img_pts_arr = np.array(img_pts, dtype=np.float32).reshape(-1, 2)
                
                center_x = (config.BOARD_SIZE[0] * config.SQUARE_LENGTH) / 2.0
                center_y = (config.BOARD_SIZE[1] * config.SQUARE_LENGTH) / 2.0
                center_offset = np.array([center_x, center_y, 0], dtype=np.float32)
                obj_pts_centered = obj_pts_arr - center_offset
                
                success, charuco_rvec, charuco_tvec = cv2.solvePnP(
                    obj_pts_centered, img_pts_arr,
                    camera_matrix, dist_coeffs
                )
                
                if success:
                    charuco_ok = True
                    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, charuco_rvec, charuco_tvec, 0.05)
        
        # --- Detect Checkerboard (ROBUST) ---
        checker_rvec, checker_tvec, checker_corners, checker_method = find_checkerboard_robust(
            gray, camera_matrix, dist_coeffs
        )
        
        checker_ok = checker_rvec is not None
        checker_count = len(checker_corners) if checker_corners is not None else 0
        
        if checker_ok:
            checker_found_count += 1
            cv2.drawChessboardCorners(display, config.CHECKER_BOARD_SIZE, checker_corners, True)
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, checker_rvec, checker_tvec, 0.03)
            
            # Log first detection
            if checker_found_count == 1:
                print(f"🟢 CHECKER DETECTED! Method: {checker_method}")
        else:
            checker_found_count = 0
        
        # --- Draw diagnostics overlay ---
        display = draw_diagnostics(
            display, gray, charuco_ok, charuco_count,
            checker_ok, checker_count, checker_method,
            capture_requested, show_debug
        )
        
        cv2.imshow("Dual-Board Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 Quit.")
            break
        
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"🔍 Debug view: {'ON' if show_debug else 'OFF'}")
        
        elif key == 32:  # SPACE
            capture_requested = True
            print("🏁 CAPTURE REQUESTED! Waiting for valid detection...")
            
        if capture_requested and charuco_ok and checker_ok:
            print()
            print("=" * 60)
            print("🔧 COMPUTING CALIBRATION...")
            print(f"   Checker method: {checker_method}")
            print("=" * 60)
            
            # Compute camera position
            camera_pos, R_world, up_dir = compute_camera_position(
                charuco_rvec, charuco_tvec, 
                checker_rvec, checker_tvec
            )
            
            # Also compute simple ChArUco-only position for comparison
            R_c, _ = cv2.Rodrigues(charuco_rvec)
            camera_pos_simple = -R_c.T @ charuco_tvec.flatten()
            
            distance = np.linalg.norm(camera_pos)
            
            print(f"📍 Camera Position (corrected with Z axis):")
            print(f"   X: {camera_pos[0]*1000:.1f} mm")
            print(f"   Y: {camera_pos[1]*1000:.1f} mm")
            print(f"   Z: {camera_pos[2]*1000:.1f} mm")
            print(f"   Distance: {distance*1000:.1f} mm")
            print()
            print(f"📍 Camera Position (ChArUco only, for comparison):")
            print(f"   X: {camera_pos_simple[0]*1000:.1f} mm")
            print(f"   Y: {camera_pos_simple[1]*1000:.1f} mm")  
            print(f"   Z: {camera_pos_simple[2]*1000:.1f} mm")
            print()
            print(f"⬆️  Up direction (Z axis): [{up_dir[0]:.3f}, {up_dir[1]:.3f}, {up_dir[2]:.3f}]")
            print()
            
            # Save to calibration.json
            calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
            try:
                with open(calib_path, "r") as f:
                    calib_data = json.load(f)
            except:
                calib_data = {}
            
            calib_data["extrinsics"] = {
                "camera_position": camera_pos.tolist(),
                "world_rotation": R_world.tolist(),
                "up_direction": up_dir.tolist(),
                "detection_method": checker_method,
                "timestamp": time.time()
            }
            
            with open(calib_path, "w") as f:
                json.dump(calib_data, f, indent=4)
                
            print(f"✅ Calibration SAVED to {calib_path}")
            print("🎉 DONE! You can close this window now.")
            
            # Flash screen
            cv2.imshow("Dual-Board Calibration", np.ones_like(display) * 255)
            cv2.waitKey(2000)
            break

    
    system.camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
