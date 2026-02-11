"""
NetraScan-3D: Dual-Board Extrinsic Calibration
================================================
Kalibrasi posisi kamera menggunakan 2 board:
1. ChArUco board DATAR di turntable → XY plane + origin
2. Checkerboard biasa TEGAK di atas ChArUco → Z axis (atas)

INSTRUCTIONS:
1. Taruh ChArUco board datar di pusat turntable
2. Taruh checkerboard biasa berdiri tegak lurus di atas ChArUco (di tengah)
3. Pastikan kedua board terlihat oleh kamera
4. Press [SPACE] ketika kedua board terdeteksi (hijau)
5. Press [Q] untuk keluar
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


def find_checkerboard(gray, camera_matrix, dist_coeffs):
    """Detect regular checkerboard and return pose (rvec, tvec)."""
    board_size = config.CHECKER_BOARD_SIZE
    square_size = config.CHECKER_SQUARE_SIZE
    
    # Find corners
    # Find corners
    # flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    # Remove FAST_CHECK for better detection on difficult images
    # Add FILTER_QUADS to filter out bad quads
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    found, corners = cv2.findChessboardCorners(gray, board_size, flags=flags)
    
    if not found:
        return None, None, None
    
    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Create object points (checkerboard in its own coordinate system)
    # Z-axis is 0 (flat on its plane), we will rotate it later
    obj_pts = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    # The order of corners in findChessboardCorners depends on the image orientation
    # Usually row by row, left to right
    for i in range(board_size[1]):
        for j in range(board_size[0]):
            obj_pts[i * board_size[0] + j] = [j * square_size, i * square_size, 0]
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, corners_refined,
        camera_matrix, dist_coeffs
    )
    
    if success:
        return rvec, tvec, corners_refined
    return None, None, None


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
    # In the checkerboard's local frame, Y goes up along the board.
    # We need to find which direction this is in camera coordinates.
    
    # Checkerboard's local Y axis in camera coordinates = R_checker's 2nd column
    checker_y_in_camera = R_checker[:, 1]  # Local Y of checkerboard
    
    # Transform this direction to ChArUco/turntable frame
    # Direction in turntable frame = R_charuco^T * direction_in_camera
    up_direction_world = R_charuco.T @ checker_y_in_camera
    
    # Normalize
    up_direction_world = up_direction_world / np.linalg.norm(up_direction_world)
    
    # Build a proper rotation matrix for the turntable frame
    # Z_world = up direction (from checkerboard)
    # X_world and Y_world from ChArUco, but corrected so Z = up
    
    z_axis = up_direction_world
    
    # ChArUco's X axis in camera frame
    charuco_x_in_camera = R_charuco[:, 0]
    charuco_x_world = R_charuco.T @ charuco_x_in_camera  # Should be [1,0,0] ideally
    
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


def main():
    print("🎯 NetraScan-3D: DUAL-BOARD EXTRINSIC CALIBRATION")
    print("=" * 52)
    print("Setup:")
    print("  1. ChArUco board DATAR di turntable (XY plane)")
    print("  2. Checkerboard biasa TEGAK di atas ChArUco (Z axis)")
    print()
    print("Controls:")
    print("  [SPACE] = Calibrate (when both boards green)")
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
    
    capture_requested = False
    
    while True:
        frame = system.camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Detect ChArUco ---
        charuco_corners, charuco_ids, marker_corners, marker_ids = system.auto_calib.charuco_detector.detectBoard(gray)
        
        # 1. Draw Markers (Debug)
        if marker_corners:
            cv2.aruco.drawDetectedMarkers(display, list(marker_corners), marker_ids)
            
        # 2. Check Charuco Corners
        charuco_ok = False
        charuco_count = 0
        if charuco_corners is not None and len(charuco_corners) >= 6:
            charuco_count = len(charuco_corners)
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))
            # Manual Pose Estimation (Robust like find_charuco logic)
            all_board_corners = system.auto_calib.charuco_board.getChessboardCorners()
            obj_pts = []
            img_pts = []
            
            # Match detected corners to board 3D points
            for j, cid in enumerate(charuco_ids):
                id_val = int(cid[0])
                if id_val < len(all_board_corners):
                    obj_pts.append(all_board_corners[id_val])
                    img_pts.append(charuco_corners[j])
            
            if len(obj_pts) >= 6:
                obj_pts_arr = np.array(obj_pts, dtype=np.float32).reshape(-1, 3)
                img_pts_arr = np.array(img_pts, dtype=np.float32).reshape(-1, 2)
                
                # Center origin to board center
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
        
        # --- Detect Checkerboard ---
        checker_rvec, checker_tvec, checker_corners = find_checkerboard(
            gray, camera_matrix, dist_coeffs
        )
        
        # Fallback: Try smaller sizes if full board fails
        if checker_corners is None:
            # Try 4x3
            # NOTE: Ideally we want full size for accuracy
            pass
        
        checker_ok = checker_rvec is not None
        checker_count = len(checker_corners) if checker_corners is not None else 0
        
        if checker_ok:
            cv2.drawChessboardCorners(display, config.CHECKER_BOARD_SIZE, checker_corners, True)
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, checker_rvec, checker_tvec, 0.03)
        else:
            # DIAGNOSTIC: Try finding a smaller board (e.g. 4x3) to see if bottom row is blocked
            # Inner corners (4,3) means 1 row is missing
            diag_size = (4, 3)
            diag_found, diag_corners = cv2.findChessboardCorners(gray, diag_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)
            if diag_found:
                cv2.drawChessboardCorners(display, diag_size, diag_corners, True)
                y_pos_diag = 200
                cv2.putText(display, "PARTIAL DETECTION (4x3)!", (20, y_pos_diag), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(display, "BOTTOM ROW BLOCKED? LIFT BOARD UP!", (20, y_pos_diag+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # --- Status UI ---
        y_pos = 30
        # ChArUco status
        charuco_color = (0, 255, 0) if charuco_ok else (0, 0, 255)
        charuco_text = f"ChArUco: {charuco_count} corners ({'OK' if charuco_ok else 'Too Few'})"
        cv2.putText(display, charuco_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, charuco_color, 2)
        
        # Checker status
        y_pos += 30
        checker_color = (0, 255, 0) if checker_ok else (0, 0, 255)
        checker_text = f"Checker: {checker_count} corners ({'OK' if checker_ok else 'Not Found'})"
        cv2.putText(display, checker_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, checker_color, 2)
        
        # Overall status
        y_pos += 40
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
        
        # Controls
        cv2.putText(display, "[SPACE] Calibrate | [Q] Quit", (20, display.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Dual-Board Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 Quit.")
            break
        
        elif key == 32:  # SPACE
            capture_requested = True
            print("🏁 CAPTURE REQUESTED! Waiting for valid detection...")
            
        if capture_requested and charuco_ok and checker_ok:
            print()
            print("=" * 52)
            print("🔧 COMPUTING CALIBRATION...")
            print("=" * 52)
            
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
