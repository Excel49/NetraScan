"""
NetraScan-3D: Extrinsic Calibration Tool
=========================================
This script calibrates the camera's position relative to the turntable center (World Origin).

INSTRUCTIONS:
1. Place the ChArUco board FLAT on the turntable center
2. The board's origin (corner with marker 0) should be at/near turntable center
3. Press [SPACE] to capture and calculate extrinsics
4. Press [Q] to quit
"""
import cv2
import numpy as np
import json
import time
from core.system import SystemController

def main():
    print("🎯 NetraScan-3D: EXTRINSIC CALIBRATION")
    print("======================================")
    print("This calibrates the camera position relative to turntable center.")
    print("INSTRUCTIONS:")
    print("1. Place ChArUco board FLAT on turntable center")
    print("2. Board origin (marker 0 corner) should be at turntable center")
    print("3. Press [SPACE] to capture and save extrinsics")
    print("4. Press [Q] to quit")
    print()
    
    system = SystemController.get_instance()
    
    # Ensure camera is ready
    if not system.camera.cap or not system.camera.cap.isOpened():
        print("Connecting to camera...")
        system.camera.connect()
        for _ in range(20): system.camera.get_frame()
    
    # Check camera calibration
    if not system.triangulation.calibrated:
        print("ERROR: Camera not calibrated! Run manual_camera_calibration.py first.")
        return
    
    camera_matrix = system.triangulation.camera_matrix
    dist_coeffs = system.triangulation.dist_coeffs
    
    print("Camera calibration loaded.")
    print()
    print("LIVE FEED STARTED")
    
    while True:
        frame = system.camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ChArUco
        corners, ids, _, _ = system.auto_calib.charuco_detector.detectBoard(gray)
        
        status_text = "No Board Detected"
        status_color = (0, 0, 255)
        rvec = None
        tvec = None
        
        if corners is not None and len(corners) > 4:
            # Draw detected corners
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids, (0, 255, 0))
            
            # Estimate pose
            all_board_corners = system.auto_calib.charuco_board.getChessboardCorners()
            obj_pts = []
            img_pts = []
            
            for j, cid in enumerate(ids):
                id_val = int(cid[0])
                if id_val < len(all_board_corners):
                    obj_pts.append(all_board_corners[id_val])
                    img_pts.append(corners[j])
            
            if len(obj_pts) >= 4:
                obj_pts_arr = np.array(obj_pts, dtype=np.float32).reshape(-1, 3)
                img_pts_arr = np.array(img_pts, dtype=np.float32).reshape(-1, 2)
                
                # Offset object points so origin = center of board (= turntable center)
                # Board: 5x5 squares × 0.025m = 0.125m → center = 0.0625m
                from core.config import SQUARE_LENGTH, BOARD_SIZE
                center_x = (BOARD_SIZE[0] * SQUARE_LENGTH) / 2.0
                center_y = (BOARD_SIZE[1] * SQUARE_LENGTH) / 2.0
                center_offset = np.array([center_x, center_y, 0], dtype=np.float32)
                obj_pts_centered = obj_pts_arr - center_offset
                
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts_centered, img_pts_arr,
                    camera_matrix, dist_coeffs
                )
                
                if success:
                    # Draw axes at board CENTER (not corner)
                    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    
                    status_text = f"Board OK - Press SPACE to calibrate"
                    status_color = (0, 255, 0)
        
        # UI
        cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display, "[SPACE] Calibrate | [Q] Quit", (20, display.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.imshow("Extrinsic Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 32:  # SPACE
            if rvec is not None and tvec is not None:
                # Calculate world transform (Camera -> World)
                R, _ = cv2.Rodrigues(rvec)
                
                # The tvec is the position of board origin in camera coordinates
                # We want the camera position in world (board) coordinates
                # Camera position in world = -R^T * t
                camera_pos_world = -R.T @ tvec.flatten()
                
                # Construct 4x4 transform matrix
                # This represents Camera -> World transform
                world_transform = np.eye(4, dtype=np.float32)
                world_transform[:3, :3] = R.T  # Rotation: Camera -> World
                world_transform[:3, 3] = camera_pos_world  # Translation
                
                print()
                print("=" * 50)
                print("EXTRINSIC CALIBRATION RESULT")
                print("=" * 50)
                print(f"Camera Position (in World/Board coords):")
                print(f"   X: {camera_pos_world[0]*1000:.1f} mm")
                print(f"   Y: {camera_pos_world[1]*1000:.1f} mm")
                print(f"   Z: {camera_pos_world[2]*1000:.1f} mm")
                print()
                
                # Update TriangulationEngine
                system.triangulation.camera_position = camera_pos_world
                
                # Save to calibration.json
                # Load existing, update, save
                try:
                    with open("calibration.json", "r") as f:
                        calib_data = json.load(f)
                except:
                    calib_data = {}
                
                calib_data["camera_position"] = camera_pos_world.tolist()
                calib_data["world_transform"] = world_transform.tolist()
                calib_data["extrinsic_rvec"] = rvec.flatten().tolist()
                calib_data["extrinsic_tvec"] = tvec.flatten().tolist()
                
                with open("calibration.json", "w") as f:
                    json.dump(calib_data, f, indent=2)
                
                print("Saved to calibration.json")
                print("=" * 50)
                
                # Flash
                cv2.imshow("Extrinsic Calibration", np.ones_like(display) * 255)
                cv2.waitKey(200)
                
                break
            else:
                print("Cannot calibrate: Board not detected properly!")
    
    system.camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
