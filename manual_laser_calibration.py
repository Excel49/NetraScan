import cv2
import numpy as np
import time
import os
from core.system import SystemController

def main():
    print("🔴 NetraScan-3D: Manual Laser Calibration Tool")
    print("=======================================")
    print("This script allows you to manually capture frames for Laser Plane Calibration.")
    print("INSTRUCTIONS:")
    print("1. Place ChArUco board in various positions (standing/tilted) where the laser line is visible.")
    print("2. Press [SPACE] to capture a frame.")
    print("3. Capture at least 10-15 frames covering different angles/depths.")
    print("4. Press [c] to calculates and save calibration.")
    print("5. Press [q] to quit without saving.")
    print("=======================================")

    # Initialize System
    print("\nInitializing System...")
    system = SystemController.get_instance()
    
    # Force Camera Init
    if system.camera.cap is None or not system.camera.cap.isOpened():
        print("📷 Opening Camera...")
        system.camera.connect()
        # Wait for auto-exposure
        for _ in range(30):
            system.camera.get_frame()

    if not system.camera.cap.isOpened():
        print("❌ Failed to open camera.")
        return

    # Load existing calibrations
    if not system.triangulation.load_calibration("calibration.json"):
        print("⚠️ Warning: Could not load calibration.json")
    
    # Check Camera Calibration
    if not system.triangulation.calibrated or system.triangulation.camera_matrix is None:
        print("❌ Camera not calibrated! Run 'python manual_camera_calibration.py' first.")
        return

    print(f"✅ Camera Calibration loaded.")

    frames = []
    
    print("\n🎥 LIVE FEED STARTED")
    print("   [SPACE] : Capture Frame")
    print("   [c]     : Calibrate & Save")
    print("   [q]     : Quit")

    while True:
        frame = system.camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        display = frame.copy()
        
        # Detect Board for visualization using AutoCalibration instance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = system.auto_calib.charuco_detector.detectBoard(gray)
        
        status_color = (0, 0, 255) # Red
        status_text = "No Board"
        
        if ids is not None and len(ids) > 4:
            status_color = (0, 255, 0) # Green
            status_text = "Board Detected"
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids, (0, 255, 0))
            
        # Draw UI
        cv2.putText(display, f"Frames Captured: {len(frames)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display, f"Status: {status_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display, "[SPACE] Capture | [c] Calibrate", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Manual Laser Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == 32: # SPACE
            if ids is not None and len(ids) > 4:
                frames.append(frame.copy())
                print(f"📸 Captured Frame {len(frames)}")
                # Flash effect
                cv2.imshow("Manual Laser Calibration", np.ones_like(display) * 255)
                cv2.waitKey(50)
            else:
                print("⚠️ Cannot capture: Board not visible or not enough markers.")
                
        elif key == ord('c'):
            if len(frames) < 5:
                print("⚠️ Not enough frames. Capture at least 5 frames.")
            else:
                print("\n🧮 Processing Laser Calibration (Plane Fitting)...")
                cv2.destroyAllWindows()
                
                # Pass Matrix and DistCoeffs explicitly
                plane = system.auto_calib.calibrate_laser(
                    frames, 
                    system.triangulation.camera_matrix,
                    system.triangulation.dist_coeffs,
                    lambda p: print(f"   Progress: {p:.1f}%")
                )
                
                if plane is not None:
                    print(f"\n✅ SUCCESS! Laser Plane detected: {plane}")
                    
                    # Update & Save
                    system.triangulation.laser_plane = plane
                    system.triangulation.save_calibration("calibration.json")
                    print("   Calibration saved to calibration.json")
                    
                    # Analyze orientation
                    normal = plane[:3]
                    # Check if vertical-ish (Y component small, X/Z large)
                    # Or if we use board axes...
                    print(f"   Normal Vector: {normal}")
                    if abs(normal[1]) > 0.8: 
                        print("   ⚠️ WARNING: Plane seems HORIZONTAL (Table-like).")
                        print("      Ensure laser line is VERTICAL and board was STANDING UP.")
                    else:
                        print("   ✅ Orientation looks good (Vertical-ish).")
                        
                else:
                    print("\n❌ FAILED. Could not detect laser plane.")
                
                break

    system.camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import cv2.aruco as aruco # helper
    main()
