import cv2
import numpy as np
import time
from core.system import SystemController

def main():
    print("📷 NetraScan-3D: MANUAL CAMERA CALIBRATION")
    print("==========================================")
    print("This script helps you calibrate the camera Intrinsic Parameters.")
    print("INSTRUCTIONS:")
    print("1. Use the ChArUco board (12.5x12.5cm).")
    print("2. Place it in various positions and angles covering the view.")
    print("3. Press [SPACE] to capture valid frames.")
    print("4. Capture at least 15-20 frames.")
    print("5. Press [C] to Calibrate and Save.")
    print("6. Press [Q] to Quit.")
    
    # Initialize System
    system = SystemController.get_instance()
    
    # Ensure camera is open
    if not system.camera.cap or not system.camera.cap.isOpened():
        print("📷 Connecting to camera...")
        system.camera.connect()
        # Warmup
        for _ in range(20): system.camera.get_frame()
        
    frames = []
    
    print("\n🎥 LIVE FEED STARTED")
    
    while True:
        frame = system.camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        display = frame.copy()
        
        # Detection Visualization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = system.auto_calib.charuco_detector.detectBoard(gray)
        
        status_color = (0, 0, 255)
        status_text = "No Board"
        
        if ids is not None and len(ids) > 6:
            status_color = (0, 255, 0)
            status_text = f"Valid ({len(ids)} markers)"
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids, (0, 255, 0))
        
        # UI
        cv2.putText(display, f"Captured: {len(frames)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(display, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display, "[SPACE] Capture | [C] Calibrate | [Q] Quit", (20, display.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.imshow("Camera Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == 32: # SPACE
            if ids is not None and len(ids) > 6:
                frames.append(frame.copy())
                print(f"📸 Captured Frame {len(frames)}")
                # Flash
                cv2.imshow("Camera Calibration", np.ones_like(display)*255)
                cv2.waitKey(50)
            else:
                print("⚠️ Cannot capture: Board not detected well enough.")
        elif key == ord('c'):
            if len(frames) < 10:
                print("⚠️ Need at least 10 frames (recommended 20+).")
            else:
                print("🧮 Running Calibration...")
                cv2.destroyAllWindows()
                
                mtx, dist = system.auto_calib.calibrate_camera_charuco(frames)
                
                if mtx is not None:
                    print("✅ Calibration Success!")
                    print(f"   Matrix:\n{mtx}")
                    print(f"   Dist:\n{dist}")
                    
                    # Save through TriangulationEngine (Central storage)
                    system.triangulation.camera_matrix = mtx
                    system.triangulation.dist_coeffs = dist
                    system.triangulation.calibrated = True
                    system.triangulation.save_calibration("calibration.json")
                    
                    print("💾 Saved to calibration.json")
                else:
                    print("❌ Calibration Failed.")
                
                break
                
    system.camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
