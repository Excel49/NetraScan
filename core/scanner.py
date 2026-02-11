import cv2
import numpy as np
import time
import core.config as config
from core.utils import get_laser_mask

class Scanner:
    def __init__(self, system_controller):
        self.system = system_controller
        self.is_scanning = False
        self.points = []
    
    def _transform_points(self, points_3d):
        """
        Apply config transforms to raw triangulated points:
        1. SCAN_ROTATION_OFFSET (rotate by degrees around X, Y, Z)
        2. POINT_CLOUD_AXIS_MAP (remap axes)
        3. POINT_CLOUD_AXIS_INVERT (flip axes)
        4. SCAN_X/Y/Z_OFFSET (translate)
        """
        if not points_3d:
            return points_3d
        
        pts = np.array(points_3d, dtype=np.float64)  # Nx3
        
        # --- 1. Rotation Offset ---
        rx, ry, rz = [np.radians(a) for a in config.SCAN_ROTATION_OFFSET]
        
        # Rotation matrix around X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx),  np.cos(rx)]
        ])
        # Rotation matrix around Y
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        # Rotation matrix around Z
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx  # Combined rotation
        pts = (R @ pts.T).T  # Apply rotation
        
        # --- 2. Axis Remap ---
        axis_map = config.POINT_CLOUD_AXIS_MAP
        pts = pts[:, axis_map]  # Remap columns
        
        # --- 3. Axis Invert ---
        invert = np.array(config.POINT_CLOUD_AXIS_INVERT, dtype=np.float64)
        pts = pts * invert
        
        # --- 4. Translation Offset ---
        offset = np.array([config.SCAN_X_OFFSET, config.SCAN_Y_OFFSET, config.SCAN_Z_OFFSET])
        pts = pts + offset
        
        return pts.tolist()

    def start_scan(self, callback_progress=None, callback_data=None):
        """
        Main Scanning Loop (Ported from NetraScan-Next)
        """
        if self.is_scanning:
            print("⚠️ Already scanning.")
            return []

        # Check Calibration
        if not self.system.triangulation.calibrated:
            print("⚠️ Warning: System not fully calibrated. Using default parameters.")

        print("🚀 Starting 3D Scan (TriangulationEngine)...")
        self.is_scanning = True
        self.system.is_scanning = True
        self.points = []
        
        try:
            # 1. Home Turntable
            print("   Homing turntable...")
            if self.system.turntable.connected:
                self.system.turntable.home()
                time.sleep(1)
            
            # 2. Scan Loop
            total_steps = config.SCAN_STEPS_TOTAL
            step_increment = 360.0 / total_steps  # Degrees per step
            
            for i in range(total_steps):
                if not self.is_scanning: 
                    print("   🛑 Scan stopped by user.")
                    break
                
                # a. Capture Frame
                # Wait for stability
                time.sleep(0.4) 
                frame = self.system.camera.get_frame()
                
                if frame is not None:
                    # b. Detect Laser
                    laser_points = self.system.laser.detect(frame)
                    
                    # c. Triangulate
                    scan_angle = i * step_increment
                    
                    # Apply INVERT_ROTATION
                    if config.INVERT_ROTATION:
                        scan_angle = -scan_angle
                    
                    points_3d = self.system.triangulation.triangulate_points(laser_points, scan_angle)
                    
                    if points_3d:
                        # d. Apply config transforms (rotation offset, axis map, invert, translation)
                        transformed = self._transform_points(points_3d)
                        
                        # Add color (Green for laser)
                        colored_points = [[p[0], p[1], p[2], 0, 255, 0] for p in transformed]
                        self.points.extend(colored_points)
                        
                        # e. Emit Data
                        if callback_data:
                            callback_data(colored_points)
                            
                        # DEBUG
                        if i % 10 == 0:
                            print(f"   Step {i}: Found {len(points_3d)} points at {abs(scan_angle):.1f}°")

                # f. Rotate
                if self.system.turntable.connected:
                    self.system.turntable.rotate_by(step_increment)
                else:
                    time.sleep(0.05) # Simulation
                
                # g. Report Progress
                if callback_progress:
                    callback_progress(i, total_steps, len(self.points))
            
            print(f"🏁 Scan complete. Total Points: {len(self.points)}")
            
        except Exception as e:
            print(f"❌ Error during scan: {e}")
            raise e 
            
        finally:
            self.is_scanning = False
            self.system.is_scanning = False
            
        return self.points

