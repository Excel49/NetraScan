import numpy as np
import json
import logging
from datetime import datetime
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TriangulationEngine:
    """Performs 3D triangulation from laser line points"""
    
    def __init__(self):
        # Default calibration parameters
        self.camera_matrix = np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.laser_plane = np.array([0.0, -0.2, 1.0, -500.0])  # ax + by + cz + d = 0
        
        # Camera position relative to turntable center (mm)
        self.camera_position = np.array([0.0, 50.0, 50.0])
        self.dist_coeffs = np.zeros((5, 1)) # Default no distortion
        self.calibrated = False
        
        # Full world transform (Camera -> World) 4x4 matrix
        # Will be loaded from calibration or computed from camera_position
        self._world_transform = None
        
        # Direct extrinsic parameters (World -> Camera, from solvePnP)
        # These can be used directly for cv2.projectPoints
        self.extrinsic_rvec = None
        self.extrinsic_tvec = None

    @property
    def world_transform(self):
        """
        Return the 4x4 Camera -> World transform matrix.
        If explicitly set from extrinsic calibration, use that.
        Otherwise compute from camera_position (translation only).
        """
        if self._world_transform is not None:
            return self._world_transform
        
        # Fallback: compute from camera_position (translation only, no rotation)
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = self.camera_position
        return T

    def triangulate_points(self, pixel_points: List[Tuple[int, int]], 
                          angle_deg: float = 0.0) -> List[List[float]]:
        """
        Triangulate 2D pixel points to 3D coordinates
        Returns list of [x, y, z] coordinates in turntable coordinate system
        """
        if not pixel_points:
            return []
        
        points_3d = []
        angle_rad = np.radians(angle_deg)
        
        # Precompute rotation terms
        # Note: This compensates for turntable rotation
        cos_a = np.cos(-angle_rad)  # Negative for compensation
        sin_a = np.sin(-angle_rad)
        
        # Precompute camera matrix inverse
        inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        
        n = self.laser_plane[:3]
        d = self.laser_plane[3]

        for x, y in pixel_points:
            # Convert pixel to normalized camera coordinates
            pixel = np.array([x, y, 1.0], dtype=np.float32)
            ray_dir = inv_camera_matrix @ pixel
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            # Intersect with laser plane
            # dot(n, (start + t*dir)) + d = 0  => t = -(d + dot(n, start)) / dot(n, dir)
            # Assuming camera center is at (0,0,0) in camera coords, so start=0. 
            # t = -d / dot(n, dir)
            
            denom = np.dot(n, ray_dir)
            if abs(denom) < 1e-6:
                continue
                
            t = -d / denom
            point_camera = t * ray_dir
            
            # Apply camera offset to get to world/turntable base frame (before rotation)
            # point_world_static = point_camera + self.camera_position
            # Actually, usually calibration gives camera pose relative to turntable.
            # If camera_position is the location of camera center in turntable frame:
            # Point in turntable frame = R_cam * point_camera + T_cam
            # Here simplified as just translation for now based on previous code
            
            point_world = point_camera + self.camera_position
            
            # Rotate based on turntable angle
            # Rotation around Y axis (vertical in this coordinate system)
            x_rot = point_world[0] * cos_a - point_world[2] * sin_a
            y_rot = point_world[1]  # Y unchanged (height)
            z_rot = point_world[0] * sin_a + point_world[2] * cos_a
            
            points_3d.append([float(x_rot), float(y_rot), float(z_rot)])
        
        return points_3d
    
    def save_calibration(self, filepath: str):
        """Save calibration parameters to file"""
        calibration = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "laser_plane": self.laser_plane.tolist(),
            "camera_position": self.camera_position.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration, f, indent=2)
            logger.info(f"Calibration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
    
    def load_calibration(self, filepath: str):
        """Load calibration parameters from file"""
        try:
            with open(filepath, 'r') as f:
                calibration = json.load(f)
            
            self.camera_matrix = np.array(calibration["camera_matrix"])
            if "dist_coeffs" in calibration:
                self.dist_coeffs = np.array(calibration["dist_coeffs"])
            self.laser_plane = np.array(calibration["laser_plane"])
            self.camera_position = np.array(calibration["camera_position"])
            
            # Load full world transform if available (from extrinsic calibration)
            if "world_transform" in calibration:
                self._world_transform = np.array(calibration["world_transform"], dtype=np.float32)
                logger.info(f"World transform loaded: shape={self._world_transform.shape}")
            
            # Load direct extrinsic rvec/tvec for axis drawing
            if "extrinsic_rvec" in calibration:
                self.extrinsic_rvec = np.array(calibration["extrinsic_rvec"], dtype=np.float32)
            if "extrinsic_tvec" in calibration:
                self.extrinsic_tvec = np.array(calibration["extrinsic_tvec"], dtype=np.float32)
            
            self.calibrated = True
            
            logger.info(f"Calibration loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
