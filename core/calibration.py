import cv2
import cv2.aruco as aruco
import numpy as np
import core.config as config
from core.utils import get_laser_mask, intersect_ray_plane
import os

class CalibrationManager:
    def __init__(self):
        self.matrix = None
        self.dist_coeffs = None
        self.plane_equation = None
        self.load_calibration()
        
        # ChArUco Board Definition
        self.board = aruco.CharucoBoard(
            config.BOARD_SIZE, 
            config.SQUARE_LENGTH, 
            config.MARKER_LENGTH, 
            config.ARUCO_DICT
        )
        
        # OpenCV 4.8+ CharucoDetector
        self.charuco_detector = aruco.CharucoDetector(self.board)

    def load_calibration(self):
        """Load both camera and laser calibration data"""
        try:
            if os.path.exists(config.CAMERA_DATA_PATH):
                cam_data = np.load(config.CAMERA_DATA_PATH)
                self.matrix = cam_data["mtx"]
                self.dist_coeffs = cam_data["dist"]
                print("✅ Camera Calibration loaded.")
            
            if os.path.exists(config.LASER_DATA_PATH):
                laser_data = np.load(config.LASER_DATA_PATH)
                self.plane_equation = laser_data["plane"]
                print("✅ Laser Calibration loaded.")
                
            # Load Extrinsics (World Transform) from calibration.json
            import json
            if os.path.exists(config.EXTRINSIC_DATA_PATH):
                with open(config.EXTRINSIC_DATA_PATH, 'r') as f:
                    ext_data = json.load(f)
                    if 'world_transform' in ext_data:
                        self.world_transform = np.array(ext_data['world_transform'], dtype=np.float32)
                        print("✅ Extrinsic Calibration (World Transform) loaded from calibration.json")
                    else:
                        self.world_transform = np.eye(4, dtype=np.float32)
            else:
                self.world_transform = np.eye(4, dtype=np.float32)
                print("⚠️ No Extrinsic Calibration found (using Identity). Run debug_extrinsics.py to calibrate.")

        except Exception as e:
            print(f"⚠️ Calibration load error: {e}")

    # --- HELPER: TRIANGULATE ---
    def triangulate(self, points_2d, angle_deg):
        """
        Triangulate 2D points to 3D using Laser Plane and World Transform.
        Also handles Turntable Rotation.
        """
        if not points_2d: return []
        if self.matrix is None or self.plane_equation is None: return []
        
        points_3d = []
        theta = np.radians(angle_deg)
        
        # Rotation Matrix (Y-axis rotation for turntable? NO. Turntable rotates around Y usually in World Frame)
        # But our World Frame has Z-Up (if we fixed it).
        # Standard Turntable rotates around Vertical Axis.
        # If we aligned Y to be Up, then we rotate around Y.
        # If we aligned Z to be Up, then we rotate around Z.
        
        # In debug_extrinsics logic: 
        # We aligned Green (Y) to be Up.
        # So we rotate around Y.
        
        # WAIT. In Next/calibration.py, the triangulation loop uses `R` defined as:
        # [ cos 0 sin ]
        # [ 0   1 0   ]
        # [ -sin 0 cos ]
        # This IS rotation around Y!
        
        # Check alignment:
        # If Y is Up, then [x, y, z] -> [x', y, z']
        
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        inv_camera = np.linalg.inv(self.matrix)
        
        # Prepare World Transform
        # Use Identity if not loaded
        T_world = getattr(self, 'world_transform', np.eye(4, dtype=np.float32))
        
        # Plane Params
        n = self.plane_equation[:3]
        d = self.plane_equation[3]
        
        for (u, v) in points_2d:
            # Ray in camera space
            ray_cam = np.array([u, v, 1.0])
            ray_world = inv_camera @ ray_cam
            ray_world /= np.linalg.norm(ray_world)
            
            # Intersection with plane
            denom = np.dot(n, ray_world)
            if abs(denom) > 1e-6:
                t = -d / denom
                p_cam = ray_world * t
                
                # Transform to World Frame (ChArUco Origin)
                p_cam_homogenous = np.append(p_cam, 1.0)
                p_world_homogenous = T_world @ p_cam_homogenous
                p_world = p_world_homogenous[:3]
                
                # Apply Turntable Rotation (Object rotates relative to sensor, or we reconstruct object)
                # We rotate p_world by angle.
                p_obj = R @ p_world
                
                # Radius Filter (optional)
                # r_sq = p_obj[0]**2 + p_obj[2]**2
                # if r_sq < 3600: ...
                
                # --- APPLY USER AXIS MAPPING (POV CONTROL) ---
                # Default [0, 1, 2] means x=x, y=y, z=z
                axis_map = getattr(config, 'POINT_CLOUD_AXIS_MAP', [0, 1, 2])
                axis_inv = getattr(config, 'POINT_CLOUD_AXIS_INVERT', [1, 1, 1])
                
                # 1. Map Axes
                p_mapped = [
                    p_obj[axis_map[0]],
                    p_obj[axis_map[1]],
                    p_obj[axis_map[2]]
                ]
                
                # 2. Invert
                p_final = [
                    p_mapped[0] * axis_inv[0],
                    p_mapped[1] * axis_inv[1],
                    p_mapped[2] * axis_inv[2]
                ]

                # --- APPLY ROTATION OFFSET (TILT) ---
                # Rotasi manual untuk memperbaiki kemiringan.
                # Urutan: Rotate X -> Rotate Y -> Rotate Z
                rot_x = np.radians(getattr(config, 'SCAN_ROTATION_OFFSET', [0, 0, 0])[0])
                rot_y = np.radians(getattr(config, 'SCAN_ROTATION_OFFSET', [0, 0, 0])[1])
                rot_z = np.radians(getattr(config, 'SCAN_ROTATION_OFFSET', [0, 0, 0])[2])

                # Rotate X
                if rot_x != 0:
                    y = p_final[1]
                    z = p_final[2]
                    p_final[1] = y * np.cos(rot_x) - z * np.sin(rot_x)
                    p_final[2] = y * np.sin(rot_x) + z * np.cos(rot_x)

                # Rotate Y
                if rot_y != 0:
                    x = p_final[0]
                    z = p_final[2]
                    p_final[0] = x * np.cos(rot_y) + z * np.sin(rot_y)
                    p_final[2] = -x * np.sin(rot_y) + z * np.cos(rot_y)

                # Rotate Z
                if rot_z != 0:
                    x = p_final[0]
                    y = p_final[1]
                    p_final[0] = x * np.cos(rot_z) - y * np.sin(rot_z)
                    p_final[1] = x * np.sin(rot_z) + y * np.cos(rot_z)

                # 3. Apply Offsets (Translation)
                # Apply after rotation/mapping so it moves the final result in the desired direction.
                off_x = getattr(config, 'SCAN_X_OFFSET', 0.0)
                off_y = getattr(config, 'SCAN_Y_OFFSET', 0.0)
                off_z = getattr(config, 'SCAN_Z_OFFSET', 0.0)

                p_final[0] += off_x
                p_final[1] += off_y
                p_final[2] += off_z
                
                points_3d.append(p_final)
                
        return points_3d

    # --- STAGE 1: CAMERA CALIBRATION ---
    def calibrate_camera(self, frames, progress_callback=None):
        """
        Calibrate Intrinsics using ChArUco
        """
        print("📷 Processing Camera Calibration...")
        all_charuco_corners = []
        all_charuco_ids = []
        
        valid_frames = 0
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use CharucoDetector (handles both markers and board interpolation)
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                valid_frames += 1

            if progress_callback:
                progress_callback((i + 1) / len(frames) * 100)
        
        if valid_frames < 3:
            print("❌ Not enough valid frames for calibration (Need > 3)")
            return None, None
            
        print(f"   Using {valid_frames} valid frames.")
        
        print(f"   Using {valid_frames} valid frames.")
        
        try:
            # Prepare data for standard calibration
            all_obj_points = []
            all_img_points = []
            
            w, h = frames[0].shape[1], frames[0].shape[0]
            
            for i in range(len(all_charuco_ids)):
                # Get object points for the detected IDs from the board
                # Current OpenCV 4.8 CharucoBoard might not have nice helper, 
                # but we can construct them or look for board.objPoints? 
                # Actually, calibrateCameraCharuco was automating this.
                
                # Let's try to use the function if it exists, otherwise fallback?
                # No, user confirms it crashes.
                
                # Let's use board.getChessboardCorners() for ALL, effectively? No.
                # We need points corresponding to IDs.
                
                # METHOD: board.matchImagePoints (available in Board class)
                # But wait, CharucoDetector.detectBoard already gives us corners/ids.
                
                # We need 3D coords. 
                # board.getObjPoints() ??
                pass
            
            # REVISION: In OpenCV 4.x, calibrateCameraCharuco IS standard. 
            # If it's missing, maybe it's `aruco.calibrateCameraCharuco`?
            # User error showed it MISSING.
            
            # Let's try importing it safely?
            
            # Alternative: Standard cv2.calibrateCamera requires object points.
            # We can generate them. 
            # The board is 5x5.
            # ID 0 is at specific location.
            # We can ask the board for the object points.
            
            # Actually, `board.matchImagePoints` is deprecated/removed too?
            
            # Let's just use `cv2.aruco.calibrateCameraCharuco` but check generic `cv2`?
            # Or use `cv2.projectPoints` logic?
            
            # WAIT. If `calibrateCameraCharuco` is missing, implies we should use
            # `board.matchImagePoints` to get `objPoints` and `imgPoints` and feed to `cv2.calibrateCamera`.
            
            # Let's try:
            # ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(...)
            
            # If that fails, we do:
            pass
            
            # IMPLEMENTATION:
            # We construct the list of object points and image points.
            current_obj_points_list = []
            current_img_points_list = []
            
            all_board_corners = self.board.getChessboardCorners()
            
            for c_corners, c_ids in zip(all_charuco_corners, all_charuco_ids):
                # c_corners is (N, 1, 2)
                # c_ids is (N, 1)
                
                obj_pts = []
                img_pts = []
                
                for j, cid in enumerate(c_ids):
                    id_val = int(cid[0])
                    # select based on ID
                    if id_val < len(all_board_corners):
                        obj_pts.append(all_board_corners[id_val])
                        img_pts.append(c_corners[j])
                
                if len(obj_pts) > 0:
                    current_obj_points_list.append(np.array(obj_pts, dtype=np.float32).reshape(-1, 3))
                    current_img_points_list.append(np.array(img_pts, dtype=np.float32).reshape(-1, 2))
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                current_obj_points_list, 
                current_img_points_list, 
                gray.shape[::-1], 
                None, None
            )

            if ret:
                calibration_error = ret # RMSE
                self.matrix = mtx
                self.dist_coeffs = dist
                # Save to disk
                np.savez(config.CAMERA_DATA_PATH, mtx=mtx, dist=dist)
                print(f"✅ Camera Calibration Saved! Error: {calibration_error:.4f} px")
                return mtx, dist, calibration_error
            else:
                print("❌ Calibration failed (cv2 return false)")
                return None, None, None
        except Exception as e:
            print(f"❌ Calibration Exception: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            return None, None, None

    # --- STAGE 2: LASER PLANE CALIBRATION ---
    def calibrate_laser(self, frames, progress_callback=None):
        """
        Calibrate Laser Plane using ChArUco pose and Laser Line intersection
        """
        if self.matrix is None:
            print("❌ Cannot calibrate laser: Camera not calibrated!")
            return None

        print("🔴 Processing Laser Calibration...")
        laser_points_3d = []
        all_board_corners = self.board.getChessboardCorners()
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use CharucoDetector
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                valid = False
                rvec = None
                tvec = None
                
                # Try standard estimatePose first
                try:
                    valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
                        charuco_corners, charuco_ids, self.board, 
                        self.matrix, self.dist_coeffs, None, None
                    )
                except AttributeError:
                    # Fallback to solvePnP
                    obj_pts = []
                    img_pts = []
                    for j, cid in enumerate(charuco_ids):
                        id_val = int(cid[0])
                        if id_val < len(all_board_corners):
                            obj_pts.append(all_board_corners[id_val])
                            img_pts.append(charuco_corners[j])
                    
                    if len(obj_pts) >= 4:
                        obj_pts_arr = np.array(obj_pts, dtype=np.float32).reshape(-1, 3)
                        img_pts_arr = np.array(img_pts, dtype=np.float32).reshape(-1, 2)
                        
                        valid, rvec, tvec = cv2.solvePnP(
                            obj_pts_arr, img_pts_arr, 
                            self.matrix, self.dist_coeffs
                        )
                
                if valid:
                    # 2. Board Plane Geometry
                    R_board, _ = cv2.Rodrigues(rvec)
                    board_normal_cam = R_board @ np.array([0, 0, 1])
                    board_point_cam = tvec.flatten()
                    
                    # 3. Detect Laser
                    mask = get_laser_mask(frame)
                    y_idxs, x_idxs = np.where(mask > 0)
                    
                    # Subsample for speed
                    step = 5
                    for i in range(0, len(x_idxs), step):
                        u, v = x_idxs[i], y_idxs[i]
                        
                        # Create Ray
                        # Undistort point first
                        src_pt = np.array([[[u, v]]], dtype=np.float32)
                        undistorted = cv2.undistortPoints(src_pt, self.matrix, self.dist_coeffs, P=self.matrix)
                        u_n, v_n = undistorted[0][0]
                        
                        ray_x = (u_n - self.matrix[0,2]) / self.matrix[0,0]
                        ray_y = (v_n - self.matrix[1,2]) / self.matrix[1,1]
                        ray_vec = np.array([ray_x, ray_y, 1.0])
                        ray_vec = ray_vec / np.linalg.norm(ray_vec)
                        
                        # Intersect Ray with Board Plane
                        denom = np.dot(ray_vec, board_normal_cam)
                        if abs(denom) > 1e-6:
                            t = np.dot(board_point_cam, board_normal_cam) / denom # Assuming Ray Origin is (0,0,0) so (P-O).N / D.N -> P.N/D.N
                            # Wait, intersect_ray_plane function uses Plane Eq (Normal + d). 
                            # Let's use the explicit logic here or adapt.
                            # Logic from calibrate_laser.py:
                            # t = np.dot(plane_point - ray_origin, plane_normal) / denom
                            # ray_origin = 0
                            t_val = np.dot(board_point_cam, board_normal_cam) / denom
                            
                            if t_val > 0:
                                p3d = t_val * ray_vec
                                laser_points_3d.append(p3d)

            if progress_callback:
                progress_callback((i + 1) / len(frames) * 100)

        print(f"   Collected {len(laser_points_3d)} 3D points on laser plane.")
        
        if len(laser_points_3d) < 50:
            print("❌ Not enough points for plane fitting.")
            return None
            
        # 4. Plane Fitting (SVD)
        points = np.array(laser_points_3d)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov = np.dot(centered.T, centered)
        u, s, vh = np.linalg.svd(cov)
        normal = vh[2, :]
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        
        plane_eq = np.array([normal[0], normal[1], normal[2], d])
        self.plane_equation = plane_eq
        
        # Save
        np.savez(config.LASER_DATA_PATH, plane=plane_eq)
        print(f"✅ Laser Plane Calibrated: {plane_eq}")
        return plane_eq
