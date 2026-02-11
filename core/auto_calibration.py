import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from core.utils import get_laser_mask

class AutoCalibration:
    def __init__(self):
        # Checkerboard configuration (standard 9x6 inner corners)
        self.checkerboard_size = (9, 6)  
        self.square_size = 25.0  # mm 
        
        # ChArUco configuration for 12.5x12.5cm board
        # 5 squares * 25mm = 125mm = 12.5cm
        self.squares_x = 5
        self.squares_y = 5
        self.square_length = 0.025  # 25mm
        self.marker_length = 0.0125  # 12.5mm
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), 
            self.square_length, 
            self.marker_length, 
            self.aruco_dict
        )
        self.charuco_board.setLegacyPattern(True)
        
        self.charuco_params = cv2.aruco.CharucoParameters()
        self.detector_params = cv2.aruco.DetectorParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.charuco_board, 
            self.charuco_params, 
            self.detector_params
        )
        
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.calibration_results = {}
        self.calibrated = False

    def calibrate_camera_charuco(self, calibration_images):
        print("Starting ChArUco calibration...")
        all_charuco_corners = []
        all_charuco_ids = []
        
        valid_images = 0
        for i, img in enumerate(calibration_images):
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                valid_images += 1
                print(f"  Image {i+1}: Found {len(charuco_corners)} ChArUco corners")
            else:
                print(f"  Image {i+1}: No ChArUco board detected")

        if valid_images < 5:
            print("Not enough valid ChArUco images.")
            return None, None
            
        print(f"Calibrating with {valid_images} images...")
        
        try:
            all_board_corners = self.charuco_board.getChessboardCorners()
            objpoints = []
            imgpoints = []
            
            for i in range(len(all_charuco_corners)):
                current_corners = all_charuco_corners[i]
                current_ids = all_charuco_ids[i]
                
                if current_corners is None or current_ids is None:
                    continue
                    
                current_obj_points = []
                current_img_points = []
                
                for k in range(len(current_ids)):
                    point_id = int(current_ids[k])
                    if point_id < len(all_board_corners):
                        current_obj_points.append(all_board_corners[point_id])
                        current_img_points.append(current_corners[k])
                
                if len(current_obj_points) > 0:
                    objpoints.append(np.array(current_obj_points, dtype=np.float32))
                    imgpoints.append(np.array(current_img_points, dtype=np.float32))

            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            if ret:
                print(f"ChArUco calibration successful! Error: {ret:.3f}")
                self.calibrated = True
                self.calibration_results['camera_calibration'] = {
                    'camera_matrix': camera_matrix.tolist(),
                    'distortion_coefficients': dist_coeffs.tolist(),
                    'reprojection_error': ret,
                    'image_size': gray.shape[::-1],
                    'calibration_type': 'charuco',
                    'timestamp': datetime.now().isoformat()
                }
                return camera_matrix, dist_coeffs
            else:
                print("Calibration failed.")
                return None, None
                
        except Exception as e:
            print(f"Error during ChArUco calibration: {e}")
            return None, None

    def calibrate_laser(self, frames, camera_matrix, dist_coeffs, progress_callback=None):
        if camera_matrix is None or dist_coeffs is None:
            print("Cannot calibrate laser: Missing Camera Parameters!")
            return None

        print("Processing Laser Calibration...")
        laser_points_3d = []
        all_board_corners = self.charuco_board.getChessboardCorners()
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                valid = False
                rvec = None
                tvec = None
                
                try:
                    valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners, charuco_ids, self.charuco_board, 
                        camera_matrix, dist_coeffs, None, None
                    )
                except Exception:
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
                        valid, rvec, tvec = cv2.solvePnP(obj_pts_arr, img_pts_arr, camera_matrix, dist_coeffs)

                if valid:
                    R_board, _ = cv2.Rodrigues(rvec)
                    board_normal_cam = R_board @ np.array([0, 0, 1])
                    board_point_cam = tvec.flatten()
                    
                    mask = get_laser_mask(frame)
                    y_idxs, x_idxs = np.where(mask > 0)
                    
                    step = 5
                    for k in range(0, len(x_idxs), step):
                        u, v = x_idxs[k], y_idxs[k]
                        src_pt = np.array([[[u, v]]], dtype=np.float32)
                        undistorted = cv2.undistortPoints(src_pt, camera_matrix, dist_coeffs, P=camera_matrix)
                        u_n, v_n = undistorted[0][0]
                        
                        ray_x = (u_n - camera_matrix[0,2]) / camera_matrix[0,0]
                        ray_y = (v_n - camera_matrix[1,2]) / camera_matrix[1,1]
                        ray_vec = np.array([ray_x, ray_y, 1.0])
                        ray_vec = ray_vec / np.linalg.norm(ray_vec)
                        
                        denom = np.dot(ray_vec, board_normal_cam)
                        if abs(denom) > 1e-6:
                            t_val = np.dot(board_point_cam, board_normal_cam) / denom
                            if t_val > 0:
                                p3d = t_val * ray_vec
                                laser_points_3d.append(p3d)
            
            if progress_callback:
                progress_callback((i + 1) / len(frames) * 100)

        print(f"   Collected {len(laser_points_3d)} 3D points on laser plane.")
        
        if len(laser_points_3d) < 50:
            print("Not enough points for plane fitting.")
            return None
            
        points = np.array(laser_points_3d)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov = np.dot(centered.T, centered)
        u, s, vh = np.linalg.svd(cov)
        normal = vh[2, :]
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        
        plane_eq = np.array([normal[0], normal[1], normal[2], d])
        print(f"Laser Plane Calibrated: {plane_eq}")
        return plane_eq
