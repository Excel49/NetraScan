
    def calibrate_laser(self, frames, camera_matrix, dist_coeffs, progress_callback=None):
        """
        Calibrate Laser Plane using ChArUco pose and Laser Line intersection
        Returns the plane equation (ax + by + cz + d = 0)
        """
        if camera_matrix is None or dist_coeffs is None:
            print("❌ Cannot calibrate laser: Missing Camera Parameters!")
            return None

        print("🔴 Processing Laser Calibration...")
        laser_points_3d = []
        all_board_corners = self.charuco_board.getChessboardCorners()
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use CharucoDetector
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                valid = False
                rvec = None
                tvec = None
                
                # Estimate Pose
                # Try standard estimatePoseCharucoBoard (might be in cv2.aruco or not depending on version)
                try:
                     # Check if we can use the board object directly
                     valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners, charuco_ids, self.charuco_board, 
                        camera_matrix, dist_coeffs, None, None
                    )
                except Exception:
                    # Fallback to solvePnP using manual obj-img points matching
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
                            camera_matrix, dist_coeffs
                        )

                if valid:
                    # 2. Board Plane Geometry (Normal and Point)
                    R_board, _ = cv2.Rodrigues(rvec)
                    # Board Z-axis is the normal in board coordinates? 
                    # Usually board is on XY plane (Z=0). Normal is Z axis (0,0,1).
                    board_normal_cam = R_board @ np.array([0, 0, 1])
                    board_point_cam = tvec.flatten()
                    
                    # 3. Detect Laser
                    mask = get_laser_mask(frame)
                    y_idxs, x_idxs = np.where(mask > 0)
                    
                    # Subsample for speed
                    step = 5
                    for k in range(0, len(x_idxs), step):
                        u, v = x_idxs[k], y_idxs[k]
                        
                        # Create Ray
                        # Undistort point first
                        src_pt = np.array([[[u, v]]], dtype=np.float32)
                        undistorted = cv2.undistortPoints(src_pt, camera_matrix, dist_coeffs, P=camera_matrix)
                        u_n, v_n = undistorted[0][0]
                        
                        ray_x = (u_n - camera_matrix[0,2]) / camera_matrix[0,0]
                        ray_y = (v_n - camera_matrix[1,2]) / camera_matrix[1,1]
                        ray_vec = np.array([ray_x, ray_y, 1.0])
                        ray_vec = ray_vec / np.linalg.norm(ray_vec)
                        
                        # Intersect Ray with Board Plane
                        # Plane defined by: (P - P0) . n = 0
                        # Ray: P = O + t*D (O=0 at camera center)
                        # (t*D - P0) . n = 0  => t*(D.n) - P0.n = 0 => t = (P0.n)/(D.n)
                        
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
            print("❌ Not enough points for plane fitting.")
            return None
            
        # 4. Plane Fitting (SVD)
        # We want to find a,b,c,d such that ax+by+cz+d=0
        points = np.array(laser_points_3d)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # SVD on covariance
        cov = np.dot(centered.T, centered)
        u, s, vh = np.linalg.svd(cov)
        normal = vh[2, :] # The eigenvector corresponding to the smallest eigenvalue
        
        # Normalize
        normal = normal / np.linalg.norm(normal)
        
        # d = - (Normal . Centroid)
        d = -np.dot(normal, centroid)
        
        plane_eq = np.array([normal[0], normal[1], normal[2], d])
        
        print(f"✅ Laser Plane Calibrated: {plane_eq}")
        return plane_eq
