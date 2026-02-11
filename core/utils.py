import cv2
import numpy as np
import core.config as config

def get_laser_mask(frame, custom_threshold=None):
    """
    Detect laser pixels using improved methods.
    Supports simple Channel Difference or HSV if needed.
    """
    threshold = custom_threshold if custom_threshold is not None else config.LASER_THRESHOLD
    
    # Method 1: Improved Channel Difference (Weighted)
    # Give more weight to Red channel relative to others
    # diff = R - max(G, B)  <-- This is stricter than avg(G,B)
    b, g, r = cv2.split(frame)
    
    # Use float for calculation
    r = r.astype(float)
    g = g.astype(float)
    b = b.astype(float)
    
    # Stricter Difference: Red must be significantly brighter than BOTH Green and Blue
    # diff = r - np.maximum(g, b) 
    
    # Original Formula (Reference):
    diff = r - (g + b) / 2.0
    
    # Additional Check: Saturation Handling
    # If pixel is VERY bright (near white), it might be the laser center.
    # But white implies R, G, B are all high.
    # We want to keep pixels where R is high, even if G/B are moderately high (saturation).
    # But we want to reject white paper / board background.
    
    # Let's stick to the difference but boost the result?
    diff[diff < 0] = 0
    diff = diff.astype(np.uint8)
    
    # Thresholding
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Method 2: HSV Filtering (Optional / Hybrid)
    # Often better for color isolation
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_red1 = np.array([0, 100, 100])
    # upper_red1 = np.array([10, 255, 255])
    # lower_red2 = np.array([160, 100, 100])
    # upper_red2 = np.array([180, 255, 255])
    # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # mask_hsv = cv2.bitwise_or(mask1, mask2)
    
    # For now, stick to the Reference Logic (Channel Diff) as base,
    # but maybe we can add a 'Brightness' boost for the mask?
    
    # Clean up noise
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Morphological Open to remove small speckles
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def get_laser_subpixel(frame, threshold=None):
    """
    Detect laser line with Sub-pixel Accuracy using Center of Mass (Centroid).
    Returns a list of (x, y) tuples where x is float.
    """
    if threshold is None: threshold = config.LASER_THRESHOLD
    
    # 1. Get rough mask first (to filter noise)
    mask = get_laser_mask(frame, custom_threshold=threshold)
    
    # 2. Extract Red channel for intensity weighting
    # We use the raw intensity to calculate the weighted average position
    _, _, r = cv2.split(frame)
    
    points = []
    h, w = mask.shape
    
    # Iterate rows
    # Optimization: Skip every 2nd line handled in scanner, but here we can return all valid
    for y in range(0, h, 2):
        row_mask = mask[y, :]
        if np.max(row_mask) > 0:
            # Find rough peak (integer)
            max_x = np.argmax(row_mask)
            
            # Sub-pixel Refinement (Center of Mass)
            # Define window: [max_x - window, max_x + window]
            window = 3 
            start = max(0, max_x - window)
            end = min(w, max_x + window + 1)
            
            # Get intensities from Red channel (or the Difference image for cleaner signal?)
            # Let's use Red channel for now, assuming laser is brightest red.
            # Or use the 'diff' from get_laser_mask? That would be cleaner.
            # To avoid re-calculating diff, let's just use R channel but masked.
            
            # Better: Calculate local centroid on R channel
            intensities = r[y, start:end].astype(float)
            indices = np.arange(start, end).astype(float)
            
            total_intensity = np.sum(intensities)
            if total_intensity > 0:
                # Center of Mass = Sum(I * x) / Sum(I)
                subpixel_x = np.sum(indices * intensities) / total_intensity
                points.append((subpixel_x, y))
            else:
                # Fallback to integer
                points.append((float(max_x), y))
                
    return points

def intersect_ray_plane(ray_origin, ray_vector, plane_equation):
    """
    Calculate intersection between a Ray and a Plane.
    Plane: ax + by + cz + d = 0
    """
    normal = plane_equation[:3]
    d = plane_equation[3]
    
    denom = np.dot(normal, ray_vector)
    
    # Avoid division by zero (parallel)
    if abs(denom) < 1e-6:
        return None
        
    t = -(np.dot(normal, ray_origin) + d) / denom
    
    # Intersection behind camera
    if t < 0:
        return None
        
    point = ray_origin + t * ray_vector
    return point

def undistort_point(u, v, mtx, dist):
    """
    Convert pixel coordinates (u, v) to normalized camera coordinates.
    Returns (x, y) on the image plane (z=1).
    """
    src_pt = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(src_pt, mtx, dist, P=mtx)
    u_n, v_n = undistorted[0][0]
    
    # Convert to normalized ray vector (x, y, 1) direction
    ray_x = (u_n - mtx[0,2]) / mtx[0,0]
    ray_y = (v_n - mtx[1,2]) / mtx[1,1]
    
    return np.array([ray_x, ray_y, 1.0])

def draw_axes(img, camera_matrix, dist_coeffs, world_transform, length=0.1, extrinsic_rvec=None, extrinsic_tvec=None):
    """
    Draw 3D axes on the image.
    If extrinsic_rvec and extrinsic_tvec are provided, use them directly.
    Otherwise, derive from world_transform (Camera -> World) by inverting.
    """
    rvec = None
    tvec = None
    
    # Use direct rvec/tvec if available (from extrinsic calibration)
    if extrinsic_rvec is not None and extrinsic_tvec is not None:
        rvec = extrinsic_rvec
        tvec = extrinsic_tvec
    elif world_transform is not None:
        # Inverse to get World -> Camera
        try:
            T_world_to_cam = np.linalg.inv(world_transform)
            rvec, _ = cv2.Rodrigues(T_world_to_cam[:3, :3])
            tvec = T_world_to_cam[:3, 3]
        except np.linalg.LinAlgError:
            return img
    else:
        return img

    # Length in same units as calibration (Meters)
    axis_length = length 
    
    points_3d = np.float32([
        [0, 0, 0],          # Origin
        [axis_length, 0, 0], # X
        [0, axis_length, 0], # Y
        [0, 0, axis_length]  # Z
    ])

    try:
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        points_2d = np.int32(points_2d).reshape(-1, 2)

        origin = tuple(points_2d[0])
        
        # Draw lines
        img = cv2.line(img, origin, tuple(points_2d[1]), (0, 0, 255), 3) # X - Red
        img = cv2.line(img, origin, tuple(points_2d[2]), (0, 255, 0), 3) # Y - Green
        img = cv2.line(img, origin, tuple(points_2d[3]), (255, 0, 0), 3) # Z - Blue
    except Exception as e:
        # print(f"Error drawing axes: {e}")
        pass
    
    return img
