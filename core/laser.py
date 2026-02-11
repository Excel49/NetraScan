import cv2
import numpy as np

class LaserDetector:
    def __init__(self):
        self.threshold = 200
        self.min_width = 3
        self.max_width = 50
        
    def detect(self, frame):
        """
        Detect laser line in frame using adaptive thresholding
        Returns list of (x, y) pixel coordinates
        """
        if frame is None: return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find non-zero pixels
        points = cv2.findNonZero(binary)
        
        if points is None:
            return []
        
        # Extract unique x positions (avoid duplicate columns)
        points_dict = {}
        for point in points:
            x, y = point[0]
            # Keep highest y for each x (lowest point in image, matching 3d_scanner_complete logic)
            if x not in points_dict or y > points_dict[x]:
                points_dict[x] = y
        
        # Convert to list and limit points
        laser_points = [(x, points_dict[x]) for x in sorted(points_dict.keys())]
        
        # Downsample if too many points
        if len(laser_points) > 1000:
            step = len(laser_points) // 500
            laser_points = laser_points[::step]
        
        return laser_points

    def set_threshold(self, val):
        self.threshold = val
