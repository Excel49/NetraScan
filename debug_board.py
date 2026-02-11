
import cv2
import cv2.aruco as aruco
import numpy as np

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((5, 5), 0.025, 0.018, dictionary)

print("Checking board methods...")
try:
    corners = board.getChessboardCorners()
    print(f"✅ getChessboardCorners exists. Shape: {corners.shape}")
    print(f"   Sample: {corners[0]}")
except AttributeError:
    print("❌ getChessboardCorners NOT found")

try:
    obj_pts = board.getObjPoints()
    print(f"✅ getObjPoints exists. Type: {type(obj_pts)}")
except AttributeError:
    print("❌ getObjPoints NOT found")
