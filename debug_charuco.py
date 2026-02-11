
import cv2
import cv2.aruco as aruco
import sys

print(f"OpenCV Version: {cv2.__version__}")

print("Checking Charuco attributes:")
attrs = dir(cv2.aruco)

functions_to_check = [
    'interpolateCornersCharuco',
    'CharucoDetector',
    'calibrateCameraCharuco',
    'estimatePoseCharucoBoard'
]

for func in functions_to_check:
    if func in attrs:
        print(f"✅ '{func}' FOUND")
    else:
        print(f"❌ '{func}' NOT FOUND")

if 'CharucoDetector' in attrs:
    print("\n💡 Recommendation: Use 'CharucoDetector.detectBoard()' instead of interpolateCorners.")
