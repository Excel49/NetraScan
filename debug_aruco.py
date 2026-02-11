
import cv2
import cv2.aruco
import sys

print(f"Python Version: {sys.version}")
print(f"OpenCV Version: {cv2.__version__}")

try:
    print("Checking cv2.aruco attributes:")
    attrs = dir(cv2.aruco)
    if 'detectMarkers' in attrs:
        print("✅ 'detectMarkers' FOUND")
    else:
        print("❌ 'detectMarkers' NOT FOUND")

    if 'ArucoDetector' in attrs:
        print("✅ 'ArucoDetector' FOUND (New API)")
    else:
        print("❌ 'ArucoDetector' NOT FOUND")
        
except Exception as e:
    print(f"Error inspecting aruco: {e}")
