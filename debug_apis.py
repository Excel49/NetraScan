
import cv2
import cv2.aruco as aruco

print(f"CV2 Version: {cv2.__version__}")
print("Searching for 'Charuco' in cv2.aruco:")
for x in dir(aruco):
    if 'Charuco' in x or 'charuco' in x:
        print(f" - aruco.{x}")

print("\nSearching for 'calibrate' in cv2.aruco:")
for x in dir(aruco):
    if 'calibrate' in x:
        print(f" - aruco.{x}")
