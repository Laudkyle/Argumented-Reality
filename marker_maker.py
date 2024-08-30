import cv2
import numpy as np
import cv2.aruco as aruco

# Define the dictionary and the ID of the marker
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
marker_id = 42  # ID of the marker you want to generate
marker_size = 200  # Size of the marker image in pixels

# Generate the marker
marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
marker_image = aruco.drawMarker(aruco_dict, marker_id, marker_size)

# Display the marker
cv2.imshow('ArUco Marker', marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the marker as an image file
cv2.imwrite(f'aruco_marker_{marker_id}.png', marker_image)
