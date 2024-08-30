import cv2
import numpy as np
import cv2.aruco as aruco
import pickle

# Load the camera calibration results (camera matrix and distortion coefficients)
with open('camera_calibration.pkl', 'rb') as f:
    calibration_data = pickle.load(f)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']

# Define the ArUco dictionary and parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Load a 3D object model (for simplicity, we'll use a cube in this example)
def draw_cube(img, corners, rvec, tvec, camera_matrix, dist_coeffs):
    axis = np.float32([
        [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
        [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]
    ]).reshape(-1, 3)

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)

    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw the base of the cube
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 3)

    # Draw pillars of the cube
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3)

    # Draw top of the cube
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img

# Capture video from the camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the ArUco markers in the image
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if np.all(ids is not None):
        # Estimate the pose of each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            # Draw the marker and axis
            aruco.drawDetectedMarkers(frame, corners)
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

            # Draw a cube or any 3D object on the marker
            frame = draw_cube(frame, corners[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

    # Display the resulting frame
    cv2.imshow('AR with ArUco Markers', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
