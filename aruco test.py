import cv2
import numpy as np
import cv2.aruco as aruco
import pickle

# Load the camera calibration results (camera matrix and distortion coefficients)
with open('camera_calibration.pkl', 'rb') as f:
    calibration_data = pickle.load(f)
    camera_matrix = calibration_data['mtx_left']  # Access the left camera matrix
    dist_coeffs = calibration_data['dist_left']   # Access the left distortion coefficients

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

# Load the overlay image (make sure it has an alpha channel)
overlay_img = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel

def overlay_on_marker(frame, corners, overlay_img):
    for corner in corners:
        # Get the marker's corners
        top_left, top_right, bottom_right, bottom_left = corner[0]

        # Calculate the width and height of the marker
        width = int(np.linalg.norm(top_right - top_left))
        height = int(np.linalg.norm(top_left - bottom_left))

        # Resize the overlay image to match the marker size
        resized_overlay = cv2.resize(overlay_img, (width, height))

        # Check if resized overlay has an alpha channel
        if resized_overlay.shape[2] == 4:
            alpha_channel = resized_overlay[:, :, 3] / 255.0
            overlay_color = resized_overlay[:, :, :3]
        else:
            alpha_channel = np.ones((height, width))
            overlay_color = resized_overlay

        # Create a mask for the overlay image
        overlay_mask = np.stack([alpha_channel] * 3, axis=-1)

        # Calculate the bounding box for the marker
        top_left = tuple(corner[0][0].astype(int))

        # Create a region of interest in the frame
        roi = frame[top_left[1]:top_left[1]+height, top_left[0]:top_left[0]+width]

        # Ensure the ROI and overlay image are the same size
        if roi.shape[:2] != (height, width):
            roi = cv2.resize(roi, (width, height))

        # Overlay the image on the ROI
        blended_roi = (1 - overlay_mask) * roi + overlay_mask * overlay_color
        frame[top_left[1]:top_left[1]+height, top_left[0]:top_left[0]+width] = blended_roi

    return frame

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

    if ids is not None:
        # Overlay the image on the detected markers
        frame = overlay_on_marker(frame, corners, overlay_img)

    # Display the resulting frame
    cv2.imshow('AR with ArUco Markers', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
