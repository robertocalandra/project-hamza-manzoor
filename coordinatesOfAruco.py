# to detect the coordinates and the pose of the aruco marker

import cv2
import numpy as np
import pyrealsense2 as rs
import cv2.aruco as aruco

# ===== Configuration =====
MARKER_SIZE = 0.1  # 5x5 cm marker (must match actual printed size)
ARUCO_DICT = aruco.DICT_5X5_1000  # Must match the marker type used

# Load camera calibration data
calib_data = np.load("camera_calibration.npz", allow_pickle=True)
mtx, dist = calib_data["mtx"], calib_data["dist"]

# ===== RealSense Setup =====
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# ===== ArUco Setup =====
aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Marker corner points (3D)
obj_points = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],  # Top-left
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],  # Top-right
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],  # Bottom-right
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]   # Bottom-left
], dtype=np.float32)

try:
    while True:
        # Get frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        frame = np.asanyarray(color_frame.get_data())

        # Detect markers
        corners, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            for i in range(len(ids)):
                # Estimate pose
                _, rvec, tvec = cv2.solvePnP(obj_points, corners[i], mtx, dist)

                # Draw marker and axes
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, MARKER_SIZE/2)
                aruco.drawDetectedMarkers(frame, corners, ids)

                # Display coordinates (convert to cm)
                x, y, z = tvec.flatten() * 100  # Convert to cm
                cv2.putText(frame, f"ID {ids[i][0]}", (10, 30 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"X: {x:+.1f} cm", (10, 60 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, f"Y: {y:+.1f} cm", (10, 90 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Z: {z:+.1f} cm", (10, 120 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("ArUco Marker Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()