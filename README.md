# TactiCal - Tactile-Assisted Hand-Eye Calibration

TactiCal introduces a novel approach to hand-eye calibration by incorporating tactile feedback along with visual data. This method utilizes a DIGIT tactile sensor and a static camera, with a free-standing 3d-printed ArUco marker for flexible calibration. Unlike traditional methods that assume rigid configurations, TactiCal enables more accurate and robust calibration in dynamic environments.

## Key Features:
- **Tactile verification**: Uses tactile feedback to verify marker contact, ensuring high accuracy.
- **Free-standing marker**: The ArUco marker is not attached to the robot or workspace, offering more setup flexibility.
- **External camera setup**: The camera remains static, avoiding interference during calibration.

## Files:
- `cameraIntrinsicCalibration.py`: Captures frames to calibrate camera intrinsic parameters using an April grid.
- `cameraIntrinsicCalibration_test.py`: Verifies camera calibration accuracy.
- `coordinatesOfAruco.py`: Detects ArUco marker pose and coordinates.
- `coordinatesOfXarm.py`: Captures the xArm end-effector pose for validation.
- `TactiCal.py`: Implements the tactile-assisted hand-eye calibration process.
- `TactiCal_verification.py`: Tests the calibration results and verifies accuracy.

## Requirements:
- `pyrealsense2`
- `numpy`
- `cv2`
- `scipy`
- `xarm`

## Usage:
Run the TactiCal script to capture calibration data, compute the transformation, and verify the results with tactile feedback.

