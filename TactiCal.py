# TactiCal - Touch-Based Hand-Eye Calibration
# -------------------------------------------
#
# Instructions on How to Use:
#
# - The script is self-guided and primarily runs through the terminal, with additional windows showing camera and DIGIT sensor outputs when needed.
# - Please note that inputs for the output windows are only registered when those windows are in focus (active).
# - The general workflow involves two main steps per sample:
#     1. Capture the pose of the ArUco marker using the camera.
#     2. Move the robot arm to physically touch the ArUco marker at the same position.
# - The ArUco marker remains fixed during each sample collection i.e., the camera captures its pose in the same position where the robot arm goes and touches it.
# - Both the robot base and the camera remain stationary at all times.

import time
import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R
from digit_interface import Digit

class HandEyeCalibrator:
    def __init__(self, arm_ip='192.168.1.227', marker_size=0.1):
        # Initialize xArm
        self.arm = XArmAPI(arm_ip)
        self.arm.connect()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)  # Position mode
        self.arm.set_state(0)  # Sport state

        self.digit = Digit("D20994")
        self.digit.connect()
        self.digit.set_fps(30)
        self.digit_running = True

        # # Start DIGIT view in a background thread
        # self.digit_thread = threading.Thread(target=self.show_digit_with_crosshair, daemon=True)
        # self.digit_thread.start()

        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # Load camera calibration
        calibration_data = np.load('camera_calibration.npz')
        self.camera_matrix = calibration_data['mtx']
        self.dist_coeffs = calibration_data['dist']

        # Aruco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_size = marker_size  # meters

        # Calibration storage
        self.T_base_cam = None
        self.last_marker_pose = None
        self.T_ee_marker = None

    def get_marker_pose(self):
        #Detect Aruco marker and return its pose in camera frame (T_cam_marker)
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("Error: No camera frame captured!")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # 1) Detect coarse markers
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.parameters)

            if ids is None:
                print("Error: No markers detected!")
                continue

            # 2) Refine corners to sub-pixel accuracy
            #   Flatten to Nx1x2 for cornerSubPix
            corners_subpix = []
            for c in corners:
                # c is 1×4×2 → 4×2 float32
                pts = c.reshape(-1, 2).astype(np.float32)
                # termination: either 30 iters or 0.01px
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                cv2.cornerSubPix(gray, pts, winSize=(5,5), zeroZone=(-1,-1), criteria=term)
                corners_subpix.append(pts.reshape(1, -1, 2))
            corners = corners_subpix

            # 3) Estimate pose using refined corners
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[0], self.marker_size, self.camera_matrix, self.dist_coeffs)

            T_marker_cam = np.eye(4)
            T_marker_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_marker_cam[:3, 3] = tvec[0][0]
            T_cam_marker = np.linalg.inv(T_marker_cam)

            euler = R.from_matrix(T_cam_marker[:3, :3]).as_euler('xyz', degrees=True)
            roll, pitch, yaw = euler


            # Print detection info
            print("\n=== Marker Detection Data ===")
            print(f"Marker ID: {ids[0][0]}")
            print(f"Position (m): X={tvec[0][0][0]:.4f}, Y={tvec[0][0][1]:.4f}, Z={tvec[0][0][2]:.4f}")
            print("Rotation Matrix:")
            print(np.array2string(T_cam_marker[:3, :3], precision=4, suppress_small=True))
            print(f"Orientation (deg): Roll={roll:+.2f}, Pitch={pitch:+.2f}, Yaw={yaw:+.2f}")


            # Visualize detection
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size)

            # Add overlay text
            instruction_text = "Press 'a' to Accept, 'r' to Retake"
            cv2.putText(color_image, instruction_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Show image
            cv2.imshow('Marker Detection', color_image)

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == ord('a'):
                print("Frame accepted.")
                return T_cam_marker
            elif key == ord('r'):
                print("Retaking frame...")
                continue
            else:
                print("Invalid input. Press 'a' to accept or 'r' to retake.")


    def show_digit_with_crosshair(self):
        self.digit_running = True
        while self.digit_running:
            frame = self.digit.get_frame()
            height, width, _ = frame.shape
            center_x = width // 2
            center_y = height // 2

            # Draw crosshair
            cv2.line(frame, (0, center_y), (width, center_y), (0, 255, 0), 1)
            cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 1)

            cv2.putText(frame, "Press 'q' to close", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("DIGIT with Crosshair", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.digit_running = False
                cv2.destroyWindow("DIGIT with Crosshair")
                break



    def get_arm_pose(self):
        """Get xArm end-effector pose in base frame (T_base_ee)"""
        pose_data = self.arm.get_position()[1]
        position_m = [p/1000.0 for p in pose_data[:3]]  # Convert mm to meters
        orientation_rad = pose_data[3:6]  # Roll, pitch, yaw 

         # Convert angles based on SDK vs. protocol (Python-SDK uses degrees! as stated in the xarmdocumentation)
        orientation_rad = np.deg2rad(pose_data[3:6])  # degree → rad (for Python-SDK)
        
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = R.from_euler('xyz', orientation_rad).as_matrix()
        T_base_ee[:3, 3] = position_m

        # Print robot pose details
        print("\n=== Robot Pose Data ===")
        print(f"Position (m): X={position_m[0]:.4f}, Y={position_m[1]:.4f}, Z={position_m[2]:.4f}")
        print(f"Orientation (rad): Roll={orientation_rad[0]:.4f}, Pitch={orientation_rad[1]:.4f}, Yaw={orientation_rad[2]:.4f}")
        print("Rotation Matrix:")
        print(np.array2string(T_base_ee[:3, :3], precision=4, suppress_small=True))
        # Convert from xArm’s left-handed to right-handed coordinates to match camera
        convert_lh_to_rh = np.array([
                    [1,  0,  0, 0],
                    [0, -1,  0, 0],
                    [0,  0, -1, 0],
                    [0,  0,  0, 1]
                ])
        T_base_ee = T_base_ee @ convert_lh_to_rh

        return T_base_ee

    def capture_sample(self):
        """Capture one calibration sample (marker pose + robot touch pose)"""
        # Step 1: Detect marker (robot not blocking view)
        print("\n" + "="*50)
        print("Step 1: Detecting Marker Pose (Camera Frame)")
        print("="*50)
        input("Ensure marker is visible, then press Enter to detect...")
        for attempt in range(3):
            T_cam_marker = self.get_marker_pose()
            if T_cam_marker is not None:
                break
        else:
            print("Failed to detect marker after 3 attempts")
            return None

        # Step 2: Move robot to touch marker
        
        print("\n" + "="*50)
        print("Step 2: Recording Robot Pose (Base Frame)")
        print("="*50)
        print("Use the DIGIT output window to precisely align the robot with the center of the marker.")
        print("When you're satisfied with the alignment, make sure the DIGIT window is active, then press 'q' to close it.")
        print("The program will then prompt you in the terminal to record the robot's pose.")
        self.show_digit_with_crosshair()
        input("Press Enter to record robot pose...")
        T_base_ee = self.get_arm_pose()
        
        return T_base_ee, T_cam_marker

    def calibrate(self, num_samples=3):
        """Main calibration routine"""
        print("\n" + "="*50)
        print(f"Starting Calibration with {num_samples} samples")
        print("="*50)
        
        samples = []
        for i in range(num_samples):
            print(f"\n=== Collecting Sample {i+1}/{num_samples} ===")
            sample = self.capture_sample()
            if sample is not None:
                samples.append(sample)
                print(f"\nSample {len(samples)} successfully recorded.")
                print(f"T_base_ee:\n{sample[0]}")
                print(f"T_cam_marker:\n{sample[1]}")
                
                # Immediate sanity check for this sample
                T_base_ee, T_cam_marker = sample
                error = np.linalg.inv(T_base_ee) @ (self.T_base_cam @ T_cam_marker) if self.T_base_cam is not None else np.eye(4)
                print("Single sample sanity check (should be identity):")
                print(np.array2string(error, precision=4, suppress_small=True))
            else:
                print("Warning: Sample collection failed!")

        if len(samples) < 1:
            print("\nError: Insufficient samples for calibration (need ≥2)!")
            return
        
        # Improved averaging using quaternions for rotation and mean for translation
        translations = []
        quaternions = []
        
        for T_base_ee, T_cam_marker in samples:
            T = T_base_ee @ T_cam_marker      #main calculation
            translations.append(T[:3, 3])
            quaternions.append(R.from_matrix(T[:3, :3]).as_quat())
        
        # Compute mean translation
        mean_translation = np.mean(translations, axis=0)

        # Compute mean rotation robustly
        rot_stack = R.from_quat(quaternions)
        mean_rotation = rot_stack.mean().as_matrix()
  
        # Build final transform
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, :3] = mean_rotation
        self.T_base_cam[:3, 3] = mean_translation

        # Print calibration result and statistics
        print("\n" + "="*50)
        print("Calibration Results")
        print("="*50)
        print("Final T_base_cam (Base to Camera Transform):")
        print(np.array2string(self.T_base_cam, precision=6, suppress_small=True))
        
        # Calculate and display consistency metrics
        deviations = []
        for T_base_ee, T_cam_marker in samples:
            T_estimated = T_base_ee @ T_cam_marker
            deviation = np.linalg.norm(T_estimated[:3, 3] - self.T_base_cam[:3, 3])
            deviations.append(deviation)
        
        print(f"\nConsistency Metrics:")
        print(f"- Average position deviation: {np.mean(deviations):.6f} m")
        print(f"- Max position deviation: {np.max(deviations):.6f} m")
        print(f"- Min position deviation: {np.min(deviations):.6f} m")

        # Comprehensive sanity checks
        print("\n" + "="*50)
        print("Sanity Checks")
        print("="*50)
        
        # 1. Check that T_base_ee ≈ T_base_cam @ T_cam_marker for all samples
        print("\nPer-sample consistency checks (should be close to identity):")
        max_error = 0
        for i, (T_base_ee, T_cam_marker) in enumerate(samples):
            error = np.linalg.inv(T_base_ee) @ (self.T_base_cam @ np.linalg.inv(T_cam_marker))
            current_error = np.linalg.norm(error[:3, 3])
            max_error = max(max_error, current_error)
            print(f"\nSample {i+1} error norm: {current_error:.6f}")
            print("Error matrix:")
            print(np.array2string(error, precision=4, suppress_small=True))
        
        # 2. Check if the transform is physically plausible
        print("\nTransform sanity checks:")
        # Check if translation is reasonable (typically <1m for tabletop setups)
        translation_norm = np.linalg.norm(self.T_base_cam[:3, 3])
        print(f"- Camera to base distance: {translation_norm:.3f} m")
        if translation_norm > 2.0:
            print("  WARNING: Unusually large camera-base distance!")
        
        # Check if rotation matrix is orthogonal
        det = np.linalg.det(self.T_base_cam[:3, :3])
        print(f"- Determinant of rotation: {det:.6f} (should be 1.0)")
        if not np.isclose(det, 1.0, atol=0.01):
            print("  WARNING: Rotation matrix is not properly orthogonal!")
        
        # Check if the transform is invertible
        try:
            inv_transform = np.linalg.inv(self.T_base_cam)
            print("- Transform is properly invertible")
        except:
            print("  WARNING: Transform is not invertible!")
        
        # Overall quality assessment
        print("\nCalibration quality assessment:")
        if max_error < 0.01:  # 1cm error threshold
            print("SUCCESS: High quality calibration (error < 1cm)")
        elif max_error < 0.05:  # 5cm error threshold
            print("ACCEPTABLE: Moderate quality calibration (error < 5cm)")
        else:
            print("WARNING: Poor calibration quality (error > 5cm)")
            print("Consider collecting more samples or checking your setup")

      # Save result with timestamp and sample count
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"hand_eye_calib_{timestamp}_{num_samples}samp.npz"
        np.savez(filename, T_base_cam=self.T_base_cam)

        print(f"\nCalibration matrix saved to '{filename}'")

    def test_calibration(self):
        """Verify calibration by moving to a detected marker pose"""
        print("\n" + "="*50)
        print("Testing Calibration")
        print("="*50)
        input("Place marker in a new location, press Enter to detect...")
        
        T_cam_marker = self.get_marker_pose()
        if T_cam_marker is None:
            print("Error: Marker detection failed during test!")
            return

        # Transform to robot coordinates
        T_base_marker = self.T_base_cam @ np.linalg.inv(T_cam_marker)
        target_pos = T_base_marker[:3, 3]
        target_ori = R.from_matrix(T_base_marker[:3, :3]).as_euler('xyz')

        # Print test movement details
        print("\n=== Movement Command ===")
        print(f"Target Position (m): X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")
        print(f"Target Orientation (deg): Roll={np.degrees(target_ori[0]):.2f}, Pitch={np.degrees(target_ori[1]):.2f}, Yaw={np.degrees(target_ori[2]):.2f}")

        # Command robot (convert meters to mm and radians to degrees)
        # self.arm.set_position(
        #     x=target_pos[0] * 1000,
        #     y=target_pos[1] * 1000,
        #     z=target_pos[2] * 1000,
        #     roll=np.degrees(target_ori[0]),
        #     pitch=np.degrees(target_ori[1]),
        #     yaw=np.degrees(target_ori[2]),
        #     wait=True
        # )
        print("x",target_pos[0] * 1000)
        print("y",target_pos[1] * 1000)
        print("z",target_pos[2] * 1000)
        print("\nRobot movement complete! Verify physical alignment.")



    def shutdown(self):
        self.arm.disconnect()
        self.pipeline.stop()
        print("\nSystem resources released.")

if __name__ == "__main__":
    calibrator = HandEyeCalibrator()
    try:
        calibrator.calibrate(num_samples=4)
        calibrator.test_calibration()
    except Exception as e:
        print(f"\nError during calibration: {str(e)}")
    finally:
        calibrator.shutdown()


        