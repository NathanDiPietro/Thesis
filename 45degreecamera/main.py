"""
    main.py
    is the python file which calls other files and executes the segmentation, pose calculation
    and robot connection code

    Original Code by 19451710 (2025)
    Auto Segmentation code is future work and shown by the global variable RUN_AUTO_SEGMENT

"""
# === IMPORT STATEMENTS === #
import cv2
import numpy as np
import torch

# === CUSTOM IMPORTS & DIRECTORY MANAGEMENT === #
from segment_anything import sam_model_registry, SamPredictor
from realsense_utils import setup_realsense, get_aligned_frames, stop_realsense, get_intrinsics
from solvePNP import estimate_pose_from_corners, get_euler_angles_from_pose
from find_centroid import find_mask_centroid
from transform import compute_homography_transform, transform_new_coords
from pose_transform import convert_camera_to_robot_quaternion
from send_to_robot import send_to_robot
from cube import Cube
# === GLOBAL VARIABLES === #

RUN_AUTO_SEGMENT = False  # Toggle between auto and click-based mode
cube_list = []

# === Load SAM Model ===
sam_checkpoint = "NathanThesis/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam.to(device))

# === Setup RealSense camera ===
pipeline, align, spatial, temporal, hole_filling, depth_scale = setup_realsense()
K, dist_coeffs = get_intrinsics(pipeline)  # Camera intrinsics matrix

# === Click Callback ===
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color_frame, depth_image = param

        # Run SAM segmentation
        image_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        mask = masks[0].astype("uint8") * 255
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(color_frame, 0.6, mask_color, 0.4, 0)

        centroid = find_mask_centroid(mask)

        if centroid is not None:
            x, y = centroid
            print(f"[INFO] Centroid in camera space: (x={x}, y={y})")

            # Load homography and transform camera coordinates to robot coordinates
            try:
                H = compute_homography_transform()
                x_robot, y_robot = transform_new_coords((x, y), H)
                print(f"[INFO] Robot space coordinate (via homography): (x={x_robot:.3f}, y={y_robot:.3f})")
            except Exception as e:
                print(f"[ERROR] Failed to apply homography: {e}")
        # === Extract corners from the SAM mask using contours ===
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("❌ No contours found.")
            return
        
      

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 6:
            print(f"❌ Only found {len(approx)} corners. Need 6.")
            return

        corners = approx.reshape(-1, 2).astype(np.float32)[:6]
        corners = sorted(corners, key=lambda pt: (pt[1], pt[0]))  # sort by Y then X
        image_points = np.array(corners, dtype=np.float32)

        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        # Store brightness values with indices
        brightness_with_index = []
        for i, (x, y) in enumerate(corners):
            brightness = gray[int(y), int(x)]
            print(f"Corner {i + 1} at ({x}, {y}) has brightness: {brightness}")
            brightness_with_index.append((brightness, i))

        # Sort and get indices of top 3 brightest corners
        brightness_with_index.sort(reverse=True)  # Sort by brightness descending
        top3_indices = [index for (_, index) in brightness_with_index[:3]]

        print("Top 3 brightest corner indices:", top3_indices)
        
        # Draw Corners
        for idx, (cx, cy) in enumerate(image_points):
            cv2.circle(blended, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(blended, f"P{idx}", (int(cx) + 5, int(cy) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Calculate Pose
        pose_matrix = estimate_pose_from_corners(image_points, K, dist_coeffs, blended, top3_indices)
        if pose_matrix is not None:
            print("✅ Pose Matrix:")
            for row in pose_matrix:
                print("  ", " ".join(f"{v: .4f}" for v in row))

            yaw, pitch, roll = get_euler_angles_from_pose(pose_matrix)
            print(f"Yaw:   {yaw:.2f}°")
            print(f"Pitch: {pitch:.2f}°")
            print(f"Roll:  {roll:.2f}°")

        cv2.imshow("SAM Click Mask", blended)
        quat, tilted_left, tilted_right, flat = convert_camera_to_robot_quaternion(yaw, pitch, roll)
        qx,qy,qz,qw = quat
        print(f"Quat: {qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f}, ")
        #Determine tilt orientation
        if tilted_left:
            cube = Cube(x_robot - 15.0, y_robot, 575.0, quat)
        elif tilted_right:
            cube = Cube(x_robot + 15.0, y_robot, 575.0, quat)
        else:
            cube = Cube(x_robot, y_robot, 575.0, quat)

        cube_list.append(cube)
        print(f"[INFO] Stored cube: {cube}")

# === Main Loop ===
if RUN_AUTO_SEGMENT:
    from auto_segmenter import run_single_frame_segmentation
    masks, color_frame = run_single_frame_segmentation()

else:
    # === Main Click-Based Loop ===
    try:
        while True:
            color_frame, depth_frame, depth_image = get_aligned_frames(pipeline, align, spatial, temporal, hole_filling)
            if color_frame is None or depth_frame is None or depth_image is None:
                continue

            cv2.imshow("Live Feed", color_frame)
            cv2.setMouseCallback("Live Feed", click_event, (color_frame, depth_image))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Sending cubes to robot...")
                # Send list of cubes to robot
                send_to_robot(cube_list)
                print("test")
                break
    finally:
        stop_realsense(pipeline)
        cv2.destroyAllWindows()