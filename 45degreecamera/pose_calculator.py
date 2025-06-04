# pose_calculator.py
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

def estimate_pose_solvepnp(corners_2d, depth_frame, intrinsics, dist_coeffs=None):
    object_points = np.array([
        [0, 0, 0],
        [50, 0, 0],
        [50, 50, 0],
        [0, 50, 0],
        [0, 0, 50],
        [50, 0, 50],
    ], dtype=np.float32)

    if len(corners_2d) < 6:
        return None

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))

    corners_2d_np = np.array(corners_2d[:6], dtype=np.float32)
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    camera_matrix = np.array([[fx,  0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(object_points, corners_2d_np, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    R_mat, _ = cv2.Rodrigues(rvec)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R_mat
    pose_matrix[:3, 3] = tvec.flatten()
    return pose_matrix

def draw_axes_on_image(image, pose_matrix, intrinsics, axis_length=30):
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((5, 1))
    origin = pose_matrix[:3, 3].reshape(1, 3)
    x_axis = origin + pose_matrix[:3, 0].reshape(1, 3) * axis_length
    y_axis = origin + pose_matrix[:3, 1].reshape(1, 3) * axis_length
    z_axis = origin + pose_matrix[:3, 2].reshape(1, 3) * axis_length
    pts_3d = np.vstack([origin, x_axis, y_axis, z_axis]).astype(np.float32)
    pts_2d, _ = cv2.projectPoints(pts_3d, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
    o, x, y, z = [tuple(pt.ravel().astype(int)) for pt in pts_2d]
    cv2.line(image, o, x, (0, 0, 255), 2)
    cv2.line(image, o, y, (0, 255, 0), 2)
    cv2.line(image, o, z, (255, 0, 0), 2)
    return image

def print_pose_matrix(matrix):
    print("📐 SolvePnP Pose Matrix:")
    for row in matrix:
        print(" ".join(f"{val: .4f}" for val in row))

def normalize_pose_to_ground(pose_camera, tilt_angle_deg=45):
    R_c2w = R.from_euler('x', -tilt_angle_deg, degrees=True).as_matrix()
    pose_world = np.eye(4)
    pose_world[:3, :3] = R_c2w @ pose_camera[:3, :3]
    pose_world[:3, 3]  = R_c2w @ pose_camera[:3, 3]
    return pose_world
