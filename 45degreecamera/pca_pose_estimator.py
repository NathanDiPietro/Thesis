# pca_pose_estimator.py
import numpy as np
import pyrealsense2 as rs

def mask_to_pointcloud(mask, depth_frame, intrinsics, stride=2):
    points = []
    h, w = mask.shape
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if mask[y, x] > 0:
                depth = depth_frame.get_distance(x, y)
                if depth > 0:
                    pt = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                    points.append(pt)
    return np.array(points)

def estimate_pose_from_pca(points):
    if len(points) < 3:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Normalize axes
    x_axis = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
    y_axis = eigvecs[:, 1] / np.linalg.norm(eigvecs[:, 1])
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = centroid
    return pose

def draw_axes_on_image(image, pose_matrix, intrinsics, axis_length=30):
    import cv2

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
    print("📐 PCA 6D Pose Matrix:")
    for row in matrix:
        print(" ".join(f"{val: .4f}" for val in row))
