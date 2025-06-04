"""
    transform.py
    Purpose is to calibrate the coordinates and transform camera to robot coordinate spaces

    Original Code by 19451710 (2025)

"""
# === transform.py ===
import numpy as np
import cv2

def compute_homography_transform():
    """
    Computes a homography matrix (3x3) that maps 2D depth camera points to robot space.
    Returns:
        H: 3x3 homography matrix
    """
    depth_points = np.array([
        (453, 228), (314, 266), (362, 208), (212, 253),
        (247, 188), (325, 165), (403, 157), (437, 271)
    ], dtype=np.float32)

    robot_points = np.array([
        (760.893, -26.124),
        (527.055, -102.593),
        (626.22735512, 11.371787342),
        (349.64635512, -58.122787342),
        (405.59135512, 81.158787342),
        (555.59135512, 131.158787342),
        (705.59135512, 131.158787342),
        (727.055, -102.593)
    ], dtype=np.float32)

    H, _ = cv2.findHomography(depth_points, robot_points)
    if H is None:
        raise ValueError("Homography could not be estimated. Check input points.")
    return H

def transform_new_coords(point, homography_matrix):
    """
    Applies a homography transform to a 2D point.
    :param point: (x, y) tuple or array from camera
    :param homography_matrix: 3x3 homography matrix
    :return: Transformed (x', y') in robot space
    """
    point_array = np.array([[point]], dtype=np.float32)  # shape (1, 1, 2)
    transformed = cv2.perspectiveTransform(point_array, homography_matrix)
    return float(transformed[0][0][0]), float(transformed[0][0][1])
