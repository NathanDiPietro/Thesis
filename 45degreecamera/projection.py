import numpy as np

def correct_depth(u, v, raw_depth, K, R, t):
    """
    Corrects the depth measurement from the camera to get true world height.

    Parameters:
    - u, v: Pixel coordinates (ints)
    - raw_depth: Depth value from camera at (u,v), in meters (float)
    - K: Intrinsic matrix (3x3 numpy array)
    - R: Rotation matrix (3x3 numpy array)
    - t: Translation vector (3x1 or 3, numpy array)

    Returns:
    - world_point: (x, y, z) numpy array in world coordinates
    - true_height: z-coordinate (float) in world frame
    """
    # Step 1: Unproject to camera coordinates
    pixel = np.array([u, v, 1.0])
    K_inv = np.linalg.inv(K)
    ray = K_inv @ pixel
    point_camera = raw_depth * ray

    # Step 2: Transform to world coordinates
    point_world = R @ point_camera + t.reshape(3)

    # Step 3: Extract true vertical depth (z)
    true_height = point_world[2]

    return point_world, true_height

def create_camera_pose(tilt_degrees, camera_height):
    """
    Create a simple camera pose (R, t) assuming tilt around x-axis only.

    Parameters:
    - tilt_degrees: Tilt angle in degrees (positive = looking down)
    - camera_height: Height of camera from ground in meters

    Returns:
    - R: Rotation matrix (3x3 numpy array)
    - t: Translation vector (3x1 numpy array)
    """
    tilt_radians = np.deg2rad(tilt_degrees)

    R = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_radians), -np.sin(tilt_radians)],
        [0, np.sin(tilt_radians),  np.cos(tilt_radians)]
    ])

    t = np.array([0, 0, camera_height])

    return R, t

# Example Usage
if __name__ == "__main__":
    # Assume some intrinsic matrix K
    K = np.array([
        [606, 0, 329],
        [0, 605, 237],
        [0,   0,   1]
    ])

    # Camera is 1.0m above ground, tilted down by 45 degrees
    R, t = create_camera_pose(tilt_degrees=45, camera_height=1.0)

    # Example pixel and raw depth
    u, v = 300, 200
    raw_depth = 1.0  # meters

    world_point, true_height = correct_depth(u, v, raw_depth, K, R, t)

    print("World Point (x, y, z):", world_point)
    print("True Height (z):", true_height)
