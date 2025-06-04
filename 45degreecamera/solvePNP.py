import cv2
import numpy as np
from itertools import permutations, combinations

def estimate_pose_from_corners(image_points, camera_matrix, dist_coeffs, image, indices_matrix):
    # Fixed object points corresponding to 6 known cube corners
    object_points = np.array([
        [0, 50, 50],   # P0: top back left
        [50, 50, 50],    # P1: top back right
        [0, 0, 50],     # P2: top front left
        [50, 50, 0],      # P3: bottom back right
        [0, 0, 0],     # P4: bottom front left
        [50, 0, 0],    # P5: bottom front right
    ], dtype=np.float32)

    object_points2 = np.array([
        [50, 50, 50],   # P0: top back right
        [0, 50, 50],    # P1: top back left
        [50, 50, 0],     # P2: bottom back right
        [0, 0, 50],      # P3: top front left
        [50, 0, 0],     # P4: bottom front right
        [0, 0, 0],    # P5: bottom front left
    ], dtype=np.float32)

    image_points = np.array(image_points, dtype=np.float32)
    if image_points.shape[0] != 6:
        print("❌ Need exactly 6 image points in the correct order.")
        return None

    # best_error = float('inf')
    # best_pose = None
    # best_order = None

    # for perm in permutations(image_points, 6):
    #     img_pts = np.array(perm, dtype=np.float32)
    #     success, rvec, tvec = cv2.solvePnP(object_points2, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    #     if not success:
    #         continue

    #     projected_pts, _ = cv2.projectPoints(object_points2, rvec, tvec, camera_matrix, dist_coeffs)
    #     error = np.mean(np.linalg.norm(projected_pts.reshape(-1, 2) - img_pts, axis=1))

    #     if error < best_error:
    #         best_error = error
    #         best_pose = (rvec, tvec)
    #         best_order = img_pts

    # if best_pose is None:
    #     print("❌ No valid pose found from any permutation.")
    #     return None
    
    # print(f"✅ Best reprojection error: {best_error:.3f}px")
    # print("📌 Best image point order:")
    # for i, pt in enumerate(best_order):
    #     print(f"  P{i}: (x={pt[0]:.1f}, y={pt[1]:.1f})")

    # rvec, tvec = best_pose
    obj_points = choose_object_points(image_points, indices_matrix)
    success, rvec, tvec = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
  

    R, _ = cv2.Rodrigues(rvec)
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R
    pose_matrix[:3, 3] = tvec.flatten()

    if image is not None:
        axis_length = 30  # mm
        origin = obj_points[0].reshape(1, 3)
        x_axis = origin + np.array([[axis_length, 0, 0]])
        y_axis = origin + np.array([[0, axis_length, 0]])
        z_axis = origin + np.array([[0, 0, axis_length]])
        pts_3d = np.vstack([origin, x_axis, y_axis, z_axis]).astype(np.float32)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, dist_coeffs)
        o, x, y, z = [tuple(pt.ravel().astype(int)) for pt in pts_2d]
        cv2.line(image, o, x, (0, 0, 255), 2)  # X = red
        cv2.line(image, o, y, (0, 255, 0), 2)  # Y = green
        cv2.line(image, o, z, (255, 0, 0), 2)  # Z = blue

    print("✅ Pose estimated using fixed corner order. Best error")
    return pose_matrix



from scipy.spatial.transform import Rotation as R



def get_euler_angles_from_pose(pose_matrix):
    R_mat = pose_matrix[:3, :3]  # Extract 3x3 rotation matrix
    r = R.from_matrix(R_mat)
    yaw, pitch, roll = r.as_euler('zyx', degrees=True)  # ZYX = yaw-pitch-roll
    return yaw, pitch, roll

def choose_object_points(image_points, indices_matrix):
    # Assume sorted order: [top-back-left, top-back-right, top-front-left, ...]
     # Fixed object points corresponding to 6 known cube corners
    #Flat cube
    indices_sum = sum(indices_matrix)

    object_points1 = np.array([
        [0, 50, 50],   # P0: top back left
        [50, 50, 50],    # P1: top back right
        [0, 0, 50],     # P2: top front left
        [50, 50, 0],      # P3: bottom back right
        [0, 0, 0],     # P4: bottom front left
        [50, 0, 0],    # P5: bottom front right
    ], dtype=np.float32)

    #Cube leaning left
    object_points2 = np.array([
        [50, 50, 50],   # P0: top back right
        [0, 50, 50],    # P1: top back left
        [50, 50, 0],     # P2: bottom back right
        [0, 0, 50],      # P3: top front left
        [50, 0, 0],     # P4: bottom front right
        [0, 0, 0],    # P5: bottom front left
    ], dtype=np.float32)

    #Cube leaning right
    object_points3 = np.array([
        [0, 50, 50],   # P0: top back left
        [50, 50, 50],    # P1: top back right
        [0, 50, 0],     # P2: bottom back left
        [50, 0, 50],      # P3: top front right
        [0, 0, 0],     # P4: bottom front left
        [50, 0, 0],    # P5: bottom front right
    ], dtype=np.float32)

    x_dif = image_points[2][0] - image_points[1][0]
    
    if x_dif > 0:  # threshold to tolerate noise
        print("2")
        return object_points2
    else:
        if indices_sum == 4:
            return object_points3
        else:
            return object_points1