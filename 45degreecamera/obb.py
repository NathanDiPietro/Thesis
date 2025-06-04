import open3d as o3d
import numpy as np

def fit_obb_from_points(points_3d: np.ndarray, visualize: bool = False):
    """
    Fit an Oriented Bounding Box (OBB) to a set of 3D points.

    Args:
    - points_3d: (N, 3) numpy array of 3D points
    - visualize: If True, show the OBB, points, and coordinate frame in an Open3D window

    Returns:
    - center: (3,) numpy array, center of the box
    - rotation_matrix: (3,3) numpy array, orientation of the box
    - extent: (3,) numpy array, box dimensions (width, height, depth)
    - obb: the fitted open3d.geometry.OrientedBoundingBox
    """
    #if len(points_3d) > 1000:
        #idx = np.random.choice(len(points_3d), size=500, replace=False)
        #qpoints_3d = np.array(points_3d)[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 0, 1)  # Set OBB color to blue

    if visualize:
        visualize_obb_and_points(points_3d, obb)

    center = obb.center
    rotation_matrix = obb.R
    extent = obb.extent

    return center, rotation_matrix, extent, obb

def visualize_obb_and_points(points_3d: np.ndarray, obb: o3d.geometry.OrientedBoundingBox):
    """
    Visualize the fitted OBB, the point cloud, and a coordinate frame.

    Args:
    - points_3d: (N, 3) numpy array of 3D points
    - obb: open3d.geometry.OrientedBoundingBox
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    o3d.visualization.draw_geometries([pcd, obb, frame])

# Example Usage
if __name__ == "__main__":
    # Dummy points for testing
    dummy_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])

    center, rotation_matrix, extent, obb = fit_obb_from_points(dummy_points, visualize=True)

    print("OBB Center:", center)
    print("OBB Rotation Matrix:\n", rotation_matrix)
    print("OBB Size (Extent):", extent)
