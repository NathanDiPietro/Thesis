"""
realsense_utils.py
Purpose is to set up the intel realsense camera
sourced externally from https://pysource.com 
defines a functions for the Intel RealSense camera to make coding easier
depth functionaility included but not used
"""
# === realsense_utils.py ===
import pyrealsense2 as rs
import numpy as np
import cv2

def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

     # ✅ Add depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Disable auto exposure
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)  # 0 = Manual

    # Set manual exposure (in microseconds)
    manual_exposure_value = 4000  # Choose between 3000–5000 typically
    if depth_sensor.supports(rs.option.exposure):
        depth_sensor.set_option(rs.option.exposure, manual_exposure_value)

    # Optionally: set laser power to max
    if depth_sensor.supports(rs.option.laser_power):
        max_power = depth_sensor.get_option_range(rs.option.laser_power).max
        depth_sensor.set_option(rs.option.laser_power, max_power)

    return pipeline, align, spatial, temporal, hole_filling, depth_scale

def get_aligned_frames(pipeline, align, spatial, temporal, hole_filling):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_frame, depth_image

def get_intrinsics(pipeline):
    """
    Get the intrinsic matrix K from RealSense camera.
    """
    # Wait for frames to stabilize
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise ValueError("No color frame received!")

    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

    dist_coeffs = np.array(intrinsics.coeffs[:5], dtype=np.float32).reshape(-1,1)

    return K, dist_coeffs

def stop_realsense(pipeline):
    pipeline.stop()
