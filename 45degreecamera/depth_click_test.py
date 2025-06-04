import cv2
import numpy as np
from realsense_utils import setup_realsense, get_aligned_frames, stop_realsense

def correct_depth_by_y(raw_depth, y_pixel, y_ref=240, pixel_size_m=0.0015, tilt_degrees=55):
    tilt_radians = np.deg2rad(tilt_degrees)
    scale_per_pixel = pixel_size_m / np.tan(tilt_radians)

    delta_y = y_pixel - y_ref
    correction = delta_y * scale_per_pixel

    corrected_depth = raw_depth - correction
    return corrected_depth

# Setup camera
pipeline, align, spatial, temporal, hole_filling, depth_scale = setup_realsense()

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color_frame, depth_image = param

        if y >= depth_image.shape[0] or x >= depth_image.shape[1]:
            print("Clicked out of bounds")
            return

        raw_depth = depth_image[y, x] * depth_scale

        corrected_depth = correct_depth_by_y(raw_depth, y_pixel=y)

        print(f"Clicked at (x={x}, y={y})")
        print(f"Raw depth: {raw_depth:.4f} m")
        #print(f"Corrected depth: {corrected_depth:.4f} m")

try:
    while True:
        color_frame, depth_frame, depth_image = get_aligned_frames(pipeline, align, spatial, temporal, hole_filling)
        if color_frame is None or depth_frame is None or depth_image is None:
            continue

        cv2.imshow("Depth Click Test", color_frame)
        cv2.setMouseCallback("Depth Click Test", click_event, (color_frame, depth_image))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    stop_realsense(pipeline)
    cv2.destroyAllWindows()