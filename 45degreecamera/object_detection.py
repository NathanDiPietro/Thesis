# === object_detection.py ===
import cv2
import numpy as np

def get_centroid(mask):
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return np.array([cX, cY])
    return None

def draw_centroid(image, centroid):
    cv2.circle(image, centroid, 5, (0, 0, 255), -1)