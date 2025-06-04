"""
find_centroid.py
Purpose is to calculate the centroid of an imported mask of a cube
Original code by 19451710 (2025)
"""


import numpy as np
import cv2

def find_mask_centroid(mask: np.ndarray):
    """
    Computes the centroid (x, y) of a binary mask and the average depth (z).
    
    Parameters:
    - mask: Binary mask of the object (dtype=np.uint8, 0 or 255).
    - depth_image: Corresponding depth image (same size as mask, dtype=float32 or uint16).
    
    Returns:
    - (x, y, z): Tuple of centroid coordinates with depth.
    """
    if mask.dtype != np.uint8:
        raise ValueError("Mask must be of type np.uint8 with values 0 or 255.")

    # Calculate moments to find centroid
    M = cv2.moments(mask)
    if M["m00"] == 0:
        print("Mask is empty; no centroid found.")
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

   
    
   

    print(f"Centroid (x, y): ({cx}, {cy})")
    return (cx, cy)

