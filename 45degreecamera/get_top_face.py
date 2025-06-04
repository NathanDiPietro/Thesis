import cv2
import numpy as np

def extract_top_face_from_mask(mask):
    """
    Extracts the top face of a cube from a binary mask by identifying 6 corners,
    estimating the internal corner as the centroid, and selecting the 4-corner combination
    with the largest area as the top face.

    Args:
        mask (np.ndarray): Binary mask image where cube pixels = 255, background = 0.

    Returns:
        np.ndarray or None: A (4, 2) array of corner points corresponding to the top face.
    """
    # Step 1: Find external contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # Step 2: Approximate external contour to polygon
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) != 6:
        return None  # Expecting exactly 6 corners

    corners = approx.reshape(6, 2)

    # Step 3: Estimate internal corner as centroid of the 6 points
    internal_corner = np.mean(corners, axis=0)

    # Step 4: Try all combinations of 3 external corners + internal corner
    best_quad = None
    max_area = -1

    for i in range(6):
        for j in range(i+1, 6):
            for k in range(j+1, 6):
                quad = np.array([corners[i], corners[j], corners[k], internal_corner])

                # Check if the quad is convex and has a reasonable shape
                quad_int = quad.astype(np.int32)
                if cv2.isContourConvex(quad_int):
                    area = cv2.contourArea(quad)
                    if area > max_area:
                        max_area = area
                        best_quad = quad

    return best_quad if best_quad is not None else None