import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from realsense_utils import setup_realsense, get_aligned_frames, stop_realsense

def run_single_frame_segmentation():
    # === Load SAM Model ===
    sam_checkpoint = "NathanThesis/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    pred_iou_thresh=0.88,          # increase to favor higher quality masks
    stability_score_thresh=0.92,   # increase to filter out unstable masks
    min_mask_region_area=500,      # filters out small areas
    box_nms_thresh=0.7             # lower this to reduce overlap between masks
)

    # === Setup RealSense ===
    pipeline, align, spatial, temporal, hole_filling, _ = setup_realsense()

    try:
        # Warm-up the RealSense auto-exposure
        for _ in range(30):
            color_frame, _, _ = get_aligned_frames(pipeline, align, spatial, temporal, hole_filling)
        if color_frame is None:
            print("Failed to capture frame.")
            return

        image_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_rgb)

        for idx, mask in enumerate(masks):
            if not (1100 <= mask['area'] <= 1350):  # Adjust upper limit as needed
                continue

            segmentation = mask['segmentation'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) >= 4:
                    cv2.drawContours(color_frame, [approx], -1, (0, 255, 0), 2)

                    # Get centroid for label placement
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        label = f"#{idx} ({mask['area']})"
                        cv2.putText(color_frame, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Single Frame Segment", color_frame)
        cv2.waitKey(0)  # Wait for key press to close
        return masks, color_frame
    finally:
        stop_realsense(pipeline)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_single_frame_segmentation()