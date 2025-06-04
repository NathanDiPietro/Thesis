"""
sam.py
Purpose is to create and use the segment anything model for cube segmentation
sourced externally from https://github.com/facebookresearch/segment-anything

"""
# === sam.py ===
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def load_sam_model():
    sam_checkpoint = "NathanThesis/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    return SamPredictor(sam.to(device))

def predict_mask_from_click(predictor, coords):
    input_point = np.array([coords])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    return masks[0].astype("uint8") * 255
