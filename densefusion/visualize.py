import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utils import compute_add_visualization_fixed, compute_rotation_difference_degrees


def draw_bboxes_on_image(rgb_image, objects_info, predictions):
    image_with_boxes = rgb_image.copy()
    h, w = image_with_boxes.shape[:2]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, (obj_info, pred_info) in enumerate(zip(objects_info, predictions)):
        color = colors[i % len(colors)]
        bbox_norm = obj_info['bbox_norm']
        xc_n, yc_n, w_n, h_n = bbox_norm
        xc_px = int(xc_n * w)
        yc_px = int(yc_n * h)
        w_px = int(w_n * w)
        h_px = int(h_n * h)
        x1 = max(0, xc_px - w_px // 2)
        y1 = max(0, yc_px - h_px // 2)
        x2 = min(w, xc_px + w_px // 2)
        y2 = min(h, yc_px + h_px // 2)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 3)
    return image_with_boxes
