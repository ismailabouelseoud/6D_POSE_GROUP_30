import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class DenseFusionSegmentationModule:
    """Segmentation module using Mask R-CNN for instance segmentation"""
    def __init__(self, confidence_threshold=0.5, device='cuda'):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._initialize_model()

    def _initialize_model(self):
        """Initialize Mask R-CNN model"""
        try:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            ).to(self.device).eval()
            print("✓ Mask R-CNN model loaded successfully")
        except Exception as e:
            print(f"Failed to initialize Mask R-CNN: {e}")

    def refine_detection(self, rgb_image, bbox_pixel, class_id=None):
        """Refine YOLO detection using Mask R-CNN segmentation"""
        if self.model is None:
            return self._bbox_to_mask(rgb_image.shape[:2], bbox_pixel), {'source': 'bbox_fallback_no_model'}

        try:
            pil_image = Image.fromarray(rgb_image)
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(image_tensor)

            if not predictions or not predictions[0]['masks']:
                return self._bbox_to_mask(rgb_image.shape[:2], bbox_pixel), {'source': 'bbox_fallback_no_detections'}

            best_mask, best_info = self._find_best_mask(predictions[0], bbox_pixel, rgb_image.shape[:2])
            
            if best_mask is not None:
                return best_mask, best_info
            else:
                return self._bbox_to_mask(rgb_image.shape[:2], bbox_pixel), {'source': 'bbox_fallback_low_overlap'}
        except Exception as e:
            return self._bbox_to_mask(rgb_image.shape[:2], bbox_pixel), {'source': 'bbox_fallback_error', 'error': str(e)}

    def _find_best_mask(self, prediction, yolo_bbox, image_shape):
        """Find the best mask that overlaps with YOLO detection"""
        x1, y1, x2, y2 = map(int, yolo_bbox)
        yolo_area = max(1, (x2 - x1) * (y2 - y1))
        best_mask, best_score, best_info = None, 0, {'source': 'bbox_fallback'}

        for mask_tensor, score, box in zip(prediction['masks'], prediction['scores'], prediction['boxes']):
            if score < self.confidence_threshold:
                continue

            mask_np = (mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            overlap_area = np.sum(mask_np[y1:y2, x1:x2])
            overlap_ratio = overlap_area / yolo_area if yolo_area > 0 else 0
            
            if overlap_ratio > 0.1 and score > best_score:
                best_score = score
                best_mask = mask_np
                best_info = {'source': 'mask_rcnn', 'confidence': float(score), 'overlap': overlap_ratio}

        return best_mask, best_info

    def _bbox_to_mask(self, image_shape, bbox_pixel):
        """Fallback: create mask from bounding box"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        x1, y1, x2, y2 = map(int, bbox_pixel)
        mask[y1:y2, x1:x2] = 1
        return mask
