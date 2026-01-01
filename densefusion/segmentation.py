import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from .config import config


class DenseFusionSegmentationModule:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = config.DEVICE
        self.model = None
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.stats = {'total_calls': 0, 'successful_segmentations': 0, 'bbox_fallbacks': 0}
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            )
            self.model.eval()
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Failed to initialize Mask R-CNN: {e}")
            self.model = None

    def refine_detection(self, rgb_image, bbox_pixel, class_id=None):
        self.stats['total_calls'] += 1
        if self.model is None:
            return self._bbox_to_mask(rgb_image.shape[:2], bbox_pixel), {'source': 'bbox_fallback_no_model'}
        try:
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                pil_image = transforms.functional.to_pil_image(rgb_image)
                image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                predictions = self.model(image_tensor)
            if len(predictions) == 0 or len(predictions[0]['masks']) == 0:
                self.stats['bbox_fallbacks'] += 1
                return self._bbox_to_mask(rgb_image.shape[:2], bbox_pixel), {'source': 'bbox_fallback_no_detections'}
            best_mask, best_info = self._find_best_mask(predictions[0], bbox_pixel, rgb_image.shape[:2])
            if best_mask is not None:
                self.stats['successful_segmentations'] += 1
                return best_mask, best_info
            else:
                self.stats['bbox_fallbacks'] += 1
                return self._bbox_to_mask(rgb_image.shape[:2], bbox_pixel), {'source': 'bbox_fallback_low_overlap'}
        except Exception as e:
            self.stats['bbox_fallbacks'] += 1
            return self._bbox_to_mask(rgb_image.shape[:2], bbox_pixel), {'source': 'bbox_fallback_error', 'error': str(e)}

    def _find_best_mask(self, prediction, yolo_bbox, image_shape):
        x1, y1, x2, y2 = map(int, yolo_bbox)
        yolo_area = max(1, (x2 - x1) * (y2 - y1))
        best_mask = None
        best_score = 0
        best_info = {'source': 'bbox_fallback'}
        masks = prediction['masks']
        scores = prediction['scores']
        boxes = prediction['boxes']
        for mask_tensor, score, box in zip(masks, scores, boxes):
            if score < self.confidence_threshold:
                continue
            mask_np = mask_tensor.squeeze().cpu().numpy()
            if mask_np.shape != image_shape:
                mask_np = cv2.resize(mask_np, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            mask_in_bbox = mask_binary[y1:y2, x1:x2]
            overlap_area = np.sum(mask_in_bbox)
            overlap_ratio = overlap_area / yolo_area if yolo_area > 0 else 0
            pred_x1, pred_y1, pred_x2, pred_y2 = box.cpu().numpy()
            inter_x1 = max(x1, pred_x1)
            inter_y1 = max(y1, pred_y1)
            inter_x2 = min(x2, pred_x2)
            inter_y2 = min(y2, pred_y2)
            iou = 0.0
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                union_area = yolo_area + pred_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
            combined_score = float(score) * 0.4 + overlap_ratio * 0.3 + iou * 0.3
            if combined_score > best_score and overlap_ratio > 0.1:
                best_score = combined_score
                best_mask = mask_binary
                best_info = {'source': 'mask_rcnn', 'confidence': float(score), 'overlap': overlap_ratio, 'iou': iou}
        return best_mask, best_info

    def _bbox_to_mask(self, image_shape, bbox_pixel):
        mask = np.zeros(image_shape, dtype=np.uint8)
        x1, y1, x2, y2 = map(int, bbox_pixel)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_shape[1], x2), min(image_shape[0], y2)
        mask[y1:y2, x1:x2] = 1
        return mask

    def print_stats(self):
        if self.stats['total_calls'] > 0:
            success_rate = self.stats['successful_segmentations'] / self.stats['total_calls']
            print(f"Segmentation Statistics:")
            print(f"  Total calls: {self.stats['total_calls']}")
            print(f"  Success rate: {success_rate:.2%}")
