
import numpy as np

from itertools import product
from typing import Dict


def _box_area(boxes: np.ndarray):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union(boxes_a: np.ndarray, boxes_b: np.ndarray):
    area_a = _box_area(boxes_a)
    area_b = _box_area(boxes_b)

    left_top = np.maximum(boxes_a[:, np.newaxis, :2], boxes_b[:, :2])
    right_bot = np.minimum(boxes_a[:, np.newaxis, 2:], boxes_b[:, 2:])

    width_height = np.clip((right_bot - left_top), a_min=0.)
    inter = width_height[..., 0] * width_height[..., 1]
    union = area_a[:, np.newaxis] + area_b - inter
    return inter, union


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray):
    inter, union = _box_inter_union(boxes_a, boxes_b)
    iou = inter / union
    return iou


def map_single(prediction: Dict[str, np.ndarray],
               target: Dict[str, np.ndarray],
               iou_threshold=0.5) -> float:
    """

    Args:
        prediction: A list consisting of dictionaries each containing the key-values
        (each dictionary corresponds to a single image):
        - ``boxes``: ``numpy.ndarray<float32>`` of shape
            [num_boxes, 4] containing `num_boxes` detection boxes of the format
            specified in the constructor. By default, this method expects
            [xmin, ymin, xmax, ymax] in absolute image coordinates.
        - ``scores``: ``numpy.ndarray<float32>`` of shape
            [num_boxes] containing detection scores for the boxes.
        - ``labels``: ``numpy.ndarray<int>`` of shape
            [num_boxes] containing 0-indexed detection classes for the boxes.

        target: A list consisting of dictionaries each containing the key-values
        (each dictionary corresponds to a single image):
        - ``boxes``: ``numpy.ndarray<float32>`` of shape
            [num_boxes, 4] containing `num_boxes` ground truth boxes of the format
            specified in the constructor. By default, this method expects
            [xmin, ymin, xmax, ymax] in absolute image coordinates.
        - ``labels``: ``numpy.ndarray<int>`` of shape
            [num_boxes] containing 1-indexed ground truth classes for the boxes.

        iou_threshold: at least IoU value to be accepted as True-Positive
            between predicted box and ground-true box

    Returns:
        Mean Average Precision.
    """
    classes = np.unique(target["labels"])
    for c in classes:
        gt_idx = np.where(target["labels"] == c)
        gt = target["labels"][gt_idx]
        det_idx = np.where(prediction["labels"] == c)
        det_boxes = prediction["boxes"][det_idx]
        det_scores = prediction["scores"][det_idx]
        det_scores_sorted_idx = np.argsort(det_scores)[::-1]
        det_boxes = det_boxes[det_scores_sorted_idx]
        iou = box_iou(det_boxes, gt)
        h, w = iou.shape
        mat = np.zeros_like(iou)
        accessed_gt_idx = list()
        for i in range(h):
            for j in np.argsort(iou[i])[::-1]:
                if iou[i, j] >= iou_threshold and j not in accessed_gt_idx:
                    mat[i, j] = 1
                    accessed_gt_idx.append(j)
                    break
    return 0.
