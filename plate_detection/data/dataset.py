
from .augmentation import default_keypoint_transform

import torch
from torch.utils.data import Dataset
import albumentations as A

import cv2 as cv
import numpy as np

from typing import List, Optional


class ImageDataset(Dataset):

    """
    Image Dataset for keypoint detection.
        For each sample of shape:
            image: (C, H, W)
            boxes: (N, 4), in format [x1, y1, x2, y2]
            labels: (N,)
            keypoints: (N, K, 3), in format [x, y, visibility]
    """

    def __init__(self,
        img_paths: List[str],
        boxes: List[np.ndarray],
        labels: List[np.ndarray],
        keypoints: List[np.ndarray],
        transform: Optional[A.Compose] = None
        ):
        super().__init__()

        self.img_paths = img_paths
        self.boxes = boxes
        self.labels = labels
        self.keypoints = keypoints
        self.transform = transform
        if self.transform is None:
            self.transform = default_keypoint_transform()

    def __getitem__(self, item):

        img = cv.imread(self.img_paths[item], cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        boxes = self.boxes[item]
        labels = self.labels[item]
        keypoints = self.keypoints[item]
        keypoints, visibility = np.split(keypoints, 2, axis=1)

        transformed = self.transform(image=img, bboxes=boxes, keypoints=keypoints)
        img, boxes, keypoints = (transformed["image"],
                                 transformed["bboxes"],
                                 transformed["keypoints"])

        img = torch.tensor(img, torch.float32) / 255.
        img = img.permute(2, 0, 1)
        boxes = torch.tensor(boxes, torch.float32)
        labels = torch.tensor(labels, torch.int64)
        keypoints = torch.tensor(np.concatenate((keypoints, visibility), axis=1), torch.float32)

        return img, {"boxes": boxes, "labels": labels, "keypoints": keypoints}

    def __len__(self):
        return len(self.img_paths)
