
from .augmentation import default_keypoint_transform, identity_keypoint_transform

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A

import cv2 as cv
import numpy as np

from typing import Dict, List, Optional, Tuple


def trim_bbox(img: np.ndarray, bbox: np.ndarray):
    h, w, c = img.shape
    bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, w)
    bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, h)
    return bbox


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
        labels: List[np.ndarray],
        boxes: List[np.ndarray],
        keypoints: List[np.ndarray],
        transform: Optional[A.Compose] = None
        ):
        super().__init__()

        self.img_paths = img_paths
        self.labels = labels
        self.boxes = boxes
        self.keypoints = keypoints
        self.transform = transform
        if self.transform is None:
            self.transform = default_keypoint_transform()

    def __getitem__(self, item):

        img = cv.imread(self.img_paths[item], cv.IMREAD_COLOR)
        img: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        labels: np.ndarray = self.labels[item]
        boxes: np.ndarray = self.boxes[item]
        boxes = trim_bbox(img, boxes)
        boxes = np.concatenate((boxes, np.expand_dims(labels, 1)), axis=1)
        keypoints: np.ndarray = self.keypoints[item]
        keypoints, visibility, _ = np.split(keypoints, [2, 3], axis=2)
        kp_n, kp_k, kp_d = keypoints.shape
        keypoints = keypoints.reshape((kp_n * kp_k, kp_d))

        transformed = self.transform(image=img, bboxes=boxes, keypoints=keypoints)
        img, boxes, keypoints = (transformed["image"],
                                 transformed["bboxes"],
                                 transformed["keypoints"])
        boxes, labels, _ = np.split(boxes, [4, 5], axis=1)
        labels = labels.squeeze(0)
        keypoints = np.asarray(keypoints).reshape((kp_n, kp_k, kp_d))

        img_h, img_w = img.shape[:2]
        kp_isin_img = (
            (np.greater_equal(keypoints[..., 0], 0) & np.less_equal(keypoints[..., 0], img_w))
            & (np.greater_equal(keypoints[..., 1], 0) & np.less_equal(keypoints[..., 1], img_h))
        )
        visibility = np.expand_dims(kp_isin_img, axis=-1).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float32) / 255.
        img = img.permute(2, 0, 1)
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        keypoints = torch.tensor(np.concatenate((keypoints, visibility), axis=2), dtype=torch.float32)

        return img, {"boxes": boxes, "labels": labels, "keypoints": keypoints}

    def __len__(self):
        return len(self.img_paths)


class ImageCollate(object):

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
        transposed = tuple(list(col) for col in zip(*batch))
        return transposed

    @staticmethod
    def collate_fn_quad(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
        img, targets = zip(*batch)
        return


def get_dataset(img_paths: List[str], labels: List[np.ndarray],
                boxes: List[np.ndarray], keypoints: List[np.ndarray],
                transform_apply_p=0.5, is_val=False) -> ImageDataset:
    if is_val:
        transform = identity_keypoint_transform()
    else:
        transform = default_keypoint_transform(p=transform_apply_p)
    return ImageDataset(img_paths, labels, boxes, keypoints, transform)


def concat_ds(*ds: Dataset) -> ConcatDataset:
    ds = ConcatDataset(list(ds))
    return ds


def get_loader(
    dataset,
    batch_size=4, shuffle=True, num_workers=0) -> DataLoader:
    loader = DataLoader(dataset, batch_size,
                        shuffle=shuffle, num_workers=num_workers,
                        collate_fn=ImageCollate.collate_fn)
    return loader
