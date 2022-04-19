
from .augmentation import default_transform

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import numpy as np
import cv2 as cv

from typing import List, Tuple, Union
from pathlib import Path


class OCRDataset(Dataset):

    def __init__(self, img_paths: List[str], labels: List[np.ndarray],
                 transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        if self.transform is None:
            self.transform = default_transform()

    def __getitem__(self, item):
        image_path = self.img_paths[item]
        target = self.labels[item]

        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        transformed_data = self.transform(image=image)
        image = transformed_data["image"]

        image = torch.tensor(image, dtype=torch.float32) / 255.
        image = torch.permute(image, (2, 0, 1))
        target = torch.tensor(target, dtype=torch.int64)
        return image, target

    def __len__(self):
        return len(self.img_paths)

    @classmethod
    def from_yaml(cls, yaml_path, data_dir: Union[Path, str]):
        import yaml
        data_dir: Path = Path(data_dir)
        with open(yaml_path, "r") as f:
            yaml_object = yaml.safe_load(f.read())
        image_paths = list()
        for rel_path in yaml_object["image_paths"]:
            image_paths.append(str((data_dir / rel_path).resolve()))
        targets = list()
        for target in yaml_object["char_labels"]:
            targets.append(np.array(target, dtype=np.int64))
        return cls(image_paths, targets)


def to_dataloader(dataset, batch_size: int = 16, num_workers: int = 4):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers)


def concat_ds(*ds: Dataset) -> ConcatDataset:
    return ConcatDataset(ds)


def split_ds(ds: Dataset, val_split: float = 0.2) -> Tuple[Dataset, Dataset]:
    indices: np.ndarray = np.arange(len(ds))
    np.random.shuffle(indices)
    split_idx = int(len(ds) * val_split)
    val_indices, trn_indices = indices[:split_idx], indices[split_idx:]
    ds_trn, ds_val = (Subset(ds, trn_indices.tolist()),
                      Subset(ds, val_indices.tolist()))

    if isinstance(ds_val.dataset, OCRDataset):
        ds_val.dataset.transform = default_transform(is_val=True)
    elif isinstance(ds_val.dataset, ConcatDataset):
        for ds in ds_val.dataset.datasets:
            if isinstance(ds, OCRDataset):
                ds.transform = default_transform(is_val=True)

    return ds_trn, ds_val
