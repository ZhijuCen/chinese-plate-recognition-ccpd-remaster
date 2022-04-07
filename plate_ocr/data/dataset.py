
from .augmentation import default_transform

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 as cv

from typing import List, Union
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
        target = torch.tensor(target, dtype=torch.int64)
        return image, target

    def __len__(self):
        return len(self.img_paths)

    def to_dataloader(self, batch_size: int = 16, num_workers: int = 4):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers)

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
