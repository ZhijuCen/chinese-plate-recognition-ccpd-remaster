
import numpy as np
import cv2 as cv
import json

from typing import List, Union
from pathlib import Path


def to_bbox_cropped_dataset(
    img_paths: List[str],
    boxes: List[np.ndarray], keypoints: List[np.ndarray],
    destination_dir: Union[Path, str], split_name: str, depth: int = 0
):
    """Create dataset by cropping bounding boxes.

    Args:
        img_paths (List[str]): absolute path to the image.
        boxes (List[np.ndarray[N, 4]]):
            bounding boxes for object in [x1, y1, x2, y2] format.
        keypoints (List[np.ndarray[N, K, 3]]):
            keypoints for object in [x, y, visibility] format.
        destination_dir (Union[Path, str]):
            directory to store cropped images.
        split_name (str): name of split for train, validation or test.
        depth (int, optional):
            the directory depth to store cropped images. Defaults to 0.
    """

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    split_filename = split_name + ".json"
    export_object = {
        "image_paths": list(),
        "keypoints": list(),
    }

    for img_index, (img_path, bounding_boxes, kps) in enumerate(
            zip(img_paths, boxes, keypoints)):
        image: np.ndarray = cv.imread(img_path)
        img_path = Path(img_path)
        subdir_parts = img_path.parts[-1-depth:-1]
        dest_img_dir = destination_dir.joinpath(*subdir_parts).resolve()
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        for obj_index, (bounding_box, kp) in enumerate(
                zip(bounding_boxes, kps)):
            x_min, y_min, x_max, y_max = bounding_box
            # adjust offset of keypoints
            kp[:, 0] = kp[:, 0] - x_min
            kp[:, 1] = kp[:, 1] - y_min
            kp: np.ndarray
            kp, _ = np.split(kp, [2,], axis=1)
            cropped_image = image[y_min:y_max, x_min:x_max].copy()
            cropped_image_filename = f"kp-{img_index:08d}-{obj_index:03d}.jpg"
            cropped_image_filepath = dest_img_dir / cropped_image_filename
            cv.imwrite(str(cropped_image_filepath), cropped_image)

            # append sample
            export_object["image_paths"].append(
                "/".join(cropped_image_filepath.parts[-1-depth:]))
            export_object["keypoints"].append(kp.tolist())

    with open(destination_dir / split_filename, "w") as f:
        json.dump(export_object, f)
