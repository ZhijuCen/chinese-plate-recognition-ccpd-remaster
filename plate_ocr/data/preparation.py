
import numpy as np
import cv2 as cv
import yaml

from typing import List, Tuple, Union
from pathlib import Path


def to_keypoints_transformed_dataset(
    img_paths: List[str],
    keypoints: List[np.ndarray],
    char_labels: List[np.ndarray],
    destination_dir: Union[Path, str], split_name: str, depth: int = 0,
    destination_image_size: Tuple[int, int] = (224, 64),
):
    """Create dataset by wrap perspective transformed image via keypoints.

    Args:
        img_paths (List[str]): absolute path to the image.
        keypoints (List[np.ndarray[N, K, 3]]):
            keypoints for object in [x, y, visibility] format.
            Ordered clockwise from bot-right corner.
        char_labels (List[np.ndarray[N, Chars]]):
            Character labels(annots) for each plate.
        destination_dir (Union[Path, str]):
            directory to store cropped images.
        split_name (str): name of split for train, validation or test.
        depth (int, optional):
            the directory depth to store cropped images. Defaults to 0.
        destination_image_size (Tuple[int, int], optional):
            Dimensional Size(W, H) of perspective transformed image.
            Defaults to (224, 64)
    Remarking:
        * cv.getPerspectiveTransform(src_points, dst_points)
            both points order by left to right, and then downward.
    """

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    w_dst, h_dst = destination_image_size
    points_dst = np.array(
        [[0, 0], [w_dst, 0], [0, h_dst], [w_dst, h_dst]], dtype=np.float32
    )

    split_filename = split_name + ".yaml"
    export_object = {
        "image_paths": list(),
        "char_labels": list(),
    }

    for img_index, (img_path, kps, char_lbs) in enumerate(
            zip(img_paths, keypoints, char_labels)):
        image: np.ndarray = cv.imread(img_path)
        img_path = Path(img_path)
        subdir_parts = img_path.parts[-1-depth:-1]
        dest_img_dir = destination_dir.joinpath(*subdir_parts).resolve()
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        for obj_index, (kp, char_lb) in enumerate(zip(kps, char_lbs)):
            kp: np.ndarray
            kp, _ = np.split(kp, [2, ], axis=1)
            points_src = np.array([kp[2], kp[3], kp[1], kp[0]], dtype=np.float32)
            perspective_transformation_matrix = cv.getPerspectiveTransform(
                points_src, points_dst)
            wrapped_image = cv.warpPerspective(
                image, perspective_transformation_matrix, destination_image_size)
            wrapped_image_filename = f"ocr-{img_index:08d}-{obj_index:03d}.jpg"
            wrapped_image_filepath = dest_img_dir / wrapped_image_filename
            cv.imwrite(str(wrapped_image_filepath), wrapped_image)

            # append sample
            export_object["image_paths"].append(
                "/".join(wrapped_image_filepath.parts[-1-depth:]))
            export_object["char_labels"].append(char_lb.tolist())

    with open(destination_dir / split_filename, "w") as f:
        yaml.safe_dump(export_object, f)
