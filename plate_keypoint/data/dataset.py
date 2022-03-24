
import albumentations as A
import tensorflow as tf
import numpy as np
import yaml

from pathlib import Path
from typing import Union, Tuple, Dict, Any
from functools import partial


def read_image(path: str):
    buffer = tf.io.read_file(path)
    image = tf.image.decode_jpeg(buffer, channels=3)
    return image


def normalize_keypoints(
    image: tf.Tensor, keypoints: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    keypoints = tf.cast(keypoints, dtype=tf.float32)
    img_shape = tf.shape(image)
    img_shape = tf.cast(img_shape, dtype=tf.float32)
    keypoints_x = keypoints[:, 0] / img_shape[1]
    keypoints_y = keypoints[:, 1] / img_shape[0]
    keypoints = tf.stack((keypoints_x, keypoints_y), axis=1)
    return image, keypoints


def image_augmentation_fn(
    image: tf.Tensor, keypoints: tf.Tensor,
    is_val: bool = False) -> tf.Tensor:
    if not is_val:
        # TODO: Solve Overfitting Trainset.
        transform = A.Compose([
            A.Resize(224, 224),
            A.InvertImg(),
            A.ToGray(),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2),
            A.Cutout(),
            A.Perspective(scale=0.003),
        ], keypoint_params=A.KeypointParams("xy", remove_invisible=False))
    else:
        transform = A.Compose([
            A.Resize(224, 224),
        ], keypoint_params=A.KeypointParams("xy", remove_invisible=False))
    data = {"image": image, "keypoints": keypoints}
    aug_data = transform(**data)
    image = aug_data["image"]
    keypoints = aug_data["keypoints"]
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    keypoints = tf.cast(keypoints, dtype=tf.float32)
    keypoints = tf.reshape(keypoints, [1, 8])
    return image, keypoints


def process_augmentation(image, keypoints, is_val=False):
    aug_img, aug_kps = tf.numpy_function(
        image_augmentation_fn,
        inp=[image, keypoints, is_val], Tout=(tf.float32, tf.float32))
    return aug_img, aug_kps


def set_shapes(image: tf.Tensor, keypoints: tf.Tensor):
    image.set_shape([224, 224, 3])
    keypoints.set_shape([1, 8])
    return image, keypoints


def get_dataset_from_yaml(
    dataset_dir: Union[str, Path], yaml_path: Union[str, Path],
    batch_size: int = 16, is_val: bool = False
) -> tf.data.Dataset:
    dataset_dir: Path = Path(dataset_dir)
    with open(yaml_path, "r") as f:
        loaded_object: Dict[str, Any] = yaml.safe_load(f)
    image_paths = np.asarray(
        [str(dataset_dir.joinpath(p).resolve())
         for p in loaded_object["image_paths"]])
    keypoints = np.asarray(loaded_object["keypoints"])

    tfds = tf.data.Dataset.from_tensor_slices((image_paths, keypoints))
    tfds = tfds.map(lambda p, k: (read_image(p), k))
    tfds = tfds.map(normalize_keypoints)
    tfds = tfds.map(partial(process_augmentation, is_val=is_val))
    tfds = tfds.map(set_shapes)

    tfds = tfds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return tfds
