
import numpy as np
from typing import Any, Dict, List, Tuple, Union, Optional
from pathlib import Path
from os import path


def load_char_annots(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        obj: dict = yaml.safe_load(f)
    return obj


def remap_ccpd_lp_annot(
    char_indices: List[int],
    ccpd_map: Dict[str, List[str]],
    reindex_map: List[str]
):
    chars_list = list()
    for seq_idx, char_idx in enumerate(char_indices):
        if seq_idx == 0:
            chars_list.append(ccpd_map["provinces"][char_idx])
        elif seq_idx == 1:
            chars_list.append(ccpd_map["alphabets"][char_idx])
        else:
            chars_list.append(ccpd_map["ads"][char_idx])
    reindicied_chars_idx = list()
    for s in chars_list:
        reindicied_chars_idx.append(reindex_map.index(s))
    return reindicied_chars_idx


def parse_ccpd_filename(
    filepath: str,
    remap_lp_annot: Optional[Dict[str, Any]] = None,
    bbox_from_kp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    filename = path.split(filepath)[-1]
    stem = filename.rsplit(".")[-2]
    (area, tilt, bbox, keypoints, lp_number,
     brightness, blurriness
    ) = stem.split("-")
    area = float(f"0.{area}")
    tilt = np.array([v for v in tilt.split("_")], dtype=np.int64)
    keypoints = np.array([[s.split("&") for s in keypoints.split("_")]], dtype=np.int64)
    if bbox_from_kp:
        bbox = np.array([[
            keypoints[..., 0].min(),
            keypoints[..., 1].min(),
            keypoints[..., 0].max(),
            keypoints[..., 1].max(),
        ]], dtype=np.int64)
    else:
        bbox = np.array([("&".join(bbox.split("_"))).split("&")], dtype=np.int64)
    keypoints = np.concatenate((keypoints, np.ones((*keypoints.shape[:2], 1))), axis=2)
    lp_number = [int(v) for v in lp_number.split("_")]
    brightness = int(brightness)
    blurriness = int(blurriness)

    if remap_lp_annot is not None:
        lp_number = remap_ccpd_lp_annot(
            lp_number,
            remap_lp_annot["ccpd_char_annotations"],
            remap_lp_annot["char_annotations"])
    lp_number = np.asarray(lp_number, dtype=int)

    return (
        bbox, keypoints, lp_number,
        {"area": area, "tilt": tilt,
         "brightness": brightness, "blurriness": blurriness}
    )


def parse_split_file_to_arrays(
    dataset_dir: Union[Path, str],
    split_file_path: Union[Path, str],
    label: int = 1,
    remap_lp_annot: Optional[Dict[str, Any]] = None,
    bbox_from_kp: bool = True,
) -> Tuple[List[str],
    List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """_summary_

    Args:
        dataset_dir (Union[Path, str]): _description_
        split_file_path (Union[Path, str]): _description_
        label (int, optional): _description_. Defaults to 1.
        remap_lp_annot (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
        bbox_from_kp (bool, optional): _description_

    Returns:
        Tuple[
            List[str], List[np.ndarray], List[np.ndarray],
            List[np.ndarray], List[np.ndarray]]:
            img_paths, boxes, labels, keypoints, license_plate_annots
    """
    dataset_dir: Path = Path(dataset_dir)
    with open(split_file_path, "rt") as f:
        img_paths, boxes, labels, keypoints, license_plate_annots = (
            list(), list(), list(), list(), list())
        for line in f:
            img_path = (dataset_dir / line.rstrip("\r\n ")).resolve()
            img_path = str(img_path)
            bbox, kp, lp, _ = parse_ccpd_filename(
                img_path, remap_lp_annot, bbox_from_kp)
            img_paths.append(img_path)
            boxes.append(bbox)
            labels.append(np.full((bbox.shape[0],), label))
            keypoints.append(kp)
            license_plate_annots.append(lp)
    return img_paths, boxes, labels, keypoints, license_plate_annots
