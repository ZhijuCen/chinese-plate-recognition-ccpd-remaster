
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


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
    filename: str,
    remap_lp_annot: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    stem, _ = filename.rsplit(".")
    (area, tilt, bbox, keypoints, lp_number,
     brightness, blurriness
    ) = stem.split("-")
    area = float(f"0.{area}")
    tilt = np.array([v for v in tilt.split("_")], dtype=np.int64)
    bbox = np.array([s.split("&") for s in bbox.split("_")], dtype=np.int64)
    keypoints = np.array([s.split("&") for s in keypoints.split("_")], dtype=np.int64)
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
