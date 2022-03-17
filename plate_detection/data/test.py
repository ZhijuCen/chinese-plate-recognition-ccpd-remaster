
from .augmentation import *
from .dataset import *
from .filename_parser import *

import numpy as np

from pathlib import Path
import unittest
import traceback


class TestParseCCPD(unittest.TestCase):

    def setUp(self) -> None:
        annot_path = Path(__file__).parents[2] / "char-annotations.yaml"
        self.test_filename = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
        self.annot_obj = load_char_annots(str(annot_path))

    def test_parse_ccpd_filenane_noerror(self):
        raised = False
        msg = ""
        try:
            parse_ccpd_filename(self.test_filename, self.annot_obj)
        except Exception:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)

    def test_bbox_and_keypoints_correct(self):
        bbox, kp, lp, _ = parse_ccpd_filename(self.test_filename, self.annot_obj)
        bbox_eq = (bbox == np.array([[154, 383, 386, 473]])).all()
        self.assertTrue(bbox_eq)
        kp_eq = (kp == np.array([[[386, 473, 1], [177, 454, 1], [154, 383, 1], [363, 402, 1]]])).all()
        self.assertTrue(kp_eq)


class TestDatasetAndLoader(unittest.TestCase):

    def setUp(self):
        self.test_suite_path = Path(__file__).parents[2] / "test_suite"
        self.test_filelist_path = self.test_suite_path / "val_for_test.txt"
        annot_path = Path(__file__).parents[2] / "char-annotations.yaml"
        self.annot_obj = load_char_annots(annot_path)

    def test_init_loader_noerror(self):
        raised = False
        msg = ""
        try:
            img_paths, boxes, labels, kps, lpas = get_dataset_from_split_file(
                self.test_suite_path,
                self.test_filelist_path,
                0, self.annot_obj)
            loader = default_loader(img_paths, labels, boxes, kps,
                                    transform_apply_p=1.)
            for imgs, targets in loader:
                continue
        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == "__main__":
    unittest.main()
