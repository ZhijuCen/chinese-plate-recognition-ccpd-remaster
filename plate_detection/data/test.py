
from .augmentation import *
from .dataset import *
from .filename_parser import *

from pathlib import Path
import unittest
import traceback

class TestParseCCPD(unittest.TestCase):

    def setUp(self) -> None:
        self.annot_path = Path(__file__).parent.parent.parent / "char-annotations.yaml"
        self.test_filename = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
        self.annot_obj = load_char_annots(str(self.annot_path))

    def test_noerror(self):
        raised = False
        msg = ""
        try:
            parse_ccpd_filename(self.test_filename, self.annot_obj)
        except Exception:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)

    def test_bbox_and_keypoints(self):
        import numpy as np
        bbox, kp, lp, _ = parse_ccpd_filename(self.test_filename, self.annot_obj)
        self.assertTrue((bbox == np.array([[154, 383], [386, 473]])).all())
        self.assertTrue((kp == np.array([[386, 473], [177, 454], [154, 383], [363, 402]])).all())


if __name__ == "__main__":
    unittest.main()
