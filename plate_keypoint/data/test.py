
from . import load_char_annots, parse_split_file_to_arrays, to_bbox_cropped_dataset

import unittest
import traceback
from pathlib import Path


class TestToBboxCroppedDataset(unittest.TestCase):

    def setUp(self) -> None:
        root_dir = Path(__file__).parents[2]
        self.test_suite_dir = root_dir / "test_suite"
        annot_obj = load_char_annots(str(root_dir / "char-annotations.yaml"))
        (self.img_paths, self.boxes,
         self.labels, self.kps,
         self.lp_char_annots) = parse_split_file_to_arrays(
            self.test_suite_dir, self.test_suite_dir / "val_for_test.txt",
            remap_lp_annot=annot_obj)

    def test_export_cropped_dataset_noerror(self):
        raised = False
        msg = ""
        try:
            to_bbox_cropped_dataset(
                self.img_paths, self.boxes, self.kps,
                self.test_suite_dir, "val_kp_for_test", depth=1
            )
        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == "__main__":
    unittest.main()
