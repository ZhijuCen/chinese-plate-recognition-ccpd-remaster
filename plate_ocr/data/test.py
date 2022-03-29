
from . import parse_split_file_to_arrays, load_char_annots
from .preparation import to_keypoints_transformed_dataset

import unittest
import traceback
from pathlib import Path


class TestToKeypointsTransformedDataset(unittest.TestCase):

    def setUp(self) -> None:
        root_dir = Path(__file__).parents[2]
        self.test_suite_dir = root_dir / "test_suite"
        annot_obj = load_char_annots(str(root_dir / "char-annotations.yaml"))
        (self.img_paths, self.boxes,
         self.labels, self.kps,
         self.lp_char_annots) = parse_split_file_to_arrays(
            self.test_suite_dir, self.test_suite_dir / "val_for_test.txt",
            remap_lp_annot=annot_obj)

    def test_to_keypoints_transformed_dataset_noerror(self):
        raised = False
        msg = ""
        try:
            to_keypoints_transformed_dataset(
                self.img_paths, self.kps, self.lp_char_annots,
                self.test_suite_dir, "val_ocr_for_test", depth=1
            )
        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == '__main__':
    unittest.main()
