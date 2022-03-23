
from . import load_char_annots, parse_split_file_to_arrays

import unittest
import traceback


class TestToBboxCroppedDataset(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_export_cropped_dataset_noerror(self):
        raised = False
        msg = ""
        try:
            pass
        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == "__main__":
    unittest.main()
