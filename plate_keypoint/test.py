
from .model import default_keypoint_model_224
from .data import get_dataset_from_yaml

import tensorflow as tf

from pathlib import Path
import unittest
import traceback


class TestDefaultKeypointModel224(unittest.TestCase):

    def setUp(self) -> None:
        self.dummy_x = tf.random.uniform((32, 224, 224, 3))
        self.dummy_y = tf.random.uniform((32, 8))

        self.test_suite_dir = Path(__file__).parents[1] / "test_suite"

    def test_dummy_default_kp_model_224_noerror(self):
        raised = False
        msg = ""
        try:
            model = default_keypoint_model_224()
            model.fit(self.dummy_x, self.dummy_y, 4, epochs=10)
        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)

    def test_train_default_keypoint_model_224_with_yaml_dataset_no_error(self):
        raised = False
        msg = ""
        try:
            ds = get_dataset_from_yaml(
                self.test_suite_dir,
                self.test_suite_dir / "val_kp_for_test.yaml")
            ds_val = get_dataset_from_yaml(
                self.test_suite_dir,
                self.test_suite_dir / "val_kp_for_test.yaml", is_val=True)
            model = default_keypoint_model_224()
            model.fit(ds, epochs=200, validation_data=ds_val, validation_freq=10)
        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == "__main__":
    unittest.main()
