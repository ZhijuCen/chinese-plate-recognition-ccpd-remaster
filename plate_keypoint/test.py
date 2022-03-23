
from .model import default_keypoint_model_224

import tensorflow as tf

import unittest
import traceback


class TestDefaultKeypointModel224(unittest.TestCase):

    def setUp(self) -> None:
        self.dummy_x = tf.random.uniform((32, 224, 224, 3))
        self.dummy_y = tf.random.uniform((32, 8))

    def tearDown(self) -> None:
        return super().tearDown()

    def test_dummy_default_kp_model_224_noerror(self):
        raised = False
        msg = ""
        try:
            model = default_keypoint_model_224()
            model.fit(self.dummy_x, self.dummy_y, 4, epochs=50)
        except:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == "__main__":
    unittest.main()
