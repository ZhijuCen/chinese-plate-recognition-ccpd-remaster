
from .acc import ExactAccuracy

import torch
import unittest


class TestExactAccuracy(unittest.TestCase):
    
    def test_result_correct(self):
        metric = ExactAccuracy()
        pred = torch.tensor([[1, 1, 0, 2, 0, 3, 0, 1],
                             [2, 3, 0, 1, 1, 0, 3, 1]], dtype=torch.int64)
        target = torch.tensor([[1, 2, 3, 1, 0],
                               [2, 3, 1, 3, 0]], dtype=torch.int64)
        metric.update(pred, target)
        acc = metric.compute()
        self.assertAlmostEqual(0.5, acc, 1)


if __name__ == "__main__":
    unittest.main()
