
from .mobile_ocr_net import MobileOCRNet

import torch

import unittest
import traceback


class TestMobileOCRNet(unittest.TestCase):

    def test_mobileocrnet_noerror(self):
        raised = False
        msg = ""
        try:
            batch_size = 2
            images = torch.randn(batch_size, 3, 64, 224, dtype=torch.float32)
            targets = torch.randint(3, (batch_size, 8))
            input_lengths = torch.full((batch_size,), 224 // 8, dtype=torch.int64)
            target_lengths = torch.full((batch_size,), 8, dtype=torch.int64)
            net = MobileOCRNet.mobile_ocr_net_small(3)
            loss_func = torch.nn.CTCLoss()
            opt = torch.optim.SGD(net.parameters(), 1e-3)
            out = net(images)
            loss = loss_func(out, targets, input_lengths, target_lengths)
            opt.zero_grad()
            loss.backward()
            opt.step()
        except Exception:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == '__main__':
    unittest.main()
