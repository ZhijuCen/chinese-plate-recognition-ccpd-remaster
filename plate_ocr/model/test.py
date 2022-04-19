
import sys
try:
    from ..data.dataset import OCRDataset
except ValueError:
    sys.path.insert(0, "..")
    from data.dataset import OCRDataset
from .mobile_ocr_net import MobileOCRNet
from .model_container import MobileOCRNetContainer

import torch
import yaml

from pathlib import Path
import unittest
import traceback


class TestMobileOCRNet(unittest.TestCase):

    def test_mobileocrnet_noerror(self):
        raised = False
        msg = ""
        try:
            n_classes = 3
            batch_size = 2
            images = torch.randn(batch_size, 3, 64, 224, dtype=torch.float32)
            targets = torch.randint(n_classes, (batch_size, 8))
            input_lengths = torch.full((batch_size,), 224 // 8, dtype=torch.int64)
            target_lengths = torch.full((batch_size,), 8, dtype=torch.int64)
            net = MobileOCRNet.mobile_ocr_net_small(n_classes)
            loss_func = torch.nn.CTCLoss()
            opt = torch.optim.SGD(net.parameters(), 1e-3)
            out: torch.Tensor = net(images)
            self.assertTrue((out.shape == (28, batch_size, n_classes)),
                            f"{out.shape} != T, N, C")
            loss = loss_func(out, targets, input_lengths, target_lengths)
            opt.zero_grad()
            loss.backward()
            opt.step()
        except Exception:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


class TestMobileOCRNetContainer(unittest.TestCase):

    def setUp(self) -> None:
        self.project_root_dir = Path(__file__).parents[2]
        self.test_suite_dir = self.project_root_dir / "test_suite"

        with open(self.project_root_dir / "char-annotations.yaml") as f:
            self.char_annots = yaml.safe_load(f)["char_annotations"]
        
    
    def test_mobileocrnet_container_noerror(self):
        raised = False
        msg = ""
        try:
            ds = OCRDataset.from_yaml(
                self.test_suite_dir / "val_ocr_for_test.yaml", self.test_suite_dir)
            model = MobileOCRNetContainer(len(self.char_annots), "cuda")
            dummy_input = torch.randn((1, 3, 64, 224), device=model.device)
            model.train(ds, 50)
            model.predict(dummy_input)
        except Exception:
            msg = traceback.format_exc()
            raised = True
        self.assertFalse(raised, msg)


if __name__ == '__main__':
    unittest.main()
