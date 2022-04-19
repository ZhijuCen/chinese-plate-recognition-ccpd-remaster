
import sys
try:
    from ...utils.abstract_model_container import AbstractModelContainer
    from ...utils.logger import init_logger
except ValueError:
    sys.path.insert(0, "../..")
    from utils.abstract_model_container import AbstractModelContainer
    from utils.logger import init_logger
try:
    from ..data.dataset import to_dataloader, split_ds
    from ..metrics.acc import ExactAccuracy
except ValueError:
    sys.path.insert(0, "..")
    from data.dataset import to_dataloader, split_ds
    from metrics.acc import ExactAccuracy

from .mobile_ocr_net import MobileOCRNet

import torch
from torch import optim, nn
from torch.utils.data import Dataset
import numpy as np

from tqdm.auto import tqdm

from typing import Union, Any
from pathlib import Path


class MobileOCRNetContainer(AbstractModelContainer):
    
    def __init__(self,
                 n_classes: int,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 runtime_output_dir: Union[str, Path] = None) -> None:
        self.device = (torch.device(device)
                       if torch.cuda.is_available()
                       else torch.device("cpu"))
        self.dtype = dtype
        self.model = MobileOCRNet.mobile_ocr_net_small(n_classes)
        self.model.to(device=self.device, dtype=self.dtype)
        self.optimizer = optim.Adam(self.model.parameters(), 1e-3)
        self.criterion = nn.CTCLoss()

        self.runtime_output_dir = (Path(runtime_output_dir)
                                   if runtime_output_dir is not None
                                   else Path("ocr_rt_tmp"))
        self.runtime_output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = init_logger(self.__class__,
                                  self.runtime_output_dir / "logs.log")
    
    def train(self,
              train_ds: Dataset,
              epochs: int,
              val_ds: Dataset = None,
              patience: int = 5):
        if val_ds is None:
            train_ds, val_ds = split_ds(train_ds)
        train_loader = to_dataloader(train_ds)
        best_val_acc = -np.inf
        best_val_acc_epoch = 0
        for e in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            summed_loss = 0.
            for imgs, targets in tqdm(train_loader, "Training step"):
                imgs: torch.Tensor = imgs.to(device=self.device)
                targets: torch.Tensor = targets.to(device=self.device)
                pred: torch.Tensor = self.model(imgs)

                input_lengths = torch.full((pred.size(1),),
                                           pred.size(0),
                                           device=self.device)
                target_lengths = torch.sum(torch.where(targets != 0, 1, 0), dim=1)

                loss: torch.Tensor = self.criterion(pred, targets,
                                                    input_lengths,
                                                    target_lengths)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                summed_loss += loss.item()
            train_loss = summed_loss / max(1, len(train_loader))
            val_acc = self.validation(val_ds)
            self.logger.info(
                f"Epoch: {e}, Train Loss: {train_loss}, Val Acc: {val_acc}")

            self.export_onnx("latest")
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_acc_epoch = e
                self.export_onnx("best")
            else:
                if e - best_val_acc_epoch > patience:
                    self.logger.info("Early stopping.")
                    break
    
    def validation(self, val_ds: Dataset):
        val_loader = to_dataloader(val_ds)
        metric = ExactAccuracy()
        self.model.eval()
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc="Evaluation"):
                imgs = imgs.to(device=self.device)
                targets = targets.to(device=self.device)
                pred = self.model(imgs)
                metric.update(pred, targets)
        return metric.compute()
    
    def predict(self, images: torch.Tensor) -> Any:
        self.model.eval()
        logits = self.model(images)
        pred = torch.argmax(logits)
        return pred
    
    def export_onnx(self, name: str = "exported"):
        self.model.eval()
        self.model.cpu()
        example_input = torch.rand(1, 3, 64, 224)
        torch.onnx.export(
            self.model, example_input,
            str(self.runtime_output_dir / f"{name}.onnx"),
            opset_version=11,
        )
        self.model.to(device=self.device)
