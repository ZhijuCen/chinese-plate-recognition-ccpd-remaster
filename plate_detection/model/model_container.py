
import torch
from torch import nn, optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision.models.detection import (keypointrcnn_resnet50_fpn,
                                          ssdlite320_mobilenet_v3_large)

from torchmetrics.detection.map import MeanAveragePrecision

from tqdm.auto import tqdm

import sys
import logging
from datetime import datetime as dt
from pathlib import Path
from typing import Optional, Union, Type, Any, List, Dict
from abc import ABC


optimizer_map = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
}


def init_logger(name: str, path: Union[str, Path]):
    logger = logging.Logger(name)
    logger.addHandler(logging.FileHandler(str(path)))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


class AbstractModelContainer(ABC):

    def train(self, *args, **kwargs): return

    def validation(self, *args, **kwargs): return

    def predict(self, *args, **kwargs) -> Any: return

    def prune_model(self, amount_map: Dict[Type[nn.Module], float]) -> None: return

    def export_onnx(self, path: Union[Path, str]): return


class SSDLiteContainer(AbstractModelContainer):

    def __init__(
        self, model: nn.Module,
        opt: Type[optim.Optimizer] = optim.Adam,
        opt_params: Dict[str, Any] = dict(lr=1e-4),
        device: Union[torch.device, str, int] = "cpu",
        dtype: torch.dtype = torch.float32,
        runtime_output_dir: Union[str, Path, None] = None,
        use_checkpoint: bool = False,
    ):
        self.device = (torch.device(device)
                       if torch.cuda.is_available()
                       else torch.device("cpu"))
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.opt_params = opt_params
        self.opt = opt(self.model.parameters(), **self.opt_params)

        self.runtime_output_dir = runtime_output_dir
        if self.runtime_output_dir is None:
            init_datetime = tuple(dt.now().timetuple())[:6]
            self.runtime_output_dir = Path(
                ("output-{:04d}{:02d}{:02d}"
                 "{:02d}{:02d}{:02d}").format(*init_datetime))
        else:
            self.runtime_output_dir = Path(self.runtime_output_dir)
        self.runtime_output_dir.mkdir(parents=True, exist_ok=True)

        self.use_checkpoint = use_checkpoint

        self.logger = init_logger(self.__class__, self.runtime_output_dir / "logs.log")

    @classmethod
    def new_model(
        cls, opt: Type[optim.Optimizer] = optim.Adam,
        opt_params: Dict[str, Any] = dict(lr=1e-4),
        device: Union[torch.device, str, int] = "cpu",
        dtype: torch.dtype = torch.float32,
        runtime_output_dir: Union[str, Path, None] = None,
        use_checkpoint: bool = False,

        num_classes: int = 2, pretrained_backbone: bool = False,
    ):
        model = ssdlite320_mobilenet_v3_large(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone
        )
        return cls(model, opt, opt_params, device, dtype,
                   runtime_output_dir, use_checkpoint)

    def train(self, loader: DataLoader, epochs: int,
              val_loader: Optional[DataLoader] = None):
        for e in tqdm(range(epochs), desc="Epoch: ", position=0):
            self.model.train()
            total_loss = 0.
            for images, targets in tqdm(loader, desc="Training batch: ",
                                        position=1):
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                losses: Dict[str, torch.FloatTensor] = self.model(images, targets)
                self.opt.zero_grad()
                # TODO: Optional: decide whether implement loss weights w.r.t keys
                summed_loss = 0.
                for _, v in losses.items():
                    summed_loss += v
                summed_loss.backward()
                total_loss += summed_loss.item()
                self.opt.step()
            if val_loader is not None:
                val_summary = self.validation(val_loader)
                self.logger.info(val_summary)
            total_loss /= max(len(loader), 1)
            self.logger.info(f"Epoch: {e}, loss: {total_loss:.4f}")

    def validation(self, loader: DataLoader):
        self.model.eval()
        mean_average_precision = MeanAveragePrecision()
        mean_average_precision.to(self.device)
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Validating batch: "):
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                prediction: List[Dict[str, torch.Tensor]] = self.model(images)
                mean_average_precision.update(prediction, targets)
        summary = mean_average_precision.compute()
        return summary

    def predict(self, images: List[torch.Tensor]):
        self.model.eval()
        return self.model(images)

    def prune_model(self, amount_map: Dict[Type[nn.Module], float] = {
        nn.Conv2d: 0.3,
        }):
        for _, module in self.model.named_modules():
            for k, v in amount_map.items():
                if isinstance(module, k):
                    prune.l1_unstructured(module, "weight", amount=v)
                    prune.remove(module, "weight")

    def export_onnx(self):
        self.model.eval()
        self.model.cpu()
        input_names = ["image"]
        output_names = ["outputs"]
        example_input = [torch.rand(3, 320, 320), torch.rand(3, 480, 640)]
        # TODO: Optional: Find out why the onnx model accepts 3-dim input.
        torch.onnx.export(
            self.model, example_input,
            str(self.runtime_output_dir / "model.onnx"),
            input_names=input_names,
            output_names=output_names,
            opset_version=11,
        )


class KeypointRCNNContainer(AbstractModelContainer):

    """
    Deprecated: too slow in training.
    """

    def __init__(
        self, model: nn.Module,
        opt: Type[optim.Optimizer] = optim.SGD,
        opt_params: Dict[str, Any] = dict(lr=1e-3),
        device: Union[torch.device, str, int] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = (torch.device(device)
                       if torch.cuda.is_available()
                       else torch.device("cpu"))
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.opt_params = opt_params
        self.opt = opt(self.model.parameters(), **self.opt_params)

        # TODO: Required: Move to function or method
        self.logger = logging.Logger(self.__class__)
        fhdlr = logging.FileHandler(
            str(Path(__file__).parents[1].joinpath("logs/model_container.log")))
        shdlr = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(fhdlr)
        self.logger.addHandler(shdlr)
        self.logger.warn("Training with KeypointRCNN will be too slow.")

    @classmethod
    def new_model(
        cls, optimizer: Type[optim.Optimizer] = optim.SGD,
        opt_params: Dict[str, Any] = dict(lr=1e-3),
        device: Union[torch.device, str, int] = "cpu",
        dtype: torch.dtype = torch.float32,

        # Args for keypointrcnn_resnet_fpn initialzation.
        pretrained: bool = False, progress: bool = True,
        num_classes: int = 2, num_keypoints: int = 4,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: Optional[Any] = None, **kwargs: Any
    ):
        model = keypointrcnn_resnet50_fpn(
            pretrained, progress, num_classes, num_keypoints,
            pretrained_backbone, trainable_backbone_layers, **kwargs)
        return cls(model, optimizer, opt_params, device, dtype)

    def train(self, loader: DataLoader, epochs: int,
              val_loader: Optional[DataLoader] = None):
        for e in tqdm(range(epochs), desc="Epoch: ", position=0):
            self.model.train()
            total_loss = 0.
            for images, targets in tqdm(loader, desc="Training batch: ",
                                        position=1):
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                losses: Dict[str, torch.FloatTensor] = self.model(images, targets)
                self.opt.zero_grad()
                # TODO: Optional: decide whether implement loss weights w.r.t keys
                summed_loss = 0.
                for _, v in losses.items():
                    summed_loss += v
                summed_loss.backward()
                total_loss += summed_loss.item()
                self.opt.step()
            if val_loader is not None:
                val_summary = self.validation(val_loader)
                self.logger.info(val_summary)
            total_loss /= max(len(loader), 1)
            self.logger.info(f"Epoch: {e}, loss: {total_loss:.4f}")

    def validation(self, loader: DataLoader):
        self.model.eval()
        mean_average_precision = MeanAveragePrecision()
        mean_average_precision.to(self.device)
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Validating batch: "):
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                prediction: List[Dict[str, torch.Tensor]] = self.model(images)
                mean_average_precision.update(prediction, targets)
        summary = mean_average_precision.compute()
        return summary

    def predict(self, images: List[torch.Tensor]):
        self.model.eval()
        return self.model(images)

    def prune_model(self, amount_map: Dict[Type[nn.Module], float] = {
        nn.Conv2d: 0.3,
        }):
        for _, module in self.model.named_modules():
            for k, v in amount_map.items():
                if isinstance(module, k):
                    prune.l1_unstructured(module, "weight", amount=v)
                    prune.remove(module, "weight")

    def save_optimizer(self, path: Union[Path, str]):
        saved_dict = {
            "optimizer_class": str(self.opt.__class__),
            "state_dict": self.opt.state_dict()
            }
        torch.save(saved_dict, str(path))

    def save_model_state_dict(self, path: Union[Path, str]):
        torch.save(self.model.state_dict(), path)

    def load_optimizer(self, path: Union[Path, str]):
        loaded_dict = torch.load(str(path))
        opt_class_string: str = loaded_dict["optimizer_class"]
        if str(self.opt.__class__) == opt_class_string:
            self.opt.load_state_dict(loaded_dict["state_dict"])
        else:
            self.logger.error(f"Attempting load state-dict of"
                              f" {opt_class_string} for {self.opt.__class__}")
            try:
                self.opt.load_state_dict(loaded_dict["state_dict"])
            except:
                pass

    def load_model_state_dict(self, path: Union[Path, str]):
        loaded_dict = torch.load(str(path))
        self.model.load_state_dict(loaded_dict)

    def export_onnx(self, path: Union[Path, str]):
        self.model.eval()
        self.model.cpu()
        input_names = ["image"]
        output_names = ["boxes", "labels", "keypoints"]
        example_input = [torch.rand(3, 320, 640), torch.rand(3, 480, 480)]
        # TODO: Optional: Find out why the onnx model accepts 3-dim input.
        torch.onnx.export(
            self.model, example_input, str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "image": {1: "h", 2: "w"}
            },
            opset_version=11,
        )
