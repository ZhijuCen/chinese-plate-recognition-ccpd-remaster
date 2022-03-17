
import torch
from torch import nn, optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from tqdm.auto import tqdm

import sys
import logging
from pathlib import Path
from typing import Optional, Union, Type, Any, List, Dict
from abc import ABC


class AbstractModelContainer(ABC):

    def train(self, *args, **kwargs): return

    def validation(self, *args, **kwargs): return

    def predict(self, *args, **kwargs) -> Any: return

    def prune_model(self, amount_map: Dict[Type, float]) -> None: return

    def export_onnx(self, path: Union[Path, str]): return

    def save_model_state_dict(self, path: Union[Path, str]): return


class KeypointRCNNContainer(AbstractModelContainer):

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

    @classmethod
    def new_keypointrcnn_resnet50_fpn(
        cls, optimizer: Type[optim.Optimizer] = optim.SGD,
        opt_default_params: Dict[str, Any] = dict(lr=1e-3),
        device: Union[torch.device, str, int] = "cpu",
        dtype: torch.dtype = torch.float32,

        # Args for keypointrcnn_resnet_fpn initialzation.
        pretrained: bool = False, progress: bool = True,
        num_classes: int = 1, num_keypoints: int = 4,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: Optional[Any] = None, **kwargs: Any
    ):
        model = keypointrcnn_resnet50_fpn(
            pretrained, progress, num_classes, num_keypoints,
            pretrained_backbone, trainable_backbone_layers, **kwargs)
        return cls(model, optimizer, opt_default_params, device, dtype)

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
                self.validation(val_loader)
            total_loss /= max(len(loader), 1)
            self.logger.info(f"Epoch: {e}, loss: {total_loss:.4f}")

    def validation(self, loader: DataLoader) -> float:
        self.model.train()
        with torch.no_grad():
            total_loss = 0.
            for images, targets in tqdm(loader, desc="Validating batch: "):
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                losses: Dict[str, torch.FloatTensor] = self.model(images, targets)
                summed_loss = 0.
                for _, v in losses.items():
                    summed_loss += v
                total_loss += summed_loss.item()
            total_loss /= max(len(loader), 1)
        self.logger.info(f"Validation loss: {total_loss:.4f}")
        return total_loss

    def predict(self, images: List[torch.Tensor]):
        self.model.eval()
        return self.model(images)

    def prune_model(self, amount_map: Dict[Type[nn.Module], float] = {
        nn.Conv2d: 0.2,
        nn.Linear: 0.2,
        }):
        for _, module in self.model.named_modules():
            for k, v in amount_map.items():
                if isinstance(module, k):
                    prune.l1_unstructured(module, "weight", amount=v)

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
