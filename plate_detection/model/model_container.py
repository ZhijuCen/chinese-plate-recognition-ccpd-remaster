
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from pathlib import Path
from typing import Optional, Union, Type, Any
from abc import ABC


class ModelContainer(ABC):

    def train(self, *args, **kwargs): return

    def export_onnx(self, path: Union[Path, str]): return

    def save_model_state_dict(self, path: Union[Path, str]): return


class KeypointModelContainer(ModelContainer):

    def __init__(
        self, model: nn.Module,
        opt: Type[optim.Optimizer], opt_default_params: dict = dict(lr=1e-3)
    ):
        self.model = model
        self.opt_default_params = opt_default_params
        self.opt = opt(self.model.parameters(), self.opt_default_params)

    def train(self, loader: DataLoader):
        return

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
        opt_class_string = loaded_dict["optimizer_class"]
        eval(f"import {opt_class_string}")
        opt_class: Type[optim.Optimizer] = eval(opt_class_string)
        self.opt = opt_class(self.model.parameters, self.opt_default_params)
        self.opt.load_state_dict(loaded_dict["state_dict"])

    def load_model_state_dict(self, path: Union[Path, str]):
        loaded_dict = torch.load(str(path))
        self.model.load_state_dict(loaded_dict)

    def export_onnx(self, path: Union[Path, str]):
        self.model.eval()
        self.model.cpu()
        input_names = ["image"]
        output_names = ["boxes", "labels", "keypoints"]
        example_input = [torch.rand(3, 320, 640), torch.rand(3, 480, 480)]
        torch.onnx.export(
            self.model, example_input, str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "image": {0: "batch", 2: "h", 3: "w"}
            },
            opset_version=11,
        )

    @classmethod
    def new_keypointrcnn_resnet50_fpn(cls,
        pretrained: bool = False, progress: bool = True,
        num_classes: int = 1, num_keypoints: int = 4,
        pretrained_backbone: bool = True, trainable_backbone_layers: Optional[Any] = None, **kwargs: Any
    ):
        model = keypointrcnn_resnet50_fpn(
            pretrained, progress, num_classes, num_keypoints,
            pretrained_backbone, trainable_backbone_layers, **kwargs)
        return cls(model, optim.SGD)
