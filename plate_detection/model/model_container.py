
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Union, Type
from abc import ABC


class ModelContainer(ABC):

    def train(self, *args, **kwargs): return

    def export_onnx(self, path: Union[Path, str]): return

    def save_model(self, path: Union[Path, str]): return


class KeypointModelContainer(ModelContainer):

    def __init__(
        self, model: nn.Module,
        opt: Type[optim.Optimizer], opt_default_params: dict = dict()
    ):
        self.model = model
        self.opt = opt(self.model.parameters(), opt_default_params)
    
    def train(self, loader: DataLoader):
        return
    
    def save_model(self, path: Union[Path, str]):
        return
    
    def export_onnx(self, path: Union[Path, str]):
        self.model.eval()
        self.model.cpu()
        input_names = ["x"]
        output_names = ["boxes", "labels", "keypoints"]
        example_input = [torch.rand(3, 320, 640), torch.rand(3, 480, 480)]
        torch.onnx.export(
            self.model, example_input, str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "x": {0: "batch", 2: "h", 3: "w"}
            },
            opset_version=11,
        )
