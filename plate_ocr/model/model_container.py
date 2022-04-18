
import sys
try:
    from ...utils.abstract_model_container import AbstractModelContainer
except ValueError:
    sys.path.insert(0, "../..")
    from utils.abstract_model_container import AbstractModelContainer

from .mobile_ocr_net import MobileOCRNet

from torch import optim, nn

from typing import Union, Any
from pathlib import Path


class MobileOCRNetContainer(AbstractModelContainer):
    
    def __init__(self, n_classes: int) -> None:
        self.model = MobileOCRNet.mobile_ocr_net_small(n_classes)
        self.optimizer = optim.SGD(self.model.parameters, 1e-3)
        self.criterion = nn.CTCLoss()
    
    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)
    
    def validation(self, *args, **kwargs):
        return super().validation(*args, **kwargs)
    
    def predict(self, *args, **kwargs) -> Any:
        return super().predict(*args, **kwargs)
    
    def export_onnx(self, path: Union[Path, str]):
        return super().export_onnx(path)
