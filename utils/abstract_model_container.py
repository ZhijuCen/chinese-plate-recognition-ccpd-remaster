
from abc import ABC
from typing import Any, Union
from pathlib import Path


class AbstractModelContainer(ABC):

    def train(self, *args, **kwargs): return

    def validation(self, *args, **kwargs): return

    def predict(self, *args, **kwargs) -> Any: return

    def export_onnx(self, path: Union[Path, str]): return
