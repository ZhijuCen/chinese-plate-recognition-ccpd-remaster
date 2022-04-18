
import sys
try:
    from ...utils.abstract_model_container import AbstractModelContainer
except ValueError:
    sys.path.insert(0, "../..")
    from utils.abstract_model_container import AbstractModelContainer

from .models import default_keypoint_model_224

import tensorflow as tf
import tf2onnx
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import numpy as np

from typing import Union
from pathlib import Path


class ToOnnxCallback(Callback):

    def __init__(self, onnx_dir: Union[Path, str]):
        self.onnx_dir: Path = Path(onnx_dir)
        self.best_val_loss = np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        example_inputs = [tf.TensorSpec((None, 224, 224, 3), dtype=tf.float32)]
        tf2onnx.convert.from_keras(self.model, example_inputs,
                                   output_path=self.onnx_dir / "latest.onnx")
        if logs["val_loss"] <= self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            tf2onnx.convert.from_keras(self.model, example_inputs,
                                       output_path=self.onnx_dir / "best.onnx")
        return super().on_epoch_end(epoch, logs)


class KeypointNetContainer(AbstractModelContainer):
    
    def __init__(self,
                 num_keypoints=4,
                 runtime_output_dir: Union[Path, str, None] = None):
        self.model = default_keypoint_model_224(num_keypoints)
        self.runtime_output_dir = (Path(runtime_output_dir)
                                   if runtime_output_dir is not None
                                   else Path("keypoint_rt_tmp"))
        self.runtime_output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.runtime_output_dir / "ckpt-best.hdf5"
        self.callbacks = [
            ModelCheckpoint(self.ckpt_path,
                            save_best_only=True),
            EarlyStopping(patience=5),
            ToOnnxCallback(self.runtime_output_dir),
        ]
    
    def train(self,
              train_dataset: tf.data.Dataset,
              epochs: int,
              val_dataset: tf.data.Dataset = None):
        validation_split = 0.
        if val_dataset is None:
            validation_split = 0.2
        self.model.fit(train_dataset,
                       epochs=epochs,
                       validation_split=validation_split,
                       validation_data=val_dataset,
                       callbacks=self.callbacks)
        self.model.load_weights(self.ckpt_path)
    
    def validation(self, val_dataset: tf.data.Dataset):
        return self.model.evaluate(val_dataset)

    def predict(self, x):
        return self.model.predict(x)
    
    def export_onnx(self, path: Union[Path, str]):
        example_inputs = [tf.TensorSpec((None, 224, 224, 3), dtype=tf.float32)]
        tf2onnx.convert.from_keras(self.model, example_inputs,
                                   output_path=str(path))
