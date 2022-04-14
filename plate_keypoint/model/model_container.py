
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

from typing import Union
from pathlib import Path


class ToOnnxCallback(Callback):

    def __init__(self, onnx_path: Union[Path, str]):
        self.onnx_path = str(onnx_path)
    
    def on_epoch_end(self, epoch, logs=None):
        example_input = tf.random.uniform((1, 224, 224, 3))
        tf2onnx.convert.from_keras(self.model, example_input,
                                   output_path=self.onnx_path)
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
            ToOnnxCallback(self.runtime_output_dir / "latest.onnx"),
        ]
    
    def train(self,
              train_dataset: tf.data.Dataset,
              epochs: int,
              val_dataset: tf.data.Dataset = None):
        self.model.fit(train_dataset,
                       epochs=epochs,
                       validation_data=val_dataset,
                       callbacks=self.callbacks)
        self.model.load_weights(self.ckpt_path)
    
    def validation(self, val_dataset: tf.data.Dataset):
        return self.model.evaluate(val_dataset)

    def predict(self, x):
        return self.model.predict(x)
    
    def export_onnx(self, path: Union[Path, str]):
        example_input = tf.random.uniform((1, 224, 224, 3))
        tf2onnx.convert.from_keras(self.model, example_input,
                                   output_path=str(path))
