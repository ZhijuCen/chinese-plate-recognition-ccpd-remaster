
import torch
from torchmetrics import Metric
import numpy as np

from typing import Optional, Dict, Any, Iterable


class ExactAccuracy(Metric):
    
    def __init__(self,
                 blank_index: int = 0,
                 compute_on_step: Optional[bool] = None,
                 **kwargs: Dict[str, Any]) -> None:
        super().__init__(compute_on_step, **kwargs)
        self.blank_index = blank_index
        self.seq_of_corrects = list()
    
    def update(self,
               pred_batch: torch.IntTensor,
               target_batch: torch.IntTensor):
        for p, t in zip(pred_batch.tolist(), target_batch.tolist()):
            self.seq_of_corrects.append(
                self._sequence_is_exactly_correct(p, t, self.blank_index)
            )

    def compute(self):
        return np.mean(np.array(self.seq_of_corrects, dtype=np.int8))

    @staticmethod
    def _sequence_is_exactly_correct(pred: Iterable,
                                     target: Iterable,
                                     drop_value: int = 0) -> bool:

        def drop_continous(seq: Iterable) -> list:
            kept_seq = list()
            for v in seq:
                if not kept_seq:
                    kept_seq.append(v)
                else:
                    if v != kept_seq[-1]:
                        kept_seq.append(v)
            return kept_seq

        def drop_value_in_seq(seq, value):
            return [v for v in seq if v != value]
        
        pred: list = drop_value_in_seq(drop_continous(pred), drop_value)
        target: list = drop_value_in_seq(drop_continous(target), drop_value)

        return pred == target
