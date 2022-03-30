
import torch
from torch import nn
from torch.nn import init
from torchvision.ops import SqueezeExcitation

from typing import Type, Union, List, Tuple, Callable, ParamSpecKwargs, Optional
from functools import partial


def calculate_n_padding(kernel_size: Tuple[int, int],
                        dilation: Tuple[int, int] = (1, 1),
                        ) -> Tuple[int, int]:

    def calc_func(k: int, d: int) -> int:
        return (k - 1) // 2 * d

    paddings = (calc_func(kernel_size[0], dilation[0]),
                calc_func(kernel_size[1], dilation[1]))
    return paddings


class MobileOCRNetConv2dBlockConfig(dict):

    in_features: int
    out_features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    groups: int = 1,
    norm_layer_class: Type[Union[nn.BatchNorm2d, nn.LayerNorm]] = nn.BatchNorm2d
    non_linear_class: Type[Union[nn.ReLU, nn.LeakyReLU, nn.Hardswish]] = nn.ReLU
    weight_init_func: Optional[Callable[[torch.Tensor], None]] = None

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 strides: Tuple[int, int] = (1, 1),
                 groups: int = 1,
                 norm_layer_class: Type[
                     Union[nn.BatchNorm2d, None]
                 ] = nn.BatchNorm2d,
                 non_linear_class: Type[
                     Union[nn.ReLU, nn.LeakyReLU, nn.Hardswish, nn.Identity]
                 ] = nn.ReLU,
                 weight_init_func: Optional[
                     Callable[[torch.Tensor, ParamSpecKwargs], None]
                 ] = None,
                 **weight_init_func_kwargs: ParamSpecKwargs,
                 ) -> None:
        locals_copy = locals()
        func = locals_copy["weight_init_func"]
        if func is not None:
            if (func in [init.kaiming_normal_, init.kaiming_uniform_]) \
                    and (non_linear_class not in [nn.ReLU, nn.LeakyReLU]):
                raise ValueError(f"{func} only supports non_linear_class"
                                 f" in [nn.ReLU, nn.LeakyReLU],"
                                 f" got {non_linear_class}")
            else:
                func = partial(func, **weight_init_func_kwargs)
            locals_copy["weight_init_func"] = func
        for k in weight_init_func_kwargs.keys():
            del locals_copy[k]
        del locals_copy["__class__"]
        del locals_copy["self"]
        super().__init__(**locals_copy)
        for k, v in locals_copy.items():
            setattr(self, k, v)


class MobileOCRNetConv2dBlock(nn.Module):

    def __init__(self,
                 cfg: MobileOCRNetConv2dBlockConfig
                 ) -> None:
        super().__init__()
        self.cfg = cfg
        paddings = calculate_n_padding(cfg.kernel_size)
        self.block = list()
        self.block.append(nn.Conv2d(self.cfg.in_features,
                                    self.cfg.out_features,
                                    self.cfg.kernel_size,
                                    self.cfg.strides,
                                    paddings,
                                    self.cfg.groups,
                                    ))
        if self.cfg.norm_layer_class is not None:
            self.block.append(self.cfg.norm_layer_class(self.cfg.out_features))
        self.block.append(self.cfg.non_linear_class())
        self.block = nn.Sequential(*self.block)

        if self.cfg.weight_init_func is not None:
            for m in self.block.modules():
                if isinstance(m, nn.Conv2d):
                    self.cfg.weight_init_func(m.weight)

    def forward(self, x):
        return self.block(x)


class InverseBottleNeckConfig(dict):

    def __init__(self):
        super().__init__()


class InverseBottleNeck(nn.Module):

    def __init__(self):
        super().__init__()


class MobileOCRNet(nn.Module):

    def __init__(self,
                 features: List[MobileOCRNetConv2dBlock],
                 ) -> None:
        super().__init__()
