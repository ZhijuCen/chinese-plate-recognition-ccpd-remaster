
import torch
from torch import nn
from torch.nn import init
from torchvision.ops import SqueezeExcitation

from typing import (Type, Union, List, Tuple, Dict, Callable,
                    Optional, Any)
from functools import partial


def calculate_n_padding(kernel_size: Tuple[int, int],
                        dilation: Tuple[int, int] = (1, 1),
                        ) -> Tuple[int, int]:

    def calc_func(k: int, d: int) -> int:
        return (k - 1) // 2 * d

    paddings = (calc_func(kernel_size[0], dilation[0]),
                calc_func(kernel_size[1], dilation[1]))
    return paddings


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Permutation(nn.Module):

    def __init__(self, dims: Union[torch.Size, List[int], Tuple[int, ...]]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, dims=self.dims)


class MobileOCRNetConv2dBlockConfig(dict):

    in_features: int
    out_features: int
    use_se: bool = False,
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    groups: int = 1,
    norm_layer_class: Type[Union[nn.BatchNorm2d, nn.LayerNorm]] = nn.BatchNorm2d
    non_linear_class: Type[Union[nn.ReLU, nn.LeakyReLU, nn.Hardswish]] = nn.ReLU
    weight_init_func: Optional[Callable[[torch.Tensor], None]] = None

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_se: bool = False,
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
                     Callable[[torch.Tensor, Any], None]
                 ] = None,
                 **weight_init_func_kwargs: Any,
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
        self.in_channels = self.cfg.in_features
        self.out_channels = self.cfg.out_features
        paddings = calculate_n_padding(cfg.kernel_size)
        self.block = list()
        self.block.append(nn.Conv2d(self.cfg.in_features,
                                    self.cfg.out_features,
                                    self.cfg.kernel_size,
                                    self.cfg.strides,
                                    paddings,
                                    groups=self.cfg.groups,
                                    ))
        if self.cfg.norm_layer_class is not None:
            self.block.append(self.cfg.norm_layer_class(self.cfg.out_features))
        self.block.append(self.cfg.non_linear_class())
        if self.cfg.use_se:
            self.block.append(SqueezeExcitation(
                self.cfg.out_features,
                _make_divisible(self.cfg.out_features, 4),
            ))
        self.block = nn.Sequential(*self.block)

        if self.cfg.weight_init_func is not None:
            for m in self.block.modules():
                if isinstance(m, nn.Conv2d):
                    self.cfg.weight_init_func(m.weight)

    def forward(self, x):
        return self.block(x)


class InverseBottleNeckConfig(dict):

    in_features: int
    out_features: int
    expand_features: int
    use_se: bool = False
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    norm_layer_class: Type[
        Union[nn.BatchNorm2d, None]
    ] = nn.BatchNorm2d
    non_linear_class: Type[
        Union[nn.ReLU, nn.LeakyReLU, nn.Hardswish, nn.Identity]
    ] = nn.ReLU
    weight_init_func: Optional[
        Callable[[torch.Tensor, Any], None]
    ] = None
    weight_init_func_kwargs: Dict[str, Any]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 expand_features: int,
                 use_se: bool = False,
                 kernel_size: Tuple[int, int] = (3, 3),
                 strides: Tuple[int, int] = (1, 1),
                 norm_layer_class: Type[
                     Union[nn.BatchNorm2d, None]
                 ] = nn.BatchNorm2d,
                 non_linear_class: Type[
                     Union[nn.ReLU, nn.LeakyReLU, nn.Hardswish, nn.Identity]
                 ] = nn.ReLU,
                 weight_init_func: Optional[
                     Callable[[torch.Tensor, Any], None]
                 ] = None,
                 **weight_init_func_kwargs: Any,
                 ) -> None:
        self.weight_init_func_kwargs = weight_init_func_kwargs
        locals_copy = locals()
        for k in self.weight_init_func_kwargs.keys():
            del locals_copy[k]
        del locals_copy["self"]
        del locals_copy["__class__"]
        super().__init__(**locals_copy)
        for k, v in locals_copy.items():
            setattr(self, k, v)
        self["weight_init_func_kwargs"] = self.weight_init_func_kwargs
        setattr(self, "weight_init_func_kwargs", self.weight_init_func_kwargs)


class InverseBottleNeck(nn.Module):

    def __init__(self, cfg: InverseBottleNeckConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.in_features
        self.out_channels = self.cfg.out_features
        self.block_main = list()
        self.block_main.append(
            MobileOCRNetConv2dBlock(MobileOCRNetConv2dBlockConfig(
                self.cfg.in_features,
                self.cfg.expand_features,
                False,
                (1, 1),
                (1, 1),
                norm_layer_class=self.cfg.norm_layer_class,
                non_linear_class=self.cfg.non_linear_class,
                weight_init_func=self.cfg.weight_init_func,
                **self.cfg.weight_init_func_kwargs,
            )))
        self.block_main.append(
            MobileOCRNetConv2dBlock(MobileOCRNetConv2dBlockConfig(
                self.cfg.expand_features,
                self.cfg.expand_features,
                self.cfg.use_se,
                self.cfg.kernel_size,
                self.cfg.strides,
                groups=self.cfg.expand_features,
                norm_layer_class=self.cfg.norm_layer_class,
                non_linear_class=self.cfg.non_linear_class,
                weight_init_func=self.cfg.weight_init_func,
                **self.cfg.weight_init_func_kwargs,
            )))
        self.block_main.append(
            MobileOCRNetConv2dBlock(MobileOCRNetConv2dBlockConfig(
                self.cfg.expand_features,
                self.cfg.out_features,
                False,
                (1, 1),
                (1, 1),
                norm_layer_class=self.cfg.norm_layer_class,
                non_linear_class=self.cfg.non_linear_class,
                weight_init_func=self.cfg.weight_init_func,
                **self.cfg.weight_init_func_kwargs,
            ))
        )
        self.block_shortcut = list()
        if self.cfg.in_features == self.cfg.out_features and self.cfg.strides == (1, 1):
            self.block_shortcut.append(nn.Identity())
        else:
            self.block_shortcut.append(
                MobileOCRNetConv2dBlock(MobileOCRNetConv2dBlockConfig(
                    self.cfg.in_features,
                    self.cfg.out_features,
                    False,
                    (3, 3),
                    self.cfg.strides,
                    norm_layer_class=self.cfg.norm_layer_class,
                    non_linear_class=self.cfg.non_linear_class,
                    weight_init_func=self.cfg.weight_init_func,
                    **self.cfg.weight_init_func_kwargs,
                ))
            )

        self.block_main = nn.Sequential(*self.block_main)
        self.block_shortcut = nn.Sequential(*self.block_shortcut)

    def forward(self, x):
        out_main = self.block_main(x)
        out_shortcut = self.block_shortcut(x)
        return out_main + out_shortcut


class MobileOCRNet(nn.Module):

    def __init__(self,
                 features: List[Union[InverseBottleNeck, MobileOCRNetConv2dBlock]],
                 n_classes: int,
                 feature_map_height: int = 2,
                 ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.features = nn.Sequential(*features)
        self.features_head = list()
        self.features_head.append(nn.Conv2d(
            self.features[-1].out_channels,
            self.features[-1].out_channels,
            (feature_map_height, 1),
            (1, 1),
            groups=self.features[-1].out_channels,
        ))
        self.features_head.append(Permutation((3, 0, 1, 2)))
        self.features_head = nn.Sequential(*self.features_head)

        hidden_size_rnn = 128
        bidirectional = True
        directions = 2 if bidirectional else 1
        self.rnn = nn.GRU(self.features[-1].out_channels, hidden_size_rnn, 2,
                          bidirectional=bidirectional, batch_first=False)

        self.head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden_size_rnn * directions, 128),
            nn.Dropout(),
            nn.Linear(128, self.n_classes),
            nn.LogSoftmax(dim=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.features_head(x)
        t, n, c, h = x.shape
        x = torch.reshape(x, (t, n, c*h))
        x, h_n = self.rnn(x)
        out = self.head(x)
        return out

    @classmethod
    def mobile_ocr_net_small(cls, n_classes: int):
        cfgs = [
            MobileOCRNetConv2dBlockConfig(
                3, 16, strides=(2, 2), non_linear_class=nn.Hardswish),
            InverseBottleNeckConfig(
                16, 16, 16, use_se=True, kernel_size=(3, 3), strides=(2, 1)),
            InverseBottleNeckConfig(
                16, 24, 72, use_se=False, kernel_size=(3, 3), strides=(2, 2)),
            InverseBottleNeckConfig(
                24, 24, 88, use_se=False, kernel_size=(3, 3), strides=(1, 1)),
            InverseBottleNeckConfig(
                24, 40, 96, use_se=True, kernel_size=(5, 5), strides=(2, 1), non_linear_class=nn.Hardswish),
            InverseBottleNeckConfig(
                40, 40, 240, use_se=True, kernel_size=(5, 5), strides=(1, 1), non_linear_class=nn.Hardswish),
            InverseBottleNeckConfig(
                40, 40, 240, use_se=True, kernel_size=(5, 5), strides=(1, 1), non_linear_class=nn.Hardswish),
            InverseBottleNeckConfig(
                40, 48, 120, use_se=True, kernel_size=(5, 5), strides=(1, 1), non_linear_class=nn.Hardswish),
            InverseBottleNeckConfig(
                48, 48, 144, use_se=True, kernel_size=(5, 5), strides=(1, 1), non_linear_class=nn.Hardswish),
            InverseBottleNeckConfig(
                48, 96, 288, use_se=True, kernel_size=(5, 5), strides=(2, 2), non_linear_class=nn.Hardswish),
            InverseBottleNeckConfig(
                96, 96, 576, use_se=True, kernel_size=(5, 5), strides=(1, 1), non_linear_class=nn.Hardswish),
            InverseBottleNeckConfig(
                96, 96, 576, use_se=True, kernel_size=(5, 5), strides=(1, 1), non_linear_class=nn.Hardswish),
            MobileOCRNetConv2dBlockConfig(
                96, 576, use_se=True, kernel_size=(1, 1), strides=(1, 1), non_linear_class=nn.Hardswish),
        ]
        features = list()
        for cfg in cfgs:
            if isinstance(cfg, MobileOCRNetConv2dBlockConfig):
                features.append(MobileOCRNetConv2dBlock(cfg))
            elif isinstance(cfg, InverseBottleNeckConfig):
                features.append(InverseBottleNeck(cfg))
            else:
                raise TypeError()
        return cls(features, n_classes)

