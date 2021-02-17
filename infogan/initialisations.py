"""Implementation of useful initialisations for different parts of the model."""
import math
import typing
from typing import Any

import numpy as np
import torch
import torch.nn as nn

@typing.no_type_check
def conv_nn_resize_(
        tensor_or_module: torch.Tensor, subinit: Any, stride: int = None
        ) -> torch.Tensor:
    """
    Nearest neighbor resize initialization for strided convolution.
    https://arxiv.org/abs/1707.02937

    Found at: https://github.com/pytorch/pytorch/pull/5429
    """
    if isinstance(tensor_or_module, nn.Module):
        tensor = tensor_or_module.weight.data
        stride = stride or tensor_or_module.stride
    else:
        tensor = tensor_or_module
    try:
        stride = tuple(stride)
    except TypeError:
        stride = (stride,) * (tensor.dim() - 2)
    if all(s == 1 for s in stride):
        return subinit(tensor)
    subtensor = tensor[(slice(None), slice(None)) +
                       tuple(slice(None, None, s) for s in stride)]
    subinit(subtensor)
    for d, s in enumerate(stride):
        subtensor = subtensor.repeat_interleave(s, 2 + d)
    return tensor.copy_(subtensor[tuple(slice(None, l) for l in tensor.shape)])


def gen_linear_init_weights(m: Any, gain: float = 1) -> None:
    """Initialise weights of a module using Xavier normal initialisation."""
    # See https://github.com/pytorch/pytorch/issues/18182 for original formulation.
    if type(m) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d}:
        nn.init.xavier_normal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.normal_(m.bias, 0, 0.01)

    if type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


def gen_relu_init_weights(m: Any, gain: float = 0.1) -> None:
    """Initialise weights of a module using Xavier normal initialisation."""
    # See https://github.com/pytorch/pytorch/issues/18182 for original formulation.
    if type(m) == nn.ConvTranspose2d:
        conv_nn_resize_(m, lambda x: nn.init.xavier_normal_(x, gain))
        if m.bias is not None:
            nn.init.normal_(m.bias, 0, 0.01)

    # if type(m) == nn.BatchNorm2d:
    #     nn.init.normal_(m.weight.data, 1.0, 0.01)
    #     nn.init.normal_(m.bias.data, 0.0, 0.01)


def gen_tanh_init_weights(m: Any, gain: float = 1) -> None:
    """Initialise weights of a module using Xavier normal initialisation."""
    # See https://github.com/pytorch/pytorch/issues/18182 for original formulation.
    assert type(m) == nn.ConvTranspose2d
    nn.init.xavier_normal_(m.weight.data, gain)
    if m.bias is not None:
        nn.init.normal_(m.bias, 0, 0.01)


def disc_lrelu_init_weights(m: Any, gain: float = 1.4, bn_weight: float = 1) -> None:
    """Initialise weights of a module using Xavier normal initialisation."""
    # See https://github.com/pytorch/pytorch/issues/18182 for original formulation.
    if type(m) in {nn.Conv2d, nn.Linear}:
        nn.init.xavier_normal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.normal_(m.bias.data, 0, 0.001)

    if type(m) in {nn.BatchNorm1d, nn.BatchNorm2d}:
        nn.init.normal_(m.weight.data, bn_weight, 0.001)
        nn.init.normal_(m.bias.data, 0.0, 0.001)


def disc_sigmoid_init_weights(m: Any, gain: float = 1) -> None:
    """Initialise weights of a module using Xavier normal initialisation."""
    # See https://github.com/pytorch/pytorch/issues/18182 for original formulation.
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.normal_(m.bias, 0, 0.01)
