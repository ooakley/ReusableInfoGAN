"""Implementation of model methods."""
import math
from typing import Any
import torch
import torch.nn as nn


def relu_kaiming_init_weights(m: Any) -> None:
    """Initialise weights of a module using Kaiming He initialisation."""
    # See https://github.com/pytorch/pytorch/issues/18182 for original formulation.
    if type(m) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d}:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            std = 1 / math.sqrt(fan_out)
            nn.init.normal_(m.bias.data, 0, std)

    if type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0.0, 0.02)


def leaky_relu_kaiming_init_weights(m: Any, alpha: float = 0.1) -> None:
    """Initialise weights of a module using Kaiming He initialisation."""
    # See https://github.com/pytorch/pytorch/issues/18182 for original formulation.
    if type(m) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d}:
        nn.init.kaiming_normal_(m.weight.data, a=alpha, mode="fan_out", nonlinearity="leaky_relu")
        if m.bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            gain = nn.init.calculate_gain('leaky_relu', alpha)
            std = gain / math.sqrt(fan_out)
            nn.init.normal_(m.bias.data, 0, std)

    if type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0.0, 0.02)


class GeneratorNetwork(nn.Module):
    """Class of neural network designed to generate images from a vector of random numbers."""

    def __init__(self, zdim: int, gen_cfg: dict) -> None:
        """Initialise structure of neural network."""
        super(GeneratorNetwork, self).__init__()
        filter_widths = gen_cfg["upconv_filter_widths"]
        self.initial_channels = filter_widths[0]
        self.linear = nn.Sequential(
            nn.Linear(zdim, filter_widths[0] * 4 * 4)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(filter_widths[0], filter_widths[1], 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filter_widths[1]),

            nn.ConvTranspose2d(filter_widths[1], filter_widths[2], 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filter_widths[2]),

            nn.ConvTranspose2d(filter_widths[2], filter_widths[3], 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filter_widths[3]),

            nn.ConvTranspose2d(filter_widths[3], filter_widths[4], 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(filter_widths[4]),

            nn.ConvTranspose2d(filter_widths[4], 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward pass through network."""
        linear_out = self.linear(x)
        filter_out = linear_out.view(-1, self.initial_channels, 4, 4)
        return self.conv(filter_out)


class DiscriminatorNetwork(nn.Module):
    """Class of neural network designed to determine between real and generated images."""

    def __init__(self, disc_cfg: dict) -> None:
        """Initialise network structure using config from configuation dictionary."""
        super(DiscriminatorNetwork, self).__init__()
        filter_widths = disc_cfg["downconv_filter_widths"]
        self.final_channels = filter_widths[4]
        self.conv = nn.Sequential(
            nn.Conv2d(1, filter_widths[0], 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(filter_widths[0]),

            nn.Conv2d(filter_widths[0], filter_widths[1], 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(filter_widths[1]),

            nn.Conv2d(filter_widths[1], filter_widths[2], 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(filter_widths[2]),

            nn.Conv2d(filter_widths[2], filter_widths[3], 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(filter_widths[3]),

            nn.Conv2d(filter_widths[3], filter_widths[4], 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(filter_widths[4]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward pass through module."""
        output_filters = self.conv(x)
        return output_filters.view(-1, self.final_channels * 4 * 4)


class ClassificationHead(nn.Module):
    """Head of discriminator network that outputs real/fake classification."""

    def __init__(self, disc_cfg: dict) -> None:
        """Initialise linear layer of classification head."""
        super(ClassificationHead, self).__init__()
        self.final_channels = disc_cfg["downconv_filter_widths"][4]
        self.linear = nn.Sequential(
            nn.Linear(self.final_channels * 4 * 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward pass through network."""
        return self.linear(x)


class AuxiliaryHead(nn.Module):
    """Head of discriminator network that predicts latent codes from discriminator features."""

    def __init__(self, cfg: dict) -> None:
        """Initialise linear layer of latent code prediction head."""
        super(AuxiliaryHead, self).__init__()
        self.final_channels = cfg["discriminator"]["downconv_filter_widths"][4]
        self.zdim = cfg["zdim"]
        self.linear = nn.Sequential(
            nn.Linear(self.final_channels * 4 * 4, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.zdim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward pass through network."""
        return self.linear(x)
